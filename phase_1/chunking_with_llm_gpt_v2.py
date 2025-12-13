#!/usr/bin/env python
"""
chunking_with_llm_gpt_v2.py

Goal
-----
Robustly chunk a single Berkshire Hathaway shareholder letter for a given year
using an LLM **without** risking truncated/invalid JSON.

Key design: split the task into passes.

Pass 1 (LLM): Produce a *chunk plan* only (boundaries + structural labels).
Pass 2 (Python): Materialize exact `chunk_text` from paragraph spans.
Pass 3 (LLM, batched): Enrich chunks with summaries/entities/principle fields.
Pass 4 (Python): Compute deterministic fields (counts/positions) + validate.

This avoids the common failure where a single completion tries to output the
entire letter text + rich metadata in one huge JSON response and gets truncated.

Directory layout (relative to this file):

    phase_1/
        chunking_rule_claude.md
        chunking_with_llm_gpt_v2.py   <-- THIS FILE
    data/
        text_extracted_letters/
            2008_cleaned.txt
            2009_cleaned.txt
            ...
        chunks_llm_gpt/
            2008_chunks_llm.jsonl
            2009_chunks_llm.jsonl

Usage
-----
    python chunking_with_llm_gpt_v2.py 2009

Environment variables
---------------------
    OPENAI_MODEL               Default: gpt-4.1-mini
    OPENAI_PLAN_TOKENS         Default: 3000
    OPENAI_ENRICH_TOKENS       Default: 3000

Optional local test mode (no API calls):
    MOCK_LLM=1 python chunking_with_llm_gpt_v2.py --text sample.txt --year 2009
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ------------------------- Paths & Constants -------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

CHUNKING_RULE_PATH = THIS_DIR / "chunking_rule_claude.md"

TEXT_DIR = PROJECT_ROOT / "data" / "text_extracted_letters"
OUT_DIR = PROJECT_ROOT / "data" / "chunks_llm_gpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
PLAN_MAX_TOKENS = int(os.getenv("OPENAI_PLAN_TOKENS", "3000"))
ENRICH_MAX_TOKENS = int(os.getenv("OPENAI_ENRICH_TOKENS", "3000"))


# ------------------------- JSON parsing helpers -------------------------

def _trim_to_jsonish(text: str) -> str:
    """Try to trim leading/trailing noise so json.loads has a chance."""
    start_candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    end_candidates = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
    if not start_candidates or not end_candidates:
        return text
    start = min(start_candidates)
    end = max(end_candidates)
    if end <= start:
        return text
    return text[start : end + 1]


def parse_json_loose(content: str) -> Any:
    """Parse JSON with two-stage fallback.

    1) json.loads(content)
    2) json.loads(trimmed)

    Raises json.JSONDecodeError if both fail.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        trimmed = _trim_to_jsonish(content)
        return json.loads(trimmed)


def salvage_balanced_objects(content: str) -> List[Dict[str, Any]]:
    """Salvage dict objects by scanning for balanced JSON objects.

    Works even if objects contain nested braces.
    """
    decoder = json.JSONDecoder()
    objs: List[Dict[str, Any]] = []
    i = 0
    n = len(content)
    while i < n:
        j = content.find("{", i)
        if j == -1:
            break
        try:
            obj, end = decoder.raw_decode(content[j:])
            if isinstance(obj, dict):
                objs.append(obj)
            i = j + end
        except json.JSONDecodeError:
            i = j + 1
    return objs


# ------------------------- Text helpers -------------------------

def load_rules() -> str:
    if not CHUNKING_RULE_PATH.exists():
        raise FileNotFoundError(f"Cannot find rules file at {CHUNKING_RULE_PATH}")
    return CHUNKING_RULE_PATH.read_text(encoding="utf-8")


def load_letter_text_from_year(year: int) -> Tuple[str, str]:
    path = TEXT_DIR / f"{year}_cleaned.txt"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find cleaned letter for year {year}: {path}")
    return path.read_text(encoding="utf-8"), path.name


def load_letter_text_from_path(path: Path) -> Tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find text file: {path}")
    return path.read_text(encoding="utf-8"), path.name


def split_paragraphs(text: str) -> List[str]:
    """Split on blank lines while preserving intra-paragraph line breaks."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return []
    paras = [p.strip("\n") for p in t.split("\n\n")]
    return [p for p in paras if p.strip()]


def join_paragraphs(paras: List[str], start: int, end: int) -> str:
    return "\n\n".join(paras[start : end + 1]).strip()


def has_financials(text: str) -> bool:
    return bool(re.search(r"(\$|%|\b\d{1,3}(?:,\d{3})+\b|\b\d+\.\d+\b)", text))


def has_quote(text: str) -> bool:
    return '"' in text or "\u201c" in text or "\u201d" in text


def has_table_like(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    numish = sum(1 for ln in lines if re.search(r"\d", ln))
    spaced = sum(1 for ln in lines if re.search(r"\s{2,}", ln))
    return (numish / len(lines) >= 0.6) and (spaced / len(lines) >= 0.5)


# ------------------------- Pass 1: Chunk plan -------------------------

@dataclass
class ChunkPlanItem:
    section_type: str
    section_title: str
    subsection: Optional[str]
    parent_section: Optional[str]
    chunk_type: str
    start_para: int
    end_para: int


def plan_prompt(year: int, source_file: str, rules: str, paragraphs: List[str]) -> Tuple[str, str]:
    """Return (system, user) messages for Pass 1."""
    system = (
        "You are an expert corpus architect for Berkshire Hathaway shareholder letters. "
        "You will produce a CHUNK PLAN only (no chunk_text, no long summaries). "
        "Your output MUST be valid JSON."
    )

    para_block = "\n\n".join([f"[P{idx:04d}] {p}" for idx, p in enumerate(paragraphs)])

    user = f"""
You are chunking the Berkshire Hathaway shareholder letter for year {year}.
Source file: {source_file}

Follow these rules (authoritative):
{rules}

TASK (Pass 1 / Chunk Plan):
- Return ONLY a JSON object with this exact shape:
  {{"plan": [ ... ]}}

Each item in plan MUST include:
- section_type: one of the section types in the rules
- section_title: exact header if present, otherwise a concise descriptive title
- subsection: string or null
- parent_section: string or null
- chunk_type: one of the chunk types in the rules
- start_para: integer paragraph index (inclusive)
- end_para: integer paragraph index (inclusive)

Constraints:
- The plan MUST cover every paragraph exactly once: start at 0 and end at {len(paragraphs)-1}.
- No overlaps and no gaps.
- Respect the boundary rules: do not split stories, tables with context, quotes, lists.
- Target 150–300 words per chunk, but allow larger for full stories / tables-with-context.

IMPORTANT:
- Do NOT include chunk_text.
- Do NOT include contextual_summary or entities.
- Do NOT include any commentary.

PARAGRAPHS (indexed):
{para_block}
""".strip()

    return system, user


def validate_and_normalize_plan(plan: List[Dict[str, Any]], n_paras: int) -> List[ChunkPlanItem]:
    if not isinstance(plan, list) or not plan:
        raise ValueError("Plan must be a non-empty list")

    items: List[ChunkPlanItem] = []
    for i, raw in enumerate(plan):
        if not isinstance(raw, dict):
            raise ValueError(f"Plan item {i} is not an object")
        required = ["section_type", "section_title", "subsection", "parent_section", "chunk_type", "start_para", "end_para"]
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(f"Plan item {i} missing keys: {missing}")
        sp = int(raw["start_para"])
        ep = int(raw["end_para"])
        if sp < 0 or ep < 0 or sp > ep:
            raise ValueError(f"Invalid paragraph span in item {i}: {sp}-{ep}")
        items.append(
            ChunkPlanItem(
                section_type=str(raw["section_type"]),
                section_title=str(raw["section_title"]),
                subsection=None if raw["subsection"] in (None, "null") else str(raw["subsection"]),
                parent_section=None if raw["parent_section"] in (None, "null") else str(raw["parent_section"]),
                chunk_type=str(raw["chunk_type"]),
                start_para=sp,
                end_para=ep,
            )
        )

    items.sort(key=lambda x: x.start_para)

    expected = 0
    for it in items:
        if it.start_para != expected:
            raise ValueError(f"Plan has a gap or overlap at paragraph {expected}; got start_para={it.start_para}")
        expected = it.end_para + 1
    if expected != n_paras:
        raise ValueError(f"Plan does not cover all paragraphs. Covered up to {expected-1}, expected {n_paras-1}")

    return items


# ------------------------- Pass 3: Enrichment -------------------------

ENRICH_FIELDS = [
    "contextual_summary",
    "prev_context",
    "next_context",
    "topics",
    "companies_mentioned",
    "people_mentioned",
    "metrics_discussed",
    "industries",
    "contains_principle",
    "principle_category",
    "principle_statement",
    "contains_example",
    "contains_comparison",
    "retrieval_priority",
    "abstraction_level",
    "time_sensitivity",
    "is_complete_thought",
    "needs_context",
]


def enrichment_prompt(rules: str, year: int, source_file: str, batch: List[Dict[str, Any]]) -> Tuple[str, str]:
    system = (
        "You are an expert analyst of Berkshire Hathaway shareholder letters. "
        "You will enrich already-defined chunks with specific summaries and entities. "
        "Output MUST be valid JSON only."
    )

    user = f"""
Follow these rules (authoritative):
{rules}

TASK (Pass 3 / Enrichment):
You are given a list of chunks (with chunk_id, section labels, and chunk_text). For EACH chunk,
fill ONLY the enrichment fields listed below.

Return ONLY a JSON object with exact shape:
{{"enriched": [ {{"chunk_id": "...", ...fields...}}, ... ]}}

Enrichment fields to output for each chunk_id:
{ENRICH_FIELDS}

Requirements:
- Summaries must be specific (no templates).
- prev_context / next_context must refer to adjacent flow in the letter.
- If contains_principle is false, principle_category and principle_statement MUST be null.
- topics should be 3–6 short tags.
- Arrays must contain strings only.
- Use retrieval_priority in {{high, medium, low}}.
- Use abstraction_level in {{high, medium, low}}.
- Use time_sensitivity in {{high, low}}.

Year = {year}; source_file = {source_file}.

CHUNKS:
{json.dumps(batch, ensure_ascii=False)}
""".strip()
    return system, user


def batch_chunks_for_enrichment(chunks: List[Dict[str, Any]], max_chars: int = 18000) -> List[List[Dict[str, Any]]]:
    """Batch chunks so we don't risk long outputs. Batching is based on input size."""
    batches: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_chars = 0
    for ch in chunks:
        payload = {
            "chunk_id": ch["chunk_id"],
            "section_type": ch["section_type"],
            "section_title": ch["section_title"],
            "subsection": ch.get("subsection"),
            "parent_section": ch.get("parent_section"),
            "chunk_type": ch["chunk_type"],
            "chunk_text": ch["chunk_text"],
        }
        s = json.dumps(payload, ensure_ascii=False)
        if cur and cur_chars + len(s) > max_chars:
            batches.append(cur)
            cur = []
            cur_chars = 0
        cur.append(payload)
        cur_chars += len(s)
    if cur:
        batches.append(cur)
    return batches


# ------------------------- LLM call wrappers -------------------------

def get_openai_client():
    """Create an OpenAI client (modern SDK)."""
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ImportError(
            "OpenAI SDK is not available. Install/upgrade with `pip install -U openai` "
            "or run with MOCK_LLM=1 for a local dry-run."
        ) from e
    return OpenAI()


def mock_llm_response(system: str, user: str) -> str:
    """Very small mock for end-to-end testing."""
    if "Pass 1" in user or "Chunk Plan" in user:
        m = re.search(r"end at (\d+)", user)
        last = int(m.group(1)) if m else 0
        plan = []
        i = 0
        while i <= last:
            start = i
            end = min(i + 1, last)
            plan.append(
                {
                    "section_type": "other",
                    "section_title": "Mock Section",
                    "subsection": None,
                    "parent_section": None,
                    "chunk_type": "philosophy",
                    "start_para": start,
                    "end_para": end,
                }
            )
            i = end + 1
        return json.dumps({"plan": plan}, ensure_ascii=False)

    if "Pass 3" in user or "Enrichment" in user:
        batch = json.loads(re.search(r"CHUNKS:\n(.*)$", user, re.DOTALL).group(1))
        enriched = []
        for b in batch:
            txt = b.get("chunk_text", "")
            enriched.append(
                {
                    "chunk_id": b["chunk_id"],
                    "contextual_summary": "Mock summary that mentions concrete content from this chunk.",
                    "prev_context": "Mock prev context.",
                    "next_context": "Mock next context.",
                    "topics": ["mock", "berkshire", "summary"],
                    "companies_mentioned": [],
                    "people_mentioned": [],
                    "metrics_discussed": [],
                    "industries": [],
                    "contains_principle": True if len(txt.split()) > 10 else False,
                    "principle_category": "risk_management" if len(txt.split()) > 10 else None,
                    "principle_statement": "Mock principle statement." if len(txt.split()) > 10 else None,
                    "contains_example": False,
                    "contains_comparison": False,
                    "retrieval_priority": "medium",
                    "abstraction_level": "medium",
                    "time_sensitivity": "low",
                    "is_complete_thought": True,
                    "needs_context": False,
                }
            )
        return json.dumps({"enriched": enriched}, ensure_ascii=False)

    return json.dumps({}, ensure_ascii=False)


def call_openai_json(
    client: Any,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    debug_path: Path,
) -> Any:
    """Call OpenAI and parse JSON with robust fallback. Saves raw content."""
    if os.getenv("MOCK_LLM") == "1":
        mock = mock_llm_response(system=system, user=user)
        debug_path.write_text(mock, encoding="utf-8")
        return parse_json_loose(mock)

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or ""
    debug_path.write_text(content, encoding="utf-8")

    try:
        return parse_json_loose(content)
    except Exception:
        objs = salvage_balanced_objects(content)
        if objs:
            return {"_salvaged": objs}
        raise


# ------------------------- Assembly & post-processing -------------------------

def assign_chunk_ids(plan_items: List[ChunkPlanItem], year: int) -> List[str]:
    """Create chunk_id = {year}_{section_type}_{sequence:03d}, sequence per section_type."""
    counters: Dict[str, int] = {}
    ids: List[str] = []
    for it in plan_items:
        counters.setdefault(it.section_type, 0)
        counters[it.section_type] += 1
        ids.append(f"{year}_{it.section_type}_{counters[it.section_type]:03d}")
    return ids


def compute_positions(chunks: List[Dict[str, Any]]) -> None:
    n = len(chunks)
    for i, ch in enumerate(chunks):
        ch["position_in_letter"] = 0.0 if n <= 1 else i / (n - 1)

    groups: Dict[Tuple[str, str, Optional[str], Optional[str]], List[int]] = {}
    for i, ch in enumerate(chunks):
        key = (
            ch.get("section_type") or "other",
            ch.get("section_title") or "",
            ch.get("subsection"),
            ch.get("parent_section"),
        )
        groups.setdefault(key, []).append(i)

    for _, idxs in groups.items():
        for pos, i in enumerate(idxs):
            chunks[i]["position_in_section"] = pos
            chunks[i]["total_chunks_in_section"] = len(idxs)


def apply_heuristic_flags(chunks: List[Dict[str, Any]]) -> None:
    for ch in chunks:
        text = ch.get("chunk_text", "") or ""
        ch["has_financials"] = bool(ch.get("has_financials")) or has_financials(text)
        ch["has_quote"] = bool(ch.get("has_quote")) or has_quote(text)
        ch["has_table"] = bool(ch.get("has_table")) or has_table_like(text)


def enforce_counts(chunks: List[Dict[str, Any]]) -> None:
    for ch in chunks:
        txt = ch.get("chunk_text", "") or ""
        ch["word_count"] = len(txt.split())
        ch["char_count"] = len(txt)


def merge_enrichment(chunks: List[Dict[str, Any]], enrich_map: Dict[str, Dict[str, Any]]) -> None:
    for ch in chunks:
        cid = ch["chunk_id"]
        extra = enrich_map.get(cid)
        if not extra:
            raise RuntimeError(f"Missing enrichment for chunk_id={cid}")
        for k in ENRICH_FIELDS:
            if k not in extra:
                raise RuntimeError(f"Enrichment missing field {k} for chunk_id={cid}")
            ch[k] = extra[k]


def validate_final_chunks(chunks: List[Dict[str, Any]]) -> None:
    required = [
        "chunk_id",
        "year",
        "source_file",
        "section_type",
        "section_title",
        "subsection",
        "parent_section",
        "position_in_letter",
        "position_in_section",
        "total_chunks_in_section",
        "chunk_text",
        "word_count",
        "char_count",
        "chunk_type",
        "has_financials",
        "has_table",
        "has_quote",
        "contains_principle",
        "contains_example",
        "contains_comparison",
        "contextual_summary",
        "prev_context",
        "next_context",
        "topics",
        "companies_mentioned",
        "people_mentioned",
        "metrics_discussed",
        "industries",
        "principle_category",
        "principle_statement",
        "retrieval_priority",
        "abstraction_level",
        "time_sensitivity",
        "is_complete_thought",
        "needs_context",
    ]
    for i, ch in enumerate(chunks):
        missing = [k for k in required if k not in ch]
        if missing:
            raise RuntimeError(f"Chunk {i} missing required keys: {missing}")
        if not isinstance(ch["topics"], list):
            raise RuntimeError(f"Chunk {i} topics must be a list")


def write_jsonl(chunks: List[Dict[str, Any]], year: int) -> Path:
    out_path = OUT_DIR / f"{year}_chunks_llm.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    return out_path


# ------------------------- Main pipeline -------------------------

def build_chunks_from_plan(
    plan_items: List[ChunkPlanItem],
    paragraphs: List[str],
    year: int,
    source_file: str,
) -> List[Dict[str, Any]]:
    chunk_ids = assign_chunk_ids(plan_items, year)
    chunks: List[Dict[str, Any]] = []
    for cid, it in zip(chunk_ids, plan_items):
        txt = join_paragraphs(paragraphs, it.start_para, it.end_para)
        chunks.append(
            {
                "chunk_id": cid,
                "year": year,
                "source_file": source_file,
                "section_type": it.section_type,
                "section_title": it.section_title,
                "subsection": it.subsection,
                "parent_section": it.parent_section,
                "paragraph_start": it.start_para,
                "paragraph_end": it.end_para,
                "chunk_text": txt,
                "chunk_type": it.chunk_type,
            }
        )
    return chunks


def run_pipeline(letter_text: str, year: int, source_file: str, model: str) -> List[Dict[str, Any]]:
    rules = load_rules()
    paragraphs = split_paragraphs(letter_text)
    if not paragraphs:
        raise RuntimeError("Letter text is empty after paragraph splitting")

    client = None if os.getenv("MOCK_LLM") == "1" else get_openai_client()

    # Pass 1: plan
    sys1, usr1 = plan_prompt(year, source_file, rules, paragraphs)
    raw_plan_path = OUT_DIR / f"{year}_pass1_plan_raw.json"
    data1 = call_openai_json(client, model, sys1, usr1, PLAN_MAX_TOKENS, raw_plan_path)
    if isinstance(data1, dict) and "plan" in data1:
        plan_raw = data1["plan"]
    else:
        raise RuntimeError(f"Pass 1 did not return {{'plan': [...]}}. See {raw_plan_path}")

    plan_items = validate_and_normalize_plan(plan_raw, n_paras=len(paragraphs))

    # Pass 2: materialize exact text
    chunks = build_chunks_from_plan(plan_items, paragraphs, year, source_file)

    # Pass 3: enrichment (batched)
    batches = batch_chunks_for_enrichment(chunks)
    enrich_map: Dict[str, Dict[str, Any]] = {}
    for bi, batch in enumerate(batches):
        sys3, usr3 = enrichment_prompt(rules, year, source_file, batch)
        raw_enrich_path = OUT_DIR / f"{year}_pass3_enrich_raw_{bi:02d}.json"
        data3 = call_openai_json(client, model, sys3, usr3, ENRICH_MAX_TOKENS, raw_enrich_path)

        if isinstance(data3, dict) and "enriched" in data3 and isinstance(data3["enriched"], list):
            enriched_list = data3["enriched"]
        elif isinstance(data3, dict) and "_salvaged" in data3:
            wrapper = next((o for o in data3["_salvaged"] if isinstance(o, dict) and "enriched" in o), None)
            if not wrapper:
                raise RuntimeError(f"Could not salvage enrichment wrapper. See {raw_enrich_path}")
            enriched_list = wrapper["enriched"]
        else:
            raise RuntimeError(f"Pass 3 returned unexpected JSON. See {raw_enrich_path}")

        for obj in enriched_list:
            if not isinstance(obj, dict) or "chunk_id" not in obj:
                continue
            enrich_map[obj["chunk_id"]] = obj

    merge_enrichment(chunks, enrich_map)

    # Pass 4: deterministic fields + flags
    enforce_counts(chunks)
    compute_positions(chunks)
    apply_heuristic_flags(chunks)

    validate_final_chunks(chunks)
    return chunks


# ------------------------- CLI -------------------------

def _parse_args(argv: List[str]) -> Tuple[int, Optional[Path]]:
    """Supports:
    - python script.py 2009
    - python script.py --year 2009 --text path/to/file.txt
    """
    if len(argv) == 2 and argv[1].isdigit():
        return int(argv[1]), None

    year: Optional[int] = None
    text_path: Optional[Path] = None
    i = 1
    while i < len(argv):
        if argv[i] == "--year":
            year = int(argv[i + 1]); i += 2
        elif argv[i] == "--text":
            text_path = Path(argv[i + 1]); i += 2
        else:
            raise SystemExit("Usage: python chunking_with_llm_gpt_v2.py <year> OR --year <year> --text <file>")
    if year is None:
        raise SystemExit("Missing --year")
    return year, text_path


def main(argv: List[str]) -> None:
    year, text_path = _parse_args(argv)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    if text_path is None:
        letter_text, source_file = load_letter_text_from_year(year)
    else:
        letter_text, source_file = load_letter_text_from_path(text_path)

    print(f"[INFO] Chunking year {year} with model={model} (Pass1+Pass3, batched)...")
    if os.getenv("MOCK_LLM") == "1":
        print("[INFO] MOCK_LLM=1 enabled: no API calls will be made.")

    chunks = run_pipeline(letter_text=letter_text, year=year, source_file=source_file, model=model)
    out_path = write_jsonl(chunks, year)
    print(f"[OK] Wrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main(sys.argv)
