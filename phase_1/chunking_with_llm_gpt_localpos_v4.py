#!/usr/bin/env python
"""chunking_with_llm_gpt_localpos_v4.py

Use an LLM (OpenAI API) to chunk a single Berkshire letter for a given year,
following the detailed rules in `chunking_rule_claude.md`.

v4 upgrades (fixes common failures in LLM offset chunking):

1) Anchored offsets (self-healing)
   - The LLM MUST return `start_anchor` (first ~60 chars of the chunk) and
     `end_anchor` (last ~60 chars of the chunk), copied EXACTLY from the LETTER TEXT.
   - Locally, we validate that the [start_char:end_char] substring matches anchors.
     If not, we repair offsets by searching anchors in the letter.

2) Strong boundary snapping
   - Replaces the weak “lowercase start” heuristic with a “safe boundary” rule:
     when a boundary looks unsafe, we snap it backward to the nearest sentence
     break (or at least a whitespace boundary) and adjust prev.end / cur.start.

3) Diagnostics
   - Writes a small JSON diagnostics file for each year (offset fixes, unsafe boundaries, etc.).

Directory layout (relative to this file):

    phase_1/
        chunking_rule_claude.md
        chunking_with_llm_gpt_localpos_v4.py   <-- THIS FILE
    data/
        text_extracted_letters/
            2008_cleaned.txt
            ...
        chunks_llm_gpt/
            localpos_v4/
                2008_chunks_llm.jsonl
                2008_offset_diagnostics.json
                2008_raw_llm_response.json
                ...

Usage:
    python chunking_with_llm_gpt_localpos_v4.py 2008

Env:
    OPENAI_MODEL (default: gpt-4.1-mini)
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ------------------------- Paths & Constants -------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

CHUNKING_STRATEGY_PATH = THIS_DIR / "chunking_rule_claude.md"

TEXT_DIR = PROJECT_ROOT / "data" / "text_extracted_letters"
OUT_DIR = PROJECT_ROOT / "data" / "chunks_llm_gpt" / "localpos_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "gpt-4.1-mini"

ALLOWED_CHUNK_TYPES = {
    "narrative_story",
    "financial_table",
    "philosophy",
    "business_analysis",
    "administrative",
}

SENT_END = {".", "!", "?"}


# ------------------------- Generic helpers -------------------------


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _norm_ws(s: str) -> str:
    """Whitespace-normalized string used for anchor matching."""
    return re.sub(r"\s+", " ", s).strip()


def extract_chunk_objects_from_response(content: str) -> List[Dict[str, Any]]:
    """Fallback parser when LLM returns malformed / truncated JSON."""
    pattern = re.compile(r'\{[^{}]*"chunk_id"\s*:\s*"[^"]+"[^{}]*\}', re.DOTALL)

    objs: List[Dict[str, Any]] = []
    for match in pattern.finditer(content):
        obj_str = match.group(0)
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "chunk_id" in obj:
                objs.append(obj)
        except json.JSONDecodeError:
            continue

    return objs


def load_chunking_strategy() -> str:
    if not CHUNKING_STRATEGY_PATH.exists():
        raise FileNotFoundError(f"Cannot find chunking rule at {CHUNKING_STRATEGY_PATH}")
    return CHUNKING_STRATEGY_PATH.read_text(encoding="utf-8")


def load_letter_text(year: int) -> str:
    path = TEXT_DIR / f"{year}_cleaned.txt"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find cleaned letter for year {year}: {path}")
    return path.read_text(encoding="utf-8")


# ------------------------- Offset anchoring / repair -------------------------


def _validate_span_with_anchors(
    letter_text: str,
    start: int,
    end: int,
    start_anchor: Optional[str],
    end_anchor: Optional[str],
) -> bool:
    """Return True if span seems consistent with anchors (whitespace-normalized)."""
    if start < 0 or end < 0 or start >= end or end > len(letter_text):
        return False

    span = letter_text[start:end]
    span_n = _norm_ws(span)

    if start_anchor:
        sa = _norm_ws(start_anchor)
        if len(sa) >= 8 and not span_n.startswith(sa):
            return False

    if end_anchor:
        ea = _norm_ws(end_anchor)
        if len(ea) >= 8 and not span_n.endswith(ea):
            return False

    return True


def _find_all(haystack: str, needle: str) -> List[int]:
    """Return all start indices where needle occurs in haystack."""
    if not needle:
        return []
    out: List[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        out.append(idx)
        start = idx + 1
    return out


def _repair_offsets_by_raw_anchors(
    letter_text: str,
    start_anchor: str,
    end_anchor: str,
    *,
    max_candidates: int = 50,
    max_span: int = 120_000,
) -> Optional[Tuple[int, int]]:
    """Try repair in raw text space first (best precision)."""
    if not start_anchor or not end_anchor:
        return None

    sa = start_anchor.strip()
    ea = end_anchor.strip()
    if len(sa) < 8 or len(ea) < 8:
        return None

    start_positions = _find_all(letter_text, sa)
    if not start_positions:
        return None

    if len(start_positions) > max_candidates:
        start_positions = start_positions[:max_candidates]

    best: Optional[Tuple[int, int]] = None
    for s in start_positions:
        search_from = s + max(1, len(sa) // 2)
        e = letter_text.find(ea, search_from)
        if e == -1:
            continue
        end_excl = e + len(ea)
        if end_excl <= s:
            continue
        if end_excl - s > max_span:
            continue

        if best is None or (end_excl - s) < (best[1] - best[0]):
            best = (s, end_excl)

    return best


def _repair_offsets_by_normalized_anchors(
    letter_text: str,
    start_anchor: str,
    end_anchor: str,
    *,
    max_candidates: int = 80,
) -> Optional[Tuple[int, int]]:
    """Fallback repair using shorter raw hints derived from normalized anchors."""
    sa_norm = _norm_ws(start_anchor)
    ea_norm = _norm_ws(end_anchor)
    if len(sa_norm) < 8 or len(ea_norm) < 8:
        return None

    # Take raw hints to map back safely.
    sa_hint = start_anchor.strip()[:40]
    ea_hint = end_anchor.strip()[-40:]
    if len(sa_hint) < 8 or len(ea_hint) < 8:
        return None

    # Raw-hint search is usually enough. Keep candidates bounded.
    return _repair_offsets_by_raw_anchors(letter_text, sa_hint, ea_hint, max_candidates=max_candidates)


def resolve_offsets_with_anchors(
    chunks: List[Dict[str, Any]],
    letter_text: str,
    *,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Validate and repair offsets using anchors when available."""
    n = len(letter_text)
    fixed = 0
    attempted = 0
    low_confidence = 0

    for c in chunks:
        s = c.get("start_char")
        e = c.get("end_char")
        sa = c.get("start_anchor")
        ea = c.get("end_anchor")

        sa = sa if isinstance(sa, str) and sa.strip() else None
        ea = ea if isinstance(ea, str) and ea.strip() else None

        if not sa or not ea:
            low_confidence += 1
            continue

        attempted += 1

        if not (isinstance(s, int) and isinstance(e, int)):
            s = None
            e = None
        else:
            s = _clamp_int(s, 0, n)
            e = _clamp_int(e, 0, n)

        if isinstance(s, int) and isinstance(e, int) and _validate_span_with_anchors(letter_text, s, e, sa, ea):
            continue

        repaired = _repair_offsets_by_raw_anchors(letter_text, sa, ea)
        if repaired is None:
            repaired = _repair_offsets_by_normalized_anchors(letter_text, sa, ea)

        if repaired is None:
            c["offset_confidence"] = "low"
            low_confidence += 1
            continue

        rs, re_ = repaired
        rs = _clamp_int(rs, 0, n)
        re_ = _clamp_int(re_, 0, n)
        if rs >= re_:
            c["offset_confidence"] = "low"
            low_confidence += 1
            continue

        c["start_char"] = rs
        c["end_char"] = re_
        c["offset_confidence"] = "repaired"
        fixed += 1

    if diagnostics is not None:
        diagnostics["anchors_attempted"] = attempted
        diagnostics["anchors_fixed"] = fixed
        diagnostics["anchors_low_confidence"] = low_confidence

    return chunks


# ------------------------- Chunk text reconstruction -------------------------


def reconstruct_chunk_text(letter_text: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reconstruct chunk_text from offsets (best effort). Also recompute counts."""
    n = len(letter_text)
    fixed: List[Dict[str, Any]] = []

    for ch in chunks:
        if not isinstance(ch, dict):
            continue

        s = ch.get("start_char")
        e = ch.get("end_char")

        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= n:
            ch["chunk_text"] = letter_text[s:e]
        else:
            ch["chunk_text"] = ""

        txt = ch.get("chunk_text") or ""
        ch["char_count"] = len(txt)
        ch["word_count"] = len(txt.split()) if txt else 0

        fixed.append(ch)

    return fixed


# ------------------------- Boundary snapping -------------------------


def _last_nonspace_char(letter_text: str, pos: int) -> Optional[str]:
    n = len(letter_text)
    k = _clamp_int(pos, 0, n) - 1
    while k >= 0 and letter_text[k].isspace():
        k -= 1
    return letter_text[k] if k >= 0 else None


def _is_safe_boundary(letter_text: str, pos: int) -> bool:
    n = len(letter_text)
    pos = _clamp_int(pos, 0, n)
    if pos == 0:
        return True

    # Paragraph break: boundary after blank line.
    if pos >= 2 and letter_text[pos - 1] == "\n" and letter_text[pos - 2] == "\n":
        return True

    last_prev = _last_nonspace_char(letter_text, pos)
    if last_prev in SENT_END:
        return True

    return False


def _find_sentence_break_before(letter_text: str, pos: int, window: int = 1200) -> Optional[int]:
    n = len(letter_text)
    pos = _clamp_int(pos, 0, n)
    left = max(0, pos - window)
    snippet = letter_text[left:pos]
    matches = list(re.finditer(r'[.!?]["\')\]]*[\s\n\r\t]+', snippet))
    if not matches:
        return None
    m = matches[-1]
    return left + m.end()


def _snap_to_whitespace_before(letter_text: str, pos: int, window: int = 800) -> int:
    n = len(letter_text)
    pos = _clamp_int(pos, 0, n)
    left = max(0, pos - window)
    k = pos
    while k > left:
        if letter_text[k - 1].isspace():
            return k
        k -= 1
    return pos


def smooth_boundaries(
    chunks: List[Dict[str, Any]],
    letter_text: str,
    *,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Adjust boundaries so chunks don't start mid-word and boundaries are safer."""
    n = len(letter_text)
    if not chunks:
        return chunks

    chunks.sort(key=lambda x: (x.get("start_char", 10**18), x.get("end_char", 10**18)))

    midword_fixes = 0
    unsafe_fixes = 0

    for j in range(1, len(chunks)):
        prev = chunks[j - 1]
        cur = chunks[j]

        pe = prev.get("end_char")
        cs = cur.get("start_char")
        ce = cur.get("end_char")

        if not (isinstance(pe, int) and isinstance(cs, int) and isinstance(ce, int)):
            continue

        pe = _clamp_int(pe, 0, n)
        cs = _clamp_int(cs, 0, n)
        ce = _clamp_int(ce, 0, n)
        if cs > ce:
            continue

        # A) Mid-word boundary
        if cs > 0 and cs < n and letter_text[cs - 1].isalnum() and letter_text[cs].isalnum():
            boundary = cs
            while boundary > 0 and letter_text[boundary - 1].isalnum() and letter_text[boundary].isalnum():
                boundary -= 1
            if boundary != cs:
                prev["end_char"] = boundary
                cur["start_char"] = boundary
                midword_fixes += 1
                continue

        # B) Unsafe boundary snapping
        if not _is_safe_boundary(letter_text, cs):
            sb = _find_sentence_break_before(letter_text, cs)
            boundary = sb if sb is not None else _snap_to_whitespace_before(letter_text, cs)
            boundary = _clamp_int(boundary, 0, n)
            if boundary != cs and boundary <= cs:
                prev["end_char"] = boundary
                cur["start_char"] = boundary
                unsafe_fixes += 1

    remaining_unsafe = 0
    for j in range(1, len(chunks)):
        cs = chunks[j].get("start_char")
        if isinstance(cs, int) and not _is_safe_boundary(letter_text, cs):
            remaining_unsafe += 1

    if diagnostics is not None:
        diagnostics["midword_boundary_fixes"] = midword_fixes
        diagnostics["unsafe_boundary_fixes"] = unsafe_fixes
        diagnostics["remaining_unsafe_boundaries"] = remaining_unsafe

    return chunks


# ------------------------- Local repair + merging -------------------------


def repair_chunks_locally(
    chunks: List[Dict[str, Any]],
    letter_text: str,
    year: int,
    source_file: str,
    *,
    min_words_to_keep: int = 60,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"year": year, "source_file": source_file}
    n = len(letter_text)
    if n == 0:
        return chunks, diagnostics

    # Normalize basics
    for c in chunks:
        if not isinstance(c, dict):
            continue

        c["year"] = year
        c["source_file"] = source_file

        c.pop("position_in_letter", None)
        c.pop("position_in_section", None)
        c.pop("total_chunks_in_section", None)

        ct = c.get("chunk_type")
        if not isinstance(ct, str) or ct not in ALLOWED_CHUNK_TYPES:
            c["chunk_type"] = "administrative"

    # Sort by offsets
    chunks.sort(key=lambda x: (x.get("start_char", 10**18), x.get("end_char", 10**18)))

    # Detect and normalize inclusive end_char -> exclusive
    deltas: List[int] = []
    for j in range(len(chunks) - 1):
        a = chunks[j].get("end_char")
        b = chunks[j + 1].get("start_char")
        if isinstance(a, int) and isinstance(b, int):
            deltas.append(b - a)

    if deltas:
        frac_plus_one = sum(1 for d in deltas if d == 1) / max(1, len(deltas))
        diagnostics["endchar_plus_one_fraction"] = frac_plus_one
        if frac_plus_one >= 0.40:
            for c in chunks:
                e = c.get("end_char")
                if isinstance(e, int):
                    c["end_char"] = _clamp_int(e + 1, 0, n)

    # Anchor-based repair
    chunks = resolve_offsets_with_anchors(chunks, letter_text, diagnostics=diagnostics)

    # Boundary smoothing
    chunks = smooth_boundaries(chunks, letter_text, diagnostics=diagnostics)

    # Reconstruct text
    chunks = reconstruct_chunk_text(letter_text, chunks)

    # Deterministic position_in_letter
    for c in chunks:
        s = c.get("start_char")
        if isinstance(s, int):
            pos = s / n
            c["position_in_letter"] = float(_clamp_int(int(pos * 1_000_000), 0, 1_000_000) / 1_000_000)

    # Merge tiny chunks
    merged: List[Dict[str, Any]] = []
    i = 0
    merges = 0
    while i < len(chunks):
        cur = chunks[i]
        cur_words = int(cur.get("word_count") or 0)
        cur_type = cur.get("chunk_type")
        cur_section = cur.get("section_type")

        can_merge = cur_words < min_words_to_keep and cur_type != "financial_table"
        if can_merge and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            if nxt.get("section_type") == cur_section:
                s1, e1 = cur.get("start_char"), cur.get("end_char")
                s2, e2 = nxt.get("start_char"), nxt.get("end_char")
                if isinstance(s1, int) and isinstance(s2, int):
                    nxt["start_char"] = min(s1, s2)
                if isinstance(e1, int) and isinstance(e2, int):
                    nxt["end_char"] = max(e1, e2)
                nxt["chunk_text"] = ""
                merges += 1
                i += 1
                continue

        merged.append(cur)
        i += 1

    diagnostics["tiny_chunk_merges"] = merges
    chunks = merged

    # Reconstruct after merges
    chunks = reconstruct_chunk_text(letter_text, chunks)

    # Recompute deterministic fields
    for c in chunks:
        txt = c.get("chunk_text") or ""
        c["char_count"] = len(txt)
        c["word_count"] = len(txt.split()) if txt else 0

        s = c.get("start_char")
        if isinstance(s, int):
            pos = s / n
            c["position_in_letter"] = float(_clamp_int(int(pos * 1_000_000), 0, 1_000_000) / 1_000_000)

    # Recompute section indices
    def _section_key(c: Dict[str, Any]) -> tuple:
        stitle = (c.get("section_title") or "").strip()
        sub = (c.get("subsection") or "").strip()
        stype = (c.get("section_type") or "other").strip()
        if stitle or sub:
            return (stitle, sub, stype)
        return ("", "", stype)

    section_counts: Dict[tuple, int] = {}
    for c in chunks:
        k = _section_key(c)
        section_counts[k] = section_counts.get(k, 0) + 1

    section_seen: Dict[tuple, int] = {}
    for c in chunks:
        k = _section_key(c)
        idx_in_section = section_seen.get(k, 0)
        c["position_in_section"] = int(idx_in_section)
        c["total_chunks_in_section"] = int(section_counts.get(k, 1))
        section_seen[k] = idx_in_section + 1

    return chunks, diagnostics


# ------------------------- LLM call -------------------------


def call_llm_for_chunks(
    letter_text: str,
    year: int,
    source_file: str,
    chunking_spec: str,
    model: str,
) -> List[Dict[str, Any]]:
    client = OpenAI()

    system_content = """
You are an expert corpus architect for Berkshire Hathaway shareholder letters.
Your job is to chunk a single letter into semantically meaningful chunks and produce rich metadata.
Follow EXACTLY the chunking rule, definitions, and metadata schema in the chunking strategy document.

Hard constraints:
- Return ONLY valid JSON (no prose).
- Top-level MUST be a JSON object: {"chunks": [ ... ]}.
- Include accurate start_char/end_char offsets into the provided LETTER TEXT.
- Also include start_anchor and end_anchor copied EXACTLY from the LETTER TEXT.
- DO generate prev_context and next_context summaries.
- DO NOT include chunk_text/word_count/char_count (they will be reconstructed locally).
- DO NOT include position_in_letter/position_in_section/total_chunks_in_section (they will be computed locally).
"""

    strategy_message = (
        "Here is the complete chunking rule specification you MUST follow:\n\n" + chunking_spec
    )

    user_content = f"""
Chunk the Berkshire Hathaway shareholder letter for year {year}.

Source file name: {source_file}

Output schema (each chunk object MUST include these fields):
```json
[
  {{
    "chunk_id": "{year}_{{section_type}}_{{sequence:03d}}",
    "year": {year},
    "source_file": "{source_file}",
    "section_type": "string (one of: performance_overview, insurance_operations, acquisitions, investments, operating_businesses, corporate_governance, management_philosophy, shareholder_matters, other)",
    "section_title": "string",
    "subsection": "string or null",
    "parent_section": "string or null",
    "start_char": "int (0-indexed, inclusive offset into LETTER TEXT)",
    "end_char": "int (0-indexed, exclusive offset into LETTER TEXT)",
    "start_anchor": "string (first ~60 characters of the chunk text, copied EXACTLY from LETTER TEXT)",
    "end_anchor": "string (last ~60 characters of the chunk text, copied EXACTLY from LETTER TEXT)",
    "chunk_type": "string (one of: narrative_story, financial_table, philosophy, business_analysis, administrative)",
    "has_financials": "bool",
    "has_table": "bool",
    "has_quote": "bool",
    "contains_principle": "bool",
    "contains_example": "bool",
    "contains_comparison": "bool",
    "contextual_summary": "string (2-3 sentences per spec)",
    "prev_context": "string (1-2 sentences summarizing preceding content, empty for first chunk)",
    "next_context": "string (1-2 sentences summarizing following content, empty for last chunk)",
    "topics": ["array", "of", "topic", "strings"],
    "companies_mentioned": ["array", "of", "company", "names"],
    "people_mentioned": ["array", "of", "people", "names"],
    "metrics_discussed": ["array", "of", "metric", "names"],
    "industries": ["array", "of", "industry", "names"],
    "principle_category": "string or null (if contains_principle: moats, valuation, management_quality, capital_allocation, risk_management, competitive_advantage, business_quality)",
    "principle_statement": "string or null (if contains_principle)",
    "retrieval_priority": "string (high, medium, low)",
    "abstraction_level": "string (high, medium, low)",
    "time_sensitivity": "string (high, low)",
    "is_complete_thought": "bool",
    "needs_context": "bool"
  }}
]
```

IMPORTANT:
- Anchors MUST be exact text from LETTER TEXT (preserve punctuation; do not paraphrase).
- DO NOT include chunk_text, word_count, char_count.
- DO NOT include position_in_letter/position_in_section/total_chunks_in_section.

LETTER TEXT START
{letter_text}
LETTER TEXT END
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_content},
            {"role": "system", "content": strategy_message},
            {"role": "user", "content": user_content},
        ],
    )

    content = response.choices[0].message.content or ""

    raw_debug_path = OUT_DIR / f"{year}_raw_llm_response.json"
    raw_debug_path.write_text(content, encoding="utf-8")

    # Parse
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "chunks" in data:
            chunks = data["chunks"]
        elif isinstance(data, list):
            chunks = data
        else:
            raise ValueError("Top-level JSON is neither {'chunks': [...]} nor a list.")
        if not isinstance(chunks, list):
            raise ValueError('"chunks" is not a list.')
        return chunks
    except Exception:
        pass

    # Trim parse
    try:
        start_brace = content.find("{")
        start_bracket = content.find("[")
        start_candidates = [i for i in (start_brace, start_bracket) if i != -1]
        end_brace = content.rfind("}")
        end_bracket = content.rfind("]")
        end_candidates = [i for i in (end_brace, end_bracket) if i != -1]

        if start_candidates and end_candidates:
            start = min(start_candidates)
            end = max(end_candidates)
            trimmed = content[start : end + 1]
            data = json.loads(trimmed)
            if isinstance(data, dict) and "chunks" in data:
                chunks = data["chunks"]
            elif isinstance(data, list):
                chunks = data
            else:
                raise ValueError("Trimmed JSON has unexpected top-level structure.")
            if not isinstance(chunks, list):
                raise ValueError('"chunks" is not a list.')
            return chunks
    except Exception:
        pass

    # Salvage objects
    salvaged = extract_chunk_objects_from_response(content)
    if not salvaged:
        raise RuntimeError(
            "LLM returned invalid or unusable JSON, and no chunk objects could be salvaged. "
            f"See raw response: {raw_debug_path}"
        )

    print(f"[WARN] Whole-response JSON parse failed. Salvaged {len(salvaged)} chunk objects from {raw_debug_path}")
    return salvaged


# ------------------------- Output -------------------------


def write_chunks_jsonl(chunks: List[Dict[str, Any]], year: int) -> Path:
    out_path = OUT_DIR / f"{year}_chunks_llm.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out_path


def write_diagnostics_json(diagnostics: Dict[str, Any], year: int) -> Path:
    out_path = OUT_DIR / f"{year}_offset_diagnostics.json"
    out_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ------------------------- CLI -------------------------


def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python chunking_with_llm_gpt_localpos_v4.py <year>")
        sys.exit(1)

    try:
        year = int(argv[1])
    except ValueError:
        print("Year must be an integer, e.g. 2008")
        sys.exit(1)

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    print(f"[INFO] Chunking letter for year {year} using model {model}...")

    chunking_spec = load_chunking_strategy()
    letter_text = load_letter_text(year)
    source_file_name = f"{year}_cleaned.txt"

    chunks_raw = call_llm_for_chunks(
        letter_text=letter_text,
        year=year,
        source_file=source_file_name,
        chunking_spec=chunking_spec,
        model=model,
    )

    chunks, diagnostics = repair_chunks_locally(
        chunks_raw,
        letter_text=letter_text,
        year=year,
        source_file=source_file_name,
    )

    out_path = write_chunks_jsonl(chunks, year)
    diag_path = write_diagnostics_json(diagnostics, year)

    print(f"[OK] Wrote {len(chunks)} chunks to {out_path}")
    print(f"[OK] Wrote diagnostics to {diag_path}")


if __name__ == "__main__":
    main(sys.argv)
