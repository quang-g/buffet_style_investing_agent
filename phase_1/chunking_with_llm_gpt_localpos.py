#!/usr/bin/env python
"""
Use an LLM (OpenAI API) to chunk a single Berkshire letter for a given year,
following the detailed rules in `chunking_strategy.md`.

Directory layout (relative to this file):

    phase_1/
        chunking_strategy.md
        chunk_with_llm.py   <-- THIS FILE
    data/
        text_extracted_letters/
            2008_cleaned.txt
            2009_cleaned.txt
            ...
        chunks_llm/
            2008_chunks_llm.jsonl
            2009_chunks_llm.jsonl
            ...

Usage:

    # Process one year
    python chunk_with_llm.py 2008

    # Optionally override model via env:
    #   export OPENAI_MODEL=gpt-4.1
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

import re


# ------------------------- Paths & Constants -------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

CHUNKING_STRATEGY_PATH = THIS_DIR / "chunking_rule_claude.md"

TEXT_DIR = PROJECT_ROOT / "data" / "text_extracted_letters"
OUT_DIR = PROJECT_ROOT / "data" / "chunks_llm_gpt" / "test2011"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default model; can be overridden via env OPENAI_MODEL
DEFAULT_MODEL = "gpt-4.1-mini"

# ------------------------- Helpers -------------------------

ALLOWED_CHUNK_TYPES = {
    "narrative_story",
    "financial_table",
    "philosophy",
    "business_analysis",
    "administrative",
}

def extract_chunk_objects_from_response(content: str) -> List[Dict[str, Any]]:
    """
    Fallback parser for when the LLM returns 'almost JSON' but the overall
    object/array is truncated or malformed.

    Strategy:
    - Use a regex to find all `{ ... }` blocks that contain `"chunk_id":`.
    - Each match is attempted as a standalone JSON object.
    - If it parses and has 'chunk_id', treat it as a chunk.

    Assumption: individual chunk objects do not contain nested `{}`.
    This holds for our chunk schema (fields are primitives or arrays of primitives).
    """
    pattern = re.compile(
        r'\{[^{}]*"chunk_id"\s*:\s*"[^"]+"[^{}]*\}',
        re.DOTALL
    )

    objs: List[Dict[str, Any]] = []
    for match in pattern.finditer(content):
        obj_str = match.group(0)
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "chunk_id" in obj:
                objs.append(obj)
        except json.JSONDecodeError:
            # Ignore bad candidates, move on
            continue

    return objs

def load_chunking_strategy() -> str:
    if not CHUNKING_STRATEGY_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find chunking_strategy.md at {CHUNKING_STRATEGY_PATH}"
        )
    return CHUNKING_STRATEGY_PATH.read_text(encoding="utf-8")


def load_letter_text(year: int) -> str:
    path = TEXT_DIR / f"{year}_cleaned.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find cleaned letter for year {year}: {path}"
        )
    return path.read_text(encoding="utf-8")


def reconstruct_chunk_text(letter_text: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Best-effort reconstruction of chunk_text from start/end offsets.

    We do NOT override LLM-generated prev_context/next_context.
    """
    n = len(letter_text)
    fixed: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        if not isinstance(ch, dict):
            continue

        # Only reconstruct when missing/blank
        text = ch.get("chunk_text")
        if not isinstance(text, str) or not text.strip():
            s = ch.get("start_char")
            e = ch.get("end_char")
            if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= n:
                ch["chunk_text"] = letter_text[s:e]
            else:
                ch["chunk_text"] = ""

        # Always recompute counts from whatever chunk_text is now
        txt = ch.get("chunk_text") or ""
        ch["char_count"] = len(txt)
        ch["word_count"] = len(txt.split()) if txt else 0

        fixed.append(ch)
    return fixed


def repair_chunks_locally(
    chunks: List[Dict[str, Any]],
    letter_text: str,
    year: int,
    source_file: str,
    *,
    min_words_to_keep: int = 60,
) -> List[Dict[str, Any]]:
    """Guaranteed local repair pass.

    Goals:
    - Ensure chunk_text exists (reconstruct from offsets when possible).
    - Ensure year/source_file are set.
    - Fix position_in_letter into [0,1] deterministically.
    - Normalize chunk_type to allowed values.
    - Merge tiny heading-like chunks into the next chunk within the same section.
    - Recompute position_in_section + total_chunks_in_section after merges.

    Note: prev_context/next_context remain LLM-generated (we do not touch them).
    """
    n = len(letter_text)
    if n == 0:
        return chunks

    # 1) Ensure chunk_text + counts exist
    chunks = reconstruct_chunk_text(letter_text, chunks)

    # 2) Normalize basics & compute deterministic position_in_letter
    for c in chunks:
        c["year"] = year
        c["source_file"] = source_file


        # Ignore any LLM-supplied positional fields; we recompute deterministically.
        c.pop("position_in_letter", None)
        c.pop("position_in_section", None)
        c.pop("total_chunks_in_section", None)

        s = c.get("start_char")
        if isinstance(s, int):
            pos = s / n
            if pos < 0.0:
                pos = 0.0
            elif pos > 1.0:
                pos = 1.0
            c["position_in_letter"] = float(pos)

        ct = c.get("chunk_type")
        if not isinstance(ct, str) or ct not in ALLOWED_CHUNK_TYPES:
            # Default to administrative rather than inventing a new enum.
            c["chunk_type"] = "administrative"

        # Recompute counts again (in case caller mutated chunk_text)
        txt = c.get("chunk_text") or ""
        c["char_count"] = len(txt)
        c["word_count"] = len(txt.split()) if txt else 0

    # 3) Sort by start_char to ensure correct document order
    chunks.sort(key=lambda x: (x.get("start_char", 10**18), x.get("end_char", 10**18)))

    # 4) Merge tiny chunks (typically headings / stubs)
    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(chunks):
        cur = chunks[i]

        # Decide if cur is merge-candidate
        cur_words = cur.get("word_count", 0) or 0
        cur_type = cur.get("chunk_type")
        cur_section = cur.get("section_type")

        can_merge = (
            isinstance(cur_words, int)
            and cur_words < min_words_to_keep
            and cur_type != "financial_table"  # don't merge tables by default
        )

        # Merge into next chunk if same section_type
        if can_merge and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            if nxt.get("section_type") == cur_section:
                # Join texts with a newline boundary (preserves readability)
                cur_txt = (cur.get("chunk_text") or "").strip()
                nxt_txt = (nxt.get("chunk_text") or "").strip()
                joined = (cur_txt + "\n" + nxt_txt).strip() if cur_txt and nxt_txt else (cur_txt or nxt_txt)
                nxt["chunk_text"] = joined

                # Expand offsets to cover both if available
                s1, e1 = cur.get("start_char"), cur.get("end_char")
                s2, e2 = nxt.get("start_char"), nxt.get("end_char")
                if isinstance(s1, int) and isinstance(s2, int):
                    nxt["start_char"] = min(s1, s2)
                if isinstance(e1, int) and isinstance(e2, int):
                    nxt["end_char"] = max(e1, e2)

                # Keep NEXT chunk's metadata as the surviving chunk.
                # We intentionally do NOT try to recompute LLM prev/next_context.
                # Counts will be recomputed later.

                i += 1
                continue

        merged.append(cur)
        i += 1

    chunks = merged

    # 5) Recompute counts + deterministic positions after merges
    for c in chunks:
        txt = c.get("chunk_text") or ""
        c["char_count"] = len(txt)
        c["word_count"] = len(txt.split()) if txt else 0

        s = c.get("start_char")
        if isinstance(s, int):
            pos = s / n
            if pos < 0.0:
                pos = 0.0
            elif pos > 1.0:
                pos = 1.0
            c["position_in_letter"] = float(pos)

    # 6) Recompute section indices (deterministic)
    # Primary grouping: (section_title, subsection). Fallback to section_type.
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

    return chunks


def call_llm_for_chunks(
    letter_text: str,
    year: int,
    source_file: str,
    chunking_spec: str,
    model: str,
) -> List[Dict[str, Any]]:
    """
    Call the OpenAI Chat Completions API and ask it to:

    - Read the full letter text for a given year.
    - Apply all rules from chunking_strategy.md.
    - Return a JSON object with a "chunks" array containing fully-populated
      chunk objects that match the "Complete Metadata Structure" in the spec.

    If the overall JSON is truncated or malformed (common when the output is
    very large), fall back to scanning the response and extracting all
    *individually valid* chunk objects.
    """
    client = OpenAI()

    system_content = """
        You are an expert corpus architect for Berkshire Hathaway shareholder letters.
        Your job is to chunk a single letter into semantically meaningful chunks and produce rich metadata.
        Follow EXACTLY the chunking rule, definitions, and metadata schema in the chunking strategy document below.

        Hard constraints:
        - Return ONLY valid JSON (no prose).
        - Top-level MUST be a JSON object: {"chunks": [ ... ]}.
        - Include accurate start_char/end_char offsets into the provided LETTER TEXT.
        - DO generate prev_context and next_context summaries.
        - DO NOT include chunk_text/word_count/char_count (they will be reconstructed locally).
        - DO NOT include position_in_letter/position_in_section/total_chunks_in_section (they will be computed locally).
    """

    strategy_message = (
        "Here is the complete chunking rule specification you MUST follow:\n\n"
        f"{chunking_spec}"
    )

    user_content = f"""
Chunk the Berkshire Hathaway shareholder letter for year {year}.

Source file name: {source_file}

Use the following requirements:

- Apply all rules from the chunking rule document you received.
- Identify sections, subsections, and the different chunk types.
- Generate chunks of ~150–300 words, except where the rule specifies
  larger/smaller sizes (e.g., front performance tables).
- Ensure each chunk is a complete thought and respects narrative flow.
- Fill in all required metadata fields for each chunk:

```json
[
  {{
    "chunk_id": "{{year}}_{{section_type}}_{{sequence:03d}}",
    "year": {year},
    "source_file": "{source_file}",
    "section_type": "string (one of: performance_overview, insurance_operations, acquisitions, investments, operating_businesses, corporate_governance, management_philosophy, shareholder_matters, other)",
    "section_title": "string",
    "subsection": "string or null",
    "parent_section": "string or null",    "start_char": "int (0-indexed, inclusive offset into LETTER TEXT)",
    "end_char": "int (0-indexed, exclusive offset into LETTER TEXT)",
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
- Use `year = {year}` and `source_file = "{source_file}"` in every chunk.
- Provide accurate `start_char` and `end_char` offsets.
- DO generate `prev_context` and `next_context` summaries.
- DO NOT include `chunk_text`, `word_count`, or `char_count`.
- Prefer this outer JSON shape, but a top-level array is also accepted:

{{
  "chunks": [
    {{ /* first chunk */ }},
    {{ /* second chunk */ }},
    ...
  ]
}}

LETTER TEXT START
{letter_text}
LETTER TEXT END
"""

    # Ask the model to output JSON
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

    # Always save raw response for debugging
    raw_debug_path = OUT_DIR / f"{year}_raw_llm_response.json"
    raw_debug_path.write_text(content, encoding="utf-8")

    # --- 1) Try to parse whole thing normally ---
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
        # Fall through to salvage mode
        pass

    # --- 2) Fallback: trimmed parse (sometimes the model wraps JSON in text) ---
    try:
        start_brace = content.find("{")
        start_bracket = content.find("[")
        start_candidates = [i for i in [start_brace, start_bracket] if i != -1]
        end_brace = content.rfind("}")
        end_bracket = content.rfind("]")
        end_candidates = [i for i in [end_brace, end_bracket] if i != -1]

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
        # Fall through to salvage mode
        pass

    # --- 3) Final fallback: scan and salvage individual chunk objects ---
    salvaged = extract_chunk_objects_from_response(content)
    if not salvaged:
        raise RuntimeError(
            f"LLM returned invalid or unusable JSON, and no chunk objects "
            f"could be salvaged. See raw response in: {raw_debug_path}"
        )

    print(
        f"[WARN] Whole-response JSON parse failed. "
        f"Salvaged {len(salvaged)} chunk objects from {raw_debug_path}"
    )
    return salvaged




def postprocess_chunks(
    chunks: List[Dict[str, Any]],
    year: int,
    source_file: str,
) -> List[Dict[str, Any]]:
    """
    Light post-processing:

    - Ensure `year` and `source_file` are set.
    - Recompute `word_count` and `char_count` from `chunk_text` to be safe.
    """
    for c in chunks:
        c.setdefault("year", year)
        c.setdefault("source_file", source_file)

        text = c.get("chunk_text", "") or ""
        # Recompute counts (LLM may be slightly off)
        c["word_count"] = len(text.split())
        c["char_count"] = len(text)

    return chunks


def write_chunks_jsonl(
    chunks: List[Dict[str, Any]],
    year: int,
) -> Path:
    out_path = OUT_DIR / f"{year}_chunks_llm.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out_path


# ------------------------- CLI -------------------------


def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python chunk_with_llm.py <year>")
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

    # 1) Ask LLM to chunk + attach metadata
    chunks_raw = call_llm_for_chunks(
        letter_text=letter_text,
        year=year,
        source_file=source_file_name,
        chunking_spec=chunking_spec,
        model=model,
    )

    # 2) Guaranteed local repair (schema stability + tiny-chunk merge + deterministic positions)
    chunks = repair_chunks_locally(
        chunks_raw,
        letter_text=letter_text,
        year=year,
        source_file=source_file_name,
    )

    # 3) Write JSONL
    out_path = write_chunks_jsonl(chunks, year)
    print(f"[OK] Wrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main(sys.argv)
