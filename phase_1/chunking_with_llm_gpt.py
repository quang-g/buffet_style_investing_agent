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
OUT_DIR = PROJECT_ROOT / "data" / "chunks_llm_gpt" / "test2009"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default model; can be overridden via env OPENAI_MODEL
DEFAULT_MODEL = "gpt-4.1-mini"

# ------------------------- Helpers -------------------------

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

def call_llm_for_chunks(
    letter_text: str,
    year: int,
    source_file: str,
    chunking_spec: str,
    model: str,
) -> List[Dict[str, Any]]:
    """
    Robust chunking:
    - Ask LLM for chunk boundaries (start_char/end_char) + metadata (NO chunk_text).
    - Reconstruct chunk_text locally from letter_text to avoid huge/truncated JSON outputs.
    """
    client = OpenAI()

    def reconstruct_chunk_text(letter_text: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n = len(letter_text)
        fixed: List[Dict[str, Any]] = []
        for i, ch in enumerate(chunks):
            if not isinstance(ch, dict):
                continue

            s = ch.get("start_char")
            e = ch.get("end_char")

            if not isinstance(s, int) or not isinstance(e, int):
                raise RuntimeError(f"Chunk {i} missing start_char/end_char. Got start_char={s}, end_char={e}")

            if s < 0 or e < 0 or s >= e or s > n or e > n:
                raise RuntimeError(f"Chunk {i} has invalid offsets: start_char={s}, end_char={e}, letter_len={n}")

            text = letter_text[s:e]
            ch["chunk_text"] = text

            # Compute counts locally (more reliable than LLM)
            ch["char_count"] = len(text)
            ch["word_count"] = len(text.split())

            fixed.append(ch)
        return fixed

    system_content = """
You are an expert corpus architect for Berkshire Hathaway shareholder letters.
Your job is to chunk a single letter into semantically meaningful chunks and produce rich metadata.

Follow EXACTLY the chunking rule, definitions, and metadata schema in the chunking strategy document below.

Hard constraints:
- Return ONLY valid JSON (no prose).
- Top-level MUST be a JSON object: { "chunks": [ ... ] }.
- DO NOT include chunk_text in the output.
- Instead, include start_char and end_char offsets into the provided LETTER TEXT.
"""

    strategy_message = (
        "Here is the complete chunking rule specification you MUST follow:\n\n"
        f"{chunking_spec}"
    )

    # IMPORTANT: We explicitly avoid chunk_text to prevent huge outputs that get truncated.
    user_content = f"""
Chunk the Berkshire Hathaway shareholder letter for year {year}.
Source file name: {source_file}

Requirements:
- Apply all rules from the chunking rule document you received.
- Identify sections, subsections, and chunk types.
- Create chunks of ~150–300 words unless the rule specifies otherwise.
- Each chunk must be a complete thought and respect narrative flow.

Output format (STRICT):
Return a JSON object with this shape:

{{
  "chunks": [
    {{
      "chunk_id": "{year}_{{section_type}}_{{sequence:03d}}",
      "year": {year},
      "source_file": "{source_file}",

      "section_type": "one of: performance_overview, insurance_operations, acquisitions, investments, operating_businesses, corporate_governance, management_philosophy, shareholder_matters, other",
      "section_title": "string",
      "subsection": "string or null",
      "parent_section": "string or null",

      "position_in_letter": "float 0.0-1.0",
      "position_in_section": "int (0-indexed)",
      "total_chunks_in_section": "int",

      "start_char": "int (0-indexed, inclusive offset into LETTER TEXT)",
      "end_char": "int (0-indexed, exclusive offset into LETTER TEXT)",

      "chunk_type": "one of: narrative_story, financial_table, philosophy, business_analysis, administrative",
      "has_financials": "bool",
      "has_table": "bool",
      "has_quote": "bool",
      "contains_principle": "bool",
      "contains_example": "bool",
      "contains_comparison": "bool",

      "contextual_summary": "string (2-3 sentences per spec)",
      "prev_context": "string (1-2 sentences, empty for first chunk)",
      "next_context": "string (1-2 sentences, empty for last chunk)",

      "topics": ["array", "of", "topic", "strings"],
      "companies_mentioned": ["array", "of", "company", "names"],
      "people_mentioned": ["array", "of", "people", "names"],
      "metrics_discussed": ["array", "of", "metric", "names"],
      "industries": ["array", "of", "industry", "names"],

      "principle_category": "string or null (if contains_principle: moats, valuation, management_quality, capital_allocation, risk_management, competitive_advantage, business_quality)",
      "principle_statement": "string or null (if contains_principle)",

      "retrieval_priority": "high|medium|low",
      "abstraction_level": "high|medium|low",
      "time_sensitivity": "high|low",
      "is_complete_thought": "bool",
      "needs_context": "bool"
    }}
  ]
}}

Important:
- Use year={year} and source_file="{source_file}" in every chunk.
- start_char/end_char must exactly slice the LETTER TEXT (no hallucinated text).
- DO NOT return chunk_text / word_count / char_count (we compute locally).

LETTER TEXT START
{letter_text}
LETTER TEXT END
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_content.strip()},
            {"role": "system", "content": strategy_message},
            {"role": "user", "content": user_content},
        ],
    )

    content = response.choices[0].message.content or ""

    # Save raw response (txt is more useful when truncated)
    raw_debug_path_json = OUT_DIR / f"{year}_raw_llm_response.json"
    # raw_debug_path_txt = OUT_DIR / f"{year}_raw_llm_response.txt"
    raw_debug_path_json.write_text(content, encoding="utf-8")
    # raw_debug_path_txt.write_text(content, encoding="utf-8")

    # Detect truncation explicitly
    finish_reason = getattr(response.choices[0], "finish_reason", None)
    if finish_reason == "length":
        raise RuntimeError(
            f"LLM output was truncated (finish_reason='length'). "
            f"This indicates the model hit the output token limit. "
            f"Raw response saved to: {raw_debug_path_txt}"
        )

    # Parse JSON normally
    try:
        data = json.loads(content)
        if not (isinstance(data, dict) and isinstance(data.get("chunks"), list)):
            raise ValueError("Top-level JSON must be {'chunks': [...]}.")

        chunks = data["chunks"]
        # Reconstruct chunk_text + counts locally
        chunks = reconstruct_chunk_text(letter_text, chunks)
        return chunks

    except Exception:
        # Optional: keep your salvage fallback, but now it should almost never happen.
        salvaged = extract_chunk_objects_from_response(content)
        if not salvaged:
            raise RuntimeError(
                f"LLM returned invalid or unusable JSON, and no chunk objects could be salvaged. "
                f"See raw response in: {raw_debug_path_txt}"
            )

        salvaged = reconstruct_chunk_text(letter_text, salvaged)
        print(f"[WARN] Whole-response JSON parse failed. Salvaged {len(salvaged)} chunk objects from {raw_debug_path_txt}")
        return salvaged

def postprocess_chunks(
    chunks: List[Dict[str, Any]],
    letter_text: str,
    year: int,
    source_file: str,
) -> List[Dict[str, Any]]:
    """
    Post-processing + guaranteed local repair step:

    - Ensure `year` and `source_file` are set.
    - If `chunk_text` is missing/empty but `start_char`/`end_char` exist and are valid,
      reconstruct `chunk_text` by slicing `letter_text[start_char:end_char]`.
    - If `position_in_letter` is missing or out of range, recompute as
      `start_char / len(letter_text)` (clamped to [0, 1]).
    - Always recompute `word_count` and `char_count` from the (repaired) chunk_text.

    Note: We do NOT override LLM-generated `prev_context` / `next_context`.
    """
    n = len(letter_text)

    for idx, c in enumerate(chunks):
        c.setdefault("year", year)
        c.setdefault("source_file", source_file)

        # -------- Guaranteed local repair step --------
        text = c.get("chunk_text")
        if not isinstance(text, str) or not text.strip():
            s = c.get("start_char")
            e = c.get("end_char")
            if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= n:
                text = letter_text[s:e]
                c["chunk_text"] = text
            else:
                # Keep as empty string if we cannot repair
                c["chunk_text"] = ""
                text = ""

        # Fix/normalize position_in_letter (must be 0.0-1.0)
        pos = c.get("position_in_letter")
        if not isinstance(pos, (int, float)) or pos < 0.0 or pos > 1.0:
            s = c.get("start_char")
            if isinstance(s, int) and n > 0:
                pos = s / n
                # clamp
                if pos < 0.0:
                    pos = 0.0
                elif pos > 1.0:
                    pos = 1.0
                c["position_in_letter"] = float(pos)

        # Recompute counts (LLM may be slightly off or chunk_text was repaired)
        c["word_count"] = len(text.split()) if text else 0
        c["char_count"] = len(text) if text else 0

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

    # 2) Light post-processing: fix counts, enforce year/source_file
    chunks = postprocess_chunks(
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
