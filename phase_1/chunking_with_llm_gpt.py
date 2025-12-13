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
OUT_DIR = PROJECT_ROOT / "data" / "chunks_llm_gpt"

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

    system_content = (
        "You are an expert corpus architect for Berkshire Hathaway "
        "shareholder letters.\n"
        "Your job is to chunk a single letter into semantically meaningful "
        "chunks and produce rich metadata.\n"
        "Follow EXACTLY the rules, definitions, and metadata schema in the "
        "chunking strategy document below.\n"
        "You must:\n"
        "- Respect section and subsection boundaries\n"
        "- Respect the target chunk sizes\n"
        "- Distinguish chunk types (philosophy, business_analysis, etc.)\n"
        "- Populate ALL fields in the 'Complete Metadata Structure' including "
        "chunk_text, word_count, char_count, and contextual summaries.\n\n"
        "Return ONLY valid JSON."
    )

    strategy_message = (
        "Here is the complete chunking strategy specification you MUST follow:\n\n"
        f"{chunking_spec}"
    )

    user_content = f"""
Chunk the Berkshire Hathaway shareholder letter for year {year}.

Source file name: {source_file}

Use the following requirements:

- Apply all rules from the chunking strategy document you received.
- Identify sections, subsections, and the different chunk types.
- Generate chunks of ~150–300 words, except where the strategy specifies
  larger/smaller sizes (e.g., front performance tables).
- Ensure each chunk is a complete thought and respects narrative flow.
- Fill in all required metadata fields for each chunk:
  - identifiers (chunk_id, year, source_file)
  - structural context (section_type, section_title, subsection, parent_section)
  - position metadata (position_in_letter, position_in_section,
    total_chunks_in_section, paragraph indices)
  - content metadata (chunk_text, word_count, char_count, chunk_type, etc.)
  - flags (contains_numbers, contains_principle, etc.)
  - contextual summaries (contextual_summary, prev_context, next_context)
  - extracted entities (topics, companies_mentioned, people_mentioned, metrics)
  - retrieval metadata (retrieval_priority, abstraction_level, time_sensitivity)
  - quality metrics, if specified.

IMPORTANT:
- Every chunk object MUST include `chunk_text` with the exact text for that chunk.
- Use `year = {year}` and `source_file = "{source_file}"` in every chunk.
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

    # 2) Light post-processing: fix counts, enforce year/source_file
    chunks = postprocess_chunks(chunks_raw, year=year, source_file=source_file_name)

    # 3) Write JSONL
    out_path = write_chunks_jsonl(chunks, year)
    print(f"[OK] Wrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main(sys.argv)
