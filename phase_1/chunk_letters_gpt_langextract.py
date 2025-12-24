#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunk Buffett shareholder letters using LangExtract as the base chunker, then
layer deterministic semantic sections + local flags + optional LLM enrichment,
and finally emit one valid JSON file per year.

Inputs (relative):
  data/text_extracted_letters/{YEAR}_cleaned.txt

Outputs (relative):
  data/chunks_llm_gpt/langextract/{YEAR}.json

Debug artifacts per year:
  data/chunks_llm_gpt/langextract/debug/{YEAR}_base_chunks.jsonl
  data/chunks_llm_gpt/langextract/debug/{YEAR}_sections.json
  data/chunks_llm_gpt/langextract/debug/{YEAR}_pre_llm.jsonl
  data/chunks_llm_gpt/langextract/debug/{YEAR}_llm_raw.jsonl (if enabled)

Usage examples:
  python chunk_letters_langextract.py --year 1977
  python chunk_letters_langextract.py --all
  python chunk_letters_langextract.py --input_dir data/text_extracted_letters --output_dir data/chunks_llm_gpt/langextract

LLM enrichment:
  - Optional. Requires OPENAI_API_KEY.
  - Uses OpenAI Responses API via the `openai` package if installed, else falls back to `requests`.
  - If no key is provided, semantic/enrichment fields are filled with safe defaults (null/[]/false).

Design goals:
  - Deterministic chunk boundaries from LangExtract (char offsets).
  - Deterministic sectioning based on robust heading heuristics.
  - Deterministic local flags.
  - LLM enrichment is a separate, retryable layer (does not affect chunk boundaries).

"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Minimal dependency shims
# ----------------------------
# LangExtract depends on `more_itertools` for `batched`. Some environments may not have it.
# We provide a tiny shim so the chunker works without extra installs.
try:
    import more_itertools  # type: ignore
except Exception:  # pragma: no cover
    import types
    def _batched(iterable, n):
        if n <= 0:
            raise ValueError("n must be > 0")
        batch = []
        for x in iterable:
            batch.append(x)
            if len(batch) == n:
                yield tuple(batch)
                batch = []
        if batch:
            yield tuple(batch)
    more_itertools = types.ModuleType("more_itertools")  # type: ignore
    more_itertools.batched = _batched  # type: ignore
    sys.modules["more_itertools"] = more_itertools  # type: ignore


# Shim for absl.logging used by langextract
try:
    from absl import logging as absl_logging  # type: ignore
except Exception:  # pragma: no cover
    import logging as _py_logging
    import types as _types
    absl = _types.ModuleType("absl")
    absl_logging = _types.ModuleType("absl.logging")
    for name in ["debug", "info", "warning", "error", "fatal", "exception"]:
        setattr(absl_logging, name, getattr(_py_logging, name))
    absl_logging.warn = _py_logging.warning  # type: ignore
    absl.logging = absl_logging  # type: ignore
    sys.modules["absl"] = absl
    sys.modules["absl.logging"] = absl_logging


# ----------------------------
# LangExtract bootstrap
# ----------------------------
def _ensure_langextract_on_path(repo_zip: Path) -> None:
    """
    Adds the extracted langextract repo to sys.path.
    We assume the user provided langextract-main.zip (downloaded from GitHub).
    """
    if not repo_zip.exists():
        return

    extract_dir = repo_zip.parent / "_vendor_langextract"
    pkg_dir = extract_dir / "langextract-main"
    if not pkg_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(repo_zip, "r") as zf:
            zf.extractall(extract_dir)

    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))


# ----------------------------
# Utilities
# ----------------------------
HEADER_LINE_RE = re.compile(r"^[A-Z][A-Za-z0-9&'’\-/(),. ]{0,60}$")
SALUTATION_RE = re.compile(r"^(To the (Shareholders|Stockholders) of Berkshire Hathaway.*:)\s*$", re.MULTILINE)

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _word_count(text: str) -> int:
    # Robust, simple: count word-like tokens (letters/digits with internal apostrophes/hyphens)
    return len(re.findall(r"[A-Za-z0-9]+(?:[’'\-][A-Za-z0-9]+)?", text))

def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "section"

def _sha1(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def _iter_lines_with_offsets(text: str) -> List[Tuple[int, str]]:
    """
    Returns a list of (line_start_char_offset, line_text_without_trailing_newline).
    Preserves exact char offsets by accumulating lengths.
    """
    out: List[Tuple[int, str]] = []
    pos = 0
    for raw in text.splitlines(True):  # keepends
        line = raw.rstrip("\r\n")
        out.append((pos, line))
        pos += len(raw)
    return out


def _is_blank_line_around(lines_with_offsets: List[Tuple[int, str]], idx: int) -> bool:
    """True if line idx is surrounded by blank lines (or edges)."""
    prev_blank = True
    next_blank = True
    if idx - 1 >= 0:
        prev_blank = (lines_with_offsets[idx-1][1].strip() == "")
    if idx + 1 < len(lines_with_offsets):
        next_blank = (lines_with_offsets[idx+1][1].strip() == "")
    return prev_blank and next_blank


def _looks_like_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 60:
        return False

    # Salutations handled separately
    if s.endswith(":"):
        return False

    # Hard rejects for prose fragments
    if "," in s or s.endswith(","):
        return False
    if s.startswith(("*", "•", "-", "(", "[", "Figure", "Table")):
        return False
    if any(x in s for x in [";"]):
        return False

    # Avoid normal sentences
    if "." in s:
        return False

    # Too many words -> probably prose
    words = s.split()
    if len(words) >= 7:
        return False

    # Common non-headers
    low = s.lower()
    if low.startswith(("dear ", "sincerely", "yours")):
        return False

    # Accept ALL CAPS short
    if s.isupper() and 2 <= len(s) <= 40:
        return True

    # Accept Title-Case-ish (few words, letters/digits/& etc)
    if not HEADER_LINE_RE.match(s):
        return False

    # Heuristic: most words start with capital letter
    cap_words = sum(1 for w in words if w and w[0].isupper())
    if cap_words < max(1, len(words) - 1):
        return False

    return True


def _detect_sections(text: str) -> List[Dict[str, Any]]:
    """
    Deterministic section detection using char offsets.
    Produces a list of sections with:
      - section_index
      - section_type (coarse)
      - section_title (exact header text or descriptive)
      - start_char, end_char
      - parent_section/subsection (null here; you can extend)
    """
    lines = _iter_lines_with_offsets(text)

    # 1) Find salutation ("To the Shareholders...")
    sal_match = SALUTATION_RE.search(text)
    sal_start = sal_match.start(1) if sal_match else None
    sal_line = sal_match.group(1).strip() if sal_match else None

    # Candidate headers: standalone lines that look like headers
    header_candidates: List[Tuple[int, str]] = []
    for off, line in lines:
        if line.strip() == "[[PAGE_BREAK]]":
            continue
        if _looks_like_header(line):
            header_candidates.append((off, line.strip()))

    # Filter: keep headers that appear after the salutation (or all if no salutation)
    if sal_start is not None:
        header_candidates = [(o, t) for (o, t) in header_candidates if o > sal_start]

    # De-duplicate adjacent duplicates
    dedup: List[Tuple[int, str]] = []
    prev_t = None
    for o, t in sorted(header_candidates, key=lambda x: x[0]):
        if prev_t == t:
            continue
        dedup.append((o, t))
        prev_t = t
    header_candidates = dedup

    sections: List[Dict[str, Any]] = []

    # Front matter: from start to salutation (or to first PAGE_BREAK if present)
    if sal_start is not None and sal_start > 0:
        sections.append({
            "section_index": 0,
            "section_type": "front_matter",
            "section_title": "Front matter / tables",
            "subsection": None,
            "parent_section": None,
            "start_char": 0,
            "end_char": sal_start,
        })

    # Main letter intro: from salutation to first header candidate (or end)
    base_idx = len(sections)
    if sal_start is not None:
        start = sal_start
        first_hdr_off = header_candidates[0][0] if header_candidates else len(text)
        sections.append({
            "section_index": base_idx,
            "section_type": "shareholder_letter",
            "section_title": sal_line or "To the Shareholders/Stockholders",
            "subsection": None,
            "parent_section": None,
            "start_char": start,
            "end_char": first_hdr_off,
        })
    else:
        # No salutation; treat beginning as letter
        first_hdr_off = header_candidates[0][0] if header_candidates else len(text)
        sections.append({
            "section_index": base_idx,
            "section_type": "shareholder_letter",
            "section_title": "Shareholder letter",
            "subsection": None,
            "parent_section": None,
            "start_char": 0,
            "end_char": first_hdr_off,
        })

    # Subsequent headers become sections; each runs until next header or end
    idx0 = len(sections)
    for i, (hdr_off, hdr_text) in enumerate(header_candidates):
        start = hdr_off
        end = header_candidates[i + 1][0] if i + 1 < len(header_candidates) else len(text)

        # Some letters have headers that are actually table captions; keep coarse type
        # Make section_type unique and stable to avoid chunk_id collisions.
        # Required chunk_id format is {year}_{section_type}_{sequence}, so section_type must not repeat.
        section_type = _slug(hdr_text)

        sections.append({
            "section_index": idx0 + i,
            "section_type": section_type,
            "section_title": hdr_text,
            "subsection": None,
            "parent_section": None,
            "start_char": start,
            "end_char": end,
        })

    # Ensure monotonic + clamp
    for s in sections:
        s["start_char"] = max(0, min(len(text), int(s["start_char"])))
        s["end_char"] = max(0, min(len(text), int(s["end_char"])))
        if s["end_char"] < s["start_char"]:
            s["end_char"] = s["start_char"]

    # Merge tiny/empty sections (common with false-positive headers)
    merged: List[Dict[str, Any]] = []
    for s in sections:
        if not merged:
            merged.append(s)
            continue
        if (s["end_char"] - s["start_char"]) < 250:  # too small to be meaningful
            # merge into previous
            merged[-1]["end_char"] = max(merged[-1]["end_char"], s["end_char"])
            continue
        merged.append(s)

    # Re-index after merges
    for i, s in enumerate(merged):
        s["section_index"] = i

    return merged


def _map_chunk_to_section(sections: List[Dict[str, Any]], chunk_start: int) -> Dict[str, Any]:
    # pick the section where chunk_start is within [start,end)
    for s in sections:
        if s["start_char"] <= chunk_start < s["end_char"]:
            return s
    # fallback: closest section by start_char
    return min(sections, key=lambda s: abs(chunk_start - s["start_char"])) if sections else {
        "section_index": 0,
        "section_type": "unknown",
        "section_title": "Unknown",
        "subsection": None,
        "parent_section": None,
        "start_char": 0,
        "end_char": chunk_start,
    }


def _compute_local_flags(chunk_text: str) -> Dict[str, bool]:
    has_financials = bool(re.search(r"[$%]", chunk_text)) or bool(re.search(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", chunk_text))
    # Table heuristic: many short lines + high digit density OR dot-leader columns
    lines = [ln for ln in chunk_text.splitlines() if ln.strip()]
    digit_chars = sum(ch.isdigit() for ch in chunk_text)
    tableish = False
    if len(lines) >= 6:
        avg_len = sum(len(ln) for ln in lines) / max(1, len(lines))
        digit_density = digit_chars / max(1, len(chunk_text))
        dot_leader = bool(re.search(r"\.\s*\.\s*\.\s*\.", chunk_text))
        aligned_nums = sum(bool(re.search(r"\s{2,}\d", ln)) for ln in lines) >= max(3, len(lines)//3)
        tableish = dot_leader or (digit_density > 0.08 and avg_len < 120) or aligned_nums
    has_table = tableish
    has_quote = ('"' in chunk_text) or ('“' in chunk_text) or ('”' in chunk_text)
    return {
        "has_financials": has_financials,
        "has_table": has_table,
        "has_quote": has_quote,
    }


def _initial_chunk_type(flags: Dict[str, bool], section_title: str) -> str:
    if flags["has_table"]:
        return "table_financials" if flags["has_financials"] else "table"
    # simple rule-based types
    if section_title.lower() in {"insurance", "investments", "banking", "textile operations"}:
        return "business_analysis"
    return "narrative"



def _word_target_langextract_chunks(
    text: str,
    doc: Any,
    tokenizer_impl: Any,
    min_words: int = 300,
    max_words: int = 400,
    hard_max_char: int = 3500,
) -> List[Dict[str, Any]]:
    """
    Produce chunks using LangExtract's tokenization + sentence boundary detection,
    but target a WORD range instead of a char buffer.

    - Never break mid-sentence (sentence-aware).
    - Prefer breaking on paragraph boundaries when within [min_words, max_words].
    - Enforces a hard_max_char fallback to avoid pathological chunks (e.g., long tables).
    """
    from langextract.chunking import SentenceIterator, get_char_interval  # type: ignore

    total_chars = max(1, len(text))

    def is_para_break(a_end: int, b_start: int) -> bool:
        # paragraph break if there is a blank line in the original text between spans
        gap = text[a_end:b_start]
        return "\n\n" in gap or "\r\n\r\n" in gap

    # Build sentence iterator over tokenized text (langextract expects TokenizedText)
    tokenized_text = tokenizer_impl.tokenize(text)
    sent_iter = SentenceIterator(tokenized_text)

    chunks: List[Dict[str, Any]] = []
    cur_start_char: Optional[int] = None
    cur_end_char: Optional[int] = None
    cur_word_count = 0
    last_para_break_end: Optional[int] = None  # best break candidate inside current chunk
    last_para_break_words: Optional[int] = None

    # helper to flush chunk
    def flush():
        nonlocal cur_start_char, cur_end_char, cur_word_count, last_para_break_end, last_para_break_words
        if cur_start_char is None or cur_end_char is None or cur_end_char <= cur_start_char:
            # reset
            cur_start_char = None
            cur_end_char = None
            cur_word_count = 0
            last_para_break_end = None
            last_para_break_words = None
            return
        chunk_text = text[cur_start_char:cur_end_char]
        chunks.append({
            "start_char": cur_start_char,
            "end_char": cur_end_char,
            "chunk_text": chunk_text,
            "char_count": len(chunk_text),
            "word_count": _word_count(chunk_text),
            "position_in_letter": round(cur_start_char / total_chars, 6),
        })
        cur_start_char = None
        cur_end_char = None
        cur_word_count = 0
        last_para_break_end = None
        last_para_break_words = None

    prev_sent_end: Optional[int] = None

    for sent in sent_iter:
        # SentenceIterator yields TokenInterval; derive char span from the tokenized text.
        ci = get_char_interval(tokenized_text, sent)
        s_start = int(ci.start_pos)
        s_end = int(ci.end_pos)
        if s_end <= s_start:
            continue
        s_text = text[s_start:s_end]
        s_words = _word_count(s_text)

        if cur_start_char is None:
            cur_start_char = s_start
            cur_end_char = s_end
            cur_word_count = s_words
            prev_sent_end = s_end
            continue

        # Track paragraph break opportunity between previous sentence and this sentence
        if prev_sent_end is not None and is_para_break(prev_sent_end, s_start):
            last_para_break_end = prev_sent_end
            last_para_break_words = cur_word_count

        # If adding this sentence stays within max_words and hard char, just add it
        projected_end = s_end
        projected_words = cur_word_count + s_words
        projected_chars = projected_end - cur_start_char

        if projected_words <= max_words and projected_chars <= hard_max_char:
            cur_end_char = projected_end
            cur_word_count = projected_words
            prev_sent_end = s_end
            continue

        # If we are already at/above min_words, try to cut at best paragraph break within range
        if cur_word_count >= min_words and last_para_break_end is not None:
            # If para-break word count is reasonable, cut there
            if last_para_break_words is not None and min_words <= last_para_break_words <= max_words:
                # Cut at paragraph boundary
                cur_end_char = last_para_break_end
                flush()
                # Start new chunk at current sentence
                cur_start_char = s_start
                cur_end_char = s_end
                cur_word_count = s_words
                prev_sent_end = s_end
                continue

        # Otherwise, cut at current end (before adding this sentence) if it's not tiny,
        # else force include the sentence (to avoid micro-chunks) unless it violates hard_max_char badly.
        if cur_word_count >= max(80, min_words // 2):
            flush()
            cur_start_char = s_start
            cur_end_char = s_end
            cur_word_count = s_words
            prev_sent_end = s_end
            continue

        # Tiny chunk case: include sentence even if it exceeds max_words, but respect hard_max_char by flushing first if needed
        if (s_end - cur_start_char) > hard_max_char and cur_word_count > 0:
            flush()
            cur_start_char = s_start
            cur_end_char = s_end
            cur_word_count = s_words
            prev_sent_end = s_end
            continue

        # include
        cur_end_char = s_end
        cur_word_count = projected_words
        prev_sent_end = s_end

    flush()
    return chunks


# ----------------------------
# LLM enrichment (optional)
# ----------------------------
ENRICH_FIELDS = [
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

def _default_enrichment() -> Dict[str, Any]:
    return {
        "contains_principle": False,
        "contains_example": False,
        "contains_comparison": False,
        "contextual_summary": None,
        "prev_context": None,
        "next_context": None,
        "topics": [],
        "companies_mentioned": [],
        "people_mentioned": [],
        "metrics_discussed": [],
        "industries": [],
        "principle_category": None,
        "principle_statement": None,
        "retrieval_priority": "medium",
        "abstraction_level": "medium",
        "time_sensitivity": "low",
        "is_complete_thought": True,
        "needs_context": False,
    }

def _build_enrich_prompt(chunk_text: str, prev_tail: str, next_head: str) -> str:
    # Keep prompt short + structured
    return f"""
You are enriching a chunk from Warren Buffett shareholder letters for retrieval and study.

Return STRICT JSON with exactly these keys:
{json.dumps(ENRICH_FIELDS, indent=2)}

Rules:
- contextual_summary: 1-2 sentences.
- prev_context / next_context: 1-2 sentences based on provided adjacent context only.
- topics: 3-6 concise tags.
- companies_mentioned/people_mentioned: extract explicit proper names only.
- metrics_discussed: named metrics (e.g., ROE, float, intrinsic value, book value, combined ratio).
- industries: sectors mentioned (insurance, railroads, utilities, manufacturing, etc.).
- retrieval_priority: high if contains a timeless principle OR key Berkshire operating economics; low for pure boilerplate.
- abstraction_level: high for principles/philosophy; low for numeric tables; medium otherwise.
- time_sensitivity: high only if references a specific deal/event whose relevance is time-bound; else low.
- is_complete_thought: true if chunk stands alone; needs_context true if it clearly relies on adjacent text.

Adjacent context (do not quote verbatim; just use for understanding):
PREV_TAIL: {prev_tail}
NEXT_HEAD: {next_head}

CHUNK:
{chunk_text}
""".strip()


def _call_openai_json(prompt: str, model: str, api_key: str, timeout_s: int = 60) -> Dict[str, Any]:
    """
    Call OpenAI and return a parsed JSON object.

    Compatibility strategy (because different OpenAI SDK/API versions differ):
      1) Try Responses API via official SDK (if installed) using text.format=json_object.
      2) Try Responses API via requests with text.format=json_object.
      3) Fallback to Chat Completions API with response_format=json_object.
    """
    # 1) Official SDK: Responses API
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=model,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )
        # Newer SDKs expose output_text
        out_text = getattr(resp, "output_text", None) or ""
        if not out_text:
            # best-effort traverse
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        out_text += getattr(c, "text", "") or ""
        return json.loads(out_text)
    except Exception:
        pass

    # 2) Requests: Responses API
    try:
        import requests
        url = "https://api.openai.com/v1/responses"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "input": prompt,
            "text": {"format": {"type": "json_object"}},
        }
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # If the error complains about unsupported params, we will fallback to chat.completions below.
            err_text = r.text[:4000]
            raise RuntimeError(f"OpenAI Responses HTTP {r.status_code}: {err_text}") from e
        data = r.json()
        out_text = data.get("output_text", "") or ""
        if not out_text:
            for item in data.get("output", []):
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        out_text += c.get("text", "")
        return json.loads(out_text)
    except Exception as e_responses:
        # 3) Fallback: Chat Completions API (widely supported)
        try:
            import requests
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You output ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            try:
                r.raise_for_status()
            except requests.HTTPError as e:
                raise RuntimeError(f"OpenAI ChatCompletions HTTP {r.status_code}: {r.text[:4000]}") from e
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e_chat:
            # Preserve both errors
            raise RuntimeError(f"LLM call failed. Responses error: {e_responses}. ChatCompletions error: {e_chat}") from e_chat



def _enrich_chunks_with_llm(chunks: List[Dict[str, Any]], model: str, api_key: str, debug_llm_path: Path) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    debug_llm_path.parent.mkdir(parents=True, exist_ok=True)

    for i, ch in enumerate(chunks):
        prev_tail = ""
        next_head = ""
        if i > 0:
            prev_text = chunks[i - 1]["chunk_text"]
            prev_tail = " ".join(prev_text.split()[-60:])[:400]
        if i + 1 < len(chunks):
            next_text = chunks[i + 1]["chunk_text"]
            next_head = " ".join(next_text.split()[:60])[:400]

        prompt = _build_enrich_prompt(ch["chunk_text"], prev_tail, next_head)

        for attempt in range(4):
            try:
                result = _call_openai_json(prompt, model=model, api_key=api_key)
                # sanitize keys
                cleaned = _default_enrichment()
                for k in ENRICH_FIELDS:
                    if k in result:
                        cleaned[k] = result[k]
                ch2 = dict(ch)
                ch2.update(cleaned)
                enriched.append(ch2)

                with debug_llm_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"chunk_id": ch["chunk_id"], "llm": cleaned}, ensure_ascii=False) + "\n")
                break
            except Exception as e:
                if attempt == 3:
                    ch2 = dict(ch)
                    ch2.update(_default_enrichment())
                    enriched.append(ch2)
                    with debug_llm_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"chunk_id": ch["chunk_id"], "error": str(e)}, ensure_ascii=False) + "\n")
                time.sleep(1.2 * (attempt + 1))

    return enriched


# ----------------------------
# Core pipeline
# ----------------------------
def chunk_one_letter(
    year: int,
    input_path: Path,
    output_dir: Path,
    vendor_zip: Optional[Path] = None,
    max_char_buffer: int = 5000,
    min_words: int = 300,
    max_words: int = 400,
    hard_max_char: int = 3500,
    enable_llm: bool = False,
    llm_model: str = "gpt-4.1-mini",
) -> Path:
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    total_chars = max(1, len(text))

    # Bootstrap langextract
    if vendor_zip:
        _ensure_langextract_on_path(vendor_zip)

    try:
        from langextract.chunking import ChunkIterator  # type: ignore
        from langextract.core import tokenizer as tokenizer_lib  # type: ignore
        from langextract.core import data as data_lib  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import langextract from vendor zip. "
            "Make sure langextract-main.zip is available and extractable."
        ) from e

    doc = data_lib.Document(text=text, additional_context=None)
    # Set a stable document_id for reproducibility
    doc.document_id = f"letter_{year}"

    tokenizer_impl = tokenizer_lib.RegexTokenizer()

    # Step 1: detect semantic sections FIRST (so base chunks never cross section boundaries)
    sections = _detect_sections(text)
    sections_path = (output_dir / "debug") / f"{year}_sections.json"  # overwritten below after debug_dir exists

    # Step 1: base chunks (LangExtract) within each section slice (offset back to full-text coordinates)
    base_chunks: List[Dict[str, Any]] = []
    for sec in sections:
        sec_start = int(sec["start_char"])
        sec_end = int(sec["end_char"])
        if sec_end <= sec_start:
            continue
        slice_text = text[sec_start:sec_end]

        # Chunk the slice with word-targeted LangExtract sentence boundaries
        slice_chunks = _word_target_langextract_chunks(
            text=slice_text,
            doc=doc,
            tokenizer_impl=tokenizer_impl,
            min_words=min_words,
            max_words=max_words,
            hard_max_char=hard_max_char,
        )
        for sc in slice_chunks:
            start_char = sec_start + int(sc["start_char"])
            end_char = sec_start + int(sc["end_char"])
            ctext = sc["chunk_text"]
            base_chunks.append({
                "start_char": start_char,
                "end_char": end_char,
                "chunk_text": ctext,
                "char_count": len(ctext),
                "position_in_letter": round(start_char / total_chars, 6),
                "section_index": int(sec["section_index"]),
                "section_type": sec["section_type"],
                "section_title": sec["section_title"],
            })
    # Sort just in case
    base_chunks.sort(key=lambda x: x["start_char"])

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Write sections debug (now that debug_dir exists)
    sections_path = debug_dir / f"{year}_sections.json"
    sections_path.write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write base chunks debug
    base_path = debug_dir / f"{year}_base_chunks.jsonl"
    with base_path.open("w", encoding="utf-8") as f:
        for bc in base_chunks:
            f.write(json.dumps(bc, ensure_ascii=False) + "\n")

    # Map chunks to sections + compute deterministic metadata + local flags (Step 3)
    # First group chunks by section
    by_section: Dict[int, List[Dict[str, Any]]] = {}
    for bc in base_chunks:
        sec_id = int(bc.get("section_index", 0))
        by_section.setdefault(sec_id, []).append(bc)

    pre_llm_chunks: List[Dict[str, Any]] = []

    for sec_id in sorted(by_section.keys()):
        grouped = by_section[sec_id]
        grouped.sort(key=lambda x: x["start_char"])
        total_in_section = len(grouped)
        section_type = grouped[0].get("section_type", "unknown")
        section_title = grouped[0].get("section_title", "Unknown")

        for pos_in_section, bc in enumerate(grouped):
            flags = _compute_local_flags(bc["chunk_text"])
            chunk_type = _initial_chunk_type(flags, section_title)

            seq = pos_in_section + 1  # start at 1 for readability, formatted 003 etc
            chunk_id = f"{year}_{_slug(section_type)}_{seq:03d}"

            item = {
                # Required identity
                "chunk_id": chunk_id,
                "year": int(year),
                "source_file": f"letter_{year}.pdf",
                # Section info
                "section_type": section_type,
                "section_title": section_title,
                "subsection": None,
                "parent_section": None,
                # Positions
                "position_in_letter": bc["position_in_letter"],
                "position_in_section": int(pos_in_section),
                "total_chunks_in_section": int(total_in_section),
                # Text
                "chunk_text": bc["chunk_text"],
                "word_count": _word_count(bc["chunk_text"]),
                "char_count": int(bc["char_count"]),
                # Types + local flags
                "chunk_type": chunk_type,
                "has_financials": bool(flags["has_financials"]),
                "has_table": bool(flags["has_table"]),
                "has_quote": bool(flags["has_quote"]),
            }

            # Fill remaining required keys with defaults for now (LLM layer later)
            item.update(_default_enrichment())

            pre_llm_chunks.append(item)

    pre_llm_path = debug_dir / f"{year}_pre_llm.jsonl"
    with pre_llm_path.open("w", encoding="utf-8") as f:
        for ch in pre_llm_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # Step 4: LLM enrichment (optional)
    final_chunks = pre_llm_chunks
    if enable_llm:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("[WARN] enable_llm was set but OPENAI_API_KEY is missing; skipping enrichment.", file=sys.stderr)
        else:
            llm_debug = debug_dir / f"{year}_llm_raw.jsonl"
            final_chunks = _enrich_chunks_with_llm(pre_llm_chunks, model=llm_model, api_key=api_key, debug_llm_path=llm_debug)

    # Step 5: Mapping (already ensured all required fields exist)
    out_path = output_dir / f"{year}.json"
    out_path.write_text(json.dumps(final_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _discover_years(input_dir: Path) -> List[int]:
    years = []
    for p in input_dir.glob("*_cleaned.txt"):
        m = re.match(r"(\d{4})_cleaned\.txt$", p.name)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="../data/text_extracted_letters", help="Relative path to cleaned letter txt files.")
    ap.add_argument("--output_dir", default="../data/chunks_llm_gpt/langextract", help="Relative output directory.")
    ap.add_argument("--vendor_zip", default="langextract-main.zip", help="Path to langextract zip (relative or absolute).")
    ap.add_argument("--year", type=int, default=None, help="Process a single year (e.g., 1977).")
    ap.add_argument("--all", action="store_true", help="Process all *_cleaned.txt files in input_dir.")
    ap.add_argument("--max_char_buffer", type=int, default=5000, help="(Deprecated) LangExtract max_char_buffer (characters). Word-target chunker is used by default.")
    ap.add_argument("--min_words", type=int, default=300, help="Target minimum words per chunk (LangExtract sentence-aware).")
    ap.add_argument("--max_words", type=int, default=400, help="Target maximum words per chunk (LangExtract sentence-aware).")
    ap.add_argument("--hard_max_char", type=int, default=3500, help="Hard max characters per chunk as safety valve (tables).")
    ap.add_argument("--enable_llm", action="store_true", help="Enable OpenAI enrichment (requires OPENAI_API_KEY).")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name for enrichment.")
    args = ap.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # vendor zip resolution
    vz = Path(args.vendor_zip)
    if not vz.is_absolute():
        # prefer CWD, else script dir
        if (Path.cwd() / vz).exists():
            vz = Path.cwd() / vz
        else:
            vz = Path(__file__).resolve().parent / vz

    if args.year is None and not args.all:
        ap.error("Provide --year YEAR or --all")

    years = [args.year] if args.year is not None else _discover_years(input_dir)
    if not years:
        print(f"[ERROR] No *_cleaned.txt files found in {input_dir}", file=sys.stderr)
        return 2

    for y in years:
        inp = input_dir / f"{y}_cleaned.txt"
        if not inp.exists():
            print(f"[WARN] Missing input: {inp}", file=sys.stderr)
            continue
        print(f"[INFO] Processing year {y} -> {output_dir}")
        out = chunk_one_letter(
            year=y,
            input_path=inp,
            output_dir=output_dir,
            vendor_zip=vz if vz.exists() else None,
            max_char_buffer=args.max_char_buffer,
            min_words=args.min_words,
            max_words=args.max_words,
            hard_max_char=args.hard_max_char,
            enable_llm=args.enable_llm,
            llm_model=args.model,
        )
        print(f"[OK] Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
