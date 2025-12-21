#!/usr/bin/env python
"""
chunking_with_llm_gpt_localpos_v4_batch.py

Batch chunking runner for Berkshire Hathaway shareholder letters.

This is a batch-enabled evolution of `chunking_with_llm_gpt_localpos_v4.py`:
- Processes multiple years in one run (positional years, --years, or --all).
- Safe resume: skips years that already have an output file unless --force.
- Per-year isolation: writes per-year raw LLM response + diagnostics, and continues on errors.
- Adds retries + backoff for transient OpenAI failures.
- Optional --dry-run to validate file discovery and I/O WITHOUT calling the LLM.

Directory layout (relative to this file):

    phase_1/
        chunking_rule_claude.md
        chunking_with_llm_gpt_localpos_v4_batch.py
    data/
        text_extracted_letters/
            1977_cleaned.txt
            ...
        chunks_llm_gpt/
            localpos_v4/
                1977_chunks_llm.jsonl
                1977_offset_diagnostics.json
                1977_raw_llm_response.json
                batch_run_summary.json

Usage examples:
    # Chunk specific years
    python chunking_with_llm_gpt_localpos_v4_batch.py 2008 2009

    # Chunk a comma-separated list
    python chunking_with_llm_gpt_localpos_v4_batch.py --years 2008,2009,2010

    # Chunk everything that exists in data/text_extracted_letters
    python chunking_with_llm_gpt_localpos_v4_batch.py --all

    # Validate discovery and output paths, but don't call LLM
    python chunking_with_llm_gpt_localpos_v4_batch.py --all --dry-run

Env:
    OPENAI_MODEL (default: gpt-4.1-mini)
    OPENAI_API_KEY (required by OpenAI SDK)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ------------------------- Paths & Constants -------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

CHUNKING_STRATEGY_PATH = THIS_DIR / "chunking_rule_claude.md"

TEXT_DIR = PROJECT_ROOT / "data" / "text_extracted_letters"
OUT_DIR = PROJECT_ROOT / "data" / "chunks_llm_gpt" / "g"
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

# Windowed chunking defaults (prevents long-year truncation / broken JSON)
# If a letter exceeds this, we chunk in overlapping windows and then merge.
MAX_SINGLECALL_CHARS = int(os.getenv("CHUNK_MAX_SINGLECALL_CHARS", "26000"))
WINDOW_MAX_CHARS = int(os.getenv("CHUNK_WINDOW_MAX_CHARS", "18000"))
WINDOW_OVERLAP_CHARS = int(os.getenv("CHUNK_WINDOW_OVERLAP_CHARS", "800"))


# ------------------------- Generic helpers -------------------------

def _now_iso() -> str:
    # Keep it simple + deterministic-ish for logs.
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


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


def discover_years_from_text_dir() -> List[int]:
    """Discover years based on files like '2009_cleaned.txt'."""
    years: List[int] = []
    if not TEXT_DIR.exists():
        return years
    for p in TEXT_DIR.glob("*_cleaned.txt"):
        m = re.match(r"^(\d{4})_cleaned\.txt$", p.name)
        if m:
            try:
                years.append(int(m.group(1)))
            except ValueError:
                pass
    years.sort()
    return years

# ------------------------- Window splitting (local, deterministic) -------------------------

def _find_paragraph_break_before(letter_text: str, pos: int, window: int = 4000) -> int | None:
    """Return a position <= pos that ends right after a paragraph break (\n\n) if found."""
    n = len(letter_text)
    pos = _clamp_int(pos, 0, n)
    left = max(0, pos - window)
    snippet = letter_text[left:pos]
    k = snippet.rfind("\n\n")
    if k == -1:
        return None
    return left + k + 2




def split_letter_into_windows(
    letter_text: str,
    *,
    max_chars: int = WINDOW_MAX_CHARS,
    overlap_chars: int = WINDOW_OVERLAP_CHARS,
) -> List[Tuple[int, int, str]]:
    """Split letter_text into overlapping windows. Returns list of (start, end, window_text)."""
    n = len(letter_text)
    if n <= max_chars:
        return [(0, n, letter_text)]

    windows: List[Tuple[int, int, str]] = []
    start = 0
    guard = 0

    while start < n:
        guard += 1
        if guard > 50_000:
            # Safety guard against infinite loops on pathological input
            break

        target_end = min(n, start + max_chars)

        if target_end >= n:
            end = n
        else:
            # Try strongest boundaries first
            end = _find_paragraph_break_before(letter_text, target_end)
            if end is None:
                sb = _find_sentence_break_before(letter_text, target_end, window=2400)
                end = sb if sb is not None else _snap_to_whitespace_before(letter_text, target_end, window=1200)

            end = _clamp_int(end, start + 2000, n)  # ensure progress + minimum window size

        win_text = letter_text[start:end]
        windows.append((start, end, win_text))

        if end >= n:
            break

        # Overlap
        next_start = max(0, end - overlap_chars)
        # Snap overlap start to whitespace to reduce mid-word starts
        next_start = _snap_to_whitespace_before(letter_text, next_start, window=600)
        next_start = _clamp_int(next_start, start + 1, n)
        start = next_start

    return windows


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

    # Renumber chunk_id deterministically by global order (important for windowed chunking)
    chunks.sort(key=lambda x: (x.get("start_char", 10**18), x.get("end_char", 10**18)))
    for idx, c in enumerate(chunks):
        st = (c.get("section_type") or "other").strip() or "other"
        c["chunk_id"] = f"{year}_{st}_{idx:03d}"


    return chunks, diagnostics


# ------------------------- LLM call (with retries) -------------------------

@dataclass
class LLMRetryConfig:
    max_attempts: int = 5
    base_sleep_s: float = 1.5
    max_sleep_s: float = 20.0
    jitter_s: float = 0.8


def _is_retryable_openai_error(exc: Exception) -> bool:
    # Keep broad: networking, rate limits, 5xx. The SDK error classes may vary by version.
    msg = (str(exc) or "").lower()
    retry_markers = [
        "rate limit",
        "timeout",
        "timed out",
        "temporarily",
        "server error",
        "overloaded",
        "connection",
        "429",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(m in msg for m in retry_markers)


def _llm_call_chunks_single(
    *,
    text_for_llm: str,
    year: int,
    source_file: str,
    chunking_spec: str,
    model: str,
    raw_debug_path: Path,
    offsets_are_relative_to_window: bool,
    window_id: Optional[int] = None,
    retry: Optional[LLMRetryConfig] = None,
) -> List[Dict[str, Any]]:
    """Single LLM call that returns chunk dicts. Offsets can be relative to the provided text."""
    client = OpenAI()
    retry = retry or LLMRetryConfig()

    rel_note = (
        "Offsets are relative to WINDOW TEXT (0-indexed within WINDOW TEXT)."
        if offsets_are_relative_to_window
        else "Offsets are relative to LETTER TEXT (0-indexed within LETTER TEXT)."
    )

    system_content = f"""
You are an expert corpus architect for Berkshire Hathaway shareholder letters.
Your job is to chunk a single text into semantically meaningful chunks and produce rich metadata.
Follow EXACTLY the chunking rule, definitions, and metadata schema in the chunking strategy document.

Hard constraints:
- Return ONLY valid JSON (no prose).
- Top-level MUST be a JSON object: {{"chunks": [ ... ]}}.
- Include accurate start_char/end_char offsets. {rel_note}
- Also include start_anchor and end_anchor copied EXACTLY from the provided TEXT.
- DO generate prev_context and next_context summaries.
- DO NOT include chunk_text/word_count/char_count (they will be reconstructed locally).
- DO NOT include position_in_letter/position_in_section/total_chunks_in_section (they will be computed locally).
"""

    strategy_message = "Here is the complete chunking rule specification you MUST follow:\n\n" + chunking_spec

    window_tag = f"window {window_id}" if window_id is not None else "full letter"

    if offsets_are_relative_to_window:
        offset_instructions = """
IMPORTANT (OFFSETS):
- start_char/end_char MUST be offsets into WINDOW TEXT ONLY (not the full letter).
- 0-indexed, start inclusive, end exclusive.
- start_anchor and end_anchor MUST be copied EXACTLY from WINDOW TEXT.
"""
    else:
        offset_instructions = """
IMPORTANT (OFFSETS):
- start_char/end_char MUST be offsets into LETTER TEXT.
- 0-indexed, start inclusive, end exclusive.
- start_anchor and end_anchor MUST be copied EXACTLY from LETTER TEXT.
"""

    user_content = f"""
Chunk the Berkshire Hathaway shareholder letter for year {year} ({window_tag}).

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
    "start_char": "int (0-indexed, inclusive offset)",
    "end_char": "int (0-indexed, exclusive offset)",
    "start_anchor": "string (first ~60 characters of the chunk text, copied EXACTLY from provided TEXT)",
    "end_anchor": "string (last ~60 characters of the chunk text, copied EXACTLY from provided TEXT)",
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

{offset_instructions}

TEXT START
{text_for_llm}
TEXT END
"""

    last_exc: Optional[Exception] = None
    for attempt in range(1, retry.max_attempts + 1):
        try:
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

            print(f"[WARN] Whole-response JSON parse failed. Salvaged {len(salvaged)} chunk objects.")
            return salvaged

        except Exception as exc:
            last_exc = exc
            if attempt >= retry.max_attempts or not _is_retryable_openai_error(exc):
                break

            sleep_s = min(
                retry.max_sleep_s,
                retry.base_sleep_s * (2 ** (attempt - 1)) + random.random() * retry.jitter_s,
            )
            print(f"[WARN] LLM call failed for {year} ({window_tag}) (attempt {attempt}/{retry.max_attempts}): {exc}")
            print(f"[INFO] Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM call failed for {year} ({window_tag}) after {retry.max_attempts} attempts: {last_exc}")


def call_llm_for_chunks_windowed(
    *,
    letter_text: str,
    year: int,
    source_file: str,
    chunking_spec: str,
    model: str,
) -> List[Dict[str, Any]]:
    """Windowed chunking to avoid truncation/broken JSON on long letters."""
    windows = split_letter_into_windows(letter_text)
    if len(windows) == 1 and len(letter_text) <= MAX_SINGLECALL_CHARS:
        raw_path = OUT_DIR / f"{year}_raw_llm_response.json"
        return _llm_call_chunks_single(
            text_for_llm=letter_text,
            year=year,
            source_file=source_file,
            chunking_spec=chunking_spec,
            model=model,
            raw_debug_path=raw_path,
            offsets_are_relative_to_window=False,
            window_id=None,
        )

    all_chunks: List[Dict[str, Any]] = []

    for w_id, (w_start, w_end, w_text) in enumerate(windows):
        raw_path = OUT_DIR / f"{year}_raw_llm_response_window_{w_id:03d}.json"
        w_chunks = _llm_call_chunks_single(
            text_for_llm=w_text,
            year=year,
            source_file=source_file,
            chunking_spec=chunking_spec,
            model=model,
            raw_debug_path=raw_path,
            offsets_are_relative_to_window=True,
            window_id=w_id,
        )

        # Map local offsets -> global offsets
        for c in w_chunks:
            if not isinstance(c, dict):
                continue
            s = c.get("start_char")
            e = c.get("end_char")
            if isinstance(s, int):
                c["start_char"] = w_start + s
            if isinstance(e, int):
                c["end_char"] = w_start + e
            # keep year/source_file normalized later
            c["_window_id"] = w_id
            c["_window_start"] = w_start
            c["_window_end"] = w_end

        all_chunks.extend([c for c in w_chunks if isinstance(c, dict)])

    # Fast de-dup on identical spans (overlap windows may duplicate chunks)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for c in sorted(all_chunks, key=lambda x: (x.get('start_char', 10**18), x.get('end_char', 10**18))):
        s, e = c.get('start_char'), c.get('end_char')
        if isinstance(s, int) and isinstance(e, int):
            k = (s, e, _norm_ws(c.get('start_anchor') or ''), _norm_ws(c.get('end_anchor') or ''))
            if k in seen:
                continue
            seen.add(k)
        deduped.append(c)

    return deduped


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


def load_existing_chunks_count(year: int) -> Optional[int]:
    p = OUT_DIR / f"{year}_chunks_llm.jsonl"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


# ------------------------- Batch runner -------------------------

def _parse_years_csv(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise ValueError(f"Invalid year in --years: {part!r}")
    return out


def _unique_sorted_years(years: List[int]) -> List[int]:
    return sorted({int(y) for y in years})


def _validate_output_integrity(letter_text: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight, local-only validation.
    This is intentionally strict on boundaries and offsets (since you've seen mid-sentence starts).
    """
    n = len(letter_text)
    total = len(chunks)

    bad_offsets = 0
    empty_text = 0
    midword_starts = 0
    unsafe_starts = 0
    overlaps = 0
    gaps = 0

    # Ensure ordered
    ordered = sorted(chunks, key=lambda x: (x.get("start_char", 10**18), x.get("end_char", 10**18)))

    prev_end: Optional[int] = None
    for c in ordered:
        s, e = c.get("start_char"), c.get("end_char")
        txt = c.get("chunk_text") or ""

        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= n):
            bad_offsets += 1
        if not txt:
            empty_text += 1

        if isinstance(s, int) and 0 < s < n and letter_text[s - 1].isalnum() and letter_text[s].isalnum():
            midword_starts += 1

        if isinstance(s, int) and not _is_safe_boundary(letter_text, s):
            unsafe_starts += 1

        if prev_end is not None and isinstance(s, int):
            if s < prev_end:
                overlaps += 1
            elif s > prev_end:
                gaps += 1

        if isinstance(e, int):
            prev_end = e

    return {
        "total_chunks": total,
        "bad_offsets": bad_offsets,
        "empty_chunk_text": empty_text,
        "midword_starts": midword_starts,
        "unsafe_starts": unsafe_starts,
        "overlaps": overlaps,
        "gaps": gaps,
    }


def run_one_year(
    year: int,
    *,
    model: str,
    chunking_spec: str,
    force: bool,
    dry_run: bool,
    min_words_to_keep: int,
) -> Dict[str, Any]:
    source_file_name = f"{year}_cleaned.txt"
    out_jsonl = OUT_DIR / f"{year}_chunks_llm.jsonl"
    out_diag = OUT_DIR / f"{year}_offset_diagnostics.json"

    if out_jsonl.exists() and not force:
        return {
            "year": year,
            "status": "skipped",
            "reason": "output_exists",
            "output": str(out_jsonl),
            "existing_chunks": load_existing_chunks_count(year),
        }

    if dry_run:
        # Validate we can read input and write to output dir.
        letter_text = load_letter_text(year)
        return {
            "year": year,
            "status": "dry_run_ok",
            "input_chars": len(letter_text),
            "would_write": [str(out_jsonl), str(out_diag)],
        }

    letter_text = load_letter_text(year)

    chunks_raw = call_llm_for_chunks_windowed(
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
        min_words_to_keep=min_words_to_keep,
    )

    # Final rebuild (ensures chunk_text present even if LLM "helpfully" included it)
    chunks = reconstruct_chunk_text(letter_text, chunks)

    integrity = _validate_output_integrity(letter_text, chunks)
    diagnostics["integrity"] = integrity

    out_path = write_chunks_jsonl(chunks, year)
    diag_path = write_diagnostics_json(diagnostics, year)

    return {
        "year": year,
        "status": "ok",
        "chunks": len(chunks),
        "output": str(out_path),
        "diagnostics": str(diag_path),
        "integrity": integrity,
    }


def write_batch_summary(summary: Dict[str, Any]) -> Path:
    out_path = OUT_DIR / "batch_run_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ------------------------- CLI -------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch chunk Berkshire letters with OpenAI + local offset repair.")
    p.add_argument("years", nargs="*", help="Years to process (e.g., 2008 2009).")
    p.add_argument("--years", dest="years_csv", default="", help="Comma-separated years (e.g., 2008,2009).")
    p.add_argument("--all", action="store_true", help="Process all *_cleaned.txt found in data/text_extracted_letters.")
    p.add_argument("--force", action="store_true", help="Re-run even if output exists.")
    p.add_argument("--dry-run", action="store_true", help="Do not call LLM; only validate discovery + I/O.")
    p.add_argument("--min-words-to-keep", type=int, default=60, help="Merge non-table chunks smaller than this into next chunk within same section.")
    return p


def main(argv: List[str]) -> int:
    args = build_arg_parser().parse_args(argv[1:])

    # Resolve years
    years: List[int] = []
    if args.all:
        years.extend(discover_years_from_text_dir())

    if args.years_csv:
        years.extend(_parse_years_csv(args.years_csv))

    if args.years:
        for y in args.years:
            try:
                years.append(int(y))
            except ValueError:
                print(f"[ERROR] Invalid year argument: {y!r}")
                return 2

    years = _unique_sorted_years(years)
    if not years:
        print("[ERROR] No years selected. Provide years, --years, or --all.")
        return 2

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    print(f"[INFO] Model={model}")
    print(f"[INFO] Years={years}")
    print(f"[INFO] OUT_DIR={OUT_DIR}")

    # Load spec once
    chunking_spec = load_chunking_strategy()

    results: List[Dict[str, Any]] = []
    started = _now_iso()
    ok = 0
    skipped = 0
    failed = 0

    for year in years:
        print(f"\n[INFO] === Year {year} ===")
        try:
            r = run_one_year(
                year,
                model=model,
                chunking_spec=chunking_spec,
                force=bool(args.force),
                dry_run=bool(args.dry_run),
                min_words_to_keep=int(args.min_words_to_keep),
            )
            results.append(r)
            status = r.get("status")
            if status in ("ok", "dry_run_ok"):
                ok += 1
                if status == "ok":
                    integ = (r.get("integrity") or {})
                    print(f"[OK] {year}: chunks={r.get('chunks')} integrity={integ}")
                else:
                    print(f"[OK] {year}: dry-run input_chars={r.get('input_chars')}")
            elif status == "skipped":
                skipped += 1
                print(f"[SKIP] {year}: {r.get('reason')} existing_chunks={r.get('existing_chunks')}")
            else:
                print(f"[WARN] {year}: status={status}")
        except Exception as exc:
            failed += 1
            err = {"year": year, "status": "failed", "error": str(exc)}
            results.append(err)
            print(f"[ERROR] {year}: {exc}")

    finished = _now_iso()

    summary = {
        "started_at": started,
        "finished_at": finished,
        "model": model,
        "out_dir": str(OUT_DIR),
        "years_requested": years,
        "counts": {"ok_or_dry_run_ok": ok, "skipped": skipped, "failed": failed, "total": len(years)},
        "results": results,
        "notes": {
            "integrity_fields": [
                "bad_offsets",
                "empty_chunk_text",
                "midword_starts",
                "unsafe_starts",
                "overlaps",
                "gaps",
            ]
        },
    }

    summary_path = write_batch_summary(summary)
    print(f"\n[INFO] Batch summary written: {summary_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
