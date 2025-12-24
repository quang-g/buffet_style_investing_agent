#!/usr/bin/env python
"""
chunking_with_llm_gpt_localpos_v6_hybrid_batch.py

Hybrid (best-of "c" + "g") batch chunker for Berkshire Hathaway shareholder letters.

Design goals (based on observed failures in sample outputs):
- Avoid truncated/broken JSON on long letters by windowing calls (g).
- Improve semantic coherence of window boundaries by *preferring* section/header breaks (c),
  but NEVER letting header detection cause dropped coverage.
- Guarantee positional integrity with strict local repair + integrity gating:
    * rebuild chunk_text from offsets
    * fix mid-word / unsafe boundaries
    * merge + de-duplicate overlap artifacts
    * validate coverage/gaps/overlaps; if gaps exceed thresholds, auto-fill missing spans
      with targeted supplemental window calls.

Outputs:
- data/chunks_llm_gpt/localpos_v6/{year}_chunks_llm.jsonl
- data/chunks_llm_gpt/localpos_v6/{year}_raw_llm_responses/  (per-window raw responses)
- data/chunks_llm_gpt/localpos_v6/{year}_offset_diagnostics.json
- data/chunks_llm_gpt/localpos_v6/batch_run_summary.json

Usage:
  python phase_1/chunking_with_llm_gpt_localpos_v6_hybrid_batch.py 2010
  python phase_1/chunking_with_llm_gpt_localpos_v6_hybrid_batch.py --years 1984,2010
  python phase_1/chunking_with_llm_gpt_localpos_v6_hybrid_batch.py --all
  python phase_1/chunking_with_llm_gpt_localpos_v6_hybrid_batch.py --all --dry-run
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
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

# ------------------------- Paths & Defaults -------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../buffet_style_investing_agent
DATA_DIR = REPO_ROOT / "data"
INPUT_DIR = DATA_DIR / "text_extracted_letters"
OUT_DIR = DATA_DIR / "chunks_llm_gpt" / "localpos_v7_2_langextract"

RULES_PATH = Path(os.getenv("CHUNKING_RULES_PATH", str(REPO_ROOT / "phase_1" / "chunking_rule_claude.md")))
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Window sizing: conservative to prevent output truncation.
WINDOW_MAX_CHARS = int(os.getenv("CHUNK_WINDOW_MAX_CHARS", "18000"))
WINDOW_MIN_CHARS = int(os.getenv("CHUNK_WINDOW_MIN_CHARS", "3500"))
WINDOW_OVERLAP_CHARS = int(os.getenv("CHUNK_WINDOW_OVERLAP_CHARS", "800"))

# LangExtract-style scaling knobs
EXTRACTION_PASSES = int(os.getenv("CHUNK_EXTRACTION_PASSES", "2"))
MAX_WORKERS = int(os.getenv("CHUNK_MAX_WORKERS", "10"))

USE_MOCK_LLM = False  # set via CLI --mock-llm for local testing
# Per-pass boundary jitter (fraction of max_chars). Helps surface content near boundaries.
PASS_START_OFFSET_FRAC = float(os.getenv("CHUNK_PASS_START_OFFSET_FRAC", "0.33"))
# Per-pass ideal-end ratio (smaller -> earlier boundaries)
PASS_IDEAL_END_RATIOS = os.getenv("CHUNK_PASS_IDEAL_END_RATIOS", "0.85,0.80,0.90")

# Integrity gates (tuned for downstream embeddings/trends reliability)
MIN_COVERAGE_FRACTION = float(os.getenv("CHUNK_MIN_COVERAGE_FRACTION", "0.99"))
MAX_SINGLE_GAP_CHARS = int(os.getenv("CHUNK_MAX_SINGLE_GAP_CHARS", "250"))
MAX_TOTAL_GAP_CHARS = int(os.getenv("CHUNK_MAX_TOTAL_GAP_CHARS", "800"))
MAX_SINGLE_OVERLAP_CHARS = int(os.getenv("CHUNK_MAX_SINGLE_OVERLAP_CHARS", "1500"))

# Auto-fill: when gaps exist, we re-run only missing spans with padding.
GAP_FILL_PADDING = int(os.getenv("CHUNK_GAP_FILL_PADDING", "1200"))
MAX_GAP_FILL_CALLS = int(os.getenv("CHUNK_MAX_GAP_FILL_CALLS", "6"))

# De-dup similarity threshold (fast heuristic)
DEDUP_JACCARD_THRESHOLD = float(os.getenv("CHUNK_DEDUP_JACCARD", "0.92"))

# ------------------------- Retry config -------------------------

@dataclass
class LLMRetryConfig:
    max_attempts: int = 5
    base_sleep_s: float = 1.6
    max_sleep_s: float = 18.0


# ------------------------- Utilities -------------------------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _list_years_from_input_dir() -> List[int]:
    years: List[int] = []
    if not INPUT_DIR.exists():
        return years
    for p in INPUT_DIR.glob("*_cleaned.txt"):
        m = re.match(r"^(\d{4})_cleaned\.txt$", p.name)
        if m:
            try:
                years.append(int(m.group(1)))
            except ValueError:
                pass
    years.sort()
    return years


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _is_safe_boundary(text: str, pos: int) -> bool:
    """Return True if boundary at pos does not split a word token."""
    if pos <= 0 or pos >= len(text):
        return True
    a = text[pos - 1]
    b = text[pos]
    # avoid splitting alnum sequences
    if a.isalnum() and b.isalnum():
        return False
    return True


def _snap_to_whitespace_before(text: str, pos: int, *, min_pos: int) -> int:
    """Find whitespace before pos; return pos if none found."""
    pos = _clamp_int(pos, min_pos, len(text))
    j = text.rfind(" ", min_pos, pos)
    k = text.rfind("\n", min_pos, pos)
    cand = max(j, k)
    return cand + 1 if cand != -1 else pos


def _snap_to_parabreak_before(text: str, pos: int, *, min_pos: int) -> Optional[int]:
    """Prefer paragraph break (\n\n) before pos."""
    pos = _clamp_int(pos, min_pos, len(text))
    idx = text.rfind("\n\n", min_pos, pos)
    if idx == -1:
        return None
    return idx + 2


def _snap_to_sentence_before(text: str, pos: int, *, min_pos: int) -> Optional[int]:
    """Find a sentence-ish boundary before pos."""
    pos = _clamp_int(pos, min_pos, len(text))
    snippet = text[min_pos:pos]
    m = re.finditer(r"[\.!?]\s", snippet)
    last = None
    for mm in m:
        last = mm.end()
    if last is None:
        return None
    return min_pos + last


# ------------------------- Header detection (soft hints) -------------------------

_HEADER_RE = re.compile(
    r"(?m)^(?P<h>[A-Z][A-Z0-9&\-\—\s]{6,}|[A-Z][A-Za-z0-9&\-\—\s]{6,})\s*$"
)

def _detect_section_breaks(letter_text: str) -> List[int]:
    """
    Soft section break hints based on header-like lines.
    We use these only as *preferred* breakpoints when selecting window boundaries.
    """
    breaks = set([0, len(letter_text)])
    for m in _HEADER_RE.finditer(letter_text):
        start = m.start()
        # Avoid headers too close to start/end
        if 0 < start < len(letter_text):
            breaks.add(start)
    # Also include strong paragraph breaks as generic hints
    for m in re.finditer(r"\n{3,}", letter_text):
        breaks.add(m.start())
    return sorted(breaks)


# ------------------------- LangExtract-inspired iterators -------------------------

@dataclass(frozen=True)
class TextChunk:
    """A span of the original document."""
    start: int
    end: int  # exclusive
    def text(self, doc: str) -> str:
        return doc[self.start:self.end]


class SentenceIterator:
    """Lightweight sentence-ish iterator that returns (start,end) spans."""

    _SENT_END_RE = re.compile(r"([\.!?][\"\']?)(\s+|$)")

    def __init__(self, doc: str):
        self.doc = doc

    def __iter__(self):
        s = self.doc
        n = len(s)
        i = 0
        while i < n:
            # Skip leading whitespace
            while i < n and s[i].isspace():
                i += 1
            if i >= n:
                break
            # Find next sentence end
            m = self._SENT_END_RE.search(s, i)
            if not m:
                yield (i, n)
                break
            end = m.end()
            yield (i, end)
            i = end


class ChunkIterator:
    """
    Packs sentence spans into ~max_chars chunks, with overlap, preferring soft break hints.

    This mirrors LangExtract's idea of producing smaller, focused contexts while keeping
    boundaries stable (sentence/paragraph aware), which tends to improve extraction quality.
    """

    def __init__(
        self,
        doc: str,
        *,
        max_chars: int,
        min_chars: int,
        overlap_chars: int,
        preferred_breaks: Optional[List[int]] = None,
        ideal_end_ratio: float = 0.85,
        start_at: int = 0,
    ):

        self.doc = doc
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars
        self.preferred_breaks = preferred_breaks or [0, len(doc)]
        self.ideal_end_ratio = ideal_end_ratio
        self.start_at = max(0, int(start_at))

        # Precompute sentence spans
        self.sentences = list(SentenceIterator(doc))

    def __iter__(self):
        doc = self.doc
        n = len(doc)
        if n == 0:
            return

        # Map from char position to closest sentence start at/after position
        sent_starts = [a for a, _ in self.sentences] or [0]

        def snap_start(pos: int) -> int:
            # choose the closest sentence start at or BEFORE pos to avoid skipping content
            for ss in reversed(sent_starts):
                if ss <= pos:
                    return ss
            return 0

        start = snap_start(self.start_at)

        chunk_id = 0
        while start < n:
            # Aim end around ideal ratio of max_chars, but allow up to max_chars
            ideal_end = min(n, start + int(self.max_chars * self.ideal_end_ratio))
            max_end = min(n, start + self.max_chars)

            end = _choose_window_end(
                doc,
                start,
                ideal_end,
                max_end,
                self.preferred_breaks,
            )

            # Ensure min_chars unless we're at the end
            if end - start < self.min_chars and end < n:
                end = min(n, start + self.min_chars)
                end = _snap_to_whitespace_before(doc, end, min_pos=start)

            # Final safety: do not split a word
            if not _is_safe_boundary(doc, end):
                end = _snap_to_whitespace_before(doc, end, min_pos=start)

            if end <= start:
                end = min(n, start + self.min_chars)

            yield TextChunk(start=start, end=end)

            chunk_id += 1

            if end >= n:
                break

            # Next start with overlap
            next_start = max(0, end - self.overlap_chars)
            next_start = snap_start(next_start)
            if next_start <= start:
                next_start = end
            start = next_start


def make_batches_of_textchunk(chunks: List[TextChunk], batch_length: int) -> List[List[TextChunk]]:
    """Group chunks into batches for inference calls."""
    if batch_length <= 1:
        return [[c] for c in chunks]
    out: List[List[TextChunk]] = []
    for i in range(0, len(chunks), batch_length):
        out.append(chunks[i:i + batch_length])
    return out



def _choose_window_end(
    text: str,
    start: int,
    ideal_end: int,
    max_end: int,
    preferred_breaks: List[int],
) -> int:
    """
    Choose end boundary for window [start, end) with preference order:
      1) preferred_break within [ideal_end-1200, max_end]
      2) paragraph break before max_end
      3) sentence boundary before max_end
      4) whitespace before max_end
      5) hard max_end
    Always ensures end > start.
    """
    n = len(text)
    start = _clamp_int(start, 0, n)
    max_end = _clamp_int(max_end, start + 1, n)
    ideal_end = _clamp_int(ideal_end, start + 1, max_end)

    # Effective minimum chunk length for this (possibly short) span.
    eff_min = min(WINDOW_MIN_CHARS, max(0, max_end - start - 1))

    # (1) preferred breaks close to ideal, but not too early
    lo = max(start + eff_min, ideal_end - 1200)
    hi = max_end
    # binary search-ish scan of breaks
    best = None
    for b in preferred_breaks:
        if lo <= b <= hi:
            if best is None or abs(b - ideal_end) < abs(best - ideal_end):
                best = b
    if best is not None and best > start:
        return best

    # (2) paragraph break
    pb = _snap_to_parabreak_before(text, max_end, min_pos=start + eff_min)
    if pb is not None and pb > start:
        return pb

    # (3) sentence boundary
    sb = _snap_to_sentence_before(text, max_end, min_pos=start + eff_min)
    if sb is not None and sb > start:
        return sb

    # (4) whitespace
    ws = _snap_to_whitespace_before(text, max_end, min_pos=start + eff_min)
    if ws > start:
        return ws

    return max_end


def _split_letter_into_hybrid_windows_legacy(
    letter_text: str,
    *,
    max_chars: int = WINDOW_MAX_CHARS,
    overlap_chars: int = WINDOW_OVERLAP_CHARS,
    start_offset: int = 0,
    ideal_end_ratio: float = 0.85,
) -> List[Tuple[int, int, str]]:
    """
    Split letter into overlapping windows.
    Uses header/section hints (c) to pick better boundaries, but guarantees full coverage (g).
    """
    n = len(letter_text)
    if n == 0:
        return []

    preferred = _detect_section_breaks(letter_text)

    windows: List[Tuple[int, int, str]] = []
    start = max(0, int(start_offset))

    # Ensure we always cover the prefix [0, start) in every pass.
    if start > 0:
        prefix_end = _choose_window_end(letter_text, 0, min(n, int(max_chars * ideal_end_ratio)), min(n, max_chars), preferred)
        windows.append((0, prefix_end, letter_text[0:prefix_end]))
        start = max(0, prefix_end - overlap_chars)

    while start < n:
        max_end = min(n, start + max_chars)
        ideal_end = min(n, start + int(max_chars * float(ideal_end_ratio)))
        end = _choose_window_end(letter_text, start, ideal_end, max_end, preferred)

        if end <= start:
            end = min(n, start + max_chars)

        win_text = letter_text[start:end]
        windows.append((start, end, win_text))

        if end >= n:
            break

        # next start with overlap
        start = max(0, end - overlap_chars)

    # ensure monotonic coverage (defensive)
    fixed: List[Tuple[int, int, str]] = []
    last_end = -1
    for s, e, t in windows:
        if e <= s:
            continue
        if s < last_end - overlap_chars - 10:
            s = max(0, last_end - overlap_chars)
            t = letter_text[s:e]
        fixed.append((s, e, t))
        last_end = e
    return fixed


# ------------------------- LLM JSON salvage -------------------------

def extract_chunk_objects_from_response(raw: str) -> List[Dict[str, Any]]:
    """
    Last-resort salvage: extract standalone JSON objects containing "chunk_id".
    Note: only catches *flat* objects. It's a safety net, not a primary path.
    """
    out: List[Dict[str, Any]] = []
    pattern = re.compile(r'\{[^{}]*"chunk_id"\s*:\s*"[^"]+"[^{}]*\}', re.DOTALL)
    for m in pattern.finditer(raw):
        s = m.group(0)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "chunk_id" in obj:
                out.append(obj)
        except Exception:
            continue
    return out


# ------------------------- Local repair + rebuild -------------------------

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

    candidates: List[int] = []
    start = 0
    while True:
        idx = letter_text.find(sa, start)
        if idx == -1:
            break
        candidates.append(idx)
        start = idx + 1
        if len(candidates) >= max_candidates:
            break

    best: Optional[Tuple[int, int]] = None
    for s in candidates:
        s_end = s + len(sa)
        search_to = min(len(letter_text), s_end + max_span)
        e = letter_text.find(ea, s_end, search_to)
        if e == -1:
            continue
        e_end = e + len(ea)
        span = e_end - s
        if span <= 0:
            continue
        if best is None or span < (best[1] - best[0]):
            best = (s, e_end)
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

    # Use first/last 40 chars as raw hints
    sa_hint = start_anchor.strip()[:40]
    ea_hint = end_anchor.strip()[-40:]
    return _repair_offsets_by_raw_anchors(letter_text, sa_hint, ea_hint, max_candidates=max_candidates)


def _rebuild_chunk_text_and_counts(letter_text: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = len(letter_text)
    fixed: List[Dict[str, Any]] = []
    for ch in chunks:
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


def _fix_midword_and_unsafe_boundaries(letter_text: str, chunks: List[Dict[str, Any]], diagnostics: Dict[str, Any]) -> None:
    """Adjust chunk boundaries if they start mid-word or end mid-word."""
    chunks.sort(key=lambda x: (x.get("start_char", 0), x.get("end_char", 0)))
    midword_fixes = 0
    unsafe_fixes = 0

    for i, ch in enumerate(chunks):
        s = ch.get("start_char")
        e = ch.get("end_char")
        if not (isinstance(s, int) and isinstance(e, int) and s < e):
            continue
        if not _is_safe_boundary(letter_text, s):
            # shift start to next safe boundary within +60 chars
            shifted = None
            for k in range(1, 61):
                p = s + k
                if p < len(letter_text) and _is_safe_boundary(letter_text, p):
                    shifted = p
                    break
            if shifted is not None and shifted < e:
                ch["start_char"] = shifted
                midword_fixes += 1

        if not _is_safe_boundary(letter_text, e):
            shifted = None
            for k in range(1, 61):
                p = e - k
                if p > s and _is_safe_boundary(letter_text, p):
                    shifted = p
                    break
            if shifted is not None and shifted > s:
                ch["end_char"] = shifted
                unsafe_fixes += 1

    diagnostics["midword_boundary_fixes"] = midword_fixes
    diagnostics["unsafe_boundary_fixes"] = unsafe_fixes


# ------------------------- Integrity validation -------------------------

def _validate_output_integrity(letter_text: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(letter_text)
    spans: List[Tuple[int, int]] = []
    for ch in chunks:
        s, e = ch.get("start_char"), ch.get("end_char")
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= n:
            spans.append((s, e))
    spans.sort()
    if not spans:
        return {"coverage_fraction": 0.0, "gaps": 0, "overlaps": 0, "max_gap": n, "max_overlap": 0, "total_gap": n}

    # union coverage
    covered = 0
    gaps = 0
    overlaps = 0
    max_gap = 0
    max_overlap = 0
    total_gap = 0

    cur_s, cur_e = spans[0]
    covered = 0
    if cur_s > 0:
        gaps += 1
        max_gap = max(max_gap, cur_s)
        total_gap += cur_s

    for s, e in spans[1:]:
        if s > cur_e:
            gap = s - cur_e
            gaps += 1
            max_gap = max(max_gap, gap)
            total_gap += gap
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
        elif s < cur_e:
            ov = cur_e - s
            overlaps += 1
            max_overlap = max(max_overlap, ov)
            cur_e = max(cur_e, e)
        else:
            cur_e = max(cur_e, e)

    covered += cur_e - cur_s
    if cur_e < n:
        gaps += 1
        gap = n - cur_e
        max_gap = max(max_gap, gap)
        total_gap += gap

    coverage_fraction = covered / n if n else 1.0
    return {
        "coverage_fraction": coverage_fraction,
        "gaps": gaps,
        "overlaps": overlaps,
        "max_gap": max_gap,
        "max_overlap": max_overlap,
        "total_gap": total_gap,
        "valid_spans": len(spans),
        "doc_chars": n,
    }


def _find_uncovered_intervals(letter_text: str, chunks: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Return uncovered intervals in [0,n)."""
    n = len(letter_text)
    spans: List[Tuple[int, int]] = []
    for ch in chunks:
        s, e = ch.get("start_char"), ch.get("end_char")
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= n:
            spans.append((s, e))
    spans.sort()
    uncovered: List[Tuple[int, int]] = []
    cur = 0
    for s, e in spans:
        if s > cur:
            uncovered.append((cur, s))
        cur = max(cur, e)
    if cur < n:
        uncovered.append((cur, n))
    # drop tiny
    uncovered = [(a,b) for a,b in uncovered if b-a > 40]
    return uncovered


# ------------------------- De-dup merge -------------------------

def _token_set(s: str) -> set:
    toks = re.findall(r"[A-Za-z0-9']+", (s or "").lower())
    # downsample very long sets for speed
    if len(toks) > 1200:
        toks = toks[:1200]
    return set(toks)

def _jaccard(a: str, b: str) -> float:
    A = _token_set(a)
    B = _token_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


def _dedup_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = sorted(chunks, key=lambda x: (x.get("start_char", 0), x.get("end_char", 0)))
    deduped: List[Dict[str, Any]] = []
    for ch in chunks:
        if not deduped:
            deduped.append(ch)
            continue
        prev = deduped[-1]
        ps, pe = prev.get("start_char"), prev.get("end_char")
        cs, ce = ch.get("start_char"), ch.get("end_char")
        if isinstance(ps, int) and isinstance(pe, int) and isinstance(cs, int) and isinstance(ce, int):
            # if significant overlap and text is near-duplicate, keep the "better" chunk (longer + safer boundary)
            if cs < pe:
                j = _jaccard(prev.get("chunk_text",""), ch.get("chunk_text",""))
                if j >= DEDUP_JACCARD_THRESHOLD:
                    prev_safe = _is_safe_boundary(prev.get("chunk_text",""), 0)
                    cur_safe = _is_safe_boundary(ch.get("chunk_text",""), 0)
                    # simple preference: safer start boundary + longer text
                    prev_len = len(prev.get("chunk_text","") or "")
                    cur_len = len(ch.get("chunk_text","") or "")
                    if (cur_safe and not prev_safe) or (cur_len > prev_len * 1.15):
                        deduped[-1] = ch
                    continue
        deduped.append(ch)
    return deduped


def _renumber_chunk_ids_in_order(chunks: List[Dict[str, Any]], year: int) -> None:
    """
    Deterministic renumbering in global document order.
    Keeps section_type if present, but ensures stable ordering for downstream analytics.
    """
    chunks.sort(key=lambda x: (x.get("start_char", 0), x.get("end_char", 0)))
    for i, ch in enumerate(chunks, 1):
        st = (ch.get("section_type") or "unknown").strip().lower()
        st = re.sub(r"[^a-z0-9_]+", "_", st) or "unknown"
        ch["chunk_id"] = f"{year}_{st}_{i:03d}"
        ch["year"] = year





def split_letter_into_hybrid_windows(
    letter_text: str,
    *,
    max_chars: int = WINDOW_MAX_CHARS,
    min_chars: int = WINDOW_MIN_CHARS,
    overlap_chars: int = WINDOW_OVERLAP_CHARS,
    start_offset: int = 0,
    ideal_end_ratio: float = 0.85,
) -> List[Tuple[int, int, str]]:
    """
    LangExtract-inspired windowing:

    - sentence-aware packing (ChunkIterator) to avoid mid-sentence boundaries when possible
    - optional per-pass start_offset jitter (used by multi-pass recall)
    - still uses soft header/paragraph hints via _detect_section_breaks + _choose_window_end
    """
    n = len(letter_text)
    if n == 0:
        return []

    preferred = _detect_section_breaks(letter_text)
    start = max(0, int(start_offset))

    windows: List[Tuple[int, int, str]] = []

    # Ensure we always cover the prefix [0, start) in every pass so no content is omitted.
    if start > 0:
        prefix_end = _choose_window_end(
            letter_text,
            0,
            min(n, int(max_chars * ideal_end_ratio)),
            min(n, max_chars),
            preferred,
        )
        windows.append((0, prefix_end, letter_text[0:prefix_end]))

    use_sentence_iterator = os.getenv("CHUNK_USE_SENTENCE_ITERATOR", "1").strip() not in {"0", "false", "False"}

    if use_sentence_iterator:
        it = ChunkIterator(
            letter_text,
            max_chars=max_chars,
            min_chars=min_chars,
            overlap_chars=overlap_chars,
            preferred_breaks=preferred,
            ideal_end_ratio=ideal_end_ratio,
            start_at=start,
        )
        for c in it:
            windows.append((c.start, c.end, letter_text[c.start:c.end]))
    else:
        # Fallback to the original hybrid splitter (char-based with snapping)
        windows.extend(_split_letter_into_hybrid_windows_legacy(
            letter_text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            start_offset=start_offset,
            ideal_end_ratio=ideal_end_ratio,
        ))

    # Deduplicate exact duplicates (can happen with prefix + iterator overlap)
    seen = set()
    out: List[Tuple[int, int, str]] = []
    for s, e, t in windows:
        key = (s, e)
        if key in seen:
            continue
        seen.add(key)
        out.append((s, e, t))
    return out


# ------------------------- LLM call (window-relative offsets) -------------------------

def _is_retryable_openai_error(exc: Exception) -> bool:
    msg = str(exc)
    retry_markers = ["timeout", "timed out", "429", "500", "502", "503", "504", "rate limit"]
    return any(m.lower() in msg.lower() for m in retry_markers)


def _parse_llm_json(content: str) -> Optional[List[Dict[str, Any]]]:
    try:
        obj = json.loads(content)
    except Exception:
        obj = None

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and isinstance(obj.get("chunks"), list):
        return [x for x in obj["chunks"] if isinstance(x, dict)]
    return None


def _llm_call_chunks_window(
    *,
    window_text: str,
    year: int,
    source_file: str,
    chunking_spec: str,
    model: str,
    raw_debug_path: Path,
    window_id: int,
    retry: Optional[LLMRetryConfig] = None,
) -> List[Dict[str, Any]]:
    if USE_MOCK_LLM:
        # Deterministic mock output: one chunk spanning the whole window (inclusive end).
        if not window_text:
            return []
        return [{"title": "MOCK", "start_char": 0, "end_char": len(window_text) - 1}]

    if OpenAI is None:
        raise RuntimeError('openai package not available; install openai or use --mock-llm/--dry-run')
    client = OpenAI()
    retry = retry or LLMRetryConfig()

    system = "You are a precise document chunking engine. Output ONLY valid JSON."
    user = f"""
Chunk this WINDOW of the Berkshire Hathaway shareholder letter for year {year}.
Source file: {source_file}
Window id: {window_id}
Offsets MUST be relative to WINDOW TEXT (0-indexed within WINDOW TEXT).

Rules you MUST follow:
- Output MUST be a JSON array of chunk objects. No surrounding text.
- start_char/end_char must match exact substring boundaries in WINDOW TEXT.
- Include start_anchor and end_anchor (short snippets) for repair.
- Prefer coherent sections; avoid splitting tables.
- section_type must be one of the allowed labels from the spec.

Chunking spec:
{chunking_spec}

WINDOW TEXT:
\"\"\"{window_text}\"\"\"
""".strip()

    last_exc: Optional[Exception] = None

    for attempt in range(1, retry.max_attempts + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            content = resp.output_text or ""
            raw_debug_path.parent.mkdir(parents=True, exist_ok=True)
            raw_debug_path.write_text(content, encoding="utf-8", errors="replace")

            parsed = _parse_llm_json(content)
            if parsed is not None:
                return parsed

            # trim and retry parse
            a = min([i for i in [content.find("["), content.find("{")] if i != -1], default=-1)
            b = max(content.rfind("]"), content.rfind("}"))
            if a != -1 and b != -1 and b > a:
                trimmed = content[a : b + 1]
                parsed2 = _parse_llm_json(trimmed)
                if parsed2 is not None:
                    return parsed2

            salvaged = extract_chunk_objects_from_response(content)
            if salvaged:
                print(f"[WARN] Window {window_id}: Whole-response JSON parse failed; salvaged {len(salvaged)} objects.")
                return salvaged

            raise RuntimeError(f"Window {window_id}: invalid/unusable JSON (see {raw_debug_path})")

        except Exception as exc:
            last_exc = exc
            if attempt >= retry.max_attempts or not _is_retryable_openai_error(exc):
                break
            sleep_s = min(retry.max_sleep_s, retry.base_sleep_s * (2 ** (attempt - 1)) + random.random())
            print(f"[WARN] LLM call failed (attempt {attempt}/{retry.max_attempts}) for window {window_id}: {exc}")
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM call failed for window {window_id}: {last_exc}")


# ------------------------- Pipeline per year -------------------------

def _run_windows_parallel(
    *,
    windows: List[Tuple[int, int, str]],
    year: int,
    in_path: Path,
    chunking_spec: str,
    model: str,
    raw_dir: Path,
    pass_id: int,
    max_workers: int,
) -> List[Dict[str, Any]]:
    """Run LLM chunking on windows in parallel and return globally-mapped chunks."""
    tasks = []
    results: List[Dict[str, Any]] = []
    raw_pass_dir = raw_dir / f"pass_{pass_id:02d}"
    raw_pass_dir.mkdir(parents=True, exist_ok=True)

    def _worker(wid: int, ws: int, we: int, wtxt: str) -> List[Dict[str, Any]]:
        raw_path = raw_pass_dir / f"window_{wid:03d}_{ws}_{we}.txt"
        win_chunks = _llm_call_chunks_window(
            window_text=wtxt,
            year=year,
            source_file=in_path.name,
            chunking_spec=chunking_spec,
            model=model,
            raw_debug_path=raw_path,
            window_id=(pass_id * 1000) + wid,
        )
        letter_text = getattr(_run_windows_parallel, '_letter_text', '')
        mapped = _map_window_offsets_to_global(win_chunks, window_start=ws, window_text=wtxt, letter_text=letter_text)
        return mapped

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for wid, (ws, we, wtxt) in enumerate(windows, 1):
            tasks.append(ex.submit(_worker, wid, ws, we, wtxt))
        for fut in as_completed(tasks):
            results.extend(fut.result())
    return results

def _map_window_offsets_to_global(
    window_chunks: List[Dict[str, Any]],
    *,
    window_start: int,
    window_text: str,
    letter_text: str,
) -> List[Dict[str, Any]]:
    """Convert window-relative offsets to global offsets, then repair using anchors if needed."""
    out: List[Dict[str, Any]] = []
    n = len(letter_text)
    wn = len(window_text)

    for ch in window_chunks:
        s = ch.get("start_char")
        e = ch.get("end_char")

        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= wn:
            gs = window_start + s
            ge = window_start + e
            ch["start_char"] = _clamp_int(gs, 0, n)
            ch["end_char"] = _clamp_int(ge, 0, n)
        else:
            # attempt anchor repair
            sa = ch.get("start_anchor") or ""
            ea = ch.get("end_anchor") or ""
            repaired = _repair_offsets_by_raw_anchors(letter_text, sa, ea)
            if repaired is None:
                repaired = _repair_offsets_by_normalized_anchors(letter_text, sa, ea)
            if repaired is not None:
                ch["start_char"], ch["end_char"] = repaired
            else:
                # mark invalid; will be filtered by rebuild
                ch["start_char"], ch["end_char"] = None, None

        out.append(ch)

    return out


def _chunk_year(
    year: int,
    *,
    model: str,
    rules_path: Path,
    force: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    in_path = INPUT_DIR / f"{year}_cleaned.txt"
    if not in_path.exists():
        return {"year": year, "status": "missing_input", "input": str(in_path)}

    out_jsonl = OUT_DIR / f"{year}_chunks_llm.jsonl"
    diag_path = OUT_DIR / f"{year}_offset_diagnostics.json"
    raw_dir = OUT_DIR / f"{year}_raw_llm_responses"

    if out_jsonl.exists() and not force:
        return {"year": year, "status": "skipped_exists", "output": str(out_jsonl)}

    letter_text = _read_text(in_path)
    chunking_spec = _read_text(rules_path)

    windows = split_letter_into_hybrid_windows(letter_text)
    if dry_run:
        return {
            "year": year,
            "status": "dry_run",
            "input_chars": len(letter_text),
            "windows": [{"start": s, "end": e, "chars": e - s} for s, e, _ in windows[:10]],
            "windows_count": len(windows),
        }

    all_chunks: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {"year": year, "model": model, "input_chars": len(letter_text), "windows_count": len(windows)}

    # LangExtract-style multiple passes: run chunking over different window boundary configurations.
    # This improves recall around boundaries and reduces the chance of systematic misses.
    ratios: List[float] = []
    for r in PASS_IDEAL_END_RATIOS.split(","):
        r = r.strip()
        if r:
            try:
                ratios.append(float(r))
            except ValueError:
                pass
    if not ratios:
        ratios = [0.85, 0.80, 0.90]

    passes = max(1, int(EXTRACTION_PASSES))
    pass_summaries: List[Dict[str, Any]] = []

    # Provide letter_text to parallel worker via function attribute (avoids copying huge strings per task)
    _run_windows_parallel._letter_text = letter_text  # type: ignore[attr-defined]

    for pid in range(1, passes + 1):
        ideal_ratio = ratios[(pid - 1) % len(ratios)]
        start_offset = 0
        if pid > 1:
            start_offset = int(WINDOW_MAX_CHARS * PASS_START_OFFSET_FRAC * (pid - 1))

        pass_windows = split_letter_into_hybrid_windows(
            letter_text,
            max_chars=WINDOW_MAX_CHARS,
            overlap_chars=WINDOW_OVERLAP_CHARS,
            start_offset=start_offset,
            ideal_end_ratio=ideal_ratio,
        )

        # run in parallel for speed (like LangExtract max_workers)
        mapped_chunks = _run_windows_parallel(
            windows=pass_windows,
            year=year,
            in_path=in_path,
            chunking_spec=chunking_spec,
            model=model,
            raw_dir=raw_dir,
            pass_id=pid,
            max_workers=MAX_WORKERS,
        )
        all_chunks.extend(mapped_chunks)
        pass_summaries.append(
            {
                "pass_id": pid,
                "ideal_end_ratio": ideal_ratio,
                "start_offset": start_offset,
                "windows_count": len(pass_windows),
                "chunks_returned": len(mapped_chunks),
            }
        )

    diagnostics["passes"] = pass_summaries


    # rebuild + boundary fixes
    all_chunks = _rebuild_chunk_text_and_counts(letter_text, all_chunks)
    _fix_midword_and_unsafe_boundaries(letter_text, all_chunks, diagnostics)
    all_chunks = _rebuild_chunk_text_and_counts(letter_text, all_chunks)

    # dedup
    before_dedup = len(all_chunks)
    all_chunks = _dedup_chunks(all_chunks)
    diagnostics["dedup_removed"] = max(0, before_dedup - len(all_chunks))

    # integrity gate
    integrity = _validate_output_integrity(letter_text, all_chunks)
    diagnostics["integrity"] = integrity

    # auto-fill uncovered spans if needed
    fill_calls = 0
    if (
        integrity["coverage_fraction"] < MIN_COVERAGE_FRACTION
        or integrity["max_gap"] > MAX_SINGLE_GAP_CHARS
        or integrity["total_gap"] > MAX_TOTAL_GAP_CHARS
    ):
        uncovered = _find_uncovered_intervals(letter_text, all_chunks)
        # prioritize biggest gaps
        uncovered.sort(key=lambda x: (x[1] - x[0]), reverse=True)
        supplements: List[Dict[str, Any]] = []

        for (a, b) in uncovered:
            if fill_calls >= MAX_GAP_FILL_CALLS:
                break
            # add padding for context
            s = max(0, a - GAP_FILL_PADDING)
            e = min(len(letter_text), b + GAP_FILL_PADDING)
            wtxt = letter_text[s:e]
            fill_calls += 1
            raw_path = raw_dir / f"gapfill_{fill_calls:02d}_{s}_{e}.txt"
            try:
                win_chunks = _llm_call_chunks_window(
                    window_text=wtxt,
                    year=year,
                    source_file=in_path.name,
                    chunking_spec=chunking_spec,
                    model=model,
                    raw_debug_path=raw_path,
                    window_id=10_000 + fill_calls,
                )
                mapped = _map_window_offsets_to_global(win_chunks, window_start=s, window_text=wtxt, letter_text=letter_text)
                supplements.extend(mapped)
            except Exception as exc:
                diagnostics.setdefault("gapfill_errors", []).append(str(exc))

        if supplements:
            all_chunks.extend(supplements)
            all_chunks = _rebuild_chunk_text_and_counts(letter_text, all_chunks)
            _fix_midword_and_unsafe_boundaries(letter_text, all_chunks, diagnostics)
            all_chunks = _rebuild_chunk_text_and_counts(letter_text, all_chunks)
            all_chunks = _dedup_chunks(all_chunks)
            integrity2 = _validate_output_integrity(letter_text, all_chunks)
            diagnostics["integrity_after_gapfill"] = integrity2
            diagnostics["gapfill_calls"] = fill_calls

            integrity = integrity2

    # final hard-gate
    hard_fail = (
        integrity["coverage_fraction"] < MIN_COVERAGE_FRACTION
        or integrity["max_gap"] > MAX_SINGLE_GAP_CHARS * 2
        or integrity["max_overlap"] > MAX_SINGLE_OVERLAP_CHARS * 3
    )
    if hard_fail:
        diagnostics["status"] = "failed_integrity_gate"
        _write_json(diag_path, diagnostics)
        # still write what we have for inspection
        _renumber_chunk_ids_in_order(all_chunks, year)
        _write_jsonl(out_jsonl, all_chunks)
        return {"year": year, "status": "failed_integrity_gate", "output": str(out_jsonl), "diagnostics": str(diag_path)}

    _renumber_chunk_ids_in_order(all_chunks, year)
    _write_jsonl(out_jsonl, all_chunks)
    _write_json(diag_path, diagnostics)

    return {"year": year, "status": "ok", "chunks": len(all_chunks), "output": str(out_jsonl), "diagnostics": str(diag_path)}


# ------------------------- CLI -------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("years", nargs="*", help="Years to process, e.g. 2009 2010")
    p.add_argument("--years", dest="years_csv", default="", help="Comma-separated years, e.g. 2009,2010")
    p.add_argument("--all", action="store_true", help="Process all *_cleaned.txt in input dir")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model (default from env OPENAI_MODEL)")
    p.add_argument("--rules", default=str(RULES_PATH), help="Path to chunking rules markdown")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--dry-run", action="store_true", help="Discover inputs/windows, no LLM calls")
    p.add_argument("--mock-llm", action="store_true", help="Use a deterministic mock LLM (no API calls)")
    p.add_argument("--self-test", action="store_true", help="Run built-in self tests and exit")
    return p.parse_args(argv)




def _self_test() -> Tuple[bool, Dict[str, Any]]:
    """Fast local tests: windowing, offsets, boundary safety, and multi-pass merge invariants."""
    report: Dict[str, Any] = {}
    # Synthetic letter with headings + punctuation to exercise snapping.
    letter = (
        "BERKSHIRE HATHAWAY INC.\n\n"
        "To the Shareholders of Berkshire Hathaway Inc.:\n\n"
        "This is sentence one. This is sentence two! This is sentence three?\n\n"
        "THE OPERATING BUSINESSES\n"
        "We like durable moats. We avoid leverage.\n\n"
        "CAPITAL ALLOCATION\n"
        "Buybacks matter when shares are below intrinsic value.\n"
        "End.\n"
    )
    preferred = _detect_section_breaks(letter)
    it = ChunkIterator(letter, max_chars=120, min_chars=60, overlap_chars=20, preferred_breaks=preferred, ideal_end_ratio=0.85, start_at=0)
    chunks = list(it)
    report["num_chunks"] = len(chunks)
    assert len(chunks) >= 1, "Expected at least one chunk"
    if len(letter) > 120:
        assert len(chunks) >= 2, "Expected multiple chunks for synthetic text"

    # Check boundaries safe + monotonic (with overlap allowed)
    for c in chunks:
        assert 0 <= c.start < c.end <= len(letter)
        assert _is_safe_boundary(letter, c.start)
        assert _is_safe_boundary(letter, c.end)

    # Check coverage with overlaps
    covered = [False] * len(letter)
    for c in chunks:
        for i in range(c.start, c.end):
            covered[i] = True
    # Measure coverage on non-whitespace characters (sentence-aware chunking may skip leading ws).
    denom = sum(1 for ch in letter if not ch.isspace())
    numer = sum(1 for i,ch in enumerate(letter) if (not ch.isspace()) and covered[i])
    coverage = numer / max(1, denom)
    report["coverage"] = coverage
    assert coverage >= 0.98, f"Coverage too low: {coverage}"


    report["ok"] = True
    return True, report

def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if getattr(args, 'self_test', False):
        ok, rep = _self_test()
        print('[SELF-TEST]', json.dumps(rep, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    global USE_MOCK_LLM
    USE_MOCK_LLM = bool(getattr(args, 'mock_llm', False))
    model = args.model
    rules_path = Path(args.rules)

    if not rules_path.exists():
        print(f"[ERROR] Rules file not found: {rules_path}", file=sys.stderr)
        return 2

    years: List[int] = []
    if args.all:
        years = _list_years_from_input_dir()
    else:
        if args.years_csv:
            for part in args.years_csv.split(","):
                part = part.strip()
                if part:
                    years.append(int(part))
        for y in args.years:
            years.append(int(y))

    years = sorted(set(years))
    if not years:
        print("[ERROR] No years specified. Use positional years, --years, or --all.", file=sys.stderr)
        return 2

    batch_summary: Dict[str, Any] = {"model": model, "rules": str(rules_path), "out_dir": str(OUT_DIR), "results": []}

    ok = 0
    for year in years:
        print(f"[INFO] Year={year} Model={model} Rules={rules_path.name}")
        try:
            res = _chunk_year(
                year,
                model=model,
                rules_path=rules_path,
                force=args.force,
                dry_run=args.dry_run,
            )
            batch_summary["results"].append(res)
            print(f"[INFO] {year}: {res['status']}")
            if res["status"] in ("ok", "dry_run", "skipped_exists"):
                ok += 1
        except Exception as exc:
            batch_summary["results"].append({"year": year, "status": "error", "error": str(exc)})
            print(f"[ERROR] {year}: {exc}", file=sys.stderr)

    _write_json(OUT_DIR / "batch_run_summary.json", batch_summary)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
