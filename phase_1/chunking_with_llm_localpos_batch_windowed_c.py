#!/usr/bin/env python
"""
chunking_with_llm_gpt_localpos_v5_batch.py

IMPROVED VERSION with Section-Aware Batched Processing

Key improvement over v4: Handles arbitrarily long letters by splitting into 
sections/segments BEFORE calling LLM, then merging results with proper offset tracking.

This prevents:
- Output token exhaustion causing truncated JSON
- Context window overflow
- Quality degradation on very long documents

Strategy:
1. Pre-split document into sections using header detection
2. Process each section (or group of small sections) in separate LLM calls
3. Track global offsets for each segment
4. Merge all chunks with corrected offsets
5. Apply existing repair/validation pipeline
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
OUT_DIR = PROJECT_ROOT / "data" / "chunks_llm_gpt" / "c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "gpt-4.1-mini"

# Segment size thresholds (in characters)
# These are conservative to leave room for output tokens
MAX_SEGMENT_CHARS = 25_000  # ~6k tokens input, leaves room for structured output
MIN_SEGMENT_CHARS = 3_000   # Don't create tiny segments
IDEAL_SEGMENT_CHARS = 15_000  # Target size when grouping small sections

ALLOWED_CHUNK_TYPES = {
    "narrative_story",
    "financial_table",
    "philosophy",
    "business_analysis",
    "administrative",
}

SENT_END = {".", "!", "?"}

# Section header patterns for pre-splitting
SECTION_HEADER_PATTERNS = [
    r"^[A-Z][A-Z\s,&'-]{5,}$",  # ALL CAPS lines
    r"^\*\*[A-Z].*\*\*$",       # Bold markdown
    r"^#{1,3}\s+",              # Markdown headers
    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*[-–—]\s*\d{4}",  # "Section Name - 2008"
    r"^\s*_{3,}\s*$",           # Horizontal rules
]

SECTION_KEYWORDS = [
    "insurance", "geico", "float", "underwriting",
    "acquisitions", "investments", "portfolio",
    "operating", "businesses", "earnings",
    "governance", "compensation", "board",
    "annual meeting", "shareholder",
    "performance", "book value",
]


# ------------------------- Generic helpers -------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


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


# ------------------------- NEW: Document Segmentation -------------------------

@dataclass
class DocumentSegment:
    """A segment of the document for batched processing."""
    text: str
    global_start_offset: int  # Offset in the original full document
    segment_index: int
    total_segments: int
    estimated_section: Optional[str] = None


def _detect_section_breaks(letter_text: str) -> List[int]:
    """
    Detect likely section break positions in the document.
    Returns list of character offsets where sections likely begin.
    """
    breaks = [0]  # Always include start
    
    lines = letter_text.split('\n')
    current_pos = 0
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Check for section header patterns
        is_header = False
        
        # Pattern 1: ALL CAPS line (likely section header)
        if len(line_stripped) > 5 and line_stripped.isupper() and not line_stripped.isdigit():
            is_header = True
        
        # Pattern 2: Contains section keywords and is relatively short
        if len(line_stripped) < 80:
            lower_line = line_stripped.lower()
            if any(kw in lower_line for kw in SECTION_KEYWORDS):
                is_header = True
        
        # Pattern 3: Regex patterns
        for pattern in SECTION_HEADER_PATTERNS:
            if re.match(pattern, line_stripped):
                is_header = True
                break
        
        # Pattern 4: Blank line followed by potential header
        if i > 0 and not lines[i-1].strip() and line_stripped and len(line_stripped) < 100:
            # Check if this looks like a header (capitalized, not too long)
            words = line_stripped.split()
            if words and words[0][0].isupper():
                is_header = True
        
        if is_header and current_pos > 0:
            # Don't add breaks too close together
            if not breaks or current_pos - breaks[-1] > MIN_SEGMENT_CHARS:
                breaks.append(current_pos)
        
        current_pos += len(line) + 1  # +1 for newline
    
    return breaks


def _find_safe_break_point(text: str, target_pos: int, window: int = 2000) -> int:
    """
    Find a safe break point near target_pos (paragraph or sentence boundary).
    """
    n = len(text)
    target_pos = _clamp_int(target_pos, 0, n)
    
    # Look for double newline (paragraph break) first
    search_start = max(0, target_pos - window)
    search_end = min(n, target_pos + window)
    
    # Prefer paragraph breaks
    para_breaks = []
    for m in re.finditer(r'\n\s*\n', text[search_start:search_end]):
        para_breaks.append(search_start + m.end())
    
    if para_breaks:
        # Find the one closest to target
        best = min(para_breaks, key=lambda x: abs(x - target_pos))
        if abs(best - target_pos) < window:
            return best
    
    # Fall back to sentence boundaries
    sentence_ends = []
    for m in re.finditer(r'[.!?]["\')\]]*\s+', text[search_start:search_end]):
        sentence_ends.append(search_start + m.end())
    
    if sentence_ends:
        best = min(sentence_ends, key=lambda x: abs(x - target_pos))
        if abs(best - target_pos) < window:
            return best
    
    return target_pos


def segment_document(letter_text: str, max_chars: int = MAX_SEGMENT_CHARS) -> List[DocumentSegment]:
    """
    Split document into manageable segments for batched LLM processing.
    
    Strategy:
    1. Detect natural section breaks
    2. Group small sections together
    3. Split large sections at paragraph/sentence boundaries
    """
    n = len(letter_text)
    
    if n <= max_chars:
        # Small enough for single call
        return [DocumentSegment(
            text=letter_text,
            global_start_offset=0,
            segment_index=0,
            total_segments=1
        )]
    
    # Detect section breaks
    section_breaks = _detect_section_breaks(letter_text)
    
    # Ensure we have end position
    if section_breaks[-1] != n:
        section_breaks.append(n)
    
    # Build segments by grouping sections
    segments: List[DocumentSegment] = []
    current_start = 0
    current_text_parts: List[str] = []
    current_chars = 0
    
    for i in range(len(section_breaks) - 1):
        section_start = section_breaks[i]
        section_end = section_breaks[i + 1]
        section_text = letter_text[section_start:section_end]
        section_len = len(section_text)
        
        # If this single section is too large, split it
        if section_len > max_chars:
            # Flush current buffer first
            if current_text_parts:
                combined = ''.join(current_text_parts)
                segments.append(DocumentSegment(
                    text=combined,
                    global_start_offset=current_start,
                    segment_index=len(segments),
                    total_segments=0  # Will update later
                ))
                current_text_parts = []
                current_chars = 0
            
            # Split the large section
            pos = section_start
            while pos < section_end:
                chunk_end = min(pos + max_chars, section_end)
                if chunk_end < section_end:
                    chunk_end = _find_safe_break_point(letter_text, chunk_end)
                
                segments.append(DocumentSegment(
                    text=letter_text[pos:chunk_end],
                    global_start_offset=pos,
                    segment_index=len(segments),
                    total_segments=0
                ))
                pos = chunk_end
            
            current_start = section_end
            continue
        
        # Check if adding this section would exceed limit
        if current_chars + section_len > max_chars and current_text_parts:
            # Flush current buffer
            combined = ''.join(current_text_parts)
            segments.append(DocumentSegment(
                text=combined,
                global_start_offset=current_start,
                segment_index=len(segments),
                total_segments=0
            ))
            current_text_parts = []
            current_chars = 0
            current_start = section_start
        
        current_text_parts.append(section_text)
        current_chars += section_len
    
    # Flush remaining
    if current_text_parts:
        combined = ''.join(current_text_parts)
        segments.append(DocumentSegment(
            text=combined,
            global_start_offset=current_start,
            segment_index=len(segments),
            total_segments=0
        ))
    
    # Update total_segments
    for seg in segments:
        seg.total_segments = len(segments)
    
    return segments


# ------------------------- LLM call (with retries) -------------------------

@dataclass
class LLMRetryConfig:
    max_attempts: int = 5
    base_sleep_s: float = 1.5
    max_sleep_s: float = 20.0
    jitter_s: float = 0.8


def _is_retryable_openai_error(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    retry_markers = [
        "rate limit", "timeout", "timed out", "temporarily",
        "server error", "overloaded", "connection",
        "429", "500", "502", "503", "504",
    ]
    return any(m in msg for m in retry_markers)


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


def call_llm_for_segment(
    segment: DocumentSegment,
    year: int,
    source_file: str,
    chunking_spec: str,
    model: str,
    *,
    retry: Optional[LLMRetryConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Call LLM for a single segment of the document.
    Returns chunks with LOCAL offsets (relative to segment start).
    """
    client = OpenAI()
    retry = retry or LLMRetryConfig()
    
    segment_info = ""
    if segment.total_segments > 1:
        segment_info = f"""
NOTE: This is segment {segment.segment_index + 1} of {segment.total_segments} from the full letter.
- Your start_char/end_char offsets should be relative to THIS SEGMENT TEXT (starting at 0).
- The offsets will be adjusted to global positions later.
- If this segment starts mid-section, continue the section from the previous segment.
"""

    system_content = f"""
You are an expert corpus architect for Berkshire Hathaway shareholder letters.
Your job is to chunk a document segment into semantically meaningful chunks and produce rich metadata.
Follow EXACTLY the chunking rule, definitions, and metadata schema in the chunking strategy document.

Hard constraints:
- Return ONLY valid JSON (no prose).
- Top-level MUST be a JSON object: {{"chunks": [ ... ]}}.
- Include accurate start_char/end_char offsets into the provided SEGMENT TEXT.
- Also include start_anchor and end_anchor copied EXACTLY from the SEGMENT TEXT.
- DO generate prev_context and next_context summaries.
- DO NOT include chunk_text/word_count/char_count (they will be reconstructed locally).
- DO NOT include position_in_letter/position_in_section/total_chunks_in_section (they will be computed locally).
{segment_info}
"""

    strategy_message = "Here is the complete chunking rule specification you MUST follow:\n\n" + chunking_spec

    user_content = f"""
Chunk this segment of the Berkshire Hathaway shareholder letter for year {year}.

Source file name: {source_file}
Segment: {segment.segment_index + 1} of {segment.total_segments}

Output schema (each chunk object MUST include these fields):
```json
[
  {{
    "chunk_id": "{year}_{{section_type}}_{{sequence:03d}}",
    "year": {year},
    "source_file": "{source_file}",
    "section_type": "string",
    "section_title": "string",
    "subsection": "string or null",
    "parent_section": "string or null",
    "start_char": "int (0-indexed, inclusive offset into SEGMENT TEXT below)",
    "end_char": "int (0-indexed, exclusive offset into SEGMENT TEXT below)",
    "start_anchor": "string (first ~60 characters, copied EXACTLY)",
    "end_anchor": "string (last ~60 characters, copied EXACTLY)",
    "chunk_type": "string (narrative_story|financial_table|philosophy|business_analysis|administrative)",
    "has_financials": "bool",
    "has_table": "bool",
    "has_quote": "bool",
    "contains_principle": "bool",
    "contains_example": "bool",
    "contains_comparison": "bool",
    "contextual_summary": "string (2-3 sentences)",
    "prev_context": "string",
    "next_context": "string",
    "topics": ["array"],
    "companies_mentioned": ["array"],
    "people_mentioned": ["array"],
    "metrics_discussed": ["array"],
    "industries": ["array"],
    "principle_category": "string or null",
    "principle_statement": "string or null",
    "retrieval_priority": "string (high|medium|low)",
    "abstraction_level": "string (high|medium|low)",
    "time_sensitivity": "string (high|low)",
    "is_complete_thought": "bool",
    "needs_context": "bool"
  }}
]
```

IMPORTANT:
- Anchors MUST be exact text from SEGMENT TEXT below.
- Offsets are relative to SEGMENT TEXT START (position 0).

SEGMENT TEXT START
{segment.text}
SEGMENT TEXT END
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

            # Parse response
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "chunks" in data:
                    chunks = data["chunks"]
                elif isinstance(data, list):
                    chunks = data
                else:
                    raise ValueError("Unexpected JSON structure")
                if isinstance(chunks, list):
                    return chunks
            except Exception:
                pass

            # Trim parse fallback
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
                        return data["chunks"]
                    elif isinstance(data, list):
                        return data
            except Exception:
                pass

            # Salvage objects
            salvaged = extract_chunk_objects_from_response(content)
            if salvaged:
                print(f"[WARN] JSON parse failed for segment {segment.segment_index + 1}. Salvaged {len(salvaged)} chunks.")
                return salvaged
            
            raise RuntimeError("LLM returned unusable JSON and no chunks could be salvaged.")

        except Exception as exc:
            last_exc = exc
            if attempt >= retry.max_attempts or not _is_retryable_openai_error(exc):
                break

            sleep_s = min(
                retry.max_sleep_s,
                retry.base_sleep_s * (2 ** (attempt - 1)) + random.random() * retry.jitter_s,
            )
            print(f"[WARN] LLM call failed for segment {segment.segment_index + 1} (attempt {attempt}): {exc}")
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM call failed for segment {segment.segment_index + 1}: {last_exc}")


def call_llm_for_chunks_batched(
    letter_text: str,
    year: int,
    source_file: str,
    chunking_spec: str,
    model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main entry point: segments document and processes each segment.
    Returns merged chunks with global offsets.
    """
    segments = segment_document(letter_text)
    
    batch_diagnostics = {
        "total_segments": len(segments),
        "segment_sizes": [len(s.text) for s in segments],
    }
    
    print(f"[INFO] Document split into {len(segments)} segments")
    for i, seg in enumerate(segments):
        print(f"  Segment {i+1}: {len(seg.text):,} chars (offset {seg.global_start_offset:,})")
    
    all_chunks: List[Dict[str, Any]] = []
    raw_responses: List[str] = []
    
    for seg in segments:
        print(f"[INFO] Processing segment {seg.segment_index + 1}/{seg.total_segments}...")
        
        segment_chunks = call_llm_for_segment(
            segment=seg,
            year=year,
            source_file=source_file,
            chunking_spec=chunking_spec,
            model=model,
        )
        
        # Adjust offsets from segment-local to global
        for chunk in segment_chunks:
            if isinstance(chunk.get("start_char"), int):
                chunk["start_char"] += seg.global_start_offset
            if isinstance(chunk.get("end_char"), int):
                chunk["end_char"] += seg.global_start_offset
            chunk["_segment_index"] = seg.segment_index  # For debugging
        
        all_chunks.extend(segment_chunks)
        print(f"  -> Got {len(segment_chunks)} chunks from segment {seg.segment_index + 1}")
    
    # Sort by start position
    all_chunks.sort(key=lambda x: (x.get("start_char", 10**18), x.get("end_char", 10**18)))
    
    # Renumber chunk_ids for consistency
    section_counters: Dict[str, int] = {}
    for chunk in all_chunks:
        section = chunk.get("section_type", "other")
        section_counters[section] = section_counters.get(section, 0) + 1
        chunk["chunk_id"] = f"{year}_{section}_{section_counters[section]:03d}"
    
    batch_diagnostics["total_chunks"] = len(all_chunks)
    
    return all_chunks, batch_diagnostics


# ------------------------- Offset anchoring / repair -------------------------
# (Keep all the existing repair functions from v4 - they work perfectly)

def _validate_span_with_anchors(
    letter_text: str,
    start: int,
    end: int,
    start_anchor: Optional[str],
    end_anchor: Optional[str],
) -> bool:
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
    if not start_anchor or not end_anchor:
        return None
    sa = start_anchor.strip()
    ea = end_anchor.strip()
    if len(sa) < 8 or len(ea) < 8:
        return None
    start_positions = _find_all(letter_text, sa)
    if not start_positions or len(start_positions) > max_candidates:
        start_positions = start_positions[:max_candidates] if start_positions else []
    if not start_positions:
        return None
    best: Optional[Tuple[int, int]] = None
    for s in start_positions:
        search_from = s + max(1, len(sa) // 2)
        e = letter_text.find(ea, search_from)
        if e == -1:
            continue
        end_excl = e + len(ea)
        if end_excl <= s or end_excl - s > max_span:
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
    sa_norm = _norm_ws(start_anchor)
    ea_norm = _norm_ws(end_anchor)
    if len(sa_norm) < 8 or len(ea_norm) < 8:
        return None
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

        if cs > 0 and cs < n and letter_text[cs - 1].isalnum() and letter_text[cs].isalnum():
            boundary = cs
            while boundary > 0 and letter_text[boundary - 1].isalnum() and letter_text[boundary].isalnum():
                boundary -= 1
            if boundary != cs:
                prev["end_char"] = boundary
                cur["start_char"] = boundary
                midword_fixes += 1
                continue

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

    for c in chunks:
        if not isinstance(c, dict):
            continue
        c["year"] = year
        c["source_file"] = source_file
        c.pop("position_in_letter", None)
        c.pop("position_in_section", None)
        c.pop("total_chunks_in_section", None)
        c.pop("_segment_index", None)  # Remove debug field
        ct = c.get("chunk_type")
        if not isinstance(ct, str) or ct not in ALLOWED_CHUNK_TYPES:
            c["chunk_type"] = "administrative"

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

    chunks = resolve_offsets_with_anchors(chunks, letter_text, diagnostics=diagnostics)
    chunks = smooth_boundaries(chunks, letter_text, diagnostics=diagnostics)
    chunks = reconstruct_chunk_text(letter_text, chunks)

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
    chunks = reconstruct_chunk_text(letter_text, chunks)

    for c in chunks:
        txt = c.get("chunk_text") or ""
        c["char_count"] = len(txt)
        c["word_count"] = len(txt.split()) if txt else 0
        s = c.get("start_char")
        if isinstance(s, int):
            pos = s / n
            c["position_in_letter"] = float(_clamp_int(int(pos * 1_000_000), 0, 1_000_000) / 1_000_000)

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


# ------------------------- Validation -------------------------

def _validate_output_integrity(letter_text: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(letter_text)
    total = len(chunks)

    bad_offsets = 0
    empty_text = 0
    midword_starts = 0
    unsafe_starts = 0
    overlaps = 0
    gaps = 0

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


# ------------------------- Batch runner -------------------------

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

    letter_text = load_letter_text(year)

    if dry_run:
        segments = segment_document(letter_text)
        return {
            "year": year,
            "status": "dry_run_ok",
            "input_chars": len(letter_text),
            "segments": len(segments),
            "segment_sizes": [len(s.text) for s in segments],
            "would_write": [str(out_jsonl), str(out_diag)],
        }

    # Use the new batched approach
    chunks_raw, batch_diag = call_llm_for_chunks_batched(
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

    chunks = reconstruct_chunk_text(letter_text, chunks)

    integrity = _validate_output_integrity(letter_text, chunks)
    diagnostics["integrity"] = integrity
    diagnostics["batch_processing"] = batch_diag

    out_path = write_chunks_jsonl(chunks, year)
    diag_path = write_diagnostics_json(diagnostics, year)

    return {
        "year": year,
        "status": "ok",
        "chunks": len(chunks),
        "segments_processed": batch_diag.get("total_segments", 1),
        "output": str(out_path),
        "diagnostics": str(diag_path),
        "integrity": integrity,
    }


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


def write_batch_summary(summary: Dict[str, Any]) -> Path:
    out_path = OUT_DIR / "batch_run_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ------------------------- CLI -------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch chunk Berkshire letters with OpenAI + batched segment processing.")
    p.add_argument("years", nargs="*", help="Years to process (e.g., 2008 2009).")
    p.add_argument("--years", dest="years_csv", default="", help="Comma-separated years (e.g., 2008,2009).")
    p.add_argument("--all", action="store_true", help="Process all *_cleaned.txt found.")
    p.add_argument("--force", action="store_true", help="Re-run even if output exists.")
    p.add_argument("--dry-run", action="store_true", help="Do not call LLM; validate segmentation only.")
    p.add_argument("--min-words-to-keep", type=int, default=60, help="Merge small chunks into next.")
    p.add_argument("--max-segment-chars", type=int, default=MAX_SEGMENT_CHARS,
                   help=f"Max chars per segment (default: {MAX_SEGMENT_CHARS})")
    return p


def main(argv: List[str]) -> int:
    args = build_arg_parser().parse_args(argv[1:])

    global MAX_SEGMENT_CHARS
    if args.max_segment_chars:
        MAX_SEGMENT_CHARS = args.max_segment_chars

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
    print(f"[INFO] Max segment chars={MAX_SEGMENT_CHARS}")
    print(f"[INFO] OUT_DIR={OUT_DIR}")

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
                    segs = r.get("segments_processed", 1)
                    print(f"[OK] {year}: chunks={r.get('chunks')} segments={segs} integrity={integ}")
                else:
                    segs = r.get("segments", 1)
                    print(f"[OK] {year}: dry-run input_chars={r.get('input_chars')} segments={segs}")
            elif status == "skipped":
                skipped += 1
                print(f"[SKIP] {year}: {r.get('reason')} existing_chunks={r.get('existing_chunks')}")
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
        "max_segment_chars": MAX_SEGMENT_CHARS,
        "out_dir": str(OUT_DIR),
        "years_requested": years,
        "counts": {"ok_or_dry_run_ok": ok, "skipped": skipped, "failed": failed, "total": len(years)},
        "results": results,
    }

    summary_path = write_batch_summary(summary)
    print(f"\n[INFO] Batch summary written: {summary_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))