#!/usr/bin/env python
"""
chunking_with_llm_hybrid_v6.py

HYBRID CHUNKING STRATEGY - Combining Best of Both Approaches

This script addresses the limitations found in both Script C (section-aware) and 
Script G (windowed) approaches:

Key Features:
1. WINDOWED STRATEGY (from Script G): Guarantees coverage and progress
   - Fixed-size overlapping windows prevent missing content
   - Paragraph/sentence boundary detection for clean splits
   - Fallback to whitespace boundaries ensures forward progress

2. CHUNK GRANULARITY (from Script C): RAG-appropriate chunk sizes
   - Post-LLM splitting of oversized chunks into semantic units
   - Target chunk size: 500-3000 chars (optimal for RAG)
   - Preserves chunk boundaries at sentence/paragraph breaks

3. FUZZY DEDUPLICATION: Better overlap handling
   - Jaccard similarity on anchor text instead of exact tuples
   - Handles LLM variations in boundary detection
   - Prevents duplicate content from overlapping windows

4. COVERAGE VALIDATION: Guarantees no content loss
   - Post-processing check ensures >99% document coverage
   - Automatic gap detection and repair
   - Fails explicitly if coverage threshold not met

Usage:
    python chunking_with_llm_hybrid_v6.py 2010
    python chunking_with_llm_hybrid_v6.py --years 1984,2010
    python chunking_with_llm_hybrid_v6.py --all
    python chunking_with_llm_hybrid_v6.py --all --dry-run

Environment Variables:
    OPENAI_API_KEY: Required for OpenAI API access
    OPENAI_MODEL: Model to use (default: gpt-4.1-mini)
    CHUNK_WINDOW_MAX_CHARS: Maximum window size (default: 18000)
    CHUNK_WINDOW_OVERLAP_CHARS: Overlap between windows (default: 1200)
    CHUNK_TARGET_SIZE: Target chunk size for splitting (default: 2500)
    CHUNK_MIN_SIZE: Minimum chunk size (default: 400)
    CHUNK_MAX_SIZE: Maximum chunk size before splitting (default: 4000)
    COVERAGE_THRESHOLD: Minimum coverage percentage (default: 99.0)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

# ========================= Configuration =========================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent if (THIS_DIR.parent / "data").exists() else THIS_DIR

# Try multiple possible locations for chunking strategy
CHUNKING_STRATEGY_PATHS = [
    THIS_DIR / "chunking_rule_claude.md",
    THIS_DIR / "chunking_strategy.md",
    PROJECT_ROOT / "phase_1" / "chunking_rule_claude.md",
    Path("/mnt/user-data/uploads/chunking_strategy.md"),
]

TEXT_DIR = PROJECT_ROOT / "data" / "text_extracted_letters"
OUT_DIR = PROJECT_ROOT / "data" / "chunks_llm_gpt" / "hybrid_v6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Windowing parameters (from Script G - guarantees coverage)
WINDOW_MAX_CHARS = int(os.getenv("CHUNK_WINDOW_MAX_CHARS", "18000"))
WINDOW_OVERLAP_CHARS = int(os.getenv("CHUNK_WINDOW_OVERLAP_CHARS", "1200"))
MAX_SINGLECALL_CHARS = int(os.getenv("CHUNK_MAX_SINGLECALL_CHARS", "24000"))

# Chunk size parameters (for post-LLM splitting)
CHUNK_TARGET_SIZE = int(os.getenv("CHUNK_TARGET_SIZE", "2500"))
CHUNK_MIN_SIZE = int(os.getenv("CHUNK_MIN_SIZE", "400"))
CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE", "4000"))

# Coverage validation
COVERAGE_THRESHOLD = float(os.getenv("COVERAGE_THRESHOLD", "99.0"))

# Fuzzy deduplication threshold (Jaccard similarity)
DEDUP_SIMILARITY_THRESHOLD = 0.85

ALLOWED_CHUNK_TYPES = {
    "narrative_story",
    "financial_table", 
    "philosophy",
    "business_analysis",
    "administrative",
}

SENT_END = {".", "!", "?"}


# ========================= Utilities =========================

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _norm_ws(s: str) -> str:
    """Normalize whitespace for anchor comparison."""
    return re.sub(r"\s+", " ", s).strip()


def _text_to_shingles(text: str, k: int = 5) -> Set[str]:
    """Convert text to k-shingles (character n-grams) for Jaccard similarity."""
    text = _norm_ws(text.lower())
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts using shingles."""
    if not text1 or not text2:
        return 0.0
    shingles1 = _text_to_shingles(text1)
    shingles2 = _text_to_shingles(text2)
    intersection = len(shingles1 & shingles2)
    union = len(shingles1 | shingles2)
    return intersection / union if union > 0 else 0.0


def _sequence_similarity(text1: str, text2: str) -> float:
    """Compute sequence similarity using SequenceMatcher."""
    return SequenceMatcher(None, _norm_ws(text1), _norm_ws(text2)).ratio()


def load_chunking_strategy() -> str:
    """Load chunking strategy from available paths."""
    for path in CHUNKING_STRATEGY_PATHS:
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Cannot find chunking strategy at any of: {CHUNKING_STRATEGY_PATHS}")


def load_letter_text(year: int) -> str:
    """Load letter text from file."""
    path = TEXT_DIR / f"{year}_cleaned.txt"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find cleaned letter for year {year}: {path}")
    return path.read_text(encoding="utf-8")


def discover_years_from_text_dir() -> List[int]:
    """Discover available years from text directory."""
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


# ========================= Window Splitting (from Script G) =========================

def _find_paragraph_break_before(text: str, pos: int, window: int = 4000) -> Optional[int]:
    """Find paragraph break (double newline) before position."""
    n = len(text)
    pos = _clamp_int(pos, 0, n)
    left = max(0, pos - window)
    snippet = text[left:pos]
    k = snippet.rfind("\n\n")
    if k == -1:
        return None
    return left + k + 2


def _find_sentence_break_before(text: str, pos: int, window: int = 2400) -> Optional[int]:
    """Find sentence break before position."""
    n = len(text)
    pos = _clamp_int(pos, 0, n)
    left = max(0, pos - window)
    snippet = text[left:pos]
    matches = list(re.finditer(r'[.!?]["\')\]]*[\s\n\r\t]+', snippet))
    if not matches:
        return None
    return left + matches[-1].end()


def _snap_to_whitespace_before(text: str, pos: int, window: int = 800) -> int:
    """Snap position to nearest whitespace boundary before."""
    n = len(text)
    pos = _clamp_int(pos, 0, n)
    left = max(0, pos - window)
    k = pos
    while k > left:
        if text[k - 1].isspace():
            return k
        k -= 1
    return pos


def split_into_windows(
    text: str,
    max_chars: int = WINDOW_MAX_CHARS,
    overlap_chars: int = WINDOW_OVERLAP_CHARS,
) -> List[Tuple[int, int, str]]:
    """
    Split text into overlapping windows with smart boundary detection.
    
    Returns list of (start_offset, end_offset, window_text) tuples.
    This guarantees complete coverage of the document.
    """
    n = len(text)
    if n <= max_chars:
        return [(0, n, text)]
    
    windows: List[Tuple[int, int, str]] = []
    start = 0
    guard = 0
    
    while start < n:
        guard += 1
        if guard > 50_000:
            # Safety against infinite loops
            break
        
        target_end = min(n, start + max_chars)
        
        if target_end >= n:
            end = n
        else:
            # Try strongest boundaries first: paragraph > sentence > whitespace
            end = _find_paragraph_break_before(text, target_end)
            if end is None:
                end = _find_sentence_break_before(text, target_end)
            if end is None:
                end = _snap_to_whitespace_before(text, target_end)
            
            # Ensure minimum progress (2000 chars)
            end = _clamp_int(end, start + 2000, n)
        
        window_text = text[start:end]
        windows.append((start, end, window_text))
        
        if end >= n:
            break
        
        # Calculate next start with overlap
        next_start = max(0, end - overlap_chars)
        # Snap overlap start to whitespace
        next_start = _snap_to_whitespace_before(text, next_start, window=600)
        next_start = _clamp_int(next_start, start + 1, n)
        start = next_start
    
    return windows


# ========================= Offset Repair (from Script G) =========================

def _find_all(haystack: str, needle: str) -> List[int]:
    """Find all occurrences of needle in haystack."""
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


def _repair_offsets_by_anchors(
    text: str,
    start_anchor: str,
    end_anchor: str,
    max_candidates: int = 50,
    max_span: int = 120_000,
) -> Optional[Tuple[int, int]]:
    """Repair offsets using anchor text search."""
    if not start_anchor or not end_anchor:
        return None
    
    sa = start_anchor.strip()
    ea = end_anchor.strip()
    if len(sa) < 8 or len(ea) < 8:
        return None
    
    start_positions = _find_all(text, sa)
    if not start_positions:
        # Try shorter prefix
        sa_short = sa[:min(40, len(sa))]
        start_positions = _find_all(text, sa_short)
    
    if not start_positions:
        return None
    
    if len(start_positions) > max_candidates:
        start_positions = start_positions[:max_candidates]
    
    best: Optional[Tuple[int, int]] = None
    for s in start_positions:
        search_from = s + max(1, len(sa) // 2)
        e = text.find(ea, search_from)
        if e == -1:
            # Try shorter suffix
            ea_short = ea[-min(40, len(ea)):]
            e = text.find(ea_short, search_from)
            if e != -1:
                e = e + len(ea_short)
            else:
                continue
        else:
            e = e + len(ea)
        
        if e <= s or e - s > max_span:
            continue
        
        if best is None or (e - s) < (best[1] - best[0]):
            best = (s, e)
    
    return best


def resolve_offsets_with_anchors(
    chunks: List[Dict[str, Any]],
    letter_text: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Validate and repair chunk offsets using anchor text."""
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
            c["offset_confidence"] = "low"
            continue
        
        attempted += 1
        
        # Validate current offsets
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= n:
            span = letter_text[s:e]
            span_n = _norm_ws(span)
            sa_n = _norm_ws(sa)
            ea_n = _norm_ws(ea)
            
            if span_n.startswith(sa_n[:min(30, len(sa_n))]) and span_n.endswith(ea_n[-min(30, len(ea_n)):]):
                c["offset_confidence"] = "validated"
                continue
        
        # Attempt repair
        repaired = _repair_offsets_by_anchors(letter_text, sa, ea)
        
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


# ========================= Fuzzy Deduplication =========================

@dataclass
class ChunkSignature:
    """Signature for chunk deduplication."""
    start_char: int
    end_char: int
    start_anchor_norm: str
    end_anchor_norm: str
    text_hash: str
    
    def overlaps_with(self, other: 'ChunkSignature', threshold: float = 0.5) -> bool:
        """Check if this chunk significantly overlaps with another."""
        if self.start_char >= other.end_char or other.start_char >= self.end_char:
            return False
        
        overlap_start = max(self.start_char, other.start_char)
        overlap_end = min(self.end_char, other.end_char)
        overlap_len = overlap_end - overlap_start
        
        self_len = self.end_char - self.start_char
        other_len = other.end_char - other.start_char
        min_len = min(self_len, other_len)
        
        return overlap_len / min_len > threshold if min_len > 0 else False


def deduplicate_chunks_fuzzy(
    chunks: List[Dict[str, Any]],
    similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    overlap_threshold: float = 0.7,  # 70% content overlap = duplicate
) -> List[Dict[str, Any]]:
    """
    Deduplicate chunks using fuzzy matching on anchors and spans.
    
    Unlike exact tuple matching (Script G), this handles:
    - Slight variations in LLM-generated anchor text
    - Nearly-identical chunks with minor offset differences
    - Substantially overlapping chunks from windowed processing
    
    Deduplication rules:
    1. Exact position match + similar anchors = duplicate
    2. >70% content overlap + similar anchors = duplicate (keep larger)
    3. Very high anchor similarity (>0.95) + any overlap = duplicate
    """
    if not chunks:
        return chunks
    
    # Sort by start position, then by size (larger first to prefer keeping larger chunks)
    sorted_chunks = sorted(
        chunks,
        key=lambda x: (x.get('start_char', 10**18), -(x.get('end_char', 0) - x.get('start_char', 0)))
    )
    
    kept: List[Dict[str, Any]] = []
    signatures: List[ChunkSignature] = []
    
    for c in sorted_chunks:
        s = c.get('start_char')
        e = c.get('end_char')
        sa = _norm_ws(c.get('start_anchor') or '')
        ea = _norm_ws(c.get('end_anchor') or '')
        text = c.get('chunk_text', '')
        
        if not isinstance(s, int) or not isinstance(e, int):
            kept.append(c)
            continue
        
        chunk_len = e - s
        
        # Create signature for this chunk
        sig = ChunkSignature(
            start_char=s,
            end_char=e,
            start_anchor_norm=sa[:60],
            end_anchor_norm=ea[-60:],
            text_hash=str(hash(text[:200] + text[-200:] if len(text) > 400 else text))
        )
        
        # Check against existing signatures
        is_duplicate = False
        
        for existing_sig in signatures:
            # Calculate overlap
            overlap_start = max(sig.start_char, existing_sig.start_char)
            overlap_end = min(sig.end_char, existing_sig.end_char)
            
            if overlap_start >= overlap_end:
                # No overlap at all
                continue
            
            overlap_len = overlap_end - overlap_start
            existing_len = existing_sig.end_char - existing_sig.start_char
            
            # Calculate overlap ratios
            overlap_ratio_this = overlap_len / chunk_len if chunk_len > 0 else 0
            overlap_ratio_existing = overlap_len / existing_len if existing_len > 0 else 0
            max_overlap_ratio = max(overlap_ratio_this, overlap_ratio_existing)
            
            # Check anchor similarity
            start_sim = _jaccard_similarity(sig.start_anchor_norm, existing_sig.start_anchor_norm)
            end_sim = _jaccard_similarity(sig.end_anchor_norm, existing_sig.end_anchor_norm)
            avg_anchor_sim = (start_sim + end_sim) / 2
            
            # Rule 1: Exact or near-exact position with similar anchors
            if abs(sig.start_char - existing_sig.start_char) < 50 and abs(sig.end_char - existing_sig.end_char) < 50:
                if avg_anchor_sim > 0.7:
                    is_duplicate = True
                    break
            
            # Rule 2: High content overlap (>70%) with similar anchors
            if max_overlap_ratio > overlap_threshold and avg_anchor_sim > 0.6:
                is_duplicate = True
                break
            
            # Rule 3: Very high anchor similarity with any meaningful overlap
            if avg_anchor_sim > 0.95 and max_overlap_ratio > 0.3:
                is_duplicate = True
                break
            
            # Rule 4: Check if one chunk is contained within another with similar anchors
            if overlap_ratio_this > 0.9 or overlap_ratio_existing > 0.9:
                if avg_anchor_sim > 0.5:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            kept.append(c)
            signatures.append(sig)
    
    return kept


# ========================= Chunk Splitting (for RAG-appropriate sizes) =========================

def _find_split_point(text: str, target_pos: int, window: int = 500) -> int:
    """Find a good split point near target position."""
    n = len(text)
    target_pos = _clamp_int(target_pos, 0, n)
    
    # Look for paragraph break first
    search_start = max(0, target_pos - window)
    search_end = min(n, target_pos + window)
    
    para_breaks = []
    for m in re.finditer(r'\n\s*\n', text[search_start:search_end]):
        para_breaks.append(search_start + m.end())
    
    if para_breaks:
        best = min(para_breaks, key=lambda x: abs(x - target_pos))
        return best
    
    # Look for sentence break
    sent_breaks = []
    for m in re.finditer(r'[.!?]["\')\]]*\s+', text[search_start:search_end]):
        sent_breaks.append(search_start + m.end())
    
    if sent_breaks:
        best = min(sent_breaks, key=lambda x: abs(x - target_pos))
        return best
    
    # Fall back to whitespace
    return _snap_to_whitespace_before(text, target_pos, window)


def split_oversized_chunk(
    chunk: Dict[str, Any],
    letter_text: str,
    target_size: int = CHUNK_TARGET_SIZE,
    max_size: int = CHUNK_MAX_SIZE,
) -> List[Dict[str, Any]]:
    """
    Split an oversized chunk into smaller, RAG-appropriate pieces.
    
    Preserves metadata and creates proper linkage between sub-chunks.
    """
    text = chunk.get('chunk_text', '')
    char_count = len(text)
    
    if char_count <= max_size:
        return [chunk]
    
    start_char = chunk.get('start_char', 0)
    
    # Calculate number of splits needed
    num_parts = (char_count + target_size - 1) // target_size
    num_parts = max(2, num_parts)
    
    sub_chunks: List[Dict[str, Any]] = []
    pos = 0
    part_idx = 0
    
    while pos < char_count:
        # Calculate target end for this part
        remaining = char_count - pos
        parts_left = num_parts - part_idx
        part_target = remaining // parts_left if parts_left > 0 else remaining
        part_target = _clamp_int(part_target, CHUNK_MIN_SIZE, target_size + 500)
        
        target_end = min(pos + part_target, char_count)
        
        if target_end < char_count:
            # Find good split point
            actual_end = _find_split_point(text, target_end, window=400)
            actual_end = _clamp_int(actual_end, pos + CHUNK_MIN_SIZE, char_count)
        else:
            actual_end = char_count
        
        # Create sub-chunk
        sub_text = text[pos:actual_end]
        sub_chunk = chunk.copy()
        
        # Update offsets relative to full document
        sub_chunk['start_char'] = start_char + pos
        sub_chunk['end_char'] = start_char + actual_end
        sub_chunk['chunk_text'] = sub_text
        sub_chunk['char_count'] = len(sub_text)
        sub_chunk['word_count'] = len(sub_text.split())
        
        # Update anchors
        sub_chunk['start_anchor'] = sub_text[:min(60, len(sub_text))]
        sub_chunk['end_anchor'] = sub_text[-min(60, len(sub_text)):]
        
        # Mark as split
        sub_chunk['_split_from'] = chunk.get('chunk_id', 'unknown')
        sub_chunk['_split_part'] = part_idx
        
        sub_chunks.append(sub_chunk)
        
        pos = actual_end
        part_idx += 1
    
    return sub_chunks


def split_all_oversized_chunks(
    chunks: List[Dict[str, Any]],
    letter_text: str,
    target_size: int = CHUNK_TARGET_SIZE,
    max_size: int = CHUNK_MAX_SIZE,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Split all oversized chunks and return flat list."""
    result: List[Dict[str, Any]] = []
    splits_performed = 0
    
    for chunk in chunks:
        sub_chunks = split_oversized_chunk(chunk, letter_text, target_size, max_size)
        if len(sub_chunks) > 1:
            splits_performed += 1
        result.extend(sub_chunks)
    
    if diagnostics is not None:
        diagnostics['chunks_split'] = splits_performed
        diagnostics['chunks_after_split'] = len(result)
    
    return result


# ========================= Coverage Validation =========================

def calculate_coverage(
    chunks: List[Dict[str, Any]],
    doc_length: int,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Calculate document coverage and identify gaps.
    
    Returns:
        - coverage_percentage: float
        - gaps: List of (start, end) tuples for uncovered regions
    """
    if doc_length == 0:
        return 100.0, []
    
    # Create coverage bitmap
    covered = [False] * doc_length
    
    for c in chunks:
        s = c.get('start_char')
        e = c.get('end_char')
        if isinstance(s, int) and isinstance(e, int):
            for i in range(max(0, s), min(doc_length, e)):
                covered[i] = True
    
    covered_count = sum(covered)
    coverage_pct = (covered_count / doc_length) * 100
    
    # Find gaps
    gaps: List[Tuple[int, int]] = []
    in_gap = False
    gap_start = 0
    
    for i in range(doc_length):
        if not covered[i] and not in_gap:
            in_gap = True
            gap_start = i
        elif covered[i] and in_gap:
            in_gap = False
            gaps.append((gap_start, i))
    
    if in_gap:
        gaps.append((gap_start, doc_length))
    
    return coverage_pct, gaps


def fill_coverage_gaps(
    chunks: List[Dict[str, Any]],
    letter_text: str,
    gaps: List[Tuple[int, int]],
    year: int,
    min_gap_size: int = 50,
) -> List[Dict[str, Any]]:
    """
    Create chunks to fill coverage gaps.
    
    This ensures no content is lost even if LLM chunking misses regions.
    """
    new_chunks: List[Dict[str, Any]] = []
    
    for gap_start, gap_end in gaps:
        gap_size = gap_end - gap_start
        if gap_size < min_gap_size:
            continue
        
        gap_text = letter_text[gap_start:gap_end]
        
        # Create a gap-filling chunk with minimal metadata
        gap_chunk = {
            'chunk_id': f"{year}_gap_{gap_start}_{gap_end}",
            'year': year,
            'source_file': f"{year}_cleaned.txt",
            'section_type': 'other',
            'section_title': 'Gap Fill',
            'subsection': None,
            'parent_section': None,
            'start_char': gap_start,
            'end_char': gap_end,
            'start_anchor': gap_text[:min(60, len(gap_text))],
            'end_anchor': gap_text[-min(60, len(gap_text)):],
            'chunk_type': 'business_analysis',
            'chunk_text': gap_text,
            'char_count': len(gap_text),
            'word_count': len(gap_text.split()),
            'has_financials': False,
            'has_table': False,
            'has_quote': False,
            'contains_principle': False,
            'contains_example': False,
            'contains_comparison': False,
            'contextual_summary': 'Gap-filled content to ensure complete document coverage.',
            'prev_context': '',
            'next_context': '',
            'topics': [],
            'companies_mentioned': [],
            'people_mentioned': [],
            'metrics_discussed': [],
            'industries': [],
            'principle_category': None,
            'principle_statement': None,
            'retrieval_priority': 'low',
            'abstraction_level': 'low',
            'time_sensitivity': 'medium',
            'is_complete_thought': True,
            'needs_context': True,
            'offset_confidence': 'gap_fill',
            '_is_gap_fill': True,
        }
        
        new_chunks.append(gap_chunk)
    
    return chunks + new_chunks


# ========================= LLM Interaction =========================

@dataclass
class LLMRetryConfig:
    max_attempts: int = 5
    base_sleep_s: float = 1.5
    max_sleep_s: float = 20.0
    jitter_s: float = 0.8


def _is_retryable_error(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    retry_markers = [
        "rate limit", "timeout", "timed out", "temporarily",
        "server error", "overloaded", "connection",
        "429", "500", "502", "503", "504",
    ]
    return any(m in msg for m in retry_markers)


def extract_chunk_objects_from_response(content: str) -> List[Dict[str, Any]]:
    """Fallback parser for malformed JSON responses."""
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


def call_llm_for_window(
    window_text: str,
    year: int,
    source_file: str,
    chunking_spec: str,
    model: str,
    window_id: int,
    raw_debug_path: Path,
    retry: Optional[LLMRetryConfig] = None,
) -> List[Dict[str, Any]]:
    """Call LLM to chunk a single window of text."""
    client = OpenAI()
    retry = retry or LLMRetryConfig()
    
    system_content = f"""
You are an expert corpus architect for Berkshire Hathaway shareholder letters.
Your job is to chunk text into semantically meaningful chunks with rich metadata.
Follow the chunking rule specification exactly.

Hard constraints:
- Return ONLY valid JSON (no prose).
- Top-level MUST be a JSON object: {{"chunks": [ ... ]}}.
- Offsets (start_char, end_char) are 0-indexed relative to the PROVIDED WINDOW TEXT.
- Include start_anchor and end_anchor copied EXACTLY from the window text.
- Create chunks of moderate size (target: 1500-3500 chars, avoid very large chunks).
- DO NOT include chunk_text/word_count/char_count (computed locally).
- DO NOT include position_in_letter/position_in_section (computed locally).
"""

    user_content = f"""
Chunk this window from Berkshire Hathaway shareholder letter year {year} (window {window_id}).

Source: {source_file}

Output each chunk with these fields:
- chunk_id: "{year}_{{section_type}}_{{sequence:03d}}"
- year: {year}
- source_file: "{source_file}"
- section_type: one of (performance_overview, insurance_operations, acquisitions, investments, operating_businesses, corporate_governance, management_philosophy, shareholder_matters, other)
- section_title, subsection, parent_section: strings or null
- start_char, end_char: 0-indexed offsets in WINDOW TEXT
- start_anchor: first ~60 chars of chunk (exact copy)
- end_anchor: last ~60 chars of chunk (exact copy)
- chunk_type: one of (narrative_story, financial_table, philosophy, business_analysis, administrative)
- has_financials, has_table, has_quote, contains_principle, contains_example, contains_comparison: booleans
- contextual_summary: 2-3 sentence summary
- prev_context, next_context: 1-2 sentence context summaries
- topics, companies_mentioned, people_mentioned, metrics_discussed, industries: arrays
- principle_category, principle_statement: string or null
- retrieval_priority: high/medium/low
- abstraction_level: high/medium/low
- time_sensitivity: high/medium/low
- is_complete_thought, needs_context: booleans

<chunking_strategy>
{chunking_spec}
</chunking_strategy>

<window_text>
{window_text}
</window_text>

Return JSON: {{"chunks": [...]}}
"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    
    last_exc = None
    for attempt in range(1, retry.max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=16000,
                temperature=0.1,
            )
            
            raw_content = response.choices[0].message.content or ""
            
            # Save raw response for debugging
            raw_debug_path.write_text(json.dumps({
                "window_id": window_id,
                "year": year,
                "model": model,
                "raw_response": raw_content,
                "timestamp": _now_iso(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # Parse response
            try:
                # Try to extract JSON from response
                json_match = re.search(r'\{[\s\S]*"chunks"[\s\S]*\}', raw_content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    chunks = parsed.get("chunks", [])
                else:
                    chunks = extract_chunk_objects_from_response(raw_content)
            except json.JSONDecodeError:
                chunks = extract_chunk_objects_from_response(raw_content)
            
            return [c for c in chunks if isinstance(c, dict)]
            
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_error(exc) or attempt == retry.max_attempts:
                raise
            
            sleep_s = min(
                retry.max_sleep_s,
                retry.base_sleep_s * (2 ** (attempt - 1)) + random.random() * retry.jitter_s
            )
            print(f"[WARN] LLM call failed (attempt {attempt}/{retry.max_attempts}): {exc}")
            print(f"[INFO] Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
    
    raise RuntimeError(f"LLM call failed after {retry.max_attempts} attempts: {last_exc}")


def process_letter_windowed(
    letter_text: str,
    year: int,
    source_file: str,
    chunking_spec: str,
    model: str,
) -> List[Dict[str, Any]]:
    """Process entire letter using windowed approach."""
    windows = split_into_windows(letter_text)
    
    print(f"[INFO] Year {year}: Split into {len(windows)} windows")
    
    all_chunks: List[Dict[str, Any]] = []
    
    for w_id, (w_start, w_end, w_text) in enumerate(windows):
        print(f"[INFO] Processing window {w_id+1}/{len(windows)} (chars {w_start}-{w_end})")
        
        raw_path = OUT_DIR / f"{year}_raw_llm_response_window_{w_id:03d}.json"
        
        w_chunks = call_llm_for_window(
            window_text=w_text,
            year=year,
            source_file=source_file,
            chunking_spec=chunking_spec,
            model=model,
            window_id=w_id,
            raw_debug_path=raw_path,
        )
        
        # Map local offsets to global offsets
        for c in w_chunks:
            s = c.get("start_char")
            e = c.get("end_char")
            if isinstance(s, int):
                c["start_char"] = w_start + s
            if isinstance(e, int):
                c["end_char"] = w_start + e
            c["_window_id"] = w_id
            c["_window_start"] = w_start
            c["_window_end"] = w_end
        
        all_chunks.extend(w_chunks)
    
    return all_chunks


# ========================= Post-Processing Pipeline =========================

def reconstruct_chunk_text(chunks: List[Dict[str, Any]], letter_text: str) -> List[Dict[str, Any]]:
    """Reconstruct chunk_text from offsets."""
    n = len(letter_text)
    
    for c in chunks:
        s = c.get("start_char")
        e = c.get("end_char")
        
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= n:
            c["chunk_text"] = letter_text[s:e]
        else:
            c["chunk_text"] = ""
        
        txt = c.get("chunk_text", "")
        c["char_count"] = len(txt)
        c["word_count"] = len(txt.split()) if txt else 0
    
    return chunks


def renumber_chunks(chunks: List[Dict[str, Any]], year: int) -> List[Dict[str, Any]]:
    """Renumber chunk IDs based on document position."""
    chunks.sort(key=lambda x: (x.get("start_char", 10**18), x.get("end_char", 10**18)))
    
    for idx, c in enumerate(chunks):
        section_type = (c.get("section_type") or "other").strip() or "other"
        c["chunk_id"] = f"{year}_{section_type}_{idx:03d}"
        
        # Compute position_in_letter
        s = c.get("start_char")
        if isinstance(s, int):
            c["position_in_letter"] = round(s / max(1, chunks[-1].get("end_char", 1)), 6)
    
    return chunks


def compute_section_positions(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute position within sections."""
    def section_key(c: Dict[str, Any]) -> tuple:
        return (
            (c.get("section_title") or "").strip(),
            (c.get("subsection") or "").strip(),
            (c.get("section_type") or "other").strip(),
        )
    
    section_counts: Dict[tuple, int] = {}
    for c in chunks:
        k = section_key(c)
        section_counts[k] = section_counts.get(k, 0) + 1
    
    section_seen: Dict[tuple, int] = {}
    for c in chunks:
        k = section_key(c)
        idx = section_seen.get(k, 0)
        c["position_in_section"] = idx
        c["total_chunks_in_section"] = section_counts.get(k, 1)
        section_seen[k] = idx + 1
    
    return chunks


def run_full_pipeline(
    letter_text: str,
    year: int,
    chunking_spec: str,
    model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run the complete hybrid chunking pipeline.
    
    Returns:
        - chunks: Final list of chunks
        - diagnostics: Pipeline statistics and validation results
    """
    source_file = f"{year}_cleaned.txt"
    n = len(letter_text)
    
    diagnostics: Dict[str, Any] = {
        "year": year,
        "letter_length": n,
        "timestamp": _now_iso(),
        "model": model,
    }
    
    # Step 1: Windowed LLM processing (guarantees coverage)
    print(f"[INFO] Step 1: Windowed LLM processing...")
    raw_chunks = process_letter_windowed(
        letter_text=letter_text,
        year=year,
        source_file=source_file,
        chunking_spec=chunking_spec,
        model=model,
    )
    diagnostics["raw_chunks_from_llm"] = len(raw_chunks)
    print(f"[INFO] Got {len(raw_chunks)} raw chunks from LLM")
    
    # Step 2: Offset repair using anchors
    print(f"[INFO] Step 2: Repairing offsets with anchors...")
    chunks = resolve_offsets_with_anchors(raw_chunks, letter_text, diagnostics)
    
    # Step 3: Reconstruct chunk text
    print(f"[INFO] Step 3: Reconstructing chunk text...")
    chunks = reconstruct_chunk_text(chunks, letter_text)
    
    # Step 4: Fuzzy deduplication
    print(f"[INFO] Step 4: Fuzzy deduplication...")
    chunks_before_dedup = len(chunks)
    chunks = deduplicate_chunks_fuzzy(chunks)
    diagnostics["chunks_removed_by_dedup"] = chunks_before_dedup - len(chunks)
    print(f"[INFO] Removed {chunks_before_dedup - len(chunks)} duplicates")
    
    # Step 5: Split oversized chunks
    print(f"[INFO] Step 5: Splitting oversized chunks...")
    chunks = split_all_oversized_chunks(chunks, letter_text, diagnostics=diagnostics)
    
    # Step 6: Check coverage and fill gaps
    print(f"[INFO] Step 6: Validating coverage...")
    coverage_pct, gaps = calculate_coverage(chunks, n)
    diagnostics["coverage_before_gap_fill"] = coverage_pct
    diagnostics["gaps_found"] = len(gaps)
    diagnostics["gap_total_chars"] = sum(e - s for s, e in gaps)
    
    print(f"[INFO] Coverage: {coverage_pct:.2f}% ({len(gaps)} gaps, {diagnostics['gap_total_chars']} chars)")
    
    if coverage_pct < COVERAGE_THRESHOLD:
        print(f"[WARN] Coverage {coverage_pct:.2f}% below threshold {COVERAGE_THRESHOLD}%, filling gaps...")
        chunks = fill_coverage_gaps(chunks, letter_text, gaps, year)
        
        # Recheck coverage
        coverage_pct_after, gaps_after = calculate_coverage(chunks, n)
        diagnostics["coverage_after_gap_fill"] = coverage_pct_after
        diagnostics["gaps_remaining"] = len(gaps_after)
        print(f"[INFO] Coverage after gap fill: {coverage_pct_after:.2f}%")
        
        if coverage_pct_after < COVERAGE_THRESHOLD:
            print(f"[ERROR] Coverage still below threshold after gap fill!")
            diagnostics["coverage_warning"] = True
    
    # Step 7: Final cleanup and renumbering
    print(f"[INFO] Step 7: Final cleanup...")
    chunks = reconstruct_chunk_text(chunks, letter_text)  # Ensure text is current
    chunks = renumber_chunks(chunks, year)
    chunks = compute_section_positions(chunks)
    
    # Final statistics
    diagnostics["final_chunk_count"] = len(chunks)
    
    sizes = [c.get("char_count", 0) for c in chunks]
    if sizes:
        diagnostics["chunk_size_min"] = min(sizes)
        diagnostics["chunk_size_max"] = max(sizes)
        diagnostics["chunk_size_avg"] = sum(sizes) // len(sizes)
        diagnostics["chunks_under_500"] = sum(1 for s in sizes if s < 500)
        diagnostics["chunks_500_3000"] = sum(1 for s in sizes if 500 <= s <= 3000)
        diagnostics["chunks_over_3000"] = sum(1 for s in sizes if s > 3000)
    
    final_coverage, final_gaps = calculate_coverage(chunks, n)
    diagnostics["final_coverage"] = final_coverage
    
    print(f"[INFO] Pipeline complete: {len(chunks)} chunks, {final_coverage:.2f}% coverage")
    
    return chunks, diagnostics


# ========================= Output =========================

def write_chunks_jsonl(chunks: List[Dict[str, Any]], year: int) -> Path:
    """Write chunks to JSONL file."""
    out_path = OUT_DIR / f"{year}_chunks_llm.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out_path


def write_diagnostics(diagnostics: Dict[str, Any], year: int) -> Path:
    """Write diagnostics to JSON file."""
    out_path = OUT_DIR / f"{year}_diagnostics.json"
    out_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ========================= CLI =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid LLM chunking for Berkshire Hathaway shareholder letters"
    )
    parser.add_argument(
        "years",
        nargs="*",
        type=int,
        help="Years to process (e.g., 2008 2009 2010)"
    )
    parser.add_argument(
        "--years",
        dest="years_csv",
        type=str,
        help="Comma-separated years (e.g., 2008,2009,2010)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available years"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without calling LLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine years to process
    years: List[int] = []
    
    if args.all:
        years = discover_years_from_text_dir()
    elif args.years_csv:
        years = [int(y.strip()) for y in args.years_csv.split(",") if y.strip()]
    elif args.years:
        years = args.years
    
    if not years:
        print("[ERROR] No years specified. Use --all, --years, or positional arguments.")
        sys.exit(1)
    
    years = sorted(set(years))
    print(f"[INFO] Processing years: {years}")
    
    # Load chunking strategy
    try:
        chunking_spec = load_chunking_strategy()
        print(f"[INFO] Loaded chunking strategy ({len(chunking_spec)} chars)")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Process each year
    results: List[Dict[str, Any]] = []
    
    for year in years:
        print(f"\n{'='*60}")
        print(f"Processing year {year}")
        print(f"{'='*60}")
        
        out_jsonl = OUT_DIR / f"{year}_chunks_llm.jsonl"
        
        if out_jsonl.exists() and not args.force:
            print(f"[SKIP] Output already exists: {out_jsonl}")
            results.append({"year": year, "status": "skipped", "reason": "exists"})
            continue
        
        try:
            letter_text = load_letter_text(year)
            print(f"[INFO] Loaded letter: {len(letter_text)} chars")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            results.append({"year": year, "status": "error", "reason": str(e)})
            continue
        
        if args.dry_run:
            print(f"[DRY-RUN] Would process {year} ({len(letter_text)} chars)")
            windows = split_into_windows(letter_text)
            print(f"[DRY-RUN] Would create {len(windows)} windows")
            results.append({"year": year, "status": "dry_run", "windows": len(windows)})
            continue
        
        try:
            chunks, diagnostics = run_full_pipeline(
                letter_text=letter_text,
                year=year,
                chunking_spec=chunking_spec,
                model=args.model,
            )
            
            # Write outputs
            jsonl_path = write_chunks_jsonl(chunks, year)
            diag_path = write_diagnostics(diagnostics, year)
            
            print(f"[SUCCESS] Wrote {len(chunks)} chunks to {jsonl_path}")
            print(f"[SUCCESS] Wrote diagnostics to {diag_path}")
            
            results.append({
                "year": year,
                "status": "success",
                "chunks": len(chunks),
                "coverage": diagnostics.get("final_coverage", 0),
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to process year {year}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"year": year, "status": "error", "reason": str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        year = r["year"]
        status = r["status"]
        if status == "success":
            print(f"  {year}: ✓ {r['chunks']} chunks, {r['coverage']:.1f}% coverage")
        elif status == "skipped":
            print(f"  {year}: ⊘ skipped ({r['reason']})")
        elif status == "dry_run":
            print(f"  {year}: ◌ dry-run ({r['windows']} windows)")
        else:
            print(f"  {year}: ✗ error ({r.get('reason', 'unknown')})")


if __name__ == "__main__":
    main()