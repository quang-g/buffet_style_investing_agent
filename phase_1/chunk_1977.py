#!/usr/bin/env python
"""
Chunk 1977 Buffett letter into sections and chunks, and emit a DataFrame + JSONL.

- Input : data/cleaned_text/1977.txt
- Output: data/chunks/1977_chunks.parquet
         data/chunks/1977_chunks.jsonl
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd

# ---- CONFIG -----------------------------------------------------------------

<<<<<<< HEAD
YEAR = 2021
INPUT_FILE = Path("../data/clean_letters/2021.txt")
OUTPUT_DIR = Path("../data/chunks")
=======
YEAR = 1977
INPUT_FILE = Path("../data/clean_letters/1977.txt")
OUTPUT_DIR = Path("../data/chunks/improved")
>>>>>>> 2c63147ffb838fb383200e74ec86f3d8e138b07c
OUTPUT_PARQUET = OUTPUT_DIR / f"{YEAR}_chunks.parquet"
OUTPUT_JSONL = OUTPUT_DIR / f"{YEAR}_chunks.jsonl"

MAX_WORDS_PER_CHUNK = 180
MIN_WORDS_PER_CHUNK = 80
PARAGRAPH_OVERLAP = 0

# ---- UTILITIES --------------------------------------------------------------


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # Normalize Windows / mixed newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs using blank lines as separators."""
    # Split on 1+ blank lines
    raw_paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs = [p.strip() for p in raw_paragraphs]
    # Drop empties
    paragraphs = [p for p in paragraphs if p]
    return paragraphs


def is_heading(paragraph: str) -> bool:
    """
    Heuristic to detect section headings.

    Characteristics we look for:
    - Not too short, not too long
    - Few words
    - Mostly uppercase OR title-case
    - Not just numbers / asterisk lines
    """
    p = paragraph.strip()

    # Exclude very short or very long
    if len(p) < 3 or len(p) > 80:
        return False

    # Exclude lines that are just numbers or decoration (* * * * *)
    if re.fullmatch(r"[\d\W]+", p):
        return False

    words = p.split()
    if len(words) > 10:
        return False

    # Take only letters for casing analysis
    letters = re.sub(r"[^A-Za-z]+", "", p)
    if not letters:
        return False

    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)

    # Condition 1: mostly uppercase (e.g. INSURANCE, OPERATING RESULTS)
    if upper_ratio > 0.8:
        return True

    # Condition 2: mostly Title Case words
    title_case_words = sum(1 for w in words if w[0].isupper())
    if title_case_words / len(words) > 0.8:
        return True

    return False


def detect_sections(paragraphs: List[str]) -> List[Tuple[str, List[int]]]:
    """
    Detect section boundaries using headings.

    Returns a list of (section_title, paragraph_indices).
    Paragraph indices refer to the original paragraphs list.
    """
    heading_indices = [i for i, p in enumerate(paragraphs) if is_heading(p)]

    sections: List[Tuple[str, List[int]]] = []

    if not heading_indices:
        # Fallback: everything is one big "Body" section
        sections.append(("Body", list(range(len(paragraphs)))))
        return sections

    # 1) Anything before first heading -> "Introduction"
    first_heading_idx = heading_indices[0]
    if first_heading_idx > 0:
        intro_indices = list(range(0, first_heading_idx))
        sections.append(("Introduction", intro_indices))

    # 2) Each heading defines a section from the paragraph AFTER the heading
    #    up to (but not including) the next heading.
    for j, h_idx in enumerate(heading_indices):
        title = paragraphs[h_idx].strip()
        start = h_idx + 1
        end = heading_indices[j + 1] if j + 1 < len(heading_indices) else len(paragraphs)
        if start >= end:
            # Empty section (heading followed by another heading or end-of-doc) -> skip
            continue
        para_indices = list(range(start, end))
        sections.append((title, para_indices))

    return sections

def get_prev_paragraphs(paragraphs: List[str], start_idx: int, n: int = 2) -> str:
    """Get up to n paragraphs before start_idx from the global paragraphs list."""
    prev_idxs = list(range(max(0, start_idx - n), start_idx))
    if not prev_idxs:
        return ""
    return "\n\n".join(paragraphs[i] for i in prev_idxs)


def get_next_paragraphs(paragraphs: List[str], end_idx: int, n: int = 2) -> str:
    """Get up to n paragraphs after end_idx from the global paragraphs list."""
    next_idxs = list(range(end_idx + 1, min(len(paragraphs), end_idx + 1 + n)))
    if not next_idxs:
        return ""
    return "\n\n".join(paragraphs[i] for i in next_idxs)


def is_data_table(text: str) -> bool:
    """Detect if chunk is primarily a data table."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 3:
        return False
    
    # Count lines with multiple numbers (tables often have many numbers)
    number_heavy_lines = sum(
        1 for line in lines 
        if len(re.findall(r'\d+', line)) >= 3
    )
    
    # If >60% of lines have 3+ numbers, likely a table
    return (number_heavy_lines / len(lines)) > 0.6

def slugify(text: str, max_len: int = 30) -> str:
    """Very small slug function for section titles."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len] or "section"


def chunk_section(
    year: int,
    source_file: str,
    section_title: str,
    para_indices: List[int],
    paragraphs: List[str],
    max_words: int = MAX_WORDS_PER_CHUNK,
    min_words: int = MIN_WORDS_PER_CHUNK,
    overlap: int = PARAGRAPH_OVERLAP,
) -> List[Dict]:
    """
    Turn a section (list of paragraph indices) into size-limited chunks.

    Chunks are built by concatenating paragraphs until max_words is exceeded.
    Ensures minimum chunk size is met before breaking.
    Overlap is measured in whole paragraphs.
    
    Args:
        year: Year of the letter
        source_file: Source filename
        section_title: Title of the section
        para_indices: List of paragraph indices in this section
        paragraphs: Global list of all paragraphs
        max_words: Target maximum words per chunk
        min_words: Minimum words per chunk
        overlap: Number of paragraphs to overlap between chunks
        
    Returns:
        List of chunk dictionaries with metadata
    """
    rows: List[Dict] = []
    section_slug = slugify(section_title)
    pos_in_section = 0

    i = 0
    n = len(para_indices)

    while i < n:
        start_idx = i
        word_count = 0
        chunk_para_idxs: List[int] = []

        # Accumulate paragraphs for this chunk
        while i < n:
            p_idx = para_indices[i]
            p_text = paragraphs[p_idx]
            p_words = len(p_text.split())

            # Check if adding this paragraph would exceed max_words
            if chunk_para_idxs and word_count + p_words > max_words:
                # Only break if we've already met minimum size
                if word_count >= min_words:
                    break
                # Otherwise continue to reach minimum, even if exceeding max

            chunk_para_idxs.append(p_idx)
            word_count += p_words
            i += 1

        # Safety: if somehow we have no paragraphs (shouldn't happen)
        if not chunk_para_idxs:
            p_idx = para_indices[i]
            chunk_para_idxs = [p_idx]
            i += 1

        pos_in_section += 1
        
        # Get global paragraph indices for this chunk
        global_start_para_idx = chunk_para_idxs[0]
        global_end_para_idx = chunk_para_idxs[-1]

        # Build chunk text
        text_parts = [paragraphs[p] for p in chunk_para_idxs]
        chunk_text = "\n\n".join(text_parts)
        
        # Recalculate from final text for accuracy
        num_words = len(chunk_text.split())

        # Get surrounding context from global paragraphs
        prev_context = get_prev_paragraphs(paragraphs, global_start_para_idx, n=2)
        next_context = get_next_paragraphs(paragraphs, global_end_para_idx, n=2)
        
        # Detect content type
        is_table = is_data_table(chunk_text)
        content_type = "table" if is_table else "prose"

        chunk_id = f"{year}_{section_slug}_{pos_in_section:03d}"

        row = {
            "year": year,
            "source_file": source_file,
            "section_title": section_title,
            "chunk_id": chunk_id,
            "position_in_section": pos_in_section,
            "start_paragraph_idx": global_start_para_idx,
            "end_paragraph_idx": global_end_para_idx,
            "num_words": num_words,
            "text": chunk_text,
            
            # Context for LLM enrichment
            "prev_context": prev_context,
            "next_context": next_context,
            "section_summary": section_title,
            
            # Content classification
            "content_type": content_type,
            "is_table": is_table,
            
            # Will be populated later in Phase 1 Task 3
            "contextual_summary": None,
        }
        rows.append(row)

        # Move to next chunk with optional overlap
        if overlap > 0:
            # Step back by overlap paragraphs, but always move forward at least 1
            i = max(i - overlap, start_idx + 1)

    return rows

def get_prev_paragraphs(paragraphs: List[str], start_idx: int, n: int = 2) -> str:
    """Get up to n paragraphs before start_idx."""
    prev_idxs = list(range(max(0, start_idx - n), start_idx))
    if not prev_idxs:
        return ""
    return "\n\n".join(paragraphs[i] for i in prev_idxs)

def get_next_paragraphs(paragraphs: List[str], end_idx: int, n: int = 2) -> str:
    """Get up to n paragraphs after end_idx."""
    next_idxs = list(range(end_idx + 1, min(len(paragraphs), end_idx + 1 + n)))
    if not next_idxs:
        return ""
    return "\n\n".join(paragraphs[i] for i in next_idxs)


# ---- MAIN PIPELINE ----------------------------------------------------------


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {INPUT_FILE} …")
    text = read_text(INPUT_FILE)

    print("Splitting into paragraphs …")
    paragraphs = split_into_paragraphs(text)
    print(f"Total paragraphs: {len(paragraphs)}")

    print("Detecting sections via headings …")
    sections = detect_sections(paragraphs)
    print(f"Detected {len(sections)} sections:")
    for title, idxs in sections:
        print(f"  - {title!r}: {len(idxs)} paragraphs")

    all_rows: List[Dict] = []
    source_file = INPUT_FILE.name

    print("Chunking sections …")
    for section_title, para_idxs in sections:
        section_rows = chunk_section(
            year=YEAR,
            source_file=source_file,
            section_title=section_title,
            para_indices=para_idxs,
            paragraphs=paragraphs,
        )
        all_rows.extend(section_rows)

    df = pd.DataFrame(all_rows)

    # Sort for readability
    df = df.sort_values(
        by=["year", "section_title", "position_in_section"]
    ).reset_index(drop=True)

    print(f"Total chunks: {len(df)}")

    print(f"Saving Parquet to {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"Saving JSONL to {OUTPUT_JSONL}")
    df.to_json(OUTPUT_JSONL, orient="records", lines=True, force_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
