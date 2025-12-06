#!/usr/bin/env python
"""
Chunk Buffett letters into sections and chunks, and emit a DataFrame + JSONL.

Version v2.6

Key behaviors:
- Detects and preserves the front performance table section:
    "Berkshire’s Corporate Performance vs. the S&P 500"
  as a single logical section (header + table + notes).
- Treats the "To the Shareholders of Berkshire Hathaway Inc.:" block
  as an Overview section.
- Infers section headings from layout/casing (no hard-coded per-year list),
  with a heading heuristic tuned to:
    * ignore table labels (e.g., "Assets", "Company", "Total"),
    * avoid long sentence-like lines (e.g., with semicolons),
    * allow typical Buffett-style section titles.
- Uses line-level structure (headings, asterisk separators, bullets, blank lines)
  to build reasonable paragraphs.
- Enforces ~MAX_WORDS_PER_CHUNK via sentence-based splitting so chunks do not
  become overly large.
- Avoids creating useless micro-chunks from pure boilerplate / noise.

Input : ../data/clean_letters/{YEAR}.txt
Output: ../data/chunks/{YEAR}_chunks.parquet
        ../data/chunks/{YEAR}_chunks.jsonl
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Change this to the year you want to process
YEAR = 2009

INPUT_FILE = Path("../data/clean_letters_updated") / f"{YEAR}.txt"
OUTPUT_DIR = Path("../data/chunks_updated")
OUTPUT_PARQUET = OUTPUT_DIR / f"{YEAR}_chunks.parquet"
OUTPUT_JSONL = OUTPUT_DIR / f"{YEAR}_chunks.jsonl"

MAX_WORDS_PER_CHUNK = 180
MIN_WORDS_PER_CHUNK = 5  # don't keep ultra-tiny boilerplate chunks

# ---------------------------------------------------------------------------
# REGEX PATTERNS
# ---------------------------------------------------------------------------

# "To the Shareholders..." – marks the Overview section
HEADER_RE = re.compile(
    r"To the (Shareholders|Stockholders) of Berkshire Hathaway Inc\.\s*:?",
    flags=re.IGNORECASE,
)

# Performance table header ("Berkshire’s Corporate Performance vs. the S&P 500")
# Allow for straight or curly apostrophe and minor variations.
PERF_TABLE_RE = re.compile(
    r"Berkshire.?s\s+Corporate\s+Performance\s+vs\.\s+the\s+S&P\s+500",
    flags=re.IGNORECASE,
)

# Asterisk separator lines: "* * * * *", possibly with different spacing
ASTERISK_LINE_RE = re.compile(r"^(?:\*+\s*){5,}$")

# Lines that are obviously just page numbers / decoration
PAGE_LINE_RE = re.compile(r"^\s*\d+\s*$")

# Rough sentence boundary splitter
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Very generic table-like labels we do NOT want as top-level sections
TABLE_LABEL_STOPWORDS = {
    "company",
    "market",
    "total",
    "results",
    "assets",
    "liabilities and equity",
    "customer",
    "quarter",
    "pre-tax earnings",
    "of net earnings",
    "included",
    "cost * market",
    "income",
    "losses",
    "equity",
}


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text.strip())


def slugify(text: str, max_len: int = 30) -> str:
    """Very small slug function for section titles."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len] or "section"


# ---------------------------------------------------------------------------
# LINE-LEVEL TAGGING
# ---------------------------------------------------------------------------

def heading_heuristic(text: str) -> bool:
    """
    Decide if a line of text is a heading *purely by shape*, not by specific content.

    Improvements:
    - Strip leading page numbers (e.g. "5 Acquisitions" -> "Acquisitions") before analysis.
    - Ignore very generic table labels (Company, Market, Total, Assets, etc.).
    - Reject lines that look like long sentences (contain ';', '://', '@', etc.).
    - Use more tolerant casing thresholds (0.6 instead of 0.8).
    - Limit headings to a small number of words (<= 8) so sentence fragments are less likely.
    """
    t = text.strip()
    if not t:
        return False

    # Strip leading page numbers like "5 Acquisitions"
    t = re.sub(r"^\d+\s+", "", t).strip()
    if not t:
        return False

    # Exclude obvious noise characters only
    if re.fullmatch(r"[\d\W]+", t):
        return False

    # Reject ultra-short or very long lines
    if len(t) < 3 or len(t) > 80:
        return False

    words = t.split()
    if len(words) > 8:  # keep headings short
        return False

    # Exclude bullets
    if t == "•" or t.startswith("• ") or t.startswith("- "):
        return False

    # Exclude obvious full sentences or sentencey fragments:
    # - ending with .?! (except allow ":" for headings)
    # - containing semicolons or URL-like / email-like patterns
    if t.endswith((".", "?", "!")):
        return False
    if ";" in t or "://" in t or "@" in t:
        return False

    # Exclude table labels by content (case-insensitive)
    t_lower = t.lower().rstrip(" .:")
    if t_lower in TABLE_LABEL_STOPWORDS:
        return False

    # Casing analysis on letters only
    letters = re.sub(r"[^A-Za-z]+", "", t)
    if not letters:
        return False

    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)

    # Title-case ratio (ignore common small words)
    small_words = {"and", "or", "of", "the", "for", "to", "a", "an", "in", "on", "at"}
    words_for_title = [w for w in words if w.lower() not in small_words]
    if not words_for_title:
        words_for_title = words  # fallback

    title_case_words = sum(1 for w in words_for_title if w and w[0].isupper())
    title_case_ratio = title_case_words / len(words_for_title)

    # More tolerant thresholds than the previous 0.8
    if not (upper_ratio > 0.6 or title_case_ratio > 0.6):
        return False

    return True


def tag_lines(text: str) -> List[Dict]:
    """
    Convert raw text into a list of line dicts with structural tags.
    """
    raw_lines = text.split("\n")
    lines: List[Dict] = []

    for idx, raw in enumerate(raw_lines):
        strip = raw.strip()
        is_blank = (strip == "")
        is_asterisk_sep = bool(ASTERISK_LINE_RE.match(strip))
        is_perf_header = bool(PERF_TABLE_RE.search(strip))
        is_overview_header = bool(HEADER_RE.search(strip))
        is_page_line = bool(PAGE_LINE_RE.match(strip))
        is_bullet = bool(
            strip == "•" or strip.startswith("• ") or strip.startswith("- ")
        )

        lines.append(
            {
                "idx": idx,
                "raw": raw,
                "text": strip,
                "is_blank": is_blank,
                "is_asterisk_sep": is_asterisk_sep,
                "is_perf_header": is_perf_header,
                "is_overview_header": is_overview_header,
                "is_page_line": is_page_line,
                "is_bullet": is_bullet,
                "is_heading_candidate": False,  # filled in second pass
            }
        )

    # Second pass: heading candidates (shape only)
    for ln in lines:
        if (
            not ln["is_blank"]
            and not ln["is_asterisk_sep"]
            and not ln["is_page_line"]
            and not ln["is_perf_header"]
            and not ln["is_overview_header"]
            and not ln["is_bullet"]
        ):
            ln["is_heading_candidate"] = heading_heuristic(ln["text"])

    return lines


# ---------------------------------------------------------------------------
# SECTION DETECTION
# ---------------------------------------------------------------------------

def detect_sections(lines: List[Dict]) -> List[Tuple[str, List[int]]]:
    """
    Detect sections as (title, list_of_line_indices).

    Uses:
    - Performance table ("Berkshire’s Corporate Performance vs. the S&P 500")
    - Overview ("To the Shareholders...")
    - Inferred headings (heading_heuristic)
    - Fallback "Body" / "Unassigned" for leftovers.

    Important v2.6 behavior:
    - The performance table region is treated as a single section from the
      performance header down to the Overview header (if present). Heading
      candidates inside this region do NOT create their own sections.
    """
    n = len(lines)
    sections: List[Tuple[str, List[int]]] = []
    used: set[int] = set()

    def first_index(pred) -> int | None:
        for i, ln in enumerate(lines):
            if pred(ln):
                return i
        return None

    # Precompute overview_start (may be None)
    overview_start = first_index(lambda ln: ln["is_overview_header"])

    # --- Performance table section -----------------------------------------
    perf_start = first_index(lambda ln: ln["is_perf_header"])
    perf_indices: List[int] = []

    if perf_start is not None:
        # Include preceding "Note:" line if present
        if perf_start > 0 and lines[perf_start - 1]["text"].startswith("Note:"):
            perf_start -= 1

        # If we have an Overview header after the performance header,
        # treat that as the end of the table section.
        if overview_start is not None and overview_start > perf_start:
            perf_end = overview_start
        else:
            # Fallback: end at first heading candidate after perf_start, or EOF
            perf_end = n
            for i in range(perf_start + 1, n):
                if lines[i]["is_heading_candidate"]:
                    perf_end = i
                    break

        perf_indices = list(range(perf_start, perf_end))
        sections.append(("Berkshire’s Corporate Performance vs. the S&P 500", perf_indices))
        used.update(perf_indices)

    # --- Overview section ---------------------------------------------------
    if overview_start is not None:
        overview_end = n
        for i in range(overview_start + 1, n):
            if lines[i]["is_heading_candidate"]:
                overview_end = i
                break

        overview_indices = [
            i for i in range(overview_start, overview_end) if i not in used
        ]
        if overview_indices:
            sections.append(("Overview", overview_indices))
            used.update(overview_indices)

    # --- Remaining sections based on inferred headings ----------------------
    heading_idxs = [
        i
        for i, ln in enumerate(lines)
        if ln["is_heading_candidate"]
        and i not in used
        and not ln["is_perf_header"]
        and not ln["is_overview_header"]
    ]
    heading_idxs.sort()

    def clean_heading_title(text: str) -> str:
        t = text.strip()
        # Remove leading page number like "5 Insurance"
        t = re.sub(r"^\d+\s+", "", t)
        t = t.rstrip(" .:")
        return t or "Section"

    for idx_pos, h_idx in enumerate(heading_idxs):
        title = clean_heading_title(lines[h_idx]["text"])
        start = h_idx + 1
        next_heading = heading_idxs[idx_pos + 1] if idx_pos + 1 < len(heading_idxs) else n
        end = next_heading

        line_indices = [
            i for i in range(start, end) if i not in used
        ]
        if line_indices:
            sections.append((title, line_indices))
            used.update(line_indices)

    # --- Leftover lines -----------------------------------------------------
    remaining = [i for i in range(n) if i not in used]
    if remaining:
        title = "Body" if not sections else "Unassigned"
        sections.append((title, remaining))

    return sections


# ---------------------------------------------------------------------------
# PARAGRAPH BUILDING WITHIN SECTIONS
# ---------------------------------------------------------------------------

def join_paragraph_lines(lines: List[str]) -> str:
    if not lines:
        return ""
    first = lines[0]
    rest = lines[1:]
    if first.startswith("•") or first.startswith("- "):
        combined = " ".join([first] + rest)
    else:
        combined = " ".join(lines)
    return normalize_spaces(combined)


def build_paragraphs_for_section(lines: List[Dict], line_indices: List[int]) -> List[str]:
    """
    Given a list of line dicts and indices belonging to a section,
    build a list of paragraph strings.

    We break paragraphs at:
    - Blank lines
    - Asterisk separators
    - Page lines
    - Bullet boundaries (bullets + their following text)
    Heading lines themselves are treated as boundaries and skipped.
    """
    indices = sorted(line_indices)
    paragraphs: List[str] = []
    current_lines: List[str] = []

    def flush_current():
        nonlocal current_lines
        if not current_lines:
            return
        para = join_paragraph_lines(current_lines)
        # Skip decorative-only paragraphs
        if para and not re.fullmatch(r"[\d\W]+", para):
            paragraphs.append(para)
        current_lines = []

    i = 0
    m = len(indices)

    while i < m:
        line_idx = indices[i]
        ln = lines[line_idx]
        t = ln["text"]

        # Structural breaks
        if ln["is_asterisk_sep"] or ln["is_blank"] or ln["is_page_line"]:
            flush_current()
            i += 1
            continue

        # Bullet paragraph
        if ln["is_bullet"]:
            flush_current()
            bullet_lines = [t]
            j = i + 1
            while j < m:
                next_ln = lines[indices[j]]
                nt = next_ln["text"]
                if (
                    next_ln["is_blank"]
                    or next_ln["is_bullet"]
                    or next_ln["is_asterisk_sep"]
                    or next_ln["is_heading_candidate"]
                    or next_ln["is_page_line"]
                ):
                    break
                bullet_lines.append(nt)
                j += 1
            para = join_paragraph_lines(bullet_lines)
            if para and not re.fullmatch(r"[\d\W]+", para):
                paragraphs.append(para)
            i = j
            continue

        # Heading candidate inside a section: treat as boundary and skip content
        if ln["is_heading_candidate"]:
            flush_current()
            i += 1
            continue

        # Normal text line
        current_lines.append(t)
        i += 1

    flush_current()
    return paragraphs


# ---------------------------------------------------------------------------
# SPLIT LARGE PARAGRAPHS BY SENTENCE
# ---------------------------------------------------------------------------

def split_large_paragraph(paragraph: str, max_words: int) -> List[str]:
    """
    Split a too-large paragraph into smaller chunks by sentence boundaries,
    each not exceeding max_words (approximately).
    """
    words = paragraph.split()
    if len(words) <= max_words:
        return [paragraph]

    # Rough sentence splitting
    sentences = SENTENCE_SPLIT_RE.split(paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: List[str] = []
    current_words: List[str] = []

    for sent in sentences:
        sent_words = sent.split()
        if not current_words:
            current_words.extend(sent_words)
        elif len(current_words) + len(sent_words) <= max_words:
            current_words.extend(sent_words)
        else:
            chunks.append(normalize_spaces(" ".join(current_words)))
            current_words = sent_words[:]

    if current_words:
        chunks.append(normalize_spaces(" ".join(current_words)))

    return chunks


def enforce_paragraph_word_limit(paragraphs: List[str], max_words: int) -> List[str]:
    """
    For each paragraph, if its word count exceeds max_words, split it into
    smaller sentence-based sub-paragraphs.
    """
    out: List[str] = []
    for p in paragraphs:
        words = p.split()
        if len(words) > max_words:
            out.extend(split_large_paragraph(p, max_words))
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------

def is_boilerplate_chunk(text: str, num_words: int) -> bool:
    """
    Heuristic to drop ultra-tiny, meaningless chunks:
    - e.g., "BERKSHIRE HATHAWAY INC." alone, page headers, etc.
    """
    if num_words >= MIN_WORDS_PER_CHUNK:
        return False

    t = text.strip()
    if not t:
        return True

    # Pure digits/punctuation
    if re.fullmatch(r"[\d\W]+", t):
        return True

    # All-caps 1–4 word boilerplate (e.g. "BERKSHIRE HATHAWAY INC.")
    if t.upper() == t and len(t.split()) <= 4:
        return True

    return False


def chunk_section(
    year: int,
    source_file: str,
    section_title: str,
    paragraphs: List[str],
    max_words: int = MAX_WORDS_PER_CHUNK,
) -> List[Dict]:
    """
    Turn a section (list of paragraph strings) into size-limited chunks.
    """
    rows: List[Dict] = []
    section_slug = slugify(section_title)
    pos_in_section = 0

    # First enforce paragraph-level max word count
    paragraphs = enforce_paragraph_word_limit(paragraphs, max_words)

    i = 0
    n = len(paragraphs)

    while i < n:
        start_idx = i
        word_count = 0
        chunk_paras: List[str] = []

        while i < n:
            p = paragraphs[i]
            p_words = len(p.split())
            if chunk_paras and word_count + p_words > max_words:
                break
            chunk_paras.append(p)
            word_count += p_words
            i += 1

        if not chunk_paras:
            break

        end_idx = i - 1
        chunk_text = "\n\n".join(chunk_paras)
        num_words = len(chunk_text.split())

        # Drop ultra-tiny boilerplate chunks
        if is_boilerplate_chunk(chunk_text, num_words):
            continue

        pos_in_section += 1

        prev_context = paragraphs[start_idx - 1] if start_idx > 0 else ""
        next_context = paragraphs[end_idx + 1] if end_idx + 1 < n else ""

        chunk_id = f"{year}_{section_slug}_{pos_in_section:03d}"

        rows.append(
            {
                "year": year,
                "source_file": source_file,
                "section_title": section_title,
                "chunk_id": chunk_id,
                "position_in_section": pos_in_section,
                "num_words": num_words,
                "text": chunk_text,
                "prev_context": prev_context,
                "next_context": next_context,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {INPUT_FILE} …")
    text = read_text(INPUT_FILE)

    print("Tagging lines …")
    lines = tag_lines(text)
    print(f"Total lines: {len(lines)}")

    print("Detecting sections …")
    sections = detect_sections(lines)
    print(f"Detected {len(sections)} sections:")
    for title, idxs in sections:
        print(f"  - {title!r}: {len(idxs)} lines")

    all_rows: List[Dict] = []
    source_file = INPUT_FILE.name

    print("Building paragraphs and chunking …")
    for section_title, line_indices in sections:
        paragraphs = build_paragraphs_for_section(lines, line_indices)
        if not paragraphs:
            continue
        section_rows = chunk_section(
            year=YEAR,
            source_file=source_file,
            section_title=section_title,
            paragraphs=paragraphs,
        )
        all_rows.extend(section_rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No chunks produced; nothing to save.")
        return

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
