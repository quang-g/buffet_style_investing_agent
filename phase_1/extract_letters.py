"""
Extract and clean Warren Buffett letters (HTML + PDF)
into plain-text files suitable for RAG / NLP.

Dependencies (install via pip):
    pip install pdfminer.six beautifulsoup4

Directory layout (relative to this script):
    data/raw/           # original .html / .pdf letters
    data/clean_letters/ # output {year}.txt

Usage:
    python extract_letters.py
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterator, Tuple

from bs4 import BeautifulSoup
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean_letters")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Header used to locate the body of the letter
HEADER_RE = re.compile(
    r"To the (Shareholders|Stockholders) of Berkshire Hathaway Inc\.\s*:",
    flags=re.IGNORECASE,
)


# -------------------------------------------------------------------
# HTML EXTRACTION
# -------------------------------------------------------------------

def extract_html_raw(path: Path) -> str:
    """
    Return raw text from an HTML letter.

    - Uses BeautifulSoup to strip tags, scripts, styles.
    - Keeps basic line breaks via separator="\\n".
    """
    html_bytes = path.read_bytes()
    soup = BeautifulSoup(html_bytes, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style"]):
        tag.decompose()

    body = soup.body or soup
    text = body.get_text(separator="\n")

    # Normalize crazy blank-line runs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def slice_letter_body_html(text: str) -> str:
    """
    For HTML letters, keep from 'To the Shareholders...' onward.

    If the header cannot be found, return the full text as a fallback
    (better to keep noise than lose the letter).
    """
    m = HEADER_RE.search(text)
    if not m:
        return text.strip()

    body = text[m.start():]
    return body.strip()


# -------------------------------------------------------------------
# PDF EXTRACTION (PDFMINER)
# -------------------------------------------------------------------

def fix_spacing_glitches(text: str) -> str:
    """
    Heuristic fixes for common no-space / fraction glitches that appear
    in Buffett letters when extracted from PDF.

    This is intentionally conservative; it won't be perfect but will
    remove many of the worst artifacts (e.g., 'Berkshire’sCorporatePerformance').
    """
    # Add a space between a lowercase letter and a following Uppercase letter
    # e.g., "floatAnd" -> "float And"
    text = re.sub(r"(?<=[a-z\u00df-\u024f])(?=[A-Z])", " ", text)

    # Add a space between digits and letters in both directions
    # e.g., "500Index" -> "500 Index", "S&P500" -> "S&P 500"
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)

    # Normalize some common "fraction" encodings
    # These are not exhaustive but handle frequent Buffett-letter cases.
    fraction_map = {
        "1⁄2": "1/2",
        "1⁄4": "1/4",
        "3⁄4": "3/4",
        "11⁄2": "11 1/2",
        "21⁄2": "21 1/2",
    }
    for bad, good in fraction_map.items():
        text = text.replace(bad, good)

    return text


def extract_pdf_raw(path: Path) -> str:
    """
    Use pdfminer.six to extract text with better spacing than pdfplumber.

    - Iterates over layout objects per page.
    - Collects text from LTTextContainer blocks.
    - Applies light spacing/fraction fixes.
    """
    page_texts: list[str] = []

    for page_layout in extract_pages(path):
        blocks: list[str] = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                t = element.get_text()
                if not t:
                    continue
                # Normalize newlines (pdfminer sometimes mixes \r\n, \r, \n)
                t = t.replace("\r\n", "\n").replace("\r", "\n")
                blocks.append(t.rstrip())

        if blocks:
            page_texts.append("\n".join(blocks))

    text = "\n\n".join(page_texts)

    # Fix common no-space issues
    text = fix_spacing_glitches(text)

    # Normalize blank-line runs
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def slice_letter_body_pdf(text: str) -> str:
    """
    For PDF letters, keep from 'To the Shareholders...' onward.

    If the header cannot be found, return the full text as a fallback.
    """
    m = HEADER_RE.search(text)
    if not m:
        return text.strip()

    body = text[m.start():]
    return body.strip()


# -------------------------------------------------------------------
# NORMALIZATION / PARAGRAPH HANDLING
# -------------------------------------------------------------------

def unwrap_paragraphs(text: str) -> str:
    """
    Join soft-wrapped lines into paragraphs.

    - Blank lines are treated as paragraph breaks.
    - Non-empty lines within a paragraph are joined with spaces.
    """
    lines = text.splitlines()
    paras: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paras.append(" ".join(current))
                current = []
        else:
            current.append(stripped)

    if current:
        paras.append(" ".join(current))

    return "\n\n".join(paras)


def normalize_text(text: str, unwrap: bool = True) -> str:
    """
    Normalize whitespace, newlines, and paragraph spacing.

    - unwrap=True  : join soft-wrapped lines into paragraphs (HTML letters).
    - unwrap=False : keep line breaks close to original (PDF letters).
    """
    # Normalize newline types first
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if unwrap:
        # For HTML, we prefer compact paragraphs
        text = unwrap_paragraphs(text)

    # Clean per-line whitespace but keep line breaks
    lines = text.splitlines()
    cleaned_lines: list[str] = []
    prev_blank = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Collapse multiple blank lines into a single blank line
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
        else:
            # Collapse internal runs of whitespace to a single space
            stripped = re.sub(r"\s{2,}", " ", stripped)
            cleaned_lines.append(stripped)
            prev_blank = False

    out = "\n".join(cleaned_lines).strip()
    # Safety: collapse very long blank-line runs again
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


# -------------------------------------------------------------------
# PIPELINES PER FILE TYPE
# -------------------------------------------------------------------

def process_html(path: Path) -> str:
    """
    Full pipeline for an HTML letter:
        HTML -> raw text -> cropped body -> normalized.
    """
    raw = extract_html_raw(path)
    body = slice_letter_body_html(raw)
    # HTML: keep paragraphs compact
    return normalize_text(body, unwrap=True)


def process_pdf(path: Path) -> str:
    """
    Full pipeline for a PDF letter:
        PDF -> raw text (pdfminer) -> cropped body -> normalized.

    We keep unwrap=False so line breaks stay closer to the original,
    which is more human-friendly.
    """
    raw = extract_pdf_raw(path)
    body = slice_letter_body_pdf(raw)
    # PDF: keep original line breaks for human-friendly reading
    return normalize_text(body, unwrap=False)


# -------------------------------------------------------------------
# FILE DISCOVERY
# -------------------------------------------------------------------

def iter_letter_files() -> Iterator[Tuple[int, Path, str]]:
    """
    Yield (year, path, kind) for all letters found in RAW_DIR.

    - kind is "html" or "pdf".
    - year is inferred from the filename (first 4-digit 19xx/20xx).
    """
    if not RAW_DIR.exists():
        print(f"[WARN] RAW_DIR does not exist: {RAW_DIR.resolve()}")
        return

    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        if ext not in {".html", ".htm", ".pdf"}:
            continue

        m = re.search(r"(19|20)\d{2}", path.stem)
        if not m:
            print(f"[WARN] Could not infer year from filename: {path}")
            continue

        year = int(m.group(0))
        kind = "html" if ext in {".html", ".htm"} else "pdf"
        yield year, path, kind


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main() -> None:
    found_any = False

    for year, path, kind in iter_letter_files():
        found_any = True

        out_path = CLEAN_DIR / f"{year}.txt"

        # If a cleaned file already exists, leave it alone
        if out_path.exists():
            print(f"[SKIP] {year}: cleaned file already exists at {out_path}")
            continue

        if kind == "html":
            print(f"[HTML] Processing {year}: {path}")
            cleaned = process_html(path)
        else:
            print(f"[PDF ] Processing {year}: {path}")
            cleaned = process_pdf(path)

        out_path.write_text(cleaned, encoding="utf-8")
        print(f"  -> saved cleaned letter to {out_path}")

    if not found_any:
        print(f"[INFO] No letter files found under {RAW_DIR.resolve()}")


if __name__ == "__main__":
    main()
