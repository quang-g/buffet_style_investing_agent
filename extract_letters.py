from pathlib import Path
import re

import pdfplumber
from bs4 import BeautifulSoup

# Folders relative to where you run this script
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean_letters")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# ---------- PARSERS ----------

def extract_html_raw(path: Path) -> str:
    """Return raw text from an HTML letter."""
    html = path.read_text(encoding="latin-1", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    pre = soup.find("pre")
    if pre:
        text = pre.get_text("\n")
    else:
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text("\n")

    return text


def extract_pdf_raw(path: Path) -> str:
    """Return raw text from a PDF letter."""
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            chunks.append(page_text)
    return "\n".join(chunks)


# ---------- HEADER/FOOTER REMOVAL ----------

def slice_letter_body_html(text: str) -> str:
    """
    For HTML-based letters: chop off everything before
    'To the Stockholders/Shareholders of Berkshire Hathaway Inc.'
    and drop obvious junk like page numbers and headers.
    """
    lines = text.splitlines()

    start_idx = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if "to the stockholders of berkshire hathaway inc" in low \
           or "to the shareholders of berkshire hathaway inc" in low:
            start_idx = i
            break

    cleaned_lines = []
    for line in lines[start_idx:]:
        stripped = line.strip()

        # Drop page numbers like "2", "3"
        if stripped.isdigit():
            continue

        # Drop obvious headers
        if stripped == "BERKSHIRE HATHAWAY INC.":
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def slice_letter_body_pdf(text: str) -> str:
    """
    For PDF-based letters: cut before the main greeting and remove
    some common performance-table / header junk.
    """
    lines = text.splitlines()

    # Find greeting line
    start_idx = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if "to the shareholders of berkshire hathaway inc" in low \
           or "to the stockholders of berkshire hathaway inc" in low:
            start_idx = i
            break

    lines = lines[start_idx:]

    cleaned = []
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # page numbers
        if stripped.isdigit():
            continue

        # common headers from PDFs (you can add more patterns later)
        if stripped.startswith("BERKSHIRE HATHAWAY INC"):
            continue
        if "berkshire’s corporate performance vs. the s&p 500" in lower:
            continue
        if "corporate performance vs. the s&p 500" in lower:
            continue
        if "average annual gain —" in lower:
            continue
        if "overall gain —" in lower:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ---------- NORMALIZATION ----------

def unwrap_paragraphs(text: str) -> str:
    """
    Join soft-wrapped lines into paragraphs, keeping blank lines
    as paragraph breaks.
    """
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines()]

    paras = []
    current = []

    for line in lines:
        if not line:  # blank line -> end of paragraph
            if current:
                paras.append(" ".join(current))
                current = []
        else:
            current.append(line)

    if current:
        paras.append(" ".join(current))

    return "\n\n".join(paras)


def normalize_text(text: str) -> str:
    """Normalize whitespace, newlines, and paragraph spacing."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unwrap_paragraphs(text)
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse >2 blank lines
    return text


# ---------- PIPELINE PER TYPE ----------

def process_html(path: Path) -> str:
    raw = extract_html_raw(path)
    body = slice_letter_body_html(raw)
    return normalize_text(body)


def process_pdf(path: Path) -> str:
    raw = extract_pdf_raw(path)
    body = slice_letter_body_pdf(raw)
    return normalize_text(body)


# ---------- ENTRY POINT ----------

def iter_letter_files():
    """
    Yield (year, path, kind) tuples for files stored like
    data/raw/<year>/letter_<year>.<ext>.
    """
    for year_dir in sorted(RAW_DIR.iterdir()):
        if not year_dir.is_dir():
            continue

        year = year_dir.name
        for path in sorted(year_dir.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in {".html", ".htm"}:
                yield year, path, "html"
            elif suffix == ".pdf":
                yield year, path, "pdf"


def main():
    found_any = False
    for year, path, kind in iter_letter_files():
        found_any = True
        if kind == "html":
            print(f"Processing HTML letter for {year}: {path}")
            cleaned = process_html(path)
        else:
            print(f"Processing PDF letter for {year}: {path}")
            cleaned = process_pdf(path)

        out_path = CLEAN_DIR / f"{year}.txt"
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"  -> saved cleaned letter to {out_path}")

    if not found_any:
        print(f"No letter files found under {RAW_DIR}")


if __name__ == "__main__":
    main()
