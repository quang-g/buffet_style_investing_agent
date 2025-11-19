from pathlib import Path
import re

import pdfplumber
from bs4 import BeautifulSoup

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Folders relative to where you run this script
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean_letters_1")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Shared header pattern used for BOTH HTML & PDF
HEADER_RE = re.compile(
    r"To the (Shareholders|Stockholders) of Berkshire Hathaway Inc\.\s*:",
    flags=re.IGNORECASE,
)


# -------------------------------------------------------------------
# LOW-LEVEL PARSERS
# -------------------------------------------------------------------

def extract_html_raw(path: Path) -> str:
    """
    Return raw text from an HTML letter (before cropping / cleaning).

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

    # Collapse very long runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def extract_pdf_raw(path: Path) -> str:
    """
    Return raw text from a PDF letter (before cropping / cleaning).

    - Concatenates all pages with newlines.
    """
    pages_text: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                pages_text.append(txt)

    text = "\n".join(pages_text)

    # Collapse very long runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


# -------------------------------------------------------------------
# CROPPING: FIND THE "REAL" LETTER BODY
# -------------------------------------------------------------------

def slice_letter_body_html(text: str) -> str:
    """
    For HTML letters, keep from 'To the Shareholders...' onward.

    If the header cannot be found, return the full text as a fallback.
    This avoids blank outputs for weird years (e.g. 1997, 1998).
    """
    m = HEADER_RE.search(text)
    if not m:
        # Fallback: we prefer messy data to losing the letter
        return text.strip()

    body = text[m.start():]
    return body.strip()


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
# NORMALIZATION / FORMATTING
# -------------------------------------------------------------------

def unwrap_paragraphs(text: str) -> str:
    """
    Join soft-wrapped lines into paragraphs.

    - Blank lines are treated as paragraph breaks.
    - Non-empty lines within a paragraph are joined with single spaces.
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

    - If unwrap=True  : join soft-wrapped lines into paragraphs
                        (good for HTML letters).
    - If unwrap=False : keep original line breaks as much as possible
                        (good for PDFs / human readability).
    """
    # Normalize newline types first
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # For HTML we typically want compact paragraphs
    if unwrap:
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
            # Collapse runs of internal whitespace to a single space
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
    PDF -> raw text -> cropped body -> normalized.

    We keep unwrap=False so line breaks stay closer to the original,
    which is more human-friendly for reading.
    """
    raw = extract_pdf_raw(path)
    body = slice_letter_body_pdf(raw)
    # PDF: keep original line breaks for human-friendly reading
    return normalize_text(body, unwrap=False)


# -------------------------------------------------------------------
# FILE DISCOVERY
# -------------------------------------------------------------------

def iter_letter_files():
    """
    Yield (year, path, kind) for all letters found in RAW_DIR.

    - kind is "html" or "pdf".
    - year is inferred from the filename (first 4-digit year found).
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

def main():
    found_any = False

    for year, path, kind in iter_letter_files():
        found_any = True

        out_path = CLEAN_DIR / f"{year}.txt"

        # ---------------- Existing cleaned file handling ----------------
        if out_path.exists():
            print(f"[SKIP] {year}: cleaned file already exists at {out_path}")
            # If you want to force re-processing, comment out the 'continue'
            # and instead implement backup/overwrite logic here.
            continue

        # ---------------- Process based on type ----------------
        if kind == "html":
            print(f"[HTML] Processing {year}: {path}")
            cleaned = process_html(path)
        else:
            print(f"[PDF]  Processing {year}: {path}")
            cleaned = process_pdf(path)

        # ---------------- Save ----------------
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"  -> saved cleaned letter to {out_path}")

    if not found_any:
        print(f"[INFO] No letter files found under {RAW_DIR.resolve()}")


if __name__ == "__main__":
    main()
