"""
Text Extraction Script for Buffett Letters (Extraction-Only Version)

Folder layout expected (relative to this script file):

  <repo_root>/
    data/
      raw/
        1977/
          letter_1977.html
        1996/
          letter_1996.html
        2002/
          letter_2002.pdf
        2009/
          letter_2009.pdf
      processed/
    phase_1/
      extract_letters_gpt.py  <-- this script

So if this script is at phase_1/extract_letters_gpt.py,
it will, by default, use ../data as its data_dir.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
import json
import argparse

try:
    from bs4 import BeautifulSoup
    import pdfplumber
except ImportError:
    print("Please install required libraries first:")
    print("  pip install beautifulsoup4 pdfplumber")
    raise


class BuffettLetterExtractor:
    """Extract and clean text from Buffett's shareholder letters (HTML + PDF)."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the extractor.

        If data_dir is None, it is resolved as:
            <this_script_dir>/../data

        Expected structure:
            data/
              raw/
                1977/letter_1977.html
                1996/letter_1996.html
                2002/letter_2002.pdf
                ...
              processed/
        """
        if data_dir is None:
            script_dir = Path(__file__).resolve().parent
            self.data_dir = (script_dir / ".." / "data").resolve()
        else:
            self.data_dir = Path(data_dir).resolve()

        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "text_extracted_letters"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # HTML boilerplate patterns to remove BEFORE parsing
        self.html_boilerplate_patterns = [
            r'<!-- Global site tag.*?</script>',  # Google Analytics
            r'<script.*?</script>',              # All scripts
            r'<style.*?</style>',               # All styles
            r'<head.*?</head>',                 # Whole head section
        ]

        # Common page header/footer patterns (for both HTML text & PDFs)
        self.page_artifacts = [
            r'^\s*\d+\s*$',  # Page numbers alone on a line
            r'^Page \d+ of \d+$',
            r'BERKSHIRE HATHAWAY INC\.\s*$',
            r'^Chairman\'?s Letter\s*$',
            r'^TO THE SHAREHOLDERS OF BERKSHIRE HATHAWAY INC\.?\s*$',
        ]

    # ------------------------------------------------------------------
    #  Extraction
    # ------------------------------------------------------------------

    def extract_from_html(self, filepath: Path) -> str:
        """Extract text from HTML file."""
        print(f"Extracting from HTML: {filepath.relative_to(self.raw_dir)}")

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        # Strip obvious boilerplate/script/style *before* parsing
        for pattern in self.html_boilerplate_patterns:
            html_content = re.sub(
                pattern,
                "",
                html_content,
                flags=re.DOTALL | re.IGNORECASE,
            )

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove any remaining script/style/head/meta/link elements
        for element in soup(["script", "style", "head", "meta", "link"]):
            element.decompose()

        # Get text, keeping some line break structure
        text = soup.get_text(separator="\n")

        return text

    def extract_from_pdf(self, filepath: Path) -> str:
        """Extract text from PDF file, preserving page boundaries."""
        print(f"Extracting from PDF: {filepath.relative_to(self.raw_dir)}")
        text_parts: List[str] = []

        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""

                    if page_text.strip():
                        page_text = page_text.replace("\r\n", "\n").replace("\r", "\n")
                        text_parts.append(page_text.strip())

                    if page_num < len(pdf.pages):
                        text_parts.append("[[PAGE_BREAK]]")

                    if page_num % 10 == 0:
                        print(f"  Processed {page_num} pages...")
        except Exception as e:
            print(f"Error extracting PDF {filepath.name}: {e}")
            return ""

        return "\n\n".join(text_parts)

    # ------------------------------------------------------------------
    #  Cleaning
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving neutral structure
        (page breaks, paragraph breaks).
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        lines = text.split("\n")
        cleaned_lines: List[str] = []
        blank_streak = 0

        for line in lines:
            raw = line

            # Preserve explicit page break markers as their own lines
            if raw.strip() == "[[PAGE_BREAK]]":
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                cleaned_lines.append("[[PAGE_BREAK]]")
                cleaned_lines.append("")
                blank_streak = 1
                continue

            stripped = raw.strip()

            # blank lines → at most one
            if not stripped:
                blank_streak += 1
                if blank_streak == 1:
                    cleaned_lines.append("")
                continue
            else:
                blank_streak = 0

            # Remove page headers/footers & page numbers
            skip_line = False
            for pattern in self.page_artifacts:
                if re.match(pattern, stripped, re.IGNORECASE):
                    skip_line = True
                    break
            if skip_line:
                continue

            # Remove control chars
            stripped = re.sub(
                r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", stripped
            )

            # Normalize internal whitespace
            stripped = re.sub(r"\s+", " ", stripped)

            cleaned_lines.append(stripped)

        cleaned = "\n".join(cleaned_lines)
        cleaned = self._fix_common_errors(cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def _fix_common_errors(self, text: str) -> str:
        """Fix common extraction errors (HTML entities, hyphen breaks, etc.)."""
        replacements = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&nbsp;": " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # join hyphen-broken words: invest- ment -> investment
        text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

        text = re.sub(r" {2,}", " ", text)
        return text

    # ------------------------------------------------------------------
    #  Metadata & Saving
    # ------------------------------------------------------------------

    def extract_year_from_filename(self, filename: str) -> Optional[int]:
        """Extract year from filename (e.g. 'letter_1977.html' → 1977)."""
        match = re.search(r"(19|20)\d{2}", filename)
        return int(match.group(0)) if match else None

    def extract_letter_metadata(self, cleaned_text: str, year: int) -> Dict:
        """Compute simple metadata for a cleaned letter."""
        paragraphs = [p for p in cleaned_text.split("\n\n") if p.strip()]
        words = cleaned_text.split()

        metadata: Dict[str, object] = {
            "year": year,
            "char_count": len(cleaned_text),
            "word_count": len(words),
            "paragraph_count": len(paragraphs),
        }

        lines = [l.strip() for l in cleaned_text.split("\n") if l.strip()]
        if lines:
            for line in lines[:15]:
                if "shareholder" in line.lower() or "berkshire hathaway" in line.lower():
                    metadata["opening_line"] = line
                    break

        return metadata

    def process_file(self, filepath: Path) -> Optional[Dict]:
        """
        Process a single HTML/PDF letter file:
        - extract raw text
        - clean text
        - compute metadata
        """
        year = self.extract_year_from_filename(filepath.name)
        if not year:
            print(f"Warning: Could not extract year from {filepath.name}")
            return None

        if filepath.suffix.lower() in [".html", ".htm"]:
            raw_text = self.extract_from_html(filepath)
        elif filepath.suffix.lower() == ".pdf":
            raw_text = self.extract_from_pdf(filepath)
        else:
            print(f"Unsupported file type: {filepath.suffix}")
            return None

        if not raw_text or not raw_text.strip():
            print(f"Warning: No text extracted from {filepath.name}")
            return None

        cleaned_text = self.clean_text(raw_text)
        metadata = self.extract_letter_metadata(cleaned_text, year)
        metadata["source_file"] = str(filepath.relative_to(self.raw_dir))
        metadata["file_type"] = filepath.suffix.lower()

        print(
            f"  Extracted {metadata['word_count']:,} words, "
            f"{metadata['paragraph_count']} paragraphs",
        )

        return {"text": cleaned_text, "metadata": metadata}

    def save_processed_letter(self, letter_data: Dict, year: int) -> None:
        """
        Save cleaned text and metadata to disk.

        Output:
            data/processed/1977_cleaned.txt
            data/processed/1977_metadata.json
        """
        output_file = self.processed_dir / f"{year}_cleaned.txt"
        metadata_file = self.processed_dir / f"{year}_metadata.json"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(letter_data["text"])

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(letter_data["metadata"], f, indent=2)

        print(f"  Saved to {output_file}")

    def process_all_letters(self) -> Dict[int, Dict]:
        """
        Process all HTML/PDF letters under data/raw/**.

        Works with nested folders like:
            data/raw/1977/letter_1977.html
            data/raw/1996/letter_1996.html
            data/raw/2002/letter_2002.pdf
        """
        html_files = list(self.raw_dir.rglob("*.html")) + list(self.raw_dir.rglob("*.htm"))
        pdf_files = list(self.raw_dir.rglob("*.pdf"))
        files = html_files + pdf_files

        if not files:
            print(f"No HTML/PDF files found under {self.raw_dir}")
            print("Expected structure like: data/raw/1977/letter_1977.html")
            return {}

        print(f"Found {len(files)} files to process (recursive under {self.raw_dir})")
        print("=" * 60)

        processed_letters: Dict[int, Dict] = {}

        for filepath in sorted(files):
            print(f"\nProcessing: {filepath.relative_to(self.raw_dir)}")
            print("-" * 60)

            letter_data = self.process_file(filepath)

            if letter_data:
                year = letter_data["metadata"]["year"]
                processed_letters[year] = letter_data
                self.save_processed_letter(letter_data, year)

        print("\n" + "=" * 60)
        print(f"Successfully processed {len(processed_letters)} letters")

        self._create_summary_report(processed_letters)

        return processed_letters

    def _create_summary_report(self, processed_letters: Dict[int, Dict]) -> None:
        """Create a simple summary report of processed letters."""
        report_file = self.processed_dir / "extraction_summary.json"

        total_words = sum(
            l["metadata"]["word_count"] for l in processed_letters.values()
        )
        total_paragraphs = sum(
            l["metadata"]["paragraph_count"] for l in processed_letters.values()
        )
        total_letters = len(processed_letters) or 1

        summary = {
            "total_letters": len(processed_letters),
            "years_processed": sorted(processed_letters.keys()),
            "total_words": total_words,
            "total_paragraphs": total_paragraphs,
            "average_words_per_letter": total_words // total_letters,
            "letters": {},
        }

        for year, data in sorted(processed_letters.items()):
            md = data["metadata"]
            summary["letters"][year] = {
                "source_file": md["source_file"],
                "word_count": md["word_count"],
                "paragraph_count": md["paragraph_count"],
                "char_count": md["char_count"],
            }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary report saved to {report_file}")
        print("\nTotal Statistics:")
        print(f"  Letters processed: {summary['total_letters']}")
        print(f"  Total words: {summary['total_words']:,}")
        print(f"  Total paragraphs: {summary['total_paragraphs']:,}")
        print(
            f"  Average words per letter: {summary['average_words_per_letter']:,}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buffett letters text extractor")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Base data directory (default: ../data relative to this script). "
            "Inside it should be raw/ and processed/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    print("Buffett Letters Text Extractor (Extraction-Only)")
    print("=" * 60)

    args = parse_args()
    extractor = BuffettLetterExtractor(data_dir=args.data_dir)
    print(f"Using data_dir: {extractor.data_dir}")
    print(f"  raw_dir: {extractor.raw_dir}")
    print(f"  processed_dir: {extractor.processed_dir}")

    processed_letters = extractor.process_all_letters()

    if processed_letters:
        print("\n✓ Extraction complete!")
        print(f"  Cleaned text files saved to: {extractor.processed_dir}")
        print("  Next phase: Section detection & chunking (separate script)")
    else:
        print("\n✗ No letters were processed.")
        print(f"  Please add letter files under: {extractor.raw_dir}/<year>/letter_<year>.html|pdf")


if __name__ == "__main__":
    main()
