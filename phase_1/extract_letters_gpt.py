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
        """Extract text from PDF file, preserving page boundaries, with anti-glue fallback."""
        print(f"Extracting from PDF: {filepath.relative_to(self.raw_dir)}")
        text_parts: List[str] = []

        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):

                    # 1) Try a few extract_text configurations (fast path)
                    candidates: List[str] = []
                    try:
                        candidates.append(page.extract_text() or "")
                    except Exception:
                        candidates.append("")

                    # These tolerances often help PDFs where spaces are "implied" by glyph gaps.
                    for xt, yt in [(1.5, 2.0), (2.0, 2.0), (3.0, 3.0)]:
                        try:
                            candidates.append(page.extract_text(x_tolerance=xt, y_tolerance=yt) or "")
                        except Exception:
                            candidates.append("")

                    # 1) Try a few extract_text configurations (fast path)
                    candidates: List[str] = []

                    # Normal extract_text
                    try:
                        candidates.append(page.extract_text() or "")
                    except Exception:
                        candidates.append("")

                    # Optional: layout=True (if supported by your pdfplumber/pdfminer combo)
                    try:
                        candidates.append(page.extract_text(layout=True) or "")
                    except Exception:
                        pass

                    # These tolerances often help PDFs where spaces are "implied" by glyph gaps.
                    for xt, yt in [(1.5, 2.0), (2.0, 2.0), (3.0, 3.0)]:
                        try:
                            candidates.append(page.extract_text(x_tolerance=xt, y_tolerance=yt) or "")
                        except Exception:
                            candidates.append("")

                    # Pick the best candidate by composite ranking (not just glue_score)
                    best = ""
                    best_rank = 10**18

                    for c in candidates:
                        c = (c or "").replace("\r\n", "\n").replace("\r", "\n").strip()
                        if not c:
                            continue

                        c = self._normalize_pdf_artifacts(c)

                        rank = self._candidate_rank(c)
                        if rank < best_rank:
                            best, best_rank = c, rank

                    # 2) If still looks bad, reconstruct from extract_words (slow but reliable)
                    # Keep your existing fallback behavior, but make it trigger on the composite rank
                    if not best or best_rank > 35:
                        best = self._reconstruct_text_from_words(page)
                        best = self._normalize_pdf_artifacts(best)


                    # 2) If still looks glued, reconstruct from extract_words (slow but reliable)
                    if not best or best_score > 25:
                        best = self._reconstruct_text_from_words(page)

                    if best.strip():
                        text_parts.append(best.strip())

                    if page_num < len(pdf.pages):
                        text_parts.append("[[PAGE_BREAK]]")

                    if page_num % 10 == 0:
                        print(f"  Processed {page_num} pages...")

        except Exception as e:
            print(f"Error extracting PDF {filepath.name}: {e}")
            return ""

        return "\n\n".join(text_parts)


    def _reconstruct_text_from_words(self, page) -> str:
        """
        Build text from pdfplumber's word boxes, forcing spaces between words.
        This is the most robust way to defeat glued-words output.
        """
        try:
            words = page.extract_words(
                keep_blank_chars=False,
                use_text_flow=True,   # tends to read in a more natural order
            )
        except Exception:
            words = []

        if not words:
            return ""

        # Group words into lines by their 'top' coordinate with a small tolerance.
        lines: List[List[dict]] = []
        line_tol = 3.0

        # Sort top-to-bottom then left-to-right
        words_sorted = sorted(words, key=lambda w: (round(w.get("top", 0.0), 1), w.get("x0", 0.0)))

        for w in words_sorted:
            top = float(w.get("top", 0.0))
            if not lines:
                lines.append([w])
                continue

            last_top = float(lines[-1][0].get("top", 0.0))
            if abs(top - last_top) <= line_tol:
                lines[-1].append(w)
            else:
                lines.append([w])

        # Within each line, sort left-to-right and join by single space
        out_lines: List[str] = []
        for line in lines:
            line_sorted = sorted(line, key=lambda w: w.get("x0", 0.0))
            line_text = " ".join((w.get("text") or "").strip() for w in line_sorted).strip()
            if line_text:
                out_lines.append(line_text)

        return "\n".join(out_lines)

    def _normalize_pdf_artifacts(self, text: str) -> str:
        """
        Normalize common PDF extraction artifacts:
        - (cid: 129) style bullets
        - weird bullet glyphs like ‹
        """
        if not text:
            return text

        # Replace "(cid: 123)" artifacts with a bullet
        text = re.sub(r"\(\s*cid\s*:\s*\d+\s*\)", "•", text, flags=re.IGNORECASE)

        # Normalize common weird bullet-like glyphs
        # (You can extend this list as you encounter more)
        bulletish = [
            "‹", "›", "·", "•", "◦", "▪", "▫", "", "", "‣", "⁃", "–", "—",
        ]
        for ch in bulletish:
            # Keep hyphens/dashes that are used as punctuation; only normalize when used like bullets.
            # Heuristic: line starts with bullet-ish + space
            text = re.sub(rf"(?m)^\s*{re.escape(ch)}\s+", "• ", text)

        return text

    def _candidate_metrics(self, text: str) -> dict:
        """
        Compute multiple signals for candidate ranking:
        - glue score
        - whitespace density
        - average token length
        - cid artifact count
        """
        t = (text or "").strip()
        if not t:
            return {
                "glue": 10**9,
                "space_ratio": 0.0,
                "avg_token_len": 999.0,
                "cid_count": 999,
                "len": 0,
            }

        cid_count = len(re.findall(r"\(\s*cid\s*:\s*\d+\s*\)", t, flags=re.IGNORECASE))
        spaces = t.count(" ")
        chars = len(t)
        space_ratio = spaces / max(chars, 1)

        tokens = re.findall(r"\S+", t)
        avg_token_len = (sum(len(x) for x in tokens) / max(len(tokens), 1)) if tokens else 999.0

        return {
            "glue": self._glue_score(t),
            "space_ratio": space_ratio,
            "avg_token_len": avg_token_len,
            "cid_count": cid_count,
            "len": chars,
        }

    def _candidate_rank(self, text: str) -> float:
        """
        Lower is better.
        Combines glue patterns + whitespace density + avg token length + cid artifacts.
        """
        m = self._candidate_metrics(text)

        # Penalties:
        # - glue is primary (already includes some whitespace penalty)
        # - CID artifacts are strong negative
        # - too-low whitespace ratio is negative
        # - unusually large avg token length is negative (glued words inflate it)
        glue = m["glue"]
        cid = m["cid_count"]
        space_ratio = m["space_ratio"]
        avg_len = m["avg_token_len"]

        space_penalty = max(0.0, 0.12 - space_ratio) * 500.0
        avg_len_penalty = max(0.0, avg_len - 6.5) * 15.0
        cid_penalty = cid * 25.0

        return float(glue) + cid_penalty + space_penalty + avg_len_penalty


    def _glue_score(self, text: str) -> int:
        """
        Heuristic: count patterns that strongly suggest missing spaces.
        Higher score => more 'glued' text.
        """
        if not text:
            return 10**9

        score = 0

        # Penalize extremely low whitespace density
        spaces = text.count(" ")
        chars = max(len(text), 1)
        space_ratio = spaces / chars
        if space_ratio < 0.06:
            score += int((0.06 - space_ratio) * 1000)

        # Count classic glue patterns
        patterns = [
            r"[a-z][A-Z]",           # camelCase-like merges
            r"[A-Za-z]\d",           # letter followed by digit, e.g. "gain2009"
            r"\d[A-Za-z]",           # digit followed by letter, e.g. "21.8billion"
            r"[.,;:!?][A-Za-z]",     # punctuation immediately followed by letter (missing space)
        ]
        for p in patterns:
            score += len(re.findall(p, text))

        # --- NEW: suspicious long lowercase tokens (e.g. "recentlypurchased") ---
        # Tokenize on whitespace
        tokens = re.findall(r"\S+", text)

        long_lower_glued = 0
        for tok in tokens:
            # Remove surrounding punctuation but keep internal letters
            core = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", tok)
            if not core:
                continue

            # all letters
            if not core.isalpha():
                continue

            # length threshold (tunable: 14–18); use 16 as a good default
            if len(core) < 16:
                continue

            # mostly lowercase (e.g., >= 90% lowercase letters)
            lowers = sum(1 for c in core if c.islower())
            letters = len(core)
            if letters == 0:
                continue
            if (lowers / letters) < 0.90:
                continue

            # not a "normal" long word pattern (start without dictionary):
            # allow some "normal" patterns to pass with low penalty:
            # - endswith common suffixes (optional): "tion", "ment", "ness", "able", etc.
            # We'll still count them, but with smaller penalty.
            long_lower_glued += 1

        # Each suspicious token adds meaningful penalty
        score += long_lower_glued * 8

        # --- NEW: CID artifacts are a strong signal of bad extraction ---
        score += len(re.findall(r"\(\s*cid\s*:\s*\d+\s*\)", text, flags=re.IGNORECASE)) * 10

        return score



    def _fix_common_errors(self, text: str) -> str:
        """Fix common extraction errors (entities, hyphens, missing spaces)."""
        replacements = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&nbsp;": " ",
        }
        text = self._normalize_pdf_artifacts(text)

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Join hyphen-broken words across line breaks/spaces: invest- ment -> investment
        # Only remove hyphen when it is clearly a line-break hyphenation.
        # Keeps true compounds like "cash-equivalent" intact.
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)


        # --- GUARANTEED local repair for glued words ---
        # Add space after punctuation when followed by a letter/number
        text = re.sub(r"([.,;:!?])([A-Za-z0-9])", r"\1 \2", text)

        # Split letter<->digit boundaries
        text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)

        # Split lower->Upper boundaries (helps some merged tokens)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        # Fix common PDF numeric spacing: "21. 8" -> "21.8", "84, 487" -> "84,487"
        text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)
        text = re.sub(r"(\d),\s+(\d)", r"\1,\2", text)

        # Normalize spaces again
        text = re.sub(r"[ \t]{2,}", " ", text)

        return text


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

    # def _fix_common_errors(self, text: str) -> str:
    #     """Fix common extraction errors (HTML entities, hyphen breaks, etc.)."""
    #     replacements = {
    #         "&amp;": "&",
    #         "&lt;": "<",
    #         "&gt;": ">",
    #         "&quot;": '"',
    #         "&#39;": "'",
    #         "&nbsp;": " ",
    #     }
    #     for old, new in replacements.items():
    #         text = text.replace(old, new)

    #     # join hyphen-broken words: invest- ment -> investment
    #     text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    #     text = re.sub(r" {2,}", " ", text)
    #     return text

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
