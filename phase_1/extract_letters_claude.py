"""
Text Extraction Script for Buffett Letters
Phase 1A: Extract and Clean Text from HTML/PDF sources

This script handles:
1. Text extraction from HTML and PDF files
2. Cleaning (remove headers, footers, page numbers, boilerplate)
3. Initial text normalization
4. Save cleaned text for next processing stage
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional
import json

# Required libraries
try:
    from bs4 import BeautifulSoup
    import pdfplumber
except ImportError:
    print("Installing required libraries...")
    print("Run: pip install beautifulsoup4 pdfplumber")
    raise


class BuffettLetterExtractor:
    """Extract and clean text from Buffett's shareholder letters."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the extractor.
        
        Args:
            data_dir: Base directory containing raw/ and processed/ subdirectories
                     If None, uses parent directory of script location
        """
        # Get the script's directory
        script_dir = Path(__file__).parent
        
        # If data_dir not provided, assume it's in parent directory
        if data_dir is None:
            self.data_dir = script_dir.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "text_extracted_claude"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Script directory: {script_dir}")
        print(f"Data directory: {self.data_dir}")
        print(f"Raw letters directory: {self.raw_dir}")
        print(f"Processed output directory: {self.processed_dir}")
        
        # HTML-specific boilerplate patterns to remove
        self.html_boilerplate_patterns = [
            r'<!-- Global site tag.*?</script>',  # Google Analytics
            r'<script.*?</script>',  # All scripts
            r'<style.*?</style>',   # All styles
            r'<HEAD>.*?</HEAD>',    # Head section
        ]
        
        # Common page header/footer patterns
        self.page_artifacts = [
            r'^\s*\d+\s*$',  # Page numbers alone on a line
            r'^Page \d+ of \d+$',
            r'BERKSHIRE HATHAWAY INC\.\s*$',
            r'^Chairman\'?s Letter\s*$',
            r'^TO THE SHAREHOLDERS OF BERKSHIRE HATHAWAY INC\.?\s*$',
        ]
    
    def extract_from_html(self, filepath: Path) -> str:
        """
        Extract text from HTML file.
        
        Args:
            filepath: Path to HTML file
            
        Returns:
            Extracted text string
        """
        print(f"Extracting from HTML: {filepath.name}")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # Remove script/style tags and other boilerplate
        for pattern in self.html_boilerplate_patterns:
            html_content = re.sub(pattern, '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove remaining script and style elements
        for element in soup(['script', 'style', 'head', 'meta', 'link']):
            element.decompose()
        
        # Get text
        text = soup.get_text()
        
        return text
    
    def extract_from_pdf(self, filepath: Path) -> str:
        """
        Extract text from PDF file.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            Extracted text string
        """
        print(f"Extracting from PDF: {filepath.name}")
        
        text_parts = []
        
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text from page
                    page_text = page.extract_text()
                    
                    if page_text:
                        text_parts.append(page_text)
                        
                    if page_num % 10 == 0:
                        print(f"  Processed {page_num} pages...")
        
        except Exception as e:
            print(f"Error extracting PDF {filepath.name}: {e}")
            return ""
        
        return "\n\n".join(text_parts)
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines (we'll add them back strategically)
            if not line.strip():
                continue
            
            # Remove page artifacts
            skip_line = False
            for pattern in self.page_artifacts:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    skip_line = True
                    break
            
            if skip_line:
                continue
            
            # Clean up the line
            line = line.strip()
            
            # Normalize whitespace
            line = re.sub(r'\s+', ' ', line)
            
            # Remove special characters that are artifacts
            line = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', line)
            
            cleaned_lines.append(line)
        
        # Join lines back together
        text = '\n'.join(cleaned_lines)
        
        # Fix common OCR/extraction errors
        text = self._fix_common_errors(text)
        
        # Normalize paragraph breaks (2+ newlines become exactly 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _fix_common_errors(self, text: str) -> str:
        """
        Fix common extraction errors.
        
        Args:
            text: Text to fix
            
        Returns:
            Fixed text
        """
        # Fix common HTML entities that might have slipped through
        replacements = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&nbsp;': ' ',
            '\r\n': '\n',
            '\r': '\n',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Fix broken words (common in PDF extraction)
        # e.g., "invest- ment" -> "investment"
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # Fix multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text
    
    def extract_year_from_filename(self, filename: str) -> Optional[int]:
        """
        Extract year from filename.
        
        Args:
            filename: Name of file (e.g., "letter_1996.html")
            
        Returns:
            Year as integer, or None if not found
        """
        # Look for 4-digit year
        match = re.search(r'(19|20)\d{2}', filename)
        if match:
            return int(match.group(0))
        return None
    
    def extract_letter_metadata(self, text: str, year: int) -> Dict:
        """
        Extract basic metadata from letter text.
        
        Args:
            text: Cleaned letter text
            year: Year of the letter
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'year': year,
            'char_count': len(text),
            'word_count': len(text.split()),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        }
        
        # Try to find the opening line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            # First meaningful line often starts with "To the Shareholders"
            for line in lines[:10]:
                if 'shareholders' in line.lower() or 'berkshire hathaway' in line.lower():
                    metadata['opening_line'] = line
                    break
        
        return metadata
    
    def process_file(self, filepath: Path) -> Optional[Dict]:
        """
        Process a single letter file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dictionary with extracted text and metadata, or None if processing fails
        """
        # Determine year from filename
        year = self.extract_year_from_filename(filepath.name)
        if not year:
            print(f"Warning: Could not extract year from {filepath.name}")
            return None
        
        # Extract text based on file type
        if filepath.suffix.lower() in ['.html', '.htm']:
            raw_text = self.extract_from_html(filepath)
        elif filepath.suffix.lower() == '.pdf':
            raw_text = self.extract_from_pdf(filepath)
        else:
            print(f"Unsupported file type: {filepath.suffix}")
            return None
        
        if not raw_text:
            print(f"Warning: No text extracted from {filepath.name}")
            return None
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Extract metadata
        metadata = self.extract_letter_metadata(cleaned_text, year)
        metadata['source_file'] = filepath.name
        metadata['file_type'] = filepath.suffix.lower()
        
        print(f"  Extracted {metadata['word_count']:,} words, "
              f"{metadata['paragraph_count']} paragraphs")
        
        return {
            'text': cleaned_text,
            'metadata': metadata
        }
    
    def save_processed_letter(self, letter_data: Dict, year: int):
        """
        Save processed letter to disk.
        
        Args:
            letter_data: Dictionary with text and metadata
            year: Year of the letter
        """
        output_file = self.processed_dir / f"{year}_cleaned.txt"
        metadata_file = self.processed_dir / f"{year}_metadata.json"
        
        # Save text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(letter_data['text'])
        
        # Save metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(letter_data['metadata'], f, indent=2)
        
        print(f"  Saved to {output_file.name}")
    
    def process_all_letters(self):
        """
        Process all letters in the raw directory, including subdirectories.
        
        Returns:
            Dictionary mapping year to processed letter data
        """
        # Find all HTML and PDF files in raw directory and subdirectories
        files = list(self.raw_dir.rglob('*.html')) + \
                list(self.raw_dir.rglob('*.htm')) + \
                list(self.raw_dir.rglob('*.pdf'))
        
        if not files:
            print(f"No files found in {self.raw_dir} or its subdirectories")
            print(f"Please add letter files (HTML or PDF) to this directory.")
            print(f"\nExpected structure:")
            print(f"  {self.raw_dir}/")
            print(f"    1977/")
            print(f"      letter_1977.html")
            print(f"    1996/")
            print(f"      letter_1996.html")
            return {}
        
        print(f"Found {len(files)} files to process")
        print("=" * 60)
        
        processed_letters = {}
        
        for filepath in sorted(files):
            print(f"\nProcessing: {filepath.name}")
            print("-" * 60)
            
            letter_data = self.process_file(filepath)
            
            if letter_data:
                year = letter_data['metadata']['year']
                processed_letters[year] = letter_data
                self.save_processed_letter(letter_data, year)
        
        print("\n" + "=" * 60)
        print(f"Successfully processed {len(processed_letters)} letters")
        
        # Create summary report
        self._create_summary_report(processed_letters)
        
        return processed_letters
    
    def _create_summary_report(self, processed_letters: Dict):
        """
        Create a summary report of processed letters.
        
        Args:
            processed_letters: Dictionary of processed letters
        """
        report_file = self.processed_dir / "extraction_summary.json"
        
        summary = {
            'total_letters': len(processed_letters),
            'years_processed': sorted(processed_letters.keys()),
            'total_words': sum(l['metadata']['word_count'] for l in processed_letters.values()),
            'total_paragraphs': sum(l['metadata']['paragraph_count'] for l in processed_letters.values()),
            'letters': {}
        }
        
        for year, data in sorted(processed_letters.items()):
            summary['letters'][year] = {
                'source_file': data['metadata']['source_file'],
                'word_count': data['metadata']['word_count'],
                'paragraph_count': data['metadata']['paragraph_count'],
                'char_count': data['metadata']['char_count'],
            }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary report saved to {report_file}")
        print(f"\nTotal Statistics:")
        print(f"  Letters processed: {summary['total_letters']}")
        print(f"  Total words: {summary['total_words']:,}")
        print(f"  Total paragraphs: {summary['total_paragraphs']:,}")
        print(f"  Average words per letter: {summary['total_words'] // summary['total_letters']:,}")


def main():
    """Main execution function."""
    print("Buffett Letters Text Extractor")
    print("=" * 60)
    
    # Initialize extractor (automatically finds parent/data directory)
    extractor = BuffettLetterExtractor()
    
    print("\n" + "=" * 60)
    
    # Process all letters
    processed_letters = extractor.process_all_letters()
    
    if processed_letters:
        print("\n✓ Extraction complete!")
        print(f"  Cleaned text files saved to: {extractor.processed_dir}")
        print(f"  Ready for next phase: Section identification and chunking")
    else:
        print("\n✗ No letters were processed.")
        print(f"  Please add letter files to: {extractor.raw_dir}")
        print(f"\n  Expected structure:")
        print(f"    {extractor.raw_dir}/")
        print(f"      1977/letter_1977.html")
        print(f"      1996/letter_1996.html")
        print(f"      2002/letter_2002.pdf")
        print(f"      etc.")


if __name__ == "__main__":
    main()