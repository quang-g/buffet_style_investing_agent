#!/usr/bin/env python3
"""
Chunking Strategy Implementation for Warren Buffett Shareholder Letters
========================================================================
Implements a three-tier hybrid semantic-structural chunking approach for agentic RAG.

Tier 1: Section-level chunks (500-2000 tokens) - Primary retrieval units
Tier 2: Paragraph-level sub-chunks (150-500 tokens) - Precision retrieval
Tier 3: Table/data chunks - Structured retrieval

Usage:
    python chunk_buffett_letters.py
    python chunk_buffett_letters.py --input-dir /path/to/letters --output-dir /path/to/chunks
    python chunk_buffett_letters.py --single-file 2009_cleaned.txt
"""

import re
import json
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime


# =============================================================================
# Configuration
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

TEXT_DIR = PROJECT_ROOT / "data" / "text_extracted_letters"
OUT_DIR = PROJECT_ROOT / "data" / "chunks"

# Chunking parameters
MIN_SECTION_TOKENS = 50
MAX_SECTION_TOKENS = 2500
MIN_PARAGRAPH_TOKENS = 30
TARGET_PARAGRAPH_TOKENS = 300
OVERLAP_SENTENCES = 1

# Approximate tokens per character (for English text)
CHARS_PER_TOKEN = 4


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChunkMetadata:
    """Metadata for each chunk."""
    chunk_id: str
    letter_year: int
    letter_date: Optional[str]
    section_title: Optional[str]
    section_hierarchy: list[str]
    chunk_tier: int
    parent_chunk_id: Optional[str]
    child_chunk_ids: list[str]
    content_type: str  # narrative, table, philosophy, mistake_confession, performance_summary
    has_table: bool
    has_financial_data: bool
    entities: dict  # companies, people, metrics
    themes: list[str]
    temporal_references: dict
    buffett_concepts: list[str]
    token_count: int
    char_count: int
    position_in_letter: int  # Order within letter
    is_overlap: bool = False


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "metadata": asdict(self.metadata)
        }


# =============================================================================
# Pattern Definitions
# =============================================================================

# Section header patterns
SECTION_HEADER_PATTERNS = [
    # Explicit headers (line by itself, title case or all caps)
    r"^([A-Z][A-Za-z\s&,\-']+)$",
    # Headers with specific keywords
    r"^((?:Insurance|Textile|Banking|Manufacturing|Retail|Utility|Finance|Investment)[\w\s&,\-']+)$",
    # "The Annual Meeting" style
    r"^(The [A-Z][A-Za-z\s]+)$",
    # Acquisition/Company specific
    r"^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})$",
]

# Section separator patterns
SEPARATOR_PATTERNS = [
    r"\*\s*\*\s*\*\s*\*\s*\*\s*\*\s*\*\s*\*\s*\*\s*\*\s*\*\s*\*",  # Asterisk separators
    r"\[\[PAGE_BREAK\]\]",  # Explicit page breaks
    r"^[-=]{10,}$",  # Line separators
]

# Table detection patterns
TABLE_PATTERNS = [
    r"(?:^\s*[-\d,.$%]+\s+[-\d,.$%]+\s*$)",  # Numeric columns
    r"(?:^\s*\d{4}\s*\.{3,})",  # Year with dots
    r"(?:^\s*[A-Za-z\s,]+\s+\$[\d,]+)",  # Label with dollar amount
    r"(?:Earnings|Revenue|Float|Premium|Cost|Market)\s*\(in\s*millions?\)",  # Table headers
]

# Buffett concept patterns
BUFFETT_CONCEPTS = {
    "float economics": [r"float", r"insurance float", r"cost.free float"],
    "circle of competence": [r"circle of competence", r"understand the business", r"one that we can understand"],
    "margin of safety": [r"margin of safety", r"attractive price", r"bargain"],
    "moat": [r"economic moat", r"competitive advantage", r"durable advantage"],
    "owner earnings": [r"owner earnings", r"look.through earnings"],
    "Mr. Market": [r"mr\.?\s*market", r"market prices", r"stock market"],
    "value vs price": [r"price is what you pay", r"value is what you get", r"intrinsic value"],
    "long-term focus": [r"long.term", r"forever", r"indefinitely", r"permanent holding"],
    "management quality": [r"management", r"manager", r"CEO", r"operator"],
    "capital allocation": [r"capital allocation", r"deploy capital", r"reinvest"],
    "tailwinds vs headwinds": [r"tailwind", r"headwind"],
    "skin in the game": [r"eat our own cooking", r"our own money", r"personal investment"],
    "return on equity": [r"return on equity", r"ROE", r"return on capital"],
    "compounding": [r"compound", r"compounding", r"compounded"],
    "mistakes and learning": [r"mistake", r"error", r"wrong", r"confession", r"foolish"],
}

# Company name patterns (will be dynamically extended)
KNOWN_COMPANIES = [
    "Berkshire Hathaway", "GEICO", "National Indemnity", "General Re", "See's Candies",
    "Nebraska Furniture Mart", "Borsheim's", "Clayton Homes", "BNSF", "Burlington Northern",
    "Mid American", "Dairy Queen", "Fruit of the Loom", "NetJets", "Net Jets",
    "Blue Chip Stamps", "Wesco", "Coca-Cola", "American Express", "Wells Fargo",
    "Washington Post", "Capital Cities", "ABC", "Gillette", "Kraft", "IBM",
    "Apple", "Bank of America", "Occidental", "Pilot", "Lubrizol", "Precision Castparts",
    "Marmon", "Iscar", "McLane", "Shaw Industries", "Johns Manville", "Acme Brick",
    "Benjamin Moore", "Duracell", "Kraft Heinz", "Heinz",
]

# Key people patterns
KNOWN_PEOPLE = [
    "Warren Buffett", "Charlie Munger", "Ajit Jain", "Greg Abel", "Tony Nicely",
    "Phil Liesche", "Tad Montross", "Dave Sokol", "Kevin Clayton", "Rich Santulli",
    "Bill Gates", "Tom Murphy", "Dan Burke", "Lou Simpson", "Gene Abegg",
    "Chuck Huggins", "Matt Rose", "Tracy Britt Cool",
]


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return len(text) // CHARS_PER_TOKEN


def generate_chunk_id(year: int, section_num: int, tier: int, sub_num: int = 0) -> str:
    """Generate a unique chunk ID."""
    if sub_num > 0:
        return f"{year}-S{section_num:02d}-T{tier}-{sub_num:03d}"
    return f"{year}-S{section_num:02d}-T{tier}"


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename like '1977_cleaned.txt' or '2009.txt'."""
    match = re.search(r"(\d{4})", filename)
    return int(match.group(1)) if match else None


def extract_letter_date(text: str, year: int) -> Optional[str]:
    """Extract the letter date from signature block."""
    # Common patterns: "March 14, 1978", "February 26, 2010"
    patterns = [
        r"([A-Z][a-z]+\s+\d{1,2}\s*,?\s*\d{4})",
        r"(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text[-2000:])  # Check end of letter
        if matches:
            return matches[-1]
    return None


def is_section_header(line: str, prev_line: str, next_line: str) -> bool:
    """Determine if a line is a section header."""
    line = line.strip()
    
    # Must be relatively short
    if len(line) > 80 or len(line) < 3:
        return False
    
    # Should be surrounded by blank lines or be at start
    if prev_line.strip() and next_line.strip():
        return False
    
    # Check against header patterns
    for pattern in SECTION_HEADER_PATTERNS:
        if re.match(pattern, line):
            # Additional checks: not a sentence, no lowercase start words
            words = line.split()
            if len(words) <= 6:  # Headers typically short
                # Most words should be capitalized
                cap_words = sum(1 for w in words if w[0].isupper() or w in ['and', 'of', 'the', 'for', '&'])
                if cap_words >= len(words) * 0.7:
                    return True
    
    return False


def is_separator(line: str) -> bool:
    """Check if line is a section separator."""
    for pattern in SEPARATOR_PATTERNS:
        if re.search(pattern, line.strip()):
            return True
    return False


def detect_table_region(lines: list[str], start_idx: int) -> tuple[int, int]:
    """Detect the boundaries of a table region starting at start_idx."""
    if start_idx >= len(lines):
        return start_idx, start_idx
    
    # Look for table indicators
    table_indicators = 0
    end_idx = start_idx
    
    for i in range(start_idx, min(start_idx + 50, len(lines))):
        line = lines[i]
        
        # Count table-like characteristics
        has_numbers = bool(re.search(r'\$[\d,]+|\d+\.\d+%?|\d{1,3}(?:,\d{3})+', line))
        has_alignment = bool(re.search(r'\s{3,}', line))  # Multiple spaces for alignment
        has_dots = bool(re.search(r'\.{3,}', line))  # Dot leaders
        
        if has_numbers or has_alignment or has_dots:
            table_indicators += 1
            end_idx = i
        elif table_indicators > 0 and line.strip() == "":
            # Allow one blank line within table
            if i + 1 < len(lines) and re.search(r'\$[\d,]+|\d+\.\d+', lines[i + 1]):
                continue
            else:
                break
        elif table_indicators > 2 and not (has_numbers or has_alignment):
            break
    
    if table_indicators >= 3:
        return start_idx, end_idx
    return start_idx, start_idx


def extract_entities(text: str) -> dict:
    """Extract named entities from text."""
    entities = {
        "companies": [],
        "people": [],
        "metrics": []
    }
    
    text_lower = text.lower()
    
    # Extract companies
    for company in KNOWN_COMPANIES:
        if company.lower() in text_lower:
            entities["companies"].append(company)
    
    # Also find companies mentioned with Inc., Corp., etc.
    corp_pattern = r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:Inc\.|Corp\.|Corporation|Company|Co\.)"
    for match in re.finditer(corp_pattern, text):
        company = match.group(1)
        if company not in entities["companies"] and len(company) > 2:
            entities["companies"].append(company)
    
    # Extract people
    for person in KNOWN_PEOPLE:
        if person.lower() in text_lower:
            entities["people"].append(person)
    
    # Extract metrics
    metric_patterns = [
        (r"(\d+(?:\.\d+)?%)", "percentage"),
        (r"\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?", "dollar_amount"),
        (r"(\d+(?:,\d{3})+)\s*(?:employees|people|shareholders)", "headcount"),
    ]
    
    for pattern, metric_type in metric_patterns:
        matches = re.findall(pattern, text)
        if matches:
            entities["metrics"].extend([f"{m} ({metric_type})" for m in matches[:5]])  # Limit
    
    # Deduplicate
    entities["companies"] = list(set(entities["companies"]))
    entities["people"] = list(set(entities["people"]))
    entities["metrics"] = list(set(entities["metrics"]))
    
    return entities


def extract_buffett_concepts(text: str) -> list[str]:
    """Identify Buffett investment concepts mentioned in text."""
    found_concepts = []
    text_lower = text.lower()
    
    for concept, patterns in BUFFETT_CONCEPTS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found_concepts.append(concept)
                break
    
    return found_concepts


def extract_temporal_references(text: str, primary_year: int) -> dict:
    """Extract temporal references from text."""
    temporal = {
        "primary_year": primary_year,
        "mentioned_years": [],
        "has_comparison": False,
        "has_forecast": False
    }
    
    # Find year mentions
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    temporal["mentioned_years"] = sorted(set(int(y) for y in years if int(y) != primary_year))
    
    # Check for comparisons
    comparison_patterns = [r"compared to", r"versus", r"vs\.?", r"year.ago", r"previous year", r"last year"]
    temporal["has_comparison"] = any(re.search(p, text.lower()) for p in comparison_patterns)
    
    # Check for forecasts
    forecast_patterns = [r"expect", r"anticipate", r"forecast", r"outlook", r"next year", r"going forward"]
    temporal["has_forecast"] = any(re.search(p, text.lower()) for p in forecast_patterns)
    
    return temporal


def classify_content_type(text: str, has_table: bool) -> str:
    """Classify the type of content in a chunk."""
    text_lower = text.lower()
    
    if has_table:
        return "financial_data"
    
    # Philosophy/principle statements
    philosophy_indicators = [
        r"we believe", r"our (view|approach|philosophy|policy)",
        r"the (key|important|crucial) (thing|point|lesson)",
        r"rule(s)? (we|i) follow", r"principle",
    ]
    if sum(1 for p in philosophy_indicators if re.search(p, text_lower)) >= 2:
        return "philosophy"
    
    # Mistake confessions
    mistake_indicators = [r"mistake", r"error", r"wrong", r"foolish", r"confession", r"failed"]
    if any(re.search(p, text_lower) for p in mistake_indicators):
        if re.search(r"\b(i|we|my|our)\b", text_lower):
            return "mistake_confession"
    
    # Performance summary
    perf_indicators = [r"earnings", r"net worth", r"book value", r"return on", r"grew", r"increased"]
    if sum(1 for p in perf_indicators if re.search(p, text_lower)) >= 3:
        return "performance_summary"
    
    # Annual meeting info
    if re.search(r"annual meeting|shareholder.+meeting", text_lower):
        return "annual_meeting"
    
    return "narrative"


def identify_themes(text: str) -> list[str]:
    """Identify thematic topics in the text."""
    themes = []
    text_lower = text.lower()
    
    theme_patterns = {
        "insurance": [r"\binsurance\b", r"\bunderwriting\b", r"\bfloat\b", r"\bpremium\b"],
        "acquisitions": [r"\bacquisition\b", r"\bpurchase\b", r"\bacquired\b", r"\bbought\b"],
        "investments": [r"\binvest", r"\bstock\b", r"\bequity\b", r"\bsecurities\b"],
        "management": [r"\bmanager\b", r"\bmanagement\b", r"\bCEO\b", r"\bleader"],
        "capital allocation": [r"\bcapital\b", r"\ballocat", r"\bdeploy"],
        "valuation": [r"\bvalue\b", r"\bvaluation\b", r"\bprice\b", r"\bworth\b"],
        "growth": [r"\bgrowth\b", r"\bgrew\b", r"\bexpand", r"\bincrease"],
        "risk": [r"\brisk\b", r"\buncertain", r"\bvolatil"],
        "dividends": [r"\bdividend", r"\byield\b", r"\bpayout"],
        "debt": [r"\bdebt\b", r"\bborrow", r"\blever", r"\bloan"],
        "taxes": [r"\btax\b", r"\btaxes\b", r"\btaxation"],
        "regulation": [r"\bregulat", r"\bgovernment\b", r"\bpolicy"],
        "competition": [r"\bcompetit", r"\brival", r"\bmarket share"],
        "technology": [r"\btechnolog", r"\bdigital\b", r"\binternet\b", r"\bsoftware"],
        "retail": [r"\bretail\b", r"\bstore\b", r"\bconsumer\b"],
        "manufacturing": [r"\bmanufactur", r"\bfactor", r"\bproduction"],
        "utilities": [r"\butility\b", r"\butilities\b", r"\benergy\b", r"\belectric"],
        "railroads": [r"\brailroad\b", r"\brail\b", r"\bBNSF\b", r"\bBurlington"],
    }
    
    for theme, patterns in theme_patterns.items():
        if any(re.search(p, text_lower) for p in patterns):
            themes.append(theme)
    
    return themes[:5]  # Limit to top 5 themes


# =============================================================================
# Main Chunking Logic
# =============================================================================

class BuffettLetterChunker:
    """Chunks Warren Buffett shareholder letters using a three-tier strategy."""
    
    def __init__(self, text: str, year: int, filename: str):
        self.text = text
        self.year = year
        self.filename = filename
        self.lines = text.split("\n")
        self.letter_date = extract_letter_date(text, year)
        self.chunks: list[Chunk] = []
        self.section_counter = 0
        self.position_counter = 0
    
    def chunk(self) -> list[Chunk]:
        """Main chunking method."""
        # Step 1: Split into sections
        sections = self._split_into_sections()
        
        # Step 2: Process each section
        for section in sections:
            self._process_section(section)
        
        # Step 3: Link parent-child relationships
        self._link_chunks()
        
        return self.chunks
    
    def _split_into_sections(self) -> list[dict]:
        """Split the letter into logical sections."""
        sections = []
        current_section = {
            "title": "Opening",
            "lines": [],
            "start_idx": 0,
            "has_table": False
        }
        
        i = 0
        while i < len(self.lines):
            line = self.lines[i]
            prev_line = self.lines[i - 1] if i > 0 else ""
            next_line = self.lines[i + 1] if i < len(self.lines) - 1 else ""
            
            # Check for section header
            if is_section_header(line, prev_line, next_line):
                # Save current section if it has content
                if current_section["lines"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": line.strip(),
                    "lines": [],
                    "start_idx": i,
                    "has_table": False
                }
                i += 1
                continue
            
            # Check for separator (creates implicit section break)
            if is_separator(line):
                if current_section["lines"]:
                    # Check if next content looks like a new topic
                    # For now, keep in same section but mark the separator
                    current_section["lines"].append(line)
                i += 1
                continue
            
            # Check for table region
            if re.search(r'\$[\d,]+|^\s*\d{4}\s*\.{3,}', line):
                table_start, table_end = detect_table_region(self.lines, i)
                if table_end > table_start:
                    current_section["has_table"] = True
            
            # Check for page break
            if "[[PAGE_BREAK]]" in line:
                current_section["lines"].append(line)
                i += 1
                continue
            
            current_section["lines"].append(line)
            i += 1
        
        # Don't forget the last section
        if current_section["lines"]:
            sections.append(current_section)
        
        return sections
    
    def _process_section(self, section: dict):
        """Process a single section into chunks."""
        self.section_counter += 1
        section_text = "\n".join(section["lines"])
        
        # Skip very short sections (likely artifacts)
        if estimate_tokens(section_text) < MIN_SECTION_TOKENS:
            return
        
        # Detect if section has table
        has_table = section["has_table"] or bool(re.search(r'\$[\d,]+\s+\$[\d,]+', section_text))
        
        # Create Tier 1 chunk (section level)
        tier1_id = generate_chunk_id(self.year, self.section_counter, 1)
        tier1_chunk = self._create_chunk(
            content=section_text,
            chunk_id=tier1_id,
            section_title=section["title"],
            tier=1,
            has_table=has_table,
            parent_id=None
        )
        self.chunks.append(tier1_chunk)
        
        # Create Tier 2 chunks (paragraph level) if section is long enough
        if estimate_tokens(section_text) > TARGET_PARAGRAPH_TOKENS * 2:
            paragraphs = self._split_into_paragraphs(section_text)
            for para_num, para in enumerate(paragraphs, 1):
                if estimate_tokens(para) >= MIN_PARAGRAPH_TOKENS:
                    tier2_id = generate_chunk_id(self.year, self.section_counter, 2, para_num)
                    tier2_chunk = self._create_chunk(
                        content=para,
                        chunk_id=tier2_id,
                        section_title=section["title"],
                        tier=2,
                        has_table=bool(re.search(r'\$[\d,]+', para)),
                        parent_id=tier1_id
                    )
                    self.chunks.append(tier2_chunk)
        
        # Create Tier 3 chunks for tables
        if has_table:
            tables = self._extract_tables(section_text)
            for table_num, table_data in enumerate(tables, 1):
                tier3_id = generate_chunk_id(self.year, self.section_counter, 3, table_num)
                tier3_chunk = self._create_chunk(
                    content=table_data["text"],
                    chunk_id=tier3_id,
                    section_title=section["title"],
                    tier=3,
                    has_table=True,
                    parent_id=tier1_id,
                    content_type_override="financial_table"
                )
                self.chunks.append(tier3_chunk)
    
    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs, preserving logical units."""
        # Split on double newlines
        raw_paragraphs = re.split(r'\n\s*\n', text)
        
        paragraphs = []
        current_para = []
        current_tokens = 0
        
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = estimate_tokens(para)
            
            # Keep numbered lists together
            if re.match(r'^\(\d+\)', para) or re.match(r'^\d+\.', para):
                if current_para:
                    # Check if this continues a list
                    if re.search(r'\(\d+\)|\d+\.\s', current_para[-1]):
                        current_para.append(para)
                        current_tokens += para_tokens
                        continue
            
            # If adding this would exceed target and we have content, save current
            if current_tokens > 0 and current_tokens + para_tokens > TARGET_PARAGRAPH_TOKENS * 1.5:
                paragraphs.append("\n\n".join(current_para))
                current_para = [para]
                current_tokens = para_tokens
            else:
                current_para.append(para)
                current_tokens += para_tokens
        
        if current_para:
            paragraphs.append("\n\n".join(current_para))
        
        return paragraphs
    
    def _extract_tables(self, text: str) -> list[dict]:
        """Extract table regions from text."""
        tables = []
        lines = text.split("\n")
        
        i = 0
        while i < len(lines):
            # Look for table start indicators
            line = lines[i]
            
            # Check for table header patterns
            if re.search(r'(?:Earnings|Revenue|Float|Premium|Cost|Market|Shares|Company)\s*(?:\(|$)', line):
                table_start, table_end = detect_table_region(lines, i)
                if table_end > table_start + 2:
                    table_text = "\n".join(lines[table_start:table_end + 1])
                    tables.append({
                        "text": table_text,
                        "start_line": table_start,
                        "end_line": table_end
                    })
                    i = table_end + 1
                    continue
            
            # Check for numeric table patterns
            if re.search(r'^\s*\d{4}\s*\.{3,}', line) or re.search(r'\$[\d,]+\s+\$[\d,]+', line):
                table_start, table_end = detect_table_region(lines, max(0, i - 2))
                if table_end > table_start + 2:
                    table_text = "\n".join(lines[table_start:table_end + 1])
                    if table_text not in [t["text"] for t in tables]:
                        tables.append({
                            "text": table_text,
                            "start_line": table_start,
                            "end_line": table_end
                        })
                    i = table_end + 1
                    continue
            
            i += 1
        
        return tables
    
    def _create_chunk(
        self,
        content: str,
        chunk_id: str,
        section_title: str,
        tier: int,
        has_table: bool,
        parent_id: Optional[str],
        content_type_override: Optional[str] = None
    ) -> Chunk:
        """Create a chunk with full metadata."""
        self.position_counter += 1
        
        # Extract metadata
        entities = extract_entities(content)
        concepts = extract_buffett_concepts(content)
        temporal = extract_temporal_references(content, self.year)
        themes = identify_themes(content)
        
        # Determine content type
        content_type = content_type_override or classify_content_type(content, has_table)
        
        # Build section hierarchy
        hierarchy = ["Letter Body"]
        if section_title:
            hierarchy.append(section_title)
        
        # Check for financial data
        has_financial = has_table or bool(re.search(r'\$[\d,]+|[\d.]+%', content))
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            letter_year=self.year,
            letter_date=self.letter_date,
            section_title=section_title,
            section_hierarchy=hierarchy,
            chunk_tier=tier,
            parent_chunk_id=parent_id,
            child_chunk_ids=[],
            content_type=content_type,
            has_table=has_table,
            has_financial_data=has_financial,
            entities=entities,
            themes=themes,
            temporal_references=temporal,
            buffett_concepts=concepts,
            token_count=estimate_tokens(content),
            char_count=len(content),
            position_in_letter=self.position_counter,
            is_overlap=False
        )
        
        return Chunk(content=content, metadata=metadata)
    
    def _link_chunks(self):
        """Link parent-child relationships between chunks."""
        chunk_map = {c.metadata.chunk_id: c for c in self.chunks}
        
        for chunk in self.chunks:
            if chunk.metadata.parent_chunk_id:
                parent_id = chunk.metadata.parent_chunk_id
                if parent_id in chunk_map:
                    chunk_map[parent_id].metadata.child_chunk_ids.append(chunk.metadata.chunk_id)


# =============================================================================
# File Processing
# =============================================================================

def process_letter_file(filepath: Path) -> list[Chunk]:
    """Process a single letter file and return chunks."""
    # Read file
    text = filepath.read_text(encoding="utf-8")
    
    # Extract year from filename
    year = extract_year_from_filename(filepath.name)
    if not year:
        print(f"Warning: Could not extract year from {filepath.name}, skipping")
        return []
    
    # Chunk the letter
    chunker = BuffettLetterChunker(text, year, filepath.name)
    chunks = chunker.chunk()
    
    return chunks


def process_all_letters(input_dir: Path, output_dir: Path):
    """Process all letter files in input directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all text files
    letter_files = sorted(input_dir.glob("*.txt"))
    
    if not letter_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    all_chunks = []
    stats = {
        "total_files": 0,
        "total_chunks": 0,
        "tier1_chunks": 0,
        "tier2_chunks": 0,
        "tier3_chunks": 0,
        "by_year": {}
    }
    
    for filepath in letter_files:
        print(f"Processing {filepath.name}...")
        chunks = process_letter_file(filepath)
        
        if chunks:
            year = chunks[0].metadata.letter_year
            
            # Save individual year file
            year_output = output_dir / f"{year}_chunks.json"
            with open(year_output, "w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)
            
            # Update stats
            stats["total_files"] += 1
            stats["total_chunks"] += len(chunks)
            stats["tier1_chunks"] += sum(1 for c in chunks if c.metadata.chunk_tier == 1)
            stats["tier2_chunks"] += sum(1 for c in chunks if c.metadata.chunk_tier == 2)
            stats["tier3_chunks"] += sum(1 for c in chunks if c.metadata.chunk_tier == 3)
            stats["by_year"][year] = {
                "chunk_count": len(chunks),
                "tier1": sum(1 for c in chunks if c.metadata.chunk_tier == 1),
                "tier2": sum(1 for c in chunks if c.metadata.chunk_tier == 2),
                "tier3": sum(1 for c in chunks if c.metadata.chunk_tier == 3),
            }
            
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks created ({year})")
    
    # Save combined file
    if all_chunks:
        combined_output = output_dir / "all_letters_chunks.json"
        with open(combined_output, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in all_chunks], f, indent=2, ensure_ascii=False)
        
        # Save stats
        stats_output = output_dir / "chunking_stats.json"
        with open(stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"CHUNKING COMPLETE")
        print(f"{'='*60}")
        print(f"Total files processed: {stats['total_files']}")
        print(f"Total chunks created:  {stats['total_chunks']}")
        print(f"  - Tier 1 (sections): {stats['tier1_chunks']}")
        print(f"  - Tier 2 (paragraphs): {stats['tier2_chunks']}")
        print(f"  - Tier 3 (tables): {stats['tier3_chunks']}")
        print(f"\nOutput directory: {output_dir}")
        print(f"Combined file: {combined_output}")


def process_single_file(filepath: Path, output_dir: Path):
    """Process a single letter file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {filepath.name}...")
    chunks = process_letter_file(filepath)
    
    if chunks:
        year = chunks[0].metadata.letter_year
        output_file = output_dir / f"{year}_chunks.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Created {len(chunks)} chunks for {year}")
        print(f"  - Tier 1: {sum(1 for c in chunks if c.metadata.chunk_tier == 1)}")
        print(f"  - Tier 2: {sum(1 for c in chunks if c.metadata.chunk_tier == 2)}")
        print(f"  - Tier 3: {sum(1 for c in chunks if c.metadata.chunk_tier == 3)}")
        print(f"Output: {output_file}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chunk Warren Buffett shareholder letters for RAG"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TEXT_DIR,
        help="Directory containing letter text files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for output chunk files"
    )
    parser.add_argument(
        "--single-file",
        type=str,
        default=None,
        help="Process only a single file (filename or path)"
    )
    
    args = parser.parse_args()
    
    if args.single_file:
        # Process single file
        if Path(args.single_file).is_absolute():
            filepath = Path(args.single_file)
        else:
            filepath = args.input_dir / args.single_file
        
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            return 1
        
        process_single_file(filepath, args.output_dir)
    else:
        # Process all files
        if not args.input_dir.exists():
            print(f"Error: Input directory not found: {args.input_dir}")
            return 1
        
        process_all_letters(args.input_dir, args.output_dir)
    
    return 0


if __name__ == "__main__":
    exit(main())
