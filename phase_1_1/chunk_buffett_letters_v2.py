#!/usr/bin/env python3
"""
Chunking Strategy Implementation for Warren Buffett Shareholder Letters (v2)
=============================================================================
CORRECTED VERSION - Fixes the following issues from v1:
1. Content duplication between Tier 1 and Tier 2 (now mutually exclusive)
2. Broken section header detection (table rows no longer become sections)
3. False table classification (proper table detection)
4. Inconsistent section counts (improved header patterns)
5. Table rows as section titles (blacklist common table terms)

Chunking Strategy:
- Short sections (<600 tokens): Single chunk with full content
- Long sections (>600 tokens): Split into paragraph chunks (no duplication)
- Tables: Extracted as separate structured chunks

Usage:
    python chunk_buffett_letters_v2.py
    python chunk_buffett_letters_v2.py --input-dir /path/to/letters --output-dir /path/to/chunks
    python chunk_buffett_letters_v2.py --single-file 2009_cleaned.txt
"""

import re
import json
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
MIN_CHUNK_TOKENS = 50
MAX_CHUNK_TOKENS = 1500
LONG_SECTION_THRESHOLD = 600  # Sections above this get split into paragraphs
TARGET_PARAGRAPH_TOKENS = 400
OVERLAP_CHARS = 100  # Small overlap for context continuity

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
    chunk_type: str  # "section", "paragraph", "table"
    parent_section_id: Optional[str]  # For paragraphs, points to conceptual section
    sibling_chunk_ids: list[str]  # Other chunks from same section
    content_type: str  # narrative, table, philosophy, mistake_confession, performance_summary
    has_table: bool
    has_financial_data: bool
    entities: dict  # companies, people, metrics
    themes: list[str]
    temporal_references: dict
    buffett_concepts: list[str]
    token_count: int
    char_count: int
    position_in_letter: int
    section_position: int  # Position within section (for paragraphs)
    total_section_chunks: int  # Total chunks in this section


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

# Table-related terms that should NEVER be section headers
TABLE_TERM_BLACKLIST = {
    # Common table headers/footers
    'total', 'subtotal', 'net', 'gross', 'amount', 'balance',
    'pre-tax', 'after-tax', 'pre-tax earnings', 'after-tax earnings',
    # Financial terms that appear as column headers
    'earnings', 'revenue', 'income', 'loss', 'gain', 'profit',
    'premium', 'float', 'cost', 'market', 'book value',
    'in-force', 'year-end', 'yearend',
    # Table structure words
    'year', 'company', 'shares', 'per share', 'per-share',
    'investments', 'investments excluded', 'sources', 'uses',
    # Numeric indicators
    'millions', 'billions', 'thousands', '(in millions)', '(in billions)',
}

# Patterns that indicate a line is part of a table, not a header
TABLE_CONTENT_PATTERNS = [
    r'^\s*\$[\d,]+',  # Starts with dollar amount
    r'^\s*[\d,]+\s*$',  # Just numbers
    r'^\s*\d+\.\d+%?\s*$',  # Decimal/percentage
    r'\.{3,}',  # Dot leaders (.....)
    r'^\s*\(\d+\)\s*$',  # Footnote markers like (1)
    r'^\s*[-–—]+\s*$',  # Dashes
    r'^\s*\*+\s*$',  # Asterisks alone
    r'\$[\d,]+\s+\$[\d,]+',  # Multiple dollar amounts (table row)
    r'^\s*\d{4}\s*\.{2,}',  # Year followed by dots
    r'[\d,]+\s{2,}[\d,]+',  # Numbers with spacing (aligned columns)
]

# Section header patterns - more restrictive
SECTION_HEADER_PATTERNS = [
    # Business segment headers
    r'^(Insurance Underwriting)$',
    r'^(Insurance Investments)$',
    r'^(Insurance Operations)$',
    r'^(Insurance)$',
    r'^(Textile Operations)$',
    r'^(Banking)$',
    r'^(Blue Chip Stamps)$',
    r'^(Regulated[,]? Capital-Intensive Businesses)$',
    r'^(Regulated Utility Business)$',
    r'^(Manufacturing, Service and Retailing Operations)$',
    r'^(Finance and Financial Products)$',
    r'^(Investments)$',
    r'^(Acquisition Criteria)$',
    r'^(ACQUISITION CRITERIA)$',
    
    # Common section headers across years
    r'^(The Annual Meeting)$',
    r'^(Annual Meeting)$',
    r'^(Shareholder-Designated Contributions)$',
    r'^(Look-Through Earnings)$',
    r'^(Sources of Reported Earnings)$',
    r'^(Acquisition Accounting)$',
    r'^(Share Repurchases)$',
    r'^(The Managers and Directors)$',
    r'^(Corporate Governance)$',
    r'^(An Inconvenient Truth.*)$',
    
    # Additional common headers from Buffett letters
    r'^(How We Measure Ourselves)$',
    r'^(What We Don\'t Do)$',
    r'^(What We Do)$',
    r'^(Intrinsic Value)$',
    r'^(Derivatives)$',
    r'^(Investments)$',
    r'^(Common Stock Investments)$',
    r'^(The Year at Berkshire)$',
    r'^(Berkshire Today)$',
    r'^(Charlie Straightens Me Out)$',
    r'^(Operating Businesses)$',
    r'^(GEICO)$',
    r'^(General Re)$',
    r'^(NetJets)$',
    r'^(Clayton Homes)$',
    r'^(Utilities)$',
    r'^(Retailing)$',
    r'^(Flight Services)$',
    r'^(Aviation Services)$',
    
    # Generic patterns (more restrictive)
    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\s+Operations)$',
    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\s+Business)$',
    r'^([A-Z][a-z]+\s+and\s+[A-Z][a-z]+)$',  # "Finance and Products"
]

# Implicit section break patterns (start of new topic)
IMPLICIT_SECTION_PATTERNS = [
    r'^Let\'?s (?:move|turn|now look) to',
    r'^Now (?:let\'?s|I\'?ll|we\'?ll)',
    r'^Finally,? (?:we|I|let me)',
    r'^Our \w+ operations? (?:continued|had|produced|generated)',
    r'^\* \* \* \* \* \* \* \* \* \* \* \*',  # Asterisk separator
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
    "management quality": [r"outstanding manager", r"exceptional manager", r"talented manager"],
    "capital allocation": [r"capital allocation", r"deploy capital", r"reinvest"],
    "tailwinds vs headwinds": [r"tailwind", r"headwind"],
    "skin in the game": [r"eat our own cooking", r"our own money", r"personal investment"],
    "return on equity": [r"return on equity", r"ROE", r"return on capital"],
    "compounding": [r"compound", r"compounding", r"compounded annually"],
    "mistakes and learning": [r"mistake", r"error", r"wrong", r"foolish", r"confession"],
}

# Known companies
KNOWN_COMPANIES = [
    "Berkshire Hathaway", "GEICO", "National Indemnity", "General Re", "See's Candies",
    "Nebraska Furniture Mart", "Borsheim's", "Clayton Homes", "BNSF", "Burlington Northern",
    "MidAmerican", "Mid American", "Dairy Queen", "Fruit of the Loom", "NetJets", "Net Jets",
    "Blue Chip Stamps", "Wesco", "Coca-Cola", "American Express", "Wells Fargo",
    "Washington Post", "Capital Cities", "ABC", "Gillette", "Kraft", "IBM",
    "Apple", "Bank of America", "Occidental", "Pilot", "Lubrizol", "Precision Castparts",
    "Marmon", "Iscar", "McLane", "Shaw Industries", "Johns Manville", "Acme Brick",
    "Benjamin Moore", "Duracell", "Kraft Heinz", "Heinz", "FlightSafety", "Executive Jet",
    "Jordan's Furniture", "Star Furniture", "RC Willey", "CORT", "XTRA",
]

# Known people
KNOWN_PEOPLE = [
    "Warren Buffett", "Charlie Munger", "Ajit Jain", "Greg Abel", "Tony Nicely",
    "Phil Liesche", "Tad Montross", "Dave Sokol", "Kevin Clayton", "Rich Santulli",
    "Bill Gates", "Tom Murphy", "Dan Burke", "Lou Simpson", "Gene Abegg",
    "Chuck Huggins", "Matt Rose", "Tracy Britt Cool", "Ted Weschler", "Todd Combs",
]


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return len(text) // CHARS_PER_TOKEN


def generate_chunk_id(year: int, section_num: int, chunk_type: str, sub_num: int = 0) -> str:
    """Generate a unique chunk ID."""
    type_code = {"section": "SEC", "paragraph": "PAR", "table": "TBL"}[chunk_type]
    if sub_num > 0:
        return f"{year}-S{section_num:02d}-{type_code}-{sub_num:03d}"
    return f"{year}-S{section_num:02d}-{type_code}"


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename like '1977_cleaned.txt'."""
    match = re.search(r"(\d{4})", filename)
    return int(match.group(1)) if match else None


def extract_letter_date(text: str, year: int) -> Optional[str]:
    """Extract the letter date from signature block."""
    patterns = [
        r"([A-Z][a-z]+\s+\d{1,2}\s*,?\s*\d{4})",
        r"(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text[-2000:])
        if matches:
            return matches[-1]
    return None


def looks_like_table_content(line: str) -> bool:
    """Check if a line looks like table content."""
    for pattern in TABLE_CONTENT_PATTERNS:
        if re.search(pattern, line):
            return True
    return False


def is_in_table_context(lines: list[str], idx: int, window: int = 3) -> bool:
    """Check if surrounding lines suggest we're in a table."""
    start = max(0, idx - window)
    end = min(len(lines), idx + window + 1)
    
    table_like_count = 0
    for i in range(start, end):
        if i != idx and looks_like_table_content(lines[i]):
            table_like_count += 1
    
    return table_like_count >= 2


def is_section_header(line: str, lines: list[str], idx: int) -> bool:
    """Determine if a line is a section header with improved detection."""
    line_stripped = line.strip()
    
    # Basic rejections
    if not line_stripped or len(line_stripped) < 3 or len(line_stripped) > 80:
        return False
    
    # Reject if in blacklist (case-insensitive)
    if line_stripped.lower() in TABLE_TERM_BLACKLIST:
        return False
    
    # Reject if looks like table content
    if looks_like_table_content(line_stripped):
        return False
    
    # Reject if in table context
    if is_in_table_context(lines, idx):
        return False
    
    # Reject if ends with comma (mid-sentence)
    if line_stripped.endswith(','):
        return False
    
    # Reject if starts with lowercase
    if line_stripped[0].islower():
        return False
    
    # Get surrounding context
    prev_line = lines[idx - 1].strip() if idx > 0 else ""
    next_line = lines[idx + 1].strip() if idx < len(lines) - 1 else ""
    
    # Check against known header patterns FIRST (these don't need blank lines)
    for pattern in SECTION_HEADER_PATTERNS:
        if re.match(pattern, line_stripped, re.IGNORECASE):
            return True
    
    # For generic headers, check surrounding blank lines
    # Headers should be preceded OR followed by blank line
    has_blank_before = (idx == 0) or (prev_line == '') or prev_line.startswith('*')
    has_blank_after = (idx == len(lines) - 1) or (next_line == '')
    
    if not (has_blank_before or has_blank_after):
        return False
    
    # Generic check for title-case short phrases
    words = line_stripped.split()
    if 2 <= len(words) <= 6:
        # Most words should be capitalized
        cap_count = sum(1 for w in words if w[0].isupper() or w.lower() in ['and', 'of', 'the', 'for', '&', 'a', 'in'])
        if cap_count >= len(words) * 0.8:
            # Additional check: not a sentence (no common verb indicators)
            sentence_indicators = ['is', 'are', 'was', 'were', 'have', 'has', 'had', 
                                  'will', 'would', 'can', 'could', 'should', 'may',
                                  'that', 'which', 'who', 'when', 'where', 'because']
            if not any(w.lower() in sentence_indicators for w in words):
                # Also check it doesn't end like a sentence continuation
                if not line_stripped.endswith('.') or len(words) <= 4:
                    return True
    
    return False


def is_separator_line(line: str) -> bool:
    """Check if line is a section separator (asterisks, etc.)."""
    line = line.strip()
    if re.match(r'^\*[\s\*]+\*$', line):  # * * * * * * *
        return True
    if re.match(r'^[-=]{10,}$', line):  # ----------
        return True
    if line == '[[PAGE_BREAK]]':
        return True
    return False


def detect_table_block(lines: list[str], start_idx: int) -> tuple[int, int, str]:
    """
    Detect a table block starting near start_idx.
    Returns (start, end, table_text) or (start_idx, start_idx, "") if no table.
    """
    # Look for table indicators
    table_lines = []
    in_table = False
    table_start = start_idx
    
    for i in range(start_idx, min(start_idx + 50, len(lines))):
        line = lines[i]
        
        if looks_like_table_content(line) or re.search(r'\s{3,}\S', line):  # Aligned content
            if not in_table:
                table_start = i
                in_table = True
            table_lines.append(line)
        elif in_table:
            # Allow one blank line within table
            if line.strip() == "" and i + 1 < len(lines):
                if looks_like_table_content(lines[i + 1]):
                    table_lines.append(line)
                    continue
            # End of table
            break
    
    if len(table_lines) >= 3:
        return table_start, table_start + len(table_lines) - 1, "\n".join(table_lines)
    return start_idx, start_idx, ""


def has_actual_table(text: str) -> bool:
    """Detect if text contains an actual table (not just dollar amounts in narrative)."""
    lines = text.split('\n')
    
    # Count lines that look like table rows (multiple aligned values)
    table_row_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Table rows typically have:
        # 1. Multiple numeric values with spacing
        # 2. Dot leaders (......)
        # 3. Year entries with dots (1965 . . . . . . 23.8)
        
        # Check for multiple dollar amounts on same line
        dollar_matches = re.findall(r'\$[\d,]+\.?\d*', line)
        if len(dollar_matches) >= 2:
            table_row_count += 1
            continue
        
        # Check for dot leaders
        if re.search(r'\.{5,}', line):
            table_row_count += 1
            continue
        
        # Check for year-style entries (1965 . . . 23.8)
        if re.match(r'^\d{4}\s*\.', line):
            table_row_count += 1
            continue
        
        # Check for multiple numbers separated by significant whitespace
        if re.search(r'\d+\.?\d*\s{3,}\d+\.?\d*\s{3,}\d+\.?\d*', line):
            table_row_count += 1
            continue
    
    # Need at least 4 table-like rows to consider it a table
    return table_row_count >= 4


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
    
    # Extract people
    for person in KNOWN_PEOPLE:
        if person.lower() in text_lower:
            entities["people"].append(person)
    
    # Extract key metrics (be selective)
    # Only extract standalone percentages and large dollar amounts
    percentages = re.findall(r'(\d+\.?\d*%)', text)
    entities["metrics"].extend(percentages[:5])
    
    dollar_amounts = re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|M|B)?', text)
    for amount, unit in dollar_amounts[:5]:
        if unit:
            entities["metrics"].append(f"${amount} {unit}")
        elif ',' in amount or len(amount) > 4:  # Significant amounts only
            entities["metrics"].append(f"${amount}")
    
    # Deduplicate
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
    
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
    
    # Find year mentions (limit to reasonable range)
    years = re.findall(r"\b(19[6-9]\d|20[0-2]\d)\b", text)
    mentioned = sorted(set(int(y) for y in years if int(y) != primary_year))
    temporal["mentioned_years"] = mentioned[:10]  # Limit
    
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
        return "financial_table"
    
    # Philosophy/principle statements
    philosophy_indicators = [
        r"we believe", r"our (view|approach|philosophy|policy)",
        r"the (key|important|crucial) (thing|point|lesson)",
        r"rule(s)? (we|i) follow", r"principle",
    ]
    if sum(1 for p in philosophy_indicators if re.search(p, text_lower)) >= 2:
        return "philosophy"
    
    # Mistake confessions
    if re.search(r"\b(i|we|my|our)\b.{0,30}\b(mistake|error|wrong|foolish|failed)\b", text_lower):
        return "mistake_confession"
    
    # Performance summary
    perf_indicators = [r"net worth", r"book value", r"per.share", r"operating earnings"]
    if sum(1 for p in perf_indicators if re.search(p, text_lower)) >= 2:
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
        "acquisitions": [r"\bacquisition\b", r"\bpurchase[d]?\b", r"\bacquired\b"],
        "investments": [r"\binvest(?:ment|ed|ing)?\b", r"\bstock\b", r"\bequity\b"],
        "management": [r"\bmanager\b", r"\bmanagement\b", r"\bCEO\b"],
        "capital allocation": [r"\bcapital\b.{0,20}\ballocat"],
        "valuation": [r"\bintrinsic value\b", r"\bvaluation\b"],
        "growth": [r"\bgrowth\b", r"\bgrew\b", r"\bexpand"],
        "risk": [r"\brisk\b", r"\buncertain"],
        "dividends": [r"\bdividend"],
        "utilities": [r"\butility\b", r"\butilities\b", r"\benergy\b", r"\belectric"],
        "railroads": [r"\brailroad\b", r"\brail\b", r"\bBNSF\b"],
        "retail": [r"\bretail\b", r"\bstore\b", r"\bfurniture\b"],
        "manufacturing": [r"\bmanufactur"],
    }
    
    for theme, patterns in theme_patterns.items():
        if any(re.search(p, text_lower) for p in patterns):
            themes.append(theme)
    
    return themes[:5]


# =============================================================================
# Main Chunking Logic
# =============================================================================

class BuffettLetterChunkerV2:
    """Improved chunker for Warren Buffett shareholder letters."""
    
    def __init__(self, text: str, year: int, filename: str):
        self.text = text
        self.year = year
        self.filename = filename
        # Normalize line endings and split
        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
        self.lines = normalized_text.split("\n")
        self.letter_date = extract_letter_date(text, year)
        self.chunks: list[Chunk] = []
        self.section_counter = 0
        self.position_counter = 0
    
    def chunk(self) -> list[Chunk]:
        """Main chunking method."""
        # Step 1: Split into sections
        sections = self._split_into_sections()
        
        # Step 2: Process each section (NO DUPLICATION)
        for section in sections:
            self._process_section(section)
        
        # Step 3: Update sibling references
        self._update_sibling_references()
        
        return self.chunks
    
    def _split_into_sections(self) -> list[dict]:
        """Split the letter into logical sections with improved detection."""
        sections = []
        current_section = {
            "title": "Opening",
            "lines": [],
            "start_idx": 0,
        }
        
        i = 0
        while i < len(self.lines):
            line = self.lines[i]
            
            # Check for explicit section header
            if is_section_header(line, self.lines, i):
                # Save current section if it has content
                if current_section["lines"]:
                    sections.append(current_section)
                
                current_section = {
                    "title": line.strip(),
                    "lines": [],
                    "start_idx": i,
                }
                i += 1
                continue
            
            # Check for separator (potential section break)
            if is_separator_line(line):
                # Check if next non-empty line could be a section start
                next_content_idx = i + 1
                while next_content_idx < len(self.lines) and not self.lines[next_content_idx].strip():
                    next_content_idx += 1
                
                if next_content_idx < len(self.lines):
                    next_line = self.lines[next_content_idx]
                    # Check for implicit section patterns
                    for pattern in IMPLICIT_SECTION_PATTERNS:
                        if re.match(pattern, next_line.strip(), re.IGNORECASE):
                            if current_section["lines"]:
                                sections.append(current_section)
                            current_section = {
                                "title": self._infer_section_title(next_line),
                                "lines": [],
                                "start_idx": next_content_idx,
                            }
                            break
                
                # Don't add separator line to content
                i += 1
                continue
            
            # Skip page breaks but don't add them
            if '[[PAGE_BREAK]]' in line:
                i += 1
                continue
            
            current_section["lines"].append(line)
            i += 1
        
        # Don't forget last section
        if current_section["lines"]:
            sections.append(current_section)
        
        return sections
    
    def _infer_section_title(self, first_line: str) -> str:
        """Infer a section title from the first line of content."""
        line = first_line.strip()
        
        # Common patterns
        if re.match(r"^Let'?s (?:move|turn) to (.+)", line, re.IGNORECASE):
            match = re.match(r"^Let'?s (?:move|turn) to (.+)", line, re.IGNORECASE)
            return match.group(1).rstrip('.')[:50]
        
        if re.match(r"^Our (\w+(?:\s+\w+)?)\s+(?:operation|business)", line, re.IGNORECASE):
            match = re.match(r"^Our (\w+(?:\s+\w+)?)", line, re.IGNORECASE)
            return match.group(1).title()
        
        # Default: use first few words
        words = line.split()[:4]
        return " ".join(words)[:40] + "..."
    
    def _process_section(self, section: dict):
        """Process a section WITHOUT creating duplicate content."""
        self.section_counter += 1
        section_text = "\n".join(section["lines"]).strip()
        
        # Skip empty or very short sections
        if not section_text or estimate_tokens(section_text) < MIN_CHUNK_TOKENS:
            return
        
        section_tokens = estimate_tokens(section_text)
        section_id = f"{self.year}-S{self.section_counter:02d}"
        
        # Extract tables first
        table_chunks = self._extract_and_create_table_chunks(section, section_id)
        
        # Remove table content from section text for paragraph processing
        section_text_no_tables = self._remove_table_content(section_text)
        section_tokens_no_tables = estimate_tokens(section_text_no_tables)
        
        if section_tokens_no_tables < MIN_CHUNK_TOKENS:
            # Only tables in this section, already processed
            return
        
        # Decision: single chunk or split into paragraphs
        if section_tokens_no_tables <= LONG_SECTION_THRESHOLD:
            # SHORT SECTION: Create single chunk
            self._create_single_chunk(section_text_no_tables, section, section_id)
        else:
            # LONG SECTION: Split into paragraph chunks (NO separate section chunk)
            self._create_paragraph_chunks(section_text_no_tables, section, section_id)
    
    def _extract_and_create_table_chunks(self, section: dict, section_id: str) -> list[Chunk]:
        """Extract tables and create table chunks."""
        table_chunks = []
        section_text = "\n".join(section["lines"])
        
        if not has_actual_table(section_text):
            return table_chunks
        
        # Find table blocks
        lines = section["lines"]
        i = 0
        table_num = 0
        
        while i < len(lines):
            if looks_like_table_content(lines[i]):
                start, end, table_text = detect_table_block(lines, i)
                if table_text and len(table_text) > 50:
                    table_num += 1
                    chunk = self._create_chunk(
                        content=table_text,
                        section=section,
                        section_id=section_id,
                        chunk_type="table",
                        sub_num=table_num,
                        section_position=table_num,
                        total_in_section=0  # Will update later
                    )
                    table_chunks.append(chunk)
                    self.chunks.append(chunk)
                    i = end + 1
                    continue
            i += 1
        
        return table_chunks
    
    def _remove_table_content(self, text: str) -> str:
        """Remove table-like content from text."""
        lines = text.split('\n')
        result_lines = []
        skip_until = -1
        
        for i, line in enumerate(lines):
            if i <= skip_until:
                continue
            
            if looks_like_table_content(line):
                start, end, _ = detect_table_block(lines, i)
                if end > start:
                    skip_until = end
                    continue
            
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _create_single_chunk(self, text: str, section: dict, section_id: str):
        """Create a single chunk for a short section."""
        chunk = self._create_chunk(
            content=text,
            section=section,
            section_id=section_id,
            chunk_type="section",
            sub_num=0,
            section_position=1,
            total_in_section=1
        )
        self.chunks.append(chunk)
    
    def _create_paragraph_chunks(self, text: str, section: dict, section_id: str):
        """Split section into paragraph chunks (no duplication with section-level chunk)."""
        paragraphs = self._smart_split_paragraphs(text)
        
        # Filter out very short paragraphs
        paragraphs = [p for p in paragraphs if estimate_tokens(p) >= MIN_CHUNK_TOKENS]
        
        if not paragraphs:
            return
        
        total_paras = len(paragraphs)
        
        for idx, para in enumerate(paragraphs, 1):
            chunk = self._create_chunk(
                content=para,
                section=section,
                section_id=section_id,
                chunk_type="paragraph",
                sub_num=idx,
                section_position=idx,
                total_in_section=total_paras
            )
            self.chunks.append(chunk)
    
    def _smart_split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs, keeping related content together."""
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
            is_list_item = bool(re.match(r'^\s*[\(\[]?\d+[\)\].]', para))
            prev_is_list = current_para and re.match(r'^\s*[\(\[]?\d+[\)\].]', current_para[-1])
            
            if is_list_item and prev_is_list:
                current_para.append(para)
                current_tokens += para_tokens
                continue
            
            # If adding this would exceed target and we have content, save current
            if current_tokens > 0 and current_tokens + para_tokens > MAX_CHUNK_TOKENS:
                paragraphs.append("\n\n".join(current_para))
                current_para = [para]
                current_tokens = para_tokens
            else:
                current_para.append(para)
                current_tokens += para_tokens
            
            # If current is large enough on its own, save it
            if current_tokens >= TARGET_PARAGRAPH_TOKENS and not is_list_item:
                paragraphs.append("\n\n".join(current_para))
                current_para = []
                current_tokens = 0
        
        if current_para:
            paragraphs.append("\n\n".join(current_para))
        
        return paragraphs
    
    def _create_chunk(
        self,
        content: str,
        section: dict,
        section_id: str,
        chunk_type: str,
        sub_num: int,
        section_position: int,
        total_in_section: int
    ) -> Chunk:
        """Create a chunk with full metadata."""
        self.position_counter += 1
        
        chunk_id = generate_chunk_id(self.year, self.section_counter, chunk_type, sub_num)
        
        # Detect actual table presence
        is_table = chunk_type == "table" or has_actual_table(content)
        has_financial = is_table or bool(re.search(r'\$[\d,]+\s*(million|billion)?|[\d.]+%', content))
        
        # Extract metadata
        entities = extract_entities(content)
        concepts = extract_buffett_concepts(content)
        temporal = extract_temporal_references(content, self.year)
        themes = identify_themes(content)
        content_type = classify_content_type(content, is_table)
        
        # Build hierarchy
        hierarchy = ["Letter Body"]
        if section["title"]:
            hierarchy.append(section["title"])
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            letter_year=self.year,
            letter_date=self.letter_date,
            section_title=section["title"],
            section_hierarchy=hierarchy,
            chunk_type=chunk_type,
            parent_section_id=section_id,
            sibling_chunk_ids=[],  # Will be filled in later
            content_type=content_type,
            has_table=is_table,
            has_financial_data=has_financial,
            entities=entities,
            themes=themes,
            temporal_references=temporal,
            buffett_concepts=concepts,
            token_count=estimate_tokens(content),
            char_count=len(content),
            position_in_letter=self.position_counter,
            section_position=section_position,
            total_section_chunks=total_in_section
        )
        
        return Chunk(content=content, metadata=metadata)
    
    def _update_sibling_references(self):
        """Update sibling references for chunks in the same section."""
        # Group chunks by section
        section_chunks: dict[str, list[Chunk]] = {}
        for chunk in self.chunks:
            sec_id = chunk.metadata.parent_section_id
            if sec_id not in section_chunks:
                section_chunks[sec_id] = []
            section_chunks[sec_id].append(chunk)
        
        # Update sibling references and total counts
        for sec_id, chunks in section_chunks.items():
            chunk_ids = [c.metadata.chunk_id for c in chunks]
            for chunk in chunks:
                chunk.metadata.sibling_chunk_ids = [cid for cid in chunk_ids if cid != chunk.metadata.chunk_id]
                chunk.metadata.total_section_chunks = len(chunks)


# =============================================================================
# File Processing
# =============================================================================

def process_letter_file(filepath: Path) -> list[Chunk]:
    """Process a single letter file and return chunks."""
    text = filepath.read_text(encoding="utf-8")
    
    year = extract_year_from_filename(filepath.name)
    if not year:
        print(f"Warning: Could not extract year from {filepath.name}, skipping")
        return []
    
    chunker = BuffettLetterChunkerV2(text, year, filepath.name)
    chunks = chunker.chunk()
    
    return chunks


def process_all_letters(input_dir: Path, output_dir: Path):
    """Process all letter files in input directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    letter_files = sorted(input_dir.glob("*.txt"))
    
    if not letter_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    all_chunks = []
    stats = {
        "total_files": 0,
        "total_chunks": 0,
        "section_chunks": 0,
        "paragraph_chunks": 0,
        "table_chunks": 0,
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
            
            # Count by type
            sec_count = sum(1 for c in chunks if c.metadata.chunk_type == "section")
            para_count = sum(1 for c in chunks if c.metadata.chunk_type == "paragraph")
            tbl_count = sum(1 for c in chunks if c.metadata.chunk_type == "table")
            
            # Update stats
            stats["total_files"] += 1
            stats["total_chunks"] += len(chunks)
            stats["section_chunks"] += sec_count
            stats["paragraph_chunks"] += para_count
            stats["table_chunks"] += tbl_count
            stats["by_year"][year] = {
                "chunk_count": len(chunks),
                "sections": sec_count,
                "paragraphs": para_count,
                "tables": tbl_count,
                "unique_section_titles": list(set(c.metadata.section_title for c in chunks))
            }
            
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks (sec:{sec_count}, para:{para_count}, tbl:{tbl_count})")
    
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
        print(f"CHUNKING COMPLETE (V2 - No Duplication)")
        print(f"{'='*60}")
        print(f"Total files processed: {stats['total_files']}")
        print(f"Total chunks created:  {stats['total_chunks']}")
        print(f"  - Section chunks:    {stats['section_chunks']}")
        print(f"  - Paragraph chunks:  {stats['paragraph_chunks']}")
        print(f"  - Table chunks:      {stats['table_chunks']}")
        print(f"\nOutput directory: {output_dir}")


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
        
        sec_count = sum(1 for c in chunks if c.metadata.chunk_type == "section")
        para_count = sum(1 for c in chunks if c.metadata.chunk_type == "paragraph")
        tbl_count = sum(1 for c in chunks if c.metadata.chunk_type == "table")
        
        print(f"\n{'='*60}")
        print(f"Created {len(chunks)} chunks for {year}")
        print(f"  - Sections:   {sec_count}")
        print(f"  - Paragraphs: {para_count}")
        print(f"  - Tables:     {tbl_count}")
        print(f"\nSection titles detected:")
        for title in sorted(set(c.metadata.section_title for c in chunks)):
            print(f"  - {title}")
        print(f"\nOutput: {output_file}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chunk Warren Buffett shareholder letters for RAG (V2 - Improved)"
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
        if Path(args.single_file).is_absolute():
            filepath = Path(args.single_file)
        else:
            filepath = args.input_dir / args.single_file
        
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            return 1
        
        process_single_file(filepath, args.output_dir)
    else:
        if not args.input_dir.exists():
            print(f"Error: Input directory not found: {args.input_dir}")
            return 1
        
        process_all_letters(args.input_dir, args.output_dir)
    
    return 0


if __name__ == "__main__":
    exit(main())
