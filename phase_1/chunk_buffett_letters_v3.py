#!/usr/bin/env python3
"""
Chunking Strategy Implementation for Warren Buffett Shareholder Letters (V3)
=============================================================================

V3 IMPROVEMENTS over V2:
1. FIXED: Table headers no longer detected as sections (word-by-word blacklist check)
2. FIXED: Page breaks no longer cause mid-sentence truncation (smart page break handling)
3. FIXED: Missing sections now detected (relaxed blank line requirement + more patterns)
4. FIXED: Signature blocks filtered out (pattern-based rejection)
5. NEW: More robust section detection using multiple signals
6. NEW: Flexible patterns that generalize across all years (1977-present)

ROOT CAUSES ADDRESSED:
- V2 checked blacklist with exact string match; V3 checks word-by-word
- V2 required blank lines around headers; some letters don't have them
- V2 didn't handle [[PAGE_BREAK]] markers properly
- V2 had incomplete section header patterns

Usage:
    python chunk_buffett_letters_v3.py
    python chunk_buffett_letters_v3.py --input-dir /path/to/letters --output-dir /path/to/chunks
    python chunk_buffett_letters_v3.py --single-file 2009_cleaned.txt
"""

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


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
LONG_SECTION_THRESHOLD = 600
TARGET_PARAGRAPH_TOKENS = 400

# Approximate tokens per character
CHARS_PER_TOKEN = 4


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChunkMetadata:
    chunk_id: str
    letter_year: int
    letter_date: Optional[str]
    section_title: Optional[str]
    section_hierarchy: list[str]
    chunk_type: str
    parent_section_id: Optional[str]
    sibling_chunk_ids: list[str]
    content_type: str
    has_table: bool
    has_financial_data: bool
    entities: dict
    themes: list[str]
    temporal_references: dict
    buffett_concepts: list[str]
    token_count: int
    char_count: int
    position_in_letter: int
    section_position: int
    total_section_chunks: int


@dataclass
class Chunk:
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> dict:
        return {"content": self.content, "metadata": asdict(self.metadata)}


# =============================================================================
# Pattern Definitions - V3 Enhanced
# =============================================================================

# Words that indicate table headers (checked word-by-word, not exact match)
TABLE_HEADER_WORDS = {
    # Column header words
    'shares', 'cost', 'market', 'value', 'price', 'amount',
    'earnings', 'revenue', 'income', 'loss', 'gain', 'profit',
    'total', 'subtotal', 'net', 'gross', 'balance',
    'premium', 'float', 'assets', 'liabilities',
    # Table structure words
    'year', 'company', 'business', 'segment',
    # Numeric context
    'millions', 'billions', 'thousands', '(000s)',
    # Time periods in tables
    'quarter', 'annual', 'q1', 'q2', 'q3', 'q4',
    # Additional table indicators
    'percentage', 'ratio', 'rate', 'losses', 'contributions',
    'performance', 'vs', 'designated',
}

# Phrases that definitively indicate table headers (regex patterns)
TABLE_HEADER_PATTERNS = [
    r'.*\bshares?\b.*\b(cost|market|value)\b',
    r'.*\b(cost|market)\b.*\bshares?\b',
    r'.*\bearnings?\s+(per\s+)?share\b',
    r'.*\bper[- ]share\b',
    r'.*\bpercentage\s+change\b',
    r'.*\b(pre|after)[- ]?tax\b.*\bearnings?\b',
    r'^\s*\d{4}\s*[-–]\s*\d{4}\s*$',  # Year ranges like "1965-2014"
    r'.*\bgrowth\s+rate\b',
    r'.*\bin\s+millions\b',
    r'.*\boperating\s+earnings\b.*\bexclud',
    # Additional patterns
    r'.*\bperformance\b.*\bvs\.?\b',  # "Performance vs. S&P 500"
    r'.*\bcorporate\b.*\bperformance\b.*\bs&p',
    r'.*\bnet\s+losses?\b.*\bpercentage\b',
    r'.*\bshareholder[- ]?designated\b',
    r'.*\bsources?\s+of\b.*\bearnings\b',
    r'^berkshire\'?s\s+(share|corporate)',  # "Berkshire's Share" / "Berkshire's Corporate..."
]

# Signature block patterns (to reject)
SIGNATURE_PATTERNS = [
    r'^warren\s+e\.?\s+buffett',
    r'^charlie\s+(t\.?\s+)?munger',
    r'^chairman\s+(of\s+(the\s+)?board)?',
    r'^vice\s+chairman',
    r'^\w+\s+\d{1,2},?\s*\d{4}$',  # Date patterns like "March 14, 1978"
    r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d',
]

# Sentence-start patterns (not headers)
SENTENCE_START_PATTERNS = [
    r'^(finally|additionally|moreover|furthermore|however|therefore|thus),?\s',
    r'^(we|i|our|the|this|that|these|those|it|there|here)\s',
    r'^(as|if|when|while|although|because|since|unless)\s',
    r'^(in|on|at|by|for|with|from|to|of)\s+(the|our|this|that|a|an)\s',
]

# Known section header patterns - comprehensive list for all years
SECTION_HEADER_PATTERNS = [
    # === Insurance Section Headers ===
    r'^Insurance(\s*[-–]\s*.+)?$',
    r'^Insurance\s+(Underwriting|Investments|Operations)(\s*[-–]\s*.+)?$',
    r'^(Super[- ]?Cat|Catastrophe)\s+Insurance$',
    r'^The\s+Economics\s+of\s+Property/Casualty\s+Insurance$',
    r'^(GEICO|General\s+Re|National\s+Indemnity)$',
    
    # === Business Segment Headers ===
    r'^(Textile|Banking|Retail|Manufacturing)\s+Operations?$',
    r'^Manufacturing,?\s+Service\s+and\s+Retailing\s+Operations$',
    r'^Regulated[,]?\s+Capital[- ]Intensive\s+Businesses$',
    r'^Regulated\s+Utility\s+Business$',
    r'^Finance\s+and\s+Financial\s+Products$',
    r'^(Blue\s+Chip\s+Stamps|See\'?s\s+Candies?)$',
    r'^(Flight\s+Safety|Net\s*Jets?|Executive\s+Jet)$',
    r'^(Clayton\s+Homes|Nebraska\s+Furniture|Borsheim\'?s)$',
    
    # === Acquisitions ===
    r'^Acquisitions?(\s+of\s+\d{4})?$',
    r'^(ACQUISITION\s+)?CRITERIA$',
    r'^Acquisition\s+Criteria$',
    
    # === Investments ===
    r'^(Common\s+Stock\s+)?Investments?$',
    r'^Marketable\s+(Equity\s+)?Securities$',
    r'^Berkshire\'?s?\s+Major\s+Investees?$',
    
    # === Financial Reporting ===
    r'^Sources?\s+of\s+Reported\s+Earnings$',
    r'^Look[- ]?Through\s+Earnings$',
    r'^Intrinsic\s+Value.*$',
    
    # === Governance & Other ===
    r'^(The\s+)?Annual\s+Meeting$',
    r'^Shareholder[- ]?Designated\s+Contributions$',
    r'^Corporate\s+Governance$',
    r'^Derivatives$',
    r'^(Taxes|Financings?|Miscellaneous)$',
    
    # === Special Topics ===
    r'^The\s+Relationship\s+of\s+Intrinsic\s+Value.*$',
    r'^(Life\s+and\s+Debt|An\s+Inconvenient\s+Truth.*)$',
    r'^(How|What|Why)\s+We\s+(Measure|Do|Don\'t).*$',
    r'^Berkshire\s+(Today|Past|Present|Future).*$',
    r'^Charlie\s+Straightens\s+Me\s+Out$',
    
    # === Company-specific ===
    r'^(MidAmerican|BNSF|Burlington|Marmon|Iscar|Lubrizol).*$',
    r'^(Home\s+)?Services$',
]

# Patterns that indicate content, not headers
TABLE_CONTENT_PATTERNS = [
    r'^\s*\$[\d,]+',
    r'^\s*[\d,]+\s*$',
    r'^\s*\d+\.\d+%?\s*$',
    r'\.{4,}',
    r'^\s*\(\d+\)\s*$',
    r'^\s*[-–—]+\s*$',
    r'\$[\d,]+\s+\$[\d,]+',
    r'[\d,]+\s{3,}[\d,]+',
    r'^\s*\d{4}\s*\.{2,}',
]

# Buffett concepts for metadata
BUFFETT_CONCEPTS = {
    "float economics": [r"float", r"insurance float", r"cost.free float"],
    "circle of competence": [r"circle of competence", r"understand the business"],
    "margin of safety": [r"margin of safety", r"attractive price", r"bargain"],
    "moat": [r"economic moat", r"competitive advantage", r"durable advantage"],
    "owner earnings": [r"owner earnings", r"look.through earnings"],
    "Mr. Market": [r"mr\.?\s*market"],
    "value vs price": [r"price is what you pay", r"value is what you get", r"intrinsic value"],
    "long-term focus": [r"long.term", r"forever", r"indefinitely", r"permanent holding"],
    "management quality": [r"outstanding manager", r"exceptional manager"],
    "capital allocation": [r"capital allocation", r"deploy capital", r"reinvest"],
    "return on equity": [r"return on equity", r"ROE", r"return on capital"],
    "compounding": [r"compound", r"compounding", r"compounded annually"],
    "mistakes and learning": [r"mistake", r"error", r"wrong", r"foolish"],
}

KNOWN_COMPANIES = [
    "Berkshire Hathaway", "GEICO", "National Indemnity", "General Re", "See's Candies",
    "Nebraska Furniture Mart", "Borsheim's", "Clayton Homes", "BNSF", "Burlington Northern",
    "MidAmerican", "Mid American", "Dairy Queen", "Fruit of the Loom", "NetJets",
    "Blue Chip Stamps", "Wesco", "Coca-Cola", "American Express", "Wells Fargo",
    "Washington Post", "Capital Cities", "Gillette", "IBM", "Apple", "Bank of America",
    "Occidental", "Lubrizol", "Precision Castparts", "Marmon", "FlightSafety",
    "Kansas Bankers Surety", "Jordan's Furniture", "McLane", "Shaw Industries",
]

KNOWN_PEOPLE = [
    "Warren Buffett", "Charlie Munger", "Ajit Jain", "Greg Abel", "Tony Nicely",
    "Phil Liesche", "Lou Simpson", "Gene Abegg", "Chuck Huggins", "Tom Murphy",
]


# =============================================================================
# Text Preprocessing - V3 New
# =============================================================================

def preprocess_text(text: str) -> str:
    """
    Preprocess text to handle page breaks and normalize formatting.
    
    ROOT CAUSE FIX: [[PAGE_BREAK]] markers in PDFs were causing mid-sentence splits.
    Solution: Remove page breaks that occur mid-paragraph and join the text.
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    lines = text.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        
        # Check if this is a page break line (or next line is)
        is_page_break = '[[PAGE_BREAK]]' in line_stripped
        
        if is_page_break:
            # Look back for last content line
            prev_content = ""
            prev_idx = len(result) - 1
            while prev_idx >= 0 and not result[prev_idx].strip():
                prev_idx -= 1
            if prev_idx >= 0:
                prev_content = result[prev_idx].strip()
            
            # Look forward for next content line
            next_content = ""
            next_idx = i + 1
            while next_idx < len(lines) and (not lines[next_idx].strip() or '[[PAGE_BREAK]]' in lines[next_idx]):
                next_idx += 1
            if next_idx < len(lines):
                next_content = lines[next_idx].strip()
            
            # Determine if this is mid-paragraph
            ends_sentence = prev_content and prev_content[-1] in '.?!:"\')'
            next_is_header = next_content and is_likely_header(next_content)
            next_starts_continuation = next_content and next_content[0].islower()
            
            # Also check if next line looks like it continues from prev
            continuation_words = ['contracts', 'people', 'shares', 'and', 'or', 'the', 'a', 'an', 
                                 'to', 'of', 'in', 'that', 'which', 'who', 'with', 'for', 'from',
                                 'value', 'at', 'by', 'as', 'on', 'this', 'these', 'those', 'it']
            first_word = next_content.split()[0].lower() if next_content else ""
            next_is_continuation = first_word in continuation_words
            
            if (not ends_sentence or next_starts_continuation or next_is_continuation) and not next_is_header:
                # This is mid-paragraph - skip the page break
                # Also remove trailing blank lines before the page break from result
                while result and not result[-1].strip():
                    result.pop()
                i += 1
                continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


def is_likely_header(line: str) -> bool:
    """Quick check if a line is likely a section header."""
    line = line.strip()
    if not line or len(line) > 80:
        return False
    
    # Check against known patterns
    for pattern in SECTION_HEADER_PATTERNS:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    
    # Title case with 2-5 words, not ending with punctuation
    words = line.split()
    if 2 <= len(words) <= 5:
        if not line[-1] in '.,;:?!':
            cap_count = sum(1 for w in words if w[0].isupper())
            if cap_count >= len(words) * 0.7:
                return True
    
    return False


# =============================================================================
# Section Header Detection - V3 Enhanced
# =============================================================================

def contains_table_header_words(line: str) -> bool:
    """
    Check if line contains table header vocabulary (word-by-word check).
    
    ROOT CAUSE FIX: V2 only checked exact string match against blacklist.
    V3 checks each word individually.
    """
    line_lower = line.lower()
    words = re.findall(r'\b\w+\b', line_lower)
    
    # Count how many table-related words
    table_word_count = sum(1 for w in words if w in TABLE_HEADER_WORDS)
    
    # If more than 40% of words are table-related, it's likely a table header
    if len(words) > 0 and table_word_count / len(words) >= 0.4:
        return True
    
    # Check for specific multi-word table header patterns
    for pattern in TABLE_HEADER_PATTERNS:
        if re.search(pattern, line_lower):
            return True
    
    return False


def is_signature_line(line: str) -> bool:
    """Check if line is part of signature block."""
    line_lower = line.lower().strip()
    for pattern in SIGNATURE_PATTERNS:
        if re.match(pattern, line_lower):
            return True
    return False


def is_sentence_start(line: str) -> bool:
    """Check if line starts like a sentence (not a header)."""
    line_lower = line.lower().strip()
    for pattern in SENTENCE_START_PATTERNS:
        if re.match(pattern, line_lower):
            return True
    return False


def looks_like_table_content(line: str) -> bool:
    """Check if line looks like table content."""
    for pattern in TABLE_CONTENT_PATTERNS:
        if re.search(pattern, line):
            return True
    return False


def is_in_table_context(lines: list[str], idx: int, window: int = 3) -> bool:
    """Check if surrounding lines suggest we're in a table."""
    start = max(0, idx - window)
    end = min(len(lines), idx + window + 1)
    
    table_like_count = sum(
        1 for i in range(start, end) 
        if i != idx and looks_like_table_content(lines[i])
    )
    
    return table_like_count >= 2


def is_section_header(line: str, lines: list[str], idx: int) -> bool:
    """
    Determine if a line is a section header.
    
    V3 IMPROVEMENTS:
    - Word-by-word blacklist checking
    - Signature block rejection
    - Sentence-start pattern rejection  
    - Relaxed blank line requirement (checks content patterns instead)
    - More comprehensive section patterns
    """
    line_stripped = line.strip()
    
    # === Basic Rejections ===
    if not line_stripped:
        return False
    if len(line_stripped) < 3 or len(line_stripped) > 80:
        return False
    if line_stripped[0].islower():
        return False
    if line_stripped.endswith(','):
        return False
    
    # === V3 Fix: Word-by-word table header check ===
    if contains_table_header_words(line_stripped):
        return False
    
    # === V3 Fix: Signature block rejection ===
    if is_signature_line(line_stripped):
        return False
    
    # === V3 Fix: Sentence-start rejection ===
    if is_sentence_start(line_stripped):
        return False
    
    # === Reject if looks like table content ===
    if looks_like_table_content(line_stripped):
        return False
    
    # === Reject if in table context ===
    if is_in_table_context(lines, idx):
        return False
    
    # === Check against known section header patterns ===
    for pattern in SECTION_HEADER_PATTERNS:
        if re.match(pattern, line_stripped, re.IGNORECASE):
            return True
    
    # === Context-based detection ===
    prev_line = lines[idx - 1].strip() if idx > 0 else ""
    next_line = lines[idx + 1].strip() if idx < len(lines) - 1 else ""
    
    # Check for surrounding blank lines or page breaks
    has_blank_before = (idx == 0) or (prev_line == '') or ('[[PAGE_BREAK]]' in prev_line)
    has_blank_after = (idx == len(lines) - 1) or (next_line == '')
    
    # For generic detection, need at least one blank line
    if not (has_blank_before or has_blank_after):
        # Exception: if prev line ends with period and this looks like title
        if not (prev_line.endswith('.') or prev_line.endswith(':')):
            return False
    
    # === Generic title-case detection ===
    words = line_stripped.split()
    if 2 <= len(words) <= 7:
        # Most words should be capitalized
        small_words = {'and', 'of', 'the', 'for', 'a', 'an', 'in', 'on', 'at', 'to', 'or', '-', '–', '&'}
        cap_count = sum(1 for w in words if w[0].isupper() or w.lower() in small_words)
        
        if cap_count >= len(words) * 0.7:
            # Reject if contains common verbs (it's a sentence)
            verbs = {'is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would', 
                    'can', 'could', 'should', 'may', 'might', 'do', 'does', 'did',
                    'being', 'been', 'be', 'get', 'got', 'make', 'made'}
            if any(w.lower() in verbs for w in words):
                return False
            
            # Accept if doesn't end with period (unless very short)
            if not line_stripped.endswith('.') or len(words) <= 3:
                return True
    
    return False


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def generate_chunk_id(year: int, section_num: int, chunk_type: str, sub_num: int = 0) -> str:
    type_code = {"section": "SEC", "paragraph": "PAR", "table": "TBL"}[chunk_type]
    if sub_num > 0:
        return f"{year}-S{section_num:02d}-{type_code}-{sub_num:03d}"
    return f"{year}-S{section_num:02d}-{type_code}"


def extract_year_from_filename(filename: str) -> Optional[int]:
    match = re.search(r"(\d{4})", filename)
    return int(match.group(1)) if match else None


def extract_letter_date(text: str, year: int) -> Optional[str]:
    patterns = [
        r"([A-Z][a-z]+\s+\d{1,2}\s*,?\s*\d{4})",
        r"(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text[-2000:])
        if matches:
            return matches[-1]
    return None


def has_actual_table(text: str) -> bool:
    """Detect if text contains an actual table."""
    lines = text.split('\n')
    table_row_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Multiple dollar amounts on same line
        if len(re.findall(r'\$[\d,]+\.?\d*', line)) >= 2:
            table_row_count += 1
            continue
        
        # Dot leaders
        if re.search(r'\.{5,}', line):
            table_row_count += 1
            continue
        
        # Year entries with dots
        if re.match(r'^\d{4}\s*\.', line):
            table_row_count += 1
            continue
        
        # Multiple numbers with significant spacing
        if re.search(r'\d+\.?\d*\s{3,}\d+\.?\d*\s{3,}\d+\.?\d*', line):
            table_row_count += 1
    
    return table_row_count >= 4


def detect_table_block(lines: list[str], start_idx: int) -> tuple[int, int, str]:
    """Detect a table block starting near start_idx."""
    table_lines = []
    in_table = False
    table_start = start_idx
    
    for i in range(start_idx, min(start_idx + 50, len(lines))):
        line = lines[i]
        
        if looks_like_table_content(line) or re.search(r'\s{3,}\S', line):
            if not in_table:
                table_start = i
                in_table = True
            table_lines.append(line)
        elif in_table:
            if line.strip() == "" and i + 1 < len(lines):
                if looks_like_table_content(lines[i + 1]):
                    table_lines.append(line)
                    continue
            break
    
    if len(table_lines) >= 3:
        return table_start, table_start + len(table_lines) - 1, "\n".join(table_lines)
    return start_idx, start_idx, ""


def extract_entities(text: str) -> dict:
    entities = {"companies": [], "people": [], "metrics": []}
    text_lower = text.lower()
    
    for company in KNOWN_COMPANIES:
        if company.lower() in text_lower:
            entities["companies"].append(company)
    
    for person in KNOWN_PEOPLE:
        if person.lower() in text_lower:
            entities["people"].append(person)
    
    # Extract metrics
    percentages = re.findall(r'(\d+\.?\d*%)', text)
    entities["metrics"].extend(percentages[:5])
    
    dollar_amounts = re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion)?', text, re.IGNORECASE)
    for amount, unit in dollar_amounts[:5]:
        if unit:
            entities["metrics"].append(f"${amount} {unit}")
        elif ',' in amount or len(amount.replace(',', '')) > 4:
            entities["metrics"].append(f"${amount}")
    
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
    
    return entities


def extract_buffett_concepts(text: str) -> list[str]:
    found_concepts = []
    text_lower = text.lower()
    
    for concept, patterns in BUFFETT_CONCEPTS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found_concepts.append(concept)
                break
    
    return found_concepts


def extract_temporal_references(text: str, primary_year: int) -> dict:
    temporal = {
        "primary_year": primary_year,
        "mentioned_years": [],
        "has_comparison": False,
        "has_forecast": False
    }
    
    years = re.findall(r"\b(19[6-9]\d|20[0-2]\d)\b", text)
    mentioned = sorted(set(int(y) for y in years if int(y) != primary_year))
    temporal["mentioned_years"] = mentioned[:10]
    
    comparison_patterns = [r"compared to", r"versus", r"vs\.?", r"year.ago", r"previous year"]
    temporal["has_comparison"] = any(re.search(p, text.lower()) for p in comparison_patterns)
    
    forecast_patterns = [r"expect", r"anticipate", r"forecast", r"outlook", r"going forward"]
    temporal["has_forecast"] = any(re.search(p, text.lower()) for p in forecast_patterns)
    
    return temporal


def classify_content_type(text: str, has_table: bool) -> str:
    text_lower = text.lower()
    
    if has_table:
        return "financial_table"
    
    philosophy_indicators = [r"we believe", r"our (view|approach|philosophy)", r"principle"]
    if sum(1 for p in philosophy_indicators if re.search(p, text_lower)) >= 2:
        return "philosophy"
    
    if re.search(r"\b(i|we|my|our)\b.{0,30}\b(mistake|error|wrong|foolish)\b", text_lower):
        return "mistake_confession"
    
    perf_indicators = [r"net worth", r"book value", r"per.share", r"operating earnings"]
    if sum(1 for p in perf_indicators if re.search(p, text_lower)) >= 2:
        return "performance_summary"
    
    if re.search(r"annual meeting|shareholder.+meeting", text_lower):
        return "annual_meeting"
    
    return "narrative"


def identify_themes(text: str) -> list[str]:
    themes = []
    text_lower = text.lower()
    
    theme_patterns = {
        "insurance": [r"\binsurance\b", r"\bunderwriting\b", r"\bfloat\b"],
        "acquisitions": [r"\bacquisition\b", r"\bpurchase[d]?\b", r"\bacquired\b"],
        "investments": [r"\binvest(?:ment|ed|ing)?\b", r"\bstock\b", r"\bequity\b"],
        "management": [r"\bmanager\b", r"\bmanagement\b", r"\bCEO\b"],
        "capital allocation": [r"\bcapital\b.{0,20}\ballocat"],
        "valuation": [r"\bintrinsic value\b", r"\bvaluation\b"],
        "growth": [r"\bgrowth\b", r"\bgrew\b"],
        "risk": [r"\brisk\b", r"\buncertain"],
        "dividends": [r"\bdividend"],
        "utilities": [r"\butility\b", r"\butilities\b", r"\benergy\b"],
        "railroads": [r"\brailroad\b", r"\bBNSF\b"],
        "retail": [r"\bretail\b", r"\bstore\b"],
        "manufacturing": [r"\bmanufactur"],
    }
    
    for theme, patterns in theme_patterns.items():
        if any(re.search(p, text_lower) for p in patterns):
            themes.append(theme)
    
    return themes[:5]


# =============================================================================
# Main Chunking Logic - V3
# =============================================================================

class BuffettLetterChunkerV3:
    """V3 Chunker with comprehensive fixes."""
    
    def __init__(self, text: str, year: int, filename: str):
        self.original_text = text
        self.year = year
        self.filename = filename
        
        # Preprocess text (V3: handle page breaks properly)
        self.text = preprocess_text(text)
        self.lines = self.text.split("\n")
        
        self.letter_date = extract_letter_date(text, year)
        self.chunks: list[Chunk] = []
        self.section_counter = 0
        self.position_counter = 0
    
    def chunk(self) -> list[Chunk]:
        # Split into sections
        sections = self._split_into_sections()
        
        # Process each section
        for section in sections:
            self._process_section(section)
        
        # Update sibling references
        self._update_sibling_references()
        
        return self.chunks
    
    def _split_into_sections(self) -> list[dict]:
        """Split letter into logical sections."""
        sections = []
        current_section = {
            "title": "Opening",
            "lines": [],
            "start_idx": 0,
        }
        
        i = 0
        while i < len(self.lines):
            line = self.lines[i]
            
            # Check for section header
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
            
            # Skip page break markers
            if '[[PAGE_BREAK]]' in line:
                i += 1
                continue
            
            # Check for asterisk separators
            if re.match(r'^\s*\*[\s\*]+\*\s*$', line):
                # Look ahead for potential new section
                next_idx = i + 1
                while next_idx < len(self.lines) and not self.lines[next_idx].strip():
                    next_idx += 1
                
                if next_idx < len(self.lines):
                    next_line = self.lines[next_idx].strip()
                    if is_likely_header(next_line):
                        if current_section["lines"]:
                            sections.append(current_section)
                        current_section = {
                            "title": next_line,
                            "lines": [],
                            "start_idx": next_idx,
                        }
                        i = next_idx + 1
                        continue
                
                i += 1
                continue
            
            current_section["lines"].append(line)
            i += 1
        
        # Don't forget last section
        if current_section["lines"]:
            sections.append(current_section)
        
        return sections
    
    def _process_section(self, section: dict):
        """Process a section into chunks."""
        self.section_counter += 1
        section_text = "\n".join(section["lines"]).strip()
        
        if not section_text or estimate_tokens(section_text) < MIN_CHUNK_TOKENS:
            return
        
        section_id = f"{self.year}-S{self.section_counter:02d}"
        
        # Extract tables first
        self._extract_and_create_table_chunks(section, section_id)
        
        # Remove table content
        section_text_no_tables = self._remove_table_content(section_text)
        section_tokens = estimate_tokens(section_text_no_tables)
        
        if section_tokens < MIN_CHUNK_TOKENS:
            return
        
        if section_tokens <= LONG_SECTION_THRESHOLD:
            self._create_single_chunk(section_text_no_tables, section, section_id)
        else:
            self._create_paragraph_chunks(section_text_no_tables, section, section_id)
    
    def _extract_and_create_table_chunks(self, section: dict, section_id: str):
        """Extract and create table chunks."""
        section_text = "\n".join(section["lines"])
        
        if not has_actual_table(section_text):
            return
        
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
                        total_in_section=0
                    )
                    self.chunks.append(chunk)
                    i = end + 1
                    continue
            i += 1
    
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
        """Split section into paragraph chunks."""
        paragraphs = self._smart_split_paragraphs(text)
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
        """Split text into paragraphs intelligently, never breaking mid-sentence."""
        lines = text.split('\n')
        
        # First pass: Join lines that are split mid-sentence (including across blank lines)
        paragraphs = []
        current_para = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                # Blank line - check if we're mid-sentence
                if current_para:
                    last_content = current_para[-1]
                    ends_sentence = last_content and last_content[-1] in '.?!:"\')'
                    
                    if not ends_sentence:
                        # Look ahead for next content
                        next_idx = i + 1
                        while next_idx < len(lines) and not lines[next_idx].strip():
                            next_idx += 1
                        
                        if next_idx < len(lines):
                            next_content = lines[next_idx].strip()
                            # Check if next line continues the sentence
                            first_word = next_content.split()[0].lower() if next_content else ""
                            continuation_words = {'contracts', 'people', 'shares', 'and', 'or', 'the', 
                                                 'a', 'an', 'to', 'of', 'in', 'that', 'which', 'who',
                                                 'with', 'for', 'from', 'value', 'at', 'by', 'as', 'on',
                                                 'this', 'these', 'those', 'it', 'its', 'their', 'our',
                                                 'but', 'so', 'yet', 'nor', 'if', 'when', 'while', 'each'}
                            
                            if first_word in continuation_words or next_content[0].islower():
                                # Join across the blank line
                                current_para[-1] = current_para[-1] + " " + next_content
                                i = next_idx + 1
                                continue
                    
                    # Normal paragraph break
                    paragraphs.append(' '.join(current_para))
                    current_para = []
                i += 1
                continue
            
            # Non-blank line
            if current_para:
                # Check if previous line ended mid-sentence
                last_content = current_para[-1]
                if last_content[-1] not in '.?!:"\')\n':
                    # Join with current line
                    current_para[-1] = current_para[-1] + " " + line
                else:
                    current_para.append(line)
            else:
                current_para.append(line)
            
            i += 1
        
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        # Second pass: Merge small paragraphs and split large ones
        final_paragraphs = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = estimate_tokens(para)
            
            # Keep numbered lists together
            is_list_item = bool(re.match(r'^\s*[\(\[]?\d+[\)\].]', para))
            prev_is_list = current_chunk and re.match(r'^\s*[\(\[]?\d+[\)\].]', current_chunk[-1])
            
            if is_list_item and prev_is_list:
                current_chunk.append(para)
                current_tokens += para_tokens
                continue
            
            if current_tokens > 0 and current_tokens + para_tokens > MAX_CHUNK_TOKENS:
                final_paragraphs.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
            
            if current_tokens >= TARGET_PARAGRAPH_TOKENS and not is_list_item:
                final_paragraphs.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
        
        if current_chunk:
            final_paragraphs.append("\n\n".join(current_chunk))
        
        return final_paragraphs
    
    def _create_chunk(self, content: str, section: dict, section_id: str,
                     chunk_type: str, sub_num: int, section_position: int,
                     total_in_section: int) -> Chunk:
        """Create a chunk with full metadata."""
        self.position_counter += 1
        
        chunk_id = generate_chunk_id(self.year, self.section_counter, chunk_type, sub_num)
        
        is_table = chunk_type == "table" or has_actual_table(content)
        has_financial = is_table or bool(re.search(r'\$[\d,]+|\d+\.?\d*%', content))
        
        entities = extract_entities(content)
        concepts = extract_buffett_concepts(content)
        temporal = extract_temporal_references(content, self.year)
        themes = identify_themes(content)
        content_type = classify_content_type(content, is_table)
        
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
            sibling_chunk_ids=[],
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
        section_chunks: dict[str, list[Chunk]] = {}
        
        for chunk in self.chunks:
            sec_id = chunk.metadata.parent_section_id
            if sec_id not in section_chunks:
                section_chunks[sec_id] = []
            section_chunks[sec_id].append(chunk)
        
        for sec_id, chunks in section_chunks.items():
            chunk_ids = [c.metadata.chunk_id for c in chunks]
            for chunk in chunks:
                chunk.metadata.sibling_chunk_ids = [
                    cid for cid in chunk_ids if cid != chunk.metadata.chunk_id
                ]
                chunk.metadata.total_section_chunks = len(chunks)


# =============================================================================
# File Processing
# =============================================================================

def process_letter_file(filepath: Path) -> list[Chunk]:
    """Process a single letter file."""
    text = filepath.read_text(encoding="utf-8")
    
    year = extract_year_from_filename(filepath.name)
    if not year:
        print(f"Warning: Could not extract year from {filepath.name}")
        return []
    
    chunker = BuffettLetterChunkerV3(text, year, filepath.name)
    return chunker.chunk()


def process_all_letters(input_dir: Path, output_dir: Path):
    """Process all letter files."""
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
            
            sec_count = sum(1 for c in chunks if c.metadata.chunk_type == "section")
            para_count = sum(1 for c in chunks if c.metadata.chunk_type == "paragraph")
            tbl_count = sum(1 for c in chunks if c.metadata.chunk_type == "table")
            
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
                "section_titles": list(set(c.metadata.section_title for c in chunks))
            }
            
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks (sec:{sec_count}, para:{para_count}, tbl:{tbl_count})")
    
    if all_chunks:
        combined_output = output_dir / "all_letters_chunks.json"
        with open(combined_output, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in all_chunks], f, indent=2, ensure_ascii=False)
        
        stats_output = output_dir / "chunking_stats.json"
        with open(stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"CHUNKING COMPLETE (V3)")
        print(f"{'='*60}")
        print(f"Total files: {stats['total_files']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"  - Sections: {stats['section_chunks']}")
        print(f"  - Paragraphs: {stats['paragraph_chunks']}")
        print(f"  - Tables: {stats['table_chunks']}")


def process_single_file(filepath: Path, output_dir: Path):
    """Process a single file."""
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
        
        print(f"\nCreated {len(chunks)} chunks for {year}")
        print(f"  Sections: {sec_count}, Paragraphs: {para_count}, Tables: {tbl_count}")
        print(f"\nSection titles:")
        for title in sorted(set(c.metadata.section_title for c in chunks)):
            print(f"  - {title}")


def main():
    parser = argparse.ArgumentParser(description="Chunk Buffett letters for RAG (V3)")
    parser.add_argument("--input-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--single-file", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.single_file:
        filepath = Path(args.single_file) if Path(args.single_file).is_absolute() else args.input_dir / args.single_file
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
