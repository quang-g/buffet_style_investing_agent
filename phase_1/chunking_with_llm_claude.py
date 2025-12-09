#!/usr/bin/env python3
"""
chunk_with_llm_claude.py

LLM-based chunking for Warren Buffett Letters to Shareholders.
Uses Claude API with prompt caching for cost efficiency.

Usage:
    python chunk_with_llm_claude.py 1996
    python chunk_with_llm_claude.py 1996 --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import anthropic


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
STRATEGY_FILE = SCRIPT_DIR / "chunking_strategy.md"
INPUT_DIR = SCRIPT_DIR / ".." / "data" / "text_extracted_letters"
OUTPUT_DIR = SCRIPT_DIR / ".." / "data" / "chunks_llm_claude"

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 16000


# ---------------------------------------------------------------------------
# System Prompt (cached)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert text processor specializing in chunking documents for RAG (Retrieval-Augmented Generation) systems.

Your task is to chunk Warren Buffett's annual letter to shareholders according to a detailed chunking strategy specification.

You will receive:
1. A chunking strategy document (CHUNKING_STRATEGY.md) - follow this precisely
2. The full text of a specific year's letter

Output Requirements:
- Return ONLY a valid JSON array of chunk objects
- Each chunk must include ALL required metadata fields per the specification
- No explanatory text before or after the JSON
- Ensure the JSON is properly formatted and parseable

Critical Rules:
- Follow all chunking rules from the strategy document
- Respect chunk boundaries (never break mid-sentence, mid-story, or mid-table)
- Generate contextual summaries for each chunk
- Extract all entities (companies, people, topics, metrics)
- Classify principles where present
- Assign appropriate retrieval_priority and abstraction_level

Quality Standards:
- Target 150-300 words per chunk (stories can go up to 500)
- Every chunk should be a complete, understandable thought
- Include section headers in every chunk from that section
"""


# ---------------------------------------------------------------------------
# User Prompt Template
# ---------------------------------------------------------------------------
USER_PROMPT_TEMPLATE = """Process the following Warren Buffett letter from {year} and chunk it according to the specification.

<CHUNKING_STRATEGY>
{strategy}
</CHUNKING_STRATEGY>

<LETTER_TEXT year="{year}">
{letter_text}
</LETTER_TEXT>

Now chunk this letter following the strategy. Return ONLY a JSON array of chunks with the following structure for each chunk:

```json
[
  {{
    "chunk_id": "{{year}}_{{section_type}}_{{sequence:03d}}",
    "year": {year},
    "source_file": "{year}_cleaned.txt",
    "section_type": "string (one of: performance_overview, insurance_operations, acquisitions, investments, operating_businesses, corporate_governance, management_philosophy, shareholder_matters, other)",
    "section_title": "string",
    "subsection": "string or null",
    "parent_section": "string or null",
    "position_in_letter": "float 0.0-1.0",
    "position_in_section": "int (0-indexed)",
    "total_chunks_in_section": "int",
    "chunk_text": "string (the actual text content)",
    "word_count": "int",
    "char_count": "int",
    "chunk_type": "string (one of: narrative_story, financial_table, philosophy, business_analysis, administrative)",
    "has_financials": "bool",
    "has_table": "bool",
    "has_quote": "bool",
    "contains_principle": "bool",
    "contains_example": "bool",
    "contains_comparison": "bool",
    "contextual_summary": "string (2-3 sentences per spec)",
    "prev_context": "string (1-2 sentences summarizing preceding content, empty for first chunk)",
    "next_context": "string (1-2 sentences summarizing following content, empty for last chunk)",
    "topics": ["array", "of", "topic", "strings"],
    "companies_mentioned": ["array", "of", "company", "names"],
    "people_mentioned": ["array", "of", "people", "names"],
    "metrics_discussed": ["array", "of", "metric", "names"],
    "industries": ["array", "of", "industry", "names"],
    "principle_category": "string or null (if contains_principle: moats, valuation, management_quality, capital_allocation, risk_management, competitive_advantage, business_quality)",
    "principle_statement": "string or null (if contains_principle)",
    "retrieval_priority": "string (high, medium, low)",
    "abstraction_level": "string (high, medium, low)",
    "time_sensitivity": "string (high, low)",
    "is_complete_thought": "bool",
    "needs_context": "bool"
  }}
]
```

Return ONLY the JSON array, no other text."""


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------
def load_chunking_strategy() -> str:
    """Load the chunking strategy document."""
    if not STRATEGY_FILE.exists():
        raise FileNotFoundError(
            f"Chunking strategy file not found: {STRATEGY_FILE}\n"
            f"Expected location: {STRATEGY_FILE.resolve()}"
        )
    return STRATEGY_FILE.read_text(encoding="utf-8")


def load_letter(year: int) -> str:
    """Load a specific year's cleaned letter."""
    letter_file = INPUT_DIR / f"{year}_cleaned.txt"
    if not letter_file.exists():
        raise FileNotFoundError(
            f"Letter file not found: {letter_file}\n"
            f"Expected location: {letter_file.resolve()}"
        )
    return letter_file.read_text(encoding="utf-8")


def chunk_with_claude(
    strategy: str,
    letter_text: str,
    year: int,
    dry_run: bool = False
) -> list[dict]:
    """
    Call Claude API to chunk the letter.
    
    Uses prompt caching:
    - System prompt: cached (static instructions)
    - Chunking strategy in user message: cached with cache_control
    - Letter text: not cached (varies per year)
    """
    client = anthropic.Anthropic()
    
    user_message = USER_PROMPT_TEMPLATE.format(
        year=year,
        strategy=strategy,
        letter_text=letter_text
    )
    
    if dry_run:
        print(f"\n[DRY RUN] Would send request to Claude API")
        print(f"  Model: {MODEL}")
        print(f"  System prompt length: {len(SYSTEM_PROMPT):,} chars")
        print(f"  User message length: {len(user_message):,} chars")
        print(f"  Strategy length: {len(strategy):,} chars")
        print(f"  Letter length: {len(letter_text):,} chars")
        return []
    
    # Use prompt caching: the strategy document is static and can be cached
    # By placing it at the beginning of the user message with cache_control,
    # subsequent calls can reuse the cached portion
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<CHUNKING_STRATEGY>\n{strategy}\n</CHUNKING_STRATEGY>",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": f"""Process the following Warren Buffett letter from {year} and chunk it according to the specification above.

<LETTER_TEXT year="{year}">
{letter_text}
</LETTER_TEXT>

Now chunk this letter following the strategy. Return ONLY a JSON array of chunks with the complete metadata structure defined in the chunking strategy.

Important: Return ONLY valid JSON - no markdown code fences, no explanatory text."""
                    }
                ]
            }
        ]
    )
    
    # Log usage for cost tracking
    usage = response.usage
    print(f"\n[API Usage]")
    print(f"  Input tokens: {usage.input_tokens:,}")
    print(f"  Output tokens: {usage.output_tokens:,}")
    if hasattr(usage, 'cache_creation_input_tokens'):
        print(f"  Cache creation tokens: {usage.cache_creation_input_tokens:,}")
    if hasattr(usage, 'cache_read_input_tokens'):
        print(f"  Cache read tokens: {usage.cache_read_input_tokens:,}")
    
    # Extract and parse JSON response
    response_text = response.content[0].text.strip()
    
    # Handle potential markdown code fences
    if response_text.startswith("```"):
        # Remove opening fence (possibly with 'json' label)
        response_text = response_text.split("\n", 1)[1]
    if response_text.endswith("```"):
        response_text = response_text.rsplit("```", 1)[0]
    
    response_text = response_text.strip()
    
    try:
        chunks = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] Failed to parse JSON response: {e}")
        print(f"Response preview (first 500 chars):\n{response_text[:500]}")
        raise
    
    return chunks


def validate_chunks(chunks: list[dict], year: int) -> list[str]:
    """Basic validation of generated chunks."""
    issues = []
    
    required_fields = [
        "chunk_id", "year", "section_type", "chunk_text", "word_count",
        "chunk_type", "contextual_summary", "retrieval_priority"
    ]
    
    for i, chunk in enumerate(chunks):
        # Check required fields
        for field in required_fields:
            if field not in chunk:
                issues.append(f"Chunk {i}: Missing required field '{field}'")
        
        # Check year matches
        if chunk.get("year") != year:
            issues.append(f"Chunk {i}: Year mismatch (got {chunk.get('year')}, expected {year})")
        
        # Check word count
        if "chunk_text" in chunk and "word_count" in chunk:
            actual_words = len(chunk["chunk_text"].split())
            stated_words = chunk["word_count"]
            if abs(actual_words - stated_words) > 10:
                issues.append(
                    f"Chunk {i}: Word count mismatch (actual: {actual_words}, stated: {stated_words})"
                )
        
        # Check chunk_id format
        if "chunk_id" in chunk:
            if not chunk["chunk_id"].startswith(f"{year}_"):
                issues.append(f"Chunk {i}: chunk_id should start with '{year}_'")
    
    return issues


def write_jsonl(chunks: list[dict], output_path: Path) -> None:
    """Write chunks to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Chunk Buffett letters using Claude API"
    )
    parser.add_argument(
        "year",
        type=int,
        help="Year of the letter to process (e.g., 1996)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent without making API call"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation of generated chunks"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Override output path"
    )
    
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"Buffett Letter Chunking - Year {args.year}")
    print(f"=" * 60)
    
    # Load inputs
    print(f"\n[1/4] Loading chunking strategy...")
    strategy = load_chunking_strategy()
    print(f"  Loaded: {len(strategy):,} characters")
    
    print(f"\n[2/4] Loading letter for year {args.year}...")
    letter_text = load_letter(args.year)
    print(f"  Loaded: {len(letter_text):,} characters ({len(letter_text.split()):,} words)")
    
    # Process with Claude
    print(f"\n[3/4] Processing with Claude API...")
    chunks = chunk_with_claude(
        strategy=strategy,
        letter_text=letter_text,
        year=args.year,
        dry_run=args.dry_run
    )
    
    if args.dry_run:
        print("\n[DRY RUN COMPLETE]")
        return
    
    print(f"  Generated: {len(chunks)} chunks")
    
    # Validate
    if not args.skip_validation:
        print(f"\n[3.5/4] Validating chunks...")
        issues = validate_chunks(chunks, args.year)
        if issues:
            print(f"  ⚠️  Found {len(issues)} validation issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"    - {issue}")
            if len(issues) > 10:
                print(f"    ... and {len(issues) - 10} more")
        else:
            print(f"  ✓ All chunks passed validation")
    
    # Write output
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"{args.year}_chunks_llm.jsonl"
    
    print(f"\n[4/4] Writing output...")
    write_jsonl(chunks, output_path)
    print(f"  Written to: {output_path.resolve()}")
    
    # Summary stats
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total chunks: {len(chunks)}")
    if chunks:
        word_counts = [c.get("word_count", 0) for c in chunks]
        print(f"  Avg words/chunk: {sum(word_counts) / len(word_counts):.1f}")
        print(f"  Min words: {min(word_counts)}")
        print(f"  Max words: {max(word_counts)}")
        
        # Count by type
        type_counts = {}
        for c in chunks:
            ct = c.get("chunk_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1
        print(f"\n  Chunks by type:")
        for ct, count in sorted(type_counts.items()):
            print(f"    {ct}: {count}")
        
        # Count principles
        principles = sum(1 for c in chunks if c.get("contains_principle"))
        print(f"\n  Chunks with principles: {principles} ({100*principles/len(chunks):.1f}%)")


if __name__ == "__main__":
    main()