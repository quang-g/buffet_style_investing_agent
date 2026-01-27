"""
Enrich chunks with adjacent chunk summaries.

This script reads chunk JSON files and adds previous_summary and next_summary
fields to each chunk based on the adjacent chunks' content_summary or contextual_summary.

Script path: phase_1/enrich_adjacent_summary.py
Input path pattern: data/chunks/{year}_chunks_2tier.json
Output path pattern: data/enriched_chunks/{year}_chunks.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional


# Get the project root directory (parent of phase_1)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def get_chunk_summary(chunk: dict) -> Optional[str]:
    """
    Extract summary from a chunk, preferring content_summary over contextual_summary.
    
    Args:
        chunk: The chunk dictionary
        
    Returns:
        The summary string or None if not found
    """
    metadata = chunk.get("metadata", {})
    
    # Prefer content_summary if available, otherwise use contextual_summary
    summary = metadata.get("content_summary") or metadata.get("contextual_summary")
    
    return summary


def enrich_chunks_with_adjacent_summaries(chunks: list[dict]) -> list[dict]:
    """
    Add previous_summary and next_summary to each chunk based on adjacent chunks.
    
    Args:
        chunks: List of chunk dictionaries in order
        
    Returns:
        List of enriched chunk dictionaries
    """
    enriched_chunks = []
    num_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        # Create a copy to avoid modifying the original
        enriched_chunk = json.loads(json.dumps(chunk))
        
        # Get previous chunk's summary (None for first chunk)
        if i > 0:
            previous_summary = get_chunk_summary(chunks[i - 1])
        else:
            previous_summary = None
        
        # Get next chunk's summary (None for last chunk)
        if i < num_chunks - 1:
            next_summary = get_chunk_summary(chunks[i + 1])
        else:
            next_summary = None
        
        # Add the adjacent summaries to metadata
        enriched_chunk["metadata"]["previous_summary"] = previous_summary
        enriched_chunk["metadata"]["next_summary"] = next_summary
        
        enriched_chunks.append(enriched_chunk)
    
    return enriched_chunks


def process_file(input_path: Path, output_path: Path) -> None:
    """
    Process a single chunk file and save the enriched version.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to save the enriched JSON file
    """
    print(f"Processing: {input_path}")
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"  Loaded {len(chunks)} chunks")
    
    # Enrich chunks with adjacent summaries
    enriched_chunks = enrich_chunks_with_adjacent_summaries(chunks)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved enriched chunks to: {output_path}")


def process_all_files(input_dir: Path, output_dir: Path) -> None:
    """
    Process all chunk files in the input directory.
    
    Args:
        input_dir: Directory containing input chunk files
        output_dir: Directory to save enriched chunk files
    """
    # Find all *_chunks_2tier.json files
    chunk_files = sorted(input_dir.glob("*_chunks_2tier.json"))
    
    if not chunk_files:
        print(f"No chunk files found in {input_dir}")
        return
    
    print(f"Found {len(chunk_files)} chunk files to process\n")
    
    for input_file in chunk_files:
        # Extract year from filename (e.g., "1977_chunks_2tier.json" -> "1977")
        year = input_file.stem.replace("_chunks_2tier", "")
        
        # Construct output path
        output_file = output_dir / f"{year}_chunks.json"
        
        process_file(input_file, output_file)
    
    print(f"\nCompleted processing {len(chunk_files)} files")


def main():
    """Main entry point."""
    # Define input and output directories relative to project root
    input_dir = PROJECT_ROOT / "data" / "chunks"
    output_dir = PROJECT_ROOT / "data" / "enriched_chunks"
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print("Please ensure the input directory exists with chunk files.")
        sys.exit(1)
    
    process_all_files(input_dir, output_dir)


if __name__ == "__main__":
    main()