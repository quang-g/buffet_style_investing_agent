import os
import sys
import json
import argparse
from google import genai
from google.genai import types
from typing import List, Dict, Any

# ==========================================
# CONFIGURATION
# ==========================================
# Default model (User requested "gemini-2.5-flash", likely meaning the latest Flash)
# Adjust this string if the specific API model ID differs.
MODEL_NAME = "gemini-2.5-flash" 

# Paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_FILE = os.path.join(SCRIPT_DIR, "chunking_rule_claude.md")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
INPUT_DIR = os.path.join(DATA_DIR, "text_extracted_letters")
OUTPUT_DIR = os.path.join(DATA_DIR, "chunks_llm_gemini")

# ==========================================
# PROMPTS
# ==========================================

SYSTEM_INSTRUCTION = """
You are the **Buffett Letter Chunking Engine**. Your sole purpose is to process Warren Buffett's Shareholder Letters into a structured JSON dataset based on a strict set of rules.

You will be provided with:
1. The `chunking_strategy.md` which contains all rules, schemas, and metadata requirements.
2. The full text of a specific year's shareholder letter.

**YOUR TASK:**
1. Analyze the letter text to identify sections and flow.
2. Break the text into semantic chunks according to 'PART 1: CHUNKING RULES' in the strategy.
3. Generate extensive metadata for *every* chunk according to 'PART 2: METADATA SCHEMA'.
4. Perform entity extraction and contextual summary generation as defined in 'PART 3'.
5. Return the result as a strictly formatted JSON List of objects.

**CRITICAL REQUIREMENTS:**
- **Strict JSON Output**: Do not output markdown code blocks (```json). Just return the raw JSON string or a JSON object.
- **Completeness**: You must process the *entire* letter text provided. Do not summarize or skip sections.
- **Metadata**: Ensure every field in the `chunk_metadata` schema is populated.
- **Context**: The `contextual_summary` field is the most important. It must tie the specific chunk to the year's broader context.
"""

def load_file(filepath: str) -> str:
    """Reads a file and returns its content as a string."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)

def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Saves a list of dictionaries to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"Success: Wrote {len(data)} chunks to {filepath}")

def process_letter_with_llm(year: str, api_key: str):
    """Orchestrates the LLM call to chunk the letter."""
    
    # 1. Setup API (Google GenAI SDK)
    client = genai.Client(api_key=api_key)

    # 2. Load Inputs
    input_path = os.path.join(INPUT_DIR, f"{year}_cleaned.txt")
    print(f"Reading letter: {input_path}...")
    letter_text = load_file(input_path)
    
    print(f"Reading strategy: {STRATEGY_FILE}...")
    strategy_text = load_file(STRATEGY_FILE)

    # 3. Construct Prompt
    # We pass the high-level system instruction, the strategy, and the letter content clearly delimited.
    prompt = f"""
    ### SYSTEM INSTRUCTION ###
    {SYSTEM_INSTRUCTION}

    *** INPUT DOCUMENT 1: CHUNKING STRATEGY ***
    {strategy_text}

    *** INPUT DOCUMENT 2: YEAR {year} LETTER TEXT ***
    {letter_text}

    *** INSTRUCTION ***
    Process "INPUT DOCUMENT 2" using the rules in "INPUT DOCUMENT 1". 
    Output the final list of chunks as a valid JSON array.
    """

    # 4. Call LLM
    print(f"Sending to Gemini ({MODEL_NAME}). This may take a moment...")
    try:
        # Google GenAI SDK client call (Gemini 2.5 Flash has a large context window)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,  # Low temperature for strict adherence to schema
            ),
        )
        
        # 5. Parse Response
        chunks = json.loads(response.text)
        
        # Basic validation
        if not isinstance(chunks, list):
            raise ValueError("LLM did not return a list of chunks.")
        
        print(f"LLM generated {len(chunks)} chunks.")
        
        # 6. Save Output
        output_path = os.path.join(OUTPUT_DIR, f"{year}_chunks_llm.jsonl")
        save_jsonl(chunks, output_path)

    except Exception as e:
        print(f"Error during LLM processing: {e}")
        # Debug: if response blocked or errored, print prompt size or safety ratings
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
            print(response.prompt_feedback)

if __name__ == "__main__":
    # Ensure API Key is set
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    # Parse Arguments
    parser = argparse.ArgumentParser(description="Chunk Buffett letters using Gemini.")
    parser.add_argument("year", help="The year of the letter to process (e.g., 1997)")
    args = parser.parse_args()

    process_letter_with_llm(args.year, api_key)