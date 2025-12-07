# Buffett Letters Chunking Strategy v1.0

## EXECUTIVE SUMMARY
This document defines the complete chunking strategy for processing Warren Buffett's Letters to Shareholders (1977-2024) into a structured corpus for RAG-based retrieval.

**Goal**: Create semantically meaningful chunks that preserve Buffett's narrative flow while enabling precise retrieval.

**Target Chunk Size**: 150-300 words (flexible based on content type)

---

## PART 1: CHUNKING RULES

### Rule 1: Section-Based Hierarchical Chunking
**Primary Principle**: Respect the natural structure of each letter.

#### 1.1 Identify Major Sections
Each letter contains these common sections (not all present in every year):

```
STANDARD_SECTIONS = {
    "performance_overview": [
        "Performance Summary",
        "Per-Share Book Value",
        "Annual Gains"
    ],
    "insurance_operations": [
        "Insurance Operations",
        "GEICO",
        "General Re",
        "Super-Cat Insurance",
        "Reinsurance",
        "Float"
    ],
    "acquisitions": [
        "Acquisitions",
        "New Businesses"
    ],
    "investments": [
        "Common Stock Investments",
        "Investment Philosophy",
        "Portfolio Holdings",
        "Look-Through Earnings"
    ],
    "operating_businesses": [
        "Sources of Reported Earnings",
        "Operating Businesses",
        "Individual Business Performance"
    ],
    "corporate_governance": [
        "Corporate Governance",
        "Audit Committee",
        "Board of Directors",
        "Executive Compensation"
    ],
    "management_philosophy": [
        "Management Principles",
        "Capital Allocation",
        "Business Quality"
    ],
    "shareholder_matters": [
        "Shareholder-Designated Contributions",
        "Annual Meeting"
    ]
}
```

#### 1.2 Section Detection Logic
```python
# Pseudocode for section identification
def identify_section(text_block):
    """
    1. Look for bold/header formatting
    2. Check against STANDARD_SECTIONS keywords
    3. Assign section_type and section_title
    4. If nested (e.g., "GEICO" under "Insurance"), 
       record both parent and subsection
    """
    return {
        "section_type": "insurance_operations",
        "section_title": "Insurance Operations - GEICO and Other Primary Operations",
        "subsection": "GEICO",
        "parent_section": "Insurance Operations"
    }
```

### Rule 2: Content-Type Specific Chunking

#### 2.1 NARRATIVE STORIES
**Identification**: Contains sequence of events, named people, specific dates, anecdotes

**Chunking Rule**: Keep complete stories intact, even if >300 words

**Example Indicators**:
- "Let me tell you about..."
- "Here's how we..."
- Contains dialogue or specific event sequences

**Max Size**: 500 words (only for complete stories)

**Metadata**: 
```python
{
    "chunk_type": "narrative_story",
    "story_subject": "Kansas Bankers Surety acquisition",
    "contains_principle": True/False
}
```

#### 2.2 FINANCIAL TABLES
**Identification**: Structured data with columns, rows, numbers

**Chunking Rule**: Include table + preceding explanatory paragraph + following interpretation (if any)

**Example**:
```
[Paragraph explaining what the table shows]
[The complete table]
[Optional: Paragraph interpreting results]
```

**Metadata**:
```python
{
    "chunk_type": "financial_table",
    "table_subject": "Insurance Float History",
    "years_covered": [1967, 1977, 1987, 1997, 2002],
    "metrics": ["float", "underwriting_loss", "cost_of_funds"]
}
```

#### 2.3 INVESTMENT PHILOSOPHY
**Identification**: Abstract principles about investing, business quality, moats

**Chunking Rule**: Break at natural conceptual boundaries, typically 150-250 words

**Key Phrases**:
- "We look for..."
- "The key is..."
- "We believe..."
- "Our goal..."
- "Charlie and I..."

**Metadata**:
```python
{
    "chunk_type": "philosophy",
    "contains_principle": True,
    "principle_category": "moats" | "valuation" | "management" | "capital_allocation",
    "abstraction_level": "high" | "medium" | "low"
}
```

#### 2.4 COMPANY/BUSINESS ANALYSIS
**Identification**: Discussion of specific companies or business segments

**Chunking Rule**: 200-300 words per business unit

**Metadata**:
```python
{
    "chunk_type": "business_analysis",
    "companies_mentioned": ["GEICO", "See's Candies"],
    "metrics_discussed": ["underwriting_profit", "market_share"],
    "contains_example": True
}
```

#### 2.5 ADMINISTRATIVE CONTENT
**Identification**: Annual meeting details, shareholder contribution program

**Chunking Rule**: Chunk together by topic, mark as low-priority

**Metadata**:
```python
{
    "chunk_type": "administrative",
    "retrieval_priority": "low",
    "section_type": "shareholder_matters"
}
```

### Rule 3: Chunk Boundary Rules

#### 3.1 NEVER Break:
- ❌ Mid-sentence
- ❌ In the middle of a complete story/anecdote
- ❌ Between a table and its explanation
- ❌ Within a quote (if >1 sentence)
- ❌ Inside a numbered/bulleted list

#### 3.2 PREFER Breaking:
- ✅ Between paragraphs
- ✅ Between complete thoughts/concepts
- ✅ Between different business discussions
- ✅ At section/subsection transitions

#### 3.3 Overlap Strategy:
- Include section header in EVERY chunk from that section
- For multi-paragraph chunks, include 1 sentence from previous chunk if it provides essential context

---

## PART 2: METADATA SCHEMA

### Complete Metadata Structure
```python
chunk_metadata = {
    # ============================================
    # BASIC IDENTIFIERS (Required)
    # ============================================
    "chunk_id": str,  # Format: "{year}_{section_type}_{sequence:03d}"
                      # Example: "1996_insurance_003"
    
    "year": int,  # 1977-2024
    
    "source_file": str,  # Original filename
    
    # ============================================
    # STRUCTURAL CONTEXT (Required)
    # ============================================
    "section_type": str,  # One of STANDARD_SECTIONS keys
    
    "section_title": str,  # Full section heading text
    
    "subsection": str | None,  # If nested section
    
    "parent_section": str | None,  # If subsection exists
    
    # ============================================
    # POSITION METADATA (Required)
    # ============================================
    "position_in_letter": float,  # 0.0 to 1.0 (percentage through letter)
    
    "position_in_section": int,  # Chunk number within section (0-indexed)
    
    "total_chunks_in_section": int,  # Total chunks in this section
    
    "paragraph_start_idx": int,  # Starting paragraph number
    
    "paragraph_end_idx": int,  # Ending paragraph number
    
    # ============================================
    # CONTENT METADATA (Required)
    # ============================================
    "chunk_text": str,  # The actual text content
    
    "word_count": int,
    
    "char_count": int,
    
    "chunk_type": str,  # "narrative_story" | "financial_table" | 
                        # "philosophy" | "business_analysis" | "administrative"
    
    # ============================================
    # CONTENT FLAGS (Required)
    # ============================================
    "has_financials": bool,  # Contains numbers, percentages, dollars
    
    "has_table": bool,  # Contains structured table
    
    "has_quote": bool,  # Contains quoted text
    
    "contains_principle": bool,  # States an investing/business principle
    
    "contains_example": bool,  # Provides concrete example
    
    "contains_comparison": bool,  # Compares multiple entities
    
    # ============================================
    # CONTEXTUAL INFORMATION (Generated by LLM)
    # ============================================
    "contextual_summary": str,  # LLM-generated semantic summary
                                # Format: "This chunk discusses [main topic] 
                                # in the context of [year's situation]. 
                                # Key point: [main insight]"
    
    "prev_context": str,  # 1-2 sentence summary of preceding content
    
    "next_context": str,  # 1-2 sentence summary of following content
    
    # ============================================
    # EXTRACTED ENTITIES (Generated by LLM or NER)
    # ============================================
    "topics": list[str],  # ["moats", "insurance_float", "capital_allocation"]
    
    "companies_mentioned": list[str],  # ["GEICO", "Coca-Cola", "Wells Fargo"]
    
    "people_mentioned": list[str],  # ["Tony Nicely", "Ajit Jain"]
    
    "metrics_discussed": list[str],  # ["underwriting_profit", "float", "ROE"]
    
    "industries": list[str],  # ["insurance", "consumer_goods", "banking"]
    
    # ============================================
    # PRINCIPLE CLASSIFICATION (If contains_principle=True)
    # ============================================
    "principle_category": str | None,  
    # Options: "moats", "valuation", "management_quality", 
    #          "capital_allocation", "risk_management", 
    #          "competitive_advantage", "business_quality"
    
    "principle_statement": str | None,  # Extract the core principle
    
    # ============================================
    # RETRIEVAL METADATA (For filtering/ranking)
    # ============================================
    "retrieval_priority": str,  # "high" | "medium" | "low"
                                # high: philosophy, principles, key examples
                                # medium: business analysis, specific stories
                                # low: administrative, meeting details
    
    "abstraction_level": str,  # "high" | "medium" | "low"
                               # high: abstract principles
                               # medium: principles with examples
                               # low: specific facts/numbers
    
    "time_sensitivity": str,  # "high" | "low"
                              # high: specific to that year's conditions
                              # low: timeless principle
    
    # ============================================
    # QUALITY FLAGS (For validation)
    # ============================================
    "is_complete_thought": bool,  # Does chunk stand alone?
    
    "needs_context": bool,  # Requires prev/next chunks to understand?
    
    "validation_status": str,  # "pending" | "validated" | "needs_review"
}
```

---

## PART 3: CONTEXTUAL SUMMARY GENERATION

### Prompt Template for Contextual Summary

```python
CONTEXTUAL_SUMMARY_PROMPT = """
You are analyzing a chunk from Warren Buffett's {year} Letter to Shareholders.

SECTION CONTEXT:
- Section: {section_title}
- Position: Chunk {position} of {total_chunks} in this section
- Previous content summary: {prev_context}
- Following content summary: {next_context}

CHUNK TEXT:
{chunk_text}

TASK: Generate a contextual summary (2-3 sentences) that:
1. Identifies the MAIN topic/concept discussed
2. Explains the CONTEXT (why this matters in {year}, what situation it addresses)
3. States the KEY INSIGHT or principle (if any)

FORMAT:
"This chunk discusses [main topic] in the context of [year's situation/broader argument]. 
Buffett [explains/argues/demonstrates] that [key insight]. 
[Optional: Connection to broader principle]."

EXAMPLE OUTPUT:
"This chunk discusses GEICO's competitive advantage stemming from its low-cost operating 
structure in the context of the 1996 insurance market. Buffett explains that low costs 
enable low prices, which attract quality policyholders who refer others, creating a 
virtuous cycle that further reduces acquisition costs. This demonstrates his principle 
of seeking durable competitive moats."

YOUR CONTEXTUAL SUMMARY:
"""
```

### Prompt Template for Entity Extraction

```python
ENTITY_EXTRACTION_PROMPT = """
Extract structured information from this chunk of Buffett's {year} letter.

CHUNK TEXT:
{chunk_text}

Extract the following (return "none" if not present):

1. TOPICS: Core concepts discussed (e.g., "moats", "float", "intrinsic_value")
2. COMPANIES: Any companies mentioned by name
3. PEOPLE: Any people mentioned by name
4. METRICS: Financial or business metrics discussed (e.g., "ROE", "underwriting_profit")
5. INDUSTRIES: Industry sectors mentioned
6. PRINCIPLES: If this states an investing principle, extract it in one sentence

Return as JSON:
{
  "topics": [],
  "companies": [],
  "people": [],
  "metrics": [],
  "industries": [],
  "principle_statement": null or "..."
}
"""
```

---

## PART 4: IMPLEMENTATION WORKFLOW

### Step-by-Step Process

```python
# PHASE 1: Initial Text Extraction and Sectioning
def process_letter(filepath):
    """
    Input: Raw letter file (HTML/PDF/TXT)
    Output: Structured sections with metadata
    """
    # 1. Extract raw text
    raw_text = extract_text(filepath)
    
    # 2. Clean text (remove headers, footers, page numbers)
    clean_text = clean_extracted_text(raw_text)
    
    # 3. Identify sections
    sections = identify_sections(clean_text)
    
    # 4. Split each section into paragraphs
    for section in sections:
        section['paragraphs'] = split_into_paragraphs(section['text'])
    
    return sections

# PHASE 2: Chunk Creation
def create_chunks(sections, year):
    """
    Input: Structured sections
    Output: List of chunks with basic metadata
    """
    chunks = []
    
    for section in sections:
        section_chunks = []
        
        # Determine chunk type
        chunk_type = classify_section_content_type(section)
        
        if chunk_type == "financial_table":
            # Keep table + context together
            chunk = create_table_chunk(section)
            section_chunks.append(chunk)
            
        elif chunk_type == "narrative_story":
            # Keep story intact
            chunk = create_story_chunk(section)
            section_chunks.append(chunk)
            
        else:
            # Standard paragraph-based chunking
            section_chunks = create_paragraph_chunks(
                section, 
                target_size=200,
                min_size=150,
                max_size=300
            )
        
        # Add metadata to each chunk
        for idx, chunk in enumerate(section_chunks):
            chunk['metadata'] = generate_basic_metadata(
                chunk=chunk,
                section=section,
                position=idx,
                total=len(section_chunks),
                year=year
            )
        
        chunks.extend(section_chunks)
    
    return chunks

# PHASE 3: Contextual Enhancement
def enhance_chunks_with_context(chunks):
    """
    Input: Chunks with basic metadata
    Output: Chunks with LLM-generated contextual information
    """
    for i, chunk in enumerate(chunks):
        # Get surrounding context
        prev_chunk = chunks[i-1] if i > 0 else None
        next_chunk = chunks[i+1] if i < len(chunks)-1 else None
        
        # Generate contextual summary
        chunk['metadata']['contextual_summary'] = generate_contextual_summary(
            chunk=chunk,
            prev_chunk=prev_chunk,
            next_chunk=next_chunk
        )
        
        # Extract entities
        entities = extract_entities(chunk)
        chunk['metadata'].update(entities)
        
        # Classify if principle
        if chunk['metadata']['contains_principle']:
            chunk['metadata']['principle_category'] = classify_principle(chunk)
    
    return chunks

# PHASE 4: Validation
def validate_chunks(chunks):
    """
    Quality control checks
    """
    for chunk in chunks:
        # Check completeness
        chunk['metadata']['is_complete_thought'] = check_completeness(chunk)
        
        # Check boundaries
        assert not chunk['text'].strip().endswith(','), "Chunk ends mid-sentence"
        
        # Check size
        assert 50 <= chunk['metadata']['word_count'] <= 500, "Chunk size out of bounds"
        
        # Flag for review if needed
        if not chunk['metadata']['is_complete_thought']:
            chunk['metadata']['validation_status'] = 'needs_review'
        else:
            chunk['metadata']['validation_status'] = 'validated'
    
    return chunks
```

---

## PART 5: SPECIAL CASES & EDGE CASES

### Case 1: Cross-Section References
**Problem**: Buffett references earlier sections ("As I mentioned earlier...")

**Solution**: 
- Add `cross_references: list[chunk_id]` to metadata
- In contextual summary, note the reference
- Don't split the reference from its context

### Case 2: Multi-Year Comparisons
**Problem**: Some chunks compare performance across years

**Solution**:
- Add `years_referenced: list[int]` to metadata
- Tag as `contains_comparison: True`
- Extract specific years mentioned

### Case 3: Long Quotes
**Problem**: Extended quotes from others (e.g., Ben Graham, Charlie Munger)

**Solution**:
- Keep quote with surrounding context
- Add metadata: `has_quote: True, quote_source: "Ben Graham"`
- Include quote in chunk even if it pushes word count slightly over

### Case 4: Technical Explanations (e.g., Derivatives in 2002)
**Problem**: Complex multi-paragraph technical discussions

**Solution**:
- Break at conceptual sub-topics
- Use strong cross-referencing
- Add `technical_topic: "derivatives"` to metadata
- Higher `abstraction_level` tags

### Case 5: Lists of Companies/Holdings
**Problem**: Portfolio holdings table or list of acquisitions

**Solution**:
- Treat as `chunk_type: "financial_table"`
- Extract company names to `companies_mentioned`
- Include introductory/concluding paragraphs

---

## PART 6: VALIDATION CHECKLIST

Before finalizing chunks, verify:

```python
VALIDATION_CHECKLIST = {
    "boundary_checks": [
        "No chunks end mid-sentence",
        "No chunks break complete stories",
        "Tables paired with explanations",
        "Section headers included in each chunk"
    ],
    
    "metadata_completeness": [
        "All required fields populated",
        "chunk_id follows naming convention",
        "Position metadata calculated correctly",
        "Content flags accurately set"
    ],
    
    "contextual_quality": [
        "Contextual summaries are 2-3 sentences",
        "Summaries identify main topic + insight",
        "Entity extraction includes all mentions",
        "Principle classification is accurate"
    ],
    
    "size_distribution": [
        "90%+ of chunks between 150-300 words",
        "Stories <500 words if kept intact",
        "No chunks <50 words unless table-only",
        "Average chunk size ~200 words"
    ],
    
    "semantic_integrity": [
        "Each chunk understandable with metadata",
        "Context references accurate",
        "Topics correctly identified",
        "Priority levels appropriate"
    ]
}
```

---

## PART 7: OUTPUT FORMAT

### Final Corpus Structure

```
corpus/
├── chunks_with_metadata.parquet  # Main corpus file
├── metadata_schema.json           # Schema documentation
├── validation_report.json         # Quality metrics
└── by_year/
    ├── 1977_chunks.json
    ├── 1978_chunks.json
    └── ...
```

### Parquet Schema
```python
import pyarrow as pa

schema = pa.schema([
    ('chunk_id', pa.string()),
    ('year', pa.int32()),
    ('chunk_text', pa.string()),
    ('word_count', pa.int32()),
    ('section_type', pa.string()),
    ('chunk_type', pa.string()),
    ('contextual_summary', pa.string()),
    ('metadata_json', pa.string()),  # All other metadata as JSON string
])
```

---

## PART 8: EXAMPLE CHUNKS

### Example 1: Philosophy Chunk
```json
{
  "chunk_id": "1996_investments_042",
  "year": 1996,
  "section_type": "investments",
  "section_title": "Common Stock Investments",
  "chunk_type": "philosophy",
  "chunk_text": "Companies such as Coca-Cola and Gillette might well be labeled \"The Inevitables.\" Forecasters may differ a bit in their predictions of exactly how much soft drink or shaving-equipment business these companies will be doing in ten or twenty years. Nor is our talk of inevitability meant to play down the vital work that these companies must continue to carry out, in such areas as manufacturing, distribution, packaging and product innovation. In the end, however, no sensible observer – not even these companies' most vigorous competitors, assuming they are assessing the matter honestly – questions that Coke and Gillette will dominate their fields worldwide for an investment lifetime.",
  "word_count": 118,
  "contains_principle": true,
  "principle_category": "competitive_advantage",
  "principle_statement": "Some companies ('The Inevitables') possess competitive advantages so durable that their long-term dominance is virtually certain.",
  "contextual_summary": "This chunk introduces Buffett's concept of 'The Inevitables' in the context of explaining his investment in Coca-Cola and Gillette. He argues that certain companies possess competitive advantages so durable that their long-term market dominance is virtually certain, making them ideal long-term investments. This reflects his core principle of seeking businesses with predictable, enduring competitive moats.",
  "companies_mentioned": ["Coca-Cola", "Gillette"],
  "topics": ["competitive_advantage", "moats", "predictability", "long_term_investing"],
  "retrieval_priority": "high",
  "abstraction_level": "high"
}
```

### Example 2: Business Analysis Chunk
```json
{
  "chunk_id": "1996_insurance_008",
  "year": 1996,
  "section_type": "insurance_operations",
  "section_title": "Insurance - GEICO and Other Primary Operations",
  "subsection": "GEICO",
  "chunk_type": "business_analysis",
  "chunk_text": "There's nothing esoteric about GEICO's success: The company's competitive strength flows directly from its position as a low-cost operator. Low costs permit low prices, and low prices attract and retain good policyholders. The final segment of a virtuous circle is drawn when policyholders recommend us to their friends. GEICO gets more than one million referrals annually and these produce more than half of our new business, an advantage that gives us enormous savings in acquisition expenses – and that makes our costs still lower.",
  "word_count": 89,
  "contains_principle": true,
  "contains_example": true,
  "principle_category": "competitive_advantage",
  "principle_statement": "Low-cost operations create a virtuous cycle: lower costs → lower prices → customer acquisition → referrals → even lower costs.",
  "contextual_summary": "This chunk explains GEICO's competitive advantage in the 1996 insurance market, emphasizing how its low-cost structure creates a self-reinforcing virtuous cycle. Buffett demonstrates that GEICO's operational efficiency leads to customer referrals (1 million annually producing 50%+ of new business), which further reduces costs. This exemplifies his principle of seeking businesses with compounding competitive advantages.",
  "companies_mentioned": ["GEICO"],
  "people_mentioned": [],
  "metrics_discussed": ["acquisition_expenses", "referral_rate"],
  "topics": ["low_cost_operator", "virtuous_cycle", "customer_referrals", "competitive_moat"],
  "has_financials": true,
  "retrieval_priority": "high",
  "abstraction_level": "medium"
}
```

### Example 3: Narrative Story Chunk
```json
{
  "chunk_id": "1996_acquisitions_003",
  "year": 1996,
  "section_type": "acquisitions",
  "section_title": "Acquisitions of 1996",
  "chunk_type": "narrative_story",
  "chunk_text": "You might be interested in the carefully-crafted and sophisticated acquisition strategy that allowed Berkshire to nab this deal. Early in 1996 I was invited to the 40th birthday party of my nephew's wife, Jane Rogers. My taste for social events being low, I immediately, and in my standard, gracious way, began to invent reasons for skipping the event. The party planners then countered brilliantly by offering me a seat next to a man I always enjoy, Jane's dad, Roy Dinsdale - so I went. The party took place on January 26. Though the music was loud - Why must bands play as if they will be paid by the decibel? - I just managed to hear Roy say he'd come from a directors meeting at Kansas Bankers Surety, a company I'd always admired. I shouted back that he should let me know if it ever became available for purchase. On February 12, I got the following letter from Roy: \"Dear Warren: Enclosed is the annual financial information on Kansas Bankers Surety. This is the company that we talked about at Janie's party. If I can be of any further help, please let me know.\" On February 13, I told Roy we would pay $75 million for the company - and before long we had a deal. I'm now scheming to get invited to Jane's next party.",
  "word_count": 239,
  "contains_example": true,
  "contains_principle": false,
  "story_subject": "Kansas Bankers Surety acquisition",
  "contextual_summary": "This chunk tells the serendipitous story of how Berkshire acquired Kansas Bankers Surety through a casual conversation at a birthday party in early 1996. Buffett uses humor to illustrate that valuable acquisition opportunities can arise unexpectedly through personal relationships and a reputation for fair dealing. The story demonstrates his principle that being known as a good buyer creates deal flow.",
  "companies_mentioned": ["Kansas Bankers Surety", "Berkshire Hathaway"],
  "people_mentioned": ["Warren Buffett", "Roy Dinsdale", "Jane Rogers"],
  "has_financials": true,
  "topics": ["acquisitions", "deal_sourcing", "reputation"],
  "retrieval_priority": "medium",
  "abstraction_level": "low",
  "time_sensitivity": "high"
}
```

---

## PART 9: IMPLEMENTATION NOTES

### Technology Stack
- **PDF Processing**: `pdfplumber` or `pymupdf`
- **HTML Processing**: `beautifulsoup4`
- **Text Processing**: `nltk` or `spaCy`
- **LLM API**: `openai` or `anthropic`
- **Data Storage**: `pandas` + `pyarrow` (Parquet)
- **Validation**: Custom Python scripts

### Performance Considerations
- **Batch LLM calls**: Process 50-100 chunks per API call for efficiency
- **Caching**: Cache LLM-generated summaries to avoid re-processing
- **Parallel processing**: Process multiple letters simultaneously
- **Incremental updates**: New letters can be added without reprocessing entire corpus

### Version Control
- Tag corpus versions: `v1.0.0` (initial), `v1.1.0` (improved chunking), etc.
- Track changes to chunking strategy
- Maintain backward compatibility with embeddings

---

## APPENDIX A: Common Section Patterns

### Pattern Recognition for Section Headers
```python
SECTION_HEADER_PATTERNS = {
    "insurance": r"Insurance|Underwriting|Float|GEICO|General Re|Reinsurance",
    "acquisitions": r"Acquisitions?|New Business|Purchases?",
    "investments": r"Investments?|Portfolio|Common Stock|Securities",
    "performance": r"Performance|Results|Book Value|Net Worth",
    "governance": r"Governance|Board|Audit|Compensation",
    "meeting": r"Annual Meeting|Shareholder"
}
```

---

## APPENDIX B: Quality Metrics

Track these metrics across the corpus:

```python
QUALITY_METRICS = {
    "chunk_count_total": int,
    "avg_chunk_size_words": float,
    "chunk_size_std_dev": float,
    "pct_chunks_with_principles": float,
    "pct_chunks_with_examples": float,
    "sections_per_letter_avg": float,
    "chunks_per_section_avg": float,
    "companies_mentioned_unique": int,
    "people_mentioned_unique": int,
    "topics_identified_unique": int
}
```

Target benchmarks:
- Average chunk size: 200 ± 50 words
- Principle chunks: 15-20% of total
- Example chunks: 30-40% of total
- Complete thoughts: 95%+ of chunks

---

END OF CHUNKING STRATEGY DOCUMENT