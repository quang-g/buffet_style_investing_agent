# Chunking Strategy for Warren Buffett Shareholder Letters
## Zero-Discretion Execution Framework

---

## 0. Canonical Chunk Schema (MANDATORY CONFORMANCE)

Every chunk emitted MUST match this schema exactly. No field may be omitted. No additional fields may be added.

### 0.1 Top-Level Chunk Object

```json
{
  "chunk_id": "2016-S3-T1-002",
  "content": "…full chunk text…",
  "metadata": { "…see §0.2…" }
}
```

**`chunk_id` Format (STRICT REGEX)**
- Pattern: `^\d{4}-S\d+-T[123]-\d{3}$`
- Components: `YYYY-S{section_index}-T{tier}-{sequence_in_section}`
- `tier` values: `1` (section/narrative), `2` (paragraph), `3` (table/data)
- `sequence_in_section`: zero-padded 3 digits starting from `001`

### 0.2 Metadata Object (ALL KEYS REQUIRED)

```json
{
  "letter_year": 2016,
  "letter_date": "2017-02-25",
  "source_file": "2016_cleaned.txt",

  "section_title": "Insurance Operations",
  "section_hierarchy": ["Letter Body", "Business Segments", "Insurance"],

  "chunk_tier": 1,
  "parent_chunk_id": null,
  "child_chunk_ids": [],

  "content_type": "narrative",
  "contextual_summary": "Describes GEICO's underwriting discipline and float generation strategy in 2016.",

  "has_table": false,
  "table_data": [],
  "has_financial_data": true,

  "entities": { "companies": [], "people": [], "metrics": [] },
  "themes": [],
  "buffett_concepts": [],
  "principles": [],

  "temporal_references": { "primary_year": 2016, "comparison_years": [], "future_outlook": false },
  "cross_references": { "related_sections_same_letter": [], "related_years": [] },

  "retrieval_priority": "high",
  "abstraction_level": "medium",

  "token_count": 0,
  "char_count": 0,

  "source_span": { "start_char": 0, "end_char_exclusive": 0 },

  "standalone_exception": false,
  "exception_reason": null,

  "boundary_note": null,
  "merged_from": []
}
```

### 0.3 Type and Value Constraints (ENFORCE AT RUNTIME)

#### Scalar Types (MUST MATCH EXACTLY)
- `letter_year`: `int` (4-digit year)
- `letter_date`: `string` format `YYYY-MM-DD` OR `null` (if null, set `boundary_note`)
- `source_file`: `string` (non-empty)
- `chunk_tier`: `int` ∈ {1, 2, 3}
- `token_count`: `int` ≥ 0
- `char_count`: `int` ≥ 0

#### Enum Types (RESTRICTED VALUES)
- `content_type` ∈ {"narrative", "financial_table", "mistake_confession", "principle_statement"}
- `retrieval_priority` ∈ {"low", "medium", "high"}
- `abstraction_level` ∈ {"low", "medium", "high"}

#### Compound Types (EXACT STRUCTURE)
- `section_hierarchy`: `list[string]` (min length 1)
- `entities`: `dict` with exactly 3 keys: `companies`, `people`, `metrics` (each `list[string]`)
- `table_data`: `list[TableData]` (may be empty `[]`, never `null`)
- `principles`: `list[dict]` where each dict has `statement: string` and `category: string`
- `temporal_references`: `dict` with exactly 3 keys: `primary_year: int`, `comparison_years: list[int]`, `future_outlook: bool`
- `cross_references`: `dict` with exactly 2 keys: `related_sections_same_letter: list[string]`, `related_years: list[int]`
- `source_span`: `dict` with exactly 2 keys: `start_char: int`, `end_char_exclusive: int`

#### Invariant Constraints (MUST HOLD TRUE)
1. `chunk_tier` value MUST equal the tier digit in `chunk_id`
2. IF `content_type == "financial_table"` THEN `has_table == true` AND `len(table_data) >= 1`
3. IF `len(table_data) >= 1` THEN `has_table == true`
4. IF `has_table == false` THEN `table_data == []`
5. IF `chunk_tier == 2` THEN `parent_chunk_id != null`
6. IF `chunk_tier == 3` THEN `parent_chunk_id != null`
7. IF `standalone_exception == true` THEN `exception_reason != null`
8. `source_span.start_char < source_span.end_char_exclusive` (unless both are 0 with `boundary_note` explanation)

### 0.4 TableData Schema (WHEN USED)

```json
{
  "table_name": "Insurance Operations Results",
  "summary": "Compares underwriting profit and float across insurance units for 2009 vs 2008.",
  "columns": ["Operation", "Underwriting Profit 2009", "Underwriting Profit 2008", "Float 2009", "Float 2008"],
  "rows": [
    ["General Re", "$477M", "$342M", "$21,014M", "$21,074M"],
    ["GEICO", "$649M", "$916M", "$9,613M", "$8,454M"]
  ],
  "row_count": 2
}
```

**Type Requirements:**
- `table_name`: `string` (non-empty)
- `summary`: `string` (1-2 sentences)
- `columns`: `list[string]` (min length 1)
- `rows`: `list[list[string]]` (each sublist length must equal `len(columns)`)
- `row_count`: `int` (must equal `len(rows)`)

---

## 1. Document Structure Analysis

### 1.1 Section Detection Algorithm (DETERMINISTIC)

**Step 1: Identify Section Headers**
A line is a section header IF:
- It is ALL CAPS (excluding articles/prepositions if short), OR
- It appears in isolation (empty lines before/after) and is NOT part of a sentence, OR
- It follows a pattern: `SECTION_NAME:` or `# SECTION_NAME`

**Step 2: Identify Section Boundaries**
- Start: First line of content after header
- End: Line before next section header OR asterisk separator OR signature block

**Step 3: Extract Section Hierarchy**
Build hierarchy from document structure:
- Top-level sections get hierarchy depth 1
- Subsections inherit parent hierarchy + current title

### 1.2 Separator Recognition (EXACT PATTERNS)

**Asterisk Separator:**
- Pattern: 10+ consecutive asterisks with optional spaces
- Action: Treat as section boundary (same weight as section header)

**Page Break:**
- Pattern: `[[PAGE_BREAK]]` or similar marker
- Action: Ignore for chunking (do NOT use as boundary unless combined with content gap)

**Signature Block:**
- Pattern: Name followed by date in format `Month DD, YYYY` within last 10 lines of document
- Action: Marks end of letter body, start of metadata section

---

## 2. Tier Assignment and Chunking Rules (ZERO DISCRETION)

### 2.1 Tier Decision Tree (EXECUTE IN ORDER)

For each content unit candidate, execute this decision tree TOP TO BOTTOM. The FIRST matching rule determines the tier.

```
START
│
├─ Contains table structure (rows/columns)?
│  ├─ YES → Assign Tier 3, proceed to §2.3
│  └─ NO → Continue
│
├─ Is this a complete section (with header)?
│  ├─ YES → Assign Tier 1, proceed to §2.2
│  └─ NO → Continue
│
├─ Is this content currently < 250 tokens?
│  ├─ YES → MERGE (do not assign tier yet), apply §2.4
│  └─ NO → Continue
│
├─ Is this a paragraph within an already-assigned Tier-1 chunk?
│  ├─ YES → Check Tier-2 criteria (§2.2.3)
│  │  ├─ Meets criteria → Assign Tier 2
│  │  └─ Fails criteria → ABSORB into Tier 1 (no new chunk)
│  └─ NO → Assign Tier 1 (default)
│
END
```

### 2.2 Tier-1 Chunking Rules (SECTIONS & NARRATIVES)

**Rule 2.2.1: Section Boundary Detection**
A new Tier-1 chunk MUST start when encountering:
- Section header (per §1.1), OR
- Asterisk separator (per §1.2), OR
- Semantic shift marker: "Now, regarding...", "Turning to...", "Let me explain..." IF preceded by paragraph break

**Rule 2.2.2: Minimum Size Enforcement**
- Minimum: 250 tokens (~180-220 words)
- Measurement: Use tiktoken `cl100k_base` encoding
- Enforcement: If candidate < 250 tokens at boundary, apply §2.4 merge rules BEFORE finalizing chunk

**Rule 2.2.3: Maximum Size Guideline**
- Target: 800 tokens (~600 words)
- Hard ceiling: 1200 tokens (~900 words)
- IF chunk reaches 1200 tokens AND natural paragraph break exists within next 100 tokens:
  - Split at paragraph break
  - Ensure both resulting chunks ≥ 250 tokens (else, keep as single chunk)

**Rule 2.2.4: Content Coherence Preservation**
NEVER split across:
- Numbered lists (keep all items together)
- Multi-paragraph teaching stories (check for narrative continuity markers)
- Quote + attribution (if separated by < 2 lines, keep together)
- Table + lead-in + interpretation (covered in §2.3)

### 2.3 Tier-3 Chunking Rules (TABLES & FINANCIAL DATA)

**Rule 2.3.1: Table Chunk Composition (MANDATORY INCLUSION)**
Every Tier-3 chunk MUST contain ALL of:
1. Lead-in text: 1-3 paragraphs immediately BEFORE table (if present)
2. Table structure: Converted to `table_data` format (§0.4)
3. Interpretation text: 1-2 paragraphs immediately AFTER table (if present)

**Rule 2.3.2: Lead-in Distance Threshold**
- IF distance between paragraph and table ≤ 2 line breaks → Include as lead-in
- IF distance > 2 line breaks → Do NOT include as lead-in (treat as separate Tier-1 content)

**Rule 2.3.3: Interpretation Distance Threshold**
- IF paragraph immediately follows table (0-1 line breaks) → Include as interpretation
- IF paragraph is 2+ line breaks after table → Do NOT include (treat as separate Tier-1 content)

**Rule 2.3.4: Multiple Tables in Proximity**
- IF 2+ tables separated by < 3 paragraphs → Create ONE Tier-3 chunk containing all tables + shared context
- IF tables separated by ≥ 3 paragraphs → Create separate Tier-3 chunks per table

**Rule 2.3.5: Tier-3 Size Limits**
- No minimum (tables can be small)
- Maximum: 1000 tokens including all context
- IF exceeds 1000 tokens: Include lead-in + table, truncate interpretation at paragraph boundary

**Rule 2.3.6: Tier-3 Metadata Requirements**
MUST set:
- `content_type = "financial_table"`
- `has_table = true`
- `table_data = [...]` (at least one TableData object)
- `chunk_tier = 3`
- `parent_chunk_id = {corresponding Tier-1 section ID}`

### 2.4 Anti-Orphan Rules (CRITICAL - NO EXCEPTIONS)

**Rule 2.4.1: No Orphan Tier-1 Chunks**
IF a Tier-1 chunk candidate < 250 tokens at section boundary:
1. Check if it is a lead-in (introduces next topic) → Merge FORWARD into next chunk
2. Check if it is a conclusion (summarizes prior topic) → Merge BACKWARD into previous chunk
3. If ambiguous: Default to FORWARD merge (prevents tail fragmentation)

**Decision Matrix for Lead-in Detection:**
```
Is last sentence a question? → Lead-in → FORWARD
Contains "Now", "Next", "Turning to", "Let me"? → Lead-in → FORWARD
Contains "In summary", "To conclude", "As mentioned"? → Conclusion → BACKWARD
Contains forward reference ("below", "following")? → Lead-in → FORWARD
Contains backward reference ("above", "previously")? → Conclusion → BACKWARD
Default case → FORWARD
```

**Rule 2.4.2: No Header-Only Chunks**
IF chunk begins with section header AND remainder < 250 tokens:
- Continue absorbing paragraphs until 250 token minimum is reached
- IF next section header is encountered before reaching 250 tokens:
  - Apply Rule 2.4.1 merge logic to the short section

**Rule 2.4.3: No Tail Fragments**
IF final paragraphs of a section would form < 250 token chunk:
- Check if they introduce a new topic (semantic shift marker present)
  - YES → Merge FORWARD into next section
  - NO → Merge BACKWARD into current chunk (expand current chunk boundary)

**Rule 2.4.4: Single-Sentence Transitions**
IF content unit is 1 sentence AND contains transition words ("Now", "However", "Meanwhile"):
- ALWAYS merge with following content (FORWARD merge)
- NEVER emit as standalone chunk

### 2.5 Tier-2 Prevention Rules (ELIMINATE LEAKAGE)

**Rule 2.5.1: Tier-2 Creation Gate (STRICT CRITERIA)**
Create Tier-2 chunk ONLY IF ALL conditions are met:
1. Content is semantically self-contained (tests for self-containment in §2.5.2)
2. Content is ≥ 120 tokens
3. Content serves high-value retrieval purpose (principle statement, confession, or quotable insight)
4. Content does NOT interrupt a narrative flow (check for story markers)
5. Parent Tier-1 chunk would remain ≥ 250 tokens after extraction

**Rule 2.5.2: Self-Containment Test (AUTOMATED)**
Content is self-contained IF:
- First sentence has explicit subject (not pronoun referencing prior content)
- Contains no unresolved references ("this approach", "that situation" without antecedent)
- Can answer: "What is this about?" without reading parent chunk

**Automated Pronoun Check:**
```python
first_sentence_starts_with = ["This", "That", "These", "Those", "It", "They", "Such"]
if any(first_sentence.startswith(word) for word in first_sentence_starts_with):
    FAIL self-containment test → Do NOT create Tier-2
```

**Rule 2.5.3: Default to Tier-1 Absorption**
IF ANY Tier-2 criterion from §2.5.1 is FALSE:
- ABSORB content into Tier-1 chunk
- Do NOT create Tier-2 chunk
- Set `boundary_note = "Tier-2 candidate absorbed due to [criterion name]"`

**Rule 2.5.4: Tier-2 Size Enforcement**
- Minimum: 120 tokens (no exceptions)
- Maximum: 400 tokens
- IF candidate ≥ 120 AND ≤ 400 AND passes §2.5.1 → Create Tier-2
- IF candidate > 400 → Keep as part of Tier-1 (too large for sub-chunk)

**Rule 2.5.5: Tier-2 Pairing Prevention**
Do NOT create consecutive Tier-2 chunks from same Tier-1 parent.
IF multiple paragraphs each qualify for Tier-2:
- Merge them into a SINGLE Tier-2 chunk (if combined size ≤ 400 tokens)
- OR keep all as part of Tier-1 (if combined size > 400 tokens)

### 2.6 Exception Handling (RARE, DOCUMENTED)

**Rule 2.6.1: Standalone Exception Criteria**
Grant `standalone_exception = true` ONLY IF ALL conditions met:
1. Content is a complete, self-contained principle statement
2. Content is 100-249 tokens (just below Tier-1 minimum)
3. Splitting or merging would REDUCE retrieval value
4. Content is high-priority: principle, confession, or critical insight

**Rule 2.6.2: Exception Documentation**
MUST set:
- `standalone_exception = true`
- `exception_reason = {1 sentence explaining why exception is warranted}`
- `retrieval_priority = "high"`

**Rule 2.6.3: Exception Limit**
- Maximum 5% of total chunks may use standalone exceptions
- IF exception count exceeds 5%, review and consolidate exceptions

---

## 3. Implementation Workflow (ORDERED EXECUTION)

### 3.1 Processing Pipeline (MUST EXECUTE IN THIS ORDER)

```
Step 1: Document Parsing
├─ Load source file
├─ Extract sections (§1.1)
├─ Identify separators (§1.2)
└─ Build section hierarchy

Step 2: Initial Tier Assignment
├─ Apply Tier Decision Tree (§2.1) to each section
├─ Identify all table chunks → Assign Tier 3 (§2.3)
├─ Identify all section-level chunks → Assign Tier 1 (§2.2)
└─ DO NOT assign Tier 2 yet

Step 3: Size Enforcement
├─ Measure all Tier-1 candidates (token count)
├─ Apply anti-orphan rules (§2.4) to candidates < 250 tokens
├─ Execute merges as determined by Rules 2.4.1-2.4.4
└─ Revalidate: NO Tier-1 chunk < 250 tokens (except standalone exceptions)

Step 4: Tier-3 Context Assembly
├─ For each table chunk:
│  ├─ Apply lead-in inclusion rules (§2.3.2)
│  ├─ Apply interpretation inclusion rules (§2.3.3)
│  └─ Set parent_chunk_id to containing Tier-1 section
└─ Revalidate: ALL tables have lead-in OR interpretation

Step 5: Tier-2 Evaluation (ONLY IF NEEDED)
├─ For each Tier-1 chunk > 800 tokens:
│  ├─ Identify paragraph boundaries
│  ├─ Test each paragraph against Tier-2 criteria (§2.5.1)
│  ├─ IF passes ALL criteria → Extract as Tier-2
│  └─ ELSE → Keep in Tier-1
└─ Apply Tier-2 prevention rules (§2.5.3-2.5.5)

Step 6: Metadata Population
├─ Generate chunk_id per §0.1 format
├─ Populate ALL metadata fields per §0.2
├─ Compute token_count and char_count
├─ Record source_span from original file
└─ Set boundary_note for merges/tradeoffs

Step 7: Schema Validation
├─ Run type checker on every chunk
├─ Validate invariant constraints (§0.3)
├─ Check for missing required fields
├─ IF validation fails → HALT with error report
└─ ELSE → Proceed to output

Step 8: Quality Assurance
├─ Execute QA checklist (§6)
├─ Verify no orphan chunks remain
├─ Verify all tables have context
└─ Log statistics (chunk counts by tier, min/max sizes)
```

### 3.2 Content Type Assignment (RULE-BASED)

**Rule 3.2.1: Automatic Content Type Detection**
Execute these checks in order:

```
IF chunk contains table_data with len ≥ 1:
    content_type = "financial_table"

ELIF chunk contains confession markers:
    Markers: "mistake", "error", "wrong", "regret", "should not have", "failed to"
    AND chunk describes resolution OR lesson learned
    content_type = "mistake_confession"

ELIF chunk is standalone principle (§2.6.1 criteria):
    content_type = "principle_statement"

ELSE:
    content_type = "narrative"
```

### 3.3 Metadata Enrichment (REQUIRED FIELDS)

**Rule 3.3.1: Entity Extraction (AUTOMATED)**
```python
# Use spaCy or similar NER
entities.companies = extract_orgs(content)
entities.people = extract_persons(content)
entities.metrics = extract_metrics(content)  # Use domain lexicon for financial terms
```

**Rule 3.3.2: Theme & Concept Assignment**
Use predefined taxonomy:

**Themes:** (select all that apply)
- insurance_operations
- investment_strategy
- capital_allocation
- corporate_governance
- business_valuation
- management_quality
- risk_management
- shareholder_relations

**Buffett Concepts:** (select all that apply)
- float_economics
- margin_of_safety
- circle_of_competence
- Mr_Market
- owner_earnings
- intrinsic_value
- moat
- look_through_earnings
- opportunity_cost

**Rule 3.3.3: Principle Extraction (PATTERN MATCHING)**
Identify sentences matching:
- "We [verb] when/only when [condition]"
- "Our approach to [X] is [principle]"
- "[Action] is important/essential/critical because [reason]"

Extract as:
```json
{
  "statement": "We repurchase shares only when they are meaningfully below intrinsic value.",
  "category": "capital_allocation"
}
```

**Principle Categories:**
- capital_allocation
- valuation
- risk_assessment
- management_evaluation
- insurance_underwriting
- accounting_practices
- behavioral_finance
- governance_philosophy
- general_business

**Rule 3.3.4: Temporal Reference Extraction**
```python
primary_year = letter_year  # Default
comparison_years = extract_years(content)  # All years mentioned
future_outlook = contains_forward_looking_statements(content)
```

Forward-looking markers: "expect", "will", "plan to", "anticipate", "going forward", "next year"

**Rule 3.3.5: Retrieval Priority Assignment (SCORING)**
Calculate priority score:
```
score = 0
IF content_type == "financial_table": score += 30
IF content_type == "mistake_confession": score += 25
IF content_type == "principle_statement": score += 20
IF len(principles) > 0: score += (10 * len(principles))
IF has_financial_data: score += 15
IF len(entities.companies) > 2: score += 10
IF len(buffett_concepts) > 0: score += (5 * len(buffett_concepts))

Assign:
score >= 40 → "high"
score >= 20 → "medium"
score < 20 → "low"
```

**Rule 3.3.6: Abstraction Level Assignment (CONTENT ANALYSIS)**
```
concrete_markers = count(numbers, company_names, specific_events, dates)
abstract_markers = count(philosophical_terms, general_principles, metaphors)

ratio = abstract_markers / (concrete_markers + abstract_markers)

ratio >= 0.7 → "high"
ratio >= 0.3 → "medium"
ratio < 0.3 → "low"
```

---

## 4. Special Content Handling (MANDATORY RULES)

### 4.1 Teaching Moments & Analogies (PRESERVE INTEGRITY)

**Rule 4.1.1: Narrative Continuity Detection**
Identify teaching narratives by:
- Multi-paragraph structure with storyline
- Contains setup → development → lesson/conclusion
- Uses analogies or examples to illustrate abstract concepts

**Rule 4.1.2: Narrative Preservation**
IF narrative spans multiple paragraphs:
- Keep entire narrative in SAME Tier-1 chunk (even if exceeds 800 tokens)
- Do NOT extract any paragraph as Tier-2
- Set `boundary_note = "Narrative preserved, exceeds target size"`

### 4.2 Investment Philosophy Sections (KEEP WHOLE)

**Rule 4.2.1: Philosophy Section Detection**
Sections matching these titles/themes:
- "How We Think About X"
- "Our Approach to Y"
- "Investment Principles"
- "Valuation Methodology"

**Rule 4.2.2: Philosophy Chunk Integrity**
- Keep entire philosophy section as ONE Tier-1 chunk
- Minimum size does NOT apply (philosophy sections are always substantial)
- Do NOT create Tier-2 sub-chunks from philosophy sections
- Set `retrieval_priority = "high"`

### 4.3 Table Context Requirements (STRICT ASSEMBLY)

**Rule 4.3.1: Table Chunk Assembly Sequence**
```
1. Locate table structure
2. Scan backward: Find lead-in (max 3 paragraphs, within 2 line breaks)
3. Scan forward: Find interpretation (max 2 paragraphs, within 1 line break)
4. Assemble: [lead-in] + [table] + [interpretation]
5. Validate: Chunk must include at least ONE of (lead-in OR interpretation)
6. IF neither present: Set boundary_note = "Table lacks context; standalone table chunk"
```

**Rule 4.3.2: Table Context Validation**
IF table chunk lacks BOTH lead-in AND interpretation:
- Search parent Tier-1 section for context
- Add cross-reference in metadata: `cross_references.related_sections_same_letter = [parent_section_title]`
- Set `retrieval_priority = "medium"` (reduced due to lack of context)

### 4.4 Confessions & Mistakes (HIGH-VALUE HANDLING)

**Rule 4.4.1: Confession Detection (KEYWORD + STRUCTURE)**
A confession chunk contains:
- Mistake acknowledgment (keywords: "mistake", "error", "wrong", "should have")
- Impact description (what went wrong)
- Lesson learned (what was gained from the experience)

**Rule 4.4.2: Confession Chunk Boundaries**
- Start: First sentence acknowledging mistake
- End: Last sentence describing lesson OR resolution
- Keep confession + lesson together (NEVER split)
- Set `content_type = "mistake_confession"`
- Set `retrieval_priority = "high"`

### 4.5 Transition Sentence Handling (ALWAYS MERGE)

**Rule 4.5.1: Transition Patterns (EXHAUSTIVE LIST)**
```
Transition markers:
- "Now, [topic]..."
- "Turning to [topic]..."
- "Before discussing [topic]..."
- "Let me explain [topic]..."
- "Another [topic]..."
- "Moving on to [topic]..."
- "Regarding [topic]..."
- "As for [topic]..."
```

**Rule 4.5.2: Transition Merge Logic**
IF sentence matches transition pattern:
- ALWAYS merge FORWARD into next paragraph
- Do NOT create standalone chunk
- Do NOT include in previous chunk
- Exception: If transition is last sentence of section, merge BACKWARD

### 4.6 Section Stub Handling (FORCED MERGE)

**Rule 4.6.1: Stub Detection**
Section stub = section header + content < 150 tokens

**Rule 4.6.2: Stub Merge Priority**
```
1. IF stub introduces next topic → Merge FORWARD
2. IF stub concludes previous topic → Merge BACKWARD
3. IF stub is isolated (no clear relation) → Merge FORWARD (default)
```

---

## 5. Quality Assurance & Validation (AUTOMATED GATES)

### 5.1 Schema Validation (BLOCKING ERRORS)

**Validator 5.1.1: Type Conformance**
For each chunk, verify:
```python
assert type(chunk["chunk_id"]) == str
assert re.match(r'^\d{4}-S\d+-T[123]-\d{3}$', chunk["chunk_id"])
assert type(chunk["content"]) == str
assert len(chunk["content"]) > 0
assert type(chunk["metadata"]) == dict
assert len(chunk.keys()) == 3  # Only chunk_id, content, metadata
```

**Validator 5.1.2: Metadata Completeness**
```python
required_keys = [
    "letter_year", "letter_date", "source_file",
    "section_title", "section_hierarchy",
    "chunk_tier", "parent_chunk_id", "child_chunk_ids",
    "content_type", "contextual_summary",
    "has_table", "table_data", "has_financial_data",
    "entities", "themes", "buffett_concepts", "principles",
    "temporal_references", "cross_references",
    "retrieval_priority", "abstraction_level",
    "token_count", "char_count", "source_span",
    "standalone_exception", "exception_reason",
    "boundary_note", "merged_from"
]
assert all(key in chunk["metadata"] for key in required_keys)
```

**Validator 5.1.3: Invariant Checks**
```python
# Invariant 1: chunk_tier matches chunk_id
tier_from_id = int(chunk["chunk_id"].split('-')[2][1])
assert chunk["metadata"]["chunk_tier"] == tier_from_id

# Invariant 2: table consistency
if chunk["metadata"]["content_type"] == "financial_table":
    assert chunk["metadata"]["has_table"] == True
    assert len(chunk["metadata"]["table_data"]) >= 1

# Invariant 3: table_data list type
assert type(chunk["metadata"]["table_data"]) == list

# Invariant 4: has_table boolean consistency
assert chunk["metadata"]["has_table"] == (len(chunk["metadata"]["table_data"]) > 0)

# Invariant 5: parent_chunk_id for Tier 2/3
if chunk["metadata"]["chunk_tier"] in [2, 3]:
    assert chunk["metadata"]["parent_chunk_id"] is not None
```

### 5.2 Boundary Quality Checks (WARNING LEVEL)

**Check 5.2.1: No Mid-Sentence Breaks**
```python
assert not chunk["content"].endswith(",")
assert not chunk["content"].endswith(";")
assert not chunk["content"].endswith("and")
assert not chunk["content"].endswith("or")
# Content should end with: . ! ? " )
```

**Check 5.2.2: No Split Lists**
```python
# If content contains "(1)" or "1.", ensure all list items are present
if re.search(r'\(\d+\)|\d+\.', chunk["content"]):
    numbers_found = extract_list_numbers(chunk["content"])
    assert numbers_found == list(range(1, max(numbers_found) + 1))  # Sequential
```

**Check 5.2.3: Quote-Attribution Pairing**
```python
if '"' in chunk["content"]:
    # Ensure no orphan quotes (must have closing quote)
    assert chunk["content"].count('"') % 2 == 0
    # Ensure attribution within 2 sentences of quote
```

### 5.3 Size Distribution Validation (STATISTICAL CHECKS)

**Check 5.3.1: Tier-1 Size Distribution**
```python
tier1_chunks = [c for c in chunks if c["metadata"]["chunk_tier"] == 1]
tier1_sizes = [c["metadata"]["token_count"] for c in tier1_chunks]

# No chunk below 250 tokens (except standalone exceptions)
non_exceptions = [s for c, s in zip(tier1_chunks, tier1_sizes) 
                  if not c["metadata"]["standalone_exception"]]
assert all(s >= 250 for s in non_exceptions)

# No chunk above 1200 tokens (unless narrative preservation)
oversized = [c for c in tier1_chunks if c["metadata"]["token_count"] > 1200]
for chunk in oversized:
    assert "narrative preserved" in chunk["metadata"].get("boundary_note", "").lower()
```

**Check 5.3.2: Tier-2 Size Distribution**
```python
tier2_chunks = [c for c in chunks if c["metadata"]["chunk_tier"] == 2]
tier2_sizes = [c["metadata"]["token_count"] for c in tier2_chunks]

# All Tier-2 chunks >= 120 tokens
assert all(s >= 120 for s in tier2_sizes)

# All Tier-2 chunks <= 400 tokens
assert all(s <= 400 for s in tier2_sizes)
```

**Check 5.3.3: Exception Rate**
```python
exception_count = sum(1 for c in chunks if c["metadata"]["standalone_exception"])
exception_rate = exception_count / len(chunks)
assert exception_rate <= 0.05  # Max 5%
```

### 5.4 Tier-2 Leakage Detection (CRITICAL)

**Detector 5.4.1: Self-Containment Audit**
For each Tier-2 chunk:
```python
content = chunk["content"]
first_sentence = content.split('.')[0]

# Check for unresolved pronouns
pronouns = ["This", "That", "These", "Those", "It", "They", "Such"]
if any(first_sentence.strip().startswith(p) for p in pronouns):
    # Flag as potential leakage
    warnings.append(f"Tier-2 chunk {chunk['chunk_id']} may lack self-containment")

# Check for forward/backward references
references = ["above", "below", "previously", "earlier", "following", "as mentioned"]
if any(ref in content.lower() for ref in references):
    warnings.append(f"Tier-2 chunk {chunk['chunk_id']} contains contextual reference")
```

**Detector 5.4.2: Parent Relationship Validation**
```python
# Ensure all Tier-2 chunks have valid parent
tier2_parents = set(c["metadata"]["parent_chunk_id"] for c in tier2_chunks)
tier1_ids = set(c["chunk_id"] for c in tier1_chunks)
assert tier2_parents.issubset(tier1_ids)

# Ensure parent chunk is sufficiently large
for tier2_chunk in tier2_chunks:
    parent_id = tier2_chunk["metadata"]["parent_chunk_id"]
    parent_chunk = next(c for c in tier1_chunks if c["chunk_id"] == parent_id)
    assert parent_chunk["metadata"]["token_count"] >= 250
```

**Detector 5.4.3: Consecutive Tier-2 Prevention**
```python
# Check that no two Tier-2 chunks share same parent AND are consecutive
tier2_by_parent = {}
for chunk in tier2_chunks:
    parent_id = chunk["metadata"]["parent_chunk_id"]
    tier2_by_parent.setdefault(parent_id, []).append(chunk)

for parent_id, children in tier2_by_parent.items():
    assert len(children) <= 1  # Max one Tier-2 per Tier-1 parent
```

### 5.5 Table Context Validation (ENFORCE COMPLETENESS)

**Check 5.5.1: All Tables Have Context**
```python
tier3_chunks = [c for c in chunks if c["metadata"]["chunk_tier"] == 3]
for chunk in tier3_chunks:
    content = chunk["content"]
    table_text = chunk["metadata"]["table_data"][0]  # At least one table exists
    
    # Check for lead-in (text before table)
    has_leadin = len(content.split(str(table_text))[0].strip()) > 50  # At least 50 chars
    
    # Check for interpretation (text after table)
    has_interpretation = len(content.split(str(table_text))[-1].strip()) > 50
    
    # At least one must be present
    assert has_leadin or has_interpretation, f"Table chunk {chunk['chunk_id']} lacks context"
```

---

## 6. Final Quality Assurance Checklist (EXECUTE BEFORE OUTPUT)

### 6.1 Boundary & Coherence Checklist

- [ ] No chunk ends mid-sentence (verified by §5.2.1)
- [ ] No teaching story is split across chunks (narratives preserved per §4.1.2)
- [ ] All numbered lists stay together (verified by §5.2.2)
- [ ] Quotes and attributions are not separated (verified by §5.2.3)
- [ ] Page breaks do not create orphan fragments (merged per §2.4.3)

### 6.2 Table Handling Checklist

- [ ] Every table has lead-in OR interpretation (verified by §5.5.1)
- [ ] `table_data` is always a list (verified by §5.1.3 Invariant 3)
- [ ] `has_table == (len(table_data) > 0)` (verified by §5.1.3 Invariant 4)
- [ ] All Tier-3 chunks use `content_type="financial_table"` (verified by §5.1.3 Invariant 2)
- [ ] All Tier-3 chunks have `parent_chunk_id` set (verified by §5.1.3 Invariant 5)

### 6.3 Anti-Orphan Checklist

- [ ] No Tier-1 chunk < 250 tokens (except standalone exceptions, verified by §5.3.1)
- [ ] No Tier-2 chunk < 120 tokens (verified by §5.3.2)
- [ ] No header-only chunks exist (enforced by §2.4.2)
- [ ] No tail-fragment chunks exist (enforced by §2.4.3)
- [ ] Standalone exception rate ≤ 5% (verified by §5.3.3)

### 6.4 Tier-2 Leakage Checklist

- [ ] All Tier-2 chunks pass self-containment test (verified by §5.4.1)
- [ ] No Tier-2 chunk starts with unresolved pronoun (verified by §5.4.1)
- [ ] All Tier-2 chunks have valid parent (verified by §5.4.2)
- [ ] Parent chunks remain ≥ 250 tokens after Tier-2 extraction (verified by §5.4.2)
- [ ] No consecutive Tier-2 chunks from same parent (verified by §5.4.3)

### 6.5 Schema Conformance Checklist

- [ ] Every chunk has exactly 3 top-level keys: `chunk_id`, `content`, `metadata` (verified by §5.1.1)
- [ ] `metadata` contains all 26 required keys (verified by §5.1.2)
- [ ] All types match §0.3 exactly (verified by §5.1.1)
- [ ] `chunk_id` matches regex and `chunk_tier` (verified by §5.1.3 Invariant 1)
- [ ] `source_span.start_char < source_span.end_char_exclusive` (verified by §5.1.3 Invariant constraint)

---

## 7. Error Handling & Remediation (AUTOMATED FIXES)

### 7.1 Validation Failure Response

**IF Schema Validation Fails (§5.1):**
```
1. Log error with chunk_id and specific violation
2. Attempt automatic repair:
   - Missing field → Add with default value + set boundary_note
   - Wrong type → Coerce to correct type if possible
   - Invalid value → Replace with nearest valid value
3. IF repair succeeds → Continue with warning
4. IF repair fails → HALT processing, output error report
```

**IF Boundary Check Fails (§5.2):**
```
1. Log warning (non-blocking)
2. Attempt to extend chunk boundary to valid endpoint
3. Revalidate after boundary adjustment
4. Set boundary_note with adjustment reason
```

**IF Tier-2 Leakage Detected (§5.4):**
```
1. Log leakage instance
2. Automatically ABSORB Tier-2 chunk back into Tier-1 parent
3. Remove Tier-2 chunk_id from parent's child_chunk_ids
4. Set boundary_note = "Tier-2 absorbed: failed self-containment"
5. Revalidate parent chunk size (must remain ≥ 250 tokens)
```

### 7.2 Size Violation Remediation

**IF Tier-1 Chunk < 250 Tokens (Non-Exception):**
```
1. Identify merge direction using §2.4.1 decision matrix
2. Execute merge with adjacent chunk
3. Regenerate chunk_id with updated sequence number
4. Set merged_from field with absorbed chunk descriptor
5. Revalidate merged chunk
```

**IF Tier-2 Chunk < 120 Tokens:**
```
1. Absorb into Tier-1 parent immediately
2. Remove Tier-2 chunk from output
3. Update parent's content and metadata
4. Set boundary_note on parent
```

**IF Tier-3 Chunk Lacks Context:**
```
1. Search parent Tier-1 section for related paragraphs
2. IF found → Include in Tier-3 chunk
3. IF not found → Set boundary_note + reduce retrieval_priority to "medium"
4. Add cross_reference to parent section
```

---

## 8. Execution Summary & Output Format

### 8.1 Processing Output Structure

```json
{
  "document_metadata": {
    "letter_year": 2009,
    "source_file": "2009_cleaned.txt",
    "processing_timestamp": "2024-01-15T10:30:00Z",
    "chunker_version": "2.0.0"
  },
  "statistics": {
    "total_chunks": 150,
    "tier1_count": 120,
    "tier2_count": 10,
    "tier3_count": 20,
    "exception_count": 5,
    "exception_rate": 0.033,
    "avg_tier1_tokens": 450,
    "avg_tier2_tokens": 200,
    "avg_tier3_tokens": 300,
    "min_tier1_tokens": 250,
    "max_tier1_tokens": 1200,
    "validation_passed": true
  },
  "chunks": [
    { "chunk_id": "...", "content": "...", "metadata": {...} },
    ...
  ],
  "warnings": [
    "Chunk 2009-S5-T1-003: Exceeds 800 token target (narrative preserved)",
    ...
  ],
  "errors": []
}
```

### 8.2 Logging & Auditability

**Required Log Entries:**
- Each merge decision with reason
- Each Tier-2 evaluation with pass/fail outcome
- Each standalone exception with justification
- Each validation failure with remediation action
- Chunk statistics summary

**Log Format:**
```
[TIMESTAMP] [LEVEL] [CHUNK_ID] [ACTION] [REASON]
2024-01-15 10:30:15 INFO 2009-S3-T1-012 MERGE_FORWARD Short section (180 tokens), lead-in detected
2024-01-15 10:30:16 WARN 2009-S5-T1-018 TIER2_REJECTED Failed self-containment test (pronoun start)
```

---

## 9. Implementation Validation Tests (UNIT TEST SUITE)

### 9.1 Test Cases for Boundary Detection

**Test 9.1.1: Orphan Prevention**
```
INPUT: Section with 180-token content followed by new section
EXPECTED: Merge forward into next section
VALIDATION: No chunk < 250 tokens emitted
```

**Test 9.1.2: Header-Only Chunk Prevention**
```
INPUT: Section header + 50-token paragraph + new section
EXPECTED: Merge with next section
VALIDATION: No header-only chunk exists
```

**Test 9.1.3: Tail Fragment Prevention**
```
INPUT: 900-token section with 100-token final paragraph
EXPECTED: Final paragraph absorbed into main chunk
VALIDATION: No tail fragment chunk < 250 tokens
```

### 9.2 Test Cases for Tier-2 Prevention

**Test 9.2.1: Pronoun Start Rejection**
```
INPUT: Paragraph starting with "This approach..."
EXPECTED: Tier-2 creation rejected, absorbed into Tier-1
VALIDATION: No Tier-2 chunk with pronoun start exists
```

**Test 9.2.2: Size Below Minimum**
```
INPUT: 100-token self-contained paragraph
EXPECTED: Tier-2 creation rejected (< 120 tokens)
VALIDATION: No Tier-2 chunk < 120 tokens exists
```

**Test 9.2.3: Consecutive Tier-2 Prevention**
```
INPUT: Three consecutive paragraphs, each 150 tokens, each self-contained
EXPECTED: Either merge into single Tier-2 OR keep in Tier-1
VALIDATION: No two Tier-2 chunks from same parent
```

### 9.3 Test Cases for Table Context

**Test 9.3.1: Table with Lead-in**
```
INPUT: 1 paragraph + table + 0 paragraphs after
EXPECTED: Tier-3 chunk containing paragraph + table
VALIDATION: Tier-3 has lead-in content before table
```

**Test 9.3.2: Table with Interpretation**
```
INPUT: 0 paragraphs + table + 1 paragraph after
EXPECTED: Tier-3 chunk containing table + paragraph
VALIDATION: Tier-3 has interpretation content after table
```

**Test 9.3.3: Table without Context**
```
INPUT: Isolated table with no adjacent paragraphs
EXPECTED: Tier-3 chunk with boundary_note, reduced priority
VALIDATION: boundary_note set, retrieval_priority = "medium"
```

### 9.4 Test Cases for Schema Validation

**Test 9.4.1: Invariant Violation Detection**
```
INPUT: Chunk with content_type="financial_table" but has_table=false
EXPECTED: Validation failure, automatic repair attempt
VALIDATION: has_table corrected to true
```

**Test 9.4.2: Missing Required Field**
```
INPUT: Chunk metadata missing "themes" key
EXPECTED: Validation failure, field added with empty list
VALIDATION: All required keys present after repair
```

**Test 9.4.3: Type Mismatch**
```
INPUT: token_count as string "450" instead of int
EXPECTED: Validation failure, coercion to int
VALIDATION: token_count type is int
```

---

## 10. Version Control & Update Protocol

### 10.1 Schema Versioning

**Current Version: 2.0.0**

IF schema changes are needed:
1. Increment version number
2. Document changes in CHANGELOG
3. Provide migration script for existing chunks
4. Update all validators and test cases
5. Reprocess corpus with new schema

### 10.2 Rule Update Protocol

To update chunking rules:
1. Document proposed change with rationale
2. Run test suite against representative sample
3. Measure impact on chunk distribution and quality metrics
4. IF improvement validated → Update rule with version note
5. ELSE → Reject change, document reason

---

## APPENDIX A: Quick Reference Decision Trees

### A.1 Merge Direction Decision
```
Is content < 250 tokens?
├─ YES
│  ├─ Contains forward marker (Now, Next, Turning to)?
│  │  └─ MERGE FORWARD
│  ├─ Contains backward marker (In summary, As mentioned)?
│  │  └─ MERGE BACKWARD
│  ├─ Last sentence is question?
│  │  └─ MERGE FORWARD
│  ├─ Contains "following", "below"?
│  │  └─ MERGE FORWARD
│  ├─ Contains "above", "previously"?
│  │  └─ MERGE BACKWARD
│  └─ DEFAULT: MERGE FORWARD
└─ NO: Continue processing
```

### A.2 Tier Assignment Decision
```
Content analysis:
├─ Contains table structure?
│  └─ Assign TIER 3 → Apply §2.3 rules
├─ Is complete section?
│  └─ Assign TIER 1 → Apply §2.2 rules
├─ Is paragraph within Tier-1?
│  ├─ Passes ALL Tier-2 criteria (§2.5.1)?
│  │  └─ Create TIER 2 chunk
│  └─ Fails any criterion?
│     └─ ABSORB into Tier-1 parent
└─ DEFAULT: TIER 1
```

### A.3 Content Type Assignment
```
Check in order:
1. Has table_data with len ≥ 1?
   └─ content_type = "financial_table"
2. Contains mistake + lesson?
   └─ content_type = "mistake_confession"
3. Is standalone principle (§2.6.1)?
   └─ content_type = "principle_statement"
4. DEFAULT:
   └─ content_type = "narrative"
```

---

## APPENDIX B: Validation Error Codes

| Code | Description | Severity | Auto-Fix |
|------|-------------|----------|----------|
| E001 | Missing required metadata field | BLOCKING | Add default value |
| E002 | chunk_id regex mismatch | BLOCKING | Regenerate ID |
| E003 | chunk_tier inconsistency | BLOCKING | Correct tier value |
| E004 | table_data type violation | BLOCKING | Convert to list |
| E005 | Invariant violation (§0.3) | BLOCKING | Apply constraint fix |
| W001 | Chunk < 250 tokens (non-exception) | WARNING | Trigger merge |
| W002 | Tier-2 self-containment failure | WARNING | Absorb to Tier-1 |
| W003 | Table lacks context | WARNING | Add cross-reference |
| W004 | Mid-sentence boundary | WARNING | Extend boundary |
| W005 | Split numbered list | WARNING | Merge list items |

---

## APPENDIX C: Glossary of Terms

**Chunk Tier:**
- Tier 1: Section-level narrative chunks (250-1200 tokens)
- Tier 2: Optional paragraph sub-chunks (120-400 tokens)
- Tier 3: Table/data chunks with context (no minimum)

**Orphan Chunk:** Any chunk below minimum token threshold for its tier, unless granted standalone exception

**Self-Containment:** Property of text where first sentence has explicit subject and no unresolved references to external context

**Lead-in:** Explanatory text immediately preceding a table (within 2 line breaks)

**Interpretation:** Explanatory text immediately following a table (within 1 line break)

**Merge Direction:**
- Forward: Absorb content into the next chunk
- Backward: Absorb content into the previous chunk

**Standalone Exception:** Special case allowing chunk below minimum token threshold when high retrieval value and self-contained

**Boundary Note:** Metadata field documenting merge decisions or chunking tradeoffs

---

**END OF STRATEGY DOCUMENT**
