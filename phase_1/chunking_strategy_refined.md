# Chunking Strategy for Warren Buffett Shareholder Letters
## Agentic RAG System Design

---

## 0. Canonical Chunk Schema (MUST MATCH EXACTLY)

This project’s downstream embedding + retrieval pipeline assumes **one stable, canonical chunk object shape** across *all* years. Every chunk emitted by any chunker (LLM or rules-based) MUST follow this schema exactly (same keys, same nesting, same types). If a field is not applicable, use the specified default.

### 0.1 Canonical Chunk Object (top-level)

```json
{
  "chunk_id": "2016-S3-T1-002",
  "content": "…full chunk text…",
  "metadata": { "…see §0.2…" }
}
```

#### `chunk_id` format (required)
- Regex: `^\d{4}-S\d+-T[123]-\d{3}$`
- Meaning: `YYYY-S{section_index}-T{tier}-{sequence_in_section}`
  - `tier`: `1` (section/narrative), `2` (optional paragraph sub-chunk), `3` (table/data)
  - `sequence_in_section`: zero-padded 3 digits (`001`, `002`, …)

### 0.2 Canonical `metadata` object (required keys, required types)

> **Rule:** These keys MUST always be present. Use defaults if unknown / not applicable.

```json
{
  "letter_year": 2016,
  "letter_date": "2017-02-25",
  "source_file": "2016_cleaned.txt",

  "section_title": "…",
  "section_hierarchy": ["Letter Body", "Business Segments", "Insurance"],

  "chunk_tier": 1,
  "parent_chunk_id": null,
  "child_chunk_ids": [],

  "content_type": "narrative",
  "contextual_summary": "1–3 sentences: what this chunk teaches / contains, with enough context to stand alone.",

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

### 0.3 Field definitions & allowed values

#### Required scalars
- `letter_year` (int)
- `letter_date` (string, `YYYY-MM-DD`)  
  - If unknown, use `null` **only if your storage layer permits nulls**; otherwise use `"0000-00-00"` and add `boundary_note`.
- `source_file` (string): filename of the cleaned/plain-text source used for `source_span`.

#### Section identity
- `section_title` (string): human-facing section title (stable across chunks in the same section).
- `section_hierarchy` (list[string]): hierarchical path; keep consistent categories such as:
  - `["Tables", "Performance"]`
  - `["Letter Body", "Business Segments", "Insurance"]`
  - `["Appendix", "Annual Meeting"]`

#### Tiering & linkage
- `chunk_tier` (int): must equal the tier encoded in `chunk_id`.
- `parent_chunk_id` (string|null): for Tier-2 or Tier-3 chunks that belong to a Tier-1 chunk. Use `null` if not used.
- `child_chunk_ids` (list[string]): empty list if not used.

#### Content type (enum)
- `content_type` (string) ∈:
  - `"narrative"` (default for Tier-1)
  - `"financial_table"` (default for Tier-3)
  - `"mistake_confession"`
  - `"principle_statement"`

**Consistency constraints**
- If `content_type == "financial_table"` → `has_table = true` AND `len(table_data) >= 1`.
- If `len(table_data) >= 1` → `has_table = true`.
- If `has_table = false` → `table_data = []` (NOT null, NOT dict).

#### Table data (always a list)
`table_data` MUST always be a list of `TableData` objects (possibly empty):

```json
{
  "table_name": "Insurance Operations Results",
  "summary": "1–2 sentences: what the table shows and why it matters.",
  "columns": ["Operation", "Underwriting Profit 2016", "Underwriting Profit 2015"],
  "rows": [["GEICO", "$…", "$…"], ["General Re", "$…", "$…"]],
  "row_count": 2
}
```

#### Entity extraction
- `entities.companies` (list[string])
- `entities.people` (list[string])
- `entities.metrics` (list[string]) – include metric names like `"float"`, `"underwriting profit"`, `"operating earnings"`, `"book value"`, etc.

Always emit all three keys with list values (empty list allowed).

#### Concepts & principles
- `themes` (list[string]): topic tags (broad).
- `buffett_concepts` (list[string]): Buffett-specific mental models (more specific than `themes`).
- `principles` (list[object]): extracted actionable principles. Each item:
  - `statement` (string): concise, self-contained.
  - `category` (string): e.g. `capital_allocation`, `valuation`, `risk`, `management`, `insurance`, `accounting`, `behavior`, `governance`, `general`.

Example:
```json
{ "statement": "We repurchase shares only when they are meaningfully below intrinsic value.", "category": "capital_allocation" }
```

#### Time & cross-links
- `temporal_references`:
  - `primary_year` (int): usually `letter_year`
  - `comparison_years` (list[int])
  - `future_outlook` (bool)
- `cross_references`:
  - `related_sections_same_letter` (list[string])
  - `related_years` (list[int])

#### Ranking & abstraction (enums)
- `retrieval_priority` ∈ `"low" | "medium" | "high"`
- `abstraction_level` ∈ `"low" | "medium" | "high"`

Guidelines:
- Tables, definitions, crisp principles → `retrieval_priority = high`
- Small housekeeping / meeting logistics → `medium` or `low`
- Abstract reflections (no facts) → `abstraction_level = high`
- Concrete facts (numbers, events) → `abstraction_level = low`

#### Counts & provenance
- `token_count` (int): token count of `content` (pick one tokenizer and stick to it).
- `char_count` (int): character count of `content`.
- `source_span` (object):
  - `start_char` (int)
  - `end_char_exclusive` (int)
  - Offsets are in `source_file` content *after* cleaning and normalization.

#### Merge/exception flags (always present)
- `standalone_exception` (bool, default `false`)
- `exception_reason` (string|null)
- `boundary_note` (string|null)
- `merged_from` (list[string], default `[]`)

---

## 1. Document Structure Analysis

### 1.1 Observed Patterns Across Letters

Based on analysis of the 1977 (early/short) and 2009 (mature/long) letters, Buffett's letters exhibit consistent structural patterns:

**Recurring Sections:**
- Opening performance summary (net worth/book value gains)
- Business segment reports (Insurance, Utilities, Manufacturing, etc.)
- Investment philosophy discussions
- Specific company/acquisition narratives
- Management commentary and confessions
- Annual meeting information
- Closing remarks

**Structural Markers:**
- Section headers (e.g., "Textile Operations", "Insurance Underwriting", "Regulated Utility Business")
- Asterisk separators (`* * * * * * * * * * * *`) marking topic transitions
- Page breaks indicated by `[[PAGE_BREAK]]`
- Tables with financial data
- Signature block with date

### 1.2 Content Characteristics

| Characteristic | 1977 Letter | 2009 Letter |
|---------------|-------------|-------------|
| Length | ~20KB / 379 lines | ~77KB / 876 lines |
| Sections | 6-7 distinct | 15+ distinct |
| Tables | 1 (holdings) | 5+ (performance, earnings, holdings) |
| Named entities | ~30 | ~100+ |
| Time references | Single year focus | 45-year retrospective |

---

## 2. Recommended Chunking Strategy: Hybrid Semantic-Structural

For agentic RAG with Warren Buffett letters, use a **three-tier chunking approach** that balances retrieval precision with contextual completeness **while preserving the canonical schema in §0**.

### 2.1 Tier 1: Section-Level Chunks (Primary Retrieval Units)

**Definition:** Each distinct thematic section becomes a chunk, bounded by:
- Explicit headers (e.g., "Insurance Underwriting")
- Asterisk separators
- Clear topic transitions

**Chunk Size Target:** 500–2000 tokens  
**Hard Minimum (Anti-Orphan Gate):** do not emit a Tier-1 chunk under **250 tokens** (~180–220 words) unless it meets an explicit exception (see §2.4).

**Chunk ID rules (Tier 1):**
- Use `T1` in `chunk_id`.
- The section index (`S{n}`) increments by **semantic sections** in reading order.
- Sequence (`-001`, `-002`, …) increments only if you must split a very long section into multiple Tier-1 chunks.

**Rationale:** Buffett's writing style creates self-contained "mental models" that are best retrieved as whole units. Over-splitting creates fragments that lose causal context and reduce agent reliability.

### 2.2 Tier 2: Paragraph-Level Sub-Chunks (Precision Retrieval) — OPTIONAL

Tier-2 is optional. Only create Tier-2 if you have a clear retrieval need and you can maintain clean boundaries.

**Definition:** Individual paragraphs or tightly-coupled paragraph pairs within a Tier-1 section.

**Chunk Size Target:** 150–500 tokens  
**Hard Minimum (Anti-Orphan Gate):** do not emit a Tier-2 chunk under **120 tokens** (~80–110 words) unless it meets an explicit exception (see §2.4).

**Linking Strategy:**
- Tier-2 chunks MUST set `parent_chunk_id` to their Tier-1 parent, and the Tier-1 chunk MUST include their IDs in `child_chunk_ids`.

**Pairing Rule (prevents tiny paragraph-chunks):**
- If a candidate paragraph < 120 tokens, *pair it* with its most semantically adjacent paragraph:
  - Transition/setup paragraph → pair forward
  - Concluding/afterthought paragraph → pair backward

### 2.3 Tier 3: Table/Data Chunks (Structured Retrieval)

**Definition:** Financial tables, holdings lists, and performance data extracted as structured chunks.

**Critical:** Tier-3 chunks must still be valid canonical chunks (§0). Do **not** emit “bare table JSON”.

**Chunk Size Target:** table-dependent, but must include:
- narrative setup (lead-in) AND
- the table itself AND
- at least a minimal interpretation if present nearby

**Canonical example (Tier 3):**
```json
{
  "chunk_id": "2009-S7-T3-001",
  "content": "Insurance Operations Results\n\n[Lead-in narrative…]\n\n[TABLE RENDERED AS TEXT FOR READABILITY]\n\n[Brief interpretation…]",
  "metadata": {
    "letter_year": 2009,
    "letter_date": "2010-02-26",
    "source_file": "2009_cleaned.txt",
    "section_title": "Insurance Operations Results",
    "section_hierarchy": ["Tables", "Insurance"],
    "chunk_tier": 3,
    "parent_chunk_id": "2009-S7-T1-001",
    "child_chunk_ids": [],
    "content_type": "financial_table",
    "contextual_summary": "Summarizes underwriting profit and float by insurance operation; used to compare performance across years.",
    "has_table": true,
    "table_data": [
      {
        "table_name": "Insurance Operations Results",
        "summary": "Underwriting profit and float by insurance unit across two years.",
        "columns": ["Operation", "Underwriting Profit 2009", "Underwriting Profit 2008", "Float 2009", "Float 2008"],
        "rows": [["General Re", "$477M", "$342M", "$21,014M", "$21,074M"], ["GEICO", "$649M", "$916M", "$9,613M", "$8,454M"]],
        "row_count": 2
      }
    ],
    "has_financial_data": true,
    "entities": { "companies": ["General Re", "GEICO"], "people": [], "metrics": ["underwriting profit", "float"] },
    "themes": ["insurance economics"],
    "buffett_concepts": ["float economics"],
    "principles": [],
    "temporal_references": { "primary_year": 2009, "comparison_years": [2008], "future_outlook": false },
    "cross_references": { "related_sections_same_letter": ["Insurance Underwriting"], "related_years": [2008, 2010] },
    "retrieval_priority": "high",
    "abstraction_level": "low",
    "token_count": 0,
    "char_count": 0,
    "source_span": { "start_char": 0, "end_char_exclusive": 0 },
    "standalone_exception": false,
    "exception_reason": null,
    "boundary_note": null,
    "merged_from": []
  }
}
```

### 2.4 Anti-Orphan & Minimum-Size Merge Rules (Critical)

Buffett letters often contain micro-content (1–2 sentences, transitional paragraphs, “header + one line”, short pre-table lead-ins). These MUST NOT become standalone chunks.

#### Definitions
- **Tier-1 minimum:** 250 tokens (~180–220 words)
- **Tier-2 minimum:** 120 tokens (~80–110 words)
- **Micro-content:** any chunk candidate below its tier minimum.

#### Rule A — No Header-Only Chunks
A section header can never be emitted as its own chunk. If a chunk begins with a header but remains under the minimum, keep absorbing content until:
- it reaches the tier minimum, OR
- you hit the next explicit header, then apply Rule B.

#### Rule B — Directional Merge for Short Section Candidates
If a Tier-1 section candidate ends up below 250 tokens, merge it instead of emitting a short chunk.

Merge preference:
1) **Merge forward** if the short section is a lead-in/setup (common before tables, segment summaries, or new themes).  
2) Otherwise **merge backward** if it reads like a concluding remark or afterthought.  
3) If both neighbors exist, choose the merge that yields stronger coherence:
   - setup → forward
   - conclusion/afterthought → backward

#### Rule C — No Tail-Fragments
If the remainder at the end of a section would form a new chunk below minimum, do NOT emit it:
- Append it to the current chunk, unless it clearly starts a new topic.
- If it clearly starts a new topic but is short, attach it forward to the next chunk.

#### Rule D — Tier-2 Micro-Chunk Prevention
Tier-2 is optional. Do not create Tier-2 sub-chunks that are under 120 tokens.
- If a paragraph is short, merge it with its adjacent paragraph using the Pairing Rule in §2.2.

#### Allowed Exceptions (rare)
Short standalone chunks are allowed ONLY if they are high-value retrieval atoms AND fully self-contained:
- A compact, complete principle statement where context is already included
- A concise “mistake/confession” unit with its resolution included
- A table chunk ONLY if narrative setup + interpretation context are included in the SAME chunk boundary

When using an exception:
- set `standalone_exception: true`
- set `exception_reason` (1 short sentence)
- set `retrieval_priority` to `high`

---

## 3. Implementation Details

### 3.1 Special Handling Rules

**Rule 1: Preserve Buffett's Teaching Moments**  
Buffett often uses analogies and stories to illustrate principles. Never break these narratives.

**Rule 2: Keep Investment Philosophy Intact**  
Sections explaining "how we think about X" should remain whole.

**Rule 3: Table Context Preservation**  
Tables must include surrounding explanatory text **in the same chunk** (Tier-3 if possible).

**Rule 4: Handle Confessions/Mistakes Specially**  
Buffett's admissions of error are high-value content for learning:
- set `content_type: "mistake_confession"`
- ensure the confession and the lesson/resolution are together

**Rule 5: Short Lead-ins Before Tables Must Stay With the Table**  
If Buffett introduces a table with a short lead-in (often < 2 paragraphs), keep the lead-in in the SAME chunk as the table and at least one paragraph of interpretation if present.

**Rule 6: One-Sentence Transitions Are Not Chunks**  
Single-sentence bridges (“Now, turning to…”, “Before discussing X…”) must be merged with the following paragraph (forward-merge) unless they clearly conclude a prior argument (then backward-merge).

**Rule 7: Section Stubs Must Merge**  
If a section appears as a stub (header + very short body), merge it using §2.4 Rule B.

**Rule 8: Record Merge Decisions (schema-consistent)**  
When you merge micro-content or make a boundary tradeoff:
- set `boundary_note` (1 short sentence)
- populate `merged_from` with any prior IDs or descriptors you have available

### 3.2 Schema Conformance Gate (required)

After chunking, run a schema validator that enforces:
- exact keys and types from §0
- `chunk_id` regex and `metadata.chunk_tier` consistency
- `table_data` is ALWAYS a list
- if `has_table=false` then `table_data=[]`
- if `content_type="financial_table"` then `has_table=true` and `len(table_data)>0`

If validation fails:
- **repair** the chunk output (do not silently drop fields), or
- **halt** with a clear error report.

---

## 4. Agent-Specific Considerations

### 4.1 Query Type to Chunk Tier Mapping

| Query Type | Primary Tier | Expansion Strategy |
|------------|--------------|-------------------|
| "What did Buffett say about X in year Y?" | Tier 1 | None needed |
| "What was GEICO's float in 2009?" | Tier 3 | Expand to parent for context |
| "How did Buffett's view on insurance evolve?" | Tier 1 | Cross-year retrieval |
| "Quote Buffett on value investing" | Tier 2 | Expand to parent if needed |
| "Compare textile vs insurance performance" | Tier 1 | Multi-section retrieval |

### 4.2 Embedding Considerations

**Recommended Approach:** Hybrid embeddings
- Dense embeddings for semantic similarity
- Sparse retrieval (BM25/SPLADE) for exact names/years/metrics

**Metadata Filtering Priority:**
1. `letter_year`
2. `section_title`
3. `entities.companies`
4. `buffett_concepts`

---

## 5. Overlap and Deduplication Strategy

### 5.1 Chunk Overlap (optional)

If you add overlap, keep schema stable. Do NOT add new keys unless you version the schema.
Preferred approach:
- Keep `content` overlap-free
- Store overlap decisions in `boundary_note` instead

### 5.2 Cross-Year Deduplication

Buffett repeats core principles across years. Handle via:
- consistent `buffett_concepts`
- `cross_references.related_years` pointing to best related explanations

---

## 6. Quality Assurance Checklist (schema + boundaries)

Before finalizing chunks, verify:

### Boundary & coherence
- [ ] No chunk ends mid-sentence
- [ ] No teaching story is split across chunks
- [ ] All numbered lists stay together (e.g., "(1)…(2)…(3)…")
- [ ] Quotes and attributions are not separated
- [ ] Page breaks do not create orphan fragments (merge across `[[PAGE_BREAK]]`)

### Table handling
- [ ] Any table has its lead-in + table + minimal interpretation together
- [ ] `table_data` is a list (empty or `>=1`)
- [ ] `has_table == (len(table_data) > 0)`
- [ ] Tier-3 chunks with tables use `content_type="financial_table"`

### Anti-orphan rules
- [ ] No Tier-1 chunk < 250 tokens unless `standalone_exception=true`
- [ ] No Tier-2 chunk < 120 tokens unless `standalone_exception=true`
- [ ] No header-only or “header + one line” chunks
- [ ] No tail-fragment chunk is emitted below minimum (must merge)

### Canonical schema conformance
- [ ] Every chunk has only the top-level keys: `chunk_id`, `content`, `metadata`
- [ ] `metadata` contains **all required keys** in §0.2 (use defaults when N/A)
- [ ] Types match §0.2 exactly (no dict-vs-list drift)
- [ ] `chunk_id` regex passes and matches `metadata.chunk_tier`
- [ ] `source_span.start_char < source_span.end_char_exclusive` (or both `0` only if unavailable and noted)

---
