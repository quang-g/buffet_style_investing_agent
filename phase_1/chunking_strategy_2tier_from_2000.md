# Chunking Strategy for Buffett Letters — Agentic RAG

## 1. Canonical Schema (IMMUTABLE)

Every chunk MUST match this exact structure. No optional keys. No type variations.

### 1.1 Chunk Object
```json
{
  "chunk_id": "2016-S03-T1-002",
  "content": "…full chunk text…",
  "metadata": { /* §1.2 */ }
}
```
**`chunk_id` regex:** `^\d{4}-S\d{2}-T[12]-\d{3}$` → `YYYY-S##-T{1|2}-###`

### 1.2 Metadata Object (ALL 27 keys required)
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
  "contextual_summary": "a 1 to 3 sentences short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk",
  "has_table": false,
  "table_data": [],
  "has_financial_data": false,
  "entities": { "companies": [], "people": [], "metrics": [] },
  "themes": [],
  "buffett_concepts": [],
  "principles": [],
  "temporal_references": { "primary_year": 2016, "comparison_years": [], "future_outlook": false },
  "cross_references": { "related_sections_same_letter": [], "related_years": [] },
  "retrieval_priority": "medium",
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

#### metadata.contextual_summary
contextual_summary (MANDATORY, STRICT)

**Purpose**
Provide a 1 to 3 sentences short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.

**Format Rules**

Length: 1–3 sentences, maximum 45 words
Style: Natural language, complete sentences
Must be standalone (readable without surrounding context)

**Prohibitions (STRICT)**
The summary MUST NOT:
- Mention “section”, “part X of Y”, or document structure
- Use templates like:
  “Buffett discusses …”
  “This section talks about …”
- Be a list of keywords or comma-separated words
- Contain ellipses (...) or truncated words
- Copy sentences verbatim from the source text

**Quality Standard**
Prefer specific meaning over general themes.

✅ Good Examples

“Buffett explains why Berkshire’s insurance operations produced record underwriting profits in 1995, emphasizing disciplined pricing and the absence of catastrophe losses.”

“The letter warns that accounting earnings can misrepresent true economic performance, using examples from capital-intensive businesses.”

❌ Bad Examples

“In the Introduction section (part 1 of 1), Buffett discusses have, over, share.”

“Buffett discusses insurance, earnings, and business.”

“This section talks about Berkshire’s performance.”

Validation Rule (for execution)

If a generated contextual_summary violates any rule above, it must be rewritten before final output.

### 1.3 Field Constraints

| Field | Type | Constraint |
|-------|------|------------|
| `chunk_tier` | int | `1` or `2` ONLY |
| `content_type` | enum | `narrative` \| `financial_table` \| `mistake_confession` \| `principle_statement` |
| `retrieval_priority` | enum | `low` \| `medium` \| `high` |
| `abstraction_level` | enum | `low` \| `medium` \| `high` |
| `table_data` | list | ALWAYS list—empty `[]` if no table |
| `entities` | object | Keys: `companies`, `people`, `metrics` (each list) |
| `principles` | list | Items: `{ "statement": str, "category": str }` |
| `letter_date` | string | `YYYY-MM-DD` or `"0000-00-00"` |

**Invariants (validation enforced):**
1. `has_table == true` ⟺ `len(table_data) >= 1`
2. `content_type == "financial_table"` → `has_table == true`
3. `chunk_tier` matches T-digit in `chunk_id`
4. `source_span.start_char < source_span.end_char_exclusive` (or both 0)

### 1.4 TableData Object
```json
{ "table_name": "str", "summary": "str", "columns": [], "rows": [[]], "row_count": int }
```

---

## 2. Chunking Rules (DETERMINISTIC)
Skip the first table "Berkshire's Corporate Performance vs. the S&P 500" in the letter. Only start chunking after the first table.

### 2.1 Tier Definitions
| Tier | Purpose | Tokens | Creation |
|------|---------|--------|----------|
| T1 | Section narrative | 250–800 | ALWAYS |
| T2 | Table + context | 150–600 | When table detected |

### 2.2 Boundary Detection (Priority Order)
1. **Headers:** `^[A-Z][A-Za-z\s]+$` + blank line
2. **Separators:** `* * *` (3+ asterisks)
3. **Page breaks:** `[[PAGE_BREAK]]` only if topic changes
4. **Signature:** "Warren E. Buffett" / "Chairman" = END

**Never split on:** mid-paragraph breaks, single transitions, list items within discussion.

### 2.3 Token Enforcement
| Condition | Action |
|-----------|--------|
| Section < 250 tokens | Merge per §2.4 |
| Section > 800 tokens | Split at paragraph nearest 500 |
| Table chunk < 150 tokens | Absorb context until ≥ 150 |

### 2.4 Merge Decision Tree (NO DISCRETION)
```
IF chunk < minimum:
  IF section-final → MERGE BACKWARD
  ELIF header + < 2 paragraphs → MERGE FORWARD
  ELIF follows table → MERGE BACKWARD
  ELSE → MERGE FORWARD
```
After merge: populate `merged_from`, `boundary_note`.

### 2.5 Table Handling
#### 2.5.1 Skip ONLY the first S&P performance table (hard rule)

Skip content **only at the very start of the file** if the first non-empty block contains either title:

- "Berkshire's Corporate Performance vs. the S&P 500"

- "Berkshire's Performance vs. the S&P 500"

**Skip span:** from file start through the first blank line after the table ends (or through the first [[PAGE_BREAK]] after the table, whichever comes first).
After this one skip, **never skip any other table.**

#### 2.5.2 Deterministic table-block detection (after the skip)

A **TABLE_BLOCK** is any block of **≥ 3 consecutive non-empty lines** satisfying **either:**

**A) Dot-leader pattern**

- At least 1 line contains ..... (3+ dots) and

- At least 2 lines in the block contain **≥ 2 numeric tokens** ($, digits, commas, parentheses, %)

**B) Column-alignment numeric pattern**

At least 3 lines contain **≥ 2 numeric tokens**, and

At least 2 lines in the block contain **2+ spaces** between fields (aligned columns)

**End of table:** first blank line followed by a “normal paragraph” line (≥ 8 alphabetic words OR contains sentence punctuation like ./?/!).

#### 2.5.3 Emit Tier-2 for every TABLE_BLOCK (no exceptions)

For each detected TABLE_BLOCK:

Create **one T2** chunk with:

1. **lead-in paragraph immediately before** the table (if any),

2. the **full TABLE_BLOCK**,

3. the **first interpretation paragraph immediately after** (if any).

Set:

chunk_tier = 2, content_type = "financial_table", has_table = true, has_financial_data = true

retrieval_priority = "high"

parent_chunk_id = <owning T1 chunk_id>; append this T2 id into the T1 child_chunk_ids

In the parent T1 content, replace the raw table text with a single line: [TABLE — see child chunk <T2 chunk_id>]

2.5.4 table_data population (must be non-empty)

table_data MUST contain exactly 1 TableData object:

- table_name: first non-empty line of TABLE_BLOCK (or lead-in’s first clause before : if present)

- summary: 1 sentence describing what the table measures (no templates)

- columns: if a clear header row exists, use it; else ["raw_line"]

- rows: split each data row on 2+ spaces; if splitting fails, store each line as ["<raw_line>"]

- row_count: number of rows



---

## 3. Classification Rules

### 3.1 content_type
| Pattern | Type |
|---------|------|
| Contains table | `financial_table` |
| "mistake/error/wrong" + 1st-person | `mistake_confession` |
| "principle/policy/we believe" + actionable | `principle_statement` |
| Default | `narrative` |

### 3.2 retrieval_priority
| Signal | Priority |
|--------|----------|
| Table / principle / mistake / acquisition | `high` |
| General commentary | `medium` |
| Logistics, boilerplate | `low` |

### 3.3 abstraction_level
| Signal | Level |
|--------|-------|
| Numbers, dates, names | `low` |
| Mixed | `medium` |
| Pure philosophy | `high` |

---

## 4. Standalone Exceptions

Chunk < minimum allowed standalone ONLY IF:
1. `content_type` ∈ `[mistake_confession, principle_statement]`
2. Complete thought (no dangling "this", "the above")
3. Set: `standalone_exception: true`, `exception_reason: "..."`, `retrieval_priority: "high"`

---

## 5. Validation Gate (REQUIRED)

### 5.1 Schema Check
```python
REQUIRED_KEYS = {"chunk_id", "content", "metadata"}
REQUIRED_META = {
    "letter_year", "letter_date", "source_file", "section_title", "section_hierarchy",
    "chunk_tier", "parent_chunk_id", "child_chunk_ids", "content_type", "contextual_summary",
    "has_table", "table_data", "has_financial_data", "entities", "themes", "buffett_concepts",
    "principles", "temporal_references", "cross_references", "retrieval_priority",
    "abstraction_level", "token_count", "char_count", "source_span", "standalone_exception",
    "exception_reason", "boundary_note", "merged_from"
}

def validate(chunk):
    assert set(chunk.keys()) == REQUIRED_KEYS
    assert re.match(r"^\d{4}-S\d{2}-T[12]-\d{3}$", chunk["chunk_id"])
    m = chunk["metadata"]
    assert set(m.keys()) == REQUIRED_META
    assert m["chunk_tier"] in [1, 2]
    assert isinstance(m["table_data"], list)
    assert m["has_table"] == (len(m["table_data"]) > 0)
    if m["content_type"] == "financial_table":
        assert m["has_table"]
```

### 5.2 Coherence Check
- No mid-sentence breaks
- No split numbered lists
- No duplicate Tier-1 `contextual_summary`
- All T2 have narrative context
- No chunk < 150 tokens unless `standalone_exception`

---

## 6. Agent Query Mapping

| Query | Target | Filter |
|-------|--------|--------|
| "What did Buffett say about X in YYYY?" | T1 | `letter_year`, `themes` |
| "[metric] for [company] in YYYY?" | T2 | `entities.companies`, `entities.metrics` |
| "Principles on [topic]" | T1 | `buffett_concepts`, `content_type` |
| "Buffett's mistakes" | T1 | `content_type: mistake_confession` |

---

## 7. Pre-Submission Checklist

- [ ] All chunks pass `validate()`
- [ ] Zero Tier-2 chunks
- [ ] No chunk < 150 tokens without `standalone_exception`
- [ ] All tables wrapped with context
- [ ] `merged_from` populated for merges
- [ ] `contextual_summary` is standalone (no "this section")
- [ ] `chunk_id` sequence contiguous per section