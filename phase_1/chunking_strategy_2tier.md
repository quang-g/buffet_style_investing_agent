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
  "contextual_summary": "1-3 standalone sentences that summarize the context of the current chunk for retrieval.",
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
T2 chunks MUST contain: lead-in paragraph + table + first interpretation paragraph.  
Never emit raw table without narrative wrapper.

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