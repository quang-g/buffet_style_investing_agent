# Chunking Rules for Warren Buffett Letters
## Instructions for LLM-Based Chunking

**Purpose:** This document provides explicit rules for chunking Warren Buffett's annual shareholder letters into semantically meaningful segments for RAG retrieval.

**Target Output:** JSON array of chunk objects with complete metadata.

---

## STEP 1: DOCUMENT ANALYSIS (Do First)

Before chunking, read the entire letter and identify:

1. **Major Section Breaks** - Look for:
   - Centered or bold headers
   - Lines of asterisks (`************`)
   - Clear topic transitions
   - Subheadings within sections

2. **Content Types Present** - Note locations of:
   - Financial tables with numbers
   - Personal anecdotes/stories
   - Abstract principles and philosophy
   - Company-specific analysis
   - Administrative announcements

3. **Key Memorable Content** - Flag:
   - Colorful metaphors or analogies
   - Self-deprecating confessions
   - Quotable one-liners
   - Named stories about people
   - Strong opinion statements

---

## STEP 2: SECTION TYPE CLASSIFICATION

Assign each section to ONE of these categories based on content (not header text):

| Section Type | Content Signals |
|--------------|-----------------|
| `performance_overview` | Year's results, book value changes, comparison to benchmarks, multi-year performance data |
| `management_philosophy` | How decisions are made, what is avoided, operating principles, measurement standards |
| `insurance_operations` | Float, underwriting, premiums, insurance subsidiaries, reinsurance |
| `operating_businesses` | Non-insurance subsidiaries, manufacturing, retail, utilities, railroads |
| `investments` | Stock portfolio, bonds, derivatives, investment decisions, market commentary |
| `acquisitions` | Deals completed, acquisition criteria, stock vs cash considerations, merger economics |
| `corporate_governance` | Board matters, CEO responsibility, executive compensation, shareholder rights |
| `shareholder_matters` | Annual meeting, charitable contributions, shareholder communications |
| `other` | Content that doesn't fit above categories |

**Rule:** Classify by WHAT is discussed, not by WHERE it appears or what the header says.

---

## STEP 3: CHUNK TYPE CLASSIFICATION

Each chunk gets ONE type based on its primary nature:

### `narrative_story`
**Identify by:**
- Specific dates, names, places in sequence
- "Let me tell you..." or "Here's what happened..."
- Dialogue or quoted conversations
- Beginning-middle-end structure
- Personal anecdotes with punchlines
- Humorous or self-deprecating accounts

**Chunking Rule:** Keep entire story in ONE chunk, even if 300-500 words.

### `financial_table`
**Identify by:**
- Columnar data with numbers
- Year-over-year comparisons
- Lists of holdings with values
- Performance metrics in structured format

**Chunking Rule:** Include table + preceding explanation + following interpretation together.

### `philosophy`
**Identify by:**
- Abstract principles without specific examples
- "We believe...", "Our approach...", "The key is..."
- Timeless wisdom applicable beyond the specific year
- Discussion of WHY not just WHAT

**Chunking Rule:** 150-250 words, break at conceptual boundaries.

### `business_analysis`
**Identify by:**
- Discussion of specific company performance
- Named managers and their achievements
- Concrete metrics for specific businesses
- Industry-specific commentary

**Chunking Rule:** 200-300 words, one business unit or theme per chunk.

### `administrative`
**Identify by:**
- Meeting logistics, dates, locations
- Procedural shareholder information
- Routine announcements

**Chunking Rule:** Group by topic, mark as low retrieval priority.

---

## STEP 4: CHUNK BOUNDARY RULES

### NEVER Break:
- ❌ Mid-sentence (check for period before break)
- ❌ Inside a story/anecdote (keep narrative intact)
- ❌ Between a table and its explanation
- ❌ Inside a quoted passage
- ❌ Inside a numbered or bulleted list
- ❌ Between a question and its answer
- ❌ Between setup and punchline of a joke

### ALWAYS Break:
- ✅ At section headers (start new chunk)
- ✅ At asterisk dividers (`************`)
- ✅ When topic changes completely
- ✅ When switching from one company to another

### PREFER Breaking:
- ✅ Between paragraphs
- ✅ After a complete principle is stated
- ✅ When transitioning from abstract to specific (or vice versa)
- ✅ After a story concludes

---

## STEP 5: CONTEXTUAL SUMMARY GENERATION

**CRITICAL:** Each chunk needs a SPECIFIC contextual summary. DO NOT use templates.

### Bad (Generic Template):
```
"This chunk discusses [topic] in the context of [year]. 
Key point: it illustrates Berkshire's approach."
```

### Good (Specific Content):
```
"This chunk explains why Berkshire uses book value rather than 
stock price as its performance metric. Buffett acknowledges book 
value understates intrinsic value but argues it provides the most 
consistent tracking mechanism, particularly given short-term stock 
price volatility."
```

### Summary Formula:
1. **What:** State the specific topic/argument (not just category)
2. **Context:** Why this matters in this letter or situation
3. **Insight:** The key takeaway or principle (if any)

### Summary Length: 2-3 sentences (40-60 words)

---

## STEP 6: METADATA EXTRACTION

### Required Fields for Every Chunk:

```json
{
  "chunk_id": "{year}_{section_type}_{sequence:03d}",
  "year": integer,
  "source_file": "letter_{year}.pdf",
  "section_type": "from Step 2",
  "section_title": "exact header text or descriptive title",
  "subsection": "if nested, otherwise null",
  "parent_section": "if subsection exists, otherwise null",
  "position_in_letter": 0.0 to 1.0,
  "position_in_section": 0-indexed integer,
  "total_chunks_in_section": integer,
  "chunk_text": "the actual text content",
  "word_count": integer,
  "char_count": integer,
  "chunk_type": "from Step 3",
  "has_financials": true if contains $, %, or numeric metrics,
  "has_table": true if structured columnar data,
  "has_quote": true if contains quoted speech,
  "contains_principle": true if states timeless investing/business wisdom,
  "contains_example": true if provides concrete illustration,
  "contains_comparison": true if compares entities or time periods,
  "contextual_summary": "from Step 5",
  "prev_context": "1-2 sentences on what came before",
  "next_context": "1-2 sentences on what follows",
  "topics": ["array", "of", "3-6", "topic", "tags"],
  "companies_mentioned": ["extract", "all", "company", "names"],
  "people_mentioned": ["extract", "all", "person", "names"],
  "metrics_discussed": ["named", "metrics", "like", "ROE", "float"],
  "industries": ["relevant", "industry", "sectors"],
  "principle_category": "if contains_principle, else null",
  "principle_statement": "one sentence distillation, else null",
  "retrieval_priority": "high|medium|low",
  "abstraction_level": "high|medium|low",
  "time_sensitivity": "high|low",
  "is_complete_thought": true if standalone comprehensible,
  "needs_context": true if requires adjacent chunks
}
```

---

## STEP 7: PRINCIPLE EXTRACTION

When `contains_principle: true`, also provide:

### principle_category (pick one):
- `moats` - competitive advantages, barriers to entry
- `valuation` - how to value businesses, price vs value
- `management_quality` - what makes good managers/governance
- `capital_allocation` - how to deploy capital, dividends, buybacks
- `risk_management` - avoiding permanent loss, leverage, liquidity
- `competitive_advantage` - sustainable business strengths
- `business_quality` - characteristics of great businesses

### principle_statement:
- One sentence capturing the timeless wisdom
- Should be quotable and standalone
- Remove year-specific references

**Example:**
- Text: "We will never become dependent on the kindness of strangers. Too-big-to-fail is not a fallback position at Berkshire."
- principle_category: `risk_management`
- principle_statement: "Never become dependent on outside financing; maintain liquidity that dwarfs any conceivable cash needs, even if holding excess cash earns poor returns."

---

## STEP 8: PRIORITY ASSIGNMENT

### retrieval_priority:

| Priority | Assign When |
|----------|-------------|
| `high` | Contains timeless principle, famous quote, key strategic insight, or memorable story |
| `medium` | Specific business analysis, year-specific performance, concrete examples |
| `low` | Administrative content, routine announcements, meeting logistics |

### abstraction_level:

| Level | Characteristics |
|-------|-----------------|
| `high` | Abstract principles, philosophical statements, no specific numbers |
| `medium` | Principles illustrated with examples, mix of abstract and concrete |
| `low` | Specific facts, numbers, dates, named transactions |

### time_sensitivity:

| Level | Characteristics |
|-------|-----------------|
| `high` | Specific to that year's events, market conditions, or transactions |
| `low` | Timeless wisdom applicable across decades |

---

## STEP 9: QUALITY CHECKS

Before finalizing, verify:

### Completeness
- [ ] Every paragraph is in exactly one chunk
- [ ] No content is missing or duplicated
- [ ] All sections have at least one chunk

### Boundaries
- [ ] No chunk ends mid-sentence
- [ ] No stories are split across chunks
- [ ] Tables include their context

### Metadata
- [ ] All required fields populated
- [ ] chunk_id follows naming convention
- [ ] Word counts are accurate
- [ ] Companies/people extracted correctly

### Summaries
- [ ] Each summary is SPECIFIC (not templated)
- [ ] Summaries mention actual content, not just categories
- [ ] Principle statements are quotable

### Size Distribution
- [ ] Most chunks are 150-300 words
- [ ] Stories can be up to 500 words
- [ ] No chunks under 80 words unless table-only
- [ ] No chunks over 500 words

---

## STEP 10: OUTPUT FORMAT

Return ONLY a valid JSON array:

```json
[
  {
    "chunk_id": "YYYY_section_type_001",
    ...all fields...
  },
  {
    "chunk_id": "YYYY_section_type_002",
    ...all fields...
  }
]
```

**No markdown code fences. No explanatory text. Just the JSON array.**

---

## EXAMPLES OF GOOD CHUNKING DECISIONS

### Example 1: Keep Story Intact
**Input text includes:** A 250-word anecdote about how an acquisition happened at a birthday party, with specific dates, names, and a humorous punchline.

**Decision:** One chunk, type `narrative_story`, even though it exceeds 200 words.

### Example 2: Split Philosophy from Example  
**Input text includes:** A principle about competitive moats (100 words) followed by detailed GEICO analysis (200 words).

**Decision:** Two chunks - first is `philosophy` about moats, second is `business_analysis` of GEICO illustrating the principle.

### Example 3: Table With Context
**Input text includes:** Explanatory paragraph → financial table → interpretation paragraph.

**Decision:** One chunk, type `financial_table`, includes all three parts.

### Example 4: Identify Hidden Principles
**Input text:** "If Charlie, I and Ajit are ever in a sinking boat – and you can only save one of us – swim to Ajit."

**Decision:** Though humorous, this is `business_analysis` about Ajit Jain's value. Set `contains_principle: false` (it's praise, not a principle), but `retrieval_priority: high` because it's memorable.

### Example 5: Administrative vs Philosophy
**Input text:** Annual meeting logistics mixed with reflection on being lucky.

**Decision:** Split into two chunks. Logistics = `administrative` (low priority). Luck/gratitude reflection = `philosophy` (medium priority).

---

## COMMON MISTAKES TO AVOID

1. **Generic Summaries** - "This discusses X in context of Y" is useless. Be specific.

2. **Over-chunking** - Don't create 80+ tiny chunks. Aim for 30-50 for a typical letter.

3. **Under-chunking** - Don't create 10 giant chunks. Each should be focused.

4. **Missing Stories** - Anecdotes are often the most retrieved content. Identify them.

5. **Wrong Section Types** - Classify by content, not by document position.

6. **Splitting Punchlines** - If there's a setup-punchline structure, keep together.

7. **Ignoring Memorable Quotes** - Famous one-liners should be in chunks marked high priority.

8. **Template Summaries** - Each summary must be written fresh for that specific content.

---

## TARGET METRICS

For a typical 15-20 page Buffett letter, aim for:

| Metric | Target |
|--------|--------|
| Total chunks | 30-50 |
| Avg words/chunk | 175-225 |
| philosophy chunks | 30-45% |
| narrative_story chunks | 10-15% |
| business_analysis chunks | 30-40% |
| Contains principle | 40-60% |
| Contains example | 40-55% |
| High priority | 50-60% |
| Complete thoughts | 100% |

---

END OF CHUNKING RULES