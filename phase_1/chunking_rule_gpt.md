# LLM Chunking Rules for Buffett Letters
## Instruction Guide for Non-Reasoning Models

---

## TASK OVERVIEW

You are chunking Warren Buffett's annual shareholder letters into semantically meaningful segments for a RAG (Retrieval-Augmented Generation) system. Your output must be a valid JSON array of chunk objects.

**Primary Goals:**
1. Preserve narrative flow and complete thoughts
2. Extract rich, specific metadata for retrieval
3. Create chunks that stand alone while maintaining context
4. Identify principles, stories, and key insights

---

## PHASE 1: DOCUMENT ANALYSIS

Before chunking, scan the entire document to identify:

### 1.1 Section Detection
Identify major sections by looking for:
- **Capitalized headers** or **bold text** that introduce new topics
- **Thematic shifts** in content (e.g., from insurance to utilities)
- **Transition phrases** like "Let's move to...", "Now let's discuss...", "Turning to..."
- **Asterisk separators** (********) which Buffett uses between sections

Assign each section a `section_type` from these categories:
| Category | Indicators |
|----------|------------|
| `performance_overview` | Annual results, book value, S&P comparison, net worth gains |
| `insurance_operations` | Insurance, underwriting, float, reinsurance, specific insurers |
| `investments` | Stock holdings, portfolio, securities, investment decisions |
| `acquisitions` | Purchases, mergers, stock issuance for deals, acquisition philosophy |
| `operating_businesses` | Subsidiaries, manufacturing, retail, utilities, railroads, specific business units |
| `management_philosophy` | Principles, what we do/don't do, how we operate, culture |
| `corporate_governance` | Boards, CEOs, compensation, accountability, risk oversight |
| `shareholder_matters` | Annual meeting, shareholder communications, administrative info |
| `other` | Content not fitting above categories |

### 1.2 Content Type Classification
For each segment, classify the content type:

| Type | Identification Criteria |
|------|------------------------|
| `philosophy` | Abstract principles, "We believe...", "Our goal...", "Charlie and I think...", investment/business wisdom |
| `business_analysis` | Discussion of specific companies, performance metrics, competitive position |
| `narrative_story` | Anecdotes with characters, dialogue, sequence of events, humor, specific dates |
| `financial_table` | Structured numerical data, year-over-year comparisons, earnings breakdowns |
| `administrative` | Meeting logistics, contact info, procedural matters |

---

## PHASE 2: CHUNKING RULES

### 2.1 Target Chunk Size
- **Standard chunks:** 150-300 words
- **Narrative stories:** Keep intact up to 500 words (never break a story)
- **Financial tables:** Include table + explanation paragraph(s) as single chunk
- **Minimum:** 80 words (unless table-only)

### 2.2 Boundary Rules

**NEVER break:**
- ❌ Mid-sentence
- ❌ Inside a story or anecdote (look for narrative arc completion)
- ❌ Between a table and its explanatory text
- ❌ Inside a quoted passage
- ❌ Inside a bulleted or numbered list
- ❌ Between a principle statement and its supporting example

**ALWAYS break at:**
- ✅ Paragraph boundaries
- ✅ Section/subsection transitions
- ✅ Topic shifts within a section
- ✅ Before/after asterisk separators (********)
- ✅ After a complete conceptual unit

### 2.3 Story Detection
Identify narrative stories by these markers:
- Named people with actions ("Roy said...", "I told him...")
- Specific dates or timeframes ("In January 1996...", "Last year...")
- Dialogue or quoted speech
- Sequence words ("Then...", "Finally...", "The next day...")
- Humorous punchlines or memorable conclusions
- First-person recollections ("I remember when...", "Let me tell you about...")

**Keep entire stories in one chunk** even if exceeding 300 words.

### 2.4 Principle Detection
Identify investment/business principles by:
- Declarative statements about how to invest or run businesses
- "We look for...", "The key is...", "We believe...", "Our approach..."
- Generalizable wisdom (not just specific facts)
- Contrast statements ("Unlike others, we...", "We don't...")
- Memorable aphorisms or quotable phrases

---

## PHASE 3: METADATA GENERATION

For each chunk, generate ALL of these fields:

```json
{
  "chunk_id": "{year}_{section_type}_{sequence:03d}",
  "year": integer,
  "source_file": "letter_{year}.pdf",
  "section_type": "string (from categories above)",
  "section_title": "string (actual header text or inferred title)",
  "subsection": "string or null",
  "parent_section": "string or null",
  "position_in_letter": float (0.0 to 1.0),
  "position_in_section": integer (0-indexed),
  "total_chunks_in_section": integer,
  "chunk_text": "string (the actual content)",
  "word_count": integer,
  "char_count": integer,
  "chunk_type": "string (philosophy|business_analysis|narrative_story|financial_table|administrative)",
  "has_financials": boolean,
  "has_table": boolean,
  "has_quote": boolean,
  "contains_principle": boolean,
  "contains_example": boolean,
  "contains_comparison": boolean,
  "contextual_summary": "string (2-3 sentences)",
  "prev_context": "string (1-2 sentences or empty for first chunk)",
  "next_context": "string (1-2 sentences or empty for last chunk)",
  "topics": ["array", "of", "topic", "tags"],
  "companies_mentioned": ["array", "of", "company", "names"],
  "people_mentioned": ["array", "of", "people", "names"],
  "metrics_discussed": ["array", "of", "metric", "names"],
  "industries": ["array", "of", "industry", "names"],
  "principle_category": "string or null",
  "principle_statement": "string or null",
  "retrieval_priority": "high|medium|low",
  "abstraction_level": "high|medium|low",
  "time_sensitivity": "high|low",
  "is_complete_thought": boolean,
  "needs_context": boolean
}
```

---

## PHASE 4: CONTEXTUAL SUMMARY GENERATION

### 4.1 Summary Formula
Write summaries following this structure:

```
"This chunk [ACTION VERB] [SPECIFIC TOPIC] in the context of [YEAR'S SITUATION/BROADER THEME]. 
[SPECIFIC DETAIL: numbers, names, or key facts]. 
[INSIGHT: what principle or conclusion Buffett draws]."
```

### 4.2 Action Verbs by Chunk Type
- **philosophy:** "articulates", "explains", "argues", "establishes"
- **business_analysis:** "analyzes", "reports", "profiles", "examines"
- **narrative_story:** "recounts", "tells", "illustrates through story"
- **financial_table:** "presents", "summarizes", "compares"
- **administrative:** "provides", "announces", "details"

### 4.3 Summary Quality Rules
- ✅ Include specific numbers when present ("$21.8 billion", "19.8%")
- ✅ Name companies and people mentioned
- ✅ State the key insight or principle if present
- ✅ Reference the year's context when relevant
- ❌ Don't use generic phrases like "discusses various aspects"
- ❌ Don't repeat the same template across chunks
- ❌ Don't exceed 3 sentences

### 4.4 Examples of Good vs Bad Summaries

**BAD (generic):**
```
"This chunk discusses performance in the context of 2009. 
Key point: it illustrates Berkshire's approach."
```

**GOOD (specific):**
```
"This chunk reports Berkshire's 2009 results: $21.8 billion net worth 
gain and 19.8% book value increase. Buffett contextualizes this within 
the 45-year track record of 20.3% compounded growth and welcomes 65,000 
new shareholders from the BNSF acquisition."
```

---

## PHASE 5: ENTITY EXTRACTION

### 5.1 Topics
Extract 2-6 topic tags per chunk. Use snake_case. Examples:
- `float`, `underwriting_profit`, `competitive_advantage`
- `capital_allocation`, `acquisition_discipline`, `management_autonomy`
- `crisis_investing`, `risk_management`, `intrinsic_value`

### 5.2 Companies
Extract ALL company names mentioned, including:
- Berkshire subsidiaries (GEICO, See's, NetJets, etc.)
- Portfolio holdings (Coca-Cola, Wells Fargo, etc.)
- Comparison companies (competitors, examples)
- Use official names, not abbreviations

### 5.3 People
Extract named individuals:
- Berkshire managers (Tony Nicely, Ajit Jain, etc.)
- Business partners (Charlie Munger)
- Historical figures mentioned (Ben Graham, etc.)
- Include Warren Buffett only when he's referenced in third person

### 5.4 Metrics
Extract financial/business metrics discussed:
- `book_value`, `float`, `underwriting_profit`
- `return_on_equity`, `profit_margin`, `market_share`
- `debt`, `cash_position`, `premium_volume`

---

## PHASE 6: CLASSIFICATION RULES

### 6.1 Retrieval Priority
| Priority | Criteria |
|----------|----------|
| `high` | Contains principle, famous quote, key philosophy, major insight |
| `medium` | Business analysis, specific examples, financial results |
| `low` | Administrative content, meeting details, routine updates |

### 6.2 Abstraction Level
| Level | Criteria |
|-------|----------|
| `high` | Timeless principles, generalizable wisdom, no specific numbers |
| `medium` | Principles illustrated with specific examples |
| `low` | Specific facts, numbers, events tied to that year |

### 6.3 Time Sensitivity
| Level | Criteria |
|-------|----------|
| `high` | Specific to that year's conditions, dated information |
| `low` | Timeless principle, applicable across years |

### 6.4 Principle Categories
When `contains_principle: true`, assign one category:
- `moats` / `competitive_advantage` - Durable advantages, market position
- `valuation` - Intrinsic value, price vs value, margin of safety
- `management_quality` - Leadership, integrity, owner-orientation
- `capital_allocation` - Reinvestment, acquisitions, dividends
- `risk_management` - Leverage, liquidity, catastrophe avoidance
- `business_quality` - Economics, returns on capital, growth

---

## PHASE 7: VALIDATION CHECKLIST

Before finalizing, verify each chunk:

- [ ] `chunk_text` is a complete thought (doesn't end mid-sentence)
- [ ] `word_count` matches actual word count
- [ ] `chunk_id` follows format: `{year}_{section_type}_{000}`
- [ ] `contextual_summary` is specific, not generic
- [ ] `companies_mentioned` includes all companies in text
- [ ] `people_mentioned` includes all named individuals
- [ ] `contains_principle` is true only if a generalizable principle exists
- [ ] Narrative stories are kept intact (not split across chunks)
- [ ] Tables include their explanatory paragraphs

---

## OUTPUT FORMAT

Return ONLY a valid JSON array. No markdown code fences. No explanatory text before or after.

```json
[
  { chunk_1 },
  { chunk_2 },
  ...
  { chunk_n }
]
```

---

## EXAMPLE CHUNKS

### Example 1: Philosophy Chunk
```json
{
  "chunk_id": "2009_management_philosophy_002",
  "year": 2009,
  "section_type": "management_philosophy",
  "section_title": "What We Don't Do",
  "subsection": "Financial Independence",
  "chunk_type": "philosophy",
  "chunk_text": "We will never become dependent on the kindness of strangers. Too-big-to-fail is not a fallback position at Berkshire. Instead, we will always arrange our affairs so that any requirements for cash we may conceivably have will be dwarfed by our own liquidity...",
  "word_count": 199,
  "contextual_summary": "This chunk articulates Berkshire's core principle of financial independence. Buffett contrasts Berkshire's position during the 2008 crisis—deploying $15.5 billion as a capital provider—with institutions requiring bailouts, emphasizing that holding $20+ billion in cash is worth the security cost.",
  "contains_principle": true,
  "principle_category": "risk_management",
  "principle_statement": "Never become dependent on outside financing; maintain liquidity that dwarfs any conceivable cash needs.",
  "retrieval_priority": "high",
  "abstraction_level": "high",
  "time_sensitivity": "low"
}
```

### Example 2: Narrative Story Chunk
```json
{
  "chunk_id": "2009_acquisitions_004",
  "year": 2009,
  "section_type": "acquisitions",
  "section_title": "An Inconvenient Truth (Boardroom Overheating)",
  "chunk_type": "narrative_story",
  "chunk_text": "I can't resist telling you a true story from long ago. We owned stock in a large well-run bank that for decades had been statutorily prevented from acquisitions. Eventually, the law was changed and our bank immediately began looking for possible purchases. Its managers – fine people and able bankers – not unexpectedly began to behave like teenage boys who had just discovered girls...",
  "word_count": 278,
  "contextual_summary": "This chunk tells a memorable story about acquisition folly. A well-run bank paid three times book value in stock for a smaller bank despite trading near book value itself. Charlie's response: 'Are we supposed to applaud because the dog that fouls our lawn is a Chihuahua rather than a Saint Bernard?'",
  "contains_principle": true,
  "contains_example": true,
  "principle_category": "capital_allocation",
  "principle_statement": "Small value-destroying deals are not excused by their size.",
  "retrieval_priority": "high",
  "abstraction_level": "medium"
}
```

### Example 3: Business Analysis Chunk
```json
{
  "chunk_id": "2009_insurance_operations_003",
  "year": 2009,
  "section_type": "insurance_operations",
  "section_title": "Insurance - GEICO",
  "chunk_type": "business_analysis",
  "chunk_text": "Let's start at GEICO, which is known to all of you because of its $800 million annual advertising budget (close to twice that of the runner-up advertiser in the auto insurance field). GEICO is managed by Tony Nicely, who joined the company at 18...",
  "word_count": 148,
  "contextual_summary": "This chunk profiles GEICO's success under CEO Tony Nicely's leadership. Since Berkshire's 1996 acquisition, GEICO's market share tripled from 2.5% to 8.1% through the addition of seven million policyholders, driven by its $800 million advertising budget and competitive pricing.",
  "companies_mentioned": ["GEICO", "Berkshire Hathaway"],
  "people_mentioned": ["Tony Nicely"],
  "metrics_discussed": ["market_share", "advertising_budget", "policyholder_count"],
  "contains_principle": false,
  "retrieval_priority": "medium",
  "abstraction_level": "low",
  "time_sensitivity": "high"
}
```

---

## COMMON MISTAKES TO AVOID

1. **Generic summaries** - Every summary must be unique and specific
2. **Breaking stories** - Keep anecdotes with punchlines intact
3. **Missing entities** - Extract ALL companies and people mentioned
4. **Over-chunking** - Don't create chunks under 80 words
5. **Under-chunking** - Don't create chunks over 350 words (except stories)
6. **Wrong section_type** - Acquisition philosophy goes in `acquisitions`, not `corporate_governance`
7. **Missing principles** - If Buffett states a generalizable truth, mark `contains_principle: true`
8. **Forgetting tables** - Financial tables need their context paragraphs

---

## FINAL INSTRUCTION

Process the provided letter year by year. For each letter:
1. Identify all sections and their boundaries
2. Chunk each section following the rules above
3. Generate complete metadata for every chunk
4. Validate all chunks before output
5. Return only the JSON array

Begin processing now.