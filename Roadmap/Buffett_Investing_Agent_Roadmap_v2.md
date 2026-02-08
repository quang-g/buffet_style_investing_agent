# Buffett Investing Agent (Vietnam) — v2 Roadmap
**Goal:** Build a Vietnam stock analyst agent that:
1) uses **Buffett shareholder-letter knowledge (1977–2024)**,  
2) queries **VN stock data via `vnstock`**,  
3) produces a **Buffett-style stock analysis report** and **answers questions** with **verifiable citations** (letters + data).

---

## North Star MVP (pick one wedge first)
### MVP A — 1‑Click Buffett‑Style Report (recommended)
- Input: `ticker` (e.g., FPT)
- Output: 1-page structured report + evidence pack (computed metrics + Buffett citations)

### MVP B — Analyst Chat (ticker + question)
- Input: `ticker` + user question
- Output: answer grounded in (a) `vnstock` numbers, (b) letter citations

**Why wedge first:** fastest way to test demand and pricing, without building everything.

---

## Guiding Principles
- **No numbers without source:** every numeric claim links to a data key + period.
- **No “Buffett says…” without citation:** every principle claim links to year/section/chunk.
- **Deterministic core, generative wrapper:** compute analysis signals deterministically, then let the LLM narrate.
- **Graceful degradation:** missing data → say so, adjust section outputs, never hallucinate.

---

## Product Output Spec (Report Template v1)
Each report must include:
1. **Business Quality** (profitability & consistency)
2. **Financial Strength** (leverage & liquidity)
3. **Cash Generation** (CFO/FCF proxies)
4. **Capital Allocation** (dividends/buybacks if available)
5. **Valuation Sanity Check** (simple, explainable)
6. **Key Risks & What Would Change the Thesis**
7. **Buffett Principles Used** (citations)
8. **Evidence Pack**
   - Data snapshot used (endpoints, periods)
   - Computed metrics table
   - Retrieved letter chunks (year/section/chunk_id)

---

## Phase 0 — Market Validation & Scope Lock (Days 1–5)
**Deliverables**
- Landing page (1 page) describing the wedge MVP + examples
- 3 demo prompts + 5 sample reports for popular VN tickers

**Acceptance Criteria**
- A fixed report template (sections + metric definitions) is written down
- Users can understand “what they get” in < 60 seconds
- At least 10–20 target users contacted (DM/interview) with feedback captured

**Success Signals**
- People ask “Can you do ticker X?”
- Requests for exports / watchlist / recurring updates
- Willingness-to-pay signals (even small)

---

## Phase 1 — VN Stock Data Layer (Days 6–15)
**Objective:** Build a robust, cached, normalized data service on top of `vnstock`.

**Deliverables**
- `data_service.py` returning a normalized `company_facts` object:
  - `price_history`
  - `financials` (IS/BS/CF, yearly/quarterly if available)
  - `company_profile`
  - optional: `valuation_multiples` / `market_overview` (if available)
- Caching + rate limiting + retries
- Data QA checks (missing values, units normalization, sanity checks)

**Acceptance Criteria**
- Running `python -m data_service FPT` returns JSON with consistent keys
- Cache hit reduces repeated calls for same (ticker, endpoint, period)
- Missing data is explicitly tagged (e.g., `null` + `missing_reason`)

**Risks & Mitigations**
- Provider throttling → exponential backoff, cached snapshots, fallback providers if possible
- Data inconsistency across tickers → strict normalization layer

---

## Phase 2 — Buffett Knowledge Base (Days 16–25)
**Objective:** Make your corpus “citation-grade” and retrieval-friendly for analysis.

**Deliverables**
- Standardized chunk schema enforcement:
  - `chunk_id`, `content`
  - `source_year`, `letter_date`
  - `section_path` (hierarchy)
  - `char_start/char_end` OR paragraph indices
  - optional: `entities`, `metrics`, `table_flag`
- Vector index (FAISS/Qdrant) + basic retrieval API
- A lightweight principle taxonomy (`principle_tags`), e.g.:
  - moat, pricing power, capital allocation, leverage, accounting, temperament, cycles, valuation

**Acceptance Criteria**
- Given a query, retrieval returns top-k chunks with stable IDs + section paths
- Citations are reproducible (same chunk retrieved under same index config)
- Chunk schema validation script passes on all years (or reports actionable errors)

---

## Phase 3 — Buffett-Style Analysis Engine (Days 26–45)
**Objective:** Convert `company_facts` → deterministic signals → structured analysis JSON.

**Deliverables**
- `analyzer.py` that outputs `analysis_object.json` with:
  - computed metrics (with formulas & periods)
  - findings (bullet claims) + confidence + missing-data notes
  - recommended Buffett principles to cite for each finding
- Metric set (v1, explainable):
  - Profitability: gross margin trend, operating margin trend
  - Quality proxies: ROA/ROE (if available), earnings consistency
  - Strength: debt/equity, interest coverage proxy, current ratio
  - Cash: CFO trend, capex proxy, FCF proxy (if possible)
  - Valuation: P/E, P/B, historical range vs current (if available)
- Mapping layer: finding → principle_tags → retrieve citations

**Acceptance Criteria**
- `analyzer.py --ticker FPT` produces JSON without LLM
- Every finding includes:
  - data sources + periods
  - formula reference
  - (optional) principle_tags to retrieve

---

## Phase 4 — Report Generator + Q&A (Days 46–65)
**Objective:** Ship the end-to-end user experience for the chosen MVP wedge.

**Deliverables**
- Single entrypoint (CLI or API):
  - `POST /analyze?ticker=FPT` → report + evidence pack
- Report generator:
  - renders report template v1
  - injects computed numbers + citations
- Q&A mode:
  - “Why did you say X?” → cite the numbers + letter excerpts
  - “What would Buffett worry about?” → cite relevant principles
- Guardrails:
  - numeric grounding enforcement
  - refusal to invent missing data

**Acceptance Criteria**
- For 10 tickers:
  - report renders successfully
  - at least 80% of numeric claims trace to a source
  - letter citations include year + section_path + chunk_id

---

## Phase 5 — Quality, Trust & Packaging (Days 66–90)
**Objective:** Make it reliable enough to sell and scale.

**Deliverables**
### 1) Evaluation Harness
- Test sets:
  - 20 “known questions” with expected citations
  - 20 numeric checks (recompute and compare)
  - hallucination tests (missing/partial data)
- Metrics tracked:
  - citation correctness rate
  - numeric correctness rate
  - missing-data disclosure rate
  - user satisfaction (simple thumbs)

### 2) Product Packaging
- Watchlist + saved reports
- Export: markdown/PDF
- Logging:
  - data snapshot hash
  - retrieved chunk IDs
  - final answers

### 3) Go-to-Market Experiments
- 20 public sample reports for popular tickers
- “Free 5 reports” offer to collect feedback
- Simple pricing test:
  - one-off report
  - monthly subscription (watchlist refresh)

**Acceptance Criteria**
- You can reproduce a report from the same data snapshot
- Regression tests catch citation drift and numeric errors
- First paying users OR strong preorders/waitlist conversion

---

## Suggested Repo Structure (practical)
```
buffett-vn-agent/
  data/
    corpora/                 # chunked letters
    indexes/                 # vector indexes
    snapshots/               # cached vnstock responses
  src/
    data_service.py
    kb_retriever.py
    analyzer.py
    report_renderer.py
    api.py
    eval/
      tests_citations.py
      tests_numbers.py
  docs/
    schema.md
    report_template.md
    roadmap_v2.md
```

---

## Key Risks (and how to handle them)
- **Data gaps / inconsistent availability:** show “unknown” explicitly; compute alternate proxies.
- **Citation drift from index changes:** version vector index configs; snapshot indexes per release.
- **Hallucinations:** enforce “no-source, no-claim” rule at generation time.
- **Overcomplexity:** keep deterministic core small; expand metrics only after demand is proven.

---

## Next Action (today)
1) Choose wedge MVP (A or B).  
2) Freeze Report Template v1 (sections + metric list).  
3) Build Phase 1 data_service with caching + normalization.
