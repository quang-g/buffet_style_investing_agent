# Phase 0 Plan (Concierge MVP) — FPT Demo
**Objective (Phase 0):** Create market-test artifacts *without building the full agent*:  
- **3 demo prompts** (copy/paste ready)  
- **1 sample Buffett-style analysis report for FPT** (repeatable process)  
- **Evidence pack**: vnstock data snapshot + computed metrics + Buffett-letter citations  

This is a **concierge MVP** approach: you manually deliver what will later be automated, to validate demand and learn what users actually want.  
References: Concierge MVP definitions and “manual delivery to validate” concept:  
- https://www.upsilonit.com/blog/what-is-a-concierge-mvp-and-when-to-use-it  
- https://www.empat.tech/blog/concierge-mvp  

---

## Inputs & Tools
### Inputs
- **Ticker:** `FPT`
- **Financial data source:** `vnstock` library (Python)
- **Buffett knowledge source:** your chunked Berkshire letters corpus (1977–2024 JSON chunks)

### Tools (Phase 0)
- **Python notebook/script** for data pull + metric computation  
- **Basic search over chunks** (keyword/BM25; embeddings optional)  
- **Product-ready LLM** (ChatGPT / Claude / Gemini) to draft the final report using *only provided numbers + citations*

vnstock docs/examples show using core classes such as `Listing, Quote, Company, Finance, Trading`:  
- https://github.com/thinh-vu/vnstock

Rate-limit and stability are known practical constraints; caching is recommended and rate-limit issues are discussed in vnstock community resources:  
- https://github.com/thinh-vu/vnstock/issues/166  
- https://vnstocks.com/blog/cap-nhat-phien-ban-vnstock-3-2-1  

---

## Deliverables (What you will produce)
1) `report_template_v1.md` — fixed one-page structure  
2) `snapshots/FPT_*` — raw vnstock outputs saved locally  
3) `snapshots/FPT_metrics.json` — computed metrics + formulas + periods  
4) `snapshots/FPT_citations_pack.json` — 10–20 Buffett chunks with IDs + excerpts  
5) `reports/FPT_report_v0.md` — the sample report  
6) `phase0_demo_prompts.md` — 3 demo prompts

---

## Step-by-step Execution Plan

### Step 1 — Freeze the “1-page Buffett-style report template” (60–120 mins)
Create a simple template you can reuse for every ticker. Keep it **1 page**.

**Template sections (v1):**
1. Business Quality (profitability + stability)
2. Financial Strength (leverage + liquidity)
3. Cash Generation (CFO/FCF proxies)
4. Capital Allocation (dividends/buybacks if available)
5. Valuation sanity check (simple & explainable)
6. Key risks + what would change the thesis
7. Buffett principles used (citations)
8. Evidence pack (data periods + chunk IDs)

**Output:** `report_template_v1.md`

---

### Step 2 — Pull FPT data with vnstock (2–4 hours)
In a notebook/script, fetch *only what you need to fill the template*.

**2.1 Setup**
- Install/upgrade vnstock
- Import core classes (typical approach shown in vnstock docs):  
  `from vnstock import Listing, Quote, Company, Finance, Trading`  
  Source: https://github.com/thinh-vu/vnstock

**2.2 Fetch**
- **Price history:** 1Y + 5Y (or max available)
- **Financial statements:** IS/BS/CF (yearly; quarterly if available)
- **Company profile:** industry/description (if available)

**2.3 Save snapshots (important for Phase 0 reproducibility + rate limits)**
Save raw outputs locally so you can regenerate reports *without re-calling providers*:
- `snapshots/FPT_price.csv`
- `snapshots/FPT_income_statement.csv`
- `snapshots/FPT_balance_sheet.csv`
- `snapshots/FPT_cashflow.csv`
- `snapshots/FPT_profile.json`

**Operational note (do this now):**
- Implement “poor man’s caching”: if snapshot exists, reuse it.
- Rate-limit constraints are a known concern; caching and proactive warnings are discussed by the vnstock ecosystem:  
  https://github.com/thinh-vu/vnstock/issues/166  
  https://vnstocks.com/blog/cap-nhat-phien-ban-vnstock-3-2-1  

---

### Step 3 — Compute a small metric set (60–120 mins)
Goal: 10–15 metrics max. Phase 0 is about *believable output*, not completeness.

**Business quality**
- Revenue CAGR (3Y/5Y if available)
- Gross margin trend
- Operating margin trend

**Financial strength**
- Debt/Equity
- Current ratio (if BS supports)

**Cash generation**
- CFO trend
- FCF proxy = CFO – Capex (only if capex exists)

**Valuation sanity**
- P/E, P/B (only if vnstock provides; otherwise mark missing)

**Output format:** `snapshots/FPT_metrics.json` containing:
- `metric_name`
- `value(s)`
- `period(s)` (e.g., 2020–2024)
- `formula`
- `source_fields` (which vnstock columns were used)
- `notes_on_missing_data`

---

### Step 4 — Build “Buffett principle citations” using basic search over your chunks (2–4 hours)
This step turns your letter corpus into a **citation pack** usable by any LLM.

#### 4.1 Create a mapping: Report section → principle queries
Example queries (edit to match your schema/sections):
- **Moat/pricing power:** “pricing power”, “brand”, “moat”, “raise prices”
- **Leverage/risk:** “debt”, “leverage”, “liquidity”, “staying power”
- **Owner earnings / cash:** “owner earnings”, “free cash flow”, “capital expenditures”
- **Capital allocation:** “repurchase”, “buyback”, “dividend”, “retained earnings”
- **Valuation discipline:** “intrinsic value”, “margin of safety”

#### 4.2 Search strategy (Phase 0)
- Start with **keyword / BM25** over `content` fields in your chunk JSON.
- Keep top ~2–3 chunks per query, total **10–20 chunks**.

#### 4.3 Save a strict citation pack
For each selected chunk, store:
- `source_year`
- `section_path` (or equivalent)
- `chunk_id`
- `excerpt` (2–4 sentences max)
- `principle_tags` (optional)
- `why_relevant` (one sentence)

**Output:** `snapshots/FPT_citations_pack.json`

---

### Step 5 — Generate the sample report using a product-ready LLM (60–120 mins)
You now have:
- Template (`report_template_v1.md`)
- Numbers (`FPT_metrics.json`)
- Citations (`FPT_citations_pack.json`)

#### 5.1 Use a “grounded generation” prompt
**Rules to enforce in the prompt:**
- **No-source, no-claim:** If a number isn’t in metrics JSON, don’t mention it.
- Every numeric claim must include: **metric name + period**
- Every Buffett claim must cite: **(year + section_path + chunk_id)**
- Missing data must be stated explicitly (e.g., “Không đủ dữ liệu từ snapshot vnstock”).

#### 5.2 Output
- `reports/FPT_report_v0.md` (1 page)
- Include “Evidence pack” section listing:
  - data snapshot files
  - computed metrics list
  - chunk IDs used

---

### Step 6 — Produce 3 demo prompts (30 mins)
Create prompts that reflect real user intent. Save as `phase0_demo_prompts.md`.

**Prompt 1 — 1-click report**
> Create a 1-page Buffett-style report for FPT using the provided vnstock metrics snapshot and Buffett citation pack. Every number must reference the metric + period. Every Buffett principle must cite year + chunk_id + section_path. If data is missing, say so.

**Prompt 2 — Moat / quality deep dive**
> Does FPT show signs of a durable moat and pricing power? Use only the provided metrics and cite Buffett letters (year + chunk info) to justify the reasoning.

**Prompt 3 — Risks & thesis breakers**
> What are the top 3 risks for FPT and what metrics would change your thesis? Ground each risk in the snapshot metrics. Cite Buffett principles about risk, leverage, and uncertainty.

---

### Step 7 — Market test loop (same week; still Phase 0)
Show the report to 10–20 target users (VN investors, analysts, finance creators). Collect feedback with these questions:
1) Would you pay for this? (per report vs subscription)
2) Which section is most valuable?
3) What’s missing to make it “investment-usable”?
4) Which tickers do you want next?
5) Vietnamese vs English vs bilingual?

**Success signals**
- Repeat intent (“do ticker X next”)
- Follow-up questions about the same report (Q&A value)
- Willingness-to-pay or preorders

---

## Folder/Files to Create (Phase 0)
```
phase0/
  report_template_v1.md
  phase0_demo_prompts.md
  snapshots/
    FPT_price.csv
    FPT_income_statement.csv
    FPT_balance_sheet.csv
    FPT_cashflow.csv
    FPT_profile.json
    FPT_metrics.json
    FPT_citations_pack.json
  reports/
    FPT_report_v0.md
  notes/
    user_feedback.md
```

---

## References (for credibility & later documentation)
- vnstock (library, examples, modules): https://github.com/thinh-vu/vnstock  
- vnstock rate-limit/stability discussion (example): https://github.com/thinh-vu/vnstock/issues/166  
- vnstock release notes mentioning proactive rate-limit warning: https://vnstocks.com/blog/cap-nhat-phien-ban-vnstock-3-2-1  
- Concierge MVP concept (manual delivery to validate demand):
  - https://www.upsilonit.com/blog/what-is-a-concierge-mvp-and-when-to-use-it  
  - https://www.empat.tech/blog/concierge-mvp  
