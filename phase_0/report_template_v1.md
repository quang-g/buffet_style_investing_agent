# Buffett-Style One-Page Stock Report (Vietnam) — {{ticker}} ({{company_name}})

> **One-line thesis:** {{thesis_one_liner}}  
> **As of:** {{as_of_date}} | **Currency/Units:** {{currency_units}} | **Period covered:** {{period_covered}}

---

## 1. Business Quality (profitability + stability)

**A) Intent:** Judge whether the business is predictably profitable and likely to stay that way.

**B) What to write here:** Summarize how the company makes money, why returns are durable, and whether results are stable across cycles.

**C) Checklist:**
- Profitability level and trend: {{ROE_or_ROIC}}, {{gross_margin}}, {{net_margin}} across {{period}} → {{value}}
- Stability: revenue/earnings volatility, drawdowns, and recovery over {{period}} → {{value}}
- Moat signals: pricing power, switching costs, brand, network effects (evidence-based)
- Competitive position: market share / relative scale / cost advantage (if available)
- Business simplicity and key drivers (volume, price, mix): {{key_driver_metric}} over {{period}} → {{value}}

**D) Write-up block:**  
{{company_name}} earns money primarily from {{revenue_streams}}. Over {{period}}, profitability has been {{stable_or_volatile}}, with {{ROE_or_ROIC}} at {{value}} and margins ({{gross_margin}} / {{net_margin}}) at {{value}} / {{value}}. Evidence of durable advantage is {{moat_evidence_summary}}; pricing power appears {{strong_or_weak}} based on {{pricing_power_evidence}}. Overall business quality is {{quality_rating}} because {{plain_language_reason}}.

**E) Citations to include:**
- [DATA:{{ROE_or_ROIC}}|{{period}}]
- [DATA:{{gross_margin}}|{{period}}]
- [DATA:{{net_margin}}|{{period}}]
- [DATA:{{revenue_growth}}|{{period}}]
- [WB:1986|Owner Earnings / Economic Reality|{{chunk_id}}]
- [WB:{{year}}|Moat / Pricing Power discussion|{{chunk_id}}]

**F) Missing data:** Missing data: {{what_is_missing}} → Needed: {{what_to_fetch_next}}

---

## 2. Financial Strength (leverage + liquidity)

**A) Intent:** Test survivability under a bad scenario and avoid leverage-driven permanent loss.

**B) What to write here:** Describe balance-sheet resilience, debt burden, liquidity, and whether the firm can endure stress without dilution.

**C) Checklist:**
- Leverage: {{debt_to_equity}} / {{net_debt_to_EBITDA}} over {{period}} → {{value}}
- Liquidity: {{current_ratio}} / {{cash_and_equivalents}} over {{period}} → {{value}}
- Interest coverage: {{interest_coverage}} over {{period}} → {{value}}
- Debt structure: maturities, FX exposure, covenants (if disclosed)
- Bad-scenario survivability: “Could it fund operations for 12–24 months?” using {{cash}} + {{CFO}} vs {{fixed_obligations}}

**D) Write-up block:**  
Financial strength is {{strong_or_weak}}. Net leverage ({{net_debt_to_EBITDA}}) is {{value}} over {{period}}, and liquidity ({{current_ratio}}) is {{value}}. Interest coverage is {{value}}, suggesting {{comfort_or_pressure}}. In a downside case of {{stress_case_assumption}}, the company likely {{can_or_cannot}} avoid dilution because {{survivability_logic_with_placeholders}}.

**E) Citations to include:**
- [DATA:{{debt_to_equity}}|{{period}}]
- [DATA:{{net_debt_to_EBITDA}}|{{period}}]
- [DATA:{{current_ratio}}|{{period}}]
- [DATA:{{interest_coverage}}|{{period}}]
- [WB:{{year}}|Balance Sheet / Conservatism / Leverage warnings|{{chunk_id}}]

**F) Missing data:** Missing data: {{what_is_missing}} → Needed: {{what_to_fetch_next}}

---

## 3. Cash Generation (CFO/FCF proxies)

**A) Intent:** Focus on cash reality and owner earnings, not just accounting profits.

**B) What to write here:** Evaluate cash from operations, reinvestment needs, and a practical proxy for owner earnings (incl. maintenance capex).

**C) Checklist:**
- Cash conversion: {{CFO}} vs {{net_income}} over {{period}} → {{value}}
- FCF proxy: {{CFO}} - {{capex}} over {{period}} → {{value}}
- Maintenance vs growth capex: estimate {{maintenance_capex_proxy}} → {{value}}
- Working capital behavior: {{working_capital_change}} over {{period}} → {{value}}
- Owner earnings proxy: {{CFO}} - {{maintenance_capex_proxy}} ± {{other_adjustments}} → {{value}}

**D) Write-up block:**  
Cash generation is {{good_or_poor}}. Over {{period}}, {{CFO}} was {{value}} versus net income {{value}}, implying cash conversion of {{cash_conversion_ratio}}. A simple FCF proxy ({{CFO}} - {{capex}}) is {{value}}. Estimated maintenance capex is {{maintenance_capex_proxy}} at {{value}}, so owner-earnings proxy is {{owner_earnings_proxy}} at {{value}}. This supports (or weakens) the thesis because {{plain_language_cash_reason}}.

**E) Citations to include:**
- [DATA:{{CFO}}|{{period}}]
- [DATA:{{capex}}|{{period}}]
- [DATA:{{net_income}}|{{period}}]
- [DATA:{{working_capital_change}}|{{period}}]
- [WB:1986|Owner Earnings definition and adjustments|{{chunk_id}}]

**F) Missing data:** Missing data: {{what_is_missing}} → Needed: {{what_to_fetch_next}}

---

## 4. Capital Allocation (dividends/buybacks if available)

**A) Intent:** Judge whether management increases per-share intrinsic value with retained earnings.

**B) What to write here:** Explain how cash is used (reinvest, dividends, buybacks, M&A, debt paydown) and whether actions are owner-friendly.

**C) Checklist:**
- Reinvestment returns: evidence that reinvested capital earns {{ROIC_or_incremental_ROE}} → {{value}}
- Dividends: payout stability and rationale: {{dividend_per_share}} / {{payout_ratio}} over {{period}} → {{value}}
- Buybacks (if any): shares reduced, price vs value, and whether repurchases were value-accretive
- M&A / expansions: track record, goodwill, and integration risk (if applicable)
- “Retained earnings test”: did retained profits translate into higher per-share value drivers?

**D) Write-up block:**  
Management’s capital allocation appears {{high_or_low_quality}}. Over {{period}}, cash was primarily used for {{uses_of_cash_ranked}}. Dividends were {{stable_or_unstable}} with {{payout_ratio}} at {{value}}. Buybacks were {{none_or_present}}; if present, share count changed by {{share_count_change}} and repurchases look {{value_accretive_or_not}} because {{buyback_value_logic}}. Overall, retained earnings have {{compounded_or_not}} per-share value as seen in {{per_share_metric}} at {{value}}.

**E) Citations to include:**
- [DATA:{{dividend_per_share}}|{{period}}]
- [DATA:{{payout_ratio}}|{{period}}]
- [DATA:{{share_count}}|{{period}}]
- [DATA:{{ROIC_or_incremental_ROE}}|{{period}}]
- [WB:{{year}}|Capital allocation / retained earnings discussion|{{chunk_id}}]
- [WB:{{year}}|Repurchases: when buybacks help owners|{{chunk_id}}]

**F) Missing data:** Missing data: {{what_is_missing}} → Needed: {{what_to_fetch_next}}

---

## 5. Valuation Sanity Check (simple & explainable)

**A) Intent:** Separate price from value and check if today’s price is sensible versus long-term owner earnings.

**B) What to write here:** Use simple valuation anchors (multiples, yields, and an intrinsic-value range) tied to owner earnings or cash power.

**C) Checklist:**
- Price context: {{market_cap}} / {{share_price}} as of {{as_of_date}} → {{value}}
- Earnings and cash multiples: {{P/E}}, {{EV/EBIT}}, {{P/FCF_proxy}} over {{period}} → {{value}}
- Owner-earnings yield: {{owner_earnings_proxy}} / {{market_cap}} → {{value}}
- Intrinsic value range: base / bear / bull using conservative assumptions
- Margin of safety: compare price to value range; note uncertainty drivers

**D) Write-up block:**  
At {{as_of_date}}, the stock trades at {{share_price}} with market cap {{market_cap}}. Valuation is {{cheap_or_expensive_or_fair}} versus history/peers using {{P/E}} at {{value}} and {{P/FCF_proxy}} at {{value}} ({{period}}). Using owner-earnings proxy of {{owner_earnings_proxy}} ({{period}}), the implied owner-earnings yield is {{owner_earnings_yield}}. A simple intrinsic value range is {{intrinsic_value_bear}}–{{intrinsic_value_base}}–{{intrinsic_value_bull}} per share, driven by {{key_assumptions}}. Current price implies {{implied_expectations_plain_language}}.

**E) Citations to include:**
- [DATA:{{share_price}}|{{as_of_date}}]
- [DATA:{{market_cap}}|{{as_of_date}}]
- [DATA:{{P/E}}|{{period}}]
- [DATA:{{P/FCF_proxy}}|{{period}}]
- [DATA:{{owner_earnings_proxy}}|{{period}}]
- [WB:2011|Per-share intrinsic business value mindset|{{chunk_id}}]
- [WB:{{year}}|Price vs value / margin of safety framing|{{chunk_id}}]

**F) Missing data:** Missing data: {{what_is_missing}} → Needed: {{what_to_fetch_next}}

---

## 6. Key Risks & What Would Change the Thesis

**A) Intent:** Identify the few risks that could cause permanent loss, and define measurable “thesis breakers.”

**B) What to write here:** List 3–5 specific, monitorable risks; then state what evidence would invalidate the thesis.

**C) Checklist:**
- Business risk: demand, regulation, competition, disruption (tie to metrics)
- Financial risk: leverage, refinancing, FX, liquidity squeeze (tie to metrics)
- Execution risk: capex overruns, product cycle, M&A integration
- Governance risk: related-party, disclosure quality, dilution history
- Thesis breakers: explicit triggers with {{metric_name}} over {{period}} → {{value}}

**D) Write-up block:**  
Top risks are: (1) {{risk_1}} linked to {{metric_name}} at {{value}} over {{period}}; (2) {{risk_2}} linked to {{metric_name}} at {{value}}; (3) {{risk_3}}. The thesis would change if {{thesis_breaker_1_metric}} falls below {{value}} for {{period}}, if leverage rises to {{metric_name}} {{value}}, or if pricing power weakens as shown by {{margin_metric}} declining to {{value}}.

**E) Citations to include:**
- [DATA:{{risk_metric_1}}|{{period}}]
- [DATA:{{risk_metric_2}}|{{period}}]
- [DATA:{{margin_metric}}|{{period}}]
- [WB:{{year}}|Risk / permanence of capital / conservatism|{{chunk_id}}]

**F) Missing data:** Missing data: {{what_is_missing}} → Needed: {{what_to_fetch_next}}

---

## 7. Buffett Principles Used (citations)

**A) Intent:** Make the reasoning explicit and grounded in Buffett doctrine, not vibes.

**B) What to write here:** Map the report’s key conclusions to Buffett principles and cite the exact letter chunks used.

**C) Checklist:**
- Owner earnings / cash reality applied to {{ticker}} (with one concrete metric)
- Intrinsic value per share mindset (price ≠ value) applied to valuation section
- Moat/pricing power evidence summarized in plain terms
- Low-leverage survivability logic tied to balance sheet and stress case
- Capital allocation (retained earnings, buybacks) tied to per-share outcomes

**D) Write-up block:**  
This report prioritizes **owner earnings** by using {{owner_earnings_proxy}} ({{period}}) rather than only GAAP earnings, reflecting Buffett’s cash-reality lens. It separates **price from intrinsic value** by presenting a conservative value range and margin-of-safety discussion. It tests **moat/pricing power** using {{margin_metric}} and qualitative evidence. It emphasizes **survivability** via leverage/liquidity checks and a downside case. It evaluates **capital allocation** by asking whether retained earnings increased per-share value and whether any buybacks were value-accretive.

**E) Citations to include:**
- [WB:1986|Owner Earnings framing|{{chunk_id}}]
- [WB:2011|Intrinsic business value per share mindset|{{chunk_id}}]
- [WB:{{year}}|Moat / pricing power|{{chunk_id}}]
- [WB:{{year}}|Leverage / conservatism / survivability|{{chunk_id}}]
- [WB:{{year}}|Repurchases: value-accretive buybacks|{{chunk_id}}]
- [DATA:{{owner_earnings_proxy}}|{{period}}]

**F) Missing data:** Missing data: {{what_is_missing}} → Needed: {{what_to_fetch_next}}

---

## 8. Evidence Pack (data periods + chunk IDs)

**A) Intent:** Make the report auditable: every key claim ties to a data point or a Buffett-letter chunk.

**B) What to write here:** List the exact periods, endpoints, and Buffett letter chunk IDs used so someone can reproduce the report.

**C) Checklist:**
- All key metrics have a [DATA:...] tag with {{period}}
- Every Buffett concept cited with [WB:{{year}}|{{section_path}}|{{chunk_id}}]
- Data endpoints are listed (vnstock) and match the metrics used
- Clearly state “as of” date and currency/units
- Note any gaps and the next fetch required

**D) Write-up block:**  
This report uses vnstock data for {{ticker}} covering {{period_covered}} and cites specific Berkshire letter chunks to ground the valuation and quality framework. Any claim not backed by a [DATA:...] or [WB:...] tag should be treated as an assumption and flagged for follow-up.

**E) Citations to include:**
- [DATA:{{metric_name}}|{{period}}]
- [DATA:{{metric_name}}|{{period}}]
- [WB:{{year}}|{{section_path}}|{{chunk_id}}]
- [WB:{{year}}|{{section_path}}|{{chunk_id}}]

**F) Missing data:** Missing data: {{what_is_missing}} → Needed: {{what_to_fetch_next}}

### Evidence Pack Table

| Item Type | ID (metric_name or chunk_id) | Period / Year | Source (vnstock endpoint or letter section_path) |
|---|---|---|---|
| Data | {{metric_name}} | {{period}} | {{vnstock_endpoint}} |
| Data | {{metric_name}} | {{period}} | {{vnstock_endpoint}} |
| Data | {{metric_name}} | {{period}} | {{vnstock_endpoint}} |
| Buffett Letter | {{chunk_id}} | {{year}} | {{section_path}} |
| Buffett Letter | {{chunk_id}} | {{year}} | {{section_path}} |

### Data Snapshot
- **{{data_endpoints_used}}**
- **{{as_of_date}}**
- **{{currency_units}}**
