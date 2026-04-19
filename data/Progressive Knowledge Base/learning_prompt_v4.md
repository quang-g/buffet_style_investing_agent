# Progressive Buffett Framework Distillation Loop — Anti-Redundancy Executor Prompt v4

## Purpose

This prompt executes a reusable Progressive Buffett Framework Distillation Loop across any pair of inputs:

1. a current Buffett master framework JSON, or a first-year extracted framework JSON, and  
2. the next year's extracted Buffett shareholder-letter framework JSON.

The goal is to produce a cleaner, more faithful, less redundant, more traceable, and more AI-agent-ready Buffett-style investing master framework.

This prompt is intentionally reusable. Do **not** assume any specific source year, item count, category count, principle count, or letter content.

---

# Role

You are a multi-role expert system combining:

1. **Warren Buffett Scholar**  
   You deeply understand Buffett's shareholder letters, recurring principles, causal reasoning, terminology, and evolution over time.

2. **Senior Knowledge Architect**  
   You maintain a cumulative knowledge base that becomes cleaner, denser, and more useful over time. You prevent shallow accumulation.

3. **AI-Agent Knowledge-Base Designer**  
   You design the framework so a future Buffett-style stock analysis agent can retrieve, reason, cite, compare, and apply the knowledge to real companies without repetitive or generic output.

4. **Investment Analyst**  
   You judge whether each retained item improves real company analysis, not merely letter summarization.

5. **Prompt and Evaluation Engineer**  
   You actively guard against regressions: duplication, category sprawl, schema drift, weak lineage, inflated metadata counts, over-merging, under-merging, vague labels, and generic investment clichés.

---

# Inputs

You will receive two JSON inputs:

```text
CURRENT_FRAMEWORK_JSON:
{{CURRENT_FRAMEWORK_JSON}}

NEXT_YEAR_EXTRACTED_JSON:
{{NEXT_YEAR_EXTRACTED_JSON}}
```

`CURRENT_FRAMEWORK_JSON` may be either:

- the first yearly extracted framework JSON, or
- an existing cumulative master framework JSON.

`NEXT_YEAR_EXTRACTED_JSON` is the newly extracted framework from the next Buffett shareholder letter.

---

# Required Output

Produce exactly one output:

> `UPDATED_MASTER_FRAMEWORK_JSON`

The output must be saved as a downloadable `.json` file containing valid JSON only.

Do **not** output the full JSON inline unless explicitly requested.

---

# System Objective

Build a continuously improving Buffett-style investing knowledge base.

This is not a concatenation task.  
This is not a yearly-summary task.  
This is not a “keep everything” task.  
This is not a “make a new category for every new theme” task.  
This is not a “compress everything into vague mega-principles” task.

The master framework should become:

- more faithful to Buffett,
- more complete only where useful,
- less redundant,
- better integrated,
- structurally cleaner,
- more traceable to source years and source item IDs,
- more useful for real stock analysis,
- more usable by a future Buffett-style AI agent,
- more resistant to bloat as more years are processed.

Each retained idea must help an analyst or AI agent evaluate at least one of:

- business quality,
- durable competitive advantage / moat,
- industry structure,
- management quality,
- capital allocation,
- accounting and earnings quality,
- owner earnings and cash generation,
- valuation,
- insurance economics,
- fixed-income or inflation risk,
- leverage and financial risk,
- long-term compounding,
- shareholder communication,
- organizational design,
- portfolio ownership discipline.

---

# Core Operating Law

## The default action is integration, not addition.

When processing an incoming item, assume it probably belongs inside the existing framework unless it clearly passes the top-level promotion gate.

Decision priority order:

1. `ADD_AS_EVIDENCE`
2. `ADD_AS_QUESTION`
3. `ADD_AS_METRIC`
4. `REINFORCE`
5. `REFINE`
6. `EXTEND`
7. `MERGE`
8. `ADD_AS_SUB_PRINCIPLE`
9. `ADD`
10. `DEFER_FOR_FUTURE_REVIEW`
11. `REJECT`

Use `ADD` only after proving the item cannot be cleanly integrated, nested, or deferred.

---

# Non-Negotiable Principles

## 1. Preserve valuable old knowledge

Do not accidentally delete valuable existing knowledge.

If an old principle is valuable but poorly structured, preserve its substance while improving its architecture.

If an old principle is weak, overloaded, or redundant, you may consolidate or restructure it, but you must preserve source lineage and explain the change in the decision/audit logs.

## 2. Learn; do not append

Every incoming item must be compared against the current master.

Never add a new top-level item merely because it appears in the new year.

Every item must be handled through one deliberate primary decision:

- `ADD`,
- `MERGE`,
- `REINFORCE`,
- `REFINE`,
- `EXTEND`,
- `ADD_AS_SUB_PRINCIPLE`,
- `ADD_AS_METRIC`,
- `ADD_AS_QUESTION`,
- `ADD_AS_EVIDENCE`,
- `REJECT`,
- `DEFER_FOR_FUTURE_REVIEW`.

## 3. One top-level principle, one primary analytical question

Each top-level master principle should answer one primary stock-analysis question.

Good examples:

- “Are reported returns real or produced by leverage/accounting distortion?”
- “Does the business preserve owner purchasing power after inflation and taxes?”
- “Is this cheap stock a bargain or a structurally poor business?”
- “Is insurance float low-cost or value-destructive?”
- “Can retained earnings be reinvested at attractive incremental returns?”

If a top-level principle answers multiple distinct analytical questions, split it or move secondary tests into sub-principles.

## 4. Prefer Buffett-specific causal logic

Do not convert Buffett's reasoning into generic investment advice.

Weak:

> Buy good companies for the long term.

Strong:

> A business is superior when it can grow earnings without requiring proportional incremental capital, because retained earnings can compound owner value instead of being consumed by maintenance capital or working-capital needs.

## 5. Trace every retained idea

Every retained incoming item must be traceable somewhere in the updated master:

- as a new top-level principle,
- inside an existing principle,
- as a sub-principle,
- as a metric,
- as an evaluation question,
- as evidence,
- or in a rejected/deferred decision record.

No incoming item may be silently ignored.

## 6. Be conservative about top-level additions

A new top-level principle is expensive. It increases retrieval complexity and future maintenance burden.

Only create one when the incoming idea is:

- Buffett-faithful,
- reusable beyond the specific year,
- analytically distinct,
- important enough for real company analysis,
- not already represented by an existing principle or sub-principle,
- not merely a metric, example, diagnostic signal, or evaluation question.

## 7. Prevent category sprawl

A new category is more expensive than a new principle.

Create a new category only when:

1. the domain is analytically distinct,
2. it is likely to hold multiple principles over time,
3. a future stock-analysis agent would intentionally retrieve that category,
4. it does not overlap heavily with existing categories,
5. it improves rather than fragments the framework.

If a category would contain only one narrow principle, prefer attaching the idea to an existing category or making it a sub-principle, unless the domain is strategically important and likely to recur.

---

# Phase 1 — Input Type Detection and Inventory

Detect whether `CURRENT_FRAMEWORK_JSON` is:

- a yearly extract, or
- an existing master framework.

Use signals such as:

- `document_type`,
- `letter_metadata`,
- `framework_items`,
- `master_framework_metadata`,
- `framework_categories`,
- `merge_decision_log`,
- `regression_test`,
- `post_merge_quality_audit`.

Internally inventory:

- source years already covered,
- incoming source year,
- existing top-level principle IDs,
- existing sub-principle IDs,
- existing categories,
- existing source item IDs,
- number of existing top-level principles,
- number of existing sub-principles,
- number of incoming framework items,
- schema version,
- current metadata fields,
- known old concepts that must not regress.

If the current file is a yearly extract, convert it into the master framework structure before merging.

---

# Phase 2 — Normalize All Incoming Knowledge Units

For every incoming item, create an internal normalized knowledge unit.

Do not output the full internal normalization unless needed in `merge_decision_log`.

For each incoming item, identify:

## 2.1 Semantic fingerprint

One sentence capturing the reusable principle independent of wording.

## 2.2 Buffett causal logic

State the cause-and-effect logic.

Examples:

- EPS can rise simply because retained capital increased; therefore EPS growth alone does not prove improved economics.
- Float is valuable only if underwriting discipline keeps its cost low or negative.
- Inflation and taxes can consume nominal compounding; therefore owner results must be judged by real after-tax purchasing power.

## 2.3 Analytical use case

What job does this item help an analyst perform?

Examples:

- evaluate ROE quality,
- test retained earnings,
- distinguish price from value,
- identify commodity economics,
- assess insurance reserving risk,
- judge premium valuation,
- evaluate management candor,
- decide buy / avoid / watch.

## 2.4 Conceptual level

Classify the item as one of:

- `CORE_PHILOSOPHY`,
- `MAJOR_PRINCIPLE`,
- `SUPPORTING_PRINCIPLE`,
- `SUB_PRINCIPLE`,
- `METRIC`,
- `EVALUATION_QUESTION`,
- `DIAGNOSTIC_SIGNAL`,
- `PRACTICAL_RULE`,
- `EXAMPLE`,
- `SOURCE_CONTEXT`,
- `NON_REUSABLE_NOISE`.

## 2.5 Domain category

Assign the cleanest Buffett-style category.

Preferred category set:

- `core_philosophy`,
- `business_quality`,
- `moat_and_competition`,
- `accounting_earnings_quality`,
- `owner_earnings_cash_generation`,
- `capital_allocation`,
- `valuation`,
- `management_quality`,
- `shareholder_communication`,
- `insurance`,
- `real_owner_returns`,
- `fixed_income_risk`,
- `risk_and_leverage`,
- `portfolio_management`,
- `organizational_design`.

Use a new category only if it passes the category creation gate in Phase 8.

## 2.6 Candidate retrieval boundary

For each item, write internally:

- `retrieves_when`: when an agent should retrieve this idea,
- `does_not_retrieve_when`: nearby situations where another principle should be retrieved instead,
- `nearest_existing_neighbors`: the 1–3 closest existing principles or sub-principles.

This is mandatory for any item being considered for `ADD` or `ADD_AS_SUB_PRINCIPLE`.

---

# Phase 3 — Build a Principle Family Map

Before merging, build an internal map of existing principle families.

For each existing top-level principle, identify:

- primary analytical question,
- causal logic,
- category,
- decision use cases,
- metrics,
- tags,
- sub-principles,
- nearest neighboring principles.

Then place every incoming item into one of these buckets:

1. existing principle family,
2. existing category but new principle family,
3. possible new category,
4. defer/reject bucket.

Important:

- If an incoming item has a nearest existing neighbor, it cannot be added as top-level until it passes the redundancy tribunal in Phase 6.
- If several incoming items belong to the same family, process them as a cluster rather than one by one to avoid creating duplicates.

---

# Phase 4 — Compare Against the Current Master

For every incoming item, compare it against existing top-level principles and sub-principles by **meaning**, not wording.

Compare across:

- normalized principle,
- causal logic,
- primary analytical question,
- evaluation questions,
- practical rules,
- metrics,
- diagnostic signals,
- tags,
- category,
- source lineage,
- recurrence history,
- retrieval boundary.

Classify the semantic relationship as exactly one:

## `SAME_CORE_IDEA`

The existing master already captures the same principle.

## `SAME_FAMILY_NEW_TEST`

The idea belongs to an existing principle family but asks a distinct analytical question.

## `REFINES_EXISTING`

The incoming item improves precision, Buffett-faithfulness, metric clarity, wording, or causal logic.

## `EXTENDS_EXISTING`

The incoming item adds a meaningful new angle, use case, risk, metric, question, or diagnostic signal.

## `EVIDENCE_FOR_EXISTING`

The incoming item mostly strengthens recurrence or evidence.

## `METRIC_FOR_EXISTING`

The incoming item mainly contributes a useful metric.

## `QUESTION_FOR_EXISTING`

The incoming item mainly contributes a useful evaluation question.

## `DISTINCT_NEW_PRINCIPLE`

The item is a distinct reusable Buffett principle.

## `DUPLICATE_NO_NEW_VALUE`

The item adds no meaningful new information.

## `TOO_SPECIFIC_OR_HISTORICAL`

The item is too company-specific, year-specific, transaction-specific, or historical without durable analytical reuse.

## `CONTRADICTION_OR_QUALIFIER`

The item conflicts with, limits, or qualifies an existing principle.

---

# Phase 5 — Choose One Primary Update Action Per Incoming Item

Each incoming item must receive exactly one primary action:

- `ADD`,
- `MERGE`,
- `REINFORCE`,
- `REFINE`,
- `EXTEND`,
- `ADD_AS_SUB_PRINCIPLE`,
- `ADD_AS_METRIC`,
- `ADD_AS_QUESTION`,
- `ADD_AS_EVIDENCE`,
- `REJECT`,
- `DEFER_FOR_FUTURE_REVIEW`.

You may record secondary effects in notes, but the count summary must use the primary action only.

## 5.1 Use `ADD` only when all conditions are true

- The item is a distinct reusable principle.
- It answers a different primary analytical question from existing principles.
- Merging would overload an existing principle.
- Nesting would make it hard for an agent to retrieve.
- It is not merely a metric, question, example, source context, or evidence.
- It improves real business or stock analysis.
- It passes the Top-Level Promotion Gate.
- It passes the Redundancy Tribunal in Phase 6.
- It passes the Category Governance Gate in Phase 8 if a new category is involved.

## 5.2 Top-Level Promotion Gate

Before using `ADD`, ask:

1. Does this idea answer a unique primary analytical question?
2. Would an analyst retrieve it directly during real stock analysis?
3. Is it broad enough to recur across companies, years, or industries?
4. Is it more than a metric, checklist item, example, or warning signal?
5. Does it avoid duplicating an existing principle or sub-principle?
6. Will adding it improve the framework more than nesting it?
7. Would keeping it separate reduce ambiguity for a future agent?
8. Would a future evaluator score non-redundancy at least 4/5 after this addition?

If any answer is no, do **not** use `ADD`.

Use `ADD_AS_SUB_PRINCIPLE`, `EXTEND`, `ADD_AS_METRIC`, `ADD_AS_QUESTION`, `ADD_AS_EVIDENCE`, `REJECT`, or `DEFER_FOR_FUTURE_REVIEW` instead.

## 5.3 Use `MERGE` when

- the incoming item expresses the same principle,
- no distinct analytical test would be lost,
- the existing principle can absorb it without becoming bloated,
- source lineage and recurrence can be updated cleanly.

## 5.4 Use `REINFORCE` when

- the item mainly confirms an existing principle,
- only source lineage, recurrence, and small supporting details should be added.

## 5.5 Use `REFINE` when

- the item improves wording,
- clarifies causal logic,
- improves Buffett-faithfulness,
- sharpens metrics or limitations,
- corrects vague or generic existing language.

## 5.6 Use `EXTEND` when

- the item adds a new but closely related angle,
- the extension improves usefulness,
- the extension does not create multiple unrelated analytical tests inside one principle.

## 5.7 Use `ADD_AS_SUB_PRINCIPLE` when

- the item belongs under an existing principle family,
- it has a distinct analytical question,
- it should remain separately retrievable,
- it is too narrow or specialized to deserve top-level status,
- the parent principle remains concise after nesting.

## 5.8 Use `ADD_AS_METRIC`, `ADD_AS_QUESTION`, or `ADD_AS_EVIDENCE` when

- the item adds useful content but not a principle.

## 5.9 Use `REJECT` when

- the item is duplicate,
- weakly reusable,
- generic,
- too historical,
- too narrow,
- already represented in stronger form,
- not useful for stock analysis,
- or not faithful to Buffett's causal reasoning.

## 5.10 Use `DEFER_FOR_FUTURE_REVIEW` when

- the item may become important if it recurs later,
- evidence is currently too thin to promote,
- retaining it now would bloat the framework,
- but it should not be forgotten entirely.

Deferred items must be recorded in `merge_decision_log` and optionally in `post_merge_quality_audit.deferred_items`.

---

# Phase 6 — Redundancy Tribunal

This phase is mandatory before finalizing any `ADD` or `ADD_AS_SUB_PRINCIPLE` decision.

For each candidate new top-level principle or sub-principle, compare it against its nearest existing neighbors.

## 6.1 Similarity dimensions

Score each dimension internally from 0 to 3:

- primary analytical question similarity,
- causal logic similarity,
- metric similarity,
- evaluation question similarity,
- practical rule similarity,
- decision use case similarity,
- retrieval alias similarity,
- source item overlap.

Then classify the candidate:

- `CLEARLY_DISTINCT`,
- `DISTINCT_BUT_RELATED`,
- `OVERLAPPING_NEEDS_BOUNDARY`,
- `DUPLICATE_SHOULD_MERGE`,
- `TOO_NARROW_SHOULD_NEST`,
- `TOO_WEAK_SHOULD_DEFER_OR_REJECT`.

## 6.2 Required boundary statement

If the candidate remains separate, write a short boundary note:

```text
Retrieve this principle when: ...
Do not retrieve it when: ...
Nearest neighbor distinction: ...
```

Store this in:

- the principle or sub-principle as `retrieval_boundary`, or
- `post_merge_quality_audit.justified_overlaps_remaining`.

## 6.3 Mandatory consolidation rule

If two principles have the same primary analytical question, they must not remain separate.

Choose one:

1. consolidate into one principle,
2. make one a sub-principle,
3. deprecate the weaker duplicate while preserving lineage.

## 6.4 Mandatory nesting rule

If two principles have the same causal logic but one is narrower, the narrower one should normally become a sub-principle, not a top-level principle.

## 6.5 Mandatory justification rule

If overlap remains, justify it explicitly. Unjustified overlap is a major issue.

---

# Phase 7 — Merge Discipline and Non-Redundancy Control

Before finalizing, run a redundancy check across the full updated framework.

Identify near-duplicate clusters where two or more principles share:

- similar primary analytical questions,
- similar causal logic,
- similar practical rules,
- overlapping metrics,
- overlapping tags,
- same source items,
- same incoming item family,
- same decision use cases,
- similar retrieval aliases.

For each cluster, choose one:

1. **Consolidate** into one principle.
2. **Make one a parent and the other a sub-principle.**
3. **Keep separate but explicitly justify the distinction.**
4. **Deprecate the weaker duplicate** while preserving source lineage and merge history.

## Required distinction test

Two principles may remain separate only if they answer meaningfully different analyst questions.

Example distinction:

- “Do reported earnings exceed the capital required to produce them?”  
  belongs to capital efficiency / earnings quality.

- “Are high reported returns real or produced by leverage/accounting distortion?”  
  belongs to leverage-adjusted earnings quality.

- “Can the company compound without relying on debt or external capital?”  
  belongs to risk and leverage / durability.

These are related but not identical. Keep boundaries clear.

Record remaining duplicates or justified overlaps in `post_merge_quality_audit`.

If `deduplication_actions_performed` equals `0`, explain why no deduplication was needed. Do not leave this as an unexplained zero after many additions.

---

# Phase 8 — Category and Taxonomy Governance

Before finalizing categories, check whether each category represents one coherent domain.

## 8.1 Category rules

A category should not mix unrelated domains merely because they appeared in the same letter.

Suggested category boundaries:

- `core_philosophy`: owner mindset, long-term orientation, business-first thinking.
- `business_quality`: economic durability, capital intensity, pricing power, business resilience.
- `moat_and_competition`: durable competitive advantage, franchise strength, customer captivity, competitive structure.
- `accounting_earnings_quality`: accounting versus economics, earnings quality, ROE validity, denominator quality.
- `owner_earnings_cash_generation`: cash earnings, owner earnings, free cash flow, working-capital absorption.
- `capital_allocation`: retention, reinvestment, acquisitions, selling discipline, repurchases, redeployment.
- `valuation`: intrinsic value, margin of safety, quality versus cheapness, private-market value, premium valuation.
- `management_quality`: integrity, rationality, candor, incentives, competence, owner orientation.
- `shareholder_communication`: reporting to owners, expectation setting, shareholder base quality.
- `insurance`: underwriting, reserves, float, claims inflation, long-tail risk, premium volume, insurance cycles.
- `real_owner_returns`: inflation, tax drag, real purchasing power, owner purchasing-power outcomes.
- `fixed_income_risk`: duration, nominal claims, bond accounting, optionality, inflation exposure.
- `risk_and_leverage`: excessive leverage, balance-sheet fragility, financing risk, low-leverage durability.
- `portfolio_management`: holding discipline, turnover, market timing avoidance, ownership of exceptional businesses.
- `organizational_design`: decentralization, autonomy, headquarters design, operating-control philosophy.

## 8.2 New category creation gate

Before creating a new category, ask:

1. Is this category analytically distinct?
2. Would a stock-analysis agent intentionally retrieve it?
3. Is it likely to hold multiple principles over time?
4. Does it avoid overlap with existing categories?
5. Does it reduce confusion rather than increase fragmentation?
6. Would adding this category improve non-redundancy rather than weaken it?
7. Can the incoming idea be better represented as a principle, sub-principle, tag, or decision use case inside an existing category?

If the answer to any question is no, attach the item to an existing category or make it a sub-principle.

## 8.3 One-item category warning

If a category has only one top-level principle after the update, add a note in `post_merge_quality_audit.category_concerns` explaining why the category is justified.

If it is not justified, merge it into a broader category before finalizing.

## 8.4 Category growth pressure test

After the update, compare category count before and after.

If category count increases, add `category_change_log` explaining:

- categories before,
- categories after,
- new categories added,
- why each new category could not be represented inside an existing category,
- whether each new category is expected to contain multiple principles over future years.

If category count increases by more than one in a single update, run a second category-consolidation pass before finalizing.

---

# Phase 9 — Anti-Bloat and Overload Test

A top-level principle is overloaded if it contains two or more separable analytical tests.

A top-level principle should not become a “kitchen sink.”

## Split or create sub-principles when one principle combines distinct ideas such as:

- EPS growth versus return on capital,
- high ROE without leverage,
- accounting gimmick detection,
- low-leverage compounding,
- real after-tax purchasing power,
- inflation-plus-tax hurdle rate,
- fixed-income duration risk,
- bond accounting marks,
- underwriting discipline,
- float economics,
- long-tail reserve risk,
- management incentives,
- decentralization trade-offs,
- valuation discipline,
- quality-over-cheapness,
- premium valuation for exceptional businesses.

## Rule

If one principle now contains multiple distinct analytical questions:

- keep a concise parent principle,
- move separable tests into `sub_principles`,
- or split into separate top-level principles only if each deserves direct retrieval.

Every top-level principle must have exactly one `primary_analytical_question`.

Every sub-principle must also have exactly one `primary_analytical_question`.

## Top-level growth pressure test

After preliminary merging, calculate:

```text
top_level_growth = final_top_level_principles - starting_top_level_principles
```

If top-level principles grow by more than the smaller of:

- 15% of starting top-level principles, or
- 5 top-level principles,

run a second consolidation pass.

This is not a hard cap. It is a forcing function to prevent shallow accumulation. If growth remains high after the second pass, justify each addition in `post_merge_quality_audit.top_level_growth_justification`.

---

# Phase 10 — Source Lineage and Evidence Integrity

Never invent source evidence.

For every incoming item, preserve in `merge_decision_log`:

- `incoming_year`,
- `incoming_item_id`,
- `incoming_title`,
- `decision`,
- `matched_master_principle_id`, if applicable,
- `matched_sub_principle_id`, if applicable,
- `source_preserved`,
- `reason`.

For every master principle, maintain:

- `first_seen_year`,
- `reinforced_by_years`,
- `updated_from_years`,
- `source_item_ids`,
- `source_refs`,
- `source_evidence`.

For each `source_evidence` entry include, when available:

- `year`,
- `source_document_type`,
- `source_item_id`,
- `principle_title_at_source`,
- `source_file`,
- `source_chunk_id`,
- `source_excerpt`.

If `source_chunk_id` or `source_excerpt` is unavailable, use `null` or omit the field.

Do not fabricate quotes, chunks, filenames, or source excerpts.

## 10.1 Sub-principle lineage

Every sub-principle must have its own source lineage:

- `first_seen_year`,
- `source_item_ids`,
- `source_evidence`.

Do not rely only on the parent principle's lineage.

## 10.2 Parent roll-up lineage

If an incoming item is added as a sub-principle, the parent principle's lineage should record that the principle family was updated, but recurrence counting must avoid double-counting the same source item.

## 10.3 Source counting rule

When computing recurrence, count each unique source item only once per principle family.

Do not double-count the same source item because it appears in both parent and sub-principle lineage.

## 10.4 Evidence-grade traceability audit

In `post_merge_quality_audit.source_lineage_concerns`, report:

- number of source evidence entries with exact excerpts,
- number of source evidence entries with null excerpts,
- number of source evidence entries with chunk IDs,
- number of source evidence entries with null chunk IDs,
- whether the limitation is due to unavailable input data.

Do not penalize the framework for missing excerpts if the input does not provide them, but record the limitation honestly.

---

# Phase 11 — Metric, Question, and Tag Normalization

Deduplicate aggressively by meaning, not only by exact string match.

For each principle and sub-principle:

- `financial_metrics` must contain unique metric names.
- `metric_definition` must contain at most one definition per metric.
- `metric_usage_context` must not repeat the same use case.
- `metric_limitations` must not repeat the same limitation.
- `evaluation_questions` must be unique by meaning.
- `diagnostic_signals` must be unique by meaning.
- `practical_rules` must be unique by meaning.
- `example_buffett_phrasing` must not contain invented Buffett quotes.
- `tags` must be normalized and unique.
- `retrieval_aliases` must be short, useful, and unique.
- `decision_use_cases` must be relevant and not overstuffed.

## 11.1 Synonym normalization

Standardize equivalent metric names unless a meaningful distinction exists.

Examples:

- Use one of `beginning_shareholder_equity` or `beginning_shareholders_equity`.
- Use one of `underwriting_profit` or `underwriting_profitability`.
- Use one of `return_on_capital` or `return_on_capital_employed` unless the framework defines a difference.
- Use one of `owner_earnings` or `look_through_owner_earnings` only if they mean the same thing in context.

## 11.2 Preferred metric structure

If changing schema is allowed, prefer structured metric objects:

```json
"metrics": [
  {
    "metric_name": "",
    "definition": "",
    "usage_context": [],
    "limitations": []
  }
]
```

If preserving the existing array-based schema, keep arrays clean and deduplicated.

Do not include both duplicated metric arrays and duplicated metric objects.

## 11.3 Duplicated wording cleanup

If a merged principle contains repeated definitions, questions, or practical rules from different source years, consolidate them into one stronger version while preserving all source lineage.

Do not let source preservation become text duplication.

---

# Phase 12 — Retrieval and AI-Agent Usefulness Layer

The master framework must be useful for retrieval and reasoning by a future Buffett-style stock analysis agent.

For every principle and sub-principle, include:

- `primary_analytical_question`,
- `agent_usefulness`,
- `retrieval_aliases`,
- `decision_use_cases`,
- `retrieval_boundary`.

## 12.1 Retrieval aliases

Add 3–8 short aliases that an AI agent might use to retrieve the principle.

Examples:

- `quality_over_cheapness`,
- `high_roe_without_leverage`,
- `real_after_tax_return`,
- `owner_purchasing_power`,
- `underwriting_discipline`,
- `long_tail_reserve_risk`,
- `look_through_earnings`,
- `retained_earnings_test`.

Do not add long sentence-like aliases.

## 12.2 Decision use cases

Add practical use cases such as:

- `buy_decision`,
- `avoid_decision`,
- `watchlist_decision`,
- `valuation_check`,
- `management_assessment`,
- `moat_assessment`,
- `earnings_quality_review`,
- `capital_allocation_review`,
- `risk_review`,
- `insurance_analysis`,
- `portfolio_review`.

Do not overstuff. Only include genuinely relevant use cases.

## 12.3 Retrieval boundary

For every principle and sub-principle, include:

```json
"retrieval_boundary": {
  "retrieves_when": "",
  "does_not_retrieve_when": "",
  "nearest_neighbor_principles": []
}
```

The boundary may be concise. Its purpose is to reduce duplicate retrieval and repetitive agent analysis.

---

# Phase 13 — Preservation and Regression Test

Before final output, run a regression test against core Buffett concepts.

For each concept, mark:

- `PASS`,
- `PARTIAL`,
- `FAIL`,
- `NOT_YET_SUPPORTED_BY_SOURCE_YEARS`.

Core concepts to check:

1. Think like a business owner
2. Durable competitive advantage / moat
3. High return on equity with little debt
4. Owner earnings
5. Economic goodwill
6. Capital allocation discipline
7. Retained earnings test
8. Margin of safety
9. Circle of competence
10. Management integrity and rationality
11. Avoid commodity-like businesses
12. Float as a source of low-cost funding
13. Look-through earnings
14. Long-term compounding
15. Intrinsic value versus accounting value
16. Avoiding excessive leverage
17. Share repurchases only below intrinsic value
18. Business quality over market timing
19. Cost of growth / incremental capital needs
20. Earnings quality and accounting conservatism

Important:

- Do not invent missing concepts.
- Do not force concepts not yet supported by the processed years.
- If a concept is absent because it has not appeared yet, mark `NOT_YET_SUPPORTED_BY_SOURCE_YEARS`.
- If an old supported concept disappears, mark `FAIL` and fix before finalizing unless the concept was intentionally consolidated with clear lineage.

Include this test in `regression_test` and summarize serious concerns in `post_merge_quality_audit`.

---

# Phase 14 — Metadata and Count Consistency

The `update_summary` must match actual decisions.

Compute counts carefully.

Required count definitions:

- `items_before_update`: number of top-level master principles before update.
- `incoming_items`: number of incoming framework items.
- `items_added`: incoming items whose primary decision is `ADD`.
- `items_merged`: incoming items whose primary decision is `MERGE`.
- `items_reinforced`: incoming items whose primary decision is `REINFORCE`.
- `items_refined`: incoming items whose primary decision is `REFINE`.
- `items_extended`: incoming items whose primary decision is `EXTEND`.
- `items_added_as_sub_principles`: incoming items whose primary decision is `ADD_AS_SUB_PRINCIPLE`.
- `items_added_as_metrics`: incoming items whose primary decision is `ADD_AS_METRIC`.
- `items_added_as_questions`: incoming items whose primary decision is `ADD_AS_QUESTION`.
- `items_added_as_evidence`: incoming items whose primary decision is `ADD_AS_EVIDENCE`.
- `items_rejected`: incoming items whose primary decision is `REJECT`.
- `items_deferred`: incoming items whose primary decision is `DEFER_FOR_FUTURE_REVIEW`.
- `items_integrated_into_existing`: count of incoming items whose primary decision is one of `MERGE`, `REINFORCE`, `REFINE`, `EXTEND`, `ADD_AS_SUB_PRINCIPLE`, `ADD_AS_METRIC`, `ADD_AS_QUESTION`, or `ADD_AS_EVIDENCE`.
- `items_after_update`: final number of top-level master principles, excluding sub-principles.
- `sub_principles_after_update`: final number of sub-principles.
- `principles_split_due_to_overload`: number of split actions.
- `principles_consolidated_due_to_redundancy`: number of consolidation actions.
- `deduplication_actions_performed`: count of principles/sub-principles where deduplication changed content.
- `category_count_before`: number of categories before update.
- `category_count_after`: number of categories after update.
- `categories_added`: categories created in this update.
- `categories_removed_or_consolidated`: categories removed or consolidated in this update.

Rules:

- `incoming_items` must equal the number of entries in `merge_decision_log`.
- The sum of all primary decision counts must equal `incoming_items`.
- `items_integrated_into_existing` must equal the sum of all non-ADD, non-REJECT, non-DEFER integration decisions.
- `items_after_update` must equal the actual number of top-level principles in `framework_categories`.
- `sub_principles_after_update` must equal the actual number of sub-principles.
- `total_weighted_score` must be on a 1–5 scale, not a percentage or raw sum.
- Do not claim `deduplication_actions_performed: 0` after large additions unless the audit explains why no deduplication was necessary.

---

# Phase 15 — Final Self-Evaluation Gate

Before writing the final JSON, score the updated framework internally.

Use this exact weighted scorecard:

| Dimension | Score 1–5 | Weight |
|---|---:|---:|
| Faithfulness to Buffett |  | 15% |
| New-year coverage |  | 10% |
| Preservation of old knowledge |  | 15% |
| Merge and distillation quality |  | 15% |
| Non-redundancy |  | 10% |
| Stock-analysis reusability |  | 10% |
| Source lineage and traceability |  | 10% |
| Structural/schema quality |  | 5% |
| Conceptual hierarchy |  | 5% |
| Downstream AI-agent usefulness |  | 5% |

Calculate:

```text
total_weighted_score = weighted average on 1–5 scale
```

## Acceptance gate

Finalize only if all are true:

- total weighted score is at least `4.2`,
- faithfulness score is at least `4`,
- preservation score is at least `4`,
- merge quality score is at least `4`,
- non-redundancy score is at least `4`,
- no blocker issues remain,
- no supported old core Buffett concept regressed to `FAIL`,
- every top-level addition has a recorded redundancy-tribunal boundary,
- every new category has a category-governance justification.

If any dimension scores below `4`, revise the JSON before finalizing.

If a score remains below `4` only because source excerpts or chunk IDs are unavailable, record the limitation clearly in `post_merge_quality_audit.source_lineage_concerns`.

Do not inflate the self-score. A low score is acceptable only if the issue is explicitly recorded and the output still follows the requested constraints.

---

# Phase 16 — Recommended Master Schema

Use schema version `2.5` unless strict backward compatibility requires preserving the current schema version.

The final JSON should follow this structure:

```json
{
  "schema_version": "2.5",
  "document_type": "buffett_master_framework",
  "master_framework_metadata": {
    "framework_name": "Buffett Master Framework",
    "framework_description": "Continuously updated master investing framework distilled from Warren Buffett shareholder letters.",
    "source_letters_covered": [],
    "latest_update_year": 0,
    "version_label": "",
    "update_summary": {
      "starting_framework_type": "initial_year_extract|existing_master",
      "items_before_update": 0,
      "incoming_items": 0,
      "items_added": 0,
      "items_merged": 0,
      "items_reinforced": 0,
      "items_refined": 0,
      "items_extended": 0,
      "items_added_as_sub_principles": 0,
      "items_added_as_metrics": 0,
      "items_added_as_questions": 0,
      "items_added_as_evidence": 0,
      "items_rejected": 0,
      "items_deferred": 0,
      "items_integrated_into_existing": 0,
      "items_after_update": 0,
      "sub_principles_after_update": 0,
      "principles_split_due_to_overload": 0,
      "principles_consolidated_due_to_redundancy": 0,
      "deduplication_actions_performed": 0,
      "category_count_before": 0,
      "category_count_after": 0,
      "categories_added": [],
      "categories_removed_or_consolidated": []
    }
  },
  "framework_categories": [
    {
      "category": "",
      "category_description": "",
      "principles": [
        {
          "master_principle_id": "",
          "principle_title": "",
          "principle_statement": "",
          "normalized_principle": "",
          "category": "",
          "conceptual_level": "CORE_PHILOSOPHY|MAJOR_PRINCIPLE|SUPPORTING_PRINCIPLE",
          "parent_principle_id": null,
          "importance": "high|medium|low",
          "time_horizon": "single_year|medium_term|long_term|cross_cycle",
          "applies_to": [],
          "primary_analytical_question": "",
          "retrieval_boundary": {
            "retrieves_when": "",
            "does_not_retrieve_when": "",
            "nearest_neighbor_principles": []
          },
          "performance_logic": [],
          "financial_metrics": [],
          "metric_definition": [],
          "metric_usage_context": [],
          "metric_limitations": [],
          "evaluation_questions": [],
          "diagnostic_signals": [],
          "practical_rules": [],
          "example_buffett_phrasing": [],
          "retrieval_aliases": [],
          "decision_use_cases": [],
          "tags": [],
          "sub_principles": [
            {
              "sub_principle_id": "",
              "sub_principle_title": "",
              "sub_principle_statement": "",
              "normalized_sub_principle": "",
              "primary_analytical_question": "",
              "retrieval_boundary": {
                "retrieves_when": "",
                "does_not_retrieve_when": "",
                "nearest_neighbor_principles": []
              },
              "performance_logic": [],
              "financial_metrics": [],
              "metric_definition": [],
              "metric_usage_context": [],
              "metric_limitations": [],
              "evaluation_questions": [],
              "diagnostic_signals": [],
              "practical_rules": [],
              "retrieval_aliases": [],
              "decision_use_cases": [],
              "tags": [],
              "source_lineage": {
                "first_seen_year": 0,
                "source_item_ids": [],
                "source_evidence": []
              }
            }
          ],
          "source_lineage": {
            "first_seen_year": 0,
            "reinforced_by_years": [],
            "updated_from_years": [],
            "source_item_ids": [],
            "source_refs": [],
            "source_evidence": [
              {
                "year": 0,
                "source_document_type": "year_extract|master_framework",
                "source_item_id": "",
                "principle_title_at_source": "",
                "source_file": null,
                "source_chunk_id": null,
                "source_excerpt": null
              }
            ]
          },
          "synthesis_state": {
            "recurrence_count": 1,
            "confidence": "high|medium|low",
            "generality": "high|medium|low",
            "agent_usefulness": "high|medium|low",
            "merge_history": [
              {
                "incoming_year": 0,
                "incoming_item_id": "",
                "action": "added|merged|reinforced|refined|extended|added_as_sub_principle|added_as_metric|added_as_question|added_as_evidence|rejected|deferred",
                "notes": ""
              }
            ]
          }
        }
      ]
    }
  ],
  "merge_decision_log": [
    {
      "incoming_year": 0,
      "incoming_item_id": "",
      "incoming_title": "",
      "semantic_fingerprint": "",
      "buffett_causal_logic": "",
      "analytical_use_case": "",
      "conceptual_level": "",
      "matched_master_principle_id": null,
      "matched_sub_principle_id": null,
      "relationship_classification": "",
      "decision": "ADD|MERGE|REINFORCE|REFINE|EXTEND|ADD_AS_SUB_PRINCIPLE|ADD_AS_METRIC|ADD_AS_QUESTION|ADD_AS_EVIDENCE|REJECT|DEFER_FOR_FUTURE_REVIEW",
      "redundancy_tribunal_result": "CLEARLY_DISTINCT|DISTINCT_BUT_RELATED|OVERLAPPING_NEEDS_BOUNDARY|DUPLICATE_SHOULD_MERGE|TOO_NARROW_SHOULD_NEST|TOO_WEAK_SHOULD_DEFER_OR_REJECT|null",
      "retrieval_boundary_summary": "",
      "reason": "",
      "source_preserved": true
    }
  ],
  "regression_test": [
    {
      "core_buffett_concept": "",
      "status": "PASS|PARTIAL|FAIL|NOT_YET_SUPPORTED_BY_SOURCE_YEARS",
      "evidence_in_updated_json": [],
      "notes": ""
    }
  ],
  "post_merge_quality_audit": {
    "old_principles_preserved": true,
    "old_principles_removed_or_consolidated": [],
    "incoming_items_handled_count": 0,
    "incoming_items_missing": [],
    "deferred_items": [],
    "rejected_items": [],
    "duplicate_principles_remaining": [],
    "justified_overlaps_remaining": [],
    "overloaded_principles_remaining": [],
    "redundancy_tribunal_notes": [],
    "top_level_growth_justification": [],
    "category_change_log": [],
    "category_concerns": [],
    "one_item_category_justifications": [],
    "metric_deduplication_notes": [],
    "source_lineage_concerns": [],
    "schema_consistency_notes": [],
    "metadata_count_check": {
      "counts_match_decision_log": true,
      "sum_of_decision_counts_equals_incoming_items": true,
      "items_integrated_into_existing_matches_decision_log": true,
      "items_after_update_matches_actual_top_level_principles": true,
      "sub_principles_after_update_matches_actual_sub_principles": true,
      "category_counts_match_actual_categories": true,
      "notes": ""
    },
    "agent_usability_notes": [],
    "blocker_issues": [],
    "major_issues": [],
    "minor_issues": [],
    "self_evaluation_scorecard": {
      "faithfulness_to_buffett": 0,
      "new_year_coverage": 0,
      "preservation_of_old_knowledge": 0,
      "merge_and_distillation_quality": 0,
      "non_redundancy": 0,
      "stock_analysis_reusability": 0,
      "source_lineage_and_traceability": 0,
      "structural_schema_quality": 0,
      "conceptual_hierarchy": 0,
      "downstream_ai_agent_usefulness": 0,
      "total_weighted_score": 0
    },
    "acceptance_gate_result": "ACCEPT|REVISE_NEEDED_BUT_OUTPUT_PROVIDED|REJECT_QUALITY_FAILURE"
  }
}
```

---

# Schema Compatibility Rule

If the current master uses an older schema and strict compatibility is required, you may preserve the existing schema version.

However, you must still apply:

- anti-redundancy rules,
- redundancy tribunal,
- anti-bloat rules,
- top-level promotion gate,
- category governance,
- source lineage rules,
- decision log completeness,
- metadata consistency,
- conceptual hierarchy,
- regression test,
- AI-agent retrieval fields where possible.

Do not preserve schema simplicity at the cost of knowledge quality.

---

# Final Validation Checklist

Before saving the output file, verify:

1. The JSON is valid and parseable.
2. Every incoming item appears exactly once in `merge_decision_log`.
3. Every retained incoming item is traceable in source lineage.
4. Every rejected or deferred item has a clear reason.
5. No valuable old principle was accidentally deleted.
6. No old supported Buffett concept regressed to `FAIL`.
7. No top-level principle has multiple unrelated primary analytical questions.
8. Any one-item category is justified or merged.
9. Near-duplicates were consolidated, nested, or explicitly justified.
10. Every top-level addition passed the redundancy tribunal.
11. Every new category passed the category creation gate.
12. Every principle and sub-principle has a retrieval boundary.
13. Metrics, definitions, questions, rules, signals, tags, aliases, and evidence are deduplicated.
14. `items_after_update` equals actual top-level principle count.
15. `sub_principles_after_update` equals actual sub-principle count.
16. Decision counts sum to `incoming_items`.
17. `items_integrated_into_existing` matches the decision log.
18. Category counts match actual categories.
19. `total_weighted_score` is on a 1–5 scale.
20. `post_merge_quality_audit` honestly records remaining concerns.
21. The downloadable file link is valid.

---

# Final Output Rules

1. Return the full updated master framework JSON as a downloadable file.
2. Do not return only changed principles.
3. Preserve all valuable old principles.
4. Every incoming item must appear exactly once in `merge_decision_log`.
5. Every retained incoming item must be traceable in source lineage.
6. Do not invent missing evidence, quotes, chunk IDs, or source files.
7. Deduplicate principles, metrics, definitions, questions, rules, signals, tags, aliases, and evidence.
8. Use sub-principles to avoid both shallow duplication and overloaded master principles.
9. Keep categories clean and retrieval-friendly.
10. Validate update-summary counts against actual decisions.
11. Include `post_merge_quality_audit`.
12. Include `regression_test`.
13. Include redundancy tribunal results.
14. Include category governance notes if categories changed.
15. Do not output explanations outside the downloadable file unless explicitly asked.
16. Make sure the download link is valid.

---

# Final Instruction

Now execute the full Progressive Framework Distillation Loop using:

- `CURRENT_FRAMEWORK_JSON`
- `NEXT_YEAR_EXTRACTED_JSON`

Return exactly one downloadable file:

`UPDATED_MASTER_FRAMEWORK_JSON`

The file must contain valid JSON only.
