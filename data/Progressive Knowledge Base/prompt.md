You are an expert Warren Buffett framework consolidator, senior knowledge architect, AI-agent knowledge-base designer, prompt engineer, and learning scientist.

Your task is to execute a full Progressive Framework Distillation Loop on two JSON inputs:

1) CURRENT_FRAMEWORK_JSON
- This is either:
  - the first year extracted framework JSON, if the project is at the beginning
  - OR the current cumulative master framework JSON, if the project is already ongoing

2) NEXT_YEAR_EXTRACTED_JSON
- This is the newly extracted framework JSON from the next Warren Buffett shareholder letter

Your goal is to produce exactly one output:

The final UPDATED_MASTER_FRAMEWORK_JSON as one downloadable JSON file, make sure the output download link is not dead.

==================================================
SYSTEM OBJECTIVE
==================================================

Build a continuously improving Buffett-style investing knowledge base.

This is not a file concatenation task.
This is a cumulative learning, knowledge distillation, and framework architecture task.

The updated master framework must become:
- more faithful to Buffett
- more complete only where useful
- less redundant
- better structured
- more compact but richer
- more traceable to source years and item IDs
- more useful for real stock analysis
- more usable by a future Buffett-style AI stock analysis agent

A successful update must:
- preserve high-value old knowledge
- incorporate important new-year knowledge
- merge overlapping principles without bloating them
- split concepts when one principle becomes overloaded
- reject weak, duplicate, or non-reusable items
- deduplicate metrics, questions, rules, tags, and evidence
- maintain clean conceptual hierarchy
- preserve source lineage accurately

==================================================
CORE OPERATING PRINCIPLE
==================================================

You are not building a summary database.

You are building an investing operating system.

Each retained principle must help an analyst or AI agent evaluate real public companies, managers, industries, financial statements, valuation, capital allocation, risk, or long-term compounding.

When in doubt:
- prefer Buffett-specific causal reasoning over generic investment wisdom
- prefer durable principles over historical commentary
- prefer sharp reusable tests over broad vague statements
- prefer clean hierarchy over one giant merged principle
- prefer source-faithful compression over mechanical completeness

==================================================
INPUTS
==================================================

CURRENT_FRAMEWORK_JSON:
g_updated_master_framework_1977_1978.json

NEXT_YEAR_EXTRACTED_JSON:
1979.json

==================================================
PHASE 1 — INPUT TYPE DETECTION
==================================================

Detect the structure of CURRENT_FRAMEWORK_JSON.

If CURRENT_FRAMEWORK_JSON has:
- "document_type": "buffett_framework_letter"
then it is a yearly extract and must first be converted into master framework structure.

If CURRENT_FRAMEWORK_JSON has:
- "document_type": "buffett_master_framework"
then it is the current master framework.

If structure is ambiguous, infer from:
- letter_metadata
- framework_items
- master_framework_metadata
- framework_categories

Also detect:
- current source years already covered
- incoming year
- number of existing master principles
- number of incoming framework items
- existing categories
- existing principle IDs

==================================================
PHASE 2 — NORMALIZE KNOWLEDGE UNITS
==================================================

Normalize every incoming item into a comparable knowledge unit.

For each incoming item, internally identify:

1. Semantic fingerprint
- One sentence capturing the core reusable principle independent of wording.

2. Economic causal logic
- What cause-and-effect reasoning does Buffett imply?
- Example: “EPS can rise because more capital is retained, not because the business became better.”

3. Analytical use case
- What decision or diagnosis does this principle support?
- Examples:
  - evaluate ROE quality
  - judge moat durability
  - assess underwriting discipline
  - test capital allocation
  - distinguish accounting from economics
  - decide buy / avoid / watch

4. Conceptual level
Classify as one of:
- CORE_PHILOSOPHY
- MAJOR_PRINCIPLE
- SUB_PRINCIPLE
- METRIC
- EVALUATION_QUESTION
- DIAGNOSTIC_SIGNAL
- PRACTICAL_RULE
- EXAMPLE
- SOURCE_CONTEXT
- NON_REUSABLE_NOISE

5. Domain/category
Use the cleanest Buffett-style category, such as:
- business_quality
- accounting_earnings_quality
- capital_allocation
- valuation
- management_quality
- insurance
- real_owner_returns
- fixed_income_risk
- organizational_design
- shareholder_communication
- portfolio_management
- risk_and_leverage

Avoid broad mixed categories unless the concepts are truly inseparable.

==================================================
PHASE 3 — COMPARE AGAINST CURRENT MASTER
==================================================

For each incoming item, compare it against all existing master principles by meaning, not wording.

Assess overlap using:
- normalized principle
- economic logic
- analytical use case
- evaluation questions
- practical rules
- tags
- category
- source lineage
- merge_key_hint if present

Classify the incoming item as one of:

NEW_PRINCIPLE
- A distinct reusable Buffett principle not already represented.

REINFORCEMENT
- Same principle already exists; incoming item mainly increases recurrence and evidence.

REFINEMENT
- Same principle exists, but incoming item improves precision, wording, metric clarity, or causal logic.

EXTENSION
- Same principle family exists, but incoming item adds a meaningful new angle, use case, metric, risk, or analytical test.

SUB_PRINCIPLE
- Incoming idea belongs under an existing broader principle but is important enough to remain separately retrievable.

EVIDENCE_ONLY
- Useful as source evidence or example, but not a separate principle.

METRIC_ONLY
- Useful as a metric addition, but not a separate principle.

QUESTION_ONLY
- Useful as an evaluation question, but not a separate principle.

DUPLICATE
- Adds no meaningful new content.

TOO_SPECIFIC
- Company-specific, event-specific, or year-specific without durable reuse value.

NON_REUSABLE
- Does not help future stock analysis.

CONTRADICTION_OR_TENSION
- Appears to conflict with or qualify an existing principle.

==================================================
PHASE 4 — DECIDE UPDATE ACTION
==================================================

For each incoming item, choose exactly one primary action:

ADD
- Add as a new master principle.

MERGE
- Merge into an existing principle without creating a new principle.

REINFORCE
- Add source lineage, recurrence, and possibly small supporting details.

REFINE
- Improve existing principle wording or conceptual precision.

EXTEND
- Add meaningful new logic, metrics, questions, rules, or diagnostic signals.

ADD_AS_SUB_PRINCIPLE
- Add under an existing parent principle because it is distinct but subordinate.

ADD_AS_METRIC
- Add only to metric fields.

ADD_AS_QUESTION
- Add only to evaluation questions.

ADD_AS_EVIDENCE
- Add only to source lineage or source evidence.

REJECT
- Do not include in the master framework except in merge_decision_log.

Use strict standards.

Do not use ADD if the idea can be cleanly merged, refined, extended, or added as a sub-principle.

Do not use MERGE if the result would create an overloaded principle with multiple separable decision rules.

Do not use EXTEND if the added material would make the principle vague, bloated, or harder for an AI agent to retrieve.

==================================================
PHASE 5 — ANTI-BLOAT AND SPLIT TEST
==================================================

Before finalizing any merged principle, run this split test.

A master principle is overloaded if it contains two or more separable ideas that would answer different analyst questions.

Split or create sub-principles when a principle combines distinct tests such as:
- EPS growth versus economic performance
- ROE adjusted for leverage
- accounting gimmick detection
- low-leverage compounding
- real after-tax purchasing power
- fixed-income duration risk
- underwriting discipline
- insurance float economics
- management incentives
- valuation discipline

Use this rule:

If one principle now contains multiple distinct analytical tests, either:
1. keep a concise parent principle and move the separable ideas into sub_principles, or
2. split into separate master principles if each idea can stand alone.

Do not create giant “kitchen sink” principles.

Each master principle should ideally answer one primary question.

Examples:
- “Is the business earning attractive returns on required capital?”
- “Are those returns real or created by leverage/gimmicks?”
- “Does the business preserve purchasing power after inflation and taxes?”
- “Is insurance float valuable or costly?”
- “Is a cheap business actually a value trap?”

==================================================
PHASE 6 — CATEGORY CLEANLINESS TEST
==================================================

Before finalizing categories, check whether each category contains one coherent conceptual domain.

Do not mix different domains merely because they appeared in the same letter.

Examples:
- real_owner_returns should contain inflation, taxes, real purchasing power, real owner outcome.
- fixed_income_risk should contain duration, fixed-rate bonds, inflation exposure, accounting marks, optionality.
- insurance should contain underwriting, reserves, float, long-tail risk, premium volume, claims inflation.
- valuation should contain price versus intrinsic value, quality versus cheapness, margin of safety, premium valuation.
- capital_allocation should contain retention, reinvestment, acquisitions, selling discipline, repurchases, redeployment.

If a category mixes two different domains, split the category.

Avoid excessive fragmentation, but never preserve a category that creates retrieval confusion.

==================================================
PHASE 7 — UPDATE MASTER PRINCIPLES
==================================================

When updating each principle:

1. Preserve stable identity
- Keep existing master_principle_id when the core principle remains the same.
- Do not renumber existing IDs unless duplicate consolidation requires it.
- New IDs must be unique and category-based.

2. Preserve old knowledge
- Do not delete old high-value logic unless it is truly redundant or incorrect.
- If old wording is stronger than incoming wording, keep the old wording.
- If incoming wording is more Buffett-faithful or more analytically precise, refine the wording.

3. Preserve source lineage
Every kept principle must include:
- first_seen_year
- reinforced_by_years
- updated_from_years
- source_item_ids
- source_refs if available
- source_evidence

4. Preserve merge history
Every changed principle must record what happened:
- incoming_year
- action
- source_item_id
- notes

5. Improve actionability
Each principle should contain, where relevant:
- performance_logic
- financial_metrics
- metric_definition
- metric_usage_context
- metric_limitations
- evaluation_questions
- diagnostic_signals
- practical_rules
- example_buffett_phrasing
- tags

6. Keep concise
Do not add every incoming question, rule, or metric.
Keep only the clearest, least redundant, most reusable ones.

==================================================
PHASE 8 — METRIC NORMALIZATION AND DEDUPLICATION
==================================================

Metrics must be deduplicated aggressively.

Do not allow duplicate metric keys inside the same principle.

Bad:
- "return_on_equity" appears twice with two definitions.
- "combined_ratio" appears twice with slightly different definitions.
- "debt_to_equity" appears twice in metric_definition.

Good:
- Keep one metric key.
- Keep the best definition.
- Add nuance in usage_context or limitations, not duplicate definitions.

For each principle:
- financial_metrics must contain unique metric names.
- metric_definition must contain one definition per metric.
- metric_usage_context must not repeat the same use case.
- metric_limitations must not repeat the same limitation.
- evaluation_questions must be unique by meaning.
- diagnostic_signals must be unique by meaning.
- practical_rules must be unique by meaning.
- tags must be unique and normalized.

Normalize synonyms when possible:
- beginning_shareholder_equity and beginning_shareholders_equity should be standardized.
- underwriting_profit and underwriting_profitability should be standardized unless meaning differs.
- capital_employed and invested_capital should be separated only if the framework defines a meaningful difference.

If the schema uses arrays of strings, keep them deduplicated.
If adding richer metric objects is allowed, use:

"metrics": [
  {
    "metric_name": "",
    "definition": "",
    "usage_context": [],
    "limitations": []
  }
]

Do not include both duplicated metric arrays and duplicated metric objects.

==================================================
PHASE 9 — SOURCE LINEAGE RULES
==================================================

Never invent source evidence.

For every incoming item that is kept in any form, record:
- incoming_year
- incoming_item_id
- incoming_title
- matched_master_principle_id if applicable
- decision
- reason

For every master principle, source_lineage must include:
- first_seen_year
- reinforced_by_years
- updated_from_years
- source_item_ids
- source_evidence

source_evidence entries must include:
- year
- source_document_type
- source_item_id
- principle_title_at_source

If source_file or source_refs are available in the input, preserve them.

If exact quote, passage, or chunk ID is unavailable, do not invent one.
Use null or omit the field.

==================================================
PHASE 10 — REJECTION DISCIPLINE
==================================================

You are allowed and expected to reject weak incoming items.

Reject or demote items that are:
- only historical commentary
- merely company-specific examples
- vague investing clichés
- duplicate with no new nuance
- too narrow to help future analysis
- macro commentary without reusable investment logic
- already fully represented in stronger form
- not faithful to Buffett’s actual causal reasoning

If rejecting an item:
- include it in merge_decision_log
- set decision to REJECT
- give a specific reason
- do not add it to master principles

A high-quality update does not need to accept every incoming item.

==================================================
PHASE 11 — SELF-AUDIT BEFORE FINAL OUTPUT
==================================================

Before producing the final JSON, run an internal quality audit.

The final JSON must pass these gates:

1. Preservation gate
- No old high-value master principle is accidentally deleted.
- If any old principle is removed or consolidated, the merge_decision_log must explain why.

2. New-year coverage gate
- Every important incoming item is handled as ADD, MERGE, REINFORCE, REFINE, EXTEND, ADD_AS_SUB_PRINCIPLE, ADD_AS_METRIC, ADD_AS_QUESTION, ADD_AS_EVIDENCE, or REJECT.
- No incoming item is silently ignored.

3. Non-redundancy gate
- No near-duplicate master principles remain unless they represent clearly distinct analytical tests.
- No duplicate metric definitions remain inside a principle.
- No repeated questions, signals, rules, or tags remain inside a principle.

4. Anti-bloat gate
- No principle contains too many unrelated analytical tests.
- If a principle has multiple related but distinct ideas, use sub_principles.

5. Category gate
- No category mixes unrelated domains.
- Real owner returns and fixed-income risk should not be forced into one category unless explicitly justified.

6. Faithfulness gate
- No generic investment cliché is introduced unless clearly grounded in Buffett source logic.
- No source year or source item is invented.

7. Agent usability gate
- Each principle should be retrievable and usable by an AI agent.
- Each principle should help ask sharper stock-analysis questions.
- Each principle should reduce generic advice rather than increase it.

8. Metadata consistency gate
- update_summary counts must match actual decisions.
- items_refined must be greater than zero if any wording or conceptual precision was improved.
- items_rejected must match REJECT decisions in merge_decision_log.
- items_after_update must equal actual final master principle count, excluding sub_principles unless explicitly counted.

==================================================
RECOMMENDED MASTER SCHEMA
==================================================

Return one valid JSON downloadable link using this structure.

Use schema_version "2.2" unless the existing master schema must be preserved for compatibility.

{
  "schema_version": "2.2",
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
      "items_after_update": 0,
      "principles_split_due_to_overload": 0,
      "principles_consolidated_due_to_redundancy": 0,
      "deduplication_actions_performed": 0
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
          "performance_logic": [],
          "financial_metrics": [],
          "metric_definition": [],
          "metric_usage_context": [],
          "metric_limitations": [],
          "evaluation_questions": [],
          "diagnostic_signals": [],
          "practical_rules": [],
          "example_buffett_phrasing": [],
          "tags": [],
          "sub_principles": [
            {
              "sub_principle_id": "",
              "sub_principle_title": "",
              "sub_principle_statement": "",
              "normalized_sub_principle": "",
              "primary_analytical_question": "",
              "performance_logic": [],
              "evaluation_questions": [],
              "diagnostic_signals": [],
              "practical_rules": [],
              "tags": [],
              "source_item_ids": []
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
                "source_chunk_id": null
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
                "action": "added|merged|reinforced|refined|extended|added_as_sub_principle|added_as_metric|added_as_question|added_as_evidence|rejected",
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
      "matched_master_principle_id": null,
      "matched_sub_principle_id": null,
      "decision": "ADD|MERGE|REINFORCE|REFINE|EXTEND|ADD_AS_SUB_PRINCIPLE|ADD_AS_METRIC|ADD_AS_QUESTION|ADD_AS_EVIDENCE|REJECT",
      "correctness_rationale": "",
      "reason": "",
      "source_preserved": true
    }
  ],
  "post_merge_quality_audit": {
    "old_principles_preserved": true,
    "old_principles_removed_or_consolidated": [],
    "incoming_items_handled_count": 0,
    "incoming_items_missing": [],
    "duplicate_principles_remaining": [],
    "overloaded_principles_remaining": [],
    "category_concerns": [],
    "metric_deduplication_notes": [],
    "source_lineage_concerns": [],
    "schema_consistency_notes": [],
    "agent_usability_notes": []
  }
}

==================================================
SCHEMA COMPATIBILITY RULE
==================================================

If CURRENT_FRAMEWORK_JSON uses schema_version "2.0" and strict backward compatibility is required, you may keep schema_version "2.0".

However:
- still apply all anti-bloat, deduplication, category-cleanliness, and self-audit rules
- include post_merge_quality_audit if possible
- include sub_principles if needed to avoid overloaded principles
- do not preserve schema simplicity at the cost of knowledge quality

==================================================
OUTPUT CONSTRUCTION RULES
==================================================

1. Always return the full updated master framework JSON.
Do not return only changed items.

2. Preserve all valuable existing principles.
Do not accidentally delete old knowledge.

3. Be conservative about adding new principles.
Create a new principle only when the idea is truly distinct and reusable.

4. Be aggressive about deduplication.
Deduplicate:
- principles
- sub-principles
- metrics
- metric definitions
- metric usage contexts
- metric limitations
- evaluation questions
- diagnostic signals
- practical rules
- tags
- source evidence

5. Use sub_principles when needed.
Use them to avoid both:
- shallow duplication
- overloaded master principles

6. Keep categories clean.
Split categories that mix unrelated domains.

7. Keep wording Buffett-faithful.
Do not convert source-specific Buffett logic into vague generic investing advice.

8. Preserve source lineage for every retained idea.
Every kept incoming item must be traceable somewhere.

9. Record all decisions.
Every incoming item must appear exactly once in merge_decision_log.

10. Validate metadata counts.
update_summary must reflect actual decisions.

11. Return valid JSON downloadable link only.

==================================================
DECISION HEURISTICS
==================================================

Use these heuristics:

ADD
Use when:
- incoming item contains a distinct reusable Buffett principle
- it answers a different analytical question from existing principles
- merging would overload an existing principle

MERGE
Use when:
- incoming item expresses the same principle
- no distinct analytical test would be lost
- the existing principle can absorb it cleanly

REINFORCE
Use when:
- incoming item mainly confirms an existing principle
- add recurrence and source lineage
- add only minimal supporting detail

REFINE
Use when:
- incoming item improves wording, precision, or Buffett-faithfulness
- update the existing principle statement or normalized principle

EXTEND
Use when:
- incoming item adds a new but closely related angle
- the extension does not overload the principle

ADD_AS_SUB_PRINCIPLE
Use when:
- incoming item belongs under an existing principle family
- but has its own distinct analytical question

ADD_AS_METRIC
Use when:
- incoming item contributes a useful metric but no new principle

ADD_AS_QUESTION
Use when:
- incoming item contributes a useful evaluation question but no new principle

ADD_AS_EVIDENCE
Use when:
- incoming item is mainly supporting evidence or example

REJECT
Use when:
- item is duplicate, too narrow, non-reusable, generic, or merely historical

==================================================
FINAL INSTRUCTION
==================================================

Now execute the full Progressive Framework Distillation Loop and return only the final UPDATED_MASTER_FRAMEWORK_JSON as one downloadable JSON file, make sure the output download link is not dead.