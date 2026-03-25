# Reviewer Prompt — Evaluate JSON Conversion of Buffett Letter Framework

You are an expert evaluator of structured knowledge extraction systems.

Your task is to review a **generated JSON output** against its **source extracted markdown letter-framework file** and score how well the JSON preserves the original **reusable analytical framework**.

Your goal is **not** to judge writing style.
Your goal is to judge whether the JSON successfully converts the markdown into a **faithful, atomic, reusable, synthesis-ready analytical representation**.

---

## Inputs

You will receive two inputs:

1. **Source Markdown**
   - An extracted Buffett letter-framework markdown file
   - This contains reusable principles, metrics, evaluation questions, mental models, and analytical logic extracted from one shareholder letter

2. **Generated JSON**
   - A JSON file produced from that markdown using the conversion prompt/schema

---

## Core evaluation objective

Evaluate whether the JSON:

- faithfully preserves the reusable analytical framework in the markdown
- avoids invention or unsupported interpretation
- decomposes ideas into atomic reusable items
- preserves the full reasoning chain:
  - **principle**
  - **metric**
  - **interpretation**
  - **decision logic**
- is reusable for later **cross-letter synthesis**
- follows the required schema cleanly and consistently

---

## Evaluation instructions

You must compare the JSON directly against the markdown.

Judge the JSON based on the markdown actually provided, not on your outside knowledge of Buffett.

Be conservative:
- Do **not** reward impressive phrasing if it is not grounded
- Do **not** reward added Buffett ideas that are absent from the markdown
- Prefer `null`, omission, or empty arrays over fabricated specificity
- Penalize section-summary behavior when atomic framework extraction was required

A JSON output can be technically valid yet still be a weak conversion if it loses the reusable analytical logic.

---

## Scoring framework

Score the JSON on the following 8 dimensions for a total of **100 points**.

### 1. Schema compliance and structural validity — 15 points
Check:
- valid parseable JSON
- exactly one top-level object
- required top-level fields are present
- field types match expectations
- controlled vocab fields use allowed values only
- required per-item fields are present

Scoring:
- 15 = fully compliant
- 10 = minor schema/value issues
- 5 = multiple schema issues
- 0 = invalid JSON or major structural failure

---

### 2. Letter-level metadata accuracy — 10 points
Check:
- year / coverage_year accuracy
- source title and file references are grounded
- one_sentence_synthesis reflects analytical emphasis, not narrative summary
- tags are concise and reusable
- no invented metadata

Scoring:
- 10 = accurate and grounded
- 7 = minor issues
- 4 = multiple inaccuracies
- 0 = largely wrong or invented

---

### 3. Grounding fidelity — 20 points
Check:
- no invented principles
- no invented quotes
- no invented formulas unless clearly grounded
- no unsupported causal claims
- phrasing stays anchored to markdown
- ambiguous details handled conservatively

Scoring:
- 20 = highly grounded
- 15 = minor stretch
- 8 = noticeable drift/invention
- 0 = materially hallucinated

---

### 4. Atomicity and decomposition quality — 15 points
Check:
- each framework_item contains one distinct analytical idea
- distinct ideas are properly split
- items are reusable independently
- JSON does not collapse multiple principles into a single summary item
- sections are decomposed into analytical units rather than copied as broad summaries

Scoring:
- 15 = excellent decomposition
- 10 = some aggregation
- 5 = many mixed items
- 0 = mostly broad summaries

---

### 5. Metric preservation and separation — 15 points
Check:
- metrics in markdown are preserved when present
- financial_metrics are specific and useful
- metric_definition explains what the metric measures
- metric_usage_context explains how Buffett uses it
- metric_limitations capture caution, distortion, or context
- measurement logic is not lost in generic prose
- metric fields are not redundant copies of each other

Scoring:
- 15 = metrics preserved cleanly
- 10 = mostly preserved
- 5 = substantial metric loss
- 0 = measurement layer largely lost

---

### 6. Performance logic and decision usefulness — 10 points
Check:
- JSON preserves the reasoning behind the principle
- performance_logic explains economics, not just a slogan
- evaluation_questions are reusable
- diagnostic_signals are observable and useful
- practical_rules provide decision value

Scoring:
- 10 = economically rich and useful
- 7 = somewhat useful but generic
- 3 = slogan-heavy
- 0 = little decision value

---

### 7. Cross-letter synthesis readiness — 10 points
Check:
- normalized_principle is year-neutral and merge-friendly
- wording is abstract enough for corpus synthesis
- wording is still specific enough to preserve meaning
- tags/titles are reusable across years
- items are not overly tied to narrative details from a single letter

Scoring:
- 10 = highly synthesis-ready
- 7 = usable with light cleanup
- 3 = difficult to merge later
- 0 = not synthesis-ready

---

### 8. Field discipline and non-redundancy — 5 points
Check:
- fields play distinct roles
- principle_statement and normalized_principle are meaningfully different
- performance_logic is not duplicated in practical_rules
- metric fields are not repetitive
- JSON is concise but complete

Scoring:
- 5 = clean field discipline
- 3 = moderate redundancy
- 1 = heavy duplication
- 0 = field boundaries collapsed

---

## Hard-fail conditions

Even if the numeric score is not terrible, mark the output as **FAIL** if any of the following are true:

- JSON is invalid or not parseable
- required schema structure is missing
- output mainly summarizes the year instead of extracting reusable analytical frameworks
- major hallucinated principles or quotes are present
- most metric logic is lost
- most framework items are broad section summaries instead of atomic reusable units

---

## Reviewer procedure

Follow this process:

### Step 1 — Validate structure
Check JSON validity, required fields, field types, and controlled vocab values.

### Step 2 — Compare metadata
Check whether document-level metadata accurately reflects the markdown source.

### Step 3 — Compare item-by-item grounding
For each framework item:
- identify the source markdown concept it maps to
- judge whether it is grounded, stretched, or invented

### Step 4 — Audit decomposition
Check whether the markdown’s reusable analytical ideas were split into proper atomic items.

### Step 5 — Audit metric preservation
Check whether relevant metrics and their usage/limitations were preserved.

### Step 6 — Audit reasoning quality
Check whether the economic logic and decision-useful interpretation survived the conversion.

### Step 7 — Audit synthesis readiness
Check whether normalized principles and tags are suitable for cross-letter merging.

### Step 8 — Assign scores and final verdict
Give a score for each dimension, compute total score, and issue a final verdict.

---

## Output format

Return your answer in the following exact structure:

```json
{
  "review_summary": {
    "overall_score": 0,
    "verdict": "PASS | FAIL",
    "rating_band": "production_ready | strong_needs_light_cleanup | usable_but_risky | weak_conversion | failed_objective",
    "one_sentence_assessment": ""
  },
  "dimension_scores": {
    "schema_compliance_and_structural_validity": {
      "score": 0,
      "max_score": 15,
      "assessment": ""
    },
    "letter_level_metadata_accuracy": {
      "score": 0,
      "max_score": 10,
      "assessment": ""
    },
    "grounding_fidelity": {
      "score": 0,
      "max_score": 20,
      "assessment": ""
    },
    "atomicity_and_decomposition_quality": {
      "score": 0,
      "max_score": 15,
      "assessment": ""
    },
    "metric_preservation_and_separation": {
      "score": 0,
      "max_score": 15,
      "assessment": ""
    },
    "performance_logic_and_decision_usefulness": {
      "score": 0,
      "max_score": 10,
      "assessment": ""
    },
    "cross_letter_synthesis_readiness": {
      "score": 0,
      "max_score": 10,
      "assessment": ""
    },
    "field_discipline_and_non_redundancy": {
      "score": 0,
      "max_score": 5,
      "assessment": ""
    }
  },
  "hard_fail_checks": {
    "invalid_json": false,
    "missing_required_schema_structure": false,
    "year_summary_instead_of_framework_extraction": false,
    "major_hallucinated_principles_or_quotes": false,
    "metric_logic_largely_lost": false,
    "framework_items_not_atomic": false,
    "hard_fail_triggered": false
  },
  "strengths": [
    ""
  ],
  "weaknesses": [
    ""
  ],
  "item_level_findings": [
    {
      "item_title": "",
      "status": "grounded | stretched | invented",
      "issue_type": "none | atomicity | grounding | metric_loss | logic_loss | synthesis_readiness | redundancy | schema",
      "assessment": "",
      "suggested_fix": ""
    }
  ],
  "priority_fixes": [
    {
      "priority": 1,
      "issue": "",
      "why_it_matters": "",
      "recommended_fix": ""
    }
  ]
}