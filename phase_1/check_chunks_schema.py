#!/usr/bin/env python3
"""
check_chunks_schema.py

Validate that chunking-result JSON files (1977–2024) share a consistent metadata schema
and are suitable for embedding.

What it checks (high-level):
- File is valid JSON array of chunk objects
- Required top-level keys: chunk_id, content, metadata
- Required metadata keys exist and have consistent types across all files
- Embedding-suitability: non-empty content, reasonable token/char counts, contextual_summary present
- Referential integrity: parent_chunk_id / child_chunk_ids resolve, tiers consistent
- Table consistency: has_table <-> table_data, financial_table implies has_table
- source_span sanity: start/end in bounds, start < end
- Cross-file collisions: duplicate chunk_id across years
- “Soft” warnings: suspicious empty lists (entities/themes), overly generic contextual_summary, etc.

Exit code:
- 0: no issues
- 1: warnings only
- 2: errors found
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Any, Dict, List, Optional, Set, Tuple

# -------------------------
# Canonical schema (IMMUTABLE) — exact match to chunking_strategy_2tier_from_2000.md
# -------------------------

REQUIRED_TOP_LEVEL_KEYS = {"chunk_id", "content", "metadata"}

# EXACT 27 required metadata keys (no optional keys)
REQUIRED_METADATA_KEYS = {
    "letter_year",
    "letter_date",
    "source_file",
    "section_title",
    "section_hierarchy",
    "chunk_tier",
    "parent_chunk_id",
    "child_chunk_ids",
    "content_type",
    "contextual_summary",
    "has_table",
    "table_data",
    "has_financial_data",
    "entities",
    "themes",
    "buffett_concepts",
    "principles",
    "temporal_references",
    "cross_references",
    "retrieval_priority",
    "abstraction_level",
    "token_count",
    "char_count",
    "source_span",
    "standalone_exception",
    "exception_reason",
    "boundary_note",
    "merged_from",
}

# Enums EXACT per spec
ALLOWED_CHUNK_TIERS = {1, 2}
ALLOWED_CONTENT_TYPE = {
    "narrative",
    "financial_table",
    "mistake_confession",
    "principle_statement",
}
ALLOWED_RETRIEVAL_PRIORITY = {"low", "medium", "high"}
ALLOWED_ABSTRACTION_LEVEL = {"low", "medium", "high"}

# Types EXACT/compatible with spec (including nested structures)
TYPE_RULES = {
    # top-level
    "chunk_id": str,
    "content": str,
    "metadata": dict,

    # metadata (27 keys)
    "letter_year": int,
    "letter_date": str,  # YYYY-MM-DD or "0000-00-00"
    "source_file": str,
    "section_title": str,
    "section_hierarchy": list,  # list[str] ideally
    "chunk_tier": int,
    "parent_chunk_id": (str, type(None)),
    "child_chunk_ids": list,  # list[str]
    "content_type": str,  # must be in ALLOWED_CONTENT_TYPE
    "contextual_summary": str,
    "has_table": bool,
    "table_data": list,  # list[TableData], [] if no table
    "has_financial_data": bool,

    # entities: object with 3 keys, each list
    "entities": dict,  # {"companies":[], "people":[], "metrics":[]}

    "themes": list,
    "buffett_concepts": list,

    # principles: list of objects {"statement": str, "category": str}
    "principles": list,

    # temporal_references: {"primary_year": int, "comparison_years": [], "future_outlook": bool}
    "temporal_references": dict,

    # cross_references: {"related_sections_same_letter": [], "related_years": []}
    "cross_references": dict,

    "retrieval_priority": str,
    "abstraction_level": str,
    "token_count": int,
    "char_count": int,

    # source_span: {"start_char": int, "end_char_exclusive": int}
    "source_span": dict,

    "standalone_exception": bool,
    "exception_reason": (str, type(None)),
    "boundary_note": (str, type(None)),
    "merged_from": list,
}

# Heuristics for embedding suitability
MIN_CONTENT_CHARS = 80
MAX_CONTENT_CHARS = 25000  # very large chunks harm retrieval; tune as needed
MIN_CONTEXTUAL_SUMMARY_CHARS = 30

# Detect overly-generic contextual_summary (heuristic)
GENERIC_SUMMARY_PATTERNS = [
    re.compile(r"develops the main ideas", re.I),
    re.compile(r"supports quantitative analysis", re.I),
    re.compile(r"provides tabular financial data", re.I),
]

CHUNK_ID_PATTERN = re.compile(r"^\d{4}-S\d{2}-T[12]-\d{3}$")


# -------------------------
# Reporting structures
# -------------------------

@dataclass
class Finding:
    level: str  # "ERROR" or "WARN"
    file: str
    chunk_id: Optional[str]
    message: str

@dataclass
class FileReport:
    file: str
    errors: int = 0
    warnings: int = 0
    findings: List[Finding] = field(default_factory=list)

    def add(self, level: str, chunk_id: Optional[str], message: str):
        self.findings.append(Finding(level=level, file=self.file, chunk_id=chunk_id, message=message))
        if level == "ERROR":
            self.errors += 1
        else:
            self.warnings += 1


# -------------------------
# Utility helpers
# -------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_instance(value: Any, expected: Any) -> bool:
    # expected can be a type or a tuple of types
    return isinstance(value, expected)

def safe_len(s: Any) -> int:
    return len(s) if isinstance(s, (str, list, dict)) else 0

def normalize_type(t: Any) -> str:
    if isinstance(t, tuple):
        return " | ".join(sorted([x.__name__ for x in t]))
    return t.__name__

def summarize_path(p: str) -> str:
    # Keep output readable
    return p.replace("\\", "/")

def get_year_from_filename(path: str) -> Optional[int]:
    # e.g., "2010_chunks_2tier.json"
    base = os.path.basename(path)
    m = re.match(r"^(\d{4})_chunks_2tier\.json$", base)
    if m:
        return int(m.group(1))
    return None


# -------------------------
# Core validations
# -------------------------

def validate_chunk_basic(report: FileReport, chunk: Dict[str, Any]) -> None:
    # Required top-level keys
    missing = REQUIRED_TOP_LEVEL_KEYS - set(chunk.keys())
    if missing:
        report.add("ERROR", chunk.get("chunk_id"), f"Missing top-level keys: {sorted(missing)}")
        return

    # Type checks for top-level
    for k in ["chunk_id", "content", "metadata"]:
        exp = TYPE_RULES[k]
        if not is_instance(chunk.get(k), exp):
            report.add(
                "ERROR",
                chunk.get("chunk_id"),
                f"Field '{k}' has wrong type: got {type(chunk.get(k)).__name__}, expected {normalize_type(exp)}",
            )

    # chunk_id format sanity (warn, not error, because you may evolve it)
    cid = chunk.get("chunk_id")
    if isinstance(cid, str) and not CHUNK_ID_PATTERN.match(cid):
        report.add("WARN", cid, f"chunk_id does not match expected pattern YYYY-S##-T#-###: '{cid}'")

    # content checks (embedding suitability)
    content = chunk.get("content", "")
    if isinstance(content, str):
        if len(content.strip()) < MIN_CONTENT_CHARS:
            report.add("WARN", cid, f"Content is very short ({len(content.strip())} chars). May embed poorly.")
        if len(content) > MAX_CONTENT_CHARS:
            report.add("WARN", cid, f"Content is very large ({len(content)} chars). May harm retrieval.")

def validate_metadata_schema(report: FileReport, chunk: Dict[str, Any], strict: bool) -> None:
    cid = chunk.get("chunk_id")
    md = chunk.get("metadata")

    if not isinstance(md, dict):
        report.add("ERROR", cid, "metadata is not an object/dict")
        return

    # ---- IMMUTABLE: exact keys, no missing, no extra ----
    meta_keys = set(md.keys())
    if meta_keys != REQUIRED_METADATA_KEYS:
        missing = REQUIRED_METADATA_KEYS - meta_keys
        extra = meta_keys - REQUIRED_METADATA_KEYS
        if missing:
            report.add("ERROR", cid, f"Missing metadata keys: {sorted(missing)}")
        if extra:
            report.add("ERROR", cid, f"Unexpected metadata keys (not allowed): {sorted(extra)}")

    # Type checks (for keys present)
    for k, exp in TYPE_RULES.items():
        if k in ("chunk_id", "content", "metadata"):
            continue
        if k in md and not isinstance(md[k], exp):
            report.add(
                "ERROR",
                cid,
                f"Metadata field '{k}' wrong type: got {type(md[k]).__name__}, expected {normalize_type(exp)}",
            )

    # ---- Enums ----
    tier = md.get("chunk_tier")
    if isinstance(tier, int) and tier not in ALLOWED_CHUNK_TIERS:
        report.add("ERROR", cid, f"chunk_tier={tier} not in allowed {sorted(ALLOWED_CHUNK_TIERS)}")

    ctype = md.get("content_type")
    if isinstance(ctype, str) and ctype not in ALLOWED_CONTENT_TYPE:
        report.add("ERROR", cid, f"content_type='{ctype}' not in {sorted(ALLOWED_CONTENT_TYPE)}")

    rp = md.get("retrieval_priority")
    if isinstance(rp, str) and rp not in ALLOWED_RETRIEVAL_PRIORITY:
        report.add("ERROR", cid, f"retrieval_priority='{rp}' not in {sorted(ALLOWED_RETRIEVAL_PRIORITY)}")

    al = md.get("abstraction_level")
    if isinstance(al, str) and al not in ALLOWED_ABSTRACTION_LEVEL:
        report.add("ERROR", cid, f"abstraction_level='{al}' not in {sorted(ALLOWED_ABSTRACTION_LEVEL)}")

    # ---- Invariants (validation enforced in spec) ----
    # 1) has_table == true <=> len(table_data) >= 1
    has_table = md.get("has_table")
    table_data = md.get("table_data")
    if isinstance(has_table, bool) and isinstance(table_data, list):
        if has_table != (len(table_data) > 0):
            report.add("ERROR", cid, "Invariant violated: has_table must equal (len(table_data) > 0)")

    # 2) content_type == "financial_table" -> has_table == true
    if ctype == "financial_table" and has_table is False:
        report.add("ERROR", cid, 'Invariant violated: content_type="financial_table" requires has_table=true')

    # 3) chunk_tier matches T-digit in chunk_id
    if isinstance(cid, str) and isinstance(tier, int):
        m = re.match(r"^\d{4}-S\d{2}-T([12])-\d{3}$", cid)
        if m:
            t_digit = int(m.group(1))
            if tier != t_digit:
                report.add("ERROR", cid, f"Invariant violated: chunk_tier={tier} != T-digit in chunk_id ({t_digit})")

    # 4) source_span.start_char < source_span.end_char_exclusive (or both 0)
    span = md.get("source_span")
    if isinstance(span, dict):
        start = span.get("start_char")
        end = span.get("end_char_exclusive")
        if isinstance(start, int) and isinstance(end, int):
            if not ((start == 0 and end == 0) or (start < end)):
                report.add("ERROR", cid, "Invariant violated: source_span must have start<end (or both 0)")

    # ---- Nested shape gates from schema ----
    # entities keys must exist and be lists: companies/people/metrics
    ent = md.get("entities")
    if isinstance(ent, dict):
        for sub in ("companies", "people", "metrics"):
            if sub not in ent:
                report.add("ERROR", cid, f"entities missing required key '{sub}'")
            elif not isinstance(ent[sub], list):
                report.add("ERROR", cid, f"entities['{sub}'] must be a list")

    # principles items must be {"statement": str, "category": str}
    principles = md.get("principles")
    if isinstance(principles, list):
        for i, p in enumerate(principles):
            if not isinstance(p, dict):
                report.add("ERROR", cid, f"principles[{i}] must be an object")
                continue
            if set(p.keys()) != {"statement", "category"}:
                report.add("ERROR", cid, f"principles[{i}] must have exactly keys {{'statement','category'}}")
            if not isinstance(p.get("statement"), str) or not isinstance(p.get("category"), str):
                report.add("ERROR", cid, f"principles[{i}] fields must be strings")

    # letter_date format allowed: YYYY-MM-DD or "0000-00-00"
    ld = md.get("letter_date")
    if isinstance(ld, str):
        if ld != "0000-00-00" and not re.match(r"^\d{4}-\d{2}-\d{2}$", ld):
            report.add("ERROR", cid, "letter_date must be YYYY-MM-DD or '0000-00-00'")


def validate_parent_child_integrity(report: FileReport, chunks: List[Dict[str, Any]]) -> None:
    # Build index
    by_id: Dict[str, Dict[str, Any]] = {}
    for ch in chunks:
        cid = ch.get("chunk_id")
        if isinstance(cid, str):
            if cid in by_id:
                report.add("ERROR", cid, "Duplicate chunk_id within file.")
            by_id[cid] = ch

    # Validate references
    for ch in chunks:
        cid = ch.get("chunk_id")
        md = ch.get("metadata", {})
        if not isinstance(md, dict):
            continue

        tier = md.get("chunk_tier")
        parent = md.get("parent_chunk_id")
        children = md.get("child_chunk_ids")

        # parent rules
        if isinstance(tier, int):
            if tier == 1 and parent is not None:
                report.add("WARN", cid, "Tier-1 chunk has non-null parent_chunk_id.")
            if tier == 2:
                if parent is None:
                    report.add("ERROR", cid, "Tier-2 chunk missing parent_chunk_id.")
                elif isinstance(parent, str) and parent not in by_id:
                    report.add("ERROR", cid, f"parent_chunk_id '{parent}' not found in this file.")
                elif isinstance(parent, str):
                    pmd = by_id[parent].get("metadata", {})
                    if isinstance(pmd, dict) and pmd.get("chunk_tier") != 1:
                        report.add("ERROR", cid, f"Tier-2 parent '{parent}' is not tier-1.")

        # child rules
        if isinstance(children, list):
            for child_id in children:
                if not isinstance(child_id, str):
                    report.add("ERROR", cid, "child_chunk_ids contains non-string value.")
                    continue
                if child_id not in by_id:
                    report.add("ERROR", cid, f"child_chunk_id '{child_id}' not found in this file.")
                    continue
                # ensure child points back
                child_md = by_id[child_id].get("metadata", {})
                if isinstance(child_md, dict):
                    if child_md.get("parent_chunk_id") != cid:
                        report.add(
                            "WARN",
                            cid,
                            f"Child '{child_id}' parent_chunk_id != '{cid}' (found '{child_md.get('parent_chunk_id')}').",
                        )

def collect_schema_signature(chunks: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    """
    Returns a set of (metadata_key, type_name) observed in a file.
    Useful to compare consistency across years.
    """
    sig: Set[Tuple[str, str]] = set()
    for ch in chunks:
        md = ch.get("metadata")
        if not isinstance(md, dict):
            continue
        for k, v in md.items():
            sig.add((k, type(v).__name__))
    return sig


# -------------------------
# Main runner
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="data/chunks_2tier",
        help="Directory containing *_chunks_2tier.json files (default: data/chunks_2tier)",
    )
    ap.add_argument(
        "--glob",
        default="*_chunks_2tier.json",
        help="Glob pattern inside --root (default: *_chunks_2tier.json)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Warn on any unexpected metadata keys beyond REQUIRED_METADATA_KEYS.",
    )
    ap.add_argument(
        "--print-ok",
        action="store_true",
        help="Also print files with no findings.",
    )
    args = ap.parse_args()

    root = args.root
    pattern = os.path.join(root, args.glob)
    files = sorted(glob(pattern))
    if not files:
        print(f"[ERROR] No files found at: {summarize_path(pattern)}", file=sys.stderr)
        return 2

    overall_errors = 0
    overall_warnings = 0

    # Cross-file checks
    seen_chunk_ids: Dict[str, str] = {}
    file_schema_sigs: Dict[str, Set[Tuple[str, str]]] = {}

    reports: List[FileReport] = []

    for fp in files:
        rep = FileReport(file=summarize_path(fp))
        reports.append(rep)

        try:
            data = load_json(fp)
        except Exception as e:
            rep.add("ERROR", None, f"Failed to parse JSON: {e}")
            continue

        if not isinstance(data, list):
            rep.add("ERROR", None, "Top-level JSON must be an array (list) of chunks.")
            continue

        # Validate each chunk
        for ch in data:
            if not isinstance(ch, dict):
                rep.add("ERROR", None, "Chunk is not an object/dict.")
                continue
            validate_chunk_basic(rep, ch)
            validate_metadata_schema(rep, ch, strict=args.strict)

            cid = ch.get("chunk_id")
            if isinstance(cid, str):
                if cid in seen_chunk_ids:
                    rep.add("ERROR", cid, f"chunk_id duplicates across files (also in {seen_chunk_ids[cid]}).")
                else:
                    seen_chunk_ids[cid] = summarize_path(fp)

        # Parent/child integrity within file
        validate_parent_child_integrity(rep, data)

        # Capture schema signature
        file_schema_sigs[summarize_path(fp)] = collect_schema_signature(data)

    # Compare schema signatures across files (consistency)
    # Use the most common signature as “reference”
    sig_counts: Dict[frozenset, int] = {}
    for sig in file_schema_sigs.values():
        sig_counts[frozenset(sig)] = sig_counts.get(frozenset(sig), 0) + 1

    reference_sig = None
    if sig_counts:
        reference_sig = max(sig_counts.items(), key=lambda kv: kv[1])[0]

    if reference_sig is not None:
        ref = set(reference_sig)
        for rep in reports:
            sig = file_schema_sigs.get(rep.file)
            if sig is None:
                continue
            extra = sig - ref
            missing = ref - sig
            # These are warnings because you may intentionally evolve schema for some years
            if extra:
                rep.add("WARN", None, f"Schema has extra (key,type) pairs vs common reference: {sorted(extra)[:10]}{' ...' if len(extra)>10 else ''}")
            if missing:
                rep.add("WARN", None, f"Schema missing (key,type) pairs vs common reference: {sorted(missing)[:10]}{' ...' if len(missing)>10 else ''}")

    # Print reports
    for rep in reports:
        overall_errors += rep.errors
        overall_warnings += rep.warnings

        if rep.errors == 0 and rep.warnings == 0 and not args.print_ok:
            continue

        header = f"\n=== {rep.file} | ERRORS={rep.errors} WARNINGS={rep.warnings} ==="
        print(header)
        for f in rep.findings:
            loc = f"[{f.level}]"
            cid = f" chunk_id={f.chunk_id}" if f.chunk_id else ""
            print(f"{loc}{cid} {f.message}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Files scanned: {len(files)}")
    print(f"Total errors: {overall_errors}")
    print(f"Total warnings: {overall_warnings}")

    if overall_errors > 0:
        return 2
    if overall_warnings > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
