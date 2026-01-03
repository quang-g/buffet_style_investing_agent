import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Union

FILES = [
    "../data/chunks/berkshire_1977_chunks.json",
    "../data/chunks/berkshire_1986_chunks.json",
    "../data/chunks/berkshire_2007_chunks.json"
]

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def type_name(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int) and not isinstance(x, bool):
        return "int"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "str"
    if isinstance(x, list):
        return "list"
    if isinstance(x, dict):
        return "dict"
    return type(x).__name__

def is_number_type(t: str) -> bool:
    return t in {"int", "float"}

def compatible_types(expected: Set[str], got: str) -> bool:
    # allow int/float interchangeability
    if got in expected:
        return True
    if is_number_type(got) and any(is_number_type(t) for t in expected):
        return True
    return False

def flatten_paths(obj: Any, prefix: str = "") -> Dict[str, Set[str]]:
    """
    Return mapping: path -> set(type_names seen at that path)
    For lists, we record the list itself and also the element types as path + "[]".
    """
    out: Dict[str, Set[str]] = defaultdict(set)

    def rec(x: Any, p: str):
        out[p].add(type_name(x))
        if isinstance(x, dict):
            for k, v in x.items():
                rec(v, f"{p}.{k}" if p else str(k))
        elif isinstance(x, list):
            # element types
            elem_types = set(type_name(e) for e in x)
            out[(p + "[]") if p else "[]"].update(elem_types)
            # also recurse into dict/list elements to capture nested paths
            for e in x:
                if isinstance(e, (dict, list)):
                    rec(e, (p + "[]") if p else "[]")

    rec(obj, prefix)
    return out

def summarize_file_schema(chunks: List[dict]) -> Dict[str, Set[str]]:
    agg: Dict[str, Set[str]] = defaultdict(set)
    for ch in chunks:
        paths = flatten_paths(ch)
        for path, types in paths.items():
            agg[path].update(types)
    return dict(agg)

def diff_schemas(baseline: Dict[str, Set[str]], other: Dict[str, Set[str]]) -> Dict[str, Any]:
    base_paths = set(baseline.keys())
    other_paths = set(other.keys())

    missing_paths = sorted(base_paths - other_paths)
    extra_paths = sorted(other_paths - base_paths)

    type_mismatches = []
    for p in sorted(base_paths & other_paths):
        expected = baseline[p]
        got_types = other[p]
        # if any got_type not compatible with expected, flag it
        bad = [gt for gt in got_types if not compatible_types(expected, gt)]
        if bad:
            type_mismatches.append((p, sorted(expected), sorted(got_types)))

    return {
        "missing_paths": missing_paths,
        "extra_paths": extra_paths,
        "type_mismatches": type_mismatches,
    }

def check_required_keys(chunks: List[dict], required_top: List[str], required_meta: List[str]) -> List[dict]:
    issues = []
    for i, ch in enumerate(chunks):
        missing_top = [k for k in required_top if k not in ch]
        meta = ch.get("metadata")
        missing_meta = []
        if isinstance(meta, dict):
            missing_meta = [k for k in required_meta if k not in meta]
        else:
            missing_meta = required_meta[:]  # metadata absent/wrong type

        if missing_top or missing_meta:
            issues.append({
                "index": i,
                "chunk_id": ch.get("chunk_id"),
                "missing_top": missing_top,
                "missing_metadata": missing_meta,
                "metadata_type": type_name(meta),
            })
    return issues

def main():
    loaded = {}
    for fp in FILES:
        data = load_json(fp)
        if not isinstance(data, list):
            raise ValueError(f"{fp} must be a JSON array (list) of chunk objects, got {type_name(data)}")
        if not all(isinstance(x, dict) for x in data):
            raise ValueError(f"{fp} array must contain objects (dicts).")
        loaded[fp] = data

    # Baseline = first file
    baseline_fp = FILES[0]
    baseline_schema = summarize_file_schema(loaded[baseline_fp])

    # Derive "required" keys from baseline presence frequency (100% of chunks in baseline)
    base_chunks = loaded[baseline_fp]
    top_key_counts = Counter()
    meta_key_counts = Counter()

    for ch in base_chunks:
        top_key_counts.update(ch.keys())
        meta = ch.get("metadata")
        if isinstance(meta, dict):
            meta_key_counts.update(meta.keys())

    n = len(base_chunks)
    required_top = sorted([k for k, c in top_key_counts.items() if c == n])
    required_meta = sorted([k for k, c in meta_key_counts.items() if c == n])

    print(f"\nBaseline file: {baseline_fp}")
    print(f"Chunks: {n}")
    print(f"Required top-level keys (100% in baseline): {required_top}")
    print(f"Required metadata keys (100% in baseline): {required_meta}")

    # Compare others to baseline
    for fp in FILES[1:]:
        print("\n" + "="*90)
        print(f"Comparing: {fp}")
        other_schema = summarize_file_schema(loaded[fp])
        d = diff_schemas(baseline_schema, other_schema)

        # High-signal summary
        print(f"Missing paths vs baseline: {len(d['missing_paths'])}")
        print(f"Extra paths vs baseline:   {len(d['extra_paths'])}")
        print(f"Type mismatches:           {len(d['type_mismatches'])}")

        # Show some details (tune limits)
        if d["missing_paths"]:
            print("\n--- Missing paths (first 60) ---")
            for p in d["missing_paths"][:60]:
                print(p)

        if d["extra_paths"]:
            print("\n--- Extra paths (first 60) ---")
            for p in d["extra_paths"][:60]:
                print(p)

        if d["type_mismatches"]:
            print("\n--- Type mismatches (first 60) ---")
            for p, exp, got in d["type_mismatches"][:60]:
                print(f"{p}\n  expected: {exp}\n  got:      {got}")

        # Per-chunk required key checks (based on baseline 100% keys)
        req_issues = check_required_keys(loaded[fp], required_top, required_meta)
        print(f"\nChunks failing required-key check: {len(req_issues)}")
        for row in req_issues[:30]:
            print(row)
        if len(req_issues) > 30:
            print(f"... ({len(req_issues)-30} more)")

if __name__ == "__main__":
    main()
