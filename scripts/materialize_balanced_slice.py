"""Freeze a repository-balanced evaluation slice disjoint from development and smoke tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

DATASET = "princeton-nlp/SWE-bench_Verified"
REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
SMOKE_IDS = {"sympy__sympy-13798", "pytest-dev__pytest-5631", "sympy__sympy-17318"}


def select_balanced(rows, excluded: set[str], limit: int):
    if limit <= 0:
        raise ValueError("limit must be positive")
    groups = defaultdict(list)
    for row in rows:
        if row["instance_id"] not in excluded:
            groups[row["repo"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: (hashlib.sha256(r["instance_id"].encode()).hexdigest(),
                                 r["instance_id"]))
    selected = []
    depth = 0
    while len(selected) < limit:
        candidates = [groups[repo][depth] for repo in sorted(groups) if depth < len(groups[repo])]
        if not candidates:
            raise ValueError("Not enough unseen tasks for the requested slice")
        selected.extend(candidates[:limit - len(selected)])
        depth += 1
    return selected


def main():
    from datasets import load_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".benchmark/balanced-50.jsonl")
    parser.add_argument("--manifest", default="configs/verified_balanced_50_manifest.json")
    parser.add_argument("--exclude-manifest", default="configs/verified_20_manifest.json")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    excluded = SMOKE_IDS | {r["instance_id"] for r in
                           json.loads(Path(args.exclude_manifest).read_text())["tasks"]}
    dataset = load_dataset(DATASET, split="test", revision=REVISION)
    rows = select_balanced(dataset, excluded, args.limit)
    records = [{key: row[key] for key in ["instance_id", "repo", "base_commit",
                                        "problem_statement", "patch"]} for row in rows]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    manifest = {"dataset": DATASET, "dataset_revision": REVISION, "split": "test",
                "selection": "Round-robin repositories alphabetically; SHA256(instance_id) within repo",
                "purpose": "Unseen repository-balanced evaluation; not a population-weighted estimate",
                "excluded_instance_ids": sorted(excluded),
                "repository_counts": dict(sorted(Counter(r["repo"] for r in records).items())),
                "jsonl_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "tasks": [{"instance_id": r["instance_id"], "repo": r["repo"],
                           "base_commit": r["base_commit"],
                           "issue_sha256": hashlib.sha256(r["problem_statement"].encode()).hexdigest(),
                           "patch_sha256": hashlib.sha256(r["patch"].encode()).hexdigest()}
                          for r in records]}
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_tasks": len(records), "repository_counts": manifest["repository_counts"]},
                     indent=2))


if __name__ == "__main__":
    main()
