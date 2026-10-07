from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stable_score(instance_id: str) -> str:
    return hashlib.sha256(instance_id.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a deterministic SWE-bench Verified development slice"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('Install benchmark dependencies with: pip install -e ".[bench]"') from exc

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    records = sorted(
        dataset,
        key=lambda row: (stable_score(str(row["instance_id"])), str(row["instance_id"])),
    )[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in records:
            payload = {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "problem_statement": row["problem_statement"],
                "patch": row["patch"],
            }
            handle.write(json.dumps(payload) + "\n")

    manifest = {
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "selection": "lowest SHA-256(instance_id), ascending",
        "limit": args.limit,
        "instance_ids": [row["instance_id"] for row in records],
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
