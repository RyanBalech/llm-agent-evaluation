from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SMOKE_INSTANCE_IDS = {
    "sympy__sympy-13798",
    "pytest-dev__pytest-5631",
    "sympy__sympy-17318",
}


def stable_score(instance_id: str) -> str:
    return hashlib.sha256(instance_id.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a deterministic SWE-bench Verified slice"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--revision", default="main", help="Dataset branch or immutable HF commit")
    parser.add_argument(
        "--exclude-smoke",
        action="store_true",
        help="Exclude the three instances used while engineering the smoke pipeline.",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError('Install benchmark dependencies with: pip install -e ".[bench]"') from exc

    revision = HfApi().dataset_info("princeton-nlp/SWE-bench_Verified", revision=args.revision).sha
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", revision=revision)
    candidates = [
        row for row in dataset
        if not args.exclude_smoke or str(row["instance_id"]) not in SMOKE_INSTANCE_IDS
    ]
    records = sorted(
        candidates,
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
        "dataset_revision": revision,
        "jsonl_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "selection": "lowest SHA-256(instance_id), ascending",
        "limit": args.limit,
        "exclude_smoke": args.exclude_smoke,
        "excluded_instance_ids": sorted(SMOKE_INSTANCE_IDS) if args.exclude_smoke else [],
        "instance_ids": [row["instance_id"] for row in records],
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
