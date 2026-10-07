from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_agent_eval.statistics import paired_bootstrap_ci


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def task_map(result: dict) -> dict[str, dict]:
    return {row["task_id"]: row for row in result["tasks"]}


def paired_summary(baseline: dict, treatment: dict) -> dict:
    b = task_map(baseline)
    t = task_map(treatment)
    ids = sorted(set(b) & set(t))
    output = {"n_paired": len(ids), "metrics": {}}
    for metric in ("recall_at_k", "reciprocal_rank", "ndcg_at_k"):
        effect, low, high = paired_bootstrap_ci(
            [b[i][metric] for i in ids],
            [t[i][metric] for i in ids],
            seed=2026,
        )
        output["metrics"][metric] = {
            "mean_difference": effect,
            "bootstrap_95_ci": [low, high],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bm25")
    parser.add_argument("dense")
    parser.add_argument("hybrid")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = {name: load(path) for name, path in [
        ("bm25", args.bm25), ("dense", args.dense), ("hybrid", args.hybrid)
    ]}
    comparison = {
        "aggregates": {name: value["aggregate"] for name, value in results.items()},
        "paired_vs_bm25": {
            "dense": paired_summary(results["bm25"], results["dense"]),
            "hybrid": paired_summary(results["bm25"], results["hybrid"]),
        },
        "interpretation_rule": "A method is not called an improvement when its paired 95% bootstrap CI includes zero.",
    }
    Path(args.output).write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
