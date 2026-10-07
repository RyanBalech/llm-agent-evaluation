from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_agent_eval.comparison import compare_benchmark_results


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
            "dense": compare_benchmark_results(results["bm25"], results["dense"]),
            "hybrid": compare_benchmark_results(results["bm25"], results["hybrid"]),
        },
        "interpretation_rule": "A method is not called an improvement when its paired 95% bootstrap CI includes zero.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
