"""Audit a complete localization artifact against its exact evaluation labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from llm_agent_eval.benchmark import load_swebench_jsonl
from llm_agent_eval.metrics import ndcg_at_k, recall_at_k


def analyze(result: dict, records: list, *, top_k: int = 5) -> dict:
    rows = result["tasks"]
    ids = [row["task_id"] for row in rows]
    if result.get("n_failed", 0) or result["n_requested"] != len(rows):
        raise ValueError("cannot analyze a failed or partial study as complete")
    if ids != [record.instance_id for record in records] or len(set(ids)) != len(ids):
        raise ValueError("ordered task IDs must exactly match the source records")
    grouped, cases = defaultdict(list), []
    for row, record in zip(rows, records):
        files, gold = row["retrieved_files"][:top_k], record.gold_files
        recall = recall_at_k(files, set(gold), top_k)
        ndcg = ndcg_at_k(files, set(gold), top_k)
        if abs(recall - row["recall_at_k"]) > 1e-10 or abs(ndcg - row["ndcg_at_k"]) > 1e-10:
            raise ValueError(f"artifact scores disagree with supplied gold labels: {record.instance_id}")
        missed = sorted(gold - set(files))
        cases.append({"instance_id": record.instance_id, "repo": record.repo,
                      "base_commit": record.base_commit, "gold_files": sorted(gold),
                      "retrieved_files": files, "missed_gold_files": missed,
                      "recall_at_5": recall, "gold_file_count": len(gold),
                      "issue_mentions_gold_path": any(path in record.problem_statement for path in gold),
                      "outcome": "complete" if recall == 1 else "partial" if recall > 0 else "miss"})
        grouped[record.repo].append(row)
    return {"n_tasks": len(rows), "outcomes": dict(Counter(case["outcome"] for case in cases)),
            "by_repository": {repo: {"n": len(group),
                                     "recall_at_5": mean(row["recall_at_k"] for row in group),
                                     "mrr": mean(row["reciprocal_rank"] for row in group)}
                              for repo, group in sorted(grouped.items())},
            "cases": cases,
            "note": "Labels are for post-hoc scoring only. This does not measure patch correctness. "
                    "Case taxonomy describes observed retrieval outcomes, not inferred root causes."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("tasks")
    parser.add_argument("--output", default="results/localization_analysis.json")
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text())
    analysis = analyze(result, load_swebench_jsonl(args.tasks))
    analysis["source_result_sha256"] = hashlib.sha256(Path(args.result).read_bytes()).hexdigest()
    analysis["source_tasks_sha256"] = hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({key: analysis[key] for key in ["n_tasks", "outcomes", "by_repository"]}, indent=2))


if __name__ == "__main__":
    main()
