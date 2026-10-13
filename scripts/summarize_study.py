"""Create a compact report from three completed, input-matched retrieval experiments."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from llm_agent_eval.benchmark import write_result
from llm_agent_eval.comparison import compare_benchmark_results, paired_bootstrap_difference


def summarize(results: dict, name: str) -> dict:
    baseline = results["bm25"]
    paired = {method: compare_benchmark_results(baseline, results[method])
              for method in ("dense", "hybrid")}
    repository = {r["instance_id"]: r["repo"] for r in baseline["input_manifest"]}
    by_repo = {}
    for method, result in results.items():
        groups = defaultdict(list)
        for task in result["tasks"]:
            groups[repository[task["task_id"]]].append(task)
        by_repo[method] = {repo: {
            "n_tasks": len(rows), "recall_at_k": mean(r["recall_at_k"] for r in rows),
            "reciprocal_rank": mean(r["reciprocal_rank"] for r in rows),
            "ndcg_at_k": mean(r["ndcg_at_k"] for r in rows),
        } for repo, rows in sorted(groups.items())}
    repositories = sorted(by_repo["bm25"])
    macro = {method: {
        "mean_recall_at_k": mean(r["recall_at_k"] for r in by_repo[method].values()),
        "paired_vs_bm25": paired_bootstrap_difference(
            [by_repo["bm25"][repo]["recall_at_k"] for repo in repositories],
            [by_repo[method][repo]["recall_at_k"] for repo in repositories]),
    } for method in results}
    task_rows = []
    for i, task in enumerate(baseline["tasks"]):
        row = {"task_id": task["task_id"], "repo": repository[task["task_id"]],
               "gold_files": task.get("gold_files", []), "methods": {}}
        for method, result in results.items():
            r = result["tasks"][i]
            row["methods"][method] = {k: r[k] for k in
                                      ("recall_at_k", "reciprocal_rank", "ndcg_at_k",
                                       "retrieved_files", "retrieval_ms")}
        task_rows.append(row)
    timings = {}
    for method, result in results.items():
        rows = result["tasks"]
        timings[method] = {"median_retrieval_ms": median(r["retrieval_ms"] for r in rows),
                           "mean_source_seconds": mean(r["source_seconds"] for r in rows),
                           "mean_index_seconds": mean(r["index_seconds"] for r in rows),
                           "source_cache_hits": sum(r.get("source_cache_hit", False) for r in rows),
                           "embedding_cache_hits": sum(bool(r.get("embedding_cache_hit")) for r in rows)}
    return {"name": name, "n_tasks": len(task_rows), "n_repositories": len(repositories),
            "repository_counts": dict(sorted(Counter(repository.values()).items())),
            "aggregates": {m: r["aggregate"] for m, r in results.items()},
            "paired_vs_bm25": paired, "repository_macro": macro,
            "by_repository": by_repo, "timings": timings, "tasks": task_rows,
            "configuration": baseline["configuration"], "input_manifest": baseline["input_manifest"],
            "method_configurations": {m: r["configuration"] for m, r in results.items()},
            "provenance": {m: r.get("study_environment", r.get("environment"))
                           for m, r in results.items()},
            "interpretation": [
                "File localization does not measure successful bug fixes.",
                "Task bootstrap treats tasks as independent; repository-macro intervals resample repository means.",
                "A paired interval including zero does not establish an improvement.",
                "Repository balancing changes the estimand; this is not a SWE-bench population-weighted score.",
                "Timings are one request per issue with recorded cache states, not a serving-load benchmark.",
            ]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    results = {m: json.loads((directory / f"{m}.json").read_text(encoding="utf-8"))
               for m in ("bm25", "dense", "hybrid")}
    report = summarize(results, args.name)
    write_result(report, args.output)
    print(json.dumps(report["aggregates"], indent=2))


if __name__ == "__main__":
    main()
