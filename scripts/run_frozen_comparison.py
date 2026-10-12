"""Run all three methods against the audited task manifest, refusing dataset drift.

python scripts/run_frozen_comparison.py .benchmark/study.jsonl
Completed method artifacts are reused only when their inputs and settings match.
Incomplete methods restart; per-task checkpoints remain available for diagnosis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

from llm_agent_eval.benchmark import evaluate_records, load_swebench_jsonl, write_result
from llm_agent_eval.comparison import compare_benchmark_results

MODEL = "microsoft/codebert-base"
REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"


def verify_manifest(records, expected):
    actual = [{"instance_id": r.instance_id, "repo": r.repo, "base_commit": r.base_commit,
               "issue_sha256": hashlib.sha256(r.problem_statement.encode()).hexdigest(),
               "patch_sha256": hashlib.sha256(r.patch.encode()).hexdigest()} for r in records]
    if actual != expected:
        raise ValueError("Task data differs from the frozen manifest; refusing an unmatched study")
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks")
    parser.add_argument("--manifest", default="configs/verified_20_manifest.json")
    parser.add_argument("--output-dir", default="results/frozen-comparison")
    args = parser.parse_args()
    records = load_swebench_jsonl(args.tasks)
    manifest = verify_manifest(records, json.loads(Path(args.manifest).read_text())["tasks"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    for source in sorted(Path("src/llm_agent_eval").glob("*.py")) + [Path(__file__)]:
        source_hash.update(source.name.encode())
        source_hash.update(source.read_bytes())
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "source_sha256": source_hash.hexdigest(),
                   "packages": {p: version(p) for p in ["torch", "transformers", "datasets"]},
                   "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                          text=True).strip(),
                   "working_tree_dirty": bool(subprocess.check_output(
                       ["git", "status", "--porcelain"], text=True).strip()),
                   "tasks_sha256": hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest()}
    write_result(environment, output / "environment.json")
    results = {}
    for method in ("bm25", "dense", "hybrid"):
        path = output / f"{method}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            config = cached.get("configuration", {})
            complete = cached.get("n_successful") == len(records) and not cached.get("n_failed")
            matches = (cached.get("input_manifest") == manifest
                       and config.get("method") == method
                       and all(config.get(k) == v for k, v in {
                           "top_k": 5, "candidate_chunks": 50, "context_budget_chars": 40000,
                           "lines_per_chunk": 80, "overlap": 20}.items())
                       and (method == "bm25" or (config.get("model_name") == MODEL
                            and config.get("model_revision") == REVISION)))
            if not matches:
                raise ValueError(f"Existing artifact has different inputs or settings: {path}")
            if complete:
                if cached.get("study_environment") != environment:
                    raise ValueError(f"Completed artifact has different code/environment: {path}")
                results[method] = cached
                continue
        print(f"Starting {method} on {len(records)} frozen tasks", flush=True)
        results[method] = evaluate_records(
            records, method=method, model_name=MODEL if method != "bm25" else None,
            model_revision=REVISION if method != "bm25" else None,
            embedding_cache=".benchmark/embeddings", checkpoint_path=path)
        results[method]["study_environment"] = environment
        write_result(results[method], path)
        if results[method]["n_failed"]:
            raise RuntimeError(f"{method} contains failed tasks; inspect {path}")
    comparison = {"aggregates": {m: r["aggregate"] for m, r in results.items()},
                  "paired_vs_bm25": {m: compare_benchmark_results(results["bm25"], results[m])
                                     for m in ("dense", "hybrid")},
                  "input_manifest": manifest, "environment": environment,
                  "interpretation": "Exploratory 20-task slice; 12 Django tasks. "
                  "No improvement claim if the paired 95% interval includes zero."}
    write_result(comparison, output / "comparison.json")
    print(json.dumps(comparison["aggregates"], indent=2))


if __name__ == "__main__":
    main()
