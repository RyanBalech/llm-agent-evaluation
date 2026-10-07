"""Reuse each base checkout for a fixed, paired sparse-retrieval ablation."""
from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from llm_agent_eval.benchmark import checkout_revision, load_swebench_jsonl, write_result
from llm_agent_eval.datasets import LocalizationTask
from llm_agent_eval.evaluation import evaluate_localization
from llm_agent_eval.indexing import chunk_repository
from llm_agent_eval.retrieval import BM25Retriever, RankedChunk
from llm_agent_eval.statistics import paired_bootstrap_ci


class FileOrderRetriever:
    def __init__(self, chunks):
        self.chunks = sorted(chunks, key=lambda chunk: chunk.chunk_id)

    def search(self, query, *, top_k=50):
        return [RankedChunk(chunk, 0.) for chunk in self.chunks[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks")
    parser.add_argument("--output", default="results/sparse_ablation.json")
    args = parser.parse_args()
    records = load_swebench_jsonl(args.tasks)
    results = defaultdict(list)
    with TemporaryDirectory(prefix="sparse-ablation-") as temporary:
        for record in records:
            root = checkout_revision(record, Path(temporary))
            chunks = chunk_repository(root, lines_per_chunk=80, overlap=20)
            task = LocalizationTask(record.instance_id, record.problem_statement, record.gold_files)
            sparse, plain = BM25Retriever(chunks), BM25Retriever(chunks, path_weight=0.)
            arms = [(f"path_prior_{budget}", sparse, budget) for budget in [5000, 10000, 20000, 40000, 80000]]
            arms += [("plain_bm25_40000", plain, 40000), ("file_order_40000", FileOrderRetriever(chunks), 40000)]
            for name, retriever, budget in arms:
                row = evaluate_localization(retriever, [task], top_k=5, candidate_chunks=50,
                                            context_budget_chars=budget)["tasks"][0]
                results[name].append(row)
            shutil.rmtree(root)
            print(f"completed {record.instance_id}", flush=True)
    summary = {name: {metric: mean(row[metric] for row in rows)
                      for metric in ["recall_at_k", "reciprocal_rank", "ndcg_at_k", "retrieved_characters"]}
               for name, rows in results.items()}
    baseline = results["path_prior_40000"]
    paired = {}
    for name, rows in results.items():
        deltas = [row["recall_at_k"] - base["recall_at_k"] for row, base in zip(rows, baseline)]
        paired[name] = {"mean_recall_difference": mean(deltas),
                        "ci95": list(paired_bootstrap_ci([0.] * len(deltas), deltas)[1:])}
    write_result({"experiment": "sparse_localization_ablation", "n_tasks": len(records),
                  "task_manifest_sha256": hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
                  "task_revisions": [{"instance_id": r.instance_id, "repo": r.repo,
                                       "base_commit": r.base_commit} for r in records],
                  "configuration": {"top_k": 5, "candidate_chunks": 50,
                                    "lines_per_chunk": 80, "overlap": 20, "path_prior_weight": .15},
                  "environment": {"python": platform.python_version()},
                  "summary": summary, "paired_vs_path_prior_40000": paired, "tasks": dict(results),
                  "limitations": "Exploratory follow-up on the original 20 tasks; task bootstrap "
                                  "does not model dependence between tasks from one repository. "
                                  "Budgets count code characters, excluding prompt/path formatting."}, args.output)


if __name__ == "__main__":
    main()
