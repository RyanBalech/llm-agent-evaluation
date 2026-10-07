from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_agent_eval.comparison import compare_task_results
from llm_agent_eval.datasets import load_jsonl
from llm_agent_eval.dense import TransformerDenseRetriever
from llm_agent_eval.evaluation import evaluate_localization
from llm_agent_eval.hybrid import HybridRetriever
from llm_agent_eval.indexing import chunk_repository
from llm_agent_eval.retrieval import BM25Retriever


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("tasks")
    p.add_argument("--model", default="microsoft/codebert-base")
    p.add_argument("--output", default="results/retrieval_comparison.json")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidate-chunks", type=int, default=50)
    p.add_argument("--context-budget-chars", type=int, default=40_000)
    args = p.parse_args()

    chunks = chunk_repository(args.repo)
    tasks = load_jsonl(args.tasks)
    sparse = BM25Retriever(chunks)
    dense = TransformerDenseRetriever(chunks, model_name=args.model)
    hybrid = HybridRetriever(sparse, dense)

    kwargs = {
        "top_k": args.top_k,
        "candidate_chunks": args.candidate_chunks,
        "context_budget_chars": args.context_budget_chars,
    }
    results = {
        "bm25": evaluate_localization(sparse, tasks, **kwargs),
        "dense": evaluate_localization(dense, tasks, **kwargs),
        "hybrid_rrf": evaluate_localization(hybrid, tasks, **kwargs),
    }
    results["paired_vs_bm25"] = {
        "dense": compare_task_results(results["bm25"], results["dense"]),
        "hybrid_rrf": compare_task_results(results["bm25"], results["hybrid_rrf"]),
    }
    results["configuration"] = {
        "dense_model": args.model,
        "top_k": args.top_k,
        "candidate_chunks": args.candidate_chunks,
        "context_budget_chars": args.context_budget_chars,
        "primary_metric": "recall_at_k",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
