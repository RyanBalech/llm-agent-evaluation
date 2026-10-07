from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    p.add_argument("--output", default="results/context_budget_ablation.json")
    p.add_argument("--budgets", type=int, nargs="+", default=[10_000, 20_000, 40_000, 80_000])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidate-chunks", type=int, default=100)
    args = p.parse_args()

    chunks = chunk_repository(args.repo)
    tasks = load_jsonl(args.tasks)
    sparse = BM25Retriever(chunks)
    dense = TransformerDenseRetriever(chunks, model_name=args.model)
    retrievers = {
        "bm25": sparse,
        "dense": dense,
        "hybrid_rrf": HybridRetriever(sparse, dense),
    }
    results = {}
    for budget in args.budgets:
        results[str(budget)] = {
            name: evaluate_localization(
                retriever,
                tasks,
                top_k=args.top_k,
                candidate_chunks=args.candidate_chunks,
                context_budget_chars=budget,
            )
            for name, retriever in retrievers.items()
        }

    artifact = {
        "experiment": "retrieval_context_budget_ablation",
        "primary_metric": "recall_at_k",
        "dense_model": args.model,
        "budgets_chars": args.budgets,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
