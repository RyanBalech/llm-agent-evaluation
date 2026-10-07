from __future__ import annotations

import argparse
import json

from .benchmark import evaluate_records, load_swebench_jsonl, write_result
from .datasets import load_jsonl
from .evaluation import evaluate_localization
from .indexing import chunk_repository
from .retrieval import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate repository-level issue localization")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a local repository")
    evaluate.add_argument("--repo", required=True)
    evaluate.add_argument("--tasks", required=True)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--candidate-chunks", type=int, default=50)
    evaluate.add_argument("--lines-per-chunk", type=int, default=80)
    evaluate.add_argument("--overlap", type=int, default=20)
    evaluate.add_argument("--context-budget-chars", type=int)

    swebench = subparsers.add_parser("swebench", help="Evaluate SWE-bench JSONL records")
    swebench.add_argument("--tasks", required=True)
    swebench.add_argument("--output", required=True)
    swebench.add_argument("--limit", type=int)
    swebench.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25")
    swebench.add_argument("--model-name")
    swebench.add_argument("--top-k", type=int, default=5)
    swebench.add_argument("--candidate-chunks", type=int, default=50)
    swebench.add_argument("--context-budget-chars", type=int, default=40_000)
    swebench.add_argument("--lines-per-chunk", type=int, default=80)
    swebench.add_argument("--overlap", type=int, default=20)

    args = parser.parse_args()

    if args.command == "evaluate":
        chunks = chunk_repository(
            args.repo, lines_per_chunk=args.lines_per_chunk, overlap=args.overlap
        )
        tasks = load_jsonl(args.tasks)
        result = evaluate_localization(
            BM25Retriever(chunks),
            tasks,
            top_k=args.top_k,
            candidate_chunks=args.candidate_chunks,
            context_budget_chars=args.context_budget_chars,
        )
        print(json.dumps(result, indent=2))
        return

    records = load_swebench_jsonl(args.tasks)
    if args.limit is not None:
        records = records[: args.limit]
    result = evaluate_records(
        records,
        method=args.method,
        model_name=args.model_name,
        top_k=args.top_k,
        candidate_chunks=args.candidate_chunks,
        context_budget_chars=args.context_budget_chars,
        lines_per_chunk=args.lines_per_chunk,
        overlap=args.overlap,
    )
    write_result(result, args.output)
    print(json.dumps({
        "output": args.output,
        "n_requested": result["n_requested"],
        "n_successful": result["n_successful"],
        "n_failed": result["n_failed"],
    }, indent=2))


if __name__ == "__main__":
    main()
