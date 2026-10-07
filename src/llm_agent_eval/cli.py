from __future__ import annotations

import argparse
import json

from .datasets import load_jsonl
from .evaluation import evaluate_localization
from .indexing import chunk_repository
from .retrieval import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate repository-level issue localization")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--repo", required=True)
    evaluate.add_argument("--tasks", required=True)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--candidate-chunks", type=int, default=50)
    evaluate.add_argument("--lines-per-chunk", type=int, default=80)
    evaluate.add_argument("--overlap", type=int, default=20)\n    evaluate.add_argument("--context-budget-chars", type=int)
    args = parser.parse_args()

    chunks = chunk_repository(
        args.repo, lines_per_chunk=args.lines_per_chunk, overlap=args.overlap
    )
    tasks = load_jsonl(args.tasks)
    result = evaluate_localization(
        BM25Retriever(chunks),
        tasks,
        top_k=args.top_k,
        candidate_chunks=args.candidate_chunks,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
