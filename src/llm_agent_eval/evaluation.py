from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from time import perf_counter

from .datasets import LocalizationTask
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from .retrieval import BM25Retriever, unique_file_ranking


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    retrieved_files: list[str]
    retrieved_characters: int
    retrieval_ms: float


def evaluate_localization(
    retriever: BM25Retriever,
    tasks: list[LocalizationTask],
    *,
    top_k: int = 10,
    candidate_chunks: int = 50,
) -> dict:
    if not tasks:
        raise ValueError("No evaluation tasks provided")

    results: list[TaskResult] = []
    for task in tasks:
        start = perf_counter()
        chunks = retriever.search(task.issue, top_k=candidate_chunks)
        elapsed_ms = (perf_counter() - start) * 1000
        files = unique_file_ranking(chunks)
        result = TaskResult(
            task_id=task.task_id,
            recall_at_k=recall_at_k(files, set(task.gold_files), top_k),
            reciprocal_rank=reciprocal_rank(files, set(task.gold_files)),
            ndcg_at_k=ndcg_at_k(files, set(task.gold_files), top_k),
            retrieved_files=files[:top_k],
            retrieved_characters=sum(len(item.chunk.text) for item in chunks),
            retrieval_ms=elapsed_ms,
        )
        results.append(result)

    return {
        "n_tasks": len(results),
        "top_k": top_k,
        "mean_recall_at_k": mean(r.recall_at_k for r in results),
        "mean_reciprocal_rank": mean(r.reciprocal_rank for r in results),
        "mean_ndcg_at_k": mean(r.ndcg_at_k for r in results),
        "mean_retrieval_ms": mean(r.retrieval_ms for r in results),
        "tasks": [asdict(r) for r in results],
    }
