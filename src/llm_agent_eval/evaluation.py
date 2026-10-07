from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from time import perf_counter

from .datasets import LocalizationTask
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from .retrieval import BM25Retriever, RankedChunk, unique_file_ranking


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    retrieved_files: list[str]
    retrieved_characters: int
    retrieval_ms: float


def select_under_character_budget(
    ranked: list[RankedChunk], budget: int | None
) -> list[RankedChunk]:
    if budget is None:
        return ranked
    if budget <= 0:
        raise ValueError("context budget must be positive")

    selected: list[RankedChunk] = []
    used = 0
    for item in ranked:
        size = len(item.chunk.text)
        if used + size > budget:
            continue
        selected.append(item)
        used += size
    return selected


def evaluate_localization(
    retriever: BM25Retriever,
    tasks: list[LocalizationTask],
    *,
    top_k: int = 10,
    candidate_chunks: int = 50,
    context_budget_chars: int | None = None,
) -> dict:
    if not tasks:
        raise ValueError("No evaluation tasks provided")

    results: list[TaskResult] = []
    for task in tasks:
        start = perf_counter()
        candidates = retriever.search(task.issue, top_k=candidate_chunks)
        elapsed_ms = (perf_counter() - start) * 1000
        selected = select_under_character_budget(candidates, context_budget_chars)
        files = unique_file_ranking(selected)
        result = TaskResult(
            task_id=task.task_id,
            recall_at_k=recall_at_k(files, set(task.gold_files), top_k),
            reciprocal_rank=reciprocal_rank(files, set(task.gold_files)),
            ndcg_at_k=ndcg_at_k(files, set(task.gold_files), top_k),
            retrieved_files=files[:top_k],
            retrieved_characters=sum(len(item.chunk.text) for item in selected),
            retrieval_ms=elapsed_ms,
        )
        results.append(result)

    return {
        "n_tasks": len(results),
        "top_k": top_k,
        "context_budget_chars": context_budget_chars,
        "mean_recall_at_k": mean(r.recall_at_k for r in results),
        "mean_reciprocal_rank": mean(r.reciprocal_rank for r in results),
        "mean_ndcg_at_k": mean(r.ndcg_at_k for r in results),
        "mean_retrieval_ms": mean(r.retrieval_ms for r in results),
        "tasks": [asdict(r) for r in results],
    }
