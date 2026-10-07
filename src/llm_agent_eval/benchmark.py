from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from .datasets import LocalizationTask
from .dense import TransformerDenseRetriever, reciprocal_rank_fusion
from .evaluation import evaluate_localization
from .indexing import chunk_repository
from .retrieval import BM25Retriever, RankedChunk
from .swebench import SWEBenchRecord, from_mapping


@dataclass(frozen=True)
class BenchmarkFailure:
    instance_id: str
    error_type: str
    message: str


class HybridRetriever:
    def __init__(self, sparse: BM25Retriever, dense: TransformerDenseRetriever, *, pool_size: int = 100):
        self.sparse = sparse
        self.dense = dense
        self.pool_size = pool_size

    def search(self, query: str, *, top_k: int = 10) -> list[RankedChunk]:
        pool = max(top_k, self.pool_size)
        return reciprocal_rank_fusion(
            [self.sparse.search(query, top_k=pool), self.dense.search(query, top_k=pool)],
            top_k=top_k,
        )


def load_swebench_jsonl(path: str | Path) -> list[SWEBenchRecord]:
    records: list[SWEBenchRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(from_mapping(json.loads(line)))
    return records


def _git(*args: str, cwd: str | Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def checkout_revision(record: SWEBenchRecord, destination: Path) -> Path:
    repo_dir = destination / record.instance_id.replace("/", "__")
    _git("clone", "--filter=blob:none", "--no-checkout", f"https://github.com/{record.repo}.git", str(repo_dir))
    _git("checkout", "--detach", record.base_commit, cwd=repo_dir)
    return repo_dir


def _aggregate(successes: list[dict]) -> dict:
    if not successes:
        return {}
    return {
        "mean_recall_at_k": mean(row["recall_at_k"] for row in successes),
        "mean_reciprocal_rank": mean(row["reciprocal_rank"] for row in successes),
        "mean_ndcg_at_k": mean(row["ndcg_at_k"] for row in successes),
        "mean_retrieval_ms": mean(row["retrieval_ms"] for row in successes),
        "mean_retrieved_characters": mean(row["retrieved_characters"] for row in successes),
    }


def evaluate_records(
    records: Iterable[SWEBenchRecord],
    *,
    method: str = "bm25",
    model_name: str | None = None,
    top_k: int = 5,
    candidate_chunks: int = 50,
    context_budget_chars: int | None = 40_000,
    lines_per_chunk: int = 80,
    overlap: int = 20,
) -> dict:
    """Evaluate localization while keeping reference patches outside model-visible inputs."""
    if method not in {"bm25", "dense", "hybrid"}:
        raise ValueError(f"Unsupported retrieval method: {method}")
    if method != "bm25" and not model_name:
        raise ValueError("model_name is required for dense and hybrid retrieval")

    successes: list[dict] = []
    failures: list[BenchmarkFailure] = []

    with TemporaryDirectory(prefix="llm-agent-eval-") as tmp:
        root = Path(tmp)
        for record in records:
            try:
                repo_dir = checkout_revision(record, root)
                chunks = chunk_repository(repo_dir, lines_per_chunk=lines_per_chunk, overlap=overlap)
                if method == "bm25":
                    retriever = BM25Retriever(chunks)
                else:
                    dense = TransformerDenseRetriever(chunks, model_name=model_name or "")
                    retriever = dense if method == "dense" else HybridRetriever(BM25Retriever(chunks), dense)

                task = LocalizationTask(
                    task_id=record.instance_id,
                    issue=record.problem_statement,
                    gold_files=record.gold_files,
                )
                result = evaluate_localization(
                    retriever,
                    [task],
                    top_k=top_k,
                    candidate_chunks=candidate_chunks,
                    context_budget_chars=context_budget_chars,
                )
                successes.append(result["tasks"][0])
            except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
                failures.append(BenchmarkFailure(record.instance_id, type(exc).__name__, str(exc)))

    return {
        "n_requested": len(successes) + len(failures),
        "n_successful": len(successes),
        "n_failed": len(failures),
        "aggregate": _aggregate(successes),
        "configuration": {
            "method": method,
            "model_name": model_name,
            "top_k": top_k,
            "candidate_chunks": candidate_chunks,
            "context_budget_chars": context_budget_chars,
            "lines_per_chunk": lines_per_chunk,
            "overlap": overlap,
        },
        "tasks": successes,
        "failures": [asdict(failure) for failure in failures],
    }


def write_result(result: dict, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
