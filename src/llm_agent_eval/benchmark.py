from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

from .datasets import LocalizationTask
from .dense import TransformerDenseRetriever, TransformerEncoder
from .evaluation import evaluate_localization
from .hybrid import HybridRetriever
from .indexing import chunk_repository
from .retrieval import BM25Retriever
from .swebench import SWEBenchRecord, from_mapping


@dataclass(frozen=True)
class BenchmarkFailure:
    instance_id: str
    error_type: str
    message: str


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
    checkpoint_path: str | Path | None = None,
    model_revision: str | None = None,
    embedding_cache: str | Path | None = None,
) -> dict:
    """Evaluate localization while keeping reference patches outside model-visible inputs."""
    if method not in {"bm25", "dense", "hybrid"}:
        raise ValueError(f"Unsupported retrieval method: {method}")
    if method != "bm25" and not model_name:
        raise ValueError("model_name is required for dense and hybrid retrieval")

    successes: list[dict] = []
    failures: list[BenchmarkFailure] = []
    records = list(records)
    if not records or len({r.instance_id for r in records}) != len(records):
        raise ValueError("benchmark requires non-empty, unique task IDs")
    manifest = [{"instance_id": r.instance_id, "repo": r.repo, "base_commit": r.base_commit,
                 "issue_sha256": hashlib.sha256(r.problem_statement.encode()).hexdigest(),
                 "patch_sha256": hashlib.sha256(r.patch.encode()).hexdigest()} for r in records]
    encoder = None
    configuration = {
        "method": method, "model_name": model_name, "model_revision": model_revision, "top_k": top_k,
        "candidate_chunks": candidate_chunks, "context_budget_chars": context_budget_chars,
        "lines_per_chunk": lines_per_chunk, "overlap": overlap,
        "bm25_path_prior_weight": 0.15,
        "rrf_candidate_multiplier": 4 if method == "hybrid" else None,
        "rrf_constant": 60 if method == "hybrid" else None,
    }

    def snapshot() -> dict:
        return {
            "n_requested": len(records), "n_successful": len(successes),
            "n_failed": len(failures), "n_completed": len(successes) + len(failures),
            "aggregate": _aggregate(successes), "configuration": configuration,
            "tasks": successes, "failures": [asdict(failure) for failure in failures],
            "input_manifest": manifest,
            "input_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            "environment": {"python": platform.python_version()},
        }

    with TemporaryDirectory(prefix="llm-agent-eval-") as tmp:
        root = Path(tmp)
        for record in records:
            repo_dir = None
            try:
                setup_start = perf_counter()
                repo_dir = checkout_revision(record, root)
                chunks = chunk_repository(repo_dir, lines_per_chunk=lines_per_chunk, overlap=overlap)
                if method == "bm25":
                    retriever = BM25Retriever(chunks)
                else:
                    if encoder is None:
                        encoder = TransformerEncoder(model_name or "", revision=model_revision)
                    dense = TransformerDenseRetriever(chunks, model_name=model_name or "",
                                                       encoder=encoder, cache_dir=embedding_cache)
                    configuration["resolved_model_revision"] = dense.resolved_revision
                    retriever = dense if method == "dense" else HybridRetriever(BM25Retriever(chunks), dense)
                setup_seconds = perf_counter() - setup_start

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
                row = result["tasks"][0]
                indexed_files = {chunk.path for chunk in chunks}
                row.update({"setup_seconds": setup_seconds, "n_indexed_chunks": len(chunks),
                            "n_indexed_files": len(indexed_files), "gold_files": sorted(record.gold_files),
                            "unindexed_gold_files": sorted(record.gold_files - indexed_files),
                            "embedding_cache_hit": dense.cache_hit if method != "bm25" else None})
                successes.append(row)
            except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
                failures.append(BenchmarkFailure(record.instance_id, type(exc).__name__, str(exc)))
            finally:
                if repo_dir is not None:
                    shutil.rmtree(repo_dir, ignore_errors=True)
            if checkpoint_path is not None:
                write_result(snapshot(), checkpoint_path)
            print(f"Completed {len(successes) + len(failures)}/{len(records)}: "
                  f"{record.instance_id} ({len(failures)} failures)", flush=True)

    return snapshot()


def write_result(result: dict, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
