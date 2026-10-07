from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Iterable

from .datasets import LocalizationTask
from .evaluation import evaluate_localization
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
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def checkout_revision(record: SWEBenchRecord, destination: Path) -> Path:
    """Materialize the exact repository revision specified by a benchmark record."""
    repo_dir = destination / record.instance_id.replace("/", "__")
    _git("clone", "--filter=blob:none", "--no-checkout", f"https://github.com/{record.repo}.git", str(repo_dir))
    _git("checkout", "--detach", record.base_commit, cwd=repo_dir)
    return repo_dir


def evaluate_records(
    records: Iterable[SWEBenchRecord],
    *,
    top_k: int = 5,
    candidate_chunks: int = 50,
    context_budget_chars: int | None = 40_000,
    lines_per_chunk: int = 80,
    overlap: int = 20,
) -> dict:
    """Evaluate localization while keeping reference patches outside model-visible inputs."""
    successes: list[dict] = []
    failures: list[BenchmarkFailure] = []

    with TemporaryDirectory(prefix="llm-agent-eval-") as tmp:
        root = Path(tmp)
        for record in records:
            try:
                repo_dir = checkout_revision(record, root)
                chunks = chunk_repository(
                    repo_dir,
                    lines_per_chunk=lines_per_chunk,
                    overlap=overlap,
                )
                task = LocalizationTask(
                    task_id=record.instance_id,
                    issue=record.problem_statement,
                    gold_files=record.gold_files,
                )
                result = evaluate_localization(
                    BM25Retriever(chunks),
                    [task],
                    top_k=top_k,
                    candidate_chunks=candidate_chunks,
                    context_budget_chars=context_budget_chars,
                )
                successes.append(result["tasks"][0])
            except Exception as exc:  # benchmark runners must preserve failed instances
                failures.append(
                    BenchmarkFailure(record.instance_id, type(exc).__name__, str(exc))
                )

    return {
        "n_requested": len(successes) + len(failures),
        "n_successful": len(successes),
        "n_failed": len(failures),
        "configuration": {
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
