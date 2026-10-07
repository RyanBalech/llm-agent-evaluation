from __future__ import annotations

import random
from statistics import mean


def paired_bootstrap_difference(
    baseline: list[float],
    challenger: list[float],
    *,
    samples: int = 10_000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Paired bootstrap CI for mean(challenger - baseline)."""
    if len(baseline) != len(challenger) or not baseline:
        raise ValueError("paired metric vectors must have the same non-zero length")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    deltas = [b - a for a, b in zip(baseline, challenger)]
    rng = random.Random(seed)
    boot = []
    n = len(deltas)
    for _ in range(samples):
        boot.append(mean(deltas[rng.randrange(n)] for _ in range(n)))
    boot.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = boot[int(alpha * (samples - 1))]
    hi = boot[int((1.0 - alpha) * (samples - 1))]
    return {
        "mean_difference": mean(deltas),
        "ci_low": lo,
        "ci_high": hi,
        "confidence": confidence,
        "bootstrap_samples": samples,
    }


def compare_task_results(baseline: dict, challenger: dict) -> dict:
    """Compare two evaluation artifacts on the exact same ordered task IDs."""
    base_tasks = baseline["tasks"]
    chal_tasks = challenger["tasks"]
    base_ids = [row["task_id"] for row in base_tasks]
    chal_ids = [row["task_id"] for row in chal_tasks]
    if len(set(base_ids)) != len(base_ids) or len(set(chal_ids)) != len(chal_ids):
        raise ValueError("duplicate task IDs are not independent observations")
    if base_ids != chal_ids:
        raise ValueError("comparisons require identical ordered task IDs")

    metrics = ["recall_at_k", "reciprocal_rank", "ndcg_at_k"]
    return {
        metric: paired_bootstrap_difference(
            [row[metric] for row in base_tasks],
            [row[metric] for row in chal_tasks],
        )
        for metric in metrics
    }


def compare_benchmark_results(baseline: dict, challenger: dict) -> dict:
    """Reject incomplete or mismatched studies before paired inference."""
    for result in (baseline, challenger):
        if result.get("n_failed", 0) or result.get("failures"):
            raise ValueError("all requested tasks must succeed before a headline comparison")
        if result.get("n_requested") != len(result["tasks"]):
            raise ValueError("incomplete benchmark: requested and completed tasks differ")
    keys = ("top_k", "candidate_chunks", "context_budget_chars", "lines_per_chunk", "overlap")
    for key in keys:
        if key not in baseline["configuration"] or key not in challenger["configuration"]:
            raise ValueError(f"missing comparison configuration: {key}")
        if baseline["configuration"][key] != challenger["configuration"][key]:
            raise ValueError(f"mismatched comparison configuration: {key}")
    return {
        "n_paired": len(baseline["tasks"]),
        "metrics": compare_task_results(baseline, challenger),
    }
