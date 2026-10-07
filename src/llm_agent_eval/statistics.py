from __future__ import annotations

import random
from statistics import mean


def paired_bootstrap_ci(
    baseline: list[float],
    treatment: list[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for the mean treatment-minus-baseline difference."""
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("baseline and treatment must have the same non-zero length")
    if not 0 < confidence < 1 or samples <= 0:
        raise ValueError("invalid bootstrap configuration")

    differences = [t - b for b, t in zip(baseline, treatment)]
    rng = random.Random(seed)
    n = len(differences)
    draws = sorted(
        mean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    alpha = 1 - confidence
    lower = draws[max(0, int((alpha / 2) * samples))]
    upper = draws[min(samples - 1, int((1 - alpha / 2) * samples) - 1)]
    return mean(differences), lower, upper
