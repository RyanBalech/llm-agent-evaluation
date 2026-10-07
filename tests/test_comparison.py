from copy import deepcopy

import pytest

from llm_agent_eval.comparison import (
    compare_benchmark_results,
    compare_task_results,
    paired_bootstrap_difference,
)


def test_paired_bootstrap_identical_has_zero_interval():
    result = paired_bootstrap_difference([0.0, 1.0], [0.0, 1.0], samples=100)
    assert result["mean_difference"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0


def test_comparison_requires_same_tasks():
    a = {"tasks": [{"task_id": "a", "recall_at_k": 1, "reciprocal_rank": 1, "ndcg_at_k": 1}]}
    b = {"tasks": [{"task_id": "b", "recall_at_k": 1, "reciprocal_rank": 1, "ndcg_at_k": 1}]}
    with pytest.raises(ValueError):
        compare_task_results(a, b)


def benchmark():
    return {
        "n_requested": 1, "n_failed": 0,
        "configuration": {"top_k": 5, "candidate_chunks": 50, "context_budget_chars": 40000,
                          "lines_per_chunk": 80, "overlap": 20},
        "tasks": [{"task_id": "a", "recall_at_k": 1, "reciprocal_rank": 1, "ndcg_at_k": 1}],
    }


@pytest.mark.parametrize("field,value", [("top_k", 10), ("context_budget_chars", 10000)])
def test_rejects_unequal_budgets(field, value):
    a = benchmark()
    b = deepcopy(a)
    b["configuration"][field] = value
    with pytest.raises(ValueError, match="mismatched"):
        compare_benchmark_results(a, b)


def test_rejects_missing_and_duplicate_tasks():
    a = benchmark()
    b = deepcopy(a)
    b["n_requested"] = 2
    with pytest.raises(ValueError, match="incomplete"):
        compare_benchmark_results(a, b)
    a["tasks"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        compare_task_results(a, a)


def test_identical_complete_benchmarks_have_zero_effect():
    result = compare_benchmark_results(benchmark(), benchmark())
    assert result["n_paired"] == 1
    assert result["metrics"]["recall_at_k"]["mean_difference"] == 0
