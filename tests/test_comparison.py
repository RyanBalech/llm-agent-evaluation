import pytest

from llm_agent_eval.comparison import compare_task_results, paired_bootstrap_difference


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
