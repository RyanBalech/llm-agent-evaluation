import pytest

from llm_agent_eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_localization_metrics_reward_early_relevant_files():
    ranked = ["wrong.py", "target.py", "other.py"]
    relevant = {"target.py"}
    assert recall_at_k(ranked, relevant, 2) == 1.0
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert 0 < ndcg_at_k(ranked, relevant, 3) < 1


def test_recall_handles_multiple_gold_files():
    assert recall_at_k(["a.py", "x.py"], {"a.py", "b.py"}, 2) == 0.5


def test_empty_relevance_is_rejected():
    with pytest.raises(ValueError):
        recall_at_k(["a.py"], set(), 1)
