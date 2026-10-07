import pytest

from llm_agent_eval.statistics import paired_bootstrap_ci


def test_paired_bootstrap_reports_direction_of_consistent_gain():
    estimate, low, high = paired_bootstrap_ci(
        [0.0, 0.2, 0.4, 0.6],
        [0.1, 0.3, 0.5, 0.7],
        samples=1000,
        seed=7,
    )
    assert estimate == pytest.approx(0.1)
    assert low > 0
    assert high > 0
