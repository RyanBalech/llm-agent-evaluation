import runpy
from pathlib import Path

import pytest

select_balanced = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/materialize_balanced_slice.py"))["select_balanced"]


def test_balanced_selection_is_deterministic_disjoint_and_uses_small_repos():
    rows = [{"instance_id": f"{repo}-{i}", "repo": repo}
            for repo, count in [("a", 20), ("b", 10), ("c", 1)] for i in range(count)]
    result = select_balanced(rows, {"a-0", "b-0"}, 7)
    assert result == select_balanced(list(reversed(rows)), {"a-0", "b-0"}, 7)
    assert not {r["instance_id"] for r in result} & {"a-0", "b-0"}
    assert [r["repo"] for r in result].count("a") == 3
    assert [r["repo"] for r in result].count("b") == 3
    assert [r["repo"] for r in result].count("c") == 1


def test_balanced_selection_rejects_insufficient_or_invalid_size():
    with pytest.raises(ValueError):
        select_balanced([], set(), 1)
    with pytest.raises(ValueError):
        select_balanced([], set(), 0)
