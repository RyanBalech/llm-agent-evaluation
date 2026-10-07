from scripts.materialize_verified_slice import stable_score


def test_stable_score_is_deterministic_and_orderable():
    assert stable_score("abc") == stable_score("abc")
    assert stable_score("abc") != stable_score("def")
