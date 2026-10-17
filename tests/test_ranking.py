from copy import deepcopy
from itertools import permutations
from typing import Any

import pytest

from evaluator.ranking import RANKING_POLICY, choose_winner, choose_winners


def candidate(
    candidate_id: str,
    score: float = 4.0,
    tokens: int = 100,
    seconds: float = 10.0,
    unreported: int = 0,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "score": score,
        "solver_tokens": tokens,
        "solver_seconds": seconds,
        "solver_unreported_attempts": unreported,
    }


def test_quality_takes_priority_over_resources_and_accounting() -> None:
    scores = [
        candidate("cheap", score=4, tokens=1, seconds=1),
        candidate("best", score=4.01, tokens=10000, seconds=1000, unreported=1),
    ]
    assert choose_winner(scores) == "best"


def test_equal_quality_prefers_fewer_tokens_before_less_time() -> None:
    scores = [candidate("fast", tokens=200, seconds=1), candidate("lean", tokens=100)]
    assert choose_winner(scores) == "lean"


def test_equal_quality_and_tokens_prefers_less_time() -> None:
    scores = [candidate("slow", seconds=100), candidate("fast", seconds=10)]
    assert choose_winner(scores) == "fast"


def test_missing_usage_cannot_appear_cheaper_than_complete_usage() -> None:
    scores = [
        candidate("unknown", tokens=0, seconds=1, unreported=1),
        candidate("complete", tokens=10000, seconds=1000),
    ]
    assert choose_winner(scores) == "complete"


def test_all_unknown_usage_skips_token_counts_and_uses_time() -> None:
    scores = [
        candidate("slow", tokens=0, seconds=20, unreported=1),
        candidate("fast", tokens=10000, seconds=10, unreported=10),
    ]
    assert choose_winner(scores) == "fast"


def test_exact_tie_is_stable_under_input_permutations_without_mutating_rows() -> None:
    rows = [candidate("z"), candidate("b"), candidate("a")]
    original = deepcopy(rows)
    for ordering in permutations(rows):
        assert choose_winner(list(ordering)) == "a"
    assert rows == original


def test_single_candidate_with_zero_measurements_is_valid() -> None:
    assert choose_winner([candidate("only", score=0, tokens=0, seconds=0)]) == "only"


def test_ranking_policy_has_a_stable_version_for_provenance() -> None:
    assert RANKING_POLICY == "score-direction-known-tokens-asc-seconds-asc-id-asc-v2"


@pytest.mark.parametrize("field", list(candidate("example")))
def test_every_measurement_is_required(field: str) -> None:
    row = candidate("bad")
    del row[field]
    with pytest.raises(ValueError, match=field):
        choose_winner([row])


@pytest.mark.parametrize("field", ["score", "solver_seconds"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1"])
def test_invalid_scores_and_times_fail(field: str, value: Any) -> None:
    row = candidate("bad")
    row[field] = value
    with pytest.raises(ValueError, match=field):
        choose_winner([candidate("good", score=5), row])


@pytest.mark.parametrize("field", ["solver_tokens", "solver_unreported_attempts"])
@pytest.mark.parametrize("value", [-1, 1.0, True, "1", None])
def test_counts_must_be_nonnegative_strict_integers(field: str, value: Any) -> None:
    row = candidate("bad")
    row[field] = value
    with pytest.raises(ValueError, match=field):
        choose_winner([row])


@pytest.mark.parametrize("value", ["", " \n", 1, None])
def test_invalid_candidate_ids_fail(value: Any) -> None:
    row = candidate("bad")
    row["candidate_id"] = value
    with pytest.raises(ValueError, match="candidate_id"):
        choose_winner([row])


def test_duplicate_candidate_ids_fail() -> None:
    with pytest.raises(ValueError, match="unique"):
        choose_winner([candidate("same"), candidate("same", score=5)])


def test_empty_results_fail() -> None:
    with pytest.raises(ValueError, match="empty"):
        choose_winner([])



def test_negative_scores_are_valid_but_negative_times_are_not() -> None:
    assert choose_winner([candidate("low", score=-4), candidate("high", score=-1)]) == "high"
    with pytest.raises(ValueError, match="solver_seconds"):
        choose_winner([candidate("negative-time", seconds=-1)])


def test_top_n_respects_minimization_and_resource_ties() -> None:
    rows = [
        candidate("best", score=-5, tokens=100),
        candidate("same-cheap", score=-5, tokens=0, seconds=1000),
        candidate("next", score=-3),
        candidate("worst", score=1, tokens=0),
    ]
    assert choose_winners(rows, count=3, direction="minimize") == [
        "same-cheap", "best", "next",
    ]
    assert choose_winner(rows, direction="minimize") == "same-cheap"
    assert choose_winners(rows) == ["worst"]


@pytest.mark.parametrize("count", [0, -1, 2, True, 1.0])
def test_top_n_requires_an_exact_available_positive_count(count: Any) -> None:
    with pytest.raises(ValueError, match="count"):
        choose_winners([candidate("only")], count=count)


def test_invalid_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="direction"):
        choose_winners([candidate("only")], direction="sideways")
