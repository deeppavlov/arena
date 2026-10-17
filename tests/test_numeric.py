import pytest

from evaluator.numeric import parse_score, validate_score


@pytest.mark.parametrize("value", [True, False, None, "1", float("nan"), float("inf")])
def test_scores_require_finite_numeric_values(value: object) -> None:
    with pytest.raises(ValueError, match="score"):
        validate_score(value)


@pytest.mark.parametrize("value", [str(2**53), str(2**53 + 1), str(-(2**53))])
def test_parsing_scores_preserves_integer_precision_until_validation(value: str) -> None:
    with pytest.raises(ValueError, match="binary64"):
        parse_score(value)


@pytest.mark.parametrize("value", [2**53 - 1, -(2**53 - 1), 0, -1])
def test_exact_supported_integers_round_trip(value: int) -> None:
    assert parse_score(str(value)) == value
    assert validate_score(value) == value


@pytest.mark.parametrize("value", ["true", "null", '"2"', "NaN", "1e309", "invalid"])
def test_invalid_text_scores_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        parse_score(value)


def test_finite_floating_scores_support_scientific_notation() -> None:
    assert parse_score("1e308") == 1e308
    assert parse_score("-2.5") == -2.5
