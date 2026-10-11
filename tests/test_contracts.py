import base64

import pytest
from pydantic import ValidationError

from evaluator.contracts import Candidate, Criterion, Metric, Submission, Vote


@pytest.mark.parametrize(
    "path",
    [
        "", ".", "..", "/prompt.txt", "../prompt.txt", "nested/../../prompt.txt",
        "nested/../prompt.txt", "./prompt.txt", "nested/./prompt.txt", "nested//prompt.txt",
        "nested/", "//prompt.txt", "nested\\prompt.txt", "prompt\x00.txt", "x" * 241,
    ],
)
def test_candidate_rejects_unsafe_or_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        Candidate(id="candidate", files={path: "YQ=="})


@pytest.mark.parametrize(
    "files",
    [
        {"folder": "YQ==", "folder/file": "Yg=="},
        {"folder/deep/file": "Yg==", "folder": "YQ=="},
    ],
)
def test_candidate_rejects_file_directory_collisions(files: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="directory"):
        Candidate(id="candidate", files=files)


@pytest.mark.parametrize("value", ["?", "YQ", "YQ==\n", "====", "YQ==garbage", "текст"])
def test_candidate_rejects_invalid_base64(value: str) -> None:
    with pytest.raises(ValidationError, match="base64"):
        Candidate(id="candidate", files={"prompt.txt": value})


@pytest.mark.parametrize("file_count", [1, 2])
def test_candidate_size_limit_covers_all_decoded_files(file_count: int) -> None:
    encoded = base64.b64encode(b"x" * (10 * 1024 * 1024 // file_count + 1)).decode()
    with pytest.raises(ValidationError, match="10 MiB"):
        Candidate(id="candidate", files={f"file-{i}": encoded for i in range(file_count)})


def test_candidate_accepts_exact_decoded_size_limit() -> None:
    encoded = base64.b64encode(b"x" * (10 * 1024 * 1024)).decode()
    assert Candidate(id="candidate", files={"data.bin": encoded}).files["data.bin"] == encoded


def test_candidate_digest_identifies_decoded_content_independent_of_order_and_name() -> None:
    first = Candidate(id="first", files={"nested/данные.bin": "AAE=", "prompt": "YQ=="})
    second = Candidate(id="second", files={"prompt": "YR==", "nested/данные.bin": "AAE="})
    assert first.digest() == second.digest()
    changed = Candidate(id="first", files={"nested/данные.bin": "AAI=", "prompt": "YQ=="})
    assert first.digest() != changed.digest()


def test_submission_rejects_duplicate_candidate_ids() -> None:
    with pytest.raises(ValidationError, match="IDs must be unique"):
        Submission(candidates=[
            Candidate(id="same", files={"prompt": "YQ=="}),
            Candidate(id="same", files={"prompt": "Yg=="}),
        ])


def test_submission_rejects_duplicate_content_under_different_ids_and_encodings() -> None:
    with pytest.raises(ValidationError, match="duplicate candidate content"):
        Submission(candidates=[
            Candidate(id="first", files={"prompt": "YQ=="}),
            Candidate(id="second", files={"prompt": "YR=="}),
        ])


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_criterion_requires_positive_finite_weight(weight: float) -> None:
    with pytest.raises(ValidationError):
        Criterion(id="correct", question="Correct?", weight=weight)


def test_metric_rejects_duplicate_criteria() -> None:
    with pytest.raises(ValidationError, match="criterion IDs must be unique"):
        Metric(version="v1", instruction="Judge.", criteria=[
            Criterion(id="correct", question="Correct?", weight=1),
            Criterion(id="correct", question="Complete?", weight=2),
        ])


@pytest.mark.parametrize("score", [True, False, "3", 3.0, 0, 6, float("nan")])
def test_vote_requires_integer_score_in_range(score: object) -> None:
    with pytest.raises(ValidationError):
        Vote.model_validate({"score": score, "reason": "Observed evidence."})


@pytest.mark.parametrize("score", [1, 5])
def test_vote_accepts_score_boundaries(score: int) -> None:
    assert Vote(score=score, reason="Observed evidence.").score == score
