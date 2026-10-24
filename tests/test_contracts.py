import base64
import hashlib
import json

import pytest
from pydantic import ValidationError

from evaluator.contracts import Candidate, CommandMetric, Criterion, Metric, Submission, Vote


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


@pytest.mark.parametrize("path", ["я" * 128, "nested/" + "😀" * 64, "я" * 128 + "/file"])
def test_candidate_rejects_components_exceeding_utf8_byte_limit(path: str) -> None:
    with pytest.raises(ValidationError, match="invalid candidate path"):
        Candidate(id="candidate", files={path: "YQ=="})


@pytest.mark.parametrize("path", ["я" * 127 + "a", "я" * 100 + "/" + "я" * 100])
def test_candidate_accepts_unicode_paths_with_valid_component_sizes(path: str) -> None:
    assert Candidate(id="candidate", files={path: "YQ=="}).files == {path: "YQ=="}


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
def test_candidate_accepts_content_above_former_size_limit(file_count: int) -> None:
    encoded = base64.b64encode(b"x" * (10 * 1024 * 1024 // file_count + 1)).decode()
    files = {f"file-{i}": encoded for i in range(file_count)}
    candidate = Candidate(id="candidate", files=files)
    assert candidate.files == files
    assert Candidate.model_validate_json(candidate.model_dump_json()) == candidate


@pytest.mark.parametrize("executable", [False, True])
def test_candidate_accepts_more_than_one_thousand_files(executable: bool) -> None:
    files = {f"file-{i}": "YQ==" for i in range(1001)}
    executable_files = list(files) if executable else []
    candidate = Candidate(id="candidate", files=files, executable_files=executable_files)
    assert candidate.files == files
    assert candidate.executable_files == sorted(executable_files)
    assert Candidate.model_validate_json(candidate.model_dump_json()) == candidate


def test_candidate_digest_identifies_decoded_content_independent_of_order_and_name() -> None:
    first = Candidate(id="first", files={"nested/данные.bin": "AAE=", "prompt": "YQ=="})
    second = Candidate(id="second", files={"prompt": "YR==", "nested/данные.bin": "AAE="})
    assert first.digest() == second.digest()
    changed = Candidate(id="first", files={"nested/данные.bin": "AAI=", "prompt": "YQ=="})
    assert first.digest() != changed.digest()


def test_candidate_digest_preserves_legacy_nonexecutable_identity() -> None:
    candidate = Candidate(id="plain", files={"helper.sh": "YQ=="})
    hashes = {"helper.sh": hashlib.sha256(b"a").hexdigest()}
    expected = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    assert candidate.digest() == expected


def test_executable_files_affect_identity_and_canonicalize_order() -> None:
    files = {"helper.sh": "YQ==", "nested/worker": "Yg=="}
    plain = Candidate(id="plain", files=files)
    executable = Candidate.model_validate({
        "id": "executable", "files": files,
        "executable_files": ["nested/worker", "helper.sh"],
    })
    reordered = Candidate.model_validate({
        "id": "reordered", "files": files,
        "executable_files": ["helper.sh", "nested/worker"],
    })
    assert executable.digest() != plain.digest()
    assert Submission(candidates=[executable]).digest() != Submission(candidates=[plain]).digest()
    assert reordered.digest() == executable.digest()
    assert executable.model_dump()["executable_files"] == ["helper.sh", "nested/worker"]
    assert len(Submission(candidates=[plain, executable]).candidates) == 2


@pytest.mark.parametrize("paths", [["missing"], ["helper.sh", "helper.sh"], ["../helper.sh"]])
def test_executable_files_must_be_unique_existing_candidate_files(paths: list[str]) -> None:
    with pytest.raises(ValidationError, match="executable"):
        Candidate.model_validate({
            "id": "invalid", "files": {"helper.sh": "YQ=="}, "executable_files": paths,
        })


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


def test_builtin_metric_defaults_to_three_votes() -> None:
    metric = Metric(version="v1", instruction="Judge.", criteria=[
        Criterion(id="correct", question="Correct?", weight=1),
    ])
    assert metric.votes == 3


@pytest.mark.parametrize("votes", [1, 7, 1000])
def test_builtin_metric_accepts_positive_vote_counts(votes: int) -> None:
    metric = Metric.model_validate({
        "version": "v1", "instruction": "Judge.", "votes": votes,
        "criteria": [{"id": "correct", "question": "Correct?", "weight": 1}],
    })
    assert metric.votes == votes
    assert metric.model_dump()["votes"] == votes


@pytest.mark.parametrize("votes", [0, -1, True, False, 1.5, 3.0, "3", None])
def test_builtin_metric_rejects_nonpositive_or_noninteger_vote_counts(votes: object) -> None:
    with pytest.raises(ValidationError):
        Metric.model_validate({
            "version": "v1", "instruction": "Judge.", "votes": votes,
            "criteria": [{"id": "correct", "question": "Correct?", "weight": 1}],
        })


@pytest.mark.parametrize("score", [True, False, "3", 3.0, 0, 6, float("nan")])
def test_vote_requires_integer_score_in_range(score: object) -> None:
    with pytest.raises(ValidationError):
        Vote.model_validate({"score": score, "reason": "Observed evidence."})


@pytest.mark.parametrize("score", [1, 5])
def test_vote_accepts_score_boundaries(score: int) -> None:
    assert Vote(score=score, reason="Observed evidence.").score == score


def test_submission_accepts_more_than_five_candidates() -> None:
    candidates = [
        Candidate(id=f"candidate-{index}", files={
            "code": base64.b64encode(str(index).encode()).decode(),
        })
        for index in range(12)
    ]
    assert len(Submission(candidates=candidates).candidates) == 12


def test_command_metric_has_no_required_llm_criteria() -> None:
    metric = Metric.model_validate({
        "version": "distance-v1", "kind": "command", "direction": "minimize",
        "command": {
            "argv": ["/opt/scorer"],
        },
    })
    assert metric.kind == "command"
    assert metric.direction == "minimize"
    assert metric.criteria == []


def test_command_metric_defaults_to_no_model_access() -> None:
    command = CommandMetric(argv=["/opt/scorer"])
    assert command.uses_model is False


@pytest.mark.parametrize("legacy_sources", [None, "invalid", {"relative": "invalid"}])
def test_command_metric_ignores_obsolete_source_checksums(legacy_sources: object) -> None:
    command = CommandMetric.model_validate({
        "argv": ["/opt/scorer"], "source_sha256": legacy_sources,
    })
    assert command == CommandMetric(argv=["/opt/scorer"])
    assert "source_sha256" not in command.model_dump()


@pytest.mark.parametrize("field", ["command", "aggregation"])
@pytest.mark.parametrize("uses_model", [False, True])
def test_command_model_access_can_be_configured(field: str, uses_model: bool) -> None:
    command = {"argv": ["/opt/scorer"]}
    metric = Metric.model_validate({
        "version": "v1", "kind": "command", "command": command,
        field: {**command, "uses_model": uses_model},
    })
    configured = metric.command if field == "command" else metric.aggregation
    assert configured is not None
    assert configured.uses_model is uses_model
    assert metric.model_dump()[field]["uses_model"] is uses_model


@pytest.mark.parametrize("uses_model", [0, 1, "true", "false", None])
def test_command_model_access_requires_boolean(uses_model: object) -> None:
    with pytest.raises(ValidationError):
        CommandMetric.model_validate({
            "argv": ["/opt/scorer"],
            "uses_model": uses_model,
        })


def test_command_metric_rejects_builtin_vote_configuration() -> None:
    with pytest.raises(ValidationError, match="votes"):
        Metric.model_validate({
            "version": "v1", "kind": "command", "votes": 7,
            "command": {
                "argv": ["/opt/scorer"],
            },
        })


@pytest.mark.parametrize("data", [
    {"version": "v1"},
    {"version": "v1", "instruction": "Judge.", "criteria": []},
    {"version": "v1", "kind": "command"},
    {"version": "v1", "kind": "command", "direction": "unknown"},
])
def test_metric_rejects_missing_profile_or_invalid_direction(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Metric.model_validate(data)


def test_command_example_may_omit_reference() -> None:
    from evaluator.contracts import Example

    assert Example(id="example", input="input").reference is None


@pytest.mark.parametrize("command", [
    {"argv": []},
    {"argv": ["scorer"]},
    {"argv": ["/opt/scorer\x00"]},
    {"argv": ["/opt/scorer"],
     "timeout_seconds": float("inf")},
])
def test_command_requires_safe_absolute_executable(command: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Metric.model_validate({"version": "v1", "kind": "command", "command": command})



def test_builtin_llm_quality_profile_cannot_minimize_without_custom_aggregation() -> None:
    with pytest.raises(ValidationError, match="maximize"):
        Metric(version="v1", direction="minimize", instruction="Judge.", criteria=[
            Criterion(id="correct", question="Correct?", weight=1),
        ])


def test_generic_example_accepts_empty_input_and_reference() -> None:
    from evaluator.contracts import Example

    example = Example(id="empty", input="", reference="")
    assert example.input == ""
    assert example.reference == ""
    with pytest.raises(ValidationError):
        Example(id="", input="", reference="")
