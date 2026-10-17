import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from evaluator.contracts import Criterion, Example, Metric
from evaluator.metrics import aggregate_results, command_provenance, evaluate_answer


def command_metric(tmp_path: Path, code: str) -> Metric:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "metric.py"
    source.write_text(code)
    executable = str(Path(sys.executable).absolute())
    return Metric.model_validate({
        "version": "test-v1", "kind": "command", "direction": "minimize",
        "command": {
            "argv": [executable, str(source)],
            "source_sha256": {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in (executable, str(source))
            },
        },
    })


def llm_metric() -> Metric:
    return Metric(version="test-v1", instruction="Judge.", criteria=[
        Criterion(id="correct", question="Correct?", weight=1),
    ])


def test_command_metric_scores_empty_answer_without_reference_or_model(tmp_path: Path) -> None:
    metric = command_metric(tmp_path, '''import json
import sys
record = json.load(sys.stdin)
assert record == {"example": {"id": "x", "input": "source", "reference": None}, "answer": ""}
print(json.dumps({"score": -2.5}))
''')
    model = Mock(side_effect=AssertionError("must not call model"))
    assert evaluate_answer(metric, Example(id="x", input="source"), "", model) == {
        "score": -2.5, "criteria": {},
    }
    model.assert_not_called()


def test_command_answer_is_json_data_not_a_shell_argument(tmp_path: Path) -> None:
    metric = command_metric(tmp_path, '''import json
import sys
record = json.load(sys.stdin)
print(json.dumps({"score": len(record["answer"])}))
''')
    answer = "$(echo injected); `echo injected`\n'\""
    result = evaluate_answer(metric, Example(id="x", input="source"), answer)
    assert result["score"] == len(answer)


@pytest.mark.parametrize("output", [
    "", "not-json", "0", '{"score": true}', '{"score": "0"}',
    '{"score": NaN}', '{"score": Infinity}', '{"score": -Infinity}',
    '{"score": 0, "extra": 1}', '{"score": null}',
])
def test_bad_command_output_is_an_error_not_zero(tmp_path: Path, output: str) -> None:
    metric = command_metric(tmp_path, f"print({output!r})\n")
    with pytest.raises(ValueError):
        evaluate_answer(metric, Example(id="x", input="source"), "answer")


def test_command_failure_does_not_convert_valid_stdout_to_score(tmp_path: Path) -> None:
    metric = command_metric(tmp_path, 'print(\'{"score": 0}\')\nraise SystemExit(1)\n')
    with pytest.raises(subprocess.CalledProcessError):
        evaluate_answer(metric, Example(id="x", input="source"), "answer")


def test_command_timeout_is_an_error_without_retry(tmp_path: Path, monkeypatch: Any) -> None:
    metric = command_metric(tmp_path, 'print(\'{"score": 0}\')\n')
    run = Mock(side_effect=subprocess.TimeoutExpired("scorer", 1))
    monkeypatch.setattr("evaluator.metrics.subprocess.run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        evaluate_answer(metric, Example(id="x", input="source"), "answer")
    assert run.call_count == 1


def test_command_provenance_contains_configuration_and_sources(tmp_path: Path) -> None:
    metric = command_metric(tmp_path, 'print(\'{"score": 0}\')\n')
    assert metric.command is not None
    assert command_provenance(metric) == {
        "command": metric.command.model_dump(), "aggregation": None,
    }
    assert command_provenance(llm_metric()) is None


def test_changed_implementation_fails_before_execution(tmp_path: Path) -> None:
    metric = command_metric(tmp_path, 'print(\'{"score": 0}\')\n')
    assert command_provenance(metric)
    (tmp_path / "metric.py").write_text('print(\'{"score": 100}\')\n')
    with pytest.raises(ValueError, match="hash"):
        evaluate_answer(metric, Example(id="x", input="source"), "answer")


def test_unpinned_source_argument_is_rejected(tmp_path: Path) -> None:
    metric = command_metric(tmp_path, 'print(\'{"score": 0}\')\n')
    data = metric.model_dump()
    del data["command"]["source_sha256"][str(tmp_path / "metric.py")]
    unpinned = Metric.model_validate(data)
    with pytest.raises(ValueError, match="pin"):
        command_provenance(unpinned)


def test_llm_metric_keeps_three_independent_votes() -> None:
    model = Mock(return_value=json.dumps({"score": 4, "reason": "Correct."}))
    result = evaluate_answer(
        llm_metric(), Example(id="x", input="source", reference="target"), "answer", model,
    )
    assert result["score"] == 4
    assert result["criteria"]["correct"]["scores"] == [4, 4, 4]
    assert model.call_count == 3


def test_llm_empty_answer_keeps_existing_candidate_failure() -> None:
    assert evaluate_answer(
        llm_metric(), Example(id="x", input="source", reference="target"), "",
    ) == {"score": 0.0, "criteria": {}, "failure": "candidate_failure"}


@pytest.mark.parametrize("reference", [None, "", " "])
def test_llm_metric_requires_reference(reference: str | None) -> None:
    with pytest.raises(ValueError, match="reference"):
        evaluate_answer(
            llm_metric(), Example(id="x", input="source", reference=reference), "answer",
        )


def test_llm_metric_requires_model_for_nonempty_answer() -> None:
    with pytest.raises(ValueError, match="model"):
        evaluate_answer(
            llm_metric(), Example(id="x", input="source", reference="target"), "answer",
        )



def test_default_aggregation_preserves_mean_without_intermediate_overflow() -> None:
    assert aggregate_results(llm_metric(), [{"score": 1e308}, {"score": 1e308}]) == 1e308
    assert aggregate_results(llm_metric(), [{"score": -3}, {"score": 1}]) == -1


@pytest.mark.parametrize("results", [[], [{"score": True}], [{"score": float("nan")}], [{}]])
def test_default_aggregation_requires_nonempty_finite_scores(results: Any) -> None:
    with pytest.raises((ValueError, KeyError)):
        aggregate_results(llm_metric(), results)


def test_dataset_metric_aggregates_sufficient_statistics_instead_of_example_mean(
    tmp_path: Path,
) -> None:
    scorer = command_metric(tmp_path / "scorer", '''import json
print(json.dumps({"score": 1.0, "statistics": {"true_positive": 1, "false_positive": 0}}))
''')
    aggregator = command_metric(tmp_path / "aggregate", '''import json
import sys
items = json.load(sys.stdin)["results"]
correct = sum(item["statistics"]["true_positive"] for item in items)
incorrect = sum(item["statistics"]["false_positive"] for item in items)
print(json.dumps({"score": correct / (correct + incorrect)}))
''')
    data = scorer.model_dump()
    data["aggregation"] = aggregator.command.model_dump() if aggregator.command else None
    metric = Metric.model_validate(data)
    result = evaluate_answer(metric, Example(id="x", input="source"), "answer")
    assert result["statistics"] == {"true_positive": 1, "false_positive": 0}
    other = {"score": 0.0, "statistics": {"true_positive": 0, "false_positive": 9}}
    assert aggregate_results(metric, [result, other]) == 0.1
    assert metric.command is not None
    assert metric.aggregation is not None
    assert command_provenance(metric) == {
        "command": metric.command.model_dump(), "aggregation": metric.aggregation.model_dump(),
    }
    (tmp_path / "aggregate" / "metric.py").write_text('print("changed")')
    with pytest.raises(ValueError, match="hash"):
        aggregate_results(metric, [result, other])


def test_llm_aggregation_can_transform_quality_to_a_minimized_loss(tmp_path: Path) -> None:
    aggregator = command_metric(tmp_path, '''import json
import sys
items = json.load(sys.stdin)["results"]
print(json.dumps({"score": 5 - sum(item["score"] for item in items) / len(items)}))
''')
    data = llm_metric().model_dump()
    data.update(direction="minimize", aggregation=aggregator.command)
    metric = Metric.model_validate(data)
    assert aggregate_results(metric, [{"score": 4}, {"score": 3}]) == 1.5


def test_command_statistics_must_be_finite_json(tmp_path: Path) -> None:
    metric = command_metric(tmp_path, '''print('{"score": 0, "statistics": {"x": NaN}}')''')
    with pytest.raises(ValueError):
        evaluate_answer(metric, Example(id="x", input="source"), "answer")


def test_relative_source_cannot_bypass_implementation_pinning(tmp_path: Path) -> None:
    metric = command_metric(tmp_path, 'print(\'{"score": 0}\')\n')
    data = metric.model_dump()
    data["command"]["argv"][1] = str((tmp_path / "metric.py").relative_to("/"))
    pinned = Metric.model_validate(data)
    assert evaluate_answer(pinned, Example(id="x", input="source"), "answer")["score"] == 0
    del data["command"]["source_sha256"][str(tmp_path / "metric.py")]
    unpinned = Metric.model_validate(data)
    with pytest.raises(ValueError, match="pin"):
        evaluate_answer(unpinned, Example(id="x", input="source"), "answer")


def test_command_metric_can_match_empty_target_and_empty_input(tmp_path: Path) -> None:
    metric = command_metric(tmp_path, '''import json
import sys
record = json.load(sys.stdin)
assert record["example"]["input"] == ""
print(json.dumps({"score": int(record["answer"] == record["example"]["reference"])}))
''')
    assert evaluate_answer(metric, Example(id="empty", input="", reference=""), "") == {
        "score": 1.0, "criteria": {},
    }
