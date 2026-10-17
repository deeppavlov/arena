import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from evaluator.contracts import Criterion, Example, Metric
from evaluator.metrics import (
    CommandMetricError,
    aggregate_results,
    command_provenance,
    evaluate_answer,
)


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
    with pytest.raises(CommandMetricError, match="status 1"):
        evaluate_answer(metric, Example(id="x", input="source"), "answer")


def test_command_timeout_is_an_error_without_retry(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    metric = command_metric(tmp_path, f'''import pathlib
import time
with pathlib.Path({str(attempts)!r}).open("a") as output:
    output.write("attempt\\n")
time.sleep(10)
''')
    assert metric.command is not None
    metric = metric.model_copy(update={"command": metric.command.model_copy(
        update={"timeout_seconds": 0.25},
    )})
    with pytest.raises(CommandMetricError, match="timed out"):
        evaluate_answer(metric, Example(id="x", input="source"), "answer")
    assert attempts.read_text() == "attempt\n"


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


@pytest.mark.parametrize("termination", ["timeout", "failure", "success"])
def test_metric_terminates_spawned_work_before_return(tmp_path: Path, termination: str) -> None:
    marker = tmp_path / "still-running"
    ready = tmp_path / "started"
    child = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); "
        "time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).touch()"
    )
    ending = {
        "timeout": "time.sleep(10)",
        "failure": "raise SystemExit(2)",
        "success": "print('{\"score\": 1}')",
    }[termination]
    metric = command_metric(tmp_path, f'''import pathlib
import subprocess
import sys
import time
subprocess.Popen([sys.executable, "-c", {child!r}])
while not pathlib.Path({str(ready)!r}).exists():
    time.sleep(0.001)
{ending}
''')
    assert metric.command is not None
    metric = metric.model_copy(update={"command": metric.command.model_copy(
        update={"timeout_seconds": 0.25},
    )})
    if termination == "success":
        assert evaluate_answer(metric, Example(id="x", input=""), "")["score"] == 1
    else:
        with pytest.raises(CommandMetricError):
            evaluate_answer(metric, Example(id="x", input=""), "")
    assert ready.exists()
    status = Path("/proc") / ready.read_text() / "stat"
    if status.exists():
        assert status.read_bytes().rsplit(b")", 1)[1].split()[0] in {b"Z", b"X"}
    time.sleep(0.55)
    assert not marker.exists()


@pytest.mark.parametrize("termination", ["failure", "timeout", "invalid"])
def test_command_failures_retain_bounded_diagnostics(tmp_path: Path, termination: str) -> None:
    ending = {
        "failure": "raise SystemExit(3)",
        "timeout": "import time; time.sleep(10)",
        "invalid": "print('not-json')",
    }[termination]
    metric = command_metric(tmp_path, f'''import sys
sys.stderr.write("AUDIT_ONLY_DIAGNOSTIC" + "x" * 100000)
sys.stderr.flush()
print("stdout-before-failure", flush=True)
{ending}
''')
    assert metric.command is not None
    metric = metric.model_copy(update={"command": metric.command.model_copy(
        update={"timeout_seconds": 0.25},
    )})
    with pytest.raises(CommandMetricError) as raised:
        evaluate_answer(metric, Example(id="x", input=""), "")
    diagnostics = raised.value.diagnostics
    assert diagnostics["stdout"].startswith("stdout-before-failure")
    assert diagnostics["stderr"].startswith("AUDIT_ONLY_DIAGNOSTIC")
    assert len(diagnostics["stderr"].encode()) <= 65536
    assert diagnostics["stderr_truncated"] is True
    assert diagnostics["stdout_truncated"] is False
    assert diagnostics["timed_out"] is (termination == "timeout")
    assert diagnostics["returncode"] == {"failure": 3, "timeout": -9, "invalid": 0}[termination]


@pytest.mark.parametrize("score", [2**53, 2**53 + 1, -(2**53), 10**1000])
def test_integer_command_score_outside_exact_range_is_rejected(tmp_path: Path, score: int) -> None:
    metric = command_metric(tmp_path, f'print(\'{json.dumps({"score": score})}\')')
    with pytest.raises(ValueError, match="binary64"):
        evaluate_answer(metric, Example(id="x", input=""), "")


@pytest.mark.parametrize("score", [2**53, 2**53 + 1, -(2**53), 10**1000])
def test_integer_aggregate_score_outside_exact_range_is_rejected(score: int) -> None:
    with pytest.raises(ValueError, match="binary64"):
        aggregate_results(llm_metric(), [{"score": score}])


def test_command_diagnostic_invalid_utf8_does_not_hide_original_failure(tmp_path: Path) -> None:
    metric = command_metric(tmp_path, '''import sys
sys.stderr.buffer.write(b"failure: \\xff")
raise SystemExit(7)
''')
    with pytest.raises(CommandMetricError, match="status 7") as raised:
        evaluate_answer(metric, Example(id="x", input=""), "")
    assert raised.value.diagnostics["stderr"] == "failure: \ufffd"
    assert raised.value.diagnostics["stderr_truncated"] is False


def test_oversized_command_output_is_rejected_with_bounded_diagnostics(tmp_path: Path) -> None:
    metric = command_metric(tmp_path, '''import json
print(json.dumps({"score": 1, "statistics": {"large": "x" * (2 * 1024 * 1024)}}))
''')
    with pytest.raises(CommandMetricError, match="exceeds 1 MiB") as raised:
        evaluate_answer(metric, Example(id="x", input=""), "")
    assert len(raised.value.diagnostics["stdout"]) == 65536
    assert raised.value.diagnostics["stdout_truncated"] is True


def test_integer_command_aggregator_score_outside_exact_range_is_rejected(tmp_path: Path) -> None:
    scorer = command_metric(tmp_path, f'print(\'{json.dumps({"score": 2**53 + 1})}\')')
    metric = llm_metric().model_copy(update={"aggregation": scorer.command})
    with pytest.raises(CommandMetricError, match="binary64"):
        aggregate_results(metric, [{"score": 1}])
