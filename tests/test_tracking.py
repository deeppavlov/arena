import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from evaluator.tracking import export_configured, export_record, flush_queue, sanitize_record


@pytest.fixture(autouse=True)
def clear_destinations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", raising=False)


def record(split: str = "train") -> dict:
    return {
        "split": split,
        "params": {"case_id": "synthetic", "campaign_id": "test", "candidate_id": "one"},
        "status": "FINISHED",
        "metrics": {"score": 0.7, "solver_tokens": 12, "unknown": None},
        "start_time_ns": 100,
        "end_time_ns": 200,
        "calls": [{"name": "model", "inputs": "question", "outputs": "answer"}],
        "examples": [{"id": "secret-example", "reference": "hidden"}],
    }


def save(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "record.json"
    path.write_text(json.dumps(value))
    return path


def test_redacts_secrets_in_keys_strings_and_scoped_urls() -> None:
    value = record()
    value["calls"] = [{
        "inputs": {
            "Authorization": "Bearer aaa",
            "api_key": "bbb",
            "model_base_url": "http://gateway/runs/run-id/capability-value/v1",
            "text": "Bearer ccc https://user:password@host?q=1&api_key=ddd",
            "trace": "POST http://gateway/runs/run-id/capability-value/v1/chat/completions",
        },
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }]
    result = sanitize_record(value)
    rendered = json.dumps(result)
    for secret in ("aaa", "bbb", "ccc", "password", "ddd", "capability-value"):
        assert secret not in rendered
    assert result["calls"][0]["usage"]["total_tokens"] == 7


def test_public_contains_only_score_and_known_candidate_metadata() -> None:
    value = record("public")
    value["error"] = "closed answer"
    value["params"]["example_id"] = "secret-example"
    value["params"]["solver_seconds"] = 12
    result = sanitize_record(value)
    assert result == {
        "split": "public",
        "params": {"case_id": "synthetic", "campaign_id": "test", "candidate_id": "one"},
        "status": "FINISHED",
        "metrics": {"score": 0.7},
    }


def test_private_requires_operator_and_keeps_closed_trace() -> None:
    with pytest.raises(ValueError, match="operator"):
        sanitize_record(record("private"))
    assert sanitize_record(record("private"), operator=True)["examples"]
    assert sanitize_record(record("public"), operator=True)["calls"]


@pytest.mark.parametrize("split,status", [("unknown", "FINISHED"), ("train", "RUNNING")])
def test_rejects_unfinished_or_unknown_records(split: str, status: str) -> None:
    value = record(split)
    value["status"] = status
    with pytest.raises(ValueError):
        sanitize_record(value)


def test_failed_export_is_durable_redacted_and_retry_does_not_repeat_solution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared-tracking")
    value = record()
    value["api_key"] = "private-key"
    path = save(tmp_path, value)
    publish = Mock(side_effect=ConnectionError("Bearer hidden-token"))
    monkeypatch.setattr("evaluator.tracking._publish", publish)
    result = export_record(path)
    assert result["status"] == "tracking_pending"
    queue = tmp_path / ".tracking"
    queued = list(queue.glob("*.json"))
    assert len(queued) == 1
    assert "private-key" not in queued[0].read_text()
    assert "hidden-token" not in queued[0].read_text()
    assert queued[0].stat().st_mode & 0o777 == 0o600
    publish.side_effect = None
    publish.return_value = "run-one"
    assert flush_queue(queue) == [{"status": "exported", "run_id": "run-one"}]
    assert path.read_text() == json.dumps(value)
    assert export_record(path) == {"status": "exported", "run_id": "run-one"}
    assert publish.call_count == 2


def test_operator_must_use_distinct_configured_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = save(tmp_path, record("private"))
    monkeypatch.delenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(ValueError, match="operator"):
        export_record(path, operator=True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking")
    monkeypatch.setenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", "http://tracking/")
    with pytest.raises(ValueError, match="distinct"):
        export_record(path, operator=True)


def test_shared_flush_rejects_operator_queue(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "entry.json").write_text(json.dumps({
        "operator": True, "status": "tracking_pending", "record": record("private"),
    }))
    with pytest.raises(ValueError, match="operator"):
        flush_queue(queue)


def test_uri_credentials_are_not_saved_in_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://user:secret@tracking")
    with pytest.raises(ValueError, match="credentials"):
        export_record(save(tmp_path, record()))
    assert not (tmp_path / ".tracking").exists()


@pytest.mark.parametrize(
    "score", [{"reference": "hidden"}, "secret", True, float("nan"), float("inf")],
)
def test_public_score_must_be_finite_numeric(score: object) -> None:
    value = record("public")
    value["metrics"]["score"] = score
    with pytest.raises(ValueError, match="score"):
        sanitize_record(value)


def test_redacts_basic_auth_and_nested_json_secret() -> None:
    from evaluator.tracking import redact

    value = 'Authorization: Basic dXNlcjpwYXNz; {"api_key": "api-secret-value"}'
    result = redact(value)
    assert "dXNlcjpwYXNz" not in result
    assert "api-secret-value" not in result


def test_redacts_escaped_known_secrets_and_exception_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluator.tracking import redact

    secret = "cap-secret"
    escaped = "".join(f"\\u{ord(char):04x}" for char in secret)
    assert escaped not in redact(escaped, secrets=(secret,))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared-tracking")
    monkeypatch.setenv("MODEL_API_KEY", "unstructured-api-secret")
    monkeypatch.setattr(
        "evaluator.tracking._publish", Mock(side_effect=RuntimeError("unstructured-api-secret")),
    )
    export_record(save(tmp_path, record()))
    queued = next((tmp_path / ".tracking").glob("*.json")).read_text()
    assert "unstructured-api-secret" not in queued


def test_retry_after_final_tag_failure_reuses_run_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from evaluator.tracking import _publish

    run = SimpleNamespace(info=SimpleNamespace(run_id="run-one"), data=SimpleNamespace(tags={}))
    client = Mock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="experiment-one")
    client.search_runs.side_effect = [[], [run]]
    client.search_traces.side_effect = [[], ["trace-one"], ["trace-one"]]
    client.set_tag.side_effect = [RuntimeError("temporary failure"), None]
    sdk = MagicMock()
    sdk.MlflowClient.return_value = client
    sdk.start_run.return_value.__enter__.return_value = run
    spans = Mock()
    monkeypatch.setattr("evaluator.tracking._spans", spans)
    monkeypatch.setattr("evaluator.tracking.importlib.import_module", Mock(return_value=sdk))
    with pytest.raises(RuntimeError):
        _publish(record(), "http://tracking", "export-one", False)
    assert _publish(record(), "http://tracking", "export-one", False) == "run-one"
    assert sdk.start_run.call_args_list[0].kwargs["tags"] == {"arena_export_id": "export-one"}
    assert sdk.start_run.call_args_list[1].kwargs["run_id"] == "run-one"
    assert spans.call_count == 1


def test_child_span_is_created_after_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from evaluator.tracking import _spans

    value = record()
    value["calls"] = [
        {"name": "child", "span_id": "child", "parent_id": "parent"},
        {"name": "parent", "span_id": "parent"},
    ]
    sdk = MagicMock()
    _spans(sdk, value, "run-one")
    assert [call.args[0] for call in sdk.start_span_no_context.call_args_list] == [
        "train_evaluation", "parent", "child",
    ]


@pytest.mark.parametrize("calls", [
    [{"name": "child", "parent_id": "missing"}],
    [{"name": "a", "span_id": "same"}, {"name": "b", "span_id": "same"}],
    [{"name": "a", "span_id": "a", "parent_id": "b"},
     {"name": "b", "span_id": "b", "parent_id": "a"}],
])
def test_rejects_invalid_span_hierarchy(calls: list[dict]) -> None:
    from unittest.mock import MagicMock

    from evaluator.tracking import _spans

    value = record()
    value["calls"] = calls
    sdk = MagicMock()
    with pytest.raises(ValueError):
        _spans(sdk, value, "run-one")
    sdk.start_span_no_context.assert_not_called()


def test_operator_export_restores_sdk_and_environment_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from evaluator.tracking import _publish

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared-tracking")
    run = SimpleNamespace(info=SimpleNamespace(run_id="run-one"))
    sdk = MagicMock()
    sdk.get_tracking_uri.return_value = "http://shared-tracking"
    sdk.set_tracking_uri.side_effect = (
        lambda uri: os.environ.__setitem__("MLFLOW_TRACKING_URI", uri)
    )
    sdk.start_run.return_value.__enter__.return_value = run
    sdk.MlflowClient.return_value.search_runs.return_value = []
    monkeypatch.setattr("evaluator.tracking.importlib.import_module", Mock(return_value=sdk))
    value = record()
    value["calls"] = []
    _publish(value, "http://operator-tracking", "export-one", True)
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://shared-tracking"
    assert sdk.set_tracking_uri.call_args.args == ("http://shared-tracking",)


def test_monitor_flush_continues_after_corrupt_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared-tracking")
    monkeypatch.setattr("evaluator.tracking._publish", Mock(side_effect=ConnectionError()))
    export_record(save(tmp_path, record()))
    queue = tmp_path / ".tracking"
    (queue / "000-corrupt.json").write_text("{")
    publish = Mock(return_value="valid-run")
    monkeypatch.setattr("evaluator.tracking._publish", publish)
    results = flush_queue(queue, continue_on_error=True)
    assert results[0]["status"] == "error"
    assert results[0]["error_type"] == "JSONDecodeError"
    assert results[1] == {"status": "exported", "run_id": "valid-run"}
    publish.assert_called_once()
    assert (queue / "000-corrupt.json").read_text() == "{"


@pytest.mark.parametrize("operator", [False, True])
def test_explicit_export_requires_configured_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operator: bool,
) -> None:
    publish = Mock()
    monkeypatch.setattr("evaluator.tracking._publish", publish)
    with pytest.raises(ValueError, match="explicitly"):
        export_record(save(tmp_path, record()), operator=operator)
    publish.assert_not_called()
    assert not (tmp_path / ".tracking").exists()


@pytest.mark.parametrize("split", ["train", "public", "private"])
def test_no_configured_destination_keeps_record_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str,
) -> None:
    path = save(tmp_path, record(split))
    original = path.read_bytes()
    publish = Mock()
    monkeypatch.setattr("evaluator.tracking._publish", publish)
    assert export_configured(path) == {}
    publish.assert_not_called()
    assert path.read_bytes() == original
    assert not (tmp_path / ".tracking").exists()


@pytest.mark.parametrize("split", ["train", "public"])
def test_shared_destination_does_not_require_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared-tracking")
    publish = Mock(return_value="shared-run")
    monkeypatch.setattr("evaluator.tracking._publish", publish)
    assert export_configured(save(tmp_path, record(split))) == {
        "shared": {"status": "exported", "run_id": "shared-run"},
    }
    published, uri, _, operator = publish.call_args.args
    assert uri == "http://shared-tracking"
    assert operator is False
    assert bool(published.get("calls")) is (split == "train")
    assert len(list((tmp_path / ".tracking" / "shared").glob("*.json"))) == 1


@pytest.mark.parametrize("split", ["train", "public", "private"])
def test_operator_destination_does_not_require_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str,
) -> None:
    monkeypatch.setenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", "http://operator-tracking")
    publish = Mock(return_value="operator-run")
    monkeypatch.setattr("evaluator.tracking._publish", publish)
    assert export_configured(save(tmp_path, record(split))) == {
        "operator": {"status": "exported", "run_id": "operator-run"},
    }
    published, uri, _, operator = publish.call_args.args
    assert uri == "http://operator-tracking"
    assert operator is True
    assert published["calls"] == record(split)["calls"]
    assert len(list((tmp_path / ".tracking" / "operator").glob("*.json"))) == 1


def test_private_record_never_uses_shared_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared-tracking")
    publish = Mock()
    monkeypatch.setattr("evaluator.tracking._publish", publish)
    with pytest.raises(ValueError, match="operator"):
        export_configured(save(tmp_path, record("private")))
    publish.assert_not_called()
    assert not (tmp_path / ".tracking").exists()


def test_configured_destinations_use_separate_filtered_durable_queues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared-tracking")
    monkeypatch.setenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", "http://operator-tracking")
    publish = Mock(side_effect=[ConnectionError("unavailable"), "operator-run"])
    monkeypatch.setattr("evaluator.tracking._publish", publish)
    queue = tmp_path / "queue"
    result = export_configured(save(tmp_path, record("public")), queue)
    assert result == {
        "shared": {"status": "tracking_pending"},
        "operator": {"status": "exported", "run_id": "operator-run"},
    }
    shared = json.loads(next((queue / "shared").glob("*.json")).read_text())
    operator = json.loads(next((queue / "operator").glob("*.json")).read_text())
    assert "calls" not in shared["record"]
    assert "examples" not in shared["record"]
    assert operator["record"]["examples"]



def test_private_export_with_both_destinations_uses_operator_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared-tracking")
    monkeypatch.setenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", "http://operator-tracking")
    publish = Mock(return_value="operator-run")
    monkeypatch.setattr("evaluator.tracking._publish", publish)
    assert export_configured(save(tmp_path, record("private"))) == {
        "operator": {"status": "exported", "run_id": "operator-run"},
    }
    publish.assert_called_once()
    assert publish.call_args.args[1] == "http://operator-tracking"
    assert publish.call_args.args[3] is True
    assert not (tmp_path / ".tracking" / "shared").exists()



def test_failed_operator_export_restores_absent_shared_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    from unittest.mock import MagicMock

    from evaluator.tracking import _publish

    sdk = MagicMock()
    sdk.get_tracking_uri.return_value = "file:///tmp/mlruns"
    sdk.set_tracking_uri.side_effect = (
        lambda uri: os.environ.__setitem__("MLFLOW_TRACKING_URI", uri)
    )
    sdk.MlflowClient.side_effect = ConnectionError("unavailable")
    monkeypatch.setattr("evaluator.tracking.importlib.import_module", Mock(return_value=sdk))
    with pytest.raises(ConnectionError):
        _publish(record(), "http://operator-tracking", "export-one", True)
    assert "MLFLOW_TRACKING_URI" not in os.environ
    assert sdk.set_tracking_uri.call_args.args == ("file:///tmp/mlruns",)
