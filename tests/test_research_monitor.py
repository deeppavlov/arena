import json
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from evaluator.trace_store import append_call
from evaluator.usage import add_usage, begin_request, create_usage, end_request, read_usage
from integrations.docker.research_monitor import Settings, flush_pending, monitor_once


def settings(tmp_path: Path) -> Settings:
    return Settings(
        active_file=tmp_path / "launch" / "active.json", usage_dir=tmp_path / "usage",
        trace_dir=tmp_path / "traces", state_dir=tmp_path / "monitor",
        params={"case_id": "synthetic", "campaign_id": "preparation", "model": "fake"},
        shared_queues=(tmp_path / "train-shared",),
        operator_queues=(tmp_path / "private-operator",),
    )


def launch(config: Settings, *, container: bool = True) -> tuple[str, str]:
    run_id = str(uuid4())
    capability = create_usage(config.usage_dir, run_id)
    config.active_file.parent.mkdir()
    config.active_file.write_text(json.dumps({
        "run_id": run_id, **({"container_id": "a" * 64} if container else {}),
    }))
    return run_id, capability


def exited(**changes: object) -> dict:
    return {
        "Status": "exited", "ExitCode": 0, "OOMKilled": False,
        "StartedAt": "2026-10-19T10:00:00Z", "FinishedAt": "2026-10-19T10:00:10Z",
        **changes,
    }


def test_before_launch_never_inspects_or_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    inspect = Mock()
    export = Mock()
    monkeypatch.setattr("integrations.docker.research_monitor._inspect_container", inspect)
    monkeypatch.setattr("integrations.docker.research_monitor.export_record", export)
    assert monitor_once(settings(tmp_path))["status"] == "not_started"
    inspect.assert_not_called()
    export.assert_not_called()


def test_launch_binding_waits_for_container_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = settings(tmp_path)
    launch(config, container=False)
    inspect = Mock()
    monkeypatch.setattr("integrations.docker.research_monitor._inspect_container", inspect)
    assert monitor_once(config)["status"] == "awaiting_container"
    inspect.assert_not_called()


def test_running_container_never_closes_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = settings(tmp_path)
    run_id, capability = launch(config)
    monkeypatch.setattr(
        "integrations.docker.research_monitor._inspect_container",
        Mock(return_value={"Status": "running"}),
    )
    assert monitor_once(config)["status"] == "running"
    begin_request(config.usage_dir, run_id, capability)
    end_request(config.usage_dir, run_id)
    assert not (config.state_dir / "records" / f"{run_id}.json").exists()


def test_exit_waits_for_inflight_calls_then_seals_and_exports_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    config = settings(tmp_path)
    run_id, capability = launch(config)
    begin_request(config.usage_dir, run_id, capability)
    inspect = Mock(return_value=exited())
    export = Mock(return_value={"status": "exported", "run_id": "mlflow-run"})
    monkeypatch.setattr("integrations.docker.research_monitor._inspect_container", inspect)
    monkeypatch.setattr("integrations.docker.research_monitor.export_record", export)
    assert monitor_once(config)["status"] == "draining"
    assert not json.loads((config.usage_dir / f"{run_id}.json").read_text())["closed"]
    add_usage(config.usage_dir, run_id, {
        "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
    })
    end_request(config.usage_dir, run_id)
    append_call(config.trace_dir, run_id, {
        "name": "model", "inputs": {"Authorization": "Bearer secret"}, "outputs": "answer",
        "attributes": {"attempt": 1}, "start_time_ns": 1792404001000000000,
        "end_time_ns": 1792404002000000000, "status": "OK", "span_type": "LLM",
    })
    assert monitor_once(config)["status"] == "finished"
    record_path = config.state_dir / "records" / f"{run_id}.json"
    saved = record_path.read_text()
    record = json.loads(saved)
    assert record["params"]["candidate_id"] == "orchestrator"
    assert record["params"]["role"] == "orchestrator"
    assert record["metrics"]["orchestrator_seconds"] == 10
    assert record["metrics"]["orchestrator_tokens"] == 7
    assert record["trace_complete"] is True
    assert "secret" not in saved
    assert "proposal_index" not in record["params"]
    assert record["calls"][0]["attributes"]["role"] == "orchestrator"
    assert export.call_count == 2
    assert {call.kwargs["operator"] for call in export.call_args_list} == {True, False}
    with pytest.raises(ValueError, match="closed"):
        begin_request(config.usage_dir, run_id, capability)
    assert monitor_once(config)["status"] == "finished"
    assert record_path.read_text() == saved
    assert inspect.call_count == 2
    assert read_usage(config.usage_dir, run_id)["attempts"] == 1


def test_failed_container_retains_failure_without_fake_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    config = settings(tmp_path)
    run_id, _ = launch(config)
    monkeypatch.setattr(
        "integrations.docker.research_monitor._inspect_container",
        Mock(return_value=exited(ExitCode=137, OOMKilled=True)),
    )
    monkeypatch.setattr("integrations.docker.research_monitor.export_record", Mock(return_value={
        "status": "tracking_pending",
    }))
    assert monitor_once(config)["status"] == "failed"
    record = json.loads((config.state_dir / "records" / f"{run_id}.json").read_text())
    assert record["status"] == "FAILED"
    assert record["exit_code"] == 137
    assert "score" not in record["metrics"]
    assert record["resources"]["orchestrator"]["usage"]["total_tokens"] == 0


@pytest.mark.parametrize("field,value", [("run_id", "../outside"), ("container_id", "--all")])
def test_rejects_untrusted_identifiers_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str,
):
    config = settings(tmp_path)
    launch(config)
    active = json.loads(config.active_file.read_text())
    active[field] = value
    config.active_file.write_text(json.dumps(active))
    inspect = Mock()
    monkeypatch.setattr("integrations.docker.research_monitor._inspect_container", inspect)
    with pytest.raises(ValueError):
        monitor_once(config)
    inspect.assert_not_called()


def test_flush_routes_are_explicit_and_failure_does_not_skip_other_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    config = settings(tmp_path)
    flush = Mock(side_effect=[RuntimeError("secret detail"), [], [], []])
    monkeypatch.setattr("integrations.docker.research_monitor.flush_queue", flush)
    result = flush_pending(config)
    assert len(result) == 4
    assert result[0]["status"] == "error"
    assert "secret detail" not in json.dumps(result)
    assert flush.call_count == 4
    for call in flush.call_args_list:
        if call.args[0] == tmp_path / "private-operator":
            assert call.kwargs["operator"] is True


def test_missing_usage_never_invents_zero_resources_and_still_flushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from integrations.docker.research_monitor import cycle

    config = settings(tmp_path)
    run_id, _ = launch(config)
    (config.usage_dir / f"{run_id}.json").unlink()
    monkeypatch.setattr(
        "integrations.docker.research_monitor._inspect_container", Mock(return_value=exited()),
    )
    flush = Mock(return_value=[])
    monkeypatch.setattr("integrations.docker.research_monitor.flush_pending", flush)
    result = cycle(config)
    assert result["monitor"]["status"] == "monitor_error"
    assert result["monitor"]["error_type"] == "FileNotFoundError"
    assert not (config.state_dir / "records" / f"{run_id}.json").exists()
    flush.assert_called_once_with(config)


def test_docker_binding_cannot_change_after_terminal_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    config = settings(tmp_path)
    launch(config)
    monkeypatch.setattr(
        "integrations.docker.research_monitor._inspect_container", Mock(return_value=exited()),
    )
    monkeypatch.setattr("integrations.docker.research_monitor.export_record", Mock(return_value={
        "status": "exported", "run_id": "mlflow-run",
    }))
    monitor_once(config)
    active = json.loads(config.active_file.read_text())
    active["container_id"] = "b" * 64
    config.active_file.write_text(json.dumps(active))
    with pytest.raises(ValueError, match="launch"):
        monitor_once(config)


def test_late_gateway_completion_is_included_in_wall_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    config = settings(tmp_path)
    run_id, _ = launch(config)
    add_usage(config.usage_dir, run_id, None)
    append_call(config.trace_dir, run_id, {
        "name": "model", "start_time_ns": 1792404009000000000,
        "end_time_ns": 1792404013000000000,
    })
    monkeypatch.setattr(
        "integrations.docker.research_monitor._inspect_container", Mock(return_value=exited()),
    )
    monkeypatch.setattr("integrations.docker.research_monitor.export_record", Mock(return_value={
        "status": "tracking_pending",
    }))
    monitor_once(config)
    record = json.loads((config.state_dir / "records" / f"{run_id}.json").read_text())
    assert record["metrics"]["orchestrator_seconds"] == 13
    assert record["metrics"]["orchestrator_unreported_attempts"] == 1
    assert record["trace_complete"] is True
