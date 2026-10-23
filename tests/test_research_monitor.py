import json
import tracemalloc
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from evaluator.gateway import GatewaySettings, create_gateway
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
    assert not (config.state_dir / "records" / f"{run_id}.manifest.json").exists()


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
    record_path = config.state_dir / "records" / f"{run_id}.manifest.json"
    saved = record_path.read_text()
    record = json.loads(saved)
    assert record["params"]["candidate_id"] == "orchestrator"
    assert record["params"]["role"] == "orchestrator"
    assert record["metrics"]["orchestrator_seconds"] == 10
    assert record["metrics"]["orchestrator_tokens"] == 7
    assert record["trace_complete"] is True
    assert "secret" not in saved
    assert "proposal_index" not in record["params"]
    assert record["calls"] == []
    assert len(record["call_files"]) == 1
    assert Path(record["call_files"][0]).is_file()
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
    record = json.loads((config.state_dir / "records" / f"{run_id}.manifest.json").read_text())
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
    assert not (config.state_dir / "records" / f"{run_id}.manifest.json").exists()
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
    record = json.loads((config.state_dir / "records" / f"{run_id}.manifest.json").read_text())
    assert record["metrics"]["orchestrator_seconds"] == 13
    assert record["metrics"]["orchestrator_unreported_attempts"] == 1
    assert record["trace_complete"] is True


def test_monitor_memory_does_not_scale_with_all_trace_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    config = settings(tmp_path)
    run_id, _ = launch(config)
    payload = "x" * 256_000
    for index in range(64):
        add_usage(config.usage_dir, run_id, None)
        append_call(config.trace_dir, run_id, {
            "name": "model", "inputs": payload, "outputs": str(index),
            "start_time_ns": 1792404001000000000,
            "end_time_ns": 1792404002000000000,
        })
    monkeypatch.setattr(
        "integrations.docker.research_monitor._inspect_container", Mock(return_value=exited()),
    )
    monkeypatch.setattr("integrations.docker.research_monitor.export_record", Mock(return_value={
        "status": "exported", "run_id": "mlflow-run",
    }))
    tracemalloc.start()
    try:
        result = monitor_once(config)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result["status"] == "finished"
    assert peak < 8_000_000
    record_path = config.state_dir / "records" / f"{run_id}.manifest.json"
    assert record_path.stat().st_size < 32_000
    record = json.loads(record_path.read_text())
    assert record["trace_complete"] is True
    assert record["metrics"]["orchestrator_trace_attempts"] == 64
    assert record["calls"] == []
    assert len(record["call_files"]) == 64


def test_legacy_monolithic_record_is_preserved_without_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    config = settings(tmp_path)
    run_id, _ = launch(config)
    legacy = config.state_dir / "records" / f"{run_id}.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"legacy_record": true}')
    monkeypatch.setattr(
        "integrations.docker.research_monitor._inspect_container", Mock(return_value=exited()),
    )
    monkeypatch.setattr("integrations.docker.research_monitor.export_record", Mock(return_value={
        "status": "exported", "run_id": "mlflow-run",
    }))
    assert monitor_once(config)["status"] == "finished"
    assert legacy.read_text() == '{"legacy_record": true}'
    assert (config.state_dir / "records" / f"{run_id}.manifest.json").exists()


def test_terminal_manifest_does_not_reload_original_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    config = settings(tmp_path)
    run_id, _ = launch(config)
    append_call(config.trace_dir, run_id, {
        "name": "model", "start_time_ns": 1792404001000000000,
        "end_time_ns": 1792404002000000000,
    })
    monkeypatch.setattr(
        "integrations.docker.research_monitor._inspect_container", Mock(return_value=exited()),
    )
    monkeypatch.setattr("integrations.docker.research_monitor.export_record", Mock(return_value={
        "status": "exported", "run_id": "mlflow-run",
    }))
    monitor_once(config)
    read_text = Path.read_text

    def read_without_traces(path: Path, *args, **kwargs):
        assert config.trace_dir not in path.parents
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_without_traces)
    assert monitor_once(config)["status"] == "finished"


def test_flush_avoids_legacy_monolithic_monitor_queues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    config = settings(tmp_path)
    flush = Mock(return_value=[])
    monkeypatch.setattr("integrations.docker.research_monitor.flush_queue", flush)
    flush_pending(config)
    directories = [call.args[0] for call in flush.call_args_list]
    assert config.state_dir / "manifest-queues" / "shared" in directories
    assert config.state_dir / "manifest-queues" / "operator" in directories
    assert config.state_dir / "queues" / "shared" not in directories
    assert config.state_dir / "queues" / "operator" not in directories


def test_failed_shared_export_does_not_skip_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    config = settings(tmp_path)
    launch(config)
    monkeypatch.setattr(
        "integrations.docker.research_monitor._inspect_container", Mock(return_value=exited()),
    )
    export = Mock(side_effect=[
        RuntimeError("secret detail"), {"status": "exported", "run_id": "operator-run"},
    ])
    monkeypatch.setattr("integrations.docker.research_monitor.export_record", export)
    result = monitor_once(config)
    assert result["status"] == "finished"
    assert result["tracking"]["shared"] == {"status": "error", "error_type": "RuntimeError"}
    assert result["tracking"]["operator"]["status"] == "exported"
    assert "secret detail" not in json.dumps(result)
    assert export.call_count == 2


def test_real_exporter_resumes_from_snapshots_without_original_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared-tracking")
    monkeypatch.setenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", "http://operator-tracking")
    monkeypatch.setenv("MONITOR_TEST_SECRET", "opaque-synthetic-value")
    config = settings(tmp_path)
    run_id, _ = launch(config)
    add_usage(config.usage_dir, run_id, None)
    append_call(config.trace_dir, run_id, {
        "name": "model", "inputs": "prefix opaque-synthetic-value suffix",
        "outputs": "full answer", "start_time_ns": 1792404001000000000,
        "end_time_ns": 1792404002000000000,
    })
    monkeypatch.setattr(
        "integrations.docker.research_monitor._inspect_container", Mock(return_value=exited()),
    )
    publish = Mock(side_effect=[
        ConnectionError("unavailable"), ConnectionError("unavailable"),
        "shared-run", "operator-run",
    ])
    monkeypatch.setattr("evaluator.tracking._publish", publish)
    first = monitor_once(config)
    assert first["status"] == "finished"
    assert all(value["status"] == "tracking_pending" for value in first["tracking"].values())
    for call in publish.call_args_list:
        exported = call.args[0]
        snapshot = json.loads(Path(exported["call_files"][0]).read_text())
        assert snapshot["attributes"]["role"] == "orchestrator"
        assert snapshot["inputs"] == "prefix [REDACTED] suffix"
        assert snapshot["outputs"] == "full answer"
    for source in (config.trace_dir / run_id).glob("*.json"):
        source.unlink()
    expected = {
        "shared": {"status": "exported", "run_id": "shared-run"},
        "operator": {"status": "exported", "run_id": "operator-run"},
    }
    assert monitor_once(config)["tracking"] == expected
    assert monitor_once(config)["tracking"] == expected
    assert publish.call_count == 4


def test_failed_gateway_accounting_cannot_look_like_complete_empty_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from evaluator import usage as usage_store

    config = settings(tmp_path)
    run_id, capability = launch(config)
    original_save = usage_store._save
    calls = []

    def fail_result_save(path, value):
        if value["total_tokens"]:
            raise OSError("synthetic storage failure")
        original_save(path, value)

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "valid answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        })

    monkeypatch.setattr(usage_store, "_save", fail_result_save)
    app = create_gateway(
        GatewaySettings(
            base_url="http://internal-model/v1", model="fake",
            usage_dir=config.usage_dir, trace_dir=config.trace_dir,
        ),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/runs/{run_id}/{capability}/v1/chat/completions",
            json={"model": "fake", "messages": [{"role": "user", "content": "input"}]},
        )
    assert response.status_code == 503
    assert len(calls) == 1
    monkeypatch.setattr(
        "integrations.docker.research_monitor._inspect_container", Mock(return_value=exited()),
    )
    monkeypatch.setattr(
        "integrations.docker.research_monitor.export_record",
        Mock(return_value={"status": "exported", "run_id": "mlflow-run"}),
    )
    result = monitor_once(config)
    assert result["trace_complete"] is False
    record = json.loads((config.state_dir / "records" / f"{run_id}.manifest.json").read_text())
    assert record["metrics"]["orchestrator_attempts"] == 1
    assert record["metrics"]["orchestrator_unreported_attempts"] == 1
    assert record["metrics"]["orchestrator_trace_attempts"] == 0
    assert record["calls"] == []
