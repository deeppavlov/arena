import hashlib
import json
import sys
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from uuid import UUID

import pytest

from evaluator.contracts import Example, Metric
from evaluator.metrics import (
    CommandMetricError,
    MetricModelAccess,
    aggregate_results,
    evaluate_answer,
)
from evaluator.reliability import empty_usage, request_timeout
from evaluator.usage import add_usage, begin_request, end_request, ensure_open, read_usage


def metric_command(tmp_path: Path, code: str, *, uses_model: bool = True) -> Metric:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "metric.py"
    source.write_text(code)
    executable = str(Path(sys.executable).absolute())
    return Metric.model_validate({
        "version": "custom-v1", "kind": "command",
        "command": {
            "argv": [executable, str(source)],
            "source_sha256": {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in (executable, str(source))
            },
            "uses_model": uses_model,
        },
    })


MODEL_CALL = '''import json
import sys
import urllib.request
record = json.load(sys.stdin)
request = urllib.request.Request(record["model_base_url"] + "/chat/completions",
    data=json.dumps({"model": record["model"], "messages": [
        {"role": "user", "content": "judge"}]}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=record["model_request_timeout"]) as response:
    result = json.load(response)
'''


@pytest.fixture
def gateway(tmp_path: Path) -> Iterator[tuple[MetricModelAccess, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []
    directory = tmp_path / "usage"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            _, _, run_id, capability, _, _, _ = self.path.split("/")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"run_id": run_id, "capability": capability, "body": body})
            begin_request(directory, run_id, capability)
            add_usage(directory, run_id, None)
            add_usage(directory, run_id, {
                "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
            })
            end_request(directory, run_id)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"answer": "good"}).encode())

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield MetricModelAccess(
            model="internal", base_url=f"http://127.0.0.1:{server.server_port}/v1",
            usage_dir=directory, model_timeout=3, retry_delay=0,
        ), calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("update", [
    {"model": " "}, {"base_url": "https://host/not-v1"},
    {"base_url": "https://name:secret@host/v1"},
    {"base_url": "https://host:bad/v1"}, {"base_url": "https://host:65536/v1"},
    {"base_url": "https://host:0/v1"}, {"base_url": "https://host/v1?secret=1"},
    {"base_url": "https://host/v1#fragment"},
    {"model_timeout": 0}, {"model_timeout": float("inf")}, {"retry_delay": -1},
])
def test_model_access_rejects_invalid_configuration(tmp_path: Path, update: dict[str, Any]) -> None:
    arguments = {"model": "internal", "base_url": "http://gateway/v1", "usage_dir": tmp_path}
    with pytest.raises(ValueError):
        MetricModelAccess(**(arguments | update))


def test_model_command_requires_scoped_access_before_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    metric = metric_command(tmp_path, f'import pathlib; pathlib.Path({str(marker)!r}).touch()')
    with pytest.raises(ValueError, match="model access"):
        evaluate_answer(metric, Example(id="x", input=""), "")
    assert not marker.exists()


def test_program_command_gets_no_model_credentials(
    tmp_path: Path, gateway: tuple[MetricModelAccess, list[dict[str, Any]]],
) -> None:
    access, calls = gateway
    metric = metric_command(tmp_path, '''import json
import sys
record = json.load(sys.stdin)
assert set(record) == {"example", "answer"}
print('{"score": 1}')
''', uses_model=False)
    usage = empty_usage()
    assert evaluate_answer(
        metric, Example(id="x", input=""), "", model_access=access, usage=usage,
    )["score"] == 1
    assert calls == []
    assert access.run_ids == []
    assert usage == empty_usage()
    assert not access.usage_dir.exists()


def test_custom_model_command_uses_trusted_ledger_and_closes_scope(
    tmp_path: Path, gateway: tuple[MetricModelAccess, list[dict[str, Any]]],
) -> None:
    access, calls = gateway
    metric = metric_command(tmp_path, MODEL_CALL + f'''
assert record["model_request_timeout"] == {request_timeout(3, retry_delay=0)!r}
print(json.dumps({{"score": 1, "statistics": {{"total_tokens": 999999,
    "accidental_url": record["model_base_url"]}}}}))
''')
    usage = empty_usage()
    result = evaluate_answer(
        metric, Example(id="x", input=""), "answer", model_access=access, usage=usage,
    )
    assert result["score"] == 1
    assert len(calls) == 1
    assert calls[0]["body"]["model"] == "internal"
    run_id = calls[0]["run_id"]
    assert access.run_ids == [run_id]
    assert str(UUID(run_id)) == run_id
    assert usage == read_usage(access.usage_dir, run_id) == {
        "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
        "attempts": 2, "unreported_attempts": 1,
    }
    assert calls[0]["capability"] not in json.dumps(result)
    with pytest.raises(ValueError, match="closed"):
        ensure_open(access.usage_dir, run_id)


@pytest.mark.parametrize("ending", [
    "raise SystemExit(5)", "print(record['model_base_url'])",
    "import time; time.sleep(10)",
])
def test_failed_custom_judge_retains_usage_and_redacts_capability(
    tmp_path: Path, gateway: tuple[MetricModelAccess, list[dict[str, Any]]], ending: str,
) -> None:
    access, calls = gateway
    metric = metric_command(tmp_path, MODEL_CALL + '''
print(record["model_base_url"], file=sys.stderr, flush=True)
''' + ending)
    assert metric.command is not None
    metric = metric.model_copy(update={"command": metric.command.model_copy(
        update={"timeout_seconds": 0.5},
    )})
    usage = empty_usage()
    with pytest.raises(CommandMetricError) as raised:
        evaluate_answer(
            metric, Example(id="x", input=""), "", model_access=access, usage=usage,
        )
    assert usage["total_tokens"] == 10
    assert usage["attempts"] == 2
    assert len(calls) == 1
    assert raised.value.diagnostics["model_run_id"] == calls[0]["run_id"]
    assert access.run_ids == [calls[0]["run_id"]]
    assert raised.value.diagnostics["model_usage_complete"] is True
    assert calls[0]["capability"] not in json.dumps(raised.value.diagnostics) + str(raised.value)
    with pytest.raises(ValueError, match="closed"):
        ensure_open(access.usage_dir, calls[0]["run_id"])


def test_custom_aggregation_gets_its_own_scope_and_adds_to_judge_usage(
    tmp_path: Path, gateway: tuple[MetricModelAccess, list[dict[str, Any]]],
) -> None:
    access, calls = gateway
    scorer = metric_command(tmp_path / "scorer", MODEL_CALL + 'print(\'{"statistics": {}}\')')
    aggregator = metric_command(tmp_path / "aggregate", MODEL_CALL + '''
assert record["results"] == [{"statistics": {}, "criteria": {}}]
print('{"score": 0.75}')
''')
    metric = scorer.model_copy(update={"aggregation": aggregator.command})
    usage = empty_usage()
    result = evaluate_answer(
        metric, Example(id="x", input=""), "", model_access=access, usage=usage,
    )
    assert aggregate_results(metric, [result], model_access=access, usage=usage) == 0.75
    assert usage["total_tokens"] == 20
    assert usage["attempts"] == 4
    assert len({call["run_id"] for call in calls}) == 2
    assert len({call["capability"] for call in calls}) == 2
    assert access.run_ids == [call["run_id"] for call in calls]


def test_inflight_model_usage_fails_explicitly_and_retains_reconciliation_id(
    tmp_path: Path, gateway: tuple[MetricModelAccess, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access, calls = gateway
    metric = metric_command(tmp_path, MODEL_CALL + 'print(\'{"score": 1}\')')
    original = __import__("evaluator.metrics", fromlist=["seal_usage"]).seal_usage

    def inflight(directory: Path, run_id: str) -> dict[str, int]:
        begin_request(directory, run_id, calls[0]["capability"])
        return original(directory, run_id)

    monkeypatch.setattr("evaluator.metrics.seal_usage", inflight)
    usage = empty_usage()
    with pytest.raises(CommandMetricError, match="accounting incomplete") as raised:
        evaluate_answer(
            metric, Example(id="x", input=""), "", model_access=access, usage=usage,
        )
    assert raised.value.diagnostics["model_run_id"] == calls[0]["run_id"]
    assert raised.value.diagnostics["model_usage_complete"] is False
    assert usage["total_tokens"] == 10
    add_usage(access.usage_dir, calls[0]["run_id"], {
        "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
    })
    end_request(access.usage_dir, calls[0]["run_id"])
    assert read_usage(access.usage_dir, calls[0]["run_id"])["total_tokens"] == 13


def test_failed_command_start_still_closes_model_scope(
    tmp_path: Path, gateway: tuple[MetricModelAccess, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access, calls = gateway
    metric = metric_command(tmp_path, 'print(\'{"score": 1}\')')

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise OSError("could not start")

    monkeypatch.setattr("evaluator.metrics.subprocess.Popen", unavailable)
    usage = empty_usage()
    with pytest.raises(CommandMetricError, match="could not start") as raised:
        evaluate_answer(
            metric, Example(id="x", input=""), "", model_access=access, usage=usage,
        )
    assert calls == []
    assert usage == empty_usage()
    assert raised.value.diagnostics["model_usage_complete"] is True
    assert access.run_ids == [raised.value.diagnostics["model_run_id"]]
    with pytest.raises(ValueError, match="closed"):
        ensure_open(access.usage_dir, raised.value.diagnostics["model_run_id"])


def test_unreadable_ledger_never_becomes_successful_zero_usage(
    tmp_path: Path, gateway: tuple[MetricModelAccess, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access, calls = gateway
    metric = metric_command(tmp_path, MODEL_CALL + 'print(\'{"score": 1}\')')

    def unavailable(*args: Any) -> dict[str, int]:
        raise OSError("ledger unavailable")

    monkeypatch.setattr("evaluator.metrics.seal_usage", unavailable)
    monkeypatch.setattr("evaluator.metrics.read_usage", unavailable)
    with pytest.raises(CommandMetricError, match="accounting incomplete") as raised:
        evaluate_answer(
            metric, Example(id="x", input=""), "", model_access=access, usage=empty_usage(),
        )
    assert raised.value.diagnostics == {
        "model_run_id": calls[0]["run_id"], "model_usage_complete": False,
    }


def test_metric_imports_do_not_mutate_frozen_package(tmp_path: Path) -> None:
    metric = metric_command(tmp_path, '''import helper
print('{"score": 1}')
''', uses_model=False)
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    assert evaluate_answer(metric, Example(id="x", input=""), "")["score"] == 1
    assert not (tmp_path / "__pycache__").exists()


@pytest.mark.parametrize("encoding", ["escaped", "utf-16"])
def test_encoded_json_cannot_persist_the_model_capability(
    tmp_path: Path, gateway: tuple[MetricModelAccess, list[dict[str, Any]]], encoding: str,
) -> None:
    access, calls = gateway
    metric = metric_command(tmp_path, MODEL_CALL + """
value = json.dumps({"score": 1, "statistics": {"url": record["model_base_url"]}})
""" + (
        r"value = value.replace(record['model_base_url'], ''.join('\\u%04x' % ord(c) "
        r"for c in record['model_base_url']))" + "\nprint(value)"
        if encoding == "escaped" else "sys.stdout.buffer.write(value.encode('utf-16'))"
    ))
    result = evaluate_answer(
        metric, Example(id="x", input=""), "", model_access=access, usage=empty_usage(),
    )
    assert calls[0]["capability"] not in json.dumps(result)


def test_non_json_diagnostics_redact_unicode_escaped_capability(
    tmp_path: Path, gateway: tuple[MetricModelAccess, list[dict[str, Any]]],
) -> None:
    access, calls = gateway
    metric = metric_command(tmp_path, MODEL_CALL +
        r"escaped = ''.join('\\u%04x' % ord(c) for c in record['model_base_url'])" +
        "\nprint('audit: ' + escaped, file=sys.stderr)\nraise SystemExit(1)")
    with pytest.raises(CommandMetricError) as raised:
        evaluate_answer(metric, Example(id="x", input=""), "", model_access=access)
    decoded = raised.value.diagnostics["stderr"].encode().decode("unicode_escape")
    assert calls[0]["capability"] not in decoded
