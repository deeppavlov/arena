import copy
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, PropertyMock

import pytest

from evaluator import tracking


def record(calls=None):
    return {
        "split": "train", "status": "FINISHED", "metrics": {"score": 1},
        "params": {"case_id": "synthetic", "campaign_id": "test", "candidate_id": "one"},
        "start_time_ns": 100, "end_time_ns": 200,
        "calls": [] if calls is None else calls,
    }


@pytest.fixture
def publisher(monkeypatch):
    artifacts = {}
    traces = {}
    run = SimpleNamespace(info=SimpleNamespace(run_id="run-one"),
                          data=SimpleNamespace(tags={}))
    sdk = MagicMock()
    client = sdk.MlflowClient.return_value
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="experiment-one")
    client.search_runs.return_value = [run]
    sdk.start_run.return_value.__enter__.return_value = run

    def set_tag(run_id, key, value, **kwargs):
        run.data.tags[key] = value

    client.set_tag.side_effect = set_tag

    def search_traces(**kwargs):
        query = kwargs.get("filter_string")
        if query is None:
            return [SimpleNamespace(data=SimpleNamespace(spans=[]))]
        result = traces.get(query.split("'")[1])
        return [] if result is None else [SimpleNamespace(data=SimpleNamespace(spans=[result]))]

    client.search_traces.side_effect = search_traces

    def start_span(name, **kwargs):
        def end(**ending):
            call_id = kwargs.get("tags", {}).get("arena_call_id", "legacy")
            traces[call_id] = SimpleNamespace(
                name=name, inputs=kwargs.get("inputs"), outputs=ending.get("outputs"),
                attributes=kwargs.get("attributes", {}),
            )
        return SimpleNamespace(end=Mock(side_effect=end))

    sdk.start_span_no_context.side_effect = start_span

    def log_artifact(mlflow, target, run_id, value, path):
        artifacts[path] = copy.deepcopy(value)

    monkeypatch.setattr(tracking.importlib, "import_module", Mock(return_value=sdk))
    monkeypatch.setattr(tracking, "log_json_artifact", Mock(side_effect=log_artifact))
    read_artifact = Mock(side_effect=lambda client, run_id, path: artifacts[path])
    monkeypatch.setattr(tracking, "read_json_artifact", read_artifact)
    return SimpleNamespace(sdk=sdk, client=client, artifacts=artifacts, traces=traces,
                           run=run, read_artifact=read_artifact)


def test_large_inline_calls_ignore_existing_partial_trace_and_preserve_parentage(publisher):
    calls = [
        {"name": "parent", "span_id": "p", "inputs": "a" * 600000,
         "start_time_ns": 110, "end_time_ns": 180, "attributes": {"role": "solver"}},
        {"name": "child", "span_id": "c", "parent_id": "p", "outputs": "b" * 600000,
         "start_time_ns": 120, "end_time_ns": 170},
    ]
    assert tracking._publish(record(calls), "http://tracking", "export-one", False) == "run-one"
    assert publisher.sdk.start_span_no_context.call_count == 2
    starts = publisher.sdk.start_span_no_context.call_args_list
    assert starts[0].kwargs["attributes"]["source_span_id"] == "p"
    assert starts[1].kwargs["attributes"]["source_parent_id"] == "p"
    assert starts[1].kwargs["start_time_ns"] == 120
    assert publisher.artifacts["record.json"]["calls"] == calls
    assert publisher.run.data.tags["arena_export_complete"] == "true"


def test_many_small_calls_are_published_as_separate_traces(publisher):
    calls = [{"name": f"model-{index}", "inputs": "small"} for index in range(65)]
    assert not tracking._large_json(record(calls))
    tracking._publish(record(calls), "http://tracking", "export-one", False)
    assert publisher.sdk.start_span_no_context.call_count == 65
    assert len(publisher.traces) == 65
    assert publisher.artifacts["record.json"]["calls"] == calls


def test_oversized_call_uses_readable_full_artifact(publisher):
    call = {"name": "model", "span_id": "source", "inputs": "a" * 2000000,
            "outputs": {"text": "complete"}, "attributes": {"role": "solver"},
            "start_time_ns": 110, "end_time_ns": 190}
    tracking._publish(record([call]), "http://tracking", "export-one", False)
    start = publisher.sdk.start_span_no_context.call_args
    reference = start.kwargs["inputs"]
    assert set(reference) == {"arena_artifact"}
    assert publisher.artifacts[reference["arena_artifact"]] == call
    assert start.kwargs["attributes"]["source_span_id"] == "source"
    assert start.kwargs["attributes"]["role"] == "solver"
    assert publisher.read_artifact.call_count >= 1


def test_reference_retry_never_fetches_legacy_giant_trace(tmp_path, publisher):
    call = {"name": "model", "inputs": "x" * 2000000}
    path = tmp_path / "call.json"
    path.write_text(json.dumps(call))
    value = record()
    value["call_files"] = [str(path)]
    legacy_trace = Mock()
    type(legacy_trace).data = PropertyMock(side_effect=AssertionError("Legacy giant trace fetched"))
    search = publisher.client.search_traces.side_effect

    def search_traces(**kwargs):
        if "tag.arena_payload_format = 'json-artifact-v1'" not in kwargs["filter_string"]:
            return [legacy_trace]
        return search(**kwargs)

    publisher.client.search_traces.side_effect = search_traces
    for _ in range(2):
        tracking._publish_call_files(
            publisher.sdk, publisher.client, value, "experiment-one", "run-one", "export-one", (),
        )
    assert publisher.sdk.start_span_no_context.call_count == 1
    assert publisher.sdk.start_span_no_context.call_args.kwargs["tags"]["arena_payload_format"] \
        == "json-artifact-v1"


def test_large_root_and_params_are_referenced_without_losing_values(publisher):
    value = record([{"name": "small", "inputs": "question"}])
    value["params"]["large/prompt~text"] = "p" * 2000000
    value["examples"] = [{"input": "e" * 2000000}]
    tracking._publish(value, "http://tracking", "export-one", False)
    parameter = publisher.sdk.log_params.call_args.args[0]["large/prompt~text"]
    assert parameter == "artifact://record.json#/params/large~1prompt~0text"
    assert publisher.artifacts["record.json"] == value
    assert publisher.sdk.start_span_no_context.call_args.args[0] == "small"
    assert len(json.dumps(publisher.sdk.start_span_no_context.call_args.kwargs)) < 10000


def test_legacy_large_param_is_not_replaced_with_immutable_reference(publisher):
    value = record()
    value["params"]["large"] = "x" * 7000
    publisher.run.data.params = {"large": "x" * 6000}
    tracking._publish(value, "http://tracking", "export-one", False)
    assert "large" not in publisher.sdk.log_params.call_args.args[0]
    assert publisher.artifacts["record.json"]["params"]["large"] == value["params"]["large"]


def test_lost_final_ack_resumes_same_run_without_duplicate_calls(publisher):
    value = record([{"name": "model", "inputs": "x" * 2000000}])
    original = publisher.client.set_tag.side_effect
    publisher.client.set_tag.side_effect = ConnectionError("lost ack")
    with pytest.raises(ConnectionError):
        tracking._publish(value, "http://tracking", "export-one", False)
    publisher.client.set_tag.side_effect = original
    assert tracking._publish(value, "http://tracking", "export-one", False) == "run-one"
    assert publisher.sdk.start_span_no_context.call_count == 1
    assert all(call.kwargs["run_id"] == "run-one"
               for call in publisher.sdk.start_run.call_args_list)


def test_missing_referenced_payload_cannot_mark_export_complete(publisher, monkeypatch):
    monkeypatch.setattr(tracking, "read_json_artifact", Mock(side_effect=FileNotFoundError("part")))
    with pytest.raises((FileNotFoundError, RuntimeError)):
        tracking._publish(record([{"name": "model", "inputs": "x" * 2000000}]),
                          "http://tracking", "export-one", False)
    publisher.client.set_tag.assert_not_called()


def test_matching_name_without_correct_payload_does_not_ack(publisher):
    call = {"name": "model", "inputs": "expected", "outputs": "complete"}
    payload = tracking._call_payload(call, "run-one", "calls/one.json")
    publisher.client.search_traces.side_effect = None
    publisher.client.search_traces.return_value = [SimpleNamespace(data=SimpleNamespace(spans=[
        SimpleNamespace(name="model", inputs="wrong", outputs="complete",
                        attributes=payload["attributes"]),
    ]))]
    assert not tracking._call_trace_available(
        publisher.client, {}, "model", expected={"payload": payload},
    )


@pytest.mark.parametrize(("original", "received"), [
    ('{"answer": "complete"}', {"answer": "complete"}),
    ('["complete", 1]', ["complete", 1]),
    ('"complete"', "complete"),
    ('"[1, 2]"', "[1, 2]"),
    ("null", "null"),
    ("true", "true"),
    ("0", "0"),
    ("ordinary", "ordinary"),
    ("", ""),
    (None, None),
])
def test_ack_preserves_original_artifact_when_mlflow_normalizes_json_strings(
    publisher, original, received,
):
    call = {"name": "model", "inputs": original, "outputs": original,
            "attributes": {"detail": original}}
    payload = tracking._call_payload(call, "run-one", "calls/one.json")
    publisher.artifacts["calls/one.json"] = copy.deepcopy(call)
    publisher.client.search_traces.side_effect = None
    publisher.client.search_traces.return_value = [SimpleNamespace(data=SimpleNamespace(spans=[
        SimpleNamespace(name="model", inputs=received, outputs=received,
                        attributes={**payload["attributes"], "detail": received}),
    ]))]
    assert tracking._call_trace_available(
        publisher.client, {}, "model", expected={
            "payload": payload, "call": call, "run_id": "run-one",
            "artifact_path": "calls/one.json",
        },
    )
    assert publisher.artifacts["calls/one.json"] == call
    if original != received:
        publisher.read_artifact.assert_called_once_with(
            publisher.client, "run-one", "calls/one.json",
        )


def test_normalized_span_without_exact_original_artifact_is_not_acknowledged(publisher):
    call = {"name": "model", "inputs": '{"value": 1}', "outputs": '"complete"'}
    payload = tracking._call_payload(call, "run-one", "calls/one.json")
    publisher.artifacts["calls/one.json"] = {**call, "inputs": {"value": 1}}
    publisher.client.search_traces.side_effect = None
    publisher.client.search_traces.return_value = [SimpleNamespace(data=SimpleNamespace(spans=[
        SimpleNamespace(name="model", inputs={"value": 1}, outputs="complete",
                        attributes=payload["attributes"]),
    ]))]
    assert not tracking._call_trace_available(
        publisher.client, {}, "model", expected={
            "payload": payload, "call": call, "run_id": "run-one",
            "artifact_path": "calls/one.json",
        },
    )
    publisher.read_artifact.assert_called_once()


@pytest.mark.parametrize("field", ["inputs", "outputs", "attributes"])
@pytest.mark.parametrize(("original", "received"), [
    ('{"value": 1}', {"value": True}),
    ('[true]', [1]),
    ('{"value": [1.0]}', {"value": [1]}),
])
def test_normalized_span_rejects_changed_nested_json_types(
    publisher, field, original, received,
):
    call = {"name": "model", "inputs": "question", "outputs": "answer", "attributes": {}}
    if field == "attributes":
        call["attributes"] = {"detail": original}
    else:
        call[field] = original
    payload = tracking._call_payload(call, "run-one", "calls/one.json")
    actual = {**payload, "attributes": {**payload["attributes"]}}
    if field == "attributes":
        actual["attributes"]["detail"] = received
    else:
        actual[field] = received
    publisher.artifacts["calls/one.json"] = copy.deepcopy(call)
    publisher.client.search_traces.side_effect = None
    publisher.client.search_traces.return_value = [SimpleNamespace(data=SimpleNamespace(
        spans=[SimpleNamespace(**actual)],
    ))]
    assert not tracking._call_trace_available(
        publisher.client, {}, "model", expected={
            "payload": payload, "call": call, "run_id": "run-one",
            "artifact_path": "calls/one.json",
        },
    )


@pytest.mark.parametrize(("original", "received"), [(1, True), (1.0, 1), (True, 1)])
def test_original_artifact_rejects_changed_nested_json_types(publisher, original, received):
    call = {"name": "model", "inputs": {"value": original}, "outputs": '{"done": true}'}
    payload = tracking._call_payload(call, "run-one", "calls/one.json")
    publisher.artifacts["calls/one.json"] = {**call, "inputs": {"value": received}}
    publisher.client.search_traces.side_effect = None
    publisher.client.search_traces.return_value = [SimpleNamespace(data=SimpleNamespace(spans=[
        SimpleNamespace(name="model", inputs=call["inputs"], outputs={"done": True},
                        attributes=payload["attributes"]),
    ]))]
    assert not tracking._call_trace_available(
        publisher.client, {}, "model", expected={
            "payload": payload, "call": call, "run_id": "run-one",
            "artifact_path": "calls/one.json",
        },
    )
    publisher.read_artifact.assert_called_once()


@pytest.mark.parametrize("received", ['{"value":1}', '{"value": 1}'])
def test_json_string_ack_accepts_exact_legacy_value_without_semantic_coercion(
    publisher, received,
):
    call = {"name": "model", "inputs": '{"value":1}'}
    payload = tracking._call_payload(call, "run-one", "calls/one.json")
    publisher.artifacts["calls/one.json"] = copy.deepcopy(call)
    publisher.client.search_traces.side_effect = None
    publisher.client.search_traces.return_value = [SimpleNamespace(data=SimpleNamespace(spans=[
        SimpleNamespace(name="model", inputs=received, outputs=None,
                        attributes=payload["attributes"]),
    ]))]
    assert tracking._call_trace_available(
        publisher.client, {}, "model", expected={
            "payload": payload, "call": call, "run_id": "run-one",
            "artifact_path": "calls/one.json",
        },
    ) is (received == call["inputs"])


@pytest.mark.parametrize("previous", [None, "true", "false"])
@pytest.mark.parametrize("fails", [False, True])
def test_export_preserves_payload_and_restores_attachment_setting(
    publisher, monkeypatch, previous, fails,
):
    name = "MLFLOW_TRACE_EXTRACT_ATTACHMENTS"
    if previous is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, previous)
    start = publisher.sdk.start_span_no_context.side_effect

    def start_span(*args, **kwargs):
        assert os.environ.get(name) == "false"
        if fails:
            raise ConnectionError("unavailable")
        return start(*args, **kwargs)

    publisher.sdk.start_span_no_context.side_effect = start_span
    value = record([{"name": "model", "inputs": "data:image/png;base64,c3ludGhldGlj"}] * 65)
    if fails:
        with pytest.raises(ConnectionError):
            tracking._publish(value, "http://tracking", "export-one", False)
    else:
        assert tracking._publish(value, "http://tracking", "export-one", False) == "run-one"
    assert os.environ.get(name) == previous


def test_oversized_attributes_use_reference_and_keep_role(publisher):
    call = {"name": "model", "span_id": "source",
            "attributes": {"role": "solver", "context": "x" * 2000000}}
    tracking._publish(record([call]), "http://tracking", "export-one", False)
    start = publisher.sdk.start_span_no_context.call_args
    assert "context" not in start.kwargs["attributes"]
    assert start.kwargs["attributes"]["role"] == "solver"
    path = start.kwargs["inputs"]["arena_artifact"]
    assert publisher.artifacts[path] == call


def test_trace_size_check_stops_without_copying_large_string():
    import tracemalloc

    value = {"inputs": "x" * (32 * 1024 * 1024)}
    tracemalloc.start()
    try:
        assert tracking._large_json(value)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 512 * 1024


def test_deferred_export_is_durable_and_flush_delivers_once(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared")
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record()))
    publish = Mock(return_value="run-one")
    monkeypatch.setattr(tracking, "_publish", publish)
    queue = tmp_path / "queue"
    assert tracking.export_record(path, queue, defer=True) == {"status": "tracking_pending"}
    publish.assert_not_called()
    entry = next(queue.glob("*.json"))
    export_id = json.loads(entry.read_text())["export_id"]
    path.unlink()
    monkeypatch.setenv("ARENA_TRACKING_DELIVERY", "queued")
    assert tracking.flush_queue(queue) == [{"status": "exported", "run_id": "run-one"}]
    assert tracking.flush_queue(queue) == []
    assert publish.call_args.args[2] == export_id


def test_legacy_pending_queue_keeps_original_export_identity(tmp_path, monkeypatch, publisher):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared")
    entry = {
        "uri": "http://shared", "operator": False, "status": "tracking_pending",
        "export_id": "legacy-identity", "attempts": 121,
        "record": record([{"name": "model", "inputs": "x" * 2000000}]),
    }
    queue = tmp_path / "queue"
    queue.mkdir()
    path = queue / "legacy.json"
    path.write_text(json.dumps(entry))
    assert tracking.flush_queue(queue) == [{"status": "exported", "run_id": "run-one"}]
    result = json.loads(path.read_text())
    assert result["export_id"] == "legacy-identity"
    assert result["record"] == entry["record"]
    assert result["attempts"] == 121
    assert publisher.sdk.set_tags.call_args.args[0]["arena_export_id"] == "legacy-identity"
    assert tracking.flush_queue(queue) == []


@pytest.mark.parametrize("split", ["train", "public", "private"])
def test_configured_queued_delivery_keeps_visibility(tmp_path, monkeypatch, split):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared")
    monkeypatch.setenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", "http://operator")
    monkeypatch.setenv("ARENA_TRACKING_DELIVERY", "queued")
    value = record([{"name": "secret-call", "inputs": "closed"}])
    value["split"] = split
    value["examples"] = [{"reference": "closed"}]
    path = tmp_path / "record.json"
    path.write_text(json.dumps(value))
    publish = Mock(return_value="unused")
    monkeypatch.setattr(tracking, "_publish", publish)
    result = tracking.export_configured(path, tmp_path / "queue")
    publish.assert_not_called()
    assert all(item["status"] == "tracking_pending" for item in result.values())
    if split == "private":
        assert set(result) == {"operator"}
    elif split == "public":
        shared = next((tmp_path / "queue/shared").glob("*.json")).read_text()
        assert "closed" not in shared and "secret-call" not in shared
    operator = next((tmp_path / "queue/operator").glob("*.json")).read_text()
    assert "closed" in operator
