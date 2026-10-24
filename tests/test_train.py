import base64
import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evaluator.reliability import empty_usage
from evaluator.runner import RunnerError, SolveResult
from evaluator.service import Settings
from evaluator.train import TrainSettings, create_app


@pytest.fixture
def settings(tmp_path: Path) -> TrainSettings:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('\n'.join(json.dumps({
        "id": f"train-{i}", "input": f"task-{i}", "reference": "reference",
    }) for i in range(2)))
    metric = tmp_path / "metric.json"
    metric.write_text(json.dumps({
        "version": "v1", "instruction": "judge", "votes": 1,
        "criteria": [{"id": "correct", "question": "Correct?", "weight": 1}],
    }))
    return TrainSettings(Settings(
        dataset_path=dataset, metric_path=metric, state_dir=tmp_path / "state",
        token="test-token", runner_image="fixed", release_id="release",
        model="internal", model_base_url="http://internal/v1", retry_delay=0,
    ), campaign_id="campaign", case_id="case", dataset_id="dataset", conditions_id="fixed",
        tracking=False)


def payload(ids=None):
    return {"candidate": {"id": "candidate", "files": {
        "prompt.md": base64.b64encode(b"prompt").decode(),
    }}, "example_ids": ids}


def api(settings, solve=None, complete=None):
    return TestClient(create_app(
        settings,
        solve=solve or (lambda candidate, task: SolveResult("answer", empty_usage(), 2.0)),
        complete=complete or (lambda messages: '{"score": 4, "reason": "good"}'),
    ), headers={"Authorization": "Bearer test-token"})


def result(client, submission):
    response = client.post("/v1/evaluations", json=submission)
    assert response.status_code == 202
    return client.get("/v1/evaluations/" + response.json()["evaluation_id"]).json()


def test_train_records_open_data_and_uses_single_answer(settings):
    calls = []
    client = api(settings, solve=lambda candidate, task: (
        calls.append(task) or SolveResult("answer", empty_usage(), 2.0)
    ))
    record = result(client, payload())
    assert record["status"] == "FINISHED"
    assert record["metrics"]["score"] == 4
    assert record["metrics"]["solver_seconds"] == 4
    assert record["metrics"]["judge_unreported_attempts"] == 2
    assert record["examples"][0]["input"] == "task-0"
    assert record["examples"][0]["reference"] == "reference"
    assert record["aggregation"]["status"] == "complete"
    assert record["progress"]["completed_examples"] == 2
    assert len(record["calls"]) == 5
    repeat = client.post("/v1/evaluations", json=payload()).json()
    assert repeat["evaluation_id"] == record["evaluation_id"]
    assert calls == ["task-0", "task-1"]
    smaller = result(client, payload(["train-0"]))
    assert smaller["metrics"]["score"] == 4
    assert calls == ["task-0", "task-1"]


def test_legacy_checksum_metadata_preserves_train_answers(settings, tmp_path):
    from evaluator.metric_package import freeze_metric

    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "score.py").write_text("print('{\"score\":2}')\n")
    (draft / "metric.json").write_text(json.dumps({
        "version": "prepared", "kind": "command", "command": {"argv": ["python", "score.py"]},
    }))
    package = tmp_path / "frozen"
    freeze_metric(draft, package)
    configured = replace(settings, evaluation=replace(settings.evaluation, metric_path=package))
    original = result(api(configured), payload())
    path = configured.evaluation.state_dir / "configuration.json"
    stored = json.loads(path.read_text())
    stored["metric"]["command"]["source_sha256"] = {"/old/source.py": "legacy"}
    stored["package"]["sha256"] = "legacy"
    for entry in stored["package"]["files"].values():
        entry["sha256"] = "legacy"
    path.write_text(json.dumps(stored))

    def repeated_solve(*args):
        pytest.fail("recorded answer must not run again")

    restarted = api(configured, solve=repeated_solve)
    repeated = restarted.post("/v1/evaluations", json=payload()).json()
    assert repeated["evaluation_id"] == original["evaluation_id"]
    assert restarted.get("/v1/evaluations/" + original["evaluation_id"]).json() == original


def test_solver_failure_retains_measured_resources(settings):
    usage = {**empty_usage(), "attempts": 2, "unreported_attempts": 2}

    def fail(candidate, task):
        raise RunnerError("unavailable", usage=usage, wall_seconds=7.0)

    record = result(api(settings, solve=fail), payload(["train-0"]))
    assert record["status"] == "FAILED"
    assert "score" not in record["metrics"]
    assert record["metrics"]["solver_unreported_attempts"] == 2
    assert record["metrics"]["solver_seconds"] == 7
    assert record["metrics"]["judge_seconds"] == 0


def test_answer_is_saved_before_judge_and_not_repeated_after_failure(settings):
    answers = []

    def complete(messages):
        paths = list(settings.evaluation.state_dir.glob("answers/*/*.json"))
        assert paths and json.loads(paths[0].read_text())["answer"] == "answer"
        raise ValueError("judge unavailable")

    client = api(settings, solve=lambda candidate, task: (
        answers.append(task) or SolveResult("answer", empty_usage(), 2)
    ), complete=complete)
    record = result(client, payload(["train-0"]))
    assert record["status"] == "FAILED"
    assert record["examples"][0]["answer"] == "answer"
    assert record["metrics"]["judge_unreported_attempts"] >= 1
    client.post("/v1/evaluations", json=payload(["train-0"]))
    assert answers == ["task-0"]


def test_subset_validation_and_authentication(settings):
    client = api(settings)
    assert client.post("/v1/evaluations", json=payload([])).status_code == 422
    assert client.post("/v1/evaluations", json=payload(["missing"])).status_code == 422
    assert client.post("/v1/evaluations", json=payload(["train-0", "train-0"])).status_code == 422
    assert TestClient(client.app).post("/v1/evaluations", json=payload()).status_code == 401
    assert client.get("/v1/evaluations/../../metric.json").status_code == 404


def test_empty_train_keeps_build_and_publisher_available(settings, tmp_path, monkeypatch):
    import sys

    settings.evaluation.dataset_path.write_text("")
    settings.evaluation.metric_path.write_text(json.dumps({
        "version": "command-v1", "kind": "command",
        "command": {"argv": [sys.executable, "-c", "print('{\"score\": 1}')"]},
    }))
    staging = tmp_path / "staging"
    (staging / "source").mkdir(parents=True)
    (staging / "source" / "input.txt").write_text("source")
    published = tmp_path / "published"
    configured = replace(settings, evaluation=replace(
        settings.evaluation, uses_model=False, model="", model_base_url="",
        runner_image="sha256:" + "a" * 64, runner_supports_build=True,
        build_dir=tmp_path / "builds", published_assets_dir=published,
        input_assets_digest="b" * 64, asset_staging_dir=staging, asset_store_dir=published,
    ))
    builds = []

    def build(self, candidate, task, *, parameters=None):
        assert self.build_output_dir is not None
        assert task == ""
        builds.append(candidate)
        (self.build_output_dir / "prepared.txt").write_text("prepared")
        return SolveResult("build complete", empty_usage(), 1)

    monkeypatch.setattr("evaluator.runner.DockerRunner.__call__", build)
    monkeypatch.setattr("evaluator.train.acquire_slot", lambda *args: (
        pytest.fail("empty train must not reserve an evaluation slot")
    ))
    monkeypatch.setattr("evaluator.train.ModelClient", lambda *args, **kwargs: (
        pytest.fail("empty train must not construct a model client")
    ))
    client = TestClient(create_app(configured), headers={"Authorization": "Bearer test-token"})
    assert client.get("/healthz").status_code == 200
    asset = client.post("/v1/assets", json={"path": "source"})
    assert asset.status_code == 200
    assert (published / asset.json()["digest"] / "input.txt").read_text() == "source"
    candidate = payload()["candidate"]
    candidate["files"]["build.py"] = base64.b64encode(b"build recipe").decode()
    response = client.post("/v1/builds", json={"candidate": candidate})
    assert response.status_code == 202
    receipt = client.get("/v1/builds/" + response.json()["candidate_digest"]).json()
    assert receipt["status"] == "FINISHED"
    assert receipt["resources"]["solver"]["usage"] == empty_usage()
    assert len(builds) == 1
    assert client.post("/v1/evaluations", json=payload()).status_code == 422
    assert not list(configured.evaluation.state_dir.glob("evaluations/*"))
    assert not list(configured.evaluation.state_dir.glob("answers/*"))
    assert len(builds) == 1


def test_configuration_frozen(settings):
    api(settings)
    settings.evaluation.dataset_path.write_text('{"id":"new","input":"new","reference":"new"}')
    with pytest.raises(ValueError, match="frozen"):
        api(settings)


def test_aggregation_failure_keeps_separate_judge_costs(settings, monkeypatch):
    from evaluator.metrics import CommandMetricError

    def aggregate(metric, results, *, model_access, usage):
        usage.update({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
                      "attempts": 2, "unreported_attempts": 1})
        raise CommandMetricError("aggregation failed", {"model_usage_complete": False})

    monkeypatch.setattr("evaluator.train.aggregate_results", aggregate)
    record = result(api(settings), payload(["train-0"]))
    assert record["status"] == "FAILED"
    assert record["metrics"]["judge_tokens"] == 7
    assert record["metrics"]["judge_unreported_attempts"] == 1
    assert record["metrics"]["judge_resources_complete"] is False
    assert record["aggregation"]["resources"]["judge"]["usage"]["total_tokens"] == 7
    assert record["examples"][0]["resources"]["judge"]["usage"]["attempts"] == 1
    assert "score" not in record["metrics"]


def test_train_redacts_gateway_capabilities_before_saving(settings):
    answer = "http://gateway/runs/" + "a" * 36 + "/" + "b" * 64 + "/v1"
    record = result(api(settings, solve=lambda candidate, task: (
        SolveResult(answer, empty_usage(), 1)
    )), payload(["train-0"]))
    assert "b" * 64 not in json.dumps(record)
    for path in settings.evaluation.state_dir.rglob("*.json"):
        assert "b" * 64 not in path.read_text()


def test_result_refuses_a_different_comparison(settings, tmp_path, monkeypatch):
    from evaluator.search import Journal, Resources
    from evaluator.train import main

    journal_path = tmp_path / "journal.sqlite"
    journal = Journal.create(journal_path, max_solutions=1)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "prompt.md").write_text("prompt")
    proposal = journal.propose("candidate", candidate, hypothesis="hypothesis", change="initial",
                               resources=Resources(0, 0))
    journal.begin("measure", "candidate", split="train", dataset_id="other-dataset",
                  subset_id="other-subset", conditions_id="other-conditions")
    record = result(api(settings), payload(["train-0"]))
    assert record["params"]["candidate_digest"] == proposal["digest"]
    monkeypatch.setattr("evaluator.train.request", lambda *args, **kwargs: record)
    with pytest.raises(ValueError, match="comparison"):
        main(["result", "--evaluation-id", record["evaluation_id"], "--measurement", "measure",
              "--output", str(tmp_path / "result.json"), "--url", "http://train",
              "--token-file", str(tmp_path / "token"), "--journal", str(journal_path)])
    assert journal.export()["measurements"][0]["state"] == "running"


def test_submit_retries_transport_without_new_measurement(settings, tmp_path, monkeypatch):
    from evaluator.client import ClientError
    from evaluator.search import Journal, Resources
    from evaluator.train import main

    journal_path = tmp_path / "journal.sqlite"
    journal = Journal.create(journal_path, max_solutions=1)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "prompt.md").write_text("prompt")
    journal.propose("candidate", candidate, hypothesis="hypothesis", change="initial",
                    resources=Resources(0, 0))
    requests = []

    def submit(*args, **kwargs):
        requests.append(args[3])
        if len(requests) == 1:
            raise ClientError("evaluator request failed")
        return {"evaluation_id": "a" * 64, "status": "RUNNING"}

    monkeypatch.setattr("evaluator.train.request", submit)
    arguments = ["submit", "--candidate", "candidate", "--measurement", "measure",
                 "--dataset", "dataset", "--conditions", "fixed", "--url", "http://train",
                 "--token-file", str(tmp_path / "token"), "--journal", str(journal_path),
                 "--train-data", str(settings.evaluation.dataset_path)]
    with pytest.raises(ClientError):
        main(arguments)
    assert main(arguments) == 0
    assert requests[0] == requests[1]
    assert len(journal.export()["measurements"]) == 1
    journal.stop("human requested stop")
    with pytest.raises(ValueError, match="stopped"):
        main(arguments)
    assert len(requests) == 2


def test_submit_empty_train_does_not_register_measurement(settings, tmp_path, monkeypatch):
    from evaluator.search import Journal, Resources
    from evaluator.train import main

    journal_path = tmp_path / "journal.sqlite"
    journal = Journal.create(journal_path, max_solutions=1)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "prompt.md").write_text("prompt")
    journal.propose("candidate", candidate, hypothesis="hypothesis", change="initial",
                    resources=Resources(0, 0))
    settings.evaluation.dataset_path.write_text("")
    monkeypatch.setattr("evaluator.train.request", lambda *args, **kwargs: (
        pytest.fail("empty train must not submit a measurement")
    ))
    with pytest.raises(ValueError, match="empty"):
        main(["submit", "--candidate", "candidate", "--measurement", "measure",
              "--dataset", "dataset", "--conditions", "fixed", "--url", "http://train",
              "--token-file", str(tmp_path / "token"), "--journal", str(journal_path),
              "--train-data", str(settings.evaluation.dataset_path)])
    assert journal.export()["measurements"] == []


def test_capacity_rejection_creates_no_evaluation(settings, tmp_path, monkeypatch):
    from evaluator.capacity import acquire_slot

    directory = tmp_path / "slots"
    directory.mkdir(mode=0o700)
    settings = replace(settings, evaluation=replace(
        settings.evaluation, slot_dir=directory, execution_slots=1,
    ))
    lease = acquire_slot(directory, 1)
    assert lease is not None
    client = api(settings)
    try:
        response = client.post("/v1/evaluations", json=payload())
        assert response.status_code == 503
        assert response.headers["retry-after"] == "60"
        assert not list(settings.evaluation.state_dir.glob("evaluations/*/record.json"))
    finally:
        lease.close()
    assert result(client, payload())["status"] == "FINISHED"


def test_setup_failure_is_durably_finished(settings, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("invalid model configuration")

    monkeypatch.setattr("evaluator.train.ModelClient", fail)
    client = TestClient(create_app(settings, solve=lambda candidate, task: (
        SolveResult("answer", empty_usage(), 1)
    )), headers={"Authorization": "Bearer test-token"})
    record = result(client, payload())
    assert record["status"] == "FAILED"
    assert record["error"]["type"] == "ValueError"
    assert record["examples"] == []
    assert record["metrics"]["solver_tokens"] == 0
    assert "end_time_ns" in record


@pytest.mark.parametrize("role", ["solver", "judge", "aggregation"])
def test_interrupted_stage_never_claims_complete_zero_accounting(settings, role):
    record = result(api(settings), payload(["train-0"]))
    record["status"] = "RUNNING"
    record["active_stage"] = {"role": role, "example_id": "train-0",
                              "start_time_ns": record["start_time_ns"]}
    if role == "solver":
        record["examples"] = []
        for answer in settings.evaluation.state_dir.glob("answers/*/*.json"):
            answer.unlink()
    path = settings.evaluation.state_dir / "evaluations" / record["evaluation_id"] / "record.json"
    path.write_text(json.dumps(record))
    restored = api(settings).get("/v1/evaluations/" + record["evaluation_id"]).json()
    resource_role = "solver" if role == "solver" else "judge"
    assert restored["status"] == "FAILED"
    assert restored["metrics"][f"{resource_role}_resources_complete"] is False
    assert "score" not in restored["metrics"]


def test_recovery_preserves_completed_aggregation(settings):
    record = result(api(settings), payload(["train-0"]))
    record["status"] = "RUNNING"
    path = settings.evaluation.state_dir / "evaluations" / record["evaluation_id"] / "record.json"
    path.write_text(json.dumps(record))
    restored = api(settings).get("/v1/evaluations/" + record["evaluation_id"]).json()
    assert restored["status"] == "FINISHED"
    assert restored["metrics"]["score"] == 4


def test_build_preflight_and_one_time_preparation_accounting(settings, tmp_path, monkeypatch):
    from evaluator.build import BuildStore
    from evaluator.contracts import Candidate

    build_dir = tmp_path / "builds"
    settings = replace(settings, evaluation=replace(
        settings.evaluation, build_dir=build_dir, runner_supports_build=True,
    ))
    submission = payload()
    submission["candidate"]["files"]["build.py"] = base64.b64encode(b"build recipe").decode()
    frozen = Candidate.model_validate(submission["candidate"])
    client = api(settings)
    assert client.post("/v1/evaluations", json=submission).status_code == 422
    assert client.post("/v1/evaluations", json=submission).status_code == 422
    assert not list(settings.evaluation.state_dir.glob("evaluations/*/package-*"))

    def build(candidate_dir, output):
        (output / "index").write_text("synthetic index")
        usage = {**empty_usage(), "prompt_tokens": 7, "total_tokens": 7, "attempts": 1}
        return SolveResult("build complete", usage, 5)

    builder = BuildStore(build_dir, tmp_path / "published", "a" * 64, "b" * 64, build)
    receipt, completion = builder.submit(frozen)
    assert completion is not None
    completion.result(timeout=10)
    receipt = builder.read(frozen.digest())
    record = result(client, submission)
    assert record["metrics"]["solver_tokens"] == 7
    assert record["metrics"]["solver_seconds"] == pytest.approx(
        receipt["resources"]["solver"]["wall_seconds"] + 4,
    )
    assert record["preparation"]["candidate_digest"] == frozen.digest()
    assert record["metrics"]["judge_tokens"] == 0


def test_ordinary_build_file_does_not_enable_preparation(settings):
    submission = payload(["train-0"])
    submission["candidate"]["files"]["build.py"] = base64.b64encode(b"ordinary helper").decode()
    record = result(api(settings), submission)
    assert record["status"] == "FINISHED"
    assert record["preparation"] is None


def test_train_uses_settings_instead_of_ambient_service_configuration(settings, monkeypatch):
    for name in ("BUILD_DIR", "SLOT_DIR", "TRACE_DIR", "ASSET_STAGING_DIR"):
        monkeypatch.setenv("ARENA_" + name, "/unavailable/" + name)
    monkeypatch.setenv("ARENA_EXECUTION_SLOTS", "invalid")
    record = result(api(settings), payload(["train-0"]))
    assert record["status"] == "FINISHED"


@pytest.mark.parametrize("failure", [OSError, ValueError])
def test_trace_read_failure_preserves_train_answer(settings, monkeypatch, failure):
    import evaluator.trace_store as traces

    def missing(*args):
        raise failure("storage unavailable")

    monkeypatch.setattr(traces, "read_calls", missing)
    configured = replace(settings, evaluation=replace(
        settings.evaluation, trace_dir=settings.evaluation.state_dir / "traces",
    ))
    record = result(api(configured, solve=lambda *args: SolveResult(
        "saved answer", empty_usage(), 2,
        run_id="11111111-1111-4111-8111-111111111111",
    )), payload(["train-0"]))
    assert record["status"] == "FINISHED"
    assert record["examples"][0]["answer"] == "saved answer"
    assert record["trace_errors"][0]["error_type"] == failure.__name__
    assert "storage unavailable" not in json.dumps(record)


def test_build_endpoint_rejects_harness_without_capability_before_execution(
    settings, tmp_path, monkeypatch,
):
    image = "sha256:" + "a" * 64
    configured = replace(settings, evaluation=replace(
        settings.evaluation, runner_image=image, runner_supports_build=True,
        build_dir=tmp_path / "builds", published_assets_dir=tmp_path / "published",
        input_assets_digest="b" * 64,
        harnesses={"ordinary": {"image": image, "supports_build": False}},
    ))
    calls = []
    monkeypatch.setattr("evaluator.runner.DockerRunner.__call__", lambda *args, **kwargs: (
        calls.append(args) or SolveResult("unexpected", empty_usage(), 1)
    ))
    client = TestClient(create_app(configured), headers={"Authorization": "Bearer test-token"})
    submission = payload()
    submission["candidate"]["files"].update({
        "build.py": base64.b64encode(b"ordinary helper").decode(),
        "solution.json": base64.b64encode(json.dumps({
            "harness": "ordinary", "parameters": {"asset_digests": "adapter-owned"},
        }).encode()).decode(),
    })
    build = {"candidate": submission["candidate"]}
    assert client.post("/v1/builds", json=build).status_code == 422
    assert client.post("/v1/builds", json=build).status_code == 422
    assert not calls
    assert not list((tmp_path / "builds" / "receipts").glob("*.json"))
    assert not list(configured.evaluation.state_dir.glob("evaluations/*/record.json"))


def test_registered_ordinary_harness_preserves_adapter_parameters(settings, monkeypatch):
    import sys

    program = settings.evaluation.metric_path.parent / "score.py"
    program.write_text(
        "import json,sys\n"
        "print(json.dumps({'score':float(json.load(sys.stdin)['answer'])}))\n"
    )
    executable = Path(sys.executable).resolve()
    settings.evaluation.metric_path.write_text(json.dumps({
        "version": "command-v1", "kind": "command",
        "command": {"argv": [str(executable), str(program)]},
    }))
    image = "sha256:" + "a" * 64
    selected_image = "sha256:" + "c" * 64
    configured = replace(settings, evaluation=replace(
        settings.evaluation, runner_image=image, uses_model=False, model="", model_base_url="",
        runner_supports_build=True, runner_supports_asset_selection=True,
        build_dir=settings.evaluation.state_dir.parent / "builds",
        published_assets_dir=settings.evaluation.state_dir.parent / "published",
        input_assets_digest="b" * 64,
        harnesses={"ordinary": {"image": selected_image, "uses_model": False}},
    ))
    calls = []

    def solve(self, candidate, task, *, parameters=None):
        assert (candidate / "build.py").read_text() == "ordinary helper"
        assert self.image == selected_image
        calls.append(parameters)
        return SolveResult("3", empty_usage(), 1)

    def receipt(*args):
        pytest.fail("ordinary harness must not read trusted build receipts")

    monkeypatch.setattr("evaluator.runner.DockerRunner.__call__", solve)
    monkeypatch.setattr("evaluator.runner.load_build_receipt", receipt)
    monkeypatch.setattr("evaluator.train.ModelClient", lambda *args, **kwargs: (
        pytest.fail("model-free train must not construct a model client")
    ))
    submission = payload(["train-0"])
    submission["candidate"]["files"].update({
        "build.py": base64.b64encode(b"ordinary helper").decode(),
        "solution.json": base64.b64encode(json.dumps({
            "harness": "ordinary", "parameters": {"asset_digests": "adapter-owned"},
        }).encode()).decode(),
    })
    client = TestClient(create_app(configured), headers={"Authorization": "Bearer test-token"})
    record = result(client, submission)
    assert record["status"] == "FINISHED"
    assert record["preparation"] is None
    assert record["metrics"]["score"] == 3
    assert record["metrics"]["solver_tokens"] == record["metrics"]["judge_tokens"] == 0
    assert record["params"]["harness"] == "ordinary"
    assert record["params"]["harness_version"] == selected_image
    assert calls == [{"asset_digests": "adapter-owned"}]


def test_builtin_judge_traces_survive_subset_reuse_and_export(settings, monkeypatch):
    from unittest.mock import MagicMock

    import httpx

    from evaluator.tracking import _spans
    from tests.test_judge import install_transport, response_payload

    requests = install_transport(monkeypatch, lambda _: httpx.Response(
        200, json=response_payload('{"score": 4, "reason": "api-secret"}'),
    ))
    exported = []

    def export(path, **kwargs):
        exported.append(json.loads(path.read_text()))
        return {"shared": {"status": "exported", "run_id": "train-run"}}

    monkeypatch.setattr("evaluator.tracking.export_configured", export)
    configured = replace(settings, tracking=True, evaluation=replace(
        settings.evaluation, model_api_key="api-secret",
    ))
    client = TestClient(create_app(configured, solve=lambda *args: SolveResult(
        "answer", empty_usage(), 1.0,
    )), headers={"Authorization": "Bearer test-token"})
    full = result(client, payload())
    subset = result(client, payload(["train-0"]))
    assert full["status"] == subset["status"] == "FINISHED"
    assert len(requests) == 2
    full_calls = [call for call in full["calls"] if call.get("span_type") == "LLM"]
    subset_calls = [call for call in subset["calls"] if call.get("span_type") == "LLM"]
    assert len(full_calls) == 2
    assert subset_calls == [full_calls[0]]
    assert subset_calls[0]["attributes"] | {
        "role": "judge", "example_id": "train-0", "criterion": "correct", "vote": 1,
    } == subset_calls[0]["attributes"]
    assert subset["examples"][0]["judge_calls"] == subset_calls
    assert subset_calls[0] in exported[-1]["calls"]
    assert "api-secret" not in json.dumps(subset_calls)
    sdk = MagicMock()
    _spans(sdk, exported[-1], "train-run")
    model_span = next(call for call in sdk.start_span_no_context.call_args_list
                      if call.kwargs.get("span_type") == "LLM")
    assert model_span.kwargs["inputs"] == subset_calls[0]["inputs"]
    assert model_span.kwargs["attributes"] == subset_calls[0]["attributes"]


def test_builtin_judge_failed_vote_attempts_are_kept_in_train(settings, monkeypatch):
    import httpx

    from tests.test_judge import install_transport, response_payload

    requests = install_transport(monkeypatch, lambda _: httpx.Response(
        200, json=response_payload("invalid"),
    ))
    client = TestClient(create_app(settings, solve=lambda *args: SolveResult(
        "answer", empty_usage(), 1.0,
    )), headers={"Authorization": "Bearer test-token"})
    record = result(client, payload(["train-0"]))
    assert record["status"] == "FAILED"
    calls = [call for call in record["calls"] if call.get("span_type") == "LLM"]
    assert len(requests) == len(calls) == 6
    assert all(call["status"] == "ERROR" for call in calls)
    assert record["examples"][0]["judge_calls"] == calls


def test_train_retains_trace_callback_error_without_repeating_judge(settings, monkeypatch):
    import httpx

    from evaluator.judge import ModelClient
    from tests.test_judge import install_transport, response_payload

    requests = install_transport(monkeypatch, lambda _: httpx.Response(
        200, json=response_payload('{"score": 4, "reason": "ok"}'),
    ))

    def fail(call):
        raise OSError("secret-diagnostic")

    def model(*args, **kwargs):
        return ModelClient(*args, **(kwargs | {"on_attempt": fail}))

    monkeypatch.setattr("evaluator.train.ModelClient", model)
    client = TestClient(create_app(settings, solve=lambda *args: SolveResult(
        "answer", empty_usage(), 1.0,
    )), headers={"Authorization": "Bearer test-token"})
    full = result(client, payload())
    subset = result(client, payload(["train-0"]))
    assert full["status"] == subset["status"] == "FINISHED"
    assert len(requests) == 2
    assert subset["trace_errors"] == [{
        "error_type": "OSError", "role": "judge", "example_id": "train-0",
    }]
    assert "secret-diagnostic" not in json.dumps(subset)
