import asyncio
import base64
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from anyio.to_thread import current_default_thread_limiter
from fastapi.testclient import TestClient

from evaluator.candidates import materialize_candidate
from evaluator.client import pack_candidate
from evaluator.judge import ModelClient
from evaluator.reliability import empty_usage
from evaluator.runner import RunnerError, SolveResult
from evaluator.service import Settings, create_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    dataset = tmp_path / "private.jsonl"
    dataset.write_text(
        '\n'.join(
            json.dumps({"id": f"secret-{i}", "input": f"task-{i}", "reference": "secret-ref"})
            for i in range(2)
        )
    )
    metric = tmp_path / "metric.json"
    metric.write_text(json.dumps({
        "version": "v1", "instruction": "judge", "criteria": [
            {"id": "correct", "question": "Correct?", "weight": 1},
        ],
    }))
    return Settings(
        dataset_path=dataset, metric_path=metric, state_dir=tmp_path / "state",
        token="test-token", model="internal-model", model_base_url="http://internal/v1",
        runner_image="fixed-runner", release_id="release-1", finalists=1,
    )


def submission(value: str = "a") -> dict[str, object]:
    return {"candidates": [{"id": "candidate", "files": {
        "prompt.txt": base64.b64encode(value.encode()).decode(),
    }}]}


def solved(answer="answer", tokens=3, seconds=2.0, unreported=0) -> SolveResult:
    return SolveResult(answer, {
        "prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens,
        "attempts": 1, "unreported_attempts": unreported,
    }, seconds)


def client(settings: Settings, solve=None, complete=None) -> TestClient:
    return TestClient(create_app(
        settings,
        solve=solve or (lambda candidate, task: solved()),
        complete=complete or (lambda messages: '{"score": 4, "reason": "secret-reason"}'),
    ), headers={"Authorization": "Bearer test-token"})


def operator_record(settings: Settings, evaluation_id: str | None = None):
    pattern = "*/record.json" if evaluation_id is None else f"*/public/{evaluation_id}/record.json"
    return json.loads(next(settings.state_dir.glob(pattern)).read_text())


def test_evaluates_once_and_exposes_only_status(settings: Settings):
    calls = []
    judges = []

    def solve(candidate, task):
        calls.append((candidate, task))
        assert (candidate / "prompt.txt").read_text() == "a"
        assert (candidate / "prompt.txt").stat().st_mode & 0o777 == 0o644
        return solved("private-answer")

    def complete(messages):
        judges.append(messages)
        return '{"score": 4, "reason": "secret-reason"}'

    api = client(settings, solve, complete)
    assert api.get("/v1/evaluations").status_code == 404
    assert api.post("/v1/evaluations", json=submission()).status_code == 202
    result = api.get("/v1/evaluations").json()
    assert result["status"] == "complete"
    assert result == {"status": "complete"}
    record = operator_record(settings)
    assert record["winners"] == ["candidate"]
    assert len(record["scores"]) == 1
    assert set(record["scores"][0]) == {
        "candidate_id", "digest", "score", "solver_tokens", "solver_seconds",
        "solver_unreported_attempts", "judge_tokens", "judge_seconds", "judge_unreported_attempts",
    }
    resources = record["scores"][0]
    assert resources["solver_tokens"] == 6
    assert resources["solver_seconds"] == 4.0
    assert resources["solver_unreported_attempts"] == 0
    assert resources["judge_tokens"] == 0
    assert resources["judge_seconds"] >= 0
    assert resources["judge_unreported_attempts"] == 6
    assert len(calls) == 2
    assert len(judges) == 6
    public = json.dumps(result)
    for secret in ["secret-", "private-answer", "task-", "internal-model"]:
        assert secret not in public
    assert api.post("/v1/evaluations", json=submission()).json() == result
    assert len(calls) == 2
    assert api.post("/v1/evaluations", json=submission("b")).status_code == 409
    restarted = client(replace(settings, release_id="new-label"))
    assert restarted.get("/v1/evaluations").json() == result
    assert restarted.post("/v1/evaluations", json=submission("c")).status_code == 409
    private_files = list(settings.state_dir.rglob("*.json"))
    assert any("private-answer" in path.read_text() for path in private_files)


def test_auth_before_parsing_and_size_limit(settings: Settings, monkeypatch):
    monkeypatch.setattr("evaluator.service.MAX_BODY_BYTES", 100)
    api = client(settings)
    assert api.post("/v1/evaluations", content="x" * 200).status_code == 413
    unauth = TestClient(api.app)
    assert unauth.post("/v1/evaluations", content="{" * 300).status_code == 401
    assert unauth.get("/v1/evaluations").status_code == 401
    assert unauth.get("/healthz").json() == {"status": "ok"}
    assert api.get("/openapi.json").status_code == 404


def test_validation_is_generic_and_finalist_count_is_fixed(settings: Settings):
    api = client(settings)
    malformed = {"candidates": [{"id": "candidate", "files": {"../secret": "abc"}}]}
    response = api.post("/v1/evaluations", json=malformed)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid submission"}
    assert api.post("/v1/evaluations", json={"candidates": []}).status_code == 422
    assert client(replace(settings, finalists=2)).post(
        "/v1/evaluations", json=submission(),
    ).status_code == 422


def test_oversized_unicode_filename_is_rejected_before_private_admission(settings: Settings):
    calls = []
    api = client(
        settings, solve=lambda candidate, task: calls.append((candidate, task)) or solved(),
    )
    malformed = {"candidates": [{"id": "candidate", "files": {"я" * 128: "YQ=="}}]}
    response = api.post("/v1/evaluations", json=malformed)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid submission"}
    assert api.get("/v1/evaluations").status_code == 404
    assert not list(settings.state_dir.rglob("record.json"))
    assert not list(settings.state_dir.rglob("candidates"))
    assert calls == []
    assert api.post("/v1/evaluations", json=submission()).status_code == 202
    assert api.get("/v1/evaluations").json() == {"status": "complete"}


def test_blank_answer_scores_zero_without_judge(settings: Settings):
    def complete(messages):
        pytest.fail("blank answer must not reach judge")

    api = client(settings, solve=lambda candidate, task: solved("  "), complete=complete)
    api.post("/v1/evaluations", json=submission())
    result = api.get("/v1/evaluations").json()
    assert result["status"] == "complete"
    assert operator_record(settings)["scores"][0]["score"] == 0
    assert "candidate_failure" in "".join(
        path.read_text() for path in settings.state_dir.rglob("*.json")
    )


def test_infrastructure_failure_is_terminal_and_private(settings: Settings):
    def solve(candidate, task):
        raise RunnerError("secret-ref database unavailable")

    api = client(settings, solve=solve)
    api.post("/v1/evaluations", json=submission())
    assert api.get("/v1/evaluations").json() == {"status": "failed"}
    assert api.post("/v1/evaluations", json=submission("b")).status_code == 409
    assert api.post("/v1/evaluations", json=submission()).json() == {"status": "failed"}


def test_concurrent_reservation_and_no_intermediate_scores(settings: Settings):
    started = threading.Event()
    finish = threading.Event()
    calls = []

    def solve(candidate, task):
        calls.append(task)
        started.set()
        assert finish.wait(5)
        return solved()

    api = client(settings, solve=solve)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(api.post, "/v1/evaluations", json=submission())
        try:
            assert started.wait(5)
            second_api = client(settings, solve=solve)
            assert second_api.get("/v1/evaluations").json() == {"status": "running"}
            assert second_api.post("/v1/evaluations", json=submission("b")).status_code == 409
            assert second_api.post("/v1/evaluations", json=submission()).json() == {
                "status": "running",
            }
        finally:
            finish.set()
        assert pending.result().status_code == 202
    assert len(calls) == 2


def test_rejects_empty_or_duplicate_examples_and_missing_token(settings: Settings):
    for contents in ["", '{"id":"x","input":"x","reference":"x"}\n' * 2]:
        settings.dataset_path.write_text(contents)
        with pytest.raises(ValueError):
            client(settings)
    with pytest.raises(ValueError):
        replace(settings, token="")


def test_changed_metric_cannot_reopen_dataset(settings: Settings):
    api = client(settings)
    api.post("/v1/evaluations", json=submission())
    metric = json.loads(settings.metric_path.read_text())
    metric["version"] = "changed"
    settings.metric_path.write_text(json.dumps(metric))
    with pytest.raises(ValueError, match="frozen"):
        client(settings)


def test_candidate_name_cannot_overwrite_receipt(settings: Settings):
    api = client(settings)
    data = {"candidates": [{"id": "record", "files": {"x": "YQ=="}}]}
    api.post("/v1/evaluations", json=data)
    assert api.get("/v1/evaluations").json() == {"status": "complete"}
    assert operator_record(settings)["winners"] == ["record"]
    assert api.post("/v1/evaluations", json=data).json()["status"] == "complete"


def test_cannot_mount_private_data_or_state_as_assets(settings: Settings):
    with pytest.raises(ValueError, match="assets"):
        client(replace(settings, assets_dir=settings.dataset_path.parent))
    with pytest.raises(ValueError, match="assets"):
        client(replace(settings, assets_dir=settings.state_dir))


def test_interrupted_batch_fails_closed_on_restart(settings: Settings):
    api = client(settings)
    api.post("/v1/evaluations", json=submission())
    record_path = next(settings.state_dir.glob("*/record.json"))
    record = json.loads(record_path.read_text())
    record["status"] = "running"
    record_path.write_text(json.dumps(record))
    restarted = client(settings)
    assert restarted.get("/v1/evaluations").json() == {"status": "failed"}
    assert restarted.post("/v1/evaluations", json=submission()).json() == {"status": "failed"}
    assert restarted.post("/v1/evaluations", json=submission("new")).status_code == 409


def test_from_env_reads_file_secrets(settings: Settings, monkeypatch):
    token_file = settings.dataset_path.parent / "token"
    token_file.write_text("from-file\n")
    values = {
        "DATASET_PATH": str(settings.dataset_path), "METRIC_PATH": str(settings.metric_path),
        "STATE_DIR": str(settings.state_dir), "TOKEN_FILE": str(token_file),
        "MODEL": "internal", "MODEL_BASE_URL": "http://internal/v1",
        "RUNNER_IMAGE": "fixed-runner", "RELEASE_ID": "release",
        "PARAMETERS": '{"temperature": 0}', "MODEL_TIMEOUT": "30000",
        "RUNNER_TIMEOUT": "900000", "PREPARE_TIMEOUT": "200000", "RETRY_DELAY": "12",
        "USAGE_DIR": str(settings.state_dir / "accounting"),
    }
    for name, value in values.items():
        monkeypatch.setenv("ARENA_" + name, value)
    loaded = Settings.from_env()
    assert loaded.token == "from-file"
    assert loaded.parameters == {"temperature": 0}
    assert loaded.runner_model_base_url == "http://model-gateway:8081/v1"
    assert loaded.model_timeout == 30000
    assert loaded.runner_timeout == 900000
    assert loaded.prepare_timeout == 200000
    assert loaded.retry_delay == 12
    assert loaded.usage_dir == settings.state_dir / "accounting"


def test_nested_candidate_directories_are_readable_with_restrictive_umask(settings: Settings):
    def solve(candidate, task):
        assert (candidate / "nested").stat().st_mode & 0o777 == 0o755
        assert (candidate / "nested" / "deep").stat().st_mode & 0o777 == 0o755
        return solved()

    mask = os.umask(0o077)
    try:
        api = client(settings, solve=solve)
        data = {"candidates": [{"id": "candidate", "files": {"nested/deep/x": "YQ=="}}]}
        api.post("/v1/evaluations", json=data)
        assert api.get("/v1/evaluations").json()["status"] == "complete"
    finally:
        os.umask(mask)


def test_storage_failure_before_acceptance_allows_clean_retry(settings: Settings, monkeypatch):
    import evaluator.service as service

    api = client(settings)
    original = service._write_json

    def fail(path, value):
        raise OSError("disk unavailable")

    monkeypatch.setattr(service, "_write_json", fail)
    assert api.post("/v1/evaluations", json=submission()).status_code == 503
    monkeypatch.setattr(service, "_write_json", original)
    assert api.post("/v1/evaluations", json=submission()).status_code == 202
    assert api.get("/v1/evaluations").json()["status"] == "complete"


def test_dataset_format_order_and_example_ids_cannot_reset_seal(settings: Settings):
    api = client(settings)
    api.post("/v1/evaluations", json=submission())
    result = api.get("/v1/evaluations").json()
    examples = [json.loads(line) for line in settings.dataset_path.read_text().splitlines()]
    reordered = [
        {"reference": item["reference"], "input": item["input"], "id": f"renamed-{index}"}
        for index, item in enumerate(reversed(examples))
    ]
    settings.dataset_path.write_text(
        "\n".join(json.dumps(item, separators=(",", ":")) for item in reordered) + "\n\n"
    )
    with pytest.raises(ValueError, match="frozen"):
        client(replace(settings, release_id="relabeled"))
    reformatted = [
        {"reference": item["reference"], "input": item["input"], "id": item["id"]}
        for item in examples
    ]
    settings.dataset_path.write_text(
        "\n".join(json.dumps(item, separators=(",", ":")) for item in reformatted) + "\n\n"
    )
    restarted = client(replace(settings, release_id="relabeled"))
    assert restarted.get("/v1/evaluations").json() == result
    assert restarted.post("/v1/evaluations", json=submission("b")).status_code == 409
    assert restarted.post("/v1/evaluations", json=submission()).json() == result


def test_settings_repr_excludes_credentials(settings: Settings):
    value = repr(replace(settings, model_api_key="secret-model-key"))
    assert "test-token" not in value
    assert "secret-model-key" not in value


def test_tied_scores_choose_cheaper_candidate_regardless_of_submission_order(settings: Settings):
    api = client(
        replace(settings, finalists=2),
        solve=lambda candidate, task: solved(tokens=7 if candidate.name == "first" else 3),
    )
    data = {"candidates": [
        {"id": "first", "files": {"prompt.txt": "YQ=="}},
        {"id": "second", "files": {"prompt.txt": "Yg=="}},
    ]}
    api.post("/v1/evaluations", json=data)
    result = api.get("/v1/evaluations").json()
    assert result["status"] == "complete"
    assert len(operator_record(settings)["scores"]) == 2
    assert operator_record(settings)["winners"] == ["second"]


@pytest.mark.parametrize("field", ["model_timeout", "runner_timeout", "prepare_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_settings_reject_invalid_timeouts(settings: Settings, field, value):
    with pytest.raises(ValueError):
        replace(settings, **{field: value})


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True])
def test_settings_reject_invalid_retry_delay(settings: Settings, value):
    with pytest.raises(ValueError):
        replace(settings, retry_delay=value)


def test_settings_allow_zero_retry_delay_and_long_defaults(settings: Settings):
    assert settings.model_timeout == 21600
    assert settings.runner_timeout == 604800
    assert settings.prepare_timeout == 172800
    assert settings.retry_delay == 30
    assert replace(settings, retry_delay=0).retry_delay == 0


def test_plain_solver_response_without_measurements_fails_closed(settings: Settings):
    api = client(settings, solve=lambda candidate, task: "unmeasured-answer")
    api.post("/v1/evaluations", json=submission())
    assert api.get("/v1/evaluations").json() == {"status": "failed"}


def test_persists_resource_measurements_and_freezes_ranking_policy(settings: Settings):
    api = client(settings)
    api.post("/v1/evaluations", json=submission())
    details = json.loads(next(settings.state_dir.glob("*/details/candidate.json")).read_text())
    example = details["examples"][0]
    assert example["resources"]["solver"] == {
        "usage": solved().usage, "wall_seconds": 2.0,
    }
    assert example["resources"]["judge"]["usage"]["unreported_attempts"] == 3
    assert example["resources"]["judge"]["wall_seconds"] >= 0
    record = json.loads(next(settings.state_dir.glob("*/record.json")).read_text())
    assert record["provenance"]["ranking_policy"]
    record["provenance"]["ranking_policy"] = "changed-policy"
    next(settings.state_dir.glob("*/record.json")).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="frozen"):
        client(settings)


def test_judge_resource_deltas_do_not_accumulate_between_examples(settings: Settings, monkeypatch):
    import evaluator.service as service

    model_client = ModelClient("http://internal/v1", "internal-model")

    def judge(metric, example, answer, complete, **kwargs):
        assert complete is model_client
        complete.usage["prompt_tokens"] += 5
        complete.usage["completion_tokens"] += 2
        complete.usage["total_tokens"] += 7
        complete.usage["attempts"] += 3
        return {"score": 4.0, "criteria": {}}

    monkeypatch.setattr(service, "evaluate_answer", judge)
    api = client(settings, complete=model_client)
    api.post("/v1/evaluations", json=submission())
    score = operator_record(settings)["scores"][0]
    assert score["judge_tokens"] == 14
    assert score["judge_unreported_attempts"] == 0
    details = json.loads(next(settings.state_dir.glob("*/details/candidate.json")).read_text())
    assert [item["resources"]["judge"]["usage"]["total_tokens"]
            for item in details["examples"]] == [7, 7]


def test_runner_receives_configurable_long_limits_and_usage_path(settings: Settings, monkeypatch):
    import evaluator.service as service

    captured = {}

    class CapturingRunner(service.DockerRunner):
        def __post_init__(self):
            captured.update(self.configuration())

        def __call__(self, candidate, task, *, parameters=None):
            return solved()

    def judge(metric, example, answer, complete, **kwargs):
        assert isinstance(complete, ModelClient)
        assert complete.timeout == 30000
        assert complete.retry_delay == 0
        return {"score": 4.0, "criteria": {}}

    monkeypatch.setattr(service, "DockerRunner", CapturingRunner)
    monkeypatch.setattr(service, "evaluate_answer", judge)
    configured = replace(
        settings, model_timeout=30000, runner_timeout=900000, prepare_timeout=200000,
        retry_delay=0, usage_dir=settings.state_dir.parent / "accounting",
    )
    api = TestClient(create_app(configured), headers={"Authorization": "Bearer test-token"})
    api.post("/v1/evaluations", json=submission())
    assert api.get("/v1/evaluations").json()["status"] == "complete"
    assert captured["timeout"] == 900000
    assert captured["model_timeout"] == 30000
    assert captured["prepare_timeout"] == 200000
    assert captured["retry_delay"] == 0
    assert captured["usage_dir"] == str(configured.usage_dir)


def test_assets_cannot_expose_resource_ledger(settings: Settings, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    with pytest.raises(ValueError, match="assets"):
        client(replace(settings, assets_dir=assets, usage_dir=assets / "accounting"))


def test_candidate_storage_cannot_contain_resource_ledger(settings: Settings):
    with pytest.raises(ValueError, match="usage"):
        client(replace(settings, usage_dir=settings.state_dir / "candidates" / "accounting"))


@pytest.mark.parametrize("key,value", [
    ("prompt_tokens", True), ("completion_tokens", -1), ("total_tokens", 10),
    ("attempts", 1.5), ("unreported_attempts", 2),
])
def test_invalid_solver_measurements_fail_closed(settings: Settings, key, value):
    def solve(candidate, task):
        usage = {**empty_usage(), "attempts": 1}
        usage[key] = value
        return SolveResult("answer", usage, 2.0)

    api = client(settings, solve=solve)
    api.post("/v1/evaluations", json=submission())
    assert api.get("/v1/evaluations").json() == {"status": "failed"}


@pytest.mark.parametrize("seconds", [-1, float("inf"), float("nan"), True, "1"])
def test_invalid_solver_time_fails_closed(settings: Settings, seconds):
    api = client(settings, solve=lambda candidate, task: solved(seconds=seconds))
    api.post("/v1/evaluations", json=submission())
    assert api.get("/v1/evaluations").json() == {"status": "failed"}


def test_failed_judging_preserves_usage_from_all_attempts(settings: Settings):
    api = client(replace(settings, retry_delay=0), complete=lambda messages: "invalid vote")
    api.post("/v1/evaluations", json=submission())
    assert api.get("/v1/evaluations").json() == {"status": "failed"}
    details = json.loads(next(settings.state_dir.glob("*/details/candidate.json")).read_text())
    assert len(details["examples"]) == 1
    example = details["examples"][0]
    assert example["failure"] == "judge_failure"
    assert example["resources"]["solver"]["usage"]["total_tokens"] == 3
    assert example["resources"]["judge"]["usage"]["attempts"] == 6
    assert example["resources"]["judge"]["usage"]["unreported_attempts"] == 6


@pytest.mark.parametrize("measured", [False, True])
def test_failed_solver_preserves_private_measurements_and_ledger_reference(
    settings: Settings, measured,
):
    run_id = "9225c5f1-2f6c-4303-a4bf-ec2c7aa65c31" if measured else None
    usage = {
        "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
        "attempts": 6, "unreported_attempts": 2,
    } if measured else None
    seconds = 43200.5 if measured else None

    def solve(candidate, task):
        raise RunnerError(
            "private-failure" * 1000, usage=usage, wall_seconds=seconds, run_id=run_id,
        )

    def complete(messages):
        pytest.fail("failed solver must not reach judge")

    api = client(settings, solve=solve, complete=complete)
    api.post("/v1/evaluations", json=submission())
    assert api.get("/v1/evaluations").json() == {"status": "failed"}
    assert api.post("/v1/evaluations", json=submission()).json() == {"status": "failed"}
    assert api.post("/v1/evaluations", json=submission("new")).status_code == 409
    details = json.loads(next(settings.state_dir.glob("*/details/candidate.json")).read_text())
    assert details["candidate_id"] == "candidate"
    assert len(details["examples"]) == 1
    example = details["examples"][0]
    assert example["example_id"] == "secret-0"
    assert example["failure"] == "solver_failure"
    assert example["resources_complete"] is False
    assert example["run_id"] == run_id
    assert example["resources"]["solver"] == {"usage": usage, "wall_seconds": seconds}
    assert example["resources"]["judge"] == {"usage": empty_usage(), "wall_seconds": 0.0}
    record = json.loads(next(settings.state_dir.glob("*/record.json")).read_text())
    assert len(record["failure"]["message"]) <= 2000


@pytest.mark.parametrize("count", [6, 9])
def test_exact_configurable_private_input_and_output_counts(settings: Settings, count):
    configured = replace(settings, finalists=count, output_count=count - 1)
    api = client(configured)
    data = {"candidates": [
        {"id": f"idea-{index}", "files": {
            "prompt.txt": base64.b64encode(str(index).encode()).decode(),
        }} for index in range(count)
    ]}
    assert api.post("/v1/evaluations", json=data).status_code == 202
    assert api.get("/v1/evaluations").json() == {"status": "complete"}
    assert len(operator_record(settings)["winners"]) == count - 1


@pytest.mark.parametrize("field,value", [
    ("finalists", 0), ("finalists", True), ("finalists", 2.5),
    ("output_count", 0), ("output_count", 2), ("output_count", True),
    ("split", "train"), ("max_body_bytes", 0), ("uses_model", "false"),
])
def test_invalid_service_configuration(settings: Settings, field, value):
    with pytest.raises(ValueError):
        replace(settings, **{field: value})


def test_public_scores_are_hidden_except_aggregate_and_repeated_runs_are_keyed(settings: Settings):
    configured = replace(settings, split="public")
    calls = []

    def solve(candidate, task):
        calls.append(task)
        return solved("private-answer")

    api = client(configured, solve=solve)
    first = api.post("/v1/evaluations", json=submission()).json()
    identifier = first["evaluation_id"]
    assert len(identifier) == 64
    status = api.get(f"/v1/evaluations/{identifier}").json()
    assert status == {
        "status": "complete", "evaluation_id": identifier,
        "scores": [{"candidate_id": "candidate", "score": 4.0}],
    }
    assert api.post("/v1/evaluations", json=submission()).json() == status
    assert len(calls) == 2
    second = api.post("/v1/evaluations", json=submission("different")).json()
    assert second["evaluation_id"] != identifier
    assert len(calls) == 4
    assert api.get(f"/v1/evaluations/{identifier}").json() == status
    assert api.get("/v1/evaluations").status_code == 422
    assert api.get("/v1/evaluations/" + "0" * 64).status_code == 404
    assert api.get("/v1/evaluations/not-an-id").status_code == 404
    details = operator_record(settings, identifier)
    assert details["scores"][0]["solver_tokens"] == 6
    assert client(configured).get(f"/v1/evaluations/{identifier}").json() == status


def test_public_batch_size_does_not_require_finalist_count(settings: Settings):
    api = client(replace(settings, split="public", finalists=9, output_count=9))
    response = api.post("/v1/evaluations", json=submission())
    assert response.status_code == 202
    identifier = response.json()["evaluation_id"]
    assert api.get(f"/v1/evaluations/{identifier}").json()["status"] == "complete"


def test_public_concurrent_batches_keep_results_and_usage_separate(settings: Settings):
    configured = replace(settings, split="public")
    barrier = threading.Barrier(2)

    def solve(candidate, task):
        barrier.wait(timeout=5)
        return solved((candidate / "prompt.txt").read_text())

    def complete(messages):
        return '{"score": 4, "reason": "hidden"}'

    api = client(configured, solve=solve, complete=complete)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(api.post, "/v1/evaluations", json=submission("a"))
        second = pool.submit(api.post, "/v1/evaluations", json=submission("b"))
        responses = [first.result(), second.result()]
    assert all(response.status_code == 202 for response in responses)
    identifiers = [response.json()["evaluation_id"] for response in responses]
    assert len(set(identifiers)) == 2
    for identifier in identifiers:
        assert api.get(f"/v1/evaluations/{identifier}").json()["status"] == "complete"
        record = operator_record(settings, identifier)
        assert record["scores"][0]["judge_unreported_attempts"] == 6
        assert record["scores"][0]["judge_tokens"] == 0


def test_private_ids_do_not_expose_closed_records(settings: Settings):
    from evaluator.contracts import Submission

    api = client(settings)
    data = submission()
    api.post("/v1/evaluations", json=data)
    identifier = Submission.model_validate(data).digest()
    assert api.get(f"/v1/evaluations/{identifier}").status_code == 404


def test_public_configuration_cannot_change_between_submissions(settings: Settings):
    configured = replace(settings, split="public")
    api = client(configured)
    api.post("/v1/evaluations", json=submission())
    with pytest.raises(ValueError, match="frozen"):
        client(replace(configured, parameters={"temperature": 1}))


def test_public_interrupted_batch_fails_without_blocking_other_submissions(settings: Settings):
    configured = replace(settings, split="public")
    api = client(configured)
    identifier = api.post("/v1/evaluations", json=submission()).json()["evaluation_id"]
    record_path = next(settings.state_dir.glob(f"*/public/{identifier}/record.json"))
    record = json.loads(record_path.read_text())
    record["status"] = "running"
    record_path.write_text(json.dumps(record))
    restarted = client(configured)
    assert restarted.get(f"/v1/evaluations/{identifier}").json() == {
        "status": "failed", "evaluation_id": identifier,
    }
    response = restarted.post("/v1/evaluations", json=submission("new"))
    assert response.status_code == 202


def test_empty_public_requires_no_evaluator_deployment(settings: Settings):
    settings.dataset_path.write_text("")
    with pytest.raises(ValueError, match="nonempty"):
        client(replace(settings, split="public"))


@pytest.fixture
def command_settings(settings: Settings):
    program = settings.metric_path.parent / "score.py"
    program.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "print(json.dumps({'score': float(payload['answer'])}))\n"
    )
    executable = Path(sys.executable).resolve()
    settings.metric_path.write_text(json.dumps({
        "version": "command-v1", "kind": "command", "direction": "minimize",
        "command": {
            "argv": [str(executable), str(program)],
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (executable, program)
            },
        },
    }))
    settings.dataset_path.write_text('{"id":"input","input":"task"}\n')
    return replace(settings, uses_model=False, model="", model_base_url="")


def test_model_free_private_metric_uses_direction_and_resource_tiebreak(
    command_settings: Settings, monkeypatch,
):
    import evaluator.service as service

    def forbidden_model(*args, **kwargs):
        pytest.fail("deterministic evaluation must not construct a model client")

    monkeypatch.setattr(service, "ModelClient", forbidden_model)
    configured = replace(command_settings, finalists=3, output_count=2)

    def solve(candidate, task):
        values = {"slow-free": ("-3", 0, 50), "fast-model": ("-3", 1, 1), "worse": ("7", 0, 1)}
        answer, tokens, seconds = values[candidate.name]
        return SolveResult(answer, {**empty_usage(), "total_tokens": tokens,
                                    "prompt_tokens": tokens, "attempts": int(tokens > 0)}, seconds)

    api = TestClient(create_app(configured, solve=solve), headers={
        "Authorization": "Bearer test-token",
    })
    data = {"candidates": [
        {"id": name, "files": {"program.txt": base64.b64encode(name.encode()).decode()}}
        for name in ("worse", "fast-model", "slow-free")
    ]}
    assert api.post("/v1/evaluations", json=data).status_code == 202
    assert api.get("/v1/evaluations").json() == {"status": "complete"}
    record = operator_record(configured)
    assert record["winners"] == ["slow-free", "fast-model"]
    assert all(row["judge_tokens"] == 0 for row in record["scores"])
    assert all(row["judge_seconds"] > 0 for row in record["scores"])


def test_trusted_harness_registry_dispatches_candidate_manifest(settings: Settings, monkeypatch):
    import evaluator.service as service

    invocations = []

    def run(self, candidate, task, *, parameters=None):
        invocations.append((self.image, self.uses_model, self.network, parameters))
        return SolveResult("answer", empty_usage(), 1)

    monkeypatch.setattr(service.DockerRunner, "__call__", run)
    image = "sha256:" + "a" * 64
    alternative = "sha256:" + "b" * 64
    configured = replace(settings, runner_image=image, harnesses={
        "program": {"image": alternative, "uses_model": False, "network": "none"},
    })
    api = TestClient(create_app(configured, complete=lambda _: '{"score":4,"reason":"ok"}'),
                     headers={"Authorization": "Bearer test-token"})
    manifest = json.dumps({"harness": "program", "parameters": {"depth": 4}})
    data = {"candidates": [{"id": "program", "files": {
        "solution.json": base64.b64encode(manifest.encode()).decode(),
    }}]}
    assert api.post("/v1/evaluations", json=data).status_code == 202
    assert api.get("/v1/evaluations").json() == {"status": "complete"}
    assert invocations == [(alternative, False, "none", {"depth": 4})] * 2
    record = operator_record(settings)
    assert record["provenance"]["runner"]["registry"]["program"]["image"] == alternative


def test_service_environment_supports_model_free_public_configuration(
    settings: Settings, monkeypatch,
):
    values = {
        "DATASET_PATH": str(settings.dataset_path), "METRIC_PATH": str(settings.metric_path),
        "STATE_DIR": str(settings.state_dir), "TOKEN": "test-token", "USES_MODEL": "false",
        "RUNNER_IMAGE": "sha256:" + "a" * 64, "RELEASE_ID": "release",
        "FINALISTS": "9", "OUTPUT_COUNT": "3", "SPLIT": "public", "MAX_BODY_BYTES": "300",
        "HARNESSES": json.dumps({"program": {"image": "sha256:" + "b" * 64}}),
    }
    for name, value in values.items():
        monkeypatch.setenv("ARENA_" + name, value)
    monkeypatch.delenv("ARENA_MODEL", raising=False)
    monkeypatch.delenv("ARENA_MODEL_BASE_URL", raising=False)
    loaded = Settings.from_env()
    assert loaded.model == loaded.model_base_url == ""
    assert loaded.split == "public"
    assert loaded.finalists == 9
    assert loaded.output_count == 3
    assert loaded.max_body_bytes == 300
    assert loaded.uses_model is False
    assert loaded.harnesses["program"]["image"].startswith("sha256:")


def test_transport_byte_limit_is_operator_configurable(settings: Settings):
    api = client(replace(settings, max_body_bytes=20))
    assert api.post("/v1/evaluations", json=submission()).status_code == 413


def test_command_metric_same_inputs_with_optional_references(command_settings: Settings):
    command_settings.dataset_path.write_text(
        '{"id":"one","input":"task"}\n'
        '{"id":"two","input":"task","reference":"reference"}\n'
    )
    api = client(command_settings, solve=lambda candidate, task: solved("2"))
    api.post("/v1/evaluations", json=submission())
    assert api.get("/v1/evaluations").json() == {"status": "complete"}


@pytest.mark.parametrize("fails", [False, True])
def test_dataset_aggregation_is_configured_and_measured(command_settings: Settings, fails):
    command_settings.dataset_path.write_text(
        '{"id":"one","input":"first"}\n'
        '{"id":"two","input":"second"}\n'
    )
    executable = Path(sys.executable).resolve()
    aggregator = command_settings.metric_path.parent / "aggregate.py"
    aggregator.write_text(
        "import json, sys\n"
        "records = json.load(sys.stdin)['results']\n"
        + ("raise RuntimeError('closed failure')\n" if fails else
           "print(json.dumps({'score': sum(item['score'] for item in records)}))\n")
    )
    metric = json.loads(command_settings.metric_path.read_text())
    metric["aggregation"] = {
        "argv": [str(executable), str(aggregator)],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (executable, aggregator)
        },
    }
    command_settings.metric_path.write_text(json.dumps(metric))
    api = client(command_settings, solve=lambda candidate, task: solved("2"))
    api.post("/v1/evaluations", json=submission())
    assert api.get("/v1/evaluations").json() == {
        "status": "failed" if fails else "complete",
    }
    path = next(command_settings.state_dir.glob("*/details/candidate.json"))
    details = json.loads(path.read_text())
    seconds = details["aggregation"]["resources"]["judge"]["wall_seconds"]
    assert seconds > 0
    assert details["aggregation"]["resources"]["judge"]["usage"] == empty_usage()
    if not fails:
        record = operator_record(command_settings)
        assert record["scores"][0]["score"] == 4
        assert record["scores"][0]["judge_seconds"] == seconds + sum(
            item["resources"]["judge"]["wall_seconds"] for item in details["examples"]
        )
    else:
        assert "scores" not in operator_record(command_settings)


@pytest.mark.parametrize("split", ["private", "public"])
def test_restart_does_not_overwrite_a_just_completed_run(settings: Settings, monkeypatch, split):
    configured = replace(settings, split=split)
    api = client(configured)
    response = api.post("/v1/evaluations", json=submission()).json()
    identifier = response.get("evaluation_id")
    pattern = "*/record.json" if split == "private" else f"*/public/{identifier}/record.json"
    path = next(settings.state_dir.glob(pattern))
    finished = json.loads(path.read_text())
    path.write_text(json.dumps({**finished, "status": "running"}))
    flock = fcntl.flock

    def finish_before_recovery(descriptor, operation):
        flock(descriptor, operation)
        if operation == fcntl.LOCK_EX | fcntl.LOCK_NB:
            path.write_text(json.dumps(finished))

    monkeypatch.setattr(fcntl, "flock", finish_before_recovery)
    restarted = client(configured)
    endpoint = "/v1/evaluations" if split == "private" else f"/v1/evaluations/{identifier}"
    assert restarted.get(endpoint).json()["status"] == "complete"
    assert json.loads(path.read_text()) == finished


def test_public_renamed_content_reuses_answer_and_score(settings: Settings):
    configured = replace(settings, split="public")
    calls = []
    judges = []

    def solve(candidate, task):
        calls.append(task)
        return solved("answer")

    def complete(messages):
        judges.append(messages)
        return '{"score":4,"reason":"hidden"}'

    api = client(configured, solve=solve, complete=complete)
    original = submission()
    first = api.post("/v1/evaluations", json=original).json()
    renamed = json.loads(json.dumps(original))
    renamed["candidates"][0]["id"] = "new-name"
    second = api.post("/v1/evaluations", json=renamed).json()
    assert api.get(f"/v1/evaluations/{first['evaluation_id']}").json()["scores"] == [
        {"candidate_id": "candidate", "score": 4.0},
    ]
    assert api.get(f"/v1/evaluations/{second['evaluation_id']}").json()["scores"] == [
        {"candidate_id": "new-name", "score": 4.0},
    ]
    assert calls == ["task-0", "task-1"]
    assert len(judges) == 6


def test_public_rejects_queued_candidate_batches(settings: Settings):
    calls = []
    api = client(replace(settings, split="public"), solve=lambda *args: calls.append(args))
    data = {"candidates": [
        {"id": name, "files": {"prompt.txt": base64.b64encode(name.encode()).decode()}}
        for name in ("first", "second", "third")
    ]}
    response = api.post("/v1/evaluations", json=data)
    assert response.status_code == 422
    assert calls == []
    assert not list(settings.state_dir.rglob("record.json"))


def test_public_concurrent_renamed_content_solves_once(settings: Settings):
    started = threading.Event()
    finish = threading.Event()
    calls = []

    def solve(candidate, task):
        calls.append(task)
        started.set()
        assert finish.wait(5)
        return solved("answer")

    api = client(replace(settings, split="public"), solve=solve)
    original = submission()
    renamed = json.loads(json.dumps(original))
    renamed["candidates"][0]["id"] = "renamed"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(api.post, "/v1/evaluations", json=original)
        try:
            assert started.wait(5)
            second = pool.submit(api.post, "/v1/evaluations", json=renamed)
        finally:
            finish.set()
        responses = [first.result(), second.result()]
    assert all(response.status_code == 202 for response in responses)
    for response in responses:
        identifier = response.json()["evaluation_id"]
        assert api.get(f"/v1/evaluations/{identifier}").json()["status"] == "complete"
    assert calls == ["task-0", "task-1"]


def test_public_failed_content_cannot_repeat_a_successful_partial_answer(settings: Settings):
    calls = []

    def solve(candidate, task):
        calls.append(task)
        if task == "task-1":
            raise RunnerError("failure", usage=empty_usage(), wall_seconds=1)
        return solved("answer")

    api = client(replace(settings, split="public"), solve=solve)
    data = submission()
    first = api.post("/v1/evaluations", json=data).json()
    renamed = json.loads(json.dumps(data))
    renamed["candidates"][0]["id"] = "new-name"
    second = api.post("/v1/evaluations", json=renamed).json()
    for identifier in (first["evaluation_id"], second["evaluation_id"]):
        assert api.get(f"/v1/evaluations/{identifier}").json()["status"] == "failed"
    assert calls == ["task-0", "task-1"]


def test_public_dataset_ids_are_frozen_for_content_cache(command_settings: Settings):
    configured = replace(command_settings, split="public")
    metric = json.loads(configured.metric_path.read_text())
    program = Path(metric["command"]["argv"][1])
    program.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "print(json.dumps({'score': int(payload['example']['id'])}))\n"
    )
    digest = hashlib.sha256(program.read_bytes()).hexdigest()
    metric["command"]["source_sha256"][str(program)] = digest
    configured.metric_path.write_text(json.dumps(metric))
    configured.dataset_path.write_text('{"id":"1","input":"task"}\n')
    api = client(configured, solve=lambda candidate, task: solved("2"))
    identifier = api.post("/v1/evaluations", json=submission()).json()["evaluation_id"]
    assert api.get(f"/v1/evaluations/{identifier}").json()["scores"] == [
        {"candidate_id": "candidate", "score": 1.0},
    ]
    configured.dataset_path.write_text('{"id":"2","input":"task"}\n')
    with pytest.raises(ValueError, match="frozen"):
        client(configured, solve=lambda candidate, task: solved("2"))


@pytest.mark.parametrize("split", ["private", "public"])
def test_dataset_order_is_frozen_for_task_aggregation(command_settings: Settings, split):
    configured = replace(command_settings, split=split)
    metric = json.loads(configured.metric_path.read_text())
    executable = Path(sys.executable).resolve()
    aggregator = configured.metric_path.parent / "ordered-aggregation.py"
    aggregator.write_text(
        "import json, sys\n"
        "records = json.load(sys.stdin)['results']\n"
        "print(json.dumps({'score': int(records[0]['example_id'])}))\n"
    )
    metric["aggregation"] = {
        "argv": [str(executable), str(aggregator)],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (executable, aggregator)
        },
    }
    configured.metric_path.write_text(json.dumps(metric))
    records = ['{"id":"1","input":"first"}', '{"id":"2","input":"second"}']
    configured.dataset_path.write_text("\n".join(records) + "\n")
    api = client(configured, solve=lambda candidate, task: solved("2"))
    response = api.post("/v1/evaluations", json=submission()).json()
    assert operator_record(configured, response.get("evaluation_id"))["scores"][0]["score"] == 1.0
    configured.dataset_path.write_text("\n".join(reversed(records)) + "\n")
    with pytest.raises(ValueError, match="frozen"):
        client(configured, solve=lambda candidate, task: solved("2"))


@pytest.mark.parametrize("text", ["", " \t"])
def test_command_metrics_allow_empty_inputs_and_references(command_settings: Settings, text):
    command_settings.dataset_path.write_text(json.dumps({
        "id": "empty", "input": text, "reference": text,
    }) + "\n")
    received = []

    def solve(candidate, task):
        received.append(task)
        return solved("2")

    api = client(command_settings, solve=solve)
    api.post("/v1/evaluations", json=submission())
    assert api.get("/v1/evaluations").json() == {"status": "complete"}
    assert received == [text]
    assert operator_record(command_settings)["scores"][0]["score"] == 2



def test_public_background_work_does_not_starve_requests(settings: Settings, monkeypatch):
    import evaluator.service as service

    configured = replace(settings, split="public")
    started = threading.Event()
    admitted = threading.Event()
    finish = threading.Event()
    lock = threading.Lock()
    calls = 0
    reservations = 0
    write_json = service._write_json

    def record_reservation(path, value):
        nonlocal reservations
        write_json(path, value)
        if path.name == "record.json" and value.get("status") == "running":
            with lock:
                reservations += 1
                if reservations == 3:
                    admitted.set()

    def solve(candidate, task):
        nonlocal calls
        with lock:
            calls += 1
            if calls == 2:
                started.set()
        assert finish.wait(5)
        return solved()

    monkeypatch.setattr(service, "_write_json", record_reservation)
    app = create_app(
        configured, solve=solve, complete=lambda _: '{"score":4,"reason":"hidden"}',
    )

    async def requests():
        limiter = current_default_thread_limiter()
        previous_tokens = limiter.total_tokens
        limiter.total_tokens = 2
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://evaluator",
            headers={"Authorization": "Bearer test-token"},
        ) as api:
            pending = [
                asyncio.create_task(api.post("/v1/evaluations", json=submission(value)))
                for value in ("first", "second")
            ]
            try:
                assert await asyncio.to_thread(started.wait, 2)
                health = await asyncio.wait_for(api.get("/healthz"), timeout=1)
                assert health.status_code == 200
                status = await asyncio.wait_for(
                    api.get("/v1/evaluations/" + "0" * 64), timeout=1,
                )
                assert status.status_code == 404
                pending.append(asyncio.create_task(api.post(
                    "/v1/evaluations", json=submission("third"),
                )))
                assert await asyncio.to_thread(admitted.wait, 1)
            finally:
                finish.set()
                await asyncio.gather(*pending)
                limiter.total_tokens = previous_tokens

    asyncio.run(requests())


@pytest.mark.parametrize("overrides", [
    {"parameters": {"stream": False}},
    {"parameters": {"messages": []}},
    {"parameters": {"temperature": float("nan")}},
    {"model_base_url": "ftp://internal/v1"},
    {"model_base_url": "http://internal:not-a-port/v1"},
    {"model_base_url": "http://internal:70000/v1"},
    {"model_base_url": "http://user:secret@internal/v1"},
])
@pytest.mark.parametrize("injected", [False, True])
def test_invalid_judge_configuration_fails_before_reserving_private(settings, overrides, injected):
    configured = replace(settings, **overrides)
    complete = (lambda _: '{"score":4,"reason":"hidden"}') if injected else None
    with pytest.raises(ValueError):
        create_app(configured, solve=lambda *args: solved(), complete=complete)
    assert not list(settings.state_dir.rglob("record.json"))
    api = client(settings)
    assert api.post("/v1/evaluations", json=submission()).status_code == 202
    assert api.get("/v1/evaluations").json() == {"status": "complete"}


def test_public_starts_without_waiting_for_shared_executor(settings):
    started = threading.Event()
    release = threading.Event()
    executor_busy = threading.Event()
    app = create_app(
        replace(settings, split="public"),
        solve=lambda *args: (started.set(), solved())[1],
        complete=lambda _: '{"score":4,"reason":"hidden"}',
    )

    def occupy():
        executor_busy.set()
        assert release.wait(5)

    async def requests():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop.set_default_executor(executor)
            busy = loop.run_in_executor(None, occupy)
            assert executor_busy.wait(2)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://evaluator",
                headers={"Authorization": "Bearer test-token"},
            ) as api:
                pending = asyncio.create_task(api.post("/v1/evaluations", json=submission()))
                try:
                    response = await asyncio.wait_for(asyncio.shield(pending), timeout=1)
                    assert response.status_code == 202
                    assert started.is_set()
                finally:
                    release.set()
                    await busy
                    await pending

    asyncio.run(requests())


@pytest.mark.parametrize("stage", ["command", "aggregation"])
@pytest.mark.parametrize("split", ["private", "public"])
def test_metric_diagnostics_are_saved_only_in_closed_logs(command_settings, stage, split):
    configured = replace(command_settings, split=split)
    executable = Path(sys.executable).resolve()
    script = configured.metric_path.parent / "failing_metric.py"
    script.write_text(
        "import sys\n"
        "print('CLOSED_STDOUT', flush=True)\n"
        "raise RuntimeError('CLOSED_METRIC_DIAGNOSTIC')\n"
    )
    metric = json.loads(configured.metric_path.read_text())
    metric[stage] = {
        "argv": [str(executable), str(script)],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (executable, script)
        },
    }
    configured.metric_path.write_text(json.dumps(metric))
    api = client(configured, solve=lambda *args: solved("2"))
    accepted = api.post("/v1/evaluations", json=submission())
    identifier = accepted.json().get("evaluation_id")
    endpoint = f"/v1/evaluations/{identifier}" if identifier else "/v1/evaluations"
    response = api.get(endpoint)
    assert response.json()["status"] == "failed"
    assert "CLOSED_" not in accepted.text + response.text
    record = operator_record(configured, identifier)
    diagnostics = record["failure"]["diagnostics"]
    assert diagnostics["returncode"] == 1
    assert "CLOSED_METRIC_DIAGNOSTIC" in diagnostics["stderr"]
    assert diagnostics["stdout"] == "CLOSED_STDOUT\n"
    paths = list(configured.state_dir.rglob("details/candidate.json"))
    assert len(paths) == 1
    details = json.loads(paths[0].read_text())
    failed = details["examples"][0] if stage == "command" else details["aggregation"]
    assert failed["diagnostics"] == diagnostics
    assert failed["resources"]["judge"]["wall_seconds"] > 0


def test_private_materialization_preserves_executable_helper(settings):
    def solve(candidate, task):
        result = subprocess.run(
            [str(candidate / "bin" / "helper.sh")], capture_output=True, check=True,
        )
        return solved(result.stdout.decode())

    source = settings.state_dir.parent / "source"
    (source / "bin").mkdir(parents=True)
    helper = source / "bin" / "helper.sh"
    helper.write_bytes(b"#!/bin/sh\nprintf answer\n")
    helper.chmod(0o755)
    candidate = pack_candidate("candidate", source)
    restored = settings.state_dir.parent / "restored"
    materialize_candidate(candidate, restored)
    assert solve(restored, "train").answer == "answer"
    assert pack_candidate("candidate", restored).digest() == candidate.digest()
    api = client(settings, solve=solve)
    data = {"candidates": [candidate.model_dump()]}
    assert api.post("/v1/evaluations", json=data).status_code == 202
    assert api.get("/v1/evaluations").json() == {"status": "complete"}
    record = operator_record(settings)
    assert record["scores"][0]["digest"] == candidate.digest()
    assert pack_candidate("candidate", Path(record["candidates"][0]["path"])).digest() == (
        candidate.digest()
    )


def test_worker_start_failure_releases_private_execution_lease(settings, monkeypatch):
    original_start = threading.Thread.start

    def fail_worker(thread):
        if getattr(getattr(thread, "_target", None), "__name__", None) == "execute":
            raise RuntimeError("cannot start worker")
        return original_start(thread)

    api = client(settings)
    monkeypatch.setattr(threading.Thread, "start", fail_worker)
    response = api.post("/v1/evaluations", json=submission())
    assert response.status_code == 503
    assert response.json() == {"detail": "evaluation unavailable"}
    assert api.get("/v1/evaluations").json() == {"status": "failed"}
    lock_path = next(settings.state_dir.glob("*/execution.lock"))
    with lock_path.open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert client(settings).post("/v1/evaluations", json=submission()).json() == {
        "status": "failed",
    }
