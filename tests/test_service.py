import base64
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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


def test_evaluates_once_and_exposes_only_final_scores(settings: Settings):
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
    assert result["winner"] == "candidate"
    assert len(result["scores"]) == 1
    assert set(result["scores"][0]) == {
        "candidate_id", "digest", "score", "solver_tokens", "solver_seconds",
        "solver_unreported_attempts", "judge_tokens", "judge_seconds", "judge_unreported_attempts",
    }
    resources = result["scores"][0]
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


def test_blank_answer_scores_zero_without_judge(settings: Settings):
    def complete(messages):
        pytest.fail("blank answer must not reach judge")

    api = client(settings, solve=lambda candidate, task: solved("  "), complete=complete)
    api.post("/v1/evaluations", json=submission())
    result = api.get("/v1/evaluations").json()
    assert result["status"] == "complete"
    assert result["scores"][0]["score"] == 0
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
    assert api.get("/v1/evaluations").json()["winner"] == "record"
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
    assert len(result["scores"]) == 2
    assert result["winner"] == "second"


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

    monkeypatch.setattr(service, "judge_answer", judge)
    api = client(settings, complete=model_client)
    api.post("/v1/evaluations", json=submission())
    score = api.get("/v1/evaluations").json()["scores"][0]
    assert score["judge_tokens"] == 14
    assert score["judge_unreported_attempts"] == 0
    details = json.loads(next(settings.state_dir.glob("*/details/candidate.json")).read_text())
    assert [item["resources"]["judge"]["usage"]["total_tokens"]
            for item in details["examples"]] == [7, 7]


def test_runner_receives_configurable_long_limits_and_usage_path(settings: Settings, monkeypatch):
    import evaluator.service as service

    captured = {}

    def runner(**kwargs):
        captured.update(kwargs)
        return lambda candidate, task: solved()

    def judge(metric, example, answer, complete, **kwargs):
        assert isinstance(complete, ModelClient)
        assert complete.timeout == 30000
        assert complete.retry_delay == 0
        return {"score": 4.0, "criteria": {}}

    monkeypatch.setattr(service, "DockerRunner", runner)
    monkeypatch.setattr(service, "judge_answer", judge)
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
    assert captured["usage_dir"] == configured.usage_dir


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
