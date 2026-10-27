import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from evaluator.client import main
from evaluator.reliability import empty_usage
from evaluator.runner import RunnerError, SolveResult
from evaluator.service import Settings, create_app


@pytest.mark.parametrize("split", ["public", "private"])
@pytest.mark.parametrize("outcome", ["failed", "complete"])
def test_wait_consumes_real_terminal_service_response_without_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    split: str, outcome: str,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({
        "id": "hidden-example", "input": "hidden-question", "reference": "hidden-reference",
    }) + "\n")
    metric = tmp_path / "metric.json"
    metric.write_text(json.dumps({
        "version": "synthetic-v1", "instruction": "judge", "criteria": [
            {"id": "correct", "question": "Correct?", "weight": 1},
        ],
    }))
    token = tmp_path / "token"
    token.write_text("test-token\n")
    settings = Settings(
        dataset_path=dataset, metric_path=metric, state_dir=tmp_path / "state",
        token="test-token", model="synthetic-model", model_base_url="http://synthetic/v1",
        runner_image="synthetic-runner", release_id="synthetic-release", finalists=1, split=split,
    )
    executions = []

    def solve(candidate, task):
        executions.append(task)
        if outcome == "failed":
            raise RunnerError("hidden-failure-details")
        return SolveResult(" ", empty_usage(), 0.0)

    def complete(messages):
        pytest.fail("These synthetic outcomes must not call a model")

    with TestClient(create_app(settings, solve=solve, complete=complete)) as api:
        submitted = api.post(
            "/v1/evaluations", headers={"Authorization": "Bearer test-token"},
            json={"candidates": [{"id": "candidate", "files": {"prompt.txt": "YQ=="}}]},
        )
        assert submitted.status_code == 202
        arguments = [
            "wait", "--url", "http://evaluator", "--token-file", str(token),
            "--output", str(tmp_path / "status.json"),
        ]
        expected: dict[str, object] = {"status": outcome}
        if split == "public":
            evaluation_id = submitted.json()["evaluation_id"]
            arguments.extend(["--evaluation-id", evaluation_id])
            expected["evaluation_id"] = evaluation_id
            if outcome == "complete":
                expected["scores"] = [{"candidate_id": "candidate", "score": 0.0}]
        calls = []

        def handle(request: httpx.Request) -> httpx.Response:
            calls.append(request.method)
            response = api.request(
                request.method, request.url.path, headers=dict(request.headers),
            )
            return httpx.Response(response.status_code, json=response.json())

        original_client = httpx.Client
        monkeypatch.setattr(
            httpx, "Client",
            lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
        )
        monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("Terminal status must not wait"))
        assert main(arguments) == (3 if outcome == "failed" else 0)
        assert calls == ["GET"]
        assert executions == ["hidden-question"]
        saved = (tmp_path / "status.json").read_text()
        assert json.loads(saved) == expected
        captured = capsys.readouterr()
        assert json.loads(captured.out) == expected
        assert captured.err == ""
        assert "hidden-" not in saved + captured.out
