import base64
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fastapi.testclient import TestClient

from evaluator.contracts import Example
from evaluator.metric_package import freeze_metric, load_metric, metric_package_provenance
from evaluator.metrics import aggregate_results, evaluate_answer
from evaluator.reliability import empty_usage
from evaluator.runner import SolveResult
from evaluator.service import Settings, create_app


def candidate(candidate_id: str, value: str) -> dict[str, Any]:
    return {"id": candidate_id, "files": {
        "prompt.txt": base64.b64encode(value.encode()).decode(),
    }}


def operator_record(settings: Settings) -> dict[str, Any]:
    paths = list(settings.state_dir.glob("*/record.json"))
    assert len(paths) == 1, paths
    return json.loads(paths[0].read_text())


def check_model_judge(root: Path) -> None:
    canary = "fictional-private-smoke-canary"
    votes = 2
    solve_calls: list[tuple[str, str]] = []
    judge_calls: list[tuple[str, str]] = []

    def solve(candidate: Path, task: str) -> SolveResult:
        prompt = (candidate / "prompt.txt").read_text()
        solve_calls.append((prompt, task))
        return SolveResult(json.dumps({"prompt": prompt, "task": task, "private": canary}), {
            "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
            "attempts": 1, "unreported_attempts": 0,
        }, 1.0)

    def complete(messages: list[dict[str, str]]) -> str:
        payload = json.loads(messages[-1]["content"])
        criterion = payload["criterion"]["id"]
        answer = json.loads(payload["answer"])
        assert canary in payload["reference"]
        judge_calls.append((payload["answer"], criterion))
        scores = {"alpha": {"correct": 4, "complete": 2}, "beta": {
            "correct": 3, "complete": 3,
        }}
        return json.dumps({"score": scores[answer["prompt"]][criterion], "reason": canary})

    dataset = root / "model-private.jsonl"
    tasks = [f"Fictional task {i}: {canary}" for i in range(2)]
    dataset.write_text("\n".join(
        json.dumps({"id": f"private-{i}", "input": task, "reference": canary})
        for i, task in enumerate(tasks)
    ))
    metric = root / "model-metric.json"
    metric.write_text(json.dumps({
        "version": "smoke-v1", "instruction": "Evaluate fictional answers.",
        "votes": votes,
        "criteria": [
            {"id": "correct", "question": "Is it correct?", "weight": 3},
            {"id": "complete", "question": "Is it complete?", "weight": 1},
        ],
    }))
    settings = Settings(
        dataset_path=dataset, metric_path=metric, state_dir=root / "model-state",
        token="smoke-token", model="fake-internal-model",
        model_base_url="http://model.invalid/v1", runner_image="fake-runner",
        release_id="smoke-release", finalists=2,
    )
    submission = {"candidates": [candidate(name, name) for name in ("alpha", "beta")]}
    with TestClient(create_app(settings, solve=solve, complete=complete)) as client:
        client.headers["Authorization"] = "Bearer smoke-token"
        accepted = client.post("/v1/evaluations", json=submission)
        assert accepted.status_code == 202, accepted.text
        assert accepted.json() == {"status": "running"}, accepted.text
        final = client.get("/v1/evaluations")
        assert final.status_code == 200, final.text
        assert final.json() == {"status": "complete"}, final.text
        result = operator_record(settings)
        assert result["status"] == "complete", result
        assert result["winners"] == ["alpha"], result
        details = list(settings.state_dir.glob("*/details/*.json"))
        assert len(details) == 2, details
        assert all(canary in path.read_text() for path in details), details
        assert all(item["solver_tokens"] == 10 for item in result["scores"]), result
        assert all(item["solver_seconds"] == 2.0 for item in result["scores"]), result
        scores = {item["candidate_id"]: item["score"] for item in result["scores"]}
        assert set(scores) == {"alpha", "beta"}, result
        assert math.isclose(scores["alpha"], 3.5), result
        assert math.isclose(scores["beta"], 3.0), result
        assert Counter(solve_calls) == Counter(
            (name, task) for name in ("alpha", "beta") for task in tasks
        ), solve_calls
        assert len(Counter(judge_calls)) == 8, judge_calls
        assert set(Counter(judge_calls).values()) == {votes}, judge_calls
        repeated = client.post("/v1/evaluations", json=submission)
        assert repeated.status_code == 200, repeated.text
        assert repeated.json() == final.json(), repeated.text
        changed = {"candidates": [candidate("alpha", "changed"), candidate("beta", "beta")]}
        rejected = client.post("/v1/evaluations", json=changed)
        assert rejected.status_code == 409, rejected.text
        for response in (accepted, final, repeated, rejected):
            assert canary not in response.text, response.text
        assert len(solve_calls) == 4, solve_calls
        assert len(judge_calls) == 8 * votes, judge_calls


def check_command_metric(root: Path) -> None:
    canary = "fictional-held-out-command-canary"
    calls: list[tuple[str, str]] = []

    def solve(candidate: Path, task: str) -> SolveResult:
        value = (candidate / "prompt.txt").read_text()
        calls.append((value, task))
        answer = float(value) + json.loads(task)["offset"]
        return SolveResult(str(answer), empty_usage(), 0.25)

    def unexpected_model(messages: list[dict[str, str]]) -> str:
        raise AssertionError("command metrics must not call a model")

    draft = root / "command-metric-draft"
    draft.mkdir()
    script = draft / "metric.py"
    script.write_text(
        "import json\n"
        "import sys\n"
        "payload = json.load(sys.stdin)\n"
        "offset = json.loads(payload['example']['input'])['offset']\n"
        "print(json.dumps({'score': float(payload['answer']) - offset}))\n"
    )
    (draft / "metric.json").write_text(json.dumps({
        "version": "command-smoke-v1", "kind": "command", "direction": "minimize",
        "command": {"argv": ["python", "metric.py"]},
    }))
    metric = root / "command-metric-frozen"
    package = freeze_metric(draft, metric)
    assert metric_package_provenance(metric) == package
    assert set(package["files"]) == {"metric.json", "metric.py"}
    prepared_metric = load_metric(metric)
    train_results = [
        evaluate_answer(
            prepared_metric, Example(id=f"train-{i}", input=json.dumps({"offset": i})),
            str(i - 3),
        )
        for i in range(2)
    ]
    assert aggregate_results(prepared_metric, train_results) == -3.0, train_results
    dataset = root / "command-dataset.jsonl"
    tasks = [json.dumps({"offset": i, "hidden": canary}) for i in range(2)]
    dataset.write_text("\n".join(
        json.dumps({"id": f"held-out-{i}", "input": task, "reference": None})
        for i, task in enumerate(tasks)
    ))
    settings = Settings(
        dataset_path=dataset, metric_path=metric, state_dir=root / "command-private-state",
        token="smoke-token", runner_image="fake-program-runner", release_id="command-smoke",
        finalists=7, output_count=6, uses_model=False,
    )
    submission = {"candidates": [candidate(f"way-{i}", str(i - 3)) for i in range(7)]}
    with TestClient(create_app(settings, solve=solve, complete=unexpected_model)) as client:
        client.headers["Authorization"] = "Bearer smoke-token"
        too_few = client.post("/v1/evaluations", json={
            "candidates": submission["candidates"][:-1],
        })
        assert too_few.status_code == 422, too_few.text
        accepted = client.post("/v1/evaluations", json=submission)
        assert accepted.status_code == 202, accepted.text
        assert accepted.json() == {"status": "running"}, accepted.text
        final = client.get("/v1/evaluations")
        assert final.status_code == 200, final.text
        assert final.json() == {"status": "complete"}, final.text
        result = operator_record(settings)
        assert result["provenance"]["metric_package"] == package, result
        assert result["winners"] == [f"way-{i}" for i in range(6)], result
        assert {item["candidate_id"]: item["score"] for item in result["scores"]} == {
            f"way-{i}": i - 3 for i in range(7)
        }, result
        for item in result["scores"]:
            assert item["solver_tokens"] == item["judge_tokens"] == 0, item
            assert item["solver_unreported_attempts"] == item["judge_unreported_attempts"] == 0
            assert item["solver_seconds"] == 0.5, item
            assert item["judge_seconds"] >= 0, item
        repeated = client.post("/v1/evaluations", json=submission)
        assert repeated.status_code == 200, repeated.text
        assert repeated.json() == {"status": "complete"}, repeated.text
        assert Counter(calls) == Counter(
            (str(i - 3), task) for i in range(7) for task in tasks
        ), calls

    calls.clear()
    public_settings = replace(
        settings, split="public", finalists=1, output_count=1,
        state_dir=root / "command-public-state",
    )
    with TestClient(create_app(
        public_settings, solve=solve, complete=unexpected_model,
    )) as client:
        client.headers["Authorization"] = "Bearer smoke-token"
        evaluations: list[str] = []
        for value in ("-2", "-4"):
            batch = {"candidates": [candidate("search", value)]}
            accepted = client.post("/v1/evaluations", json=batch)
            assert accepted.status_code == 202, accepted.text
            assert set(accepted.json()) == {"status", "evaluation_id"}, accepted.text
            evaluation_id = accepted.json()["evaluation_id"]
            assert evaluation_id not in evaluations, evaluations
            evaluations.append(evaluation_id)
            final = client.get(f"/v1/evaluations/{evaluation_id}")
            assert final.status_code == 200, final.text
            assert final.json() == {
                "status": "complete", "evaluation_id": evaluation_id,
                "scores": [{"candidate_id": "search", "score": float(value)}],
            }, final.text
            repeated = client.post("/v1/evaluations", json=batch)
            assert repeated.status_code == 200, repeated.text
            assert repeated.json() == final.json(), repeated.text
            assert canary not in final.text, final.text
            records = list(public_settings.state_dir.glob(f"*/public/{evaluation_id}/record.json"))
            assert len(records) == 1, records
            assert json.loads(records[0].read_text())["provenance"]["metric_package"] == package
        first = client.get(f"/v1/evaluations/{evaluations[0]}")
        assert first.json()["scores"] == [{"candidate_id": "search", "score": -2.0}]
        assert Counter(calls) == Counter(
            (value, task) for value in ("-2", "-4") for task in tasks
        ), calls


def main() -> None:
    with TemporaryDirectory(prefix="arena-smoke-") as temporary:
        root = Path(temporary)
        check_model_judge(root)
        check_command_metric(root)
    print(
        "smoke passed: single answers, configurable criterion votes, sealed private, "
        "multiple outputs, score-only public, prepared metric across splits, model-free usage"
    )


if __name__ == "__main__":
    main()
