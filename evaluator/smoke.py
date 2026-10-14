import base64
import json
import math
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from evaluator.runner import SolveResult
from evaluator.service import Settings, create_app


def main() -> None:
    canary = "fictional-private-smoke-canary"
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

    with TemporaryDirectory(prefix="arena-smoke-") as temporary:
        root = Path(temporary)
        dataset = root / "private.jsonl"
        tasks = [f"Fictional task {i}: {canary}" for i in range(2)]
        dataset.write_text("\n".join(
            json.dumps({"id": f"private-{i}", "input": task, "reference": canary})
            for i, task in enumerate(tasks)
        ))
        metric = root / "metric.json"
        metric.write_text(json.dumps({
            "version": "smoke-v1", "instruction": "Evaluate fictional answers.",
            "criteria": [
                {"id": "correct", "question": "Is it correct?", "weight": 3},
                {"id": "complete", "question": "Is it complete?", "weight": 1},
            ],
        }))
        settings = Settings(
            dataset_path=dataset, metric_path=metric, state_dir=root / "state",
            token="smoke-token", model="fake-internal-model",
            model_base_url="http://model.invalid/v1", runner_image="fake-runner",
            release_id="smoke-release", finalists=2,
        )
        submission = {"candidates": [
            {"id": name, "files": {"prompt.txt": base64.b64encode(name.encode()).decode()}}
            for name in ("alpha", "beta")
        ]}
        with TestClient(create_app(settings, solve=solve, complete=complete)) as client:
            client.headers["Authorization"] = "Bearer smoke-token"
            accepted = client.post("/v1/evaluations", json=submission)
            assert accepted.status_code == 202, accepted.text
            final = client.get("/v1/evaluations")
            assert final.status_code == 200, final.text
            result = final.json()
            assert result["status"] == "complete", result
            assert result["winner"] == "alpha", result
            assert all(item["solver_tokens"] == 10 for item in result["scores"]), result
            assert all(item["solver_seconds"] == 2.0 for item in result["scores"]), result
            scores = {item["candidate_id"]: item["score"] for item in result["scores"]}
            assert set(scores) == {"alpha", "beta"}, result
            assert math.isclose(scores["alpha"], 3.5), result
            assert math.isclose(scores["beta"], 3.0), result
            assert len(solve_calls) == 4, solve_calls
            assert Counter(solve_calls) == Counter(
                (name, task) for name in ("alpha", "beta") for task in tasks
            ), solve_calls
            assert len(judge_calls) == 24, judge_calls
            assert len(Counter(judge_calls)) == 8, judge_calls
            assert set(Counter(judge_calls).values()) == {3}, judge_calls
            repeated = client.post("/v1/evaluations", json=submission)
            assert repeated.status_code == 200, repeated.text
            assert repeated.json() == result, repeated.text
            changed = json.loads(json.dumps(submission))
            changed["candidates"][0]["files"]["prompt.txt"] = "Y2hhbmdlZA=="
            rejected = client.post("/v1/evaluations", json=changed)
            assert rejected.status_code == 409, rejected.text
            for response in (accepted, final, repeated, rejected):
                assert canary not in response.text, response.text
            assert len(solve_calls) == 4
            assert len(judge_calls) == 24
    print("smoke passed: 4 solves, 24 criterion votes, private responses sealed, retry idempotent")


if __name__ == "__main__":
    main()
