import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE.parents[2]))

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
EXPECTED = "КОД-СЕВЕР-731"
SCENARIOS = ("retrieval", "delegation", "judge")
WEIGHTS = {"completeness": 0.35, "accuracy": 0.35, "groundedness": 0.3}


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def fixture_digest():
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(PACKAGE)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def load_state(path, settings, fixture_hash):
    fingerprint = hashlib.sha256(json.dumps({
        "settings": settings, "fixture_sha256": fixture_hash,
    }, sort_keys=True).encode()).hexdigest()
    if path.exists():
        state = json.loads(path.read_text())
        if state["configuration_sha256"] != fingerprint:
            raise ValueError("acceptance configuration changed; saved answers must be preserved")
        return state
    return {
        "configuration_sha256": fingerprint, "fixture_sha256": fixture_hash,
        "created_at_unix": time.time(), "synthetic_only": True,
        "solution_configurations": 1, "scenario_count": 3,
        "settings": settings, "scenarios": {},
    }


def verify_model(calls):
    if not calls:
        raise ValueError("trusted model traces are missing")
    for call in calls:
        payload = call["inputs"]
        if payload.get("model") != MODEL or payload.get("reasoning_effort") != "max":
            raise ValueError("fixed model or reasoning effort differs")


def first_user(call):
    return next((message.get("content", "")
                 for message in call["inputs"].get("messages", [])
                 if message.get("role") == "user"), "")


def tool_signature(function):
    arguments = json.dumps(
        json.loads(function["arguments"]), sort_keys=True, allow_nan=False,
        separators=(",", ":"),
    )
    return function["name"], arguments


def tool_aliases(calls):
    originals = {}
    normalized = {}
    aliases = {}
    for call in calls:
        for message in call["inputs"].get("messages", []):
            for tool in message.get("tool_calls") or []:
                identifier = tool["id"]
                original = identifier if identifier in originals else normalized.get(identifier)
                if original is None or originals[original] != tool_signature(tool["function"]):
                    raise ValueError("unmatched normalized tool call in trusted transcript")
                aliases[identifier] = original
        if call["status"] == "OK":
            for choice in call["outputs"].get("choices", []):
                tools = choice["message"].get("tool_calls") or []
                normalize = len(tools) >= 2 and all(
                    tool["id"].strip().split("|", 1)[0].strip().startswith("chatcmpl-tool-")
                    for tool in tools
                )
                for tool in tools:
                    originals[tool["id"]] = tool_signature(tool["function"])
                    if normalize:
                        identifier, *suffix = tool["id"].strip().split("|", 1)
                        digest = hashlib.sha256(identifier.strip().encode(
                            "utf-8", "surrogatepass",
                        )).hexdigest()[:12]
                        alias = "call_" + digest + ("|" + suffix[0] if suffix else "")
                        normalized[alias] = tool["id"]
    return {key: key for key in originals} | aliases


def verify_solver(scenario, answer, calls):
    verify_model(calls)
    if answer.strip() != EXPECTED:
        raise ValueError("synthetic answer differs from the exact expected code")
    requested = {}
    responses = {}
    aliases = tool_aliases(calls)
    for call in calls:
        if call["status"] == "OK":
            for choice in call["outputs"].get("choices", []):
                for tool in choice["message"].get("tool_calls", []):
                    requested[tool["id"]] = tool["function"]["name"]
        for message in call["inputs"].get("messages", []):
            if message.get("role") == "tool" and EXPECTED in (message.get("content") or ""):
                identifier = aliases.get(message["tool_call_id"], message["tool_call_id"])
                responses[identifier] = message["content"]
    completed = set()
    for identifier, content in responses.items():
        if identifier not in requested:
            continue
        response = json.loads(content)
        if not isinstance(response, dict) or response.get("error"):
            continue
        results = response.get("results", [])
        if not results:
            continue
        if requested[identifier] == "delegate_task" and any(
            item.get("status") != "completed" for item in results
        ):
            continue
        completed.add(identifier)
    counts = Counter(requested[identifier] for identifier in completed if identifier in requested)
    if counts["arena_search"] < 1:
        raise ValueError("mandatory retrieval was not completed in the trusted trace")
    root_search = any(
        tool["id"] in completed and tool["function"]["name"] == "arena_search"
        for call in calls if call["status"] == "OK" and first_user(call) == first_user(calls[0])
        for choice in call["outputs"].get("choices", [])
        for tool in choice["message"].get("tool_calls", [])
    )
    if not root_search:
        raise ValueError("mandatory root retrieval was not completed in the trusted trace")
    if scenario == "delegation" and counts["delegate_task"] < 2:
        raise ValueError("nested delegation was not completed in the trusted trace")
    nested = verify_nested(calls, completed) if scenario == "delegation" else {}
    return {"exact_answer": True, "completed_tool_calls": dict(counts)} | nested


def verify_nested(calls, completed):
    child_goals = set()
    grandchild_goals = set()
    leaf_search = False
    depth_marker = False
    for call in calls:
        if call["status"] != "OK":
            continue
        goal_message = first_user(call)
        if goal_message in child_goals:
            depth_marker |= any(
                message.get("role") == "system" and (
                    "NOTE: You are at depth 1. The delegation tree is capped at max_spawn_depth=3."
                    in (message.get("content") or "")
                ) for message in call["inputs"].get("messages", [])
            )
        for choice in call["outputs"].get("choices", []):
            for tool in choice["message"].get("tool_calls", []):
                if tool["id"] not in completed:
                    continue
                function = tool["function"]
                if function["name"] == "arena_search" and goal_message in grandchild_goals:
                    leaf_search = True
                if function["name"] != "delegate_task":
                    continue
                arguments = json.loads(function["arguments"])
                tasks = arguments.get("tasks") or [arguments]
                for task in tasks:
                    goal = task.get("goal", "")
                    if (
                        goal.startswith("SC71_ACCEPT_CHILD\n")
                        and goal_message == first_user(calls[0])
                    ):
                        child_goals.add(goal)
                    if goal_message in child_goals and goal.startswith("SC71_ACCEPT_GRANDCHILD\n"):
                        grandchild_goals.add(goal)
    if not child_goals or not grandchild_goals or not leaf_search:
        raise ValueError("nested root-child-grandchild lineage is missing from trusted traces")
    return {"verified_delegation_depth": 2, "native_child_depth_marker": depth_marker}


def verify_judge(result, calls):
    verify_model(calls)
    valid = [call for call in calls if call["status"] == "OK"]
    if len(valid) != 9:
        raise ValueError("judge must produce exactly nine valid votes")
    trace_scores = []
    for call in valid:
        vote = json.loads(call["outputs"]["choices"][0]["message"]["content"])
        score = vote.get("score")
        if (
            type(score) is not int or not 1 <= score <= 5
            or not isinstance(vote.get("reason"), str) or not vote["reason"].strip()
        ):
            raise ValueError("invalid judge vote")
        trace_scores.append(score)
    scores = result["statistics"]["judge_scores"]
    if set(scores) != set(WEIGHTS) or any(len(scores[name]) != 3 for name in WEIGHTS):
        raise ValueError("judge must produce three votes per criterion")
    if trace_scores != [score for name in WEIGHTS for score in scores[name]]:
        raise ValueError("reported judge votes differ from trusted traces")
    expected = sum(WEIGHTS[name] * statistics.mean(scores[name]) for name in WEIGHTS)
    if not math.isclose(result["score"], expected, abs_tol=1e-12):
        raise ValueError("weighted judge score differs")
    if result["score"] < 4:
        raise ValueError("fixed correct answer did not pass the declared judge threshold of 4")
    return {"valid_votes": 9, "weighted_score": expected, "minimum_score": 4}


def verify_usage(usage, calls):
    from evaluator.reliability import add_usage, empty_usage

    if (
        usage["attempts"] < 1 or usage["total_tokens"] < 1
        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
    ):
        raise ValueError("trusted token accounting is missing or inconsistent")
    traced = empty_usage()
    for call in calls:
        add_usage(traced, call["attributes"]["usage"])
    if usage != traced:
        raise ValueError("trusted traces do not cover all attempts and tokens")
    return {"token_accounting_complete": usage["unreported_attempts"] == 0}


def run_scenario(name, state_path, state, operation, verify):
    if name in state["scenarios"]:
        return
    record = {"status": "running", "started_at_unix": time.time()}
    state["scenarios"][name] = record
    save_json(state_path, state)
    started = time.monotonic()
    try:
        record["result"] = operation()
        record["status"] = "answered"
        save_json(state_path, state)
        record["evidence"] = verify(record["result"])
        record["status"] = "accepted"
    except Exception as error:
        record["status"] = "rejected" if "result" in record else "failed"
        record["error_type"] = type(error).__name__
        if "result" in record:
            record["verification_error"] = str(error)
        for field in ("usage", "wall_seconds", "run_id"):
            value = getattr(error, field, None)
            if value is not None:
                record[field] = value
    finally:
        record["wall_seconds"] = time.monotonic() - started
        record["finished_at_unix"] = time.time()
        save_json(state_path, state)
        print(json.dumps({"scenario": name, "status": record["status"]}), flush=True)


def solve(settings, scenario):
    from evaluator.runner import DockerRunner

    parameters = json.loads((PACKAGE / "candidate" / "solution.json").read_text())["parameters"]
    runner = DockerRunner(**{
        **settings["runner"], "model": MODEL, "assets_dir": PACKAGE / "assets",
        "usage_dir": Path(settings["usage_dir"]),
    })
    inputs = json.loads((PACKAGE / "scenarios.json").read_text())
    return asdict(runner(PACKAGE / "candidate", inputs[scenario], parameters=parameters))


def judge(settings):
    from evaluator.contracts import Example
    from evaluator.metric_package import load_metric
    from evaluator.metrics import MetricModelAccess, evaluate_answer
    from evaluator.reliability import empty_usage

    fixture = json.loads((PACKAGE / "judge.json").read_text())
    fixture["example"]["reference"] = json.dumps(
        fixture["example"]["reference"], ensure_ascii=False,
    )
    access = MetricModelAccess(
        model=MODEL, base_url=settings["judge_gateway_url"],
        usage_dir=Path(settings["usage_dir"]),
    )
    usage = empty_usage()
    os.environ["ARENA_SC71_KB_ROOT"] = str(PACKAGE / "assets")
    try:
        result = evaluate_answer(
            load_metric(Path(settings["metric_package"])),
            Example.model_validate(fixture["example"]), fixture["answer"],
            model_access=access, usage=usage,
        )
    except Exception as error:
        vars(error)["usage"] = usage
        if access.run_ids:
            vars(error)["run_id"] = access.run_ids[-1]
        raise
    if len(access.run_ids) != 1:
        raise ValueError("one judge accounting scope is required")
    return {"evaluation": result, "usage": usage, "run_id": access.run_ids[0]}


def verify_result(settings, scenario, result):
    from evaluator import trace_store
    from evaluator.usage import read_usage

    calls = trace_store.read_calls(Path(settings["trace_dir"]), result["run_id"])
    usage = read_usage(Path(settings["usage_dir"]), result["run_id"])
    if usage != result["usage"]:
        raise ValueError("saved usage differs from trusted accounting")
    accounting = verify_usage(usage, calls)
    if scenario == "judge":
        return accounting | verify_judge(result["evaluation"], calls)
    return accounting | verify_solver(scenario, result["answer"], calls)


def main():
    from evaluator.metric_package import load_metric

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    metric_path = Path(settings["metric_package"])
    load_metric(metric_path)
    settings["metric_lock_sha256"] = hashlib.sha256(
        (metric_path / "metric.lock.json").read_bytes(),
    ).hexdigest()
    args.state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.state.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = load_state(args.state, settings, fixture_digest())
        save_json(args.state, state)
        for name in (args.scenario,) if args.scenario else SCENARIOS:
            if args.verify_only:
                record = state["scenarios"][name]
                verify_result(settings, name, record["result"])
            else:
                operation = (lambda: judge(settings)) if name == "judge" else (
                    lambda scenario=name: solve(settings, scenario)
                )
                run_scenario(
                    name, args.state, state, operation,
                    lambda result, scenario=name: verify_result(settings, scenario, result),
                )
        selected = (args.scenario,) if args.scenario else SCENARIOS
        return int(any(state["scenarios"].get(name, {}).get("status") != "accepted"
                       for name in selected))


if __name__ == "__main__":
    raise SystemExit(main())
