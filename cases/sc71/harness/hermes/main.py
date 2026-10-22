import contextlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from cases.sc71.harness.hermes import runtime

MAX_BYTES = 1024 * 1024
MAX_TASK_BYTES = 8 * MAX_BYTES
TOOLSETS = {"terminal", "file", "delegation", "todo"}


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def read_text(path: Path) -> str:
    with path.open("rb") as source:
        raw = source.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("file too large")
    return raw.decode("utf-8")


def candidate_file(candidate: Path, name: str) -> Path:
    if not isinstance(name, str):
        raise ValueError("invalid candidate path")
    path = candidate / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(candidate.resolve()):
        raise ValueError("invalid candidate path")
    if not path.is_file():
        raise ValueError("candidate file missing")
    return path


def run_script(path: Path, payload: dict[str, Any], workspace: Path, timeout: float) -> dict:
    with tempfile.TemporaryFile() as output:
        subprocess.run(
            [sys.executable, "-E", "-s", str(path)],
            input=json.dumps(payload, ensure_ascii=False).encode(),
            cwd=workspace,
            stdout=output,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=True,
        )
        output.seek(0)
        raw = output.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("candidate output too large")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("candidate output must be an object")
    return result


def validate_task(task: dict[str, Any]) -> dict[str, Any]:
    if task.get("model") != runtime.MODEL or task.get("uses_model", True) is not True:
        raise ValueError("fixed model required")
    operation = task.get("operation", "solve")
    if operation not in ("solve", "build"):
        raise ValueError("invalid operation")
    if operation == "build":
        if "input" in task:
            raise ValueError("build cannot receive input")
    elif not isinstance(task.get("input"), str):
        raise ValueError("invalid input")
    if not isinstance(task.get("model_base_url"), str) or not task["model_base_url"]:
        raise ValueError("invalid gateway")
    for name in ("model_request_timeout", "prepare_timeout"):
        value = task.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("invalid timeout")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid timeout")
    parameters = task.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("invalid parameters")
    if parameters.get("mode", "files") not in ("files", "retrieval"):
        raise ValueError("invalid mode")
    toolsets = parameters.get("toolsets", ["terminal", "file", "delegation"])
    if not isinstance(toolsets, list) or any(
        not isinstance(value, str) or value not in TOOLSETS for value in toolsets
    ):
        raise ValueError("invalid toolsets")
    iterations = parameters.get("max_iterations", 10000)
    if type(iterations) is not int or iterations < 1 or iterations > 10000:
        raise ValueError("invalid max_iterations")
    return parameters


def register_tools(candidate, payload, workspace, timeout, registry, required):
    definitions = []
    if (candidate / "tools.json").is_file():
        definitions = read_json(candidate / "tools.json")
        if not isinstance(definitions, list):
            raise ValueError("invalid tools manifest")
    if required:
        definitions.append({
            "name": "arena_search",
            "description": "Search the case knowledge base. MUST use before answering.",
            "parameters": {
                "type": "object", "properties": {"query": {"type": "string"}},
                "required": ["query"], "additionalProperties": False,
            },
            "script": "retrieve.py",
        })
    calls = []
    names = set()
    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError("invalid tool definition")
        name = definition.get("name")
        if (
            not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)
            or name in names or name in {"terminal", "read_file", "write_file", "delegate_task"}
            or not isinstance(definition.get("description"), str)
            or not isinstance(definition.get("parameters"), dict)
        ):
            raise ValueError("invalid tool definition")
        names.add(name)
        script = candidate_file(candidate, definition["script"])

        def handler(arguments, tool_name=name, tool_script=script, **kwargs):
            result = run_script(tool_script, payload | {"arguments": arguments}, workspace, timeout)
            calls.append(tool_name)
            return json.dumps(result, ensure_ascii=False)

        registry.register(
            name=name,
            toolset="arena_candidate",
            schema={key: definition[key] for key in ("name", "description", "parameters")},
            handler=handler,
            execution_timeout=timeout,
        )
    return names, calls


def solve(
    task, candidate, assets, workspace, agent_factory, registry, build_output=Path("/output")
) -> str:
    parameters = validate_task(task)
    workspace.mkdir(parents=True, exist_ok=True)
    payload = {key: task[key] for key in (
        "input", "model", "model_base_url", "model_request_timeout"
    ) if key in task}
    payload.update(
        parameters=parameters, candidate_dir=str(candidate), assets_dir=str(assets),
        workspace_dir=str(workspace),
    )
    if task.get("operation", "solve") == "build":
        payload["build_dir"] = str(build_output)
        run_script(
            candidate_file(candidate, "build.py"), payload, workspace, task["prepare_timeout"]
        )
        return "build complete"
    prompt = read_text(candidate_file(candidate, parameters.get("prompt_file", "prompt.md")))
    if (candidate / "prepare.py").is_file():
        prepared = run_script(candidate / "prepare.py", payload, workspace, task["prepare_timeout"])
        context = prepared.get("context", "")
        if not isinstance(context, str):
            raise ValueError("invalid preparation context")
        prompt += "\nPrepared context:\n" + context
    required = parameters.get("mode", "files") == "retrieval"
    names, calls = register_tools(
        candidate, payload, workspace, task["prepare_timeout"], registry, required
    )
    prompt += (
        f"\nKnowledge base files: {assets}. Candidate files: {candidate}. "
        f"Method workspace: {workspace}. Use only this case knowledge base as answer evidence."
    )
    if required:
        prompt += "\nYou MUST call arena_search before producing the final answer."
    toolsets = list(parameters.get("toolsets", ["terminal", "file", "delegation"]))
    if names:
        toolsets.append("arena_candidate")
    agent = agent_factory(
        model=runtime.MODEL,
        base_url=runtime.CONFIG_BASE_URL,
        api_key="arena",
        provider="custom",
        api_mode="chat_completions",
        reasoning_config={"enabled": True, "effort": "max"},
        enabled_toolsets=toolsets,
        max_iterations=parameters.get("max_iterations", 10000),
        cwd=str(workspace),
        skip_memory=True,
        skip_background_review=True,
        skip_context_files=True,
        load_soul_identity=False,
        save_trajectories=False,
        quiet_mode=True,
        verbose_logging=False,
    )
    try:
        result = agent.run_conversation(task["input"], system_message=prompt)
    finally:
        agent.close()
    if result.get("failed") or not result.get("completed"):
        raise ValueError("Hermes failed")
    if required and "arena_search" not in calls:
        raise ValueError("required retrieval tool was not called")
    answer = result.get("final_response")
    if not isinstance(answer, str):
        raise ValueError("invalid Hermes answer")
    return answer


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_TASK_BYTES + 1)
        if len(raw) > MAX_TASK_BYTES:
            raise ValueError("task too large")
        task = json.loads(raw)
        if not isinstance(task, dict):
            raise ValueError("invalid task")
        validate_task(task)
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                runtime.prepare_home(Path("/tmp/arena-hermes-home"))
                state = runtime.install(task["model_base_url"], task["model_request_timeout"])
                from run_agent import AIAgent
                from tools.registry import registry

                answer = solve(
                    task, Path("/candidate"), Path("/assets"), Path("/tmp/arena-workspace"),
                    AIAgent, registry,
                )
                state.check()
        output = json.dumps({"answer": answer}, ensure_ascii=False)
        if len(output.encode()) + 1 > MAX_BYTES:
            raise ValueError("answer too large")
        print(output)
        return 0
    except Exception:
        print("Hermes runner failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
