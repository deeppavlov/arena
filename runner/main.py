import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MAX_BYTES = 1024 * 1024
MAX_TASK_BYTES = 8 * 1024 * 1024


def read_limited(path: Path) -> str:
    with path.open("rb") as source:
        content = source.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("file too large")
    return content.decode()


def run_program(path: Path, payload: dict, timeout: float | None = None) -> dict:
    with tempfile.TemporaryFile() as output:
        subprocess.run(
            [sys.executable, "-E", "-s", str(path)],
            input=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                errors="backslashreplace",
            ),
            stdout=output,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=True,
            cwd="/tmp",
        )
        output.seek(0)
        raw = output.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("program output too large")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("invalid program output")
    return result


def build(task: dict) -> str:
    uses_model = task.get("uses_model", True)
    payload = {"assets_dir": "/assets", "build_dir": "/output", "uses_model": uses_model}
    if "parameters" in task:
        payload["parameters"] = task["parameters"]
    if uses_model:
        payload.update({key: task[key] for key in (
            "model", "model_base_url", "model_request_timeout"
        )})
    run_program(Path("/candidate/build.py"), payload, task["prepare_timeout"])
    return "build complete"


def solve(task: dict) -> str:
    payload = {"input": task["input"], "assets_dir": "/assets"}
    if "parameters" in task:
        payload["parameters"] = task["parameters"]
    if not task.get("uses_model", True):
        result = run_program(Path("/candidate/solve.py"), payload)
        answer = result.get("answer")
        if not isinstance(answer, str):
            raise ValueError("invalid answer")
        return answer
    prompt = read_limited(Path("/candidate/prompt.md"))
    context = ""
    preparation = Path("/candidate/prepare.py")
    if preparation.is_file():
        payload.update({
            "model": task["model"],
            "model_base_url": task["model_base_url"],
            "model_request_timeout": task["model_request_timeout"],
        })
        result = run_program(preparation, payload, task["prepare_timeout"])
        context = result["context"]
        if not isinstance(context, str):
            raise ValueError("invalid context")
    messages = [{"role": "system", "content": prompt}]
    if context:
        messages.append({"role": "user", "content": "Task context:\n" + context})
    messages.append({"role": "user", "content": task["input"]})
    request = urllib.request.Request(
        task["model_base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(
            {"model": task["model"], "messages": messages},
            ensure_ascii=False, separators=(",", ":"),
        ).encode(errors="backslashreplace"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=task["model_request_timeout"]) as response:
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError("response too large")
    answer = json.loads(body)["choices"][0]["message"]["content"]
    if not isinstance(answer, str):
        raise ValueError("invalid answer")
    return answer


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_TASK_BYTES + 1)
        if len(raw) > MAX_TASK_BYTES:
            raise ValueError("task too large")
        task = json.loads(raw)
        if not isinstance(task, dict) or type(task.get("uses_model", True)) is not bool:
            raise ValueError("invalid task")
        uses_model = task.get("uses_model", True)
        operation = task.get("operation", "solve")
        if operation not in ("solve", "build"):
            raise ValueError("invalid operation")
        if (
            operation == "solve" and not isinstance(task.get("input"), str)
            or operation == "build" and "input" in task
            or not isinstance(task.get("parameters", {}), dict)
        ):
            raise ValueError("invalid task")
        if uses_model and any(
            not isinstance(task.get(key), str) or not task[key].strip()
            for key in ("model", "model_base_url")
        ):
            raise ValueError("invalid task")
        answer = build(task) if operation == "build" else solve(task)
        output = json.dumps({"answer": answer})
        if len(output.encode()) + 1 > MAX_BYTES:
            raise ValueError("answer too large")
        print(output)
        return 0
    except Exception:
        print("runner failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
