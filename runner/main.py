import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MAX_BYTES = 1024 * 1024


def read_limited(path: Path) -> str:
    with path.open("rb") as source:
        content = source.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("file too large")
    return content.decode()


def solve(task: dict) -> str:
    prompt = read_limited(Path("/candidate/prompt.md"))
    context = ""
    preparation = Path("/candidate/prepare.py")
    if preparation.is_file():
        with tempfile.TemporaryFile() as output:
            subprocess.run(
                [sys.executable, "-I", str(preparation)],
                input=json.dumps({"input": task["input"], "assets_dir": "/assets"}).encode(),
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=True,
                cwd="/tmp",
            )
            output.seek(0)
            raw = output.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("context too large")
        result = json.loads(raw)
        context = result["context"]
        if not isinstance(context, str):
            raise ValueError("invalid context")
    messages = [{"role": "system", "content": prompt}]
    if context:
        messages.append({"role": "user", "content": "Task context:\n" + context})
    messages.append({"role": "user", "content": task["input"]})
    request = urllib.request.Request(
        task["model_base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps({"model": task["model"], "messages": messages}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError("response too large")
    answer = json.loads(body)["choices"][0]["message"]["content"]
    if not isinstance(answer, str):
        raise ValueError("invalid answer")
    return answer


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("task too large")
        task = json.loads(raw)
        if not isinstance(task, dict) or any(
            not isinstance(task.get(key), str) or not task[key].strip()
            for key in ("input", "model", "model_base_url")
        ):
            raise ValueError("invalid task")
        output = json.dumps({"answer": solve(task)})
        if len(output.encode()) + 1 > MAX_BYTES:
            raise ValueError("answer too large")
        print(output)
        return 0
    except Exception:
        print("runner failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
