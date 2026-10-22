"""Reproduce scoped and repository checks; all new outputs stay in this package."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
WORK = PACKAGE.parent.parent
REPO = WORK.parents[2]
PLANNED = PACKAGE.parent / "planned_bm25"


def main():
    output = PACKAGE / "checks" / ("qa-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / "temp"
    temporary.mkdir()
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
        TEMP=str(temporary),
        TMP=str(temporary),
        TIKTOKEN_CACHE_DIR=str(temporary / "tiktoken"),
    )
    checks = {
        "candidate_lint": ["ruff", "check", str(PACKAGE), str(PLANNED), "--no-cache"],
        "structure_types": ["pyright", "--project", str(PACKAGE / "pyrightconfig.json")],
        "planned_types": ["pyright", "--project", str(PLANNED / "pyrightconfig.json")],
        "candidate_tests": [
            "pytest",
            str(PACKAGE / "tests"),
            str(PLANNED / "tests"),
            str(WORK / "tests"),
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(output / "pytest"),
        ],
        "project_lint": ["ruff", "check", ".", "--no-cache"],
        "project_types": ["pyright"],
        "project_tests": [
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--maxfail=1",
            "--basetemp",
            str(output / "project-pytest"),
        ],
        "project_smoke": ["app.spikes.walking_skeleton"],
    }
    results = {}
    for name, arguments in checks.items():
        command = [sys.executable, "-B", "-m", *arguments]
        started = time.monotonic()
        print(f"Checking {name}", flush=True)
        try:
            run = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
            log, code = run.stdout + "\n" + run.stderr, run.returncode
        except subprocess.TimeoutExpired:
            log, code = "Check exceeded 300 seconds; not reported as passed.", None
        (output / f"{name}.txt").write_text(log, encoding="utf-8")
        results[name] = {
            "exit_code": code,
            "seconds": time.monotonic() - started,
            "command": command,
        }
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {code}", flush=True)
    protected = json.loads((PACKAGE / "checks/protected-before.json").read_text(encoding="utf-8"))
    changed = [
        name
        for name, expected in protected.items()
        if not (WORK / name).is_file()
        or hashlib.sha256((WORK / name).read_bytes()).hexdigest() != expected
    ]
    (output / "protected-after.json").write_text(
        json.dumps({"checked": len(protected), "changed": changed}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Protected existing files changed: {len(changed)}", flush=True)
    print(output, flush=True)
    if changed or any(result["exit_code"] != 0 for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
