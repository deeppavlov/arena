"""Reproduce local checks without installs; all outputs stay in this new package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-checks", action="store_true")
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    repo = package.parents[4]
    packages = package.parent
    names = ("file_agent", "amendment_bm25")
    output = package / "checks" / datetime.now(UTC).strftime("qa-%Y%m%d-%H%M%S")
    output.mkdir()
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
        UV_CACHE_DIR=str(output / "uv-cache"),
        UV_OFFLINE="true",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
    )
    python = [sys.executable, "-X", "utf8", "-B"]
    jobs = [
        (
            "new_lint",
            python
            + ["-m", "ruff", "check", "--no-cache"]
            + [str(packages / name) for name in names],
        ),
        *[
            (
                name + "_types",
                python
                + ["-m", "pyright", "--project", str(packages / name / "pyrightconfig.json")],
            )
            for name in names
        ],
        (
            "new_tests",
            python
            + ["-m", "pytest"]
            + [str(packages / name / "tests") for name in names]
            + ["-q", "--tb=short", "-p", "no:cacheprovider", "--basetemp", str(output / "pytest")],
        ),
        (
            "prior_new_tests",
            python
            + ["-m", "pytest"]
            + [str(packages / name / "tests") for name in ("structure_bm25", "planned_bm25")]
            + [
                "-q",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(output / "prior-pytest"),
            ],
        ),
    ]
    if args.project_checks:
        uv = shutil.which("uv")
        if not uv:
            local_uv = Path.home() / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv")
            if local_uv.is_file():
                uv = str(local_uv)
        if not uv:
            raise SystemExit("uv is unavailable; no installation attempted")
        prefix = [uv, "run", "--offline", "--no-sync"]
        jobs.extend(
            [
                (
                    "legacy_candidate_tests",
                    python
                    + [
                        "-m",
                        "pytest",
                        str(packages.parent / "tests"),
                        "-q",
                        "--tb=short",
                        "-p",
                        "no:cacheprovider",
                        "--basetemp",
                        str(output / "legacy-pytest"),
                    ],
                ),
                ("project_lint", prefix + ["ruff", "check", "--no-cache", "."]),
                ("project_types", prefix + ["pyright"]),
                (
                    "project_tests",
                    prefix
                    + [
                        "python",
                        "-B",
                        "-m",
                        "pytest",
                        "-q",
                        "--tb=short",
                        "-p",
                        "no:cacheprovider",
                        "--basetemp",
                        str(output / "project-pytest"),
                    ],
                ),
                ("project_smoke", prefix + ["python", "-B", "-m", "app.spikes.walking_skeleton"]),
            ]
        )
    results = []
    for name, command in jobs:
        started = time.monotonic()
        run = subprocess.run(
            command,
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        (output / (name + ".txt")).write_text(run.stdout + run.stderr, encoding="utf-8")
        results.append(
            {
                "name": name,
                "command": command,
                "exit_code": run.returncode,
                "seconds": time.monotonic() - started,
            }
        )
        print(name, run.returncode, flush=True)
    (output / "results.json").write_text(
        json.dumps(
            {"python": sys.version, "jobs": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Output:", output)
    raise SystemExit(int(any(x["exit_code"] for x in results)))


if __name__ == "__main__":
    main()
