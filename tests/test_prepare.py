import json
import subprocess
import sys
from pathlib import Path


def test_cli_freezes_and_verifies_package_without_execution(tmp_path: Path) -> None:
    source = tmp_path / "metric"
    source.mkdir()
    (source / "metric.json").write_text(json.dumps({
        "version": "synthetic-v1", "kind": "command",
        "command": {"argv": ["python", "score.py"]},
    }))
    (source / "score.py").write_text("raise RuntimeError('not a fixture runner')\n")
    frozen = tmp_path / "frozen"
    for arguments in (
        ["freeze", "--metric", str(source), "--output", str(frozen)],
        ["verify", "--metric", str(frozen)],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "evaluator.prepare", *arguments],
            text=True, capture_output=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert len(report["sha256"]) == 64
        assert report["semantic_tests_run"] is False
    (frozen / "score.py").write_text("print('changed')\n")
    failure = subprocess.run(
        [sys.executable, "-m", "evaluator.prepare", "verify", "--metric", str(frozen)],
        text=True, capture_output=True, check=False,
    )
    assert failure.returncode != 0
    assert "lock" in failure.stderr
