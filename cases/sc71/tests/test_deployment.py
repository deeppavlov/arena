import os
import subprocess
from pathlib import Path

LAUNCHER = Path(__file__).parents[1] / "deploy/gpu6/launch.sh"


def test_historical_ready_cannot_launch_refactored_layout(tmp_path):
    operator = tmp_path / "operator"
    operator.mkdir()
    (operator / "READY").write_text("historical preparation passed")
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=dict(os.environ, ARENA_ROOT=str(tmp_path)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "local refactor not validated/deployed" in result.stderr
    assert not (operator / "launch").exists()


def test_initialized_campaign_is_rejected_before_docker(tmp_path):
    operator = tmp_path / "operator"
    (operator / "launch").mkdir(parents=True)
    (operator / "layout-version").write_text("sc71-case-layout-v2")
    (operator / "READY").write_text("prepared")
    (operator / "launch/active.json").write_text('{"run_id":"existing"}')
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=dict(os.environ, ARENA_ROOT=str(tmp_path)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "A researcher run was already initialized" in result.stderr
