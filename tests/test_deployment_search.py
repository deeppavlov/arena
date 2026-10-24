import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def render_compose(tmp_path: Path, values: dict[str, str]) -> dict[str, Any]:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is required to render deployment configuration")
    environment = {key: value for key, value in os.environ.items() if not key.startswith(
        ("ARENA_", "COMPOSE_"),
    )}
    version = subprocess.run(
        [docker, "compose", "version"], capture_output=True, env=environment, check=False,
    )
    if version.returncode:
        pytest.skip("Docker Compose is required to render deployment configuration")
    env_file = tmp_path / "operator.env"
    env_file.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")
    compose = ROOT / "compose.yaml"
    result = subprocess.run(
        [docker, "compose", "--file", str(compose), "--env-file", str(env_file),
         "--profile", "*", "config", "--no-env-resolution", "--format", "json"],
        capture_output=True, text=True, env=environment, check=True,
    )
    return json.loads(result.stdout)


def example_environment() -> dict[str, str]:
    return dict(line.split("=", 1) for line in (ROOT / "deploy/operator.env.example").read_text(
    ).splitlines() if line and not line.startswith("#"))


def test_standard_compose_passes_campaign_guard_and_shared_mounts(tmp_path: Path) -> None:
    values = example_environment() | {
        "ARENA_SEARCH_DIR": str(tmp_path / "journal"),
        "ARENA_SEARCH_CONTROL_DIR": str(tmp_path / "operator-control"),
        "ARENA_SEARCH_MAX_SOLUTIONS": "7",
    }
    services = render_compose(tmp_path, values)["services"]
    evaluator = services["evaluator"]
    environment = evaluator["environment"]
    assert environment["ARENA_SEARCH_MODE"] == "campaign"
    assert environment["ARENA_SEARCH_JOURNAL"] == "/search/search.sqlite3"
    assert environment["ARENA_SEARCH_MAX_SOLUTIONS"] == "7"
    assert environment["ARENA_SEARCH_STOP_PATH"] == "/search-control/stop.json"
    volumes = {volume["target"]: volume for volume in evaluator["volumes"]}
    for path, source in (("/search", "ARENA_SEARCH_DIR"),
                         ("/search-control", "ARENA_SEARCH_CONTROL_DIR")):
        assert volumes[path]["source"] == values[source]
        assert volumes[path]["read_only"] is True
        assert volumes[path]["bind"]["create_host_path"] is False
    researcher = services["researcher"]
    research_volumes = {volume["target"]: volume for volume in researcher["volumes"]}
    assert research_volumes["/search"]["source"] == values["ARENA_SEARCH_DIR"]
    assert not research_volumes["/search"].get("read_only", False)
    assert "/search-control" not in research_volumes
    assert researcher["environment"]["ARENA_SEARCH_JOURNAL"] == "/search/search.sqlite3"


def test_standard_compose_cannot_silently_default_to_standalone(tmp_path: Path) -> None:
    values = example_environment()
    values.pop("ARENA_SEARCH_MODE", None)
    environment = render_compose(tmp_path, values)["services"]["evaluator"]["environment"]
    assert environment["ARENA_SEARCH_MODE"] == "campaign"


def test_standard_compose_allows_explicit_standalone(tmp_path: Path) -> None:
    values = example_environment() | {
        "ARENA_SEARCH_MODE": "standalone", "ARENA_SEARCH_JOURNAL": "",
        "ARENA_SEARCH_MAX_SOLUTIONS": "", "ARENA_SEARCH_STOP_PATH": "",
    }
    environment = render_compose(tmp_path, values)["services"]["evaluator"]["environment"]
    assert environment["ARENA_SEARCH_MODE"] == "standalone"
    assert environment["ARENA_SEARCH_JOURNAL"] == ""
    assert environment["ARENA_SEARCH_MAX_SOLUTIONS"] == ""
    assert environment["ARENA_SEARCH_STOP_PATH"] == ""
