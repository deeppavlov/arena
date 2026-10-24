import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

COMPOSE = Path(__file__).resolve().parents[1] / "deploy/gpu6/compose.yaml"


def render_compose(tmp_path: Path, values: dict[str, str]) -> dict[str, Any]:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is required to render deployment configuration")
    environment = {key: value for key, value in os.environ.items() if not key.startswith(
        ("ARENA_", "SC71_", "COMPOSE_"),
    )}
    version = subprocess.run(
        [docker, "compose", "version"], capture_output=True, env=environment, check=False,
    )
    if version.returncode:
        pytest.skip("Docker Compose is required to render deployment configuration")
    env_file = tmp_path / "operator.env"
    env_file.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")
    result = subprocess.run(
        [docker, "compose", "--file", str(COMPOSE), "--env-file", str(env_file),
         "--profile", "*", "config", "--no-env-resolution", "--format", "json"],
        capture_output=True, text=True, env=environment, check=True,
    )
    return json.loads(result.stdout)


@pytest.fixture
def sc71_environment(tmp_path: Path) -> dict[str, str]:
    return {
        "ARENA_ROOT": str(tmp_path), "ARENA_DOCKER_GID": "1000",
        "ARENA_EVALUATOR_IMAGE": "sha256:" + "1" * 64,
        "ARENA_HERMES_IMAGE": "sha256:" + "2" * 64,
        "ARENA_INFRA_IMAGE": "sha256:" + "3" * 64,
        "SC71_INPUT_ASSETS_DIGEST": "4" * 64,
        "SC71_FINALISTS": "10", "SC71_OUTPUT_COUNT": "4",
        "SC71_PRIVATE_RELEASE_ID": "synthetic-private-v1",
    }


def test_sc71_compose_declares_private_campaign_guard(
    tmp_path: Path, sc71_environment: dict[str, str],
) -> None:
    services = render_compose(tmp_path, sc71_environment)["services"]
    service = services["private-evaluator"]
    assert service["environment"]["ARENA_SEARCH_MODE"] == "campaign"
    assert service["environment"]["ARENA_SEARCH_JOURNAL"] == "/search/search.sqlite3"
    assert service["environment"]["ARENA_SEARCH_MAX_SOLUTIONS"] == "100"
    volumes = {volume["target"]: volume for volume in service["volumes"]}
    assert volumes["/search"]["read_only"] is True
    assert volumes["/operator-search"]["read_only"] is True
    command = services["researcher"]["command"]
    for argument, variable, expected in (
        ("--private-finalists", "ARENA_FINALISTS", "10"),
        ("--private-output-count", "ARENA_OUTPUT_COUNT", "4"),
        ("--private-release-id", "ARENA_RELEASE_ID", "synthetic-private-v1"),
    ):
        assert command[command.index(argument) + 1] == expected
        assert service["environment"][variable] == expected


@pytest.mark.parametrize("missing", [
    "SC71_FINALISTS", "SC71_OUTPUT_COUNT", "SC71_PRIVATE_RELEASE_ID",
])
def test_sc71_compose_rejects_missing_private_conditions(
    tmp_path: Path, sc71_environment: dict[str, str], missing: str,
) -> None:
    sc71_environment.pop(missing)
    with pytest.raises(subprocess.CalledProcessError) as error:
        render_compose(tmp_path, sc71_environment)
    assert missing in error.value.stderr
