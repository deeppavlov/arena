import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILES = (
    "cases/sc71/deploy/gpu6/mlflow/compose.yaml",
    "cases/sc71/campaigns/sc71-second-20261024/deploy/mlflow-compose.yaml",
    "cases/sc35/campaigns/sc35-first-20261028/deploy/mlflow-compose.yaml",
)


def test_operator_proxy_deployments_are_covered():
    discovered = set()
    for directory in (ROOT / "cases", ROOT / "deploy"):
        for pattern in ("*.yaml", "*.yml"):
            for path in directory.rglob(pattern):
                content = path.read_text()
                if "operator-tracking:" in content and "image: nginx:" in content:
                    discovered.add(path.relative_to(ROOT).as_posix())
    assert discovered == set(COMPOSE_FILES)


@pytest.mark.parametrize("relative", COMPOSE_FILES)
def test_operator_proxy_mounts_common_config(relative):
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is required")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("ARENA_", "SC35_", "SC71_", "COMPOSE_"))
    }
    environment.update({
        "ARENA_ROOT": str(ROOT.parent),
        "SC35_MLFLOW_IMAGE": "mlflow:synthetic",
        "SC71_MLFLOW_IMAGE": "mlflow:synthetic",
    })
    result = subprocess.run(
        [docker, "compose", "-f", str(ROOT / relative), "config",
         "--no-env-resolution", "--format", "json"],
        capture_output=True, text=True, check=True, env=environment,
    )
    service = json.loads(result.stdout)["services"]["operator-tracking"]
    mounts = [item for item in service["volumes"] if item["target"] == "/etc/nginx/nginx.conf"]
    assert len(mounts) == 1
    assert mounts[0]["read_only"] is True
    source = Path(mounts[0]["source"]).resolve()
    assert source == ROOT / "deploy/mlflow/operator-nginx.conf"
    assert source.is_file()
