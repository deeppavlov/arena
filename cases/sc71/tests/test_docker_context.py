import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HERMES = Path("cases/sc71/harness/hermes")


@pytest.mark.parametrize("hermes", [False, True])
def test_build_context_excludes_other_cases_and_sensitive_files(
    tmp_path: Path, hermes: bool,
) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker BuildKit is required to inspect the build context")
    for arguments in (("buildx", "version"), ("info", "--format", "{{.ServerVersion}}")):
        available = subprocess.run(
            [docker, *arguments], capture_output=True, check=False, timeout=30,
        )
        if available.returncode:
            pytest.skip("Docker BuildKit is required to inspect the build context")

    context = tmp_path / "context"
    context.mkdir()
    common = {"pyproject.toml", "uv.lock", "evaluator/source.py", "runner/main.py"}
    template = {"cases/template/README.md"}
    adapter = {str(HERMES / path) for path in (
        "main.py", "config.yaml", "source-cache/source.tar",
    )}
    excluded = {
        "cases/other/harness/main.py", "cases/sc71/dataset/train.jsonl",
        "cases/sc71/campaigns/run/state.json", "cases/sc71/proposals/method.py",
        "archive/cases/sc71/result.json", "evaluator/.env", "evaluator/access-token.txt",
        *(str(HERMES / path) for path in (
            ".env", "secret.json", "private/data.json", "source-cache/.env",
            "source-cache/operator.env", "source-cache/private/value.txt",
            "source-cache/.git/config", "source-cache/data/value.json",
        )),
    }
    for name in common | template | adapter | excluded:
        path = context / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic")
    shutil.copyfile(ROOT / ".dockerignore", context / ".dockerignore")
    dockerfile = HERMES / "Dockerfile" if hermes else Path("Dockerfile")
    if hermes:
        shutil.copyfile(
            ROOT / HERMES / "Dockerfile.dockerignore",
            context / HERMES / "Dockerfile.dockerignore",
        )
    (context / dockerfile).write_text("FROM scratch\nCOPY . /\n")
    output = tmp_path / "output"
    result = subprocess.run(
        [docker, "buildx", "build", "--progress=quiet", "--file", str(context / dockerfile),
         "--output", f"type=local,dest={output}", str(context)],
        text=True, capture_output=True, check=False, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    allowed = common | (adapter if hermes else template)
    denied = excluded | (template if hermes else adapter)
    for name in allowed:
        assert (output / name).is_file(), name
    for name in denied:
        assert not (output / name).exists(), name
