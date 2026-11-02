from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from typing import Any

EXPECTED_DELAYS = [30, 90, 240, 540, 900]
EXPECTED_TIMEOUT = 131460
SERVICES = (
    "model-gateway", "researcher-gateway", "train-evaluator", "public-evaluator",
    "private-evaluator",
)
PROBE = '''
import json
import os
import re
import sys
from pathlib import Path
from evaluator.reliability import MODEL_TIMEOUT, RETRY_DELAY, request_timeout, retry_delays

mode = sys.argv[1]

def value(key, default):
    file = os.environ.get(key + "_FILE") if mode == "evaluator" else None
    return Path(file).read_text().strip() if file else os.environ.get(key, default)

timeout, delay = MODEL_TIMEOUT, RETRY_DELAY
if mode in {"gateway", "evaluator"}:
    prefix = "MODEL_" if mode == "gateway" else "ARENA_"
    timeout_key = "MODEL_TIMEOUT" if mode == "gateway" else "ARENA_MODEL_TIMEOUT"
    timeout = float(value(timeout_key, MODEL_TIMEOUT))
    delay = float(value(prefix + "RETRY_DELAY", RETRY_DELAY))
result = {"retry_delays": retry_delays(retry_delay=delay),
          "request_timeout": request_timeout(timeout, retry_delay=delay)}
if mode == "researcher-access":
    path = os.environ.get("ARENA_MODEL_ACCESS_FILE", "/run/arena/model-access.json")
    access = json.loads(Path(path).read_text())
    result["request_timeout"] = access.get("model_request_timeout", request_timeout(
        access.get("model_timeout", MODEL_TIMEOUT)))
if mode == "evaluator":
    result["runner_image"] = value("ARENA_RUNNER_IMAGE", "")
    registry = json.loads(value("ARENA_HARNESSES", "{}"))
    result["harnesses"] = {name: item.get("image", result["runner_image"])
                           for name, item in registry.items()}
if mode == "hermes":
    from cases.sc71.harness.hermes import runtime
    result["hermes_timeout"] = runtime.REQUEST_TIMEOUT
    config = Path(runtime.__file__).with_name("config.yaml").read_text()
    values = re.findall(r"^    timeout: ([0-9]+)\\s*$", config, re.MULTILINE)
    if len(values) != 1:
        raise ValueError("Hermes config timeout is unavailable")
    result["config_timeout"] = int(values[0])
print(json.dumps(result, allow_nan=False))
'''


class ProbeError(ValueError):
    pass


def docker(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["docker", *arguments], capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ProbeError("Docker probe could not complete") from None
    if result.returncode:
        raise ProbeError("Docker probe failed")
    return result.stdout


def document(arguments: list[str]) -> Any:
    try:
        return json.loads(docker(arguments))
    except json.JSONDecodeError:
        raise ProbeError("Docker probe returned invalid JSON") from None


def image_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", value):
        raise ProbeError("Image ID is unavailable")
    return value


def timings(value: Any, hermes: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProbeError("Runtime timings are unavailable")
    keys = ["request_timeout"]
    if hermes:
        keys += ["hermes_timeout", "config_timeout"]
    delays = value.get("retry_delays")
    if not isinstance(delays, list) or len(delays) != 5:
        raise ProbeError("Runtime retry intervals are invalid")
    numbers = delays + [value.get(key) for key in keys]
    if any(type(item) not in {int, float} or not math.isfinite(item) or item < 0
           for item in numbers):
        raise ProbeError("Runtime timings are invalid")
    return {"retry_delays": delays, **{key: value[key] for key in keys}}


def audit(project_name: str, researcher_image: str | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    cached_images: dict[str, dict[str, Any]] = {}

    def fail(role: str, reason: str) -> None:
        failures.append({"role": role, "reason": reason})

    def record(role: str, identifier: str, data: Any, hermes: bool = False) -> None:
        values = timings(data, hermes)
        checks.append({"role": role, "image_id": identifier, **values})
        if values["retry_delays"] != EXPECTED_DELAYS:
            fail(role, "Retry intervals differ from the required 30-minute schedule")
        if any(value != EXPECTED_TIMEOUT for key, value in values.items() if key != "retry_delays"):
            fail(role, "Client timeout differs from the required 131460 seconds")

    def inspect_hermes(role: str, image: Any) -> str:
        if not isinstance(image, str) or not image:
            raise ProbeError("Configured runner image is unavailable")
        identifier = image_id(document(["image", "inspect", image])[0]["Id"])
        if identifier not in cached_images:
            cached_images[identifier] = document([
                "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true", "--user", "10001:10001",
                "--entrypoint", "python", "-w", "/opt/arena", identifier, "-c", PROBE, "hermes",
            ])
        record(role, identifier, cached_images[identifier], hermes=True)
        return identifier

    containers = {}
    try:
        identifiers = docker([
            "ps", "-aq", "--filter", "label=com.docker.compose.project=" + project_name,
        ]).split()
        if identifiers:
            for item in document(["inspect", *identifiers]):
                labels = item["Config"]["Labels"]
                if str(labels.get("com.docker.compose.oneoff", "false")).lower() == "true":
                    continue
                role = labels.get("com.docker.compose.service")
                if role in (*SERVICES, "researcher"):
                    if role in containers:
                        raise ProbeError("Multiple containers exist for one service")
                    containers[role] = item
    except (ProbeError, KeyError, TypeError):
        fail("deployment", "Container inventory could not be verified")

    for role in SERVICES:
        try:
            container = containers.get(role)
            if container is None or container["State"]["Status"] != "running":
                raise ProbeError("Required service is not running")
            identifier = image_id(container["Image"])
            mode = "gateway" if role.endswith("gateway") else "evaluator"
            data = document([
                "exec", "-w", "/opt/arena", container["Id"], "python", "-c", PROBE, mode,
            ])
            record(role, identifier, data)
            if mode == "evaluator":
                inspect_hermes(role + "/runner", data.get("runner_image"))
                registry = data.get("harnesses")
                if not isinstance(registry, dict):
                    raise ProbeError("Harness registry is unavailable")
                for name, image in registry.items():
                    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                        raise ProbeError("Harness name is invalid")
                    inspect_hermes(role + "/harness/" + name, image)
        except ProbeError as error:
            fail(role, str(error))
        except (KeyError, TypeError, IndexError):
            fail(role, "Runtime probe returned incomplete data")

    try:
        researcher = containers.get("researcher")
        image = researcher["Image"] if researcher is not None else researcher_image
        if image is None:
            raise ProbeError("Researcher is absent; provide --researcher-image for initial launch")
        identifier = inspect_hermes("researcher", image)
        if researcher is not None:
            access_path = "/run/arena/model-access.json"
            for item in researcher["Config"].get("Env", []):
                if item.startswith("ARENA_MODEL_ACCESS_FILE="):
                    access_path = item.split("=", 1)[1]
            data = document([
                "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true", "--user", "10001:10001",
                "--volumes-from", researcher["Id"] + ":ro",
                "-e", "ARENA_MODEL_ACCESS_FILE=" + access_path,
                "--entrypoint", "python", "-w", "/opt/arena", identifier,
                "-c", PROBE, "researcher-access",
            ])
            record("researcher/access", identifier, data)
    except ProbeError as error:
        fail("researcher", str(error))
    except (KeyError, TypeError, IndexError):
        fail("researcher", "Runtime probe returned incomplete data")
    return {"status": "failed" if failures else "ok", "checks": checks, "failures": failures}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", default="arena-sc71-second")
    parser.add_argument("--researcher-image")
    args = parser.parse_args(arguments)
    result = audit(args.project_name, args.researcher_image)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return int(result["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
