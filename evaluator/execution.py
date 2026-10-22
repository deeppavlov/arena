from pathlib import Path
from typing import TYPE_CHECKING, Any

from evaluator import trace_store
from evaluator.runner import DockerRunner, HarnessRunner

if TYPE_CHECKING:
    from evaluator.service import Settings


def make_runner(settings: "Settings", usage_dir: Path) -> HarnessRunner:
    def create(config: dict[str, Any], *, default: bool = False) -> DockerRunner:
        return DockerRunner(
            image=config.get("image", settings.runner_image),
            network=config.get("network", settings.runner_network),
            uses_model=config.get("uses_model", settings.uses_model),
            supports_build=config.get(
                "supports_build", settings.runner_supports_build if default else False,
            ),
            supports_asset_selection=config.get(
                "supports_asset_selection",
                settings.runner_supports_asset_selection if default else False,
            ),
            model=settings.model, model_base_url=settings.runner_model_base_url,
            assets_dir=settings.assets_dir, timeout=settings.runner_timeout,
            model_timeout=settings.model_timeout, prepare_timeout=settings.prepare_timeout,
            retry_delay=settings.retry_delay, usage_dir=usage_dir,
            memory=settings.runner_memory, cpus=settings.runner_cpus,
            pids_limit=settings.runner_pids_limit, tmpfs_size=settings.runner_tmpfs_size,
            tmpfs_exec=settings.runner_tmpfs_exec,
            published_assets_dir=settings.published_assets_dir, build_dir=settings.build_dir,
        )

    return HarnessRunner(create({}, default=True), {
        name: create(config) for name, config in settings.harnesses.items()
    })


def collect_model_calls(
    trace_dir: Path | None, run_ids: list[str], *,
    attributes: dict[str, Any] | None = None, parent_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    calls: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if trace_dir is None:
        return calls, errors
    for run_id in run_ids:
        try:
            model_calls = trace_store.read_calls(trace_dir, run_id)
        except (OSError, ValueError) as error:
            errors.append({"run_id": run_id, "error_type": type(error).__name__})
            continue
        for call in model_calls:
            calls.append({
                **call,
                **({"parent_id": call.get("parent_id", parent_id)} if parent_id else {}),
                "attributes": {**call.get("attributes", {}), **(attributes or {}),
                               "run_id": run_id},
            })
    return calls, errors
