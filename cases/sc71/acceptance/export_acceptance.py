import argparse
import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SCENARIOS = ("retrieval", "delegation", "judge")


def build_record(state, scenario, model_calls):
    from evaluator.reliability import empty_usage
    from evaluator.tracking import sanitize_record

    if state.get("synthetic_only") is not True or scenario not in SCENARIOS:
        raise ValueError("Only declared synthetic acceptance scenarios can be exported")
    saved = state["scenarios"][scenario]
    if saved["status"] not in {"accepted", "rejected", "failed"}:
        raise ValueError("Acceptance scenario is not terminal")
    result = saved.get("result", {})
    usage = result.get("usage", saved.get("usage"))
    role = "judge" if scenario == "judge" else "solver"
    started = min([int(saved["started_at_unix"] * 1e9)] + [
        call["start_time_ns"] for call in model_calls
    ])
    finished = max([int(saved["finished_at_unix"] * 1e9)] + [
        call["end_time_ns"] for call in model_calls
    ])
    evaluation = result.get("evaluation", {})
    seconds = (evaluation.get("wall_seconds", saved["wall_seconds"]) if role == "judge"
               else result.get("wall_seconds", saved.get("wall_seconds")))
    resources = {name: {"usage": empty_usage(), "wall_seconds": 0} for name in ("solver", "judge")}
    resources[role] = {"usage": usage, "wall_seconds": seconds}
    metrics = {"wall_seconds": saved["wall_seconds"]}
    for name, resource in resources.items():
        counters = resource["usage"]
        metrics[f"{name}_tokens"] = counters["total_tokens"] if counters is not None else None
        metrics[f"{name}_seconds"] = resource["wall_seconds"]
        metrics[f"{name}_unreported_attempts"] = (
            counters["unreported_attempts"] if counters is not None else None
        )
    if "score" in evaluation:
        metrics["score"] = evaluation["score"]
    status = "FINISHED" if saved["status"] == "accepted" else "FAILED"
    calls = [{
        "name": f"acceptance.{scenario}", "span_id": "scenario", "span_type": "CHAIN",
        "inputs": {"scenario": scenario}, "outputs": saved,
        "start_time_ns": started, "end_time_ns": finished,
        "attributes": {"role": role}, "status": "OK" if status == "FINISHED" else "ERROR",
    }]
    for index, call in enumerate(model_calls):
        metadata = call.get("attributes", {}).get("caller_metadata", {}) or {}
        calls.append({
            **call, "span_id": f"model-{index}", "parent_id": "scenario",
            "attributes": {
                **call.get("attributes", {}), "role": role,
                **{key: metadata[key] for key in ("criterion", "vote") if key in metadata},
            },
        })
    configuration = dict(state["settings"])
    configuration.pop("metric_lock_sha256", None)
    return sanitize_record({
        "split": "train", "status": status,
        "params": {
            "case_id": "sc71-preparation-acceptance",
            "campaign_id": state["settings"].get("campaign_id", "sc71-first-20261019"),
            "candidate_id": "sc71-synthetic-acceptance", "scenario": scenario,
            "configuration": configuration,
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731", "harness": "hermes",
            "harness_version": state["settings"]["runner"]["image"],
            "parameters": {"reasoning_effort": "max"}, "dataset_id": "synthetic-acceptance",
        },
        "resources": resources, "metrics": metrics, "calls": calls,
        "start_time_ns": started, "end_time_ns": finished, "synthetic_only": True,
    })


def verify_export(result, record, operator):
    if result["status"] != "exported":
        return result
    from evaluator.tracking import _EXPORT_LOCK, _credentials

    with _EXPORT_LOCK, _credentials(operator):
        mlflow = importlib.import_module("mlflow")
        key = "ARENA_OPERATOR_MLFLOW_TRACKING_URI" if operator else "MLFLOW_TRACKING_URI"
        uri = os.environ[key]
        client = mlflow.MlflowClient(tracking_uri=uri)
        run = client.get_run(result["run_id"])
        traces = client.search_traces(locations=[run.info.experiment_id], run_id=result["run_id"])
        if len(traces) != 1 or len(traces[0].data.spans) != len(record["calls"]) + 1:
            raise ValueError("Acceptance trace spans differ from saved record")
        if run.info.status != record["status"]:
            raise ValueError("Acceptance tracking status differs from saved record")
        exported = Path(client.download_artifacts(result["run_id"], "record.json"))
        if json.loads(exported.read_text()) != record:
            raise ValueError("Acceptance tracking artifact differs from saved record")
        return {**result, "trace_id": traces[0].info.trace_id, "spans": len(traces[0].data.spans)}

def main():
    from evaluator.trace_store import read_calls
    from evaluator.tracking import _environment_secrets, _save, export_record, redact

    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    state["settings"].pop("metric_lock_sha256", None)
    if state.get("synthetic_only") is not True:
        raise ValueError("Only synthetic acceptance records may be exported")
    path = args.output / "records" / f"{args.scenario}.json"
    if path.exists():
        record = json.loads(path.read_text())
        configuration = redact(state["settings"], secrets=_environment_secrets())
        params = record["params"]
        if "configuration" in params:
            if params["configuration"] != configuration:
                raise ValueError("Acceptance export belongs to another fixed configuration")
        elif (params.get("scenario") != args.scenario
              or params.get("harness_version") != state["settings"]["runner"]["image"]):
            raise ValueError("Acceptance export belongs to another fixed configuration")
    else:
        saved = state["scenarios"][args.scenario]
        run_id = saved.get("result", {}).get("run_id", saved.get("run_id"))
        calls = read_calls(Path(state["settings"]["trace_dir"]), run_id) if run_id else []
        record = build_record(state, args.scenario, calls)
        _save(path, record)
    results = {}
    for operator, name in ((False, "shared"), (True, "operator")):
        result = export_record(path, args.output / name, operator=operator)
        results[name] = verify_export(result, record, operator) if args.verify else result
    _save(args.output / f"{args.scenario}-exports.json", results)
    print(json.dumps({"scenario": args.scenario, "exports": results}))


if __name__ == "__main__":
    main()
