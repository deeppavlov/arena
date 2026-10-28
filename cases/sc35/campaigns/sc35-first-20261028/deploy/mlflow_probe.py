import argparse
import importlib
import json
import os
import time
from pathlib import Path

CAMPAIGN = "sc35-first-20261028"
MARKER = "SC35_FIRST_SYNTHETIC_TRACE_20261028"


def verify_payload(client, receipt):
    run_id = receipt["run_id"]
    run = client.get_run(run_id)
    if (run.info.status != "FINISHED" or run.data.tags.get("split") != "train"
            or run.data.tags.get("campaign_id") != receipt["campaign_id"]):
        raise ValueError("Synthetic run metadata differs from receipt")
    traces = client.search_traces(
        locations=[run.info.experiment_id], run_id=run_id, include_spans=True,
    )
    spans = [span for trace in traces for span in trace.data.spans]
    expected = {"message": receipt["marker"]}
    if (len(spans) != 1 or spans[0].name != "synthetic.echo"
            or spans[0].inputs != expected or spans[0].outputs != expected):
        raise ValueError("Synthetic trace payload is missing or changed")
    record = json.loads(Path(client.download_artifacts(run_id, "record.json")).read_text())
    artifacts = record.get("call_artifacts", [])
    if (record.get("synthetic_only") is not True
            or record["params"]["campaign_id"] != receipt["campaign_id"]
            or len(artifacts) != 1 or Path(artifacts[0]).parent != Path("calls")):
        raise ValueError("Synthetic record artifact differs from receipt")
    call = json.loads(Path(client.download_artifacts(run_id, artifacts[0])).read_text())
    if (call["name"] != "synthetic.echo" or call["inputs"] != expected
            or call["outputs"] != expected):
        raise ValueError("Synthetic call artifact payload changed")
    return {
        **receipt, "experiment_id": run.info.experiment_id,
        "trace_ids": [trace.info.trace_id for trace in traces], "payload_verified": True,
    }


def export_probe(directory):
    from evaluator.tracking import export_record

    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "record.json"
    if not record_path.exists():
        started = time.time_ns()
        finished = started + 1_000_000
        call_path = directory / "call.json"
        call_path.write_text(json.dumps({
            "name": "synthetic.echo", "span_type": "TOOL",
            "inputs": {"message": MARKER}, "outputs": {"message": MARKER},
            "attributes": {"role": "preparation", "synthetic_only": True},
            "status": "OK", "start_time_ns": started, "end_time_ns": finished,
        }))
        record_path.write_text(json.dumps({
            "split": "train", "status": "FINISHED", "synthetic_only": True,
            "params": {"case_id": "sc35-preparation-tracking", "campaign_id": CAMPAIGN,
                       "candidate_id": "synthetic-tracking-transport", "role": "preparation"},
            "metrics": {"solver_tokens": 0, "solver_seconds": 0.001,
                        "solver_unreported_attempts": 0, "judge_tokens": 0,
                        "judge_seconds": 0, "judge_unreported_attempts": 0},
            "calls": [], "call_files": [str(call_path)],
            "start_time_ns": started, "end_time_ns": finished,
        }))
    result = export_record(record_path, directory / "shared")
    if result["status"] != "exported":
        raise RuntimeError("Synthetic tracking export is pending; retain and flush its queue")
    receipt = {
        "campaign_id": CAMPAIGN, "marker": MARKER, "run_id": result["run_id"],
        "uri": os.environ["MLFLOW_TRACKING_URI"],
    }
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--directory", required=True, type=Path)
    read = commands.add_parser("read")
    read.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "export":
        result = export_probe(args.directory)
    else:
        receipt = json.loads(args.receipt.read_text())
        mlflow = importlib.import_module("mlflow")
        client = mlflow.MlflowClient(tracking_uri=receipt["uri"])
        result = verify_payload(client, receipt)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
