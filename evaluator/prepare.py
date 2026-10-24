import argparse
import json
from pathlib import Path

from evaluator.metric_package import freeze_metric, load_metric, metric_package_provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or verify a prepared metric package without executing metric code",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--metric", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--metric", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.action == "freeze":
            provenance = freeze_metric(arguments.metric, arguments.output)
        else:
            load_metric(arguments.metric)
            provenance = metric_package_provenance(arguments.metric)
        if provenance is None:
            raise ValueError("prepare requires a metric package directory")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({
        "action": arguments.action,
        "files": len(provenance["files"]), "semantic_tests_run": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
