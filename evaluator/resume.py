import argparse
import json
import sys

from evaluator.service import Settings, create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume the sealed failed private evaluation")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--check", action="store_true", help="Validate without resuming evaluation")
    arguments = parser.parse_args()
    try:
        result = create_app(Settings.from_env()).state.resume_private(
            arguments.reason, check_only=arguments.check,
        )
    except Exception as error:
        print(json.dumps({"status": "resume_rejected", "error_type": type(error).__name__}))
        return 1
    print(json.dumps(result))
    return 0 if result["status"] in {"complete", "resume_ready"} else 1


if __name__ == "__main__":
    sys.exit(main())
