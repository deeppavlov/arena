import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from cases.sc71.harness.hermes import runtime
from evaluator.reliability import request_timeout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials-file", type=Path, default=os.environ.get(
        "ARENA_MODEL_ACCESS_FILE", "/run/arena/model-access.json"
    ))
    parser.add_argument("--prompt-file", type=Path, default=os.environ.get(
        "ARENA_RESEARCH_PROMPT_FILE", "/workspace/cases/sc71/campaigns/sc71-first-20261019/start.md"
    ))
    parser.add_argument("--cwd", type=Path, default="/workspace")
    args = parser.parse_args()
    try:
        access = runtime.credentials(args.credentials_file)
        if access.get("model", runtime.MODEL) != runtime.MODEL:
            raise ValueError("fixed model required")
        timeout = access.get("model_request_timeout", request_timeout(
            access.get("model_timeout", 21600)
        ))
        prompt = args.prompt_file.read_text()
        home = Path(os.environ.get("HERMES_HOME", "/workspace/.hermes"))
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                runtime.prepare_home(home)
                state = runtime.install(
                    access["model_base_url"], timeout, access.get("api_key", "arena")
                )
                from run_agent import AIAgent

                agent = AIAgent(
                    model=runtime.MODEL,
                    base_url=runtime.CONFIG_BASE_URL,
                    api_key="arena",
                    provider="custom",
                    api_mode="chat_completions",
                    reasoning_config={"enabled": True, "effort": "max"},
                    enabled_toolsets=["terminal", "file", "delegation", "web", "todo"],
                    max_iterations=10000,
                    cwd=str(args.cwd.resolve()),
                    skip_memory=True,
                    skip_background_review=True,
                    skip_context_files=True,
                    save_trajectories=False,
                    quiet_mode=True,
                )
                try:
                    result = agent.run_conversation(
                        prompt,
                        system_message=(
                            "Follow the arena repository instructions and the supplied campaign. "
                            "Internet access is only for general research ideas. "
                            "Never send dataset "
                            "examples or knowledge-base contents to external services. All model "
                            "requests must use the fixed internal DeepSeek route."
                        ),
                    )
                finally:
                    agent.close()
                state.check()
                if result.get("failed") or not result.get("completed"):
                    raise ValueError("Hermes failed")
        print(json.dumps({"answer": result["final_response"]}, ensure_ascii=False))
        return 0
    except Exception:
        print("Hermes researcher failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
