import argparse
import contextlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cases.sc71.harness.hermes import runtime
from evaluator.client import ClientError, request, search_readiness
from evaluator.reliability import request_timeout
from evaluator.search import Journal

SYSTEM_MESSAGE = (
    "Follow the arena repository instructions and the supplied campaign. "
    "Internet access is only for general research ideas. Never send dataset "
    "examples or knowledge-base contents to external services. All model "
    "requests must use the fixed internal DeepSeek route."
)


class ResearchError(ValueError):
    pass


def private_status(url: str, token_file: Path) -> str | None:
    try:
        result = request("GET", url, token_file, timeout=60)
    except ClientError as error:
        if error.status_code == 404 and error.error_code == "no_evaluation":
            return None
        raise ResearchError("private status unavailable; search was not continued") from None
    status = result.get("status")
    if (
        set(result) != {"status"} or not isinstance(status, str)
        or status not in {"running", "complete", "failed"}
    ):
        raise ResearchError("invalid private status; search was not continued")
    return status


def continuation(status: dict[str, Any]) -> str:
    if status["stopped"]:
        instruction = (
            "The journal records a stop. Do not propose new solutions and do not start "
            "new evaluations. Wait only for evaluations already running, then select and "
            "finalize the frozen batch using existing evidence. The private service must "
            "authorize this early stop; do not invent a human instruction."
        )
    elif status["remaining_proposals"]:
        instruction = (
            f"The search has {status['remaining_proposals']} proposal slots remaining "
            f"({status['proposed']}/{status['max_solutions']} proposed). Continue the search "
            "under the existing campaign conditions. Do not finalize or submit private yet. "
            "Elapsed time, tokens and an adequate finalist count are not stop conditions."
        )
    else:
        instruction = (
            "The proposal budget is exhausted. Do not propose new solutions. Finish planned "
            "evaluations of registered solutions, then select and finalize the frozen batch."
        )
    if status["running"]:
        instruction += (
            f" The journal still has running measurements: {json.dumps(status['running'])}. "
            "Collect train results with evaluator.train result; collect public status and "
            "record its permitted result with evaluator.search finish. Service completion "
            "does not update the journal automatically. If jobs are still running, use "
            "blocking tool/process waits for the calculated remaining duration under "
            "common/long-runs.md, then collect results within this conversation turn. "
            "Do not end turns or repeatedly call the model just to poll status."
        )
    return (
        "Your conversation turn ended, but the campaign is not complete. " + instruction
        + " Campaign completion requires its single private submission to complete; "
        "after authorized finalization, wait for its status only. "
        "Never request private scores or create a replacement private attempt. "
        "If infrastructure prevents progress, report the exact blocker instead of claiming success."
    )


def wait_private(
    private: str | None,
    status: dict[str, Any],
    read_private: Callable[[], str | None],
    wait: Callable[[float], None],
) -> bool:
    if private is None:
        return False
    if (
        status["running"] or not status.get("finalization")
        or (status["remaining_proposals"] and not status["stopped"])
    ):
        raise ResearchError("private attempt already exists before valid search finalization")
    while private == "running":
        wait(60)
        private = read_private()
    if private != "complete":
        raise ResearchError(f"private evaluation is {private or 'unavailable'}; search incomplete")
    return True


def run_search(
    agent_factory: Callable[[], Any],
    prompt: str,
    journal: Journal,
    read_private: Callable[[], str | None],
    check_gateway: Callable[[], None],
    wait: Callable[[float], None] = time.sleep,
    *,
    check_readiness: Callable[[int], object],
) -> dict[str, Any]:
    previous = journal.status()
    try:
        check_readiness(previous["max_solutions"])
    except ClientError:
        raise ResearchError(
            "private search admission is unavailable or mismatched; search was not continued"
        ) from None
    if wait_private(read_private(), previous, read_private, wait):
        return {
            "completed": True, "failed": False,
            "final_response": "Private evaluation complete; no new researcher turn was started.",
        }
    agent = agent_factory()
    history = None
    idle_turns = 0
    try:
        while True:
            result = agent.run_conversation(
                prompt, system_message=SYSTEM_MESSAGE, conversation_history=history,
            )
            check_gateway()
            if result.get("failed") or not result.get("completed"):
                raise ResearchError("Hermes failed; search remains incomplete")
            status = journal.status()
            if wait_private(read_private(), status, read_private, wait):
                return result
            idle_turns = idle_turns + 1 if status == previous else 0
            if idle_turns >= 2:
                raise ResearchError(
                    "two completed turns made no journal progress; search incomplete"
                )
            history = result.get("messages")
            if not isinstance(history, list) or not history:
                raise ResearchError("Hermes returned no conversation history; search incomplete")
            prompt = continuation(status)
            previous = status
    finally:
        agent.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials-file", type=Path, default=os.environ.get(
        "ARENA_MODEL_ACCESS_FILE", "/run/arena/model-access.json"
    ))
    parser.add_argument("--prompt-file", type=Path, default=os.environ.get(
        "ARENA_RESEARCH_PROMPT_FILE", "/workspace/cases/sc71/campaigns/sc71-first-20261019/start.md"
    ))
    parser.add_argument("--cwd", type=Path, default="/workspace")
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--private-url", required=True)
    parser.add_argument("--private-token-file", type=Path, required=True)
    parser.add_argument("--private-finalists", type=int, required=True)
    parser.add_argument("--private-release-id", required=True)
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

                result = run_search(
                    lambda: AIAgent(
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
                    ),
                    prompt, Journal(args.journal),
                    lambda: private_status(args.private_url, args.private_token_file), state.check,
                    check_readiness=lambda maximum: search_readiness(
                        args.private_url, args.private_token_file, timeout=60,
                        require_campaign=True, max_solutions=maximum,
                        finalists=args.private_finalists, release_id=args.private_release_id,
                    ),
                )
        print(json.dumps({"answer": result["final_response"]}, ensure_ascii=False))
        return 0
    except Exception as error:
        reason = str(error) if isinstance(error, ResearchError) else "Hermes researcher failed"
        print(json.dumps({"status": "incomplete", "reason": reason}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
