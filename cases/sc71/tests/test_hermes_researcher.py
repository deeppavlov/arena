import json
import sys
from types import SimpleNamespace
from typing import cast

import httpx
import pytest

from cases.sc71.harness.hermes import researcher
from evaluator import search, train
from evaluator.client import ClientError, pack_candidate, search_readiness
from evaluator.contracts import Submission
from evaluator.search import Journal, Resources


@pytest.fixture
def campaign():
    return {
        "max_solutions": 100,
        "proposed": 11,
        "remaining_proposals": 89,
        "running": [],
        "completed_measurements": 21,
        "stopped": False,
        "stopped_at": None,
        "stop_reason": None,
        "finalization": None,
    }


class Agent:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []
        self.results = []
        self.closed = False

    def run_conversation(self, prompt, *, system_message, conversation_history=None):
        self.calls.append((prompt, system_message, conversation_history))
        action = next(self.actions)
        action()
        messages = [
            *(conversation_history or []), {"role": "user", "content": prompt},
            {"role": "assistant", "content": "done"},
        ]
        result = {
            "completed": True, "failed": False, "final_response": "done", "messages": messages,
        }
        self.results.append(result)
        return result

    def close(self):
        self.closed = True


def execute(campaign, agent, private, check=lambda: None):
    journal = cast(Journal, SimpleNamespace(status=lambda: campaign.copy()))
    return researcher.run_search(
        lambda: agent, "start", journal, lambda: private[0], check,
        check_readiness=lambda _: None,
    )


def finalize(campaign, private):
    campaign.update(proposed=100, remaining_proposals=0, finalization={"batch_digest": "a" * 64})
    private[0] = "complete"


def test_completed_conversation_at_11_of_100_continues_same_history(campaign):
    private = [None]
    agent = Agent([lambda: None, lambda: finalize(campaign, private)])
    result = execute(campaign, agent, private)
    assert result == agent.results[-1]
    assert len(agent.calls) == 2
    assert agent.calls[1][2] is agent.results[0]["messages"]
    assert "89" in agent.calls[1][0]
    assert "Continue" in agent.calls[1][0]
    assert agent.closed


def test_continuation_uses_compacted_history_without_reconstructing_it(campaign):
    private = [None]
    compacted = [{"role": "user", "content": "Compacted prior search"}]

    class CompactingAgent(Agent):
        def run_conversation(self, *args, **kwargs):
            result = super().run_conversation(*args, **kwargs)
            if len(self.calls) == 1:
                result["messages"] = compacted
            return result

    agent = CompactingAgent([lambda: None, lambda: finalize(campaign, private)])
    execute(campaign, agent, private)
    assert agent.calls[1][2] is compacted


@pytest.mark.parametrize("stopped", [False, True])
def test_running_measurement_returns_control_to_result_owner(monkeypatch, tmp_path, stopped):
    private: list[str | None] = [None]
    journal = Journal.create(tmp_path / "search.sqlite", max_solutions=2 if stopped else 1)
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "solve.py").write_text("print('synthetic answer')")
    journal.propose(
        "idea", source, hypothesis="synthetic hypothesis", change="synthetic change",
        resources=Resources(0, 0),
    )
    train_data = tmp_path / "train.jsonl"
    train_data.write_text(json.dumps({"id": "one", "input": "synthetic", "reference": "answer"}))
    evaluation_id = "a" * 64
    record = {}
    requests = []

    def request(method, *args, **kwargs):
        requests.append(method)
        return {"evaluation_id": evaluation_id, "status": "RUNNING"} if method == "POST" else record

    monkeypatch.setattr(train, "request", request)
    common = [
        "--url", "http://train", "--token-file", str(tmp_path / "token"),
        "--journal", str(journal.path), "--measurement", "measurement",
    ]

    def submit():
        assert train.main([
            "submit", *common, "--candidate", "idea", "--dataset", "synthetic",
            "--conditions", "fixed", "--train-data", str(train_data),
        ]) == 0
        measurement = journal.export()["measurements"][0]
        record.update({
            "status": "FINISHED",
            "params": {
                **measurement["comparison"], "candidate_digest": measurement["digest"],
            },
            "metrics": {
                "score": 1,
                **{
                    f"{role}_{name}": value
                    for role in ("solver", "judge")
                    for name, value in {
                        "tokens": 0, "seconds": 1,
                        "unreported_attempts": 0, "resources_complete": True,
                    }.items()
                },
            },
        })
        if stopped:
            journal.stop("human requested selection")

    def collect():
        assert journal.status()["running"] == ["measurement"]
        assert train.main([
            "result", *common, "--evaluation-id", evaluation_id,
            "--output", str(tmp_path / "result.json"),
        ]) == 0
        assert journal.status()["running"] == []
        journal.finalize(Submission(candidates=[pack_candidate("idea", source)]))
        private[0] = "complete"

    def wait(_):
        raise AssertionError("waiting cannot collect the train result")

    agent = Agent([submit, collect])
    researcher.run_search(
        lambda: agent, "start", journal, lambda: private[0], lambda: None, wait,
        check_readiness=lambda _: None,
    )
    assert len(agent.calls) == 2
    assert agent.calls[1][2] is agent.results[0]["messages"]
    assert "measurement" in agent.calls[1][0]
    assert "evaluator.train result" in agent.calls[1][0]
    assert "common/long-runs.md" in agent.calls[1][0]
    if stopped:
        assert "do not start" in agent.calls[1][0]
    assert requests == ["POST", "GET"]
    assert journal.status()["completed_measurements"] == 1
    assert agent.closed


def test_running_public_measurement_can_be_recorded_on_next_turn(tmp_path):
    journal = Journal.create(tmp_path / "search.sqlite", max_solutions=1)
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "solve.py").write_text("print('synthetic answer')")
    journal.propose(
        "idea", source, hypothesis="synthetic hypothesis", change="synthetic change",
        resources=Resources(0, 0),
    )
    private: list[str | None] = [None]

    def submit():
        journal.begin(
            "public-measurement", "idea", split="public", dataset_id="synthetic",
            subset_id="all", conditions_id="fixed",
        )

    def collect():
        assert search.main([
            "--journal", str(journal.path), "finish", "--id", "public-measurement", "--score", "1",
        ]) == 0
        journal.finalize(Submission(candidates=[pack_candidate("idea", source)]))
        private[0] = "complete"

    def wait(_):
        raise AssertionError("waiting cannot record the public result")

    agent = Agent([submit, collect])
    researcher.run_search(
        lambda: agent, "start", journal, lambda: private[0], lambda: None, wait,
        check_readiness=lambda _: None,
    )
    assert len(agent.calls) == 2
    assert journal.status()["completed_measurements"] == 1
    assert "evaluator.search finish" in agent.calls[1][0]


@pytest.mark.parametrize("running", [[], ["measurement"]])
def test_two_completed_turns_without_progress_fail_instead_of_looping(campaign, running):
    campaign["running"] = running
    agent = Agent([lambda: None, lambda: None])
    with pytest.raises(researcher.ResearchError, match="no journal progress"):
        execute(campaign, agent, [None])
    assert len(agent.calls) == 2
    assert agent.closed


def test_completed_measurements_count_as_progress(campaign):
    private = [None]

    def measure():
        campaign["completed_measurements"] += 1

    agent = Agent([measure, measure, measure, lambda: finalize(campaign, private)])
    execute(campaign, agent, private)
    assert len(agent.calls) == 4


@pytest.mark.parametrize("status", ["running", "failed", "complete"])
def test_existing_private_attempt_never_creates_agent(campaign, status):
    def forbidden():
        raise AssertionError("must not create Hermes for a consumed private attempt")

    journal = cast(Journal, SimpleNamespace(status=lambda: campaign.copy()))
    with pytest.raises(researcher.ResearchError, match="private attempt already exists"):
        researcher.run_search(
            forbidden, "start", journal, lambda: status, lambda: None,
            check_readiness=lambda _: None,
        )


def test_failed_private_attempt_prevents_continuation(campaign):
    private: list[str | None] = [None]

    def submit():
        finalize(campaign, private)
        private[0] = "failed"

    agent = Agent([submit])
    with pytest.raises(researcher.ResearchError, match="private evaluation is failed"):
        execute(campaign, agent, private)
    assert len(agent.calls) == 1
    assert agent.closed


@pytest.mark.parametrize("state", [
    {"remaining_proposals": 89},
    {"running": ["measurement"]},
    {"finalization": None},
])
def test_private_complete_does_not_hide_incomplete_search(campaign, state):
    private = [None]

    def submit():
        finalize(campaign, private)
        campaign.update(state)

    agent = Agent([submit])
    with pytest.raises(researcher.ResearchError, match="before valid search finalization"):
        execute(campaign, agent, private)
    assert len(agent.calls) == 1


def test_running_private_is_waited_without_another_model_turn(campaign):
    private = [None]
    responses = iter([None, "running", "running", "complete"])
    waits = []
    agent = Agent([lambda: finalize(campaign, private)])
    journal = cast(Journal, SimpleNamespace(status=lambda: campaign.copy()))
    result = researcher.run_search(
        lambda: agent, "start", journal, lambda: next(responses), lambda: None, waits.append,
        check_readiness=lambda _: None,
    )
    assert result is agent.results[0]
    assert waits == [60, 60]
    assert len(agent.calls) == 1
    assert agent.closed


@pytest.mark.parametrize("statuses", [["complete"], ["running", "running", "complete"]])
def test_finalized_private_can_finish_without_creating_agent(campaign, statuses):
    finalize(campaign, [None])
    responses = iter(statuses)
    waits = []

    def forbidden():
        raise AssertionError("must not create Hermes")

    journal = cast(Journal, SimpleNamespace(status=lambda: campaign.copy()))
    result = researcher.run_search(
        forbidden, "start", journal, lambda: next(responses), lambda: None, waits.append,
        check_readiness=lambda _: None,
    )
    assert result["completed"] is True
    assert len(waits) == len(statuses) - 1


def test_disappearing_private_attempt_is_never_treated_as_unused(campaign):
    finalize(campaign, [None])
    responses = iter(["running", None])

    def forbidden():
        raise AssertionError("must not create Hermes")

    journal = cast(Journal, SimpleNamespace(status=lambda: campaign.copy()))
    with pytest.raises(researcher.ResearchError, match="private evaluation is unavailable"):
        researcher.run_search(
            forbidden, "start", journal, lambda: next(responses), lambda: None, lambda _: None,
            check_readiness=lambda _: None,
        )


def test_exhausted_budget_continues_only_checks_and_finalization(campaign):
    private = [None]
    campaign.update(proposed=100, remaining_proposals=0)
    agent = Agent([lambda: None, lambda: finalize(campaign, private)])
    execute(campaign, agent, private)
    assert "Do not propose" in agent.calls[1][0]
    assert "finalize" in agent.calls[1][0]


def test_stopped_search_continues_only_drain_and_selection(campaign):
    private: list[str | None] = [None]
    campaign.update(stopped=True, stopped_at="2026-10-23T00:00:00+00:00")

    def submit():
        campaign["finalization"] = {"batch_digest": "a" * 64}
        private[0] = "complete"

    agent = Agent([lambda: None, submit])
    execute(campaign, agent, private)
    assert "Do not propose" in agent.calls[1][0]
    assert "do not start" in agent.calls[1][0]
    assert campaign["remaining_proposals"] == 89


@pytest.mark.parametrize("failure", ["failed", "incomplete", "gateway"])
def test_model_failure_never_starts_another_conversation(campaign, failure):
    agent = Agent([lambda: None])
    original = agent.run_conversation

    def run(*args, **kwargs):
        result = original(*args, **kwargs)
        if failure == "failed":
            result["failed"] = True
        elif failure == "incomplete":
            result["completed"] = False
        return result

    agent.run_conversation = run

    def check():
        if failure == "gateway":
            raise RuntimeError("gateway failed")

    with pytest.raises((researcher.ResearchError, RuntimeError)):
        execute(campaign, agent, [None], check)
    assert len(agent.calls) == 1
    assert agent.closed


def test_missing_conversation_history_is_explicit_failure(campaign):
    agent = Agent([lambda: None])
    original = agent.run_conversation

    def run(*args, **kwargs):
        result = original(*args, **kwargs)
        del result["messages"]
        return result

    agent.run_conversation = run
    with pytest.raises(researcher.ResearchError, match="conversation history"):
        execute(campaign, agent, [None])
    assert len(agent.calls) == 1


@pytest.mark.parametrize("status", ["running", "failed", "complete"])
def test_private_status_reads_only_status(monkeypatch, tmp_path, status):
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": status}

    monkeypatch.setattr(researcher, "request", request)
    token_file = tmp_path / "token"
    assert researcher.private_status("http://private", token_file) == status
    assert calls[0][0] == ("GET", "http://private", token_file)
    assert calls[0][1] == {"timeout": 60}


@pytest.mark.parametrize(("code", "body", "unused"), [
    (404, {"detail": {"code": "no_evaluation"}}, True),
    (404, {"detail": "Not Found"}, False),
    (404, {"detail": "no evaluation"}, False),
    (401, {"detail": {"code": "no_evaluation"}}, False),
    (503, {"detail": "unavailable"}, False),
])
def test_only_structured_private_absence_means_unused(monkeypatch, tmp_path, code, body, unused):
    token = tmp_path / "token"
    token.write_text("synthetic-token")
    original_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(lambda _: httpx.Response(code, json=body)), **kwargs,
    ))
    if unused:
        assert researcher.private_status("http://private", tmp_path / "token") is None
    else:
        with pytest.raises(researcher.ResearchError, match="private status unavailable"):
            researcher.private_status("http://private", tmp_path / "token")


def test_wrong_private_api_path_fails_before_agent_is_created(monkeypatch, tmp_path, campaign):
    token = tmp_path / "token"
    token.write_text("synthetic-token")
    paths = []

    def handle(request):
        paths.append(request.url.path)
        return httpx.Response(404, json={"detail": "Not Found"})

    original_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handle), **kwargs,
    ))

    def forbidden():
        raise AssertionError("must not create Hermes for an invalid private URL")

    journal = cast(Journal, SimpleNamespace(status=lambda: campaign.copy()))
    with pytest.raises(researcher.ResearchError, match="private status unavailable"):
        researcher.run_search(
            forbidden, "start", journal,
            lambda: researcher.private_status("http://private/v1", token), lambda: None,
            check_readiness=lambda _: None,
        )
    assert paths == ["/v1/v1/evaluations"]


@pytest.mark.parametrize("failure", ["unavailable", "disabled", "budget"])
def test_search_readiness_blocks_agent_before_private_lookup(
    monkeypatch, tmp_path, campaign, failure,
):
    def request(*args, **kwargs):
        if failure == "unavailable":
            raise ClientError("evaluator returned HTTP 404", status_code=404)
        return {
            "mode": "campaign", "gate_enabled": failure != "disabled", "journal_ready": True,
            "max_solutions": 99 if failure == "budget" else 100,
            "finalists": 10, "output_count": 4, "evaluation_enabled": True,
            "release_id": "synthetic-release",
        }

    monkeypatch.setattr("evaluator.client.request", request)

    def forbidden():
        raise AssertionError("must not create agent or read private until readiness succeeds")

    journal = cast(Journal, SimpleNamespace(status=lambda: campaign.copy()))
    with pytest.raises(researcher.ResearchError, match="search admission is unavailable"):
        researcher.run_search(
            forbidden, "start", journal, forbidden, lambda: None,
            check_readiness=lambda maximum: search_readiness(
                "http://private", tmp_path / "token", require_campaign=True, max_solutions=maximum,
            ),
        )


def test_private_response_with_results_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(researcher, "request", lambda *args, **kwargs: {
        "status": "complete", "scores": [],
    })
    with pytest.raises(researcher.ResearchError, match="invalid private status"):
        researcher.private_status("http://private", tmp_path / "token")


@pytest.mark.parametrize("mismatch", [None, "finalists", "output_count", "release_id", "empty"])
def test_launcher_verifies_private_conditions_before_starting(
    monkeypatch, tmp_path, campaign, capsys, mismatch,
):
    prompt = tmp_path / "start.md"
    prompt.write_text("Synthetic campaign")
    monkeypatch.setattr(sys, "argv", [
        "researcher", "--prompt-file", str(prompt), "--journal", str(tmp_path / "search.sqlite"),
        "--private-url", "http://private", "--private-token-file", str(tmp_path / "token"),
        "--private-finalists", "10", "--private-output-count", "4",
        "--private-release-id", "synthetic-release",
    ])
    monkeypatch.setattr(researcher.runtime, "credentials", lambda path: {
        "model_base_url": "http://gateway/runs/synthetic/secret/v1",
    })
    monkeypatch.setattr(researcher.runtime, "prepare_home", lambda path: None)
    monkeypatch.setattr(
        researcher.runtime, "install", lambda *args: SimpleNamespace(check=lambda: None),
    )
    monkeypatch.setattr(
        researcher, "Journal", lambda path: SimpleNamespace(status=lambda: campaign),
    )
    requests = []
    private_requests = []

    def readiness(*args, **kwargs):
        requests.append((args, kwargs))
        return {
            "mode": "campaign", "gate_enabled": True, "journal_ready": True, "max_solutions": 100,
            "finalists": 9 if mismatch == "finalists" else 10,
            "output_count": 3 if mismatch == "output_count" else 4,
            "evaluation_enabled": mismatch != "empty",
            "release_id": "other-release" if mismatch == "release_id" else "synthetic-release",
        }

    def private(*args):
        private_requests.append(args)
        return "complete"

    monkeypatch.setattr("evaluator.client.request", readiness)
    monkeypatch.setattr(researcher, "private_status", private)

    def forbidden(**kwargs):
        raise AssertionError("must not create Hermes")

    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=forbidden))
    assert researcher.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert requests == [(
        ("GET", "http://private", tmp_path / "token"),
        {"timeout": 60, "resource": "search-readiness"},
    )]
    assert bool(private_requests) is (mismatch is None)
    assert json.loads(captured.err) == {
        "status": "incomplete",
        "reason": (
            "private search admission is unavailable or mismatched; search was not continued"
            if mismatch else "private attempt already exists before valid search finalization"
        ),
    }


@pytest.mark.parametrize("missing", [
    "--private-finalists", "--private-output-count", "--private-release-id",
])
def test_launcher_requires_expected_private_conditions(monkeypatch, tmp_path, capsys, missing):
    arguments = [
        "researcher", "--journal", str(tmp_path / "search.sqlite"),
        "--private-url", "http://private", "--private-token-file", str(tmp_path / "token"),
    ]
    for flag, value in (
        ("--private-finalists", "10"), ("--private-output-count", "4"),
        ("--private-release-id", "synthetic"),
    ):
        if flag != missing:
            arguments.extend([flag, value])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as error:
        researcher.main()
    assert error.value.code == 2
    assert missing in capsys.readouterr().err
