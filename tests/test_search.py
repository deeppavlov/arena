import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from evaluator.client import pack_candidate
from evaluator.search import Journal, Resources, SearchError, main

ZERO = Resources(total_tokens=0, seconds=0)


def propose(journal: Journal, directory: Path, identifier: str, parents: tuple[str, ...] = ()):
    return journal.propose(
        identifier,
        directory,
        parents=parents,
        hypothesis="A general mechanism improves the task metric",
        change="A recorded change to the solution",
        resources=ZERO,
    )


def begin(journal: Journal, identifier: str, candidate: str, **overrides):
    arguments = {
        "split": "train",
        "dataset_id": "dataset-v1",
        "subset_id": "all",
        "conditions_id": "metric-v1-fixed-conditions",
    }
    arguments.update(overrides)
    return journal.begin(identifier, candidate, **arguments)


def finish(journal: Journal, identifier: str, score: float):
    return journal.finish(identifier, score=score, solver=ZERO, judge=ZERO)


def curve(journal: Journal, **overrides):
    arguments = {
        "split": "train",
        "dataset_id": "dataset-v1",
        "subset_id": "all",
        "conditions_id": "metric-v1-fixed-conditions",
        "direction": "minimize",
        "baseline": 10,
    }
    arguments.update(overrides)
    return journal.curve(**arguments)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "solve.py").write_text("print(42)\n", encoding="utf-8")
    (directory / "solve.py").chmod(0o755)
    return directory


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal.create(tmp_path / "search.sqlite", max_solutions=10)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit(tmp_path: Path, limit):
    with pytest.raises(SearchError, match="positive integer"):
        Journal.create(tmp_path / "invalid.sqlite", max_solutions=limit)


def test_limit_retains_failed_parents_and_allows_last_check(tmp_path: Path, source: Path):
    journal = Journal.create(tmp_path / "search.sqlite", max_solutions=2)
    propose(journal, source, "failed")
    begin(journal, "check-1", "failed")
    journal.finish("check-1", error="execution failed", solver=ZERO, judge=ZERO)
    child = propose(journal, source, "child", ("failed",))
    assert child["proposal"] == 2
    assert child["parents"] == ["failed"]
    begin(journal, "check-2", "child")
    finish(journal, "check-2", 3)
    with pytest.raises(SearchError, match="limit"):
        propose(journal, source, "excess")
    assert journal.status()["proposed"] == 2
    assert journal.status()["stopped"] is False
    assert len(journal.export()["candidates"]) == 2


def _race_proposal(database: str, source: str, identifier: str) -> bool:
    try:
        propose(Journal(Path(database)), Path(source), identifier)
        return True
    except SearchError:
        return False


def test_processes_cannot_overfill_limit(tmp_path: Path, source: Path):
    path = tmp_path / "race.sqlite"
    Journal.create(path, max_solutions=1)
    with ProcessPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(
            _race_proposal, [str(path)] * 4, [str(source)] * 4, ["a", "b", "c", "d"]
        ))
    assert outcomes.count(True) == 1
    assert Journal(path).status()["proposed"] == 1


def test_stop_drains_running_only_and_persists(journal: Journal, source: Path):
    propose(journal, source, "running")
    propose(journal, source, "queued")
    begin(journal, "started", "running")
    stopped = journal.stop("human requested selection")
    assert stopped["running"] == ["started"]
    reopened = Journal(journal.path)
    with pytest.raises(SearchError, match="stopped"):
        begin(reopened, "new", "queued")
    with pytest.raises(SearchError, match="stopped"):
        propose(reopened, source, "new")
    finish(reopened, "started", 1)
    assert reopened.status()["running"] == []
    assert reopened.status()["stop_reason"] == "human requested selection"
    assert reopened.stop("another reason")["stop_reason"] == "human requested selection"


def test_snapshot_and_lineage_are_immutable(journal: Journal, source: Path, tmp_path: Path):
    original = propose(journal, source, "one")
    assert original["digest"] == pack_candidate("one", source).digest()
    (source / "solve.py").write_text("changed\n", encoding="utf-8")
    second = propose(journal, source, "two", ("one",))
    assert second["digest"] != original["digest"]
    propose(journal, source, "combined", ("one", "two"))
    target = tmp_path / "restored"
    journal.restore("one", target)
    assert (target / "solve.py").read_text(encoding="utf-8") == "print(42)\n"
    assert (target / "solve.py").stat().st_mode & 0o111
    with pytest.raises(SearchError):
        journal.restore("one", target)
    with pytest.raises(SearchError, match="already exists"):
        propose(journal, source, "one")
    with pytest.raises(SearchError, match="parent"):
        propose(journal, source, "invalid", ("missing",))
    with pytest.raises(SearchError, match="parent"):
        propose(journal, source, "invalid", ("one", "one"))
    assert journal.status()["proposed"] == 3


@pytest.mark.parametrize("path", [".env", ".git/config", "nested/.env.local"])
def test_snapshot_rejects_secret_paths(journal: Journal, source: Path, path: str):
    secret = source / path
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("secret", encoding="utf-8")
    with pytest.raises(SearchError, match="restricted"):
        propose(journal, source, "unsafe")
    assert journal.status()["proposed"] == 0


def test_snapshot_rejects_symlinks(journal: Journal, source: Path):
    (source / "linked").symlink_to(source / "solve.py")
    with pytest.raises(SearchError, match="symlink"):
        propose(journal, source, "unsafe")


def test_records_resources_failures_and_train_evidence(journal: Journal, source: Path):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    solver = Resources(total_tokens=17, seconds=2.5, tokens_complete=False)
    record = journal.finish(
        "check", error="infrastructure failure", solver=solver, judge=ZERO,
        evidence={"trace_path": "traces/check.json", "failed_examples": ["synthetic"]},
    )
    assert record["score"] is None
    assert record["solver"] == {"total_tokens": 17, "seconds": 2.5, "tokens_complete": False}
    assert record["judge"] == {"total_tokens": 0, "seconds": 0, "tokens_complete": True}
    assert record["state"] == "failed"
    assert record["completion_order"] == 1
    assert record["proposal_count_at_finish"] == 1
    assert record["elapsed_seconds"] >= 0
    with pytest.raises(SearchError, match="finished"):
        finish(journal, "check", 5)


def test_public_accepts_only_aggregate_and_private_is_rejected(journal: Journal, source: Path):
    propose(journal, source, "one")
    with pytest.raises(SearchError, match="train or public"):
        begin(journal, "private", "one", split="private")
    begin(journal, "public", "one", split="public")
    with pytest.raises(SearchError, match="public"):
        journal.finish("public", score=1, evidence={"trace": "x"})
    result = journal.finish("public", score=1)
    assert result["solver"] is None
    assert result["judge"] is None
    assert result["elapsed_seconds"] >= 0
    assert curve(journal, baseline=None)["points"] == []


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True])
def test_nonfinite_or_non_numeric_scores_rejected(journal: Journal, source: Path, score):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    with pytest.raises(SearchError, match="finite"):
        finish(journal, "check", score)
    assert journal.status()["running"] == ["check"]


def test_cannot_repeat_valid_same_conditions(journal: Journal, source: Path):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    with pytest.raises(SearchError, match="already"):
        begin(journal, "overlap", "one")
    finish(journal, "check", 1)
    with pytest.raises(SearchError, match="already"):
        begin(journal, "repeat", "one")
    begin(journal, "different-subset", "one", subset_id="other")


def test_curve_uses_completion_frontier_and_step_auc(journal: Journal, source: Path):
    propose(journal, source, "slow")
    begin(journal, "slow-check", "slow")
    propose(journal, source, "second")
    begin(journal, "second-check", "second")
    finish(journal, "second-check", 8)
    propose(journal, source, "third")
    begin(journal, "third-check", "third")
    finish(journal, "third-check", 9)
    propose(journal, source, "fourth")
    finish(journal, "slow-check", 5)
    propose(journal, source, "failed")
    begin(journal, "failed-check", "failed")
    journal.finish("failed-check", error="failure", solver=ZERO, judge=ZERO)
    result = curve(journal)
    assert result["horizon"] == 5
    assert result["auc"] == 41
    assert result["mean_quality"] == 8.2
    assert [(point["proposals"], point["quality"]) for point in result["points"]] == [
        (0, 10), (2, 8), (3, 8), (4, 5), (5, 5)
    ]
    assert result["points"][3]["candidate_id"] == "slow"


def test_curve_requires_comparable_evidence_and_explicit_initial_quality(
    journal: Journal, source: Path,
):
    propose(journal, source, "one")
    begin(journal, "wrong-subset", "one", subset_id="tiny")
    finish(journal, "wrong-subset", 1000)
    begin(journal, "train", "one")
    finish(journal, "train", -5)
    propose(journal, source, "two")
    begin(journal, "train-two", "two")
    finish(journal, "train-two", -2)
    propose(journal, source, "three")
    result = curve(journal, direction="maximize", baseline=None)
    assert result["auc"] is None
    assert result["observed_auc"] == -7
    assert result["observed_from"] == 1
    assert result["covered_proposals"] == 2
    assert result["points"][-1]["quality"] == -2
    assert curve(journal, dataset_id="missing", baseline=None)["observed_auc"] is None


def test_cli_round_trip(tmp_path: Path, source: Path, capsys: pytest.CaptureFixture[str]):
    prefix = ["--journal", str(tmp_path / "search.sqlite")]
    assert main([*prefix, "init", "--max-solutions", "1"]) == 0
    assert main([
        *prefix, "propose", "--id", "one", "--directory", str(source),
        "--hypothesis", "synthetic", "--change", "initial", "--tokens", "0", "--seconds", "1",
    ]) == 0
    assert main([
        *prefix, "begin", "--id", "check", "--candidate", "one", "--split", "train",
        "--dataset", "v1", "--subset", "all", "--conditions", "metric-v1",
    ]) == 0
    assert main([*prefix, "stop", "--reason", "human"]) == 0
    assert main([
        *prefix, "finish", "--id", "check", "--score", "1", "--solver-tokens", "0",
        "--solver-seconds", "1", "--judge-tokens", "0", "--judge-seconds", "0",
    ]) == 0
    assert main([*prefix, "export"]) == 0
    assert "one" in capsys.readouterr().out
    assert main([*prefix, "status"]) == 0
    assert json.loads(capsys.readouterr().out)["running"] == []


def test_large_positive_limit_is_not_a_resource_budget(tmp_path: Path):
    journal = Journal.create(tmp_path / "huge.sqlite", max_solutions=10**30)
    assert journal.status()["max_solutions"] == 10**30


def test_early_stopped_curve_can_use_explicit_common_horizon(journal: Journal, source: Path):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    finish(journal, "check", 2)
    propose(journal, source, "two")
    journal.stop("human")
    result = curve(journal, horizon=5)
    assert result["actual_proposals"] == 2
    assert result["horizon"] == 5
    assert result["auc"] == 18
    assert result["mean_quality"] == 3.6
    assert result["points"][-1] == {"proposals": 5, "quality": 2, "source": "horizon"}
    with pytest.raises(SearchError, match="horizon"):
        curve(journal, horizon=1)


def test_archive_integrity_covers_executable_bits(journal: Journal, source: Path, tmp_path: Path):
    import io
    import sqlite3
    import zipfile

    original = propose(journal, source, "one")
    (source / "solve.py").chmod(0o644)
    changed = propose(journal, source, "two")
    assert original["digest"] == changed["digest"]
    assert original["snapshot_digest"] != changed["snapshot_digest"]
    with sqlite3.connect(journal.path) as connection:
        snapshot = connection.execute(
            "SELECT snapshot FROM candidates WHERE id = 'one'"
        ).fetchone()[0]
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(snapshot)) as old, zipfile.ZipFile(output, "w") as new:
            for member in old.infolist():
                member.external_attr = 0o644 << 16
                new.writestr(member, old.read(member))
        connection.execute(
            "UPDATE candidates SET snapshot = ? WHERE id = 'one'", (output.getvalue(),)
        )
    with pytest.raises(SearchError, match="digest"):
        journal.restore("one", tmp_path / "tampered")


def test_oversized_snapshot_rejected_before_admission(journal: Journal, source: Path):
    with (source / "large.bin").open("wb") as stream:
        stream.truncate(10 * 1024 * 1024 + 1)
    with pytest.raises(SearchError, match="10 MiB"):
        propose(journal, source, "too-large")
    assert journal.status()["proposed"] == 0


def test_public_failure_does_not_accept_detailed_error(journal: Journal, source: Path):
    propose(journal, source, "one")
    begin(journal, "check", "one", split="public")
    with pytest.raises(SearchError, match="public"):
        journal.finish("check", error="hidden example details")
    result = journal.finish("check", error="evaluation_failed")
    assert result["error"] == "evaluation_failed"


@pytest.mark.parametrize("resources", [{}, {"solver": ZERO}, {"judge": ZERO}])
def test_train_requires_both_resource_measurements(journal: Journal, source: Path, resources):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    with pytest.raises(SearchError, match="train requires"):
        journal.finish("check", score=1, **resources)
    assert journal.status()["running"] == ["check"]


@pytest.mark.parametrize("resources", [
    {"solver": ZERO}, {"judge": ZERO}, {"solver": ZERO, "judge": ZERO},
])
def test_public_rejects_resource_accounting(journal: Journal, source: Path, resources):
    propose(journal, source, "one")
    begin(journal, "check", "one", split="public")
    with pytest.raises(SearchError, match="public"):
        journal.finish("check", score=1, **resources)
    assert journal.status()["running"] == ["check"]


def test_public_cli_finishes_with_score_only(
    journal: Journal, source: Path, capsys: pytest.CaptureFixture[str],
):
    propose(journal, source, "one")
    begin(journal, "check", "one", split="public")
    assert main([
        "--journal", str(journal.path), "finish", "--id", "check", "--score", "-3",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["score"] == -3
    assert result["solver"] is None
    assert result["judge"] is None


@pytest.mark.parametrize("arguments", [
    [], ["--solver-tokens", "0"], ["--solver-seconds", "1"],
    ["--judge-tokens", "0"], ["--judge-seconds", "1"],
    ["--solver-tokens-incomplete"], ["--judge-tokens-incomplete"],
    ["--solver-tokens", "0", "--solver-seconds", "1"],
])
def test_train_cli_rejects_missing_or_partial_resources(
    journal: Journal, source: Path, arguments: list[str], capsys: pytest.CaptureFixture[str],
):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    assert main([
        "--journal", str(journal.path), "finish", "--id", "check", "--score", "1", *arguments,
    ]) == 1
    assert "arena search:" in capsys.readouterr().err
    assert journal.status()["running"] == ["check"]
