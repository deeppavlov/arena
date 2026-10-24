import argparse
import base64
import io
import json
import math
import os
import sqlite3
import stat
import sys
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import ValidationError

from evaluator.candidates import candidate_digest, materialize_files, validate_candidate_path
from evaluator.contracts import Candidate, Submission
from evaluator.filesystem import candidate_paths
from evaluator.numeric import parse_score, validate_score


class SearchError(ValueError):
    pass


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SearchError(f"{field} must be nonempty text")


def _json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise SearchError("record must contain finite JSON values") from error


@dataclass(frozen=True)
class Resources:
    total_tokens: int
    seconds: float
    tokens_complete: bool = True

    def __post_init__(self) -> None:
        if type(self.total_tokens) is not int or self.total_tokens < 0:
            raise SearchError("total_tokens must be a nonnegative integer")
        if not _finite(self.seconds) or self.seconds < 0:
            raise SearchError("seconds must be finite and nonnegative")
        if type(self.tokens_complete) is not bool:
            raise SearchError("tokens_complete must be boolean")


def _snapshot(directory: Path) -> bytes:
    if directory.is_symlink():
        raise SearchError("snapshot symlinks are forbidden")
    if not directory.is_dir():
        raise SearchError("snapshot source must be a directory")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path in sorted(candidate_paths(directory)):
            relative = path.relative_to(directory)
            try:
                validate_candidate_path(relative.as_posix())
            except ValueError as error:
                raise SearchError("snapshot contains an unsafe path") from error
            if path.is_symlink():
                raise SearchError("snapshot symlinks are forbidden")
            if any(
                part in {".git", ".env"} or part.startswith(".env.") for part in relative.parts
            ):
                raise SearchError("snapshot contains a restricted path")
            if path.is_dir():
                continue
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                mode = os.fstat(stream.fileno()).st_mode
                if not stat.S_ISREG(mode):
                    raise SearchError("snapshot must contain regular files and directories")
                member = zipfile.ZipInfo(relative.as_posix())
                member.external_attr = (stat.S_IFREG | (0o755 if mode & 0o111 else 0o644)) << 16
                archive.writestr(member, stream.read())
    return output.getvalue()


def _snapshot_files(snapshot: bytes) -> tuple[dict[str, str], list[str]]:
    with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
        files = {
            member.filename: base64.b64encode(archive.read(member)).decode("ascii")
            for member in archive.infolist() if not member.is_dir()
        }
        executable_files = [
            member.filename for member in archive.infolist()
            if not member.is_dir() and (member.external_attr >> 16) & 0o111
        ]
    return files, executable_files


def _comparison(split: str, dataset_id: str, subset_id: str, conditions_id: str) -> dict[str, str]:
    if split not in {"train", "public"}:
        raise SearchError("split must be train or public")
    result = {
        "split": split,
        "dataset_id": dataset_id,
        "subset_id": subset_id,
        "conditions_id": conditions_id,
    }
    for key, value in result.items():
        _text(value, key)
    return result


class Journal:
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def create(cls, path: Path, *, max_solutions: int) -> "Journal":
        if type(max_solutions) is not int or max_solutions <= 0:
            raise SearchError("max_solutions must be a positive integer")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb"):
                pass
        except FileExistsError as error:
            raise SearchError("journal already exists") from error
        journal = cls(path)
        with journal._connection() as connection:
            connection.execute(
                "CREATE TABLE campaign (max_solutions TEXT NOT NULL, stopped_at TEXT, "
                "stop_reason TEXT, finalization TEXT)"
            )
            connection.execute(
                "INSERT INTO campaign (max_solutions) VALUES (?)", (str(max_solutions),)
            )
            connection.execute(
                "CREATE TABLE candidates (proposal INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, "
                "record TEXT NOT NULL, snapshot BLOB NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE measurements (id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, "
                "record TEXT NOT NULL, completion_order INTEGER UNIQUE)"
            )
            connection.execute("CREATE INDEX measurements_candidate ON measurements(candidate_id)")
        return journal

    @contextmanager
    def _connection(self, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
        uri = "file:" + quote(str(self.path.resolve())) + ("?mode=ro" if readonly else "?mode=rw")
        connection = sqlite3.connect(uri, uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                connection.execute("BEGIN" if readonly else "BEGIN IMMEDIATE")
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _open(connection: sqlite3.Connection) -> sqlite3.Row:
        campaign = connection.execute("SELECT * FROM campaign").fetchone()
        if "finalization" in campaign.keys() and campaign["finalization"] is not None:
            raise SearchError("search is finalized")
        if campaign["stopped_at"] is not None:
            raise SearchError("search is stopped")
        return campaign

    def propose(
        self,
        candidate_id: str,
        directory: Path,
        *,
        parents: Sequence[str] = (),
        hypothesis: str,
        change: str,
        resources: Resources,
    ) -> dict[str, Any]:
        for field, value in (("candidate_id", candidate_id), ("hypothesis", hypothesis),
                             ("change", change)):
            _text(value, field)
        if len(set(parents)) != len(parents):
            raise SearchError("parents must be unique")
        snapshot = b""
        snapshot_error = None
        bundle_error = None
        digest = None
        try:
            snapshot = _snapshot(directory)
        except SearchError as error:
            snapshot_error = str(error)
        except UnicodeError:
            snapshot_error = "snapshot filenames must be UTF-8"
        except OSError:
            snapshot_error = "cannot read snapshot files"
        if snapshot_error is None:
            files, executable_files = _snapshot_files(snapshot)
            digest = candidate_digest(files, executable_files)
            try:
                Candidate(id=candidate_id, files=files, executable_files=executable_files)
            except ValidationError:
                bundle_error = "snapshot does not satisfy the candidate bundle contract"
        with self._connection() as connection:
            campaign = self._open(connection)
            count = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            if count >= int(campaign["max_solutions"]):
                raise SearchError("proposal limit reached")
            if connection.execute(
                "SELECT 1 FROM candidates WHERE id = ?", (candidate_id,)
            ).fetchone():
                raise SearchError("candidate already exists")
            for parent in parents:
                if not connection.execute(
                    "SELECT 1 FROM candidates WHERE id = ?", (parent,)
                ).fetchone():
                    raise SearchError("parent must already exist")
            record = {
                "proposal": count + 1,
                "candidate_id": candidate_id,
                "created_at": datetime.now(UTC).isoformat(),
                "parents": list(parents),
                "hypothesis": hypothesis,
                "change": change,
                "resources": asdict(resources),
                "digest": digest,
                "digest_version": 2,
                "snapshot_error": snapshot_error,
                "bundle_error": bundle_error,
            }
            connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?)",
                (count + 1, candidate_id, _json(record), snapshot),
            )
        return record

    def begin(
        self,
        measurement_id: str,
        candidate_id: str,
        *,
        split: str,
        dataset_id: str,
        subset_id: str,
        conditions_id: str,
        job: str = "",
    ) -> dict[str, Any]:
        comparison = _comparison(split, dataset_id, subset_id, conditions_id)
        _text(measurement_id, "measurement_id")
        with self._connection() as connection:
            self._open(connection)
            candidate = connection.execute(
                "SELECT record FROM candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if candidate is None:
                raise SearchError("candidate does not exist")
            candidate_record = json.loads(candidate["record"])
            if candidate_record.get("snapshot_error"):
                raise SearchError("snapshot is unavailable")
            if connection.execute(
                "SELECT 1 FROM measurements WHERE id = ?", (measurement_id,)
            ).fetchone():
                raise SearchError("measurement already exists")
            for previous in connection.execute(
                "SELECT record FROM measurements WHERE candidate_id = ?", (candidate_id,)
            ):
                value = json.loads(previous["record"])
                if value["comparison"] == comparison and value["state"] != "failed":
                    raise SearchError("measurement already running or succeeded")
            record = {
                "measurement_id": measurement_id,
                "candidate_id": candidate_id,
                "digest": candidate_record["digest"],
                "comparison": comparison,
                "job": job,
                "started_at": datetime.now(UTC).isoformat(),
                "state": "running",
            }
            connection.execute(
                "INSERT INTO measurements (id, candidate_id, record) VALUES (?, ?, ?)",
                (measurement_id, candidate_id, _json(record)),
            )
        return record

    def finish(
        self,
        measurement_id: str,
        *,
        solver: Resources | None = None,
        judge: Resources | None = None,
        score: float | None = None,
        error: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if score is not None:
            try:
                score = validate_score(score)
            except ValueError as failure:
                raise SearchError(str(failure)) from failure
        if (score is None) == (error is None):
            raise SearchError("provide either score or error")
        if error is not None:
            _text(error, "error")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT record FROM measurements WHERE id = ?", (measurement_id,)
            ).fetchone()
            if row is None:
                raise SearchError("measurement does not exist")
            record = json.loads(row["record"])
            if record["state"] != "running":
                raise SearchError("measurement is already finished")
            if record["comparison"]["split"] == "public":
                if evidence is not None or solver is not None or judge is not None:
                    raise SearchError("public accepts only aggregate score; no resource accounting")
                if error not in {None, "evaluation_failed"}:
                    raise SearchError("public failure must use the opaque code evaluation_failed")
            elif solver is None or judge is None:
                raise SearchError("train requires solver and judge resource measurements")
            now = datetime.now(UTC)
            completion_order = connection.execute(
                "SELECT COALESCE(MAX(completion_order), 0) + 1 FROM measurements"
            ).fetchone()[0]
            record.update({
                "state": "succeeded" if error is None else "failed",
                "score": score,
                "error": error,
                "evidence": evidence,
                "solver": None if solver is None else asdict(solver),
                "judge": None if judge is None else asdict(judge),
                "finished_at": now.isoformat(),
                "elapsed_seconds": (
                    now - datetime.fromisoformat(record["started_at"])
                ).total_seconds(),
                "completion_order": completion_order,
                "proposal_count_at_finish": connection.execute(
                    "SELECT COUNT(*) FROM candidates"
                ).fetchone()[0],
            })
            connection.execute(
                "UPDATE measurements SET record = ?, completion_order = ? WHERE id = ?",
                (_json(record), completion_order, measurement_id),
            )
        return record

    @staticmethod
    def _status(connection: sqlite3.Connection) -> dict[str, Any]:
        campaign = dict(connection.execute("SELECT * FROM campaign").fetchone())
        campaign["max_solutions"] = int(campaign["max_solutions"])
        campaign["finalization"] = (
            json.loads(campaign["finalization"]) if campaign.get("finalization") else None
        )
        if campaign["finalization"] is not None and not isinstance(campaign["finalization"], dict):
            raise ValueError("invalid search finalization")
        proposed = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        running = [row[0] for row in connection.execute(
            "SELECT id FROM measurements WHERE completion_order IS NULL ORDER BY rowid"
        )]
        return {
            **campaign,
            "stopped": campaign["stopped_at"] is not None,
            "proposed": proposed,
            "remaining_proposals": campaign["max_solutions"] - proposed,
            "running": running,
            "completed_measurements": connection.execute(
                "SELECT COUNT(*) FROM measurements WHERE completion_order IS NOT NULL"
            ).fetchone()[0],
        }

    @staticmethod
    def _finalists(connection: sqlite3.Connection, submission: Submission) -> None:
        for candidate in submission.candidates:
            row = connection.execute(
                "SELECT record FROM candidates WHERE id = ?", (candidate.id,),
            ).fetchone()
            if row is None:
                raise SearchError("finalist must be registered")
            record = json.loads(row["record"])
            if not isinstance(record, dict):
                raise ValueError("invalid registered candidate")
            if record.get("digest_version") != 2 or record.get("digest") != candidate.digest():
                raise SearchError("finalist digest differs from registered snapshot")
            if record.get("snapshot_error") or record.get("bundle_error"):
                raise SearchError("finalist snapshot is unavailable")

    def finalize(self, submission: Submission, *, replace: bool = False) -> dict[str, Any]:
        with self._connection() as connection:
            status = self._status(connection)
            previous = status["finalization"]
            if previous is not None:
                if previous["batch_digest"] == submission.digest():
                    return previous
                if not replace:
                    raise SearchError("search has a different draft batch; use --replace")
            if status["remaining_proposals"] > 0 and not status["stopped"]:
                raise SearchError("proposal limit is not reached and search is not stopped")
            if status["running"]:
                raise SearchError("search has running evaluations")
            self._finalists(connection, submission)
            finalization: dict[str, Any] = {
                "batch_digest": submission.digest(), "finalized_at": datetime.now(UTC).isoformat(),
            }
            if previous is not None:
                finalization["previous"] = [
                    *previous.get("previous", []),
                    {key: value for key, value in previous.items() if key != "previous"},
                ]
            columns = {row[1] for row in connection.execute("PRAGMA table_info(campaign)")}
            if "finalization" not in columns:
                connection.execute("ALTER TABLE campaign ADD COLUMN finalization TEXT")
            connection.execute("UPDATE campaign SET finalization = ?", (_json(finalization),))
            return finalization

    @contextmanager
    def private_admission(
        self, submission: Submission, *, max_solutions: int, human_stop_path: Path | None = None,
    ) -> Iterator[dict[str, Any]]:
        with self._connection(readonly=True) as connection:
            status = self._status(connection)
            if status["max_solutions"] != max_solutions or status["remaining_proposals"] < 0:
                raise SearchError("search proposal limit differs from configured campaign")
            finalization = status["finalization"]
            if finalization is None or finalization.get("batch_digest") != submission.digest():
                raise SearchError("search is not finalized with this batch")
            if status["running"]:
                raise SearchError("search has running evaluations")
            self._finalists(connection, submission)
            human_stop = None
            if status["remaining_proposals"] > 0:
                if human_stop_path is not None:
                    human_stop = json.loads(human_stop_path.read_text())
                if not isinstance(human_stop, dict) or not status["stopped"]:
                    raise SearchError("early private admission requires a human stop command")
                if (
                    human_stop.get("batch_digest") != submission.digest()
                    or human_stop.get("stopped_at") != status["stopped_at"]
                    or human_stop.get("reason") != status["stop_reason"]
                    or not isinstance(human_stop.get("source_ref"), str)
                    or not human_stop["source_ref"].strip()
                ):
                    raise SearchError("human stop command differs from finalized search")
            yield {**status, "human_stop": human_stop}

    def stop(self, reason: str) -> dict[str, Any]:
        _text(reason, "reason")
        with self._connection() as connection:
            connection.execute(
                "UPDATE campaign SET stopped_at = ?, stop_reason = ? WHERE stopped_at IS NULL",
                (datetime.now(UTC).isoformat(), reason),
            )
            return self._status(connection)

    def status(self) -> dict[str, Any]:
        with self._connection(readonly=True) as connection:
            return self._status(connection)

    def export(self) -> dict[str, Any]:
        with self._connection() as connection:
            return {
                "schema_version": 1,
                "campaign": self._status(connection),
                "candidates": [json.loads(row[0]) for row in connection.execute(
                    "SELECT record FROM candidates ORDER BY proposal"
                )],
                "measurements": [json.loads(row[0]) for row in connection.execute(
                    "SELECT record FROM measurements ORDER BY rowid"
                )],
            }

    def restore(self, candidate_id: str, directory: Path) -> None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT record, snapshot FROM candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if row is None:
                raise SearchError("candidate does not exist")
            snapshot = row["snapshot"]
            record = json.loads(row["record"])
            if record.get("snapshot_error"):
                raise SearchError("snapshot is unavailable")
            legacy = record.get("digest_version", 1) < 2
        if directory.exists() or directory.is_symlink():
            raise SearchError("restore directory already exists")
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            materialize_files(directory, (
                (member.filename, archive.read(member), bool((member.external_attr >> 16) & 0o111))
                for member in archive.infolist() if not member.is_dir()
            ))
            if legacy:
                for member in archive.infolist():
                    name = member.filename.removesuffix("/")
                    validate_candidate_path(name)
                    path = directory / name
                    if member.is_dir():
                        path.mkdir(parents=True, exist_ok=True, mode=0o755)
                        path.chmod(0o755)
                    else:
                        path.chmod((member.external_attr >> 16) & 0o777)

    def curve(
        self,
        *,
        split: str,
        dataset_id: str,
        subset_id: str,
        conditions_id: str,
        direction: str,
        baseline: float | None = None,
        horizon: int | None = None,
    ) -> dict[str, Any]:
        comparison = _comparison(split, dataset_id, subset_id, conditions_id)
        if direction not in {"maximize", "minimize"}:
            raise SearchError("direction must be maximize or minimize")
        if baseline is not None:
            try:
                baseline = validate_score(baseline)
            except ValueError as failure:
                raise SearchError(str(failure)) from failure
        exported = self.export()
        actual_proposals = exported["campaign"]["proposed"]
        if horizon is None:
            horizon = actual_proposals
        if type(horizon) is not int or horizon < actual_proposals:
            raise SearchError("horizon must be an integer at least actual_proposals")
        records = sorted(
            (record for record in exported["measurements"]
             if record["state"] != "running" and record["comparison"] == comparison),
            key=lambda record: record["completion_order"],
        )
        points: list[dict[str, Any]] = []
        quality = baseline
        previous_x = 0
        area = 0.0
        observed_from = 0 if baseline is not None else None
        if baseline is not None:
            points.append({"proposals": 0, "quality": baseline, "source": "baseline"})
        for record in records:
            x = record["proposal_count_at_finish"]
            if quality is not None:
                area += (x - previous_x) * quality
            score = record["score"]
            if score is not None and (
                quality is None or (score > quality if direction == "maximize" else score < quality)
            ):
                quality = score
            if quality is not None:
                if observed_from is None:
                    observed_from = x
                points.append({
                    "proposals": x,
                    "quality": quality,
                    "candidate_id": record["candidate_id"],
                    "measurement_id": record["measurement_id"],
                    "completion_order": record["completion_order"],
                    "finished_at": record["finished_at"],
                    "score": score,
                    "state": record["state"],
                })
            previous_x = x
        if quality is not None:
            area += (horizon - previous_x) * quality
            if previous_x < horizon:
                points.append({"proposals": horizon, "quality": quality, "source": "horizon"})
        if not math.isfinite(area):
            raise SearchError("curve integral exceeds finite numeric range")
        covered = 0 if observed_from is None else horizon - observed_from
        full_auc = area if observed_from == 0 else None
        return {
            "comparison": comparison,
            "direction": direction,
            "baseline": baseline,
            "integration": "left_step",
            "horizon": horizon,
            "actual_proposals": actual_proposals,
            "running": exported["campaign"]["running"],
            "points": points,
            "final_quality": quality,
            "auc": full_auc,
            "mean_quality": None if full_auc is None or not horizon else full_auc / horizon,
            "observed_auc": None if quality is None else area,
            "observed_from": observed_from,
            "covered_proposals": covered,
        }


def _resource_arguments(
    parser: argparse.ArgumentParser, prefix: str = "", *, required: bool = True,
) -> None:
    parser.add_argument(f"--{prefix}tokens", type=int, required=required)
    parser.add_argument(f"--{prefix}seconds", type=float, required=required)
    parser.add_argument(f"--{prefix}tokens-incomplete", action="store_true")


def _resources(arguments: argparse.Namespace, prefix: str = "") -> Resources:
    return Resources(
        total_tokens=getattr(arguments, prefix + "tokens"),
        seconds=getattr(arguments, prefix + "seconds"),
        tokens_complete=not getattr(arguments, prefix + "tokens_incomplete"),
    )


def _optional_resources(arguments: argparse.Namespace, prefix: str) -> Resources | None:
    tokens = getattr(arguments, prefix + "tokens")
    seconds = getattr(arguments, prefix + "seconds")
    incomplete = getattr(arguments, prefix + "tokens_incomplete")
    if tokens is None and seconds is None and not incomplete:
        return None
    if tokens is None or seconds is None:
        raise SearchError("resource tokens and seconds must be supplied together")
    return _resources(arguments, prefix)


def _comparison_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=["train", "public"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--conditions", required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record proposals, lineage and search evidence")
    parser.add_argument("--journal", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--max-solutions", type=int, required=True)
    proposal = commands.add_parser("propose")
    proposal.add_argument("--id", required=True)
    proposal.add_argument("--directory", required=True, type=Path)
    proposal.add_argument("--parent", action="append", default=[])
    proposal.add_argument("--hypothesis", required=True)
    proposal.add_argument("--change", required=True)
    _resource_arguments(proposal)
    starting = commands.add_parser("begin")
    starting.add_argument("--id", required=True)
    starting.add_argument("--candidate", required=True)
    starting.add_argument("--job", default="")
    _comparison_arguments(starting)
    finishing = commands.add_parser("finish")
    finishing.add_argument("--id", required=True)
    outcome = finishing.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--score", type=parse_score)
    outcome.add_argument("--error")
    finishing.add_argument("--evidence", type=Path)
    _resource_arguments(finishing, "solver-", required=False)
    _resource_arguments(finishing, "judge-", required=False)
    stopping = commands.add_parser("stop")
    stopping.add_argument("--reason", required=True)
    finalizing = commands.add_parser("finalize")
    finalizing.add_argument("--bundle", type=Path, required=True)
    finalizing.add_argument("--replace", action="store_true")
    commands.add_parser("status")
    commands.add_parser("export")
    restoring = commands.add_parser("restore")
    restoring.add_argument("--candidate", required=True)
    restoring.add_argument("--directory", required=True, type=Path)
    plotting = commands.add_parser("curve")
    _comparison_arguments(plotting)
    plotting.add_argument("--direction", choices=["maximize", "minimize"], required=True)
    plotting.add_argument("--baseline", type=parse_score)
    plotting.add_argument("--horizon", type=int)
    arguments = parser.parse_args(argv)
    try:
        journal = Journal(arguments.journal)
        if arguments.command == "init":
            result = Journal.create(
                arguments.journal, max_solutions=arguments.max_solutions
            ).status()
        elif arguments.command == "propose":
            result = journal.propose(
                arguments.id, arguments.directory, parents=arguments.parent,
                hypothesis=arguments.hypothesis, change=arguments.change,
                resources=_resources(arguments),
            )
        elif arguments.command == "begin":
            result = journal.begin(
                arguments.id, arguments.candidate, split=arguments.split,
                dataset_id=arguments.dataset, subset_id=arguments.subset,
                conditions_id=arguments.conditions, job=arguments.job,
            )
        elif arguments.command == "finish":
            evidence = None
            if arguments.evidence is not None:
                evidence = json.loads(arguments.evidence.read_text(encoding="utf-8"))
                if not isinstance(evidence, dict):
                    raise SearchError("evidence must be a JSON object")
            result = journal.finish(
                arguments.id, score=arguments.score, error=arguments.error, evidence=evidence,
                solver=_optional_resources(arguments, "solver_"),
                judge=_optional_resources(arguments, "judge_"),
            )
        elif arguments.command == "stop":
            result = journal.stop(arguments.reason)
        elif arguments.command == "finalize":
            result = journal.finalize(
                Submission.model_validate_json(arguments.bundle.read_bytes()),
                replace=arguments.replace,
            )
        elif arguments.command == "status":
            result = journal.status()
        elif arguments.command == "export":
            result = journal.export()
        elif arguments.command == "restore":
            journal.restore(arguments.candidate, arguments.directory)
            result = {"candidate_id": arguments.candidate, "directory": str(arguments.directory)}
        else:
            result = journal.curve(
                split=arguments.split, dataset_id=arguments.dataset, subset_id=arguments.subset,
                conditions_id=arguments.conditions, direction=arguments.direction,
                baseline=arguments.baseline, horizon=arguments.horizon,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except SearchError as error:
        print(f"arena search: {error}", file=sys.stderr)
        return 1
    except (OSError, sqlite3.Error, ValueError):
        print("arena search: cannot read or write valid journal data", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
