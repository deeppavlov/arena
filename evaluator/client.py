import argparse
import base64
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import httpx
from pydantic import ValidationError

from evaluator.contracts import Candidate, Submission
from evaluator.filesystem import candidate_paths

MAX_CANDIDATE_BYTES = 10 * 1024 * 1024
MAX_BUNDLE_BYTES = 72 * 1024 * 1024


class ClientError(Exception):
    def __init__(
        self, message: str, *, status_code: int | None = None, error_code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


def pack_candidate(identifier: str, directory: Path) -> Candidate:
    files: dict[str, str] = {}
    executable_files: list[str] = []
    total = 0
    try:
        if directory.is_symlink():
            raise ClientError("candidate symlinks are forbidden")
        if not directory.is_dir():
            raise ClientError("candidate must be a directory")
        for path in sorted(candidate_paths(directory)):
            relative = path.relative_to(directory)
            if path.is_symlink():
                raise ClientError("candidate symlinks are forbidden")
            if any(
                part in {".git", ".env"} or part.startswith(".env.")
                for part in relative.parts
            ):
                raise ClientError("candidate contains a restricted path")
            if path.is_dir():
                continue
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                mode = os.fstat(stream.fileno()).st_mode
                if not stat.S_ISREG(mode):
                    raise ClientError("candidate must contain only regular files")
                data = stream.read(MAX_CANDIDATE_BYTES - total + 1)
            total += len(data)
            if total > MAX_CANDIDATE_BYTES:
                raise ClientError("candidate exceeds 10 MiB")
            files[relative.as_posix()] = base64.b64encode(data).decode("ascii")
            if mode & 0o111:
                executable_files.append(relative.as_posix())
            if len(files) > 1000:
                raise ClientError("candidate exceeds 1000 files")
        return Candidate(id=identifier, files=files, executable_files=executable_files)
    except OSError as error:
        raise ClientError("cannot read candidate files") from error
    except ValidationError as error:
        raise ClientError("invalid candidate ID or contents") from error


def request(
    method: str,
    base_url: str,
    token_file: Path,
    payload: Mapping[str, object] | None = None,
    *,
    timeout: float | str | None = None,
    evaluation_id: str | None = None,
    resource: Literal["evaluations", "search-readiness"] = "evaluations",
) -> dict[str, object]:
    if resource not in {"evaluations", "search-readiness"} or (
        resource != "evaluations" and evaluation_id is not None
    ):
        raise ClientError("invalid evaluator resource")
    if evaluation_id is not None and not re.fullmatch(r"[0-9a-f]{64}", evaluation_id):
        raise ClientError("invalid evaluation ID")
    try:
        seconds = float(
            os.environ.get("ARENA_EVALUATOR_TIMEOUT", "3600") if timeout is None else timeout
        )
    except (TypeError, ValueError):
        raise ClientError("evaluator timeout must be positive and finite") from None
    if isinstance(timeout, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ClientError("evaluator timeout must be positive and finite")
    try:
        url = httpx.URL(base_url)
    except httpx.InvalidURL as error:
        raise ClientError("invalid evaluator URL") from error
    if (
        url.scheme not in {"http", "https"}
        or not url.host
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ClientError("invalid evaluator URL")
    try:
        with token_file.open(encoding="utf-8") as stream:
            token = stream.read(4097).strip()
    except (OSError, UnicodeError) as error:
        raise ClientError("cannot read evaluator token file") from error
    if not token or len(token) > 4096 or any(not 33 <= ord(char) <= 126 for char in token):
        raise ClientError("invalid evaluator token file")
    suffix = f"/{evaluation_id}" if evaluation_id is not None else ""
    endpoint = url.copy_with(path=url.path.rstrip("/") + "/v1/" + resource + suffix)
    try:
        with httpx.Client(follow_redirects=False, trust_env=False, timeout=seconds) as client:
            response = client.request(
                method, endpoint, headers={"Authorization": f"Bearer {token}"}, json=payload
            )
    except httpx.HTTPError as error:
        raise ClientError("evaluator request failed") from error
    if not 200 <= response.status_code < 300:
        error_code = None
        try:
            body = response.json()
        except ValueError:
            body = None
        if body == {"detail": {"code": "no_evaluation"}}:
            error_code = "no_evaluation"
        raise ClientError(
            f"evaluator returned HTTP {response.status_code}",
            status_code=response.status_code, error_code=error_code,
        )
    try:
        result = response.json()
    except ValueError as error:
        raise ClientError("evaluator returned invalid JSON") from error
    if not isinstance(result, dict):
        raise ClientError("evaluator returned invalid JSON")
    return result


def search_readiness(
    base_url: str,
    token_file: Path,
    *,
    timeout: float | str | None = None,
    require_campaign: bool = False,
    max_solutions: int | None = None,
    finalists: int | None = None,
    release_id: str | None = None,
) -> dict[str, object]:
    result = request("GET", base_url, token_file, timeout=timeout, resource="search-readiness")
    if (
        set(result) != {"mode", "gate_enabled", "journal_ready", "max_solutions",
                        "finalists", "release_id"}
        or result["mode"] not in ("campaign", "standalone")
        or type(result["gate_enabled"]) is not bool
        or (result["journal_ready"] is not None and type(result["journal_ready"]) is not bool)
        or (result["max_solutions"] is not None and (
            type(result["max_solutions"]) is not int or result["max_solutions"] <= 0
        ))
        or type(result["finalists"]) is not int or result["finalists"] <= 0
        or not isinstance(result["release_id"], str)
    ):
        raise ClientError("invalid search readiness response")
    if require_campaign and (
        result["mode"] != "campaign" or result["gate_enabled"] is not True
        or result["journal_ready"] is not True or result["max_solutions"] is None
    ):
        raise ClientError("search readiness requires an enabled campaign admission gate")
    for name, expected in (
        ("max_solutions", max_solutions), ("finalists", finalists), ("release_id", release_id),
    ):
        if expected is not None and result[name] != expected:
            raise ClientError(f"search readiness differs from expected {name}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze candidates and use the evaluator")
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="freeze explicitly selected candidate directories")
    pack.add_argument("--candidate", action="append", required=True, metavar="ID=DIRECTORY")
    pack.add_argument("--output", type=Path, required=True)
    submit = commands.add_parser("submit", help="submit one frozen finalist batch")
    submit.add_argument("--bundle", type=Path, required=True)
    submit.add_argument(
        "--max-bundle-bytes",
        default=os.environ.get("ARENA_MAX_BUNDLE_BYTES", str(MAX_BUNDLE_BYTES)),
        help="bundle size limit in bytes; overrides ARENA_MAX_BUNDLE_BYTES (default: 75497472)",
    )
    status = commands.add_parser("status", help="read evaluation status and permitted results")
    status.add_argument("--evaluation-id", help="public evaluation ID returned by submit")
    readiness = commands.add_parser("search-readiness", help="verify campaign admission settings")
    readiness.add_argument("--require-campaign", action="store_true")
    readiness.add_argument("--max-solutions", type=int)
    readiness.add_argument("--finalists", type=int)
    readiness.add_argument("--release-id")
    for command in (submit, status, readiness):
        command.add_argument("--url", required=True)
        command.add_argument("--token-file", type=Path, required=True)
        command.add_argument(
            "--timeout",
            help="HTTP timeout in seconds; overrides ARENA_EVALUATOR_TIMEOUT (default: 3600)",
        )
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "pack":
            candidates = []
            for value in arguments.candidate:
                identifier, separator, directory = value.partition("=")
                if not separator or not identifier or not directory:
                    raise ClientError("candidate must use ID=DIRECTORY")
                candidates.append(pack_candidate(identifier, Path(directory)))
            try:
                submission = Submission(candidates=candidates)
            except ValidationError as error:
                raise ClientError("invalid finalist batch") from error
            try:
                with arguments.output.open("x", encoding="utf-8") as stream:
                    stream.write(submission.model_dump_json(indent=2) + "\n")
            except FileExistsError as error:
                raise ClientError("bundle output already exists") from error
            result = {
                "bundle": str(arguments.output),
                "digest": submission.digest(),
                "candidates": [
                    {"id": candidate.id, "digest": candidate.digest()}
                    for candidate in submission.candidates
                ],
            }
        elif arguments.command == "submit":
            try:
                max_bundle_bytes = int(arguments.max_bundle_bytes)
            except ValueError:
                raise ClientError("bundle size limit must be a positive integer") from None
            if max_bundle_bytes <= 0:
                raise ClientError("bundle size limit must be a positive integer")
            try:
                if arguments.bundle.stat().st_size > max_bundle_bytes:
                    raise ClientError("invalid bundle")
                submission = Submission.model_validate_json(arguments.bundle.read_bytes())
            except (OSError, ValidationError) as error:
                raise ClientError("invalid bundle") from error
            result = request(
                "POST",
                arguments.url,
                arguments.token_file,
                submission.model_dump(mode="json"),
                timeout=arguments.timeout,
            )
        elif arguments.command == "search-readiness":
            result = search_readiness(
                arguments.url, arguments.token_file, timeout=arguments.timeout,
                require_campaign=arguments.require_campaign, max_solutions=arguments.max_solutions,
                finalists=arguments.finalists, release_id=arguments.release_id,
            )
        else:
            result = request(
                "GET", arguments.url, arguments.token_file,
                timeout=arguments.timeout, evaluation_id=arguments.evaluation_id,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ClientError as error:
        print(f"arena client: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("arena client: cannot read or write local file", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
