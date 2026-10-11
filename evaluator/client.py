import argparse
import base64
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
from pydantic import ValidationError

from evaluator.contracts import Candidate, Submission

MAX_CANDIDATE_BYTES = 10 * 1024 * 1024
MAX_BUNDLE_BYTES = 72 * 1024 * 1024


class ClientError(Exception):
    pass


def pack_candidate(identifier: str, directory: Path) -> Candidate:
    files: dict[str, str] = {}
    total = 0
    try:
        if directory.is_symlink():
            raise ClientError("candidate symlinks are forbidden")
        if not directory.is_dir():
            raise ClientError("candidate must be a directory")
        for path in sorted(directory.rglob("*")):
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
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ClientError("candidate must contain only regular files")
                data = stream.read(MAX_CANDIDATE_BYTES - total + 1)
            total += len(data)
            if total > MAX_CANDIDATE_BYTES:
                raise ClientError("candidate exceeds 10 MiB")
            files[relative.as_posix()] = base64.b64encode(data).decode("ascii")
            if len(files) > 1000:
                raise ClientError("candidate exceeds 1000 files")
        return Candidate(id=identifier, files=files)
    except OSError as error:
        raise ClientError("cannot read candidate files") from error
    except ValidationError as error:
        raise ClientError("invalid candidate ID or contents") from error


def request(
    method: str,
    base_url: str,
    token_file: Path,
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
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
    endpoint = url.copy_with(path=url.path.rstrip("/") + "/v1/evaluations")
    try:
        with httpx.Client(follow_redirects=False, trust_env=False, timeout=30) as client:
            response = client.request(
                method, endpoint, headers={"Authorization": f"Bearer {token}"}, json=payload
            )
    except httpx.HTTPError as error:
        raise ClientError("evaluator request failed") from error
    if not 200 <= response.status_code < 300:
        raise ClientError(f"evaluator returned HTTP {response.status_code}")
    try:
        result = response.json()
    except ValueError as error:
        raise ClientError("evaluator returned invalid JSON") from error
    if not isinstance(result, dict):
        raise ClientError("evaluator returned invalid JSON")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze candidates and use the private evaluator")
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="freeze explicitly selected candidate directories")
    pack.add_argument("--candidate", action="append", required=True, metavar="ID=DIRECTORY")
    pack.add_argument("--output", type=Path, required=True)
    submit = commands.add_parser("submit", help="submit one frozen finalist batch")
    submit.add_argument("--bundle", type=Path, required=True)
    status = commands.add_parser("status", help="read the sealed batch status and final aggregates")
    for command in (submit, status):
        command.add_argument("--url", required=True)
        command.add_argument("--token-file", type=Path, required=True)
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
                if arguments.bundle.stat().st_size > MAX_BUNDLE_BYTES:
                    raise ClientError("invalid bundle")
                submission = Submission.model_validate_json(arguments.bundle.read_bytes())
            except (OSError, ValidationError) as error:
                raise ClientError("invalid bundle") from error
            result = request(
                "POST", arguments.url, arguments.token_file, submission.model_dump(mode="json")
            )
        else:
            result = request("GET", arguments.url, arguments.token_file)
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
