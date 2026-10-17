import base64
import json
from pathlib import Path

import httpx
import pytest

from evaluator.client import ClientError, main, pack_candidate, request
from evaluator.contracts import Submission


def test_pack_preserves_relative_paths_and_binary_content(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "data.bin").write_bytes(b"\x00\xff")
    candidate = pack_candidate("idea-1", root)
    assert candidate.id == "idea-1"
    assert candidate.files == {"nested/data.bin": base64.b64encode(b"\x00\xff").decode()}


@pytest.mark.parametrize("mode", [0o700, 0o010, 0o001, 0o6755])
def test_pack_records_executable_semantics_and_distinguishes_identity(
    tmp_path: Path, mode: int
) -> None:
    helper = tmp_path / "helper.sh"
    helper.write_text("#!/bin/sh\nprintf 'answer'\n", encoding="utf-8")
    helper.chmod(0o600)
    plain = pack_candidate("plain", tmp_path)
    helper.chmod(mode | 0o400)
    executable = pack_candidate("executable", tmp_path)
    assert executable.model_dump()["executable_files"] == ["helper.sh"]
    assert plain.digest() != executable.digest()


@pytest.mark.parametrize("name", [".git/config", ".env", ".env.local", "nested/.env"])
def test_pack_rejects_secret_and_repository_paths(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("do-not-print-this", encoding="utf-8")
    with pytest.raises(ClientError, match="restricted path"):
        pack_candidate("idea", tmp_path)


@pytest.mark.parametrize("kind", ["file", "directory", "root"])
def test_pack_rejects_symlinks(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("secret", encoding="utf-8")
    if kind == "root":
        root = tmp_path / "linked"
        root.symlink_to(outside, target_is_directory=True)
    elif kind == "directory":
        (root / "linked").symlink_to(outside, target_is_directory=True)
    else:
        (root / "linked").symlink_to(outside / "secret")
    with pytest.raises(ClientError, match="symlinks"):
        pack_candidate("idea", root)


def test_pack_rejects_oversized_content(tmp_path: Path) -> None:
    with (tmp_path / "large").open("wb") as stream:
        stream.truncate(10 * 1024 * 1024 + 1)
    with pytest.raises(ClientError, match="10 MiB"):
        pack_candidate("idea", tmp_path)


def test_pack_freezes_bundle_and_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "solve.py").write_text("print('answer')", encoding="utf-8")
    output = tmp_path / "frozen.json"
    arguments = ["pack", "--candidate", f"idea={root}", "--output", str(output)]
    assert main(arguments) == 0
    submission = Submission.model_validate_json(output.read_text(encoding="utf-8"))
    assert submission.candidates[0].id == "idea"
    original = output.read_bytes()
    assert main(arguments) == 1
    assert output.read_bytes() == original


def test_submit_uses_bearer_token_and_expected_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text("test-token\n", encoding="utf-8")
    received: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(202, json={"status": "running"})

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    payload = {"candidates": [{"id": "idea", "files": {"a": "eA=="}}]}
    assert request("POST", "http://evaluator:8080", token, payload) == {"status": "running"}
    assert str(received[0].url) == "http://evaluator:8080/v1/evaluations"
    assert received[0].headers["Authorization"] == "Bearer test-token"
    assert json.loads(received[0].content) == payload


@pytest.mark.parametrize("evaluation_id", [None, "abcdef0123456789" * 4])
def test_status_looks_up_the_requested_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evaluation_id: str | None
) -> None:
    token = tmp_path / "token"
    token.write_text("test-token", encoding="utf-8")
    received: list[httpx.Request] = []

    def handle(outgoing: httpx.Request) -> httpx.Response:
        received.append(outgoing)
        return httpx.Response(200, json={"status": "running"})

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    assert request(
        "GET", "https://evaluator/arena/", token, evaluation_id=evaluation_id
    ) == {"status": "running"}
    suffix = f"/{evaluation_id}" if evaluation_id is not None else ""
    assert str(received[0].url) == f"https://evaluator/arena/v1/evaluations{suffix}"


@pytest.mark.parametrize("evaluation_id", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "../x"])
def test_status_rejects_invalid_evaluation_ids(tmp_path: Path, evaluation_id: str) -> None:
    with pytest.raises(ClientError, match="^invalid evaluation ID$"):
        request(
            "GET", "http://evaluator", tmp_path / "missing-token", evaluation_id=evaluation_id
        )


def test_cli_status_accepts_evaluation_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text("test-token", encoding="utf-8")
    received: list[httpx.Request] = []

    def handle(outgoing: httpx.Request) -> httpx.Response:
        received.append(outgoing)
        return httpx.Response(200, json={"status": "running"})

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    evaluation_id = "a" * 64
    assert main([
        "status", "--url", "http://evaluator", "--token-file", str(token),
        "--evaluation-id", evaluation_id,
    ]) == 0
    assert str(received[0].url) == f"http://evaluator/v1/evaluations/{evaluation_id}"


@pytest.mark.parametrize("source", ["environment", "argument"])
@pytest.mark.parametrize("limit", ["1", "1000", "0", "-1", "private-secret"])
def test_submit_bundle_byte_limit_is_configurable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: str,
    limit: str,
) -> None:
    token = tmp_path / "token"
    token.write_text("test-token", encoding="utf-8")
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps({"candidates": [{"id": "idea", "files": {"a": "eA=="}}]}),
        encoding="utf-8",
    )
    received: list[httpx.Request] = []

    def handle(outgoing: httpx.Request) -> httpx.Response:
        received.append(outgoing)
        return httpx.Response(202, json={"status": "running"})

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    monkeypatch.setenv("ARENA_MAX_BUNDLE_BYTES", limit if source == "environment" else "1")
    arguments = [
        "submit", "--url", "http://evaluator", "--token-file", str(token),
        "--bundle", str(bundle),
    ]
    if source == "argument":
        arguments.extend(["--max-bundle-bytes", limit])
    assert main(arguments) == (0 if limit == "1000" else 1)
    output = capsys.readouterr()
    assert "private-secret" not in output.err
    if limit == "1000":
        assert len(received) == 1
    else:
        assert not received
        assert ("invalid bundle" if limit == "1" else "positive integer") in output.err


def test_status_does_not_follow_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text("test-token", encoding="utf-8")
    received: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(307, headers={"Location": "http://untrusted/private"})

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    with pytest.raises(ClientError, match="HTTP 307"):
        request("GET", "http://evaluator:8080", token)
    assert len(received) == 1
    assert str(received[0].url) == "http://evaluator:8080/v1/evaluations"


def test_http_failure_does_not_expose_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text("test-token", encoding="utf-8")
    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(500, text="private-example test-token")
            ),
            **kwargs,
        ),
    )
    with pytest.raises(ClientError) as error:
        request("GET", "http://evaluator:8080", token)
    assert str(error.value) == "evaluator returned HTTP 500"


def test_invalid_bundle_is_not_echoed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text('{"secret": "never-print-me"}', encoding="utf-8")
    assert main([
        "submit", "--url", "http://evaluator", "--token-file", str(tmp_path / "token"),
        "--bundle", str(bundle),
    ]) == 1
    output = capsys.readouterr()
    assert "never-print-me" not in output.err
    assert "invalid bundle" in output.err


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize(
    ("environment", "override", "expected"),
    [(None, None, 3600), ("7200", None, 7200), ("invalid-secret", 10800, 10800)],
)
def test_request_timeout_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    environment: str | None,
    override: float | None,
    expected: float,
) -> None:
    token = tmp_path / "token"
    token.write_text("test-token", encoding="utf-8")
    if environment is None:
        monkeypatch.delenv("ARENA_EVALUATOR_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("ARENA_EVALUATOR_TIMEOUT", environment)
    received: list[httpx.Request] = []

    def handle(outgoing: httpx.Request) -> httpx.Response:
        received.append(outgoing)
        return httpx.Response(202, json={"status": "running"})

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    result = request(method, "http://evaluator", token, timeout=override)
    assert result == {"status": "running"}
    assert received[0].extensions["timeout"] == {
        key: expected for key in ("connect", "read", "write", "pool")
    }
    assert len(received) == 1


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "", "private-secret"])
@pytest.mark.parametrize("source", ["environment", "argument"])
def test_invalid_timeout_is_rejected_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, source: str
) -> None:
    monkeypatch.setenv("ARENA_EVALUATOR_TIMEOUT", value if source == "environment" else "3600")
    with pytest.raises(ClientError, match="^evaluator timeout must be positive and finite$"):
        request(
            "GET",
            "http://evaluator",
            tmp_path / "missing-token",
            timeout=value if source == "argument" else None,
        )


@pytest.mark.parametrize("command", ["submit", "status"])
@pytest.mark.parametrize("value", ["7200", "nan", "private-secret"])
def test_cli_timeout_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    value: str,
) -> None:
    token = tmp_path / "token"
    token.write_text("test-token", encoding="utf-8")
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps({"candidates": [{"id": "idea", "files": {"a": "eA=="}}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARENA_EVALUATOR_TIMEOUT", "invalid-env-secret")
    received: list[httpx.Request] = []

    def handle(outgoing: httpx.Request) -> httpx.Response:
        received.append(outgoing)
        return httpx.Response(202, json={"status": "running"})

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    arguments = [
        command,
        "--url",
        "http://evaluator",
        "--token-file",
        str(token),
        "--timeout",
        value,
    ]
    if command == "submit":
        arguments.extend(["--bundle", str(bundle)])
    assert main(arguments) == (0 if value == "7200" else 1)
    output = capsys.readouterr()
    if value == "7200":
        assert received[0].extensions["timeout"]["read"] == 7200
        assert received[0].method == ("POST" if command == "submit" else "GET")
    else:
        assert not received
        assert output.err == "arena client: evaluator timeout must be positive and finite\n"
