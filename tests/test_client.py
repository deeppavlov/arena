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
