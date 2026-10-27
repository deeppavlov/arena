import base64
import json
import os
from pathlib import Path

import httpx
import pytest

from evaluator.client import ClientError, main, pack_candidate, request, search_readiness
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


@pytest.mark.parametrize("failure", [PermissionError, FileNotFoundError])
def test_pack_rejects_unreadable_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[OSError],
) -> None:
    (tmp_path / "solve.py").write_text("print(42)", encoding="utf-8")
    blocked = tmp_path / "config"
    blocked.mkdir()
    (blocked / "options.json").write_text("{}", encoding="utf-8")
    original_scandir = os.scandir

    def scandir(path):
        if Path(path) == blocked:
            raise failure("directory cannot be read")
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    with pytest.raises(ClientError, match="cannot read candidate files"):
        pack_candidate("idea", tmp_path)


def test_pack_handles_deep_empty_directories(tmp_path: Path) -> None:
    (tmp_path / "solve.py").write_text("print(42)", encoding="utf-8")
    directories = []
    directory = tmp_path
    try:
        for _ in range(1100):
            directory /= "d"
            directory.mkdir()
            directories.append(directory)
        assert set(pack_candidate("idea", tmp_path).files) == {"solve.py"}
    finally:
        for directory in reversed(directories):
            directory.rmdir()


def test_pack_preserves_content_above_former_size_limit(tmp_path: Path) -> None:
    size = 10 * 1024 * 1024 + 1
    with (tmp_path / "large").open("wb") as stream:
        stream.truncate(size)
    candidate = pack_candidate("idea", tmp_path)
    assert base64.b64decode(candidate.files["large"]) == b"\x00" * size


@pytest.mark.parametrize("executable", [False, True])
def test_pack_preserves_more_than_one_thousand_files(tmp_path: Path, executable: bool) -> None:
    files = {f"file-{i}": str(i).encode() for i in range(1001)}
    for name, content in files.items():
        path = tmp_path / name
        path.write_bytes(content)
        path.chmod(0o700 if executable else 0o600)
    candidate = pack_candidate("idea", tmp_path)
    assert {name: base64.b64decode(value) for name, value in candidate.files.items()} == files
    assert candidate.executable_files == (sorted(files) if executable else [])


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


@pytest.mark.parametrize(("body", "expected"), [
    ({"detail": {"code": "no_evaluation"}}, "no_evaluation"),
    ({"detail": "Not Found"}, None),
    ({"detail": "no evaluation"}, None),
    ({"detail": {"code": "private-example test-token"}}, None),
    ({"detail": {"code": "no_evaluation", "message": "private-example"}}, None),
    ({"detail": {"code": ["no_evaluation"]}}, None),
    ([], None),
])
def test_http_failure_keeps_only_known_structured_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: object, expected: str | None,
) -> None:
    token = tmp_path / "token"
    token.write_text("test-token")
    original_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(lambda _: httpx.Response(404, json=body)), **kwargs,
    ))
    with pytest.raises(ClientError) as error:
        request("GET", "http://evaluator", token)
    assert error.value.status_code == 404
    assert error.value.error_code == expected
    assert str(error.value) == "evaluator returned HTTP 404"


def test_non_http_client_error_has_no_response_code(tmp_path: Path) -> None:
    with pytest.raises(ClientError) as error:
        request("GET", "http://evaluator", tmp_path / "missing-token")
    assert error.value.status_code is None
    assert error.value.error_code is None


@pytest.fixture
def campaign_readiness():
    return {
        "mode": "campaign", "gate_enabled": True, "journal_ready": True,
        "max_solutions": 100, "finalists": 10, "output_count": 3,
        "evaluation_enabled": True, "release_id": "synthetic-release",
    }


def test_search_readiness_cli_checks_published_campaign_conditions(
    monkeypatch, tmp_path, campaign_readiness, capsys,
):
    token = tmp_path / "token"
    token.write_text("test-token")
    received = []

    def handle(outgoing):
        received.append(outgoing)
        return httpx.Response(200, json=campaign_readiness)

    original_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handle), **kwargs,
    ))
    assert main([
        "search-readiness", "--url", "http://evaluator", "--token-file", str(token),
        "--require-campaign", "--max-solutions", "100", "--finalists", "10",
        "--output-count", "3", "--release-id", "synthetic-release",
    ]) == 0
    assert str(received[0].url) == "http://evaluator/v1/search-readiness"
    assert received[0].method == "GET"
    assert received[0].headers["Authorization"] == "Bearer test-token"
    assert json.loads(capsys.readouterr().out) == campaign_readiness


@pytest.mark.parametrize("changed", [
    {"mode": "standalone", "gate_enabled": False, "journal_ready": None, "max_solutions": None},
    {"gate_enabled": False},
    {"gate_enabled": 1},
    {"journal_ready": False},
    {"evaluation_enabled": False},
    {"max_solutions": 99},
    {"max_solutions": True},
    {"finalists": 9},
    {"output_count": 2},
    {"release_id": "another-release"},
    {"score": 4},
])
def test_search_readiness_rejects_disabled_or_mismatched_campaign(
    monkeypatch, tmp_path, campaign_readiness, changed,
):
    monkeypatch.setattr("evaluator.client.request", lambda *args, **kwargs: {
        **campaign_readiness, **changed,
    })
    with pytest.raises(ClientError, match="search readiness"):
        search_readiness(
            "http://evaluator", tmp_path / "token", require_campaign=True,
            max_solutions=100, finalists=10, output_count=3, release_id="synthetic-release",
        )


@pytest.mark.parametrize("changed", [
    {"output_count": 0}, {"output_count": -1}, {"output_count": 11},
    {"output_count": True}, {"output_count": 3.0}, {"output_count": "3"},
    {"output_count": None}, {"evaluation_enabled": 1}, {"evaluation_enabled": 0},
    {"evaluation_enabled": "true"}, {"evaluation_enabled": None},
])
def test_search_readiness_validates_output_count_and_availability_without_expectations(
    monkeypatch, tmp_path, campaign_readiness, changed,
):
    monkeypatch.setattr("evaluator.client.request", lambda *args, **kwargs: {
        **campaign_readiness, **changed,
    })
    with pytest.raises(ClientError, match="invalid search readiness response"):
        search_readiness("http://evaluator", tmp_path / "token")


def test_search_readiness_rejects_legacy_response_without_output_count_and_availability(
    monkeypatch, tmp_path, campaign_readiness,
):
    legacy = {key: value for key, value in campaign_readiness.items()
              if key not in {"output_count", "evaluation_enabled"}}
    monkeypatch.setattr("evaluator.client.request", lambda *args, **kwargs: legacy)
    with pytest.raises(ClientError, match="invalid search readiness response"):
        search_readiness("http://evaluator", tmp_path / "token", require_campaign=True)


@pytest.mark.parametrize("output_count", [0, -1, True, 3.0, "3"])
def test_search_readiness_rejects_invalid_expected_output_count(
    monkeypatch, tmp_path, output_count,
):
    def request(*args, **kwargs):
        raise AssertionError("invalid expected counts must not query the evaluator")

    monkeypatch.setattr("evaluator.client.request", request)
    with pytest.raises(ClientError, match="output_count"):
        search_readiness("http://evaluator", tmp_path / "token", output_count=output_count)


def test_search_readiness_can_inspect_disabled_evaluation_without_campaign_requirement(
    monkeypatch, tmp_path, campaign_readiness,
):
    published = {**campaign_readiness, "evaluation_enabled": False}
    monkeypatch.setattr("evaluator.client.request", lambda *args, **kwargs: published)
    assert search_readiness("http://evaluator", tmp_path / "token") == published


def test_search_readiness_cli_rejects_mismatched_output_count(
    monkeypatch, tmp_path, campaign_readiness, capsys,
):
    monkeypatch.setattr("evaluator.client.request", lambda *args, **kwargs: campaign_readiness)
    assert main([
        "search-readiness", "--url", "http://evaluator", "--token-file", str(tmp_path / "token"),
        "--require-campaign", "--output-count", "2",
    ]) == 1
    assert "differs from expected output_count" in capsys.readouterr().err


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


@pytest.mark.parametrize("state", ["complete", "failed"])
@pytest.mark.parametrize("public", [False, True])
def test_wait_saves_terminal_response_without_resubmitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    state: str, public: bool,
) -> None:
    token = tmp_path / "token"
    token.write_text("test-token")
    output = tmp_path / "status.json"
    records: list[dict[str, object]] = [{"status": "running"}, {"status": state}]
    if public:
        for record in records:
            record["evaluation_id"] = "a" * 64
        if state == "complete":
            records[-1]["scores"] = [{"candidate_id": "idea", "score": 0.0}]
    received: list[httpx.Request] = []
    sleeps = []

    def handle(outgoing: httpx.Request) -> httpx.Response:
        received.append(outgoing)
        return httpx.Response(200, json=records[len(received) - 1])

    def sleep(seconds: float) -> None:
        assert json.loads(output.read_text()) == records[0]
        sleeps.append(seconds)

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    monkeypatch.setattr("time.sleep", sleep)
    args = ["wait", "--url", "http://evaluator", "--token-file", str(token),
            "--output", str(output), "--poll-seconds", "2.5", "--timeout", "42"]
    if public:
        args.extend(["--evaluation-id", "a" * 64])
    assert main(args) == (0 if state == "complete" else 3)
    assert json.loads(output.read_text()) == records[-1]
    assert json.loads(capsys.readouterr().out) == records[-1]
    assert sleeps == [2.5]
    assert len(received) == 2
    assert all(item.method == "GET" for item in received)
    assert all(item.url.path == "/v1/evaluations" + ("/" + "a" * 64 if public else "")
               for item in received)
    assert all(item.extensions["timeout"]["read"] == 42 for item in received)


@pytest.mark.parametrize("failure", ["unknown", "missing", "unhashable", "http", "network", "json"])
def test_wait_stops_on_invalid_response_or_transport_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], failure: str,
) -> None:
    token = tmp_path / "token"
    token.write_text("test-token")
    output = tmp_path / "status.json"
    received = []

    def handle(outgoing: httpx.Request) -> httpx.Response:
        received.append(outgoing)
        if failure == "network":
            raise httpx.ConnectError("request failed", request=outgoing)
        if failure == "http":
            return httpx.Response(503, json={"detail": "unavailable"})
        if failure == "json":
            return httpx.Response(200, content=b"invalid JSON")
        record = {"unknown": {"status": "FAILED"}, "missing": {}, "unhashable": {"status": []}}
        return httpx.Response(200, json=record[failure])

    def sleep(seconds: float) -> None:
        raise AssertionError("Invalid status or request failure must stop waiting")

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    monkeypatch.setattr("time.sleep", sleep)
    assert main(["wait", "--url", "http://evaluator", "--token-file", str(token),
                 "--output", str(output)]) == 1
    assert len(received) == 1
    assert received[0].method == "GET"
    assert not output.exists()
    assert capsys.readouterr().err.startswith("arena client: ")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "bad-interval"])
def test_wait_rejects_invalid_poll_interval_before_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str,
) -> None:
    assert main(["wait", "--url", "http://evaluator", "--token-file", str(tmp_path / "missing"),
                 "--output", str(tmp_path / "status.json"), "--poll-seconds=" + value]) == 1
    assert capsys.readouterr().err == "arena client: poll interval must be positive and finite\n"
