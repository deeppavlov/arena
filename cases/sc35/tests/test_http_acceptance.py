import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "acceptance/http_check.py"
spec = importlib.util.spec_from_file_location("sc35_http_check", PATH)
assert spec and spec.loader
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.mark.parametrize("terminal", ["FINISHED", "FAILED"])
def test_wait_preserves_both_terminal_states(terminal):
    responses = iter([{"status": "RUNNING"}, {"status": terminal, "marker": "retained"}])
    sleeps = []
    result = check.wait_for_result(lambda: next(responses), sleep=sleeps.append)
    assert result == {"status": terminal, "marker": "retained"}
    assert sleeps == [1]


def test_unknown_status_fails_instead_of_polling():
    with pytest.raises(ValueError, match="Unknown train status"):
        check.wait_for_result(lambda: {"status": "complete"})


def test_wait_uses_requested_deadline():
    clock = iter([0, 0, 2])
    with pytest.raises(TimeoutError):
        check.wait_for_result(
            lambda: {"status": "RUNNING"}, timeout=1,
            sleep=lambda _: None, clock=lambda: next(clock),
        )


def test_mock_transport_keeps_the_fixed_hermes_model():
    from cases.sc71.harness.hermes import runtime
    from cases.sc71.harness.hermes.main import validate_task

    assert check.MODEL == runtime.MODEL
    assert validate_task({
        "input": "synthetic transport fixture", "model": check.MODEL,
        "model_base_url": "http://sc35-http-gateway:8081/runs/synthetic/fixture/v1",
        "model_request_timeout": runtime.REQUEST_TIMEOUT, "prepare_timeout": 172800,
    }) == {}
