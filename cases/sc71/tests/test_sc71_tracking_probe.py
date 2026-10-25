import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

PROBE = (
    Path(__file__).parents[1] / "campaigns" / "sc71-second-20261024" / "deploy" / "mlflow_probe.py"
)


@pytest.mark.parametrize("payload_available", [True, False])
def test_probe_requires_trace_payload_from_researcher_client(tmp_path, payload_available):
    probe = SimpleNamespace(**runpy.run_path(str(PROBE)))
    receipt = {
        "run_id": "probe-run", "campaign_id": "sc71-second-20261024",
        "uri": "http://shared-tracking:5000", "marker": "synthetic-only",
    }
    expected = {"message": receipt["marker"]}
    span = SimpleNamespace(
        name="synthetic.echo", inputs=expected, outputs=expected if payload_available else None,
    )
    trace = SimpleNamespace(info=SimpleNamespace(trace_id="probe-trace"),
                            data=SimpleNamespace(spans=[span]))
    client = Mock()
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(experiment_id="probe-experiment", status="FINISHED"),
        data=SimpleNamespace(tags={"campaign_id": receipt["campaign_id"], "split": "train"}),
    )
    client.search_traces.return_value = [trace]
    call_path = tmp_path / "call.json"
    call_path.write_text(json.dumps({"name": "synthetic.echo", "inputs": expected,
                                     "outputs": expected}))
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps({
        "synthetic_only": True, "params": {"campaign_id": receipt["campaign_id"]},
        "call_artifacts": ["calls/saved-snapshot.json"],
    }))
    client.download_artifacts.side_effect = lambda run_id, path: (
        str(record_path) if path == "record.json" else str(call_path)
    )
    if payload_available:
        result = probe.verify_payload(client, receipt)
        assert result["trace_ids"] == ["probe-trace"]
        assert result["payload_verified"] is True
        assert client.search_traces.call_args.kwargs["include_spans"] is True
        client.download_artifacts.assert_any_call("probe-run", "calls/saved-snapshot.json")
    else:
        with pytest.raises(ValueError, match="payload"):
            probe.verify_payload(client, receipt)
