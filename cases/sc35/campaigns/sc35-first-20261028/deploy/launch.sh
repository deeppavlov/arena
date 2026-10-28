set -eu
ARENA_ROOT=${ARENA_ROOT:-/cephfs/home/malyshev_sa/arena-sc35-first-20261028}
export ARENA_ROOT
if [ ! -f "$ARENA_ROOT/operator/layout-version" ] || [ "$(cat "$ARENA_ROOT/operator/layout-version")" != "sc35-first-20261028-layout-v1" ]; then
  printf '%s\n' 'SC-35 sc35-first-20261028 deployment has not passed its own readiness checks.' >&2
  exit 1
fi
test -f "$ARENA_ROOT/operator/READY"
if [ -e "$ARENA_ROOT/operator/launch/active.json" ]; then
  printf '%s\n' 'A researcher run was already initialized; inspect its status before resuming' >&2
  exit 1
fi
cd "$ARENA_ROOT/release"
set -a
. "$ARENA_ROOT/operator/images.env"
set +a
docker run --rm --cap-add SYS_MODULE -v /lib/modules:/lib/modules:ro --entrypoint modprobe "$ARENA_INFRA_IMAGE" br_netfilter
test "$(cat /proc/sys/net/bridge/bridge-nf-call-iptables)" = 1
docker run --rm --network host --cap-add NET_ADMIN --entrypoint sh "$ARENA_INFRA_IMAGE" /opt/firewall.sh
docker compose --env-file "$ARENA_ROOT/operator/images.env" -f cases/sc35/campaigns/sc35-first-20261028/deploy/compose.yaml ps --format json > "$ARENA_ROOT/operator/services-at-launch.json"
python3 -c '
import json, os
from pathlib import Path
rows = [json.loads(line) for line in (Path(os.environ["ARENA_ROOT"]) / "operator/services-at-launch.json").read_text().splitlines() if line.strip()]
services = {row["Service"]: row for row in rows}
for name in ("train-evaluator", "public-evaluator", "private-evaluator", "model-gateway", "researcher-gateway", "internet-proxy", "research-monitor"):
    row = services.get(name, {})
    if row.get("State") != "running" or (name not in {"internet-proxy", "research-monitor"} and row.get("Health") != "healthy"):
        raise SystemExit("Required service is not healthy: " + name)
'
for service in train-evaluator public-evaluator private-evaluator; do
  docker compose --env-file "$ARENA_ROOT/operator/images.env" -f cases/sc35/campaigns/sc35-first-20261028/deploy/compose.yaml exec -T "$service" python -c '
import json, os, urllib.parse, urllib.request
from pathlib import Path
from evaluator.capacity import acquire_slot
directory = os.environ["ARENA_SLOT_DIR"]
if not directory:
    raise ValueError("ARENA_SLOT_DIR must be configured before search")
lease = acquire_slot(Path(directory), int(os.environ.get("ARENA_EXECUTION_SLOTS", "4")))
assert lease is not None
lease.close()
url = urllib.parse.urlsplit(os.environ["ARENA_RUNNER_MODEL_BASE_URL"])
health = url._replace(path="/healthz", query="", fragment="").geturl()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(health, timeout=5) as response:
    if json.load(response).get("status") != "ok":
        raise ValueError("Configured model gateway is not healthy")
'
done
docker compose --env-file "$ARENA_ROOT/operator/images.env" -f cases/sc35/campaigns/sc35-first-20261028/deploy/compose.yaml exec -T private-evaluator \
  python -m evaluator.client search-readiness --url http://127.0.0.1:8080 \
  --token-file /run/secrets/private-token --require-campaign \
  --max-solutions "$SC35_MAX_SOLUTIONS" --finalists "$SC35_FINALISTS" \
  --output-count "$SC35_OUTPUT_COUNT" --release-id "$SC35_PRIVATE_RELEASE_ID"
docker run --rm --network none -v "$ARENA_ROOT/operator/metric:/metric:ro" "$ARENA_EVALUATOR_IMAGE" python -m evaluator.prepare verify --metric /metric
docker compose --env-file "$ARENA_ROOT/operator/images.env" -f cases/sc35/campaigns/sc35-first-20261028/deploy/compose.yaml \
  run --rm --no-deps -T --entrypoint /opt/arena/.venv/bin/python researcher \
  /workspace/cases/sc35/campaigns/sc35-first-20261028/deploy/mlflow_probe.py read \
  --receipt /run/arena/mlflow-probe-receipt.json
docker run --rm --network none --user 10001:10001 \
  -v "$ARENA_ROOT/operator/outer-usage:$ARENA_ROOT/operator/outer-usage" \
  -v "$ARENA_ROOT/operator/launch:$ARENA_ROOT/operator/launch" \
  -v "$ARENA_ROOT/research-access:$ARENA_ROOT/research-access" \
  -e ARENA_ROOT "$ARENA_EVALUATOR_IMAGE" python -c '
import json, os
from pathlib import Path
from uuid import uuid4
from evaluator.usage import create_usage
root = Path(os.environ["ARENA_ROOT"])
active = root / "operator/launch/active.json"
if active.exists():
    raise SystemExit("A researcher run was already initialized; inspect its status before resuming")
run_id = str(uuid4())
capability = create_usage(root / "operator/outer-usage", run_id)
access = {
    "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "model_base_url": f"http://researcher-gateway:8081/runs/{run_id}/{capability}/v1",
    "model_timeout": 21600,
    "parameters": {"reasoning_effort": "max"},
}
(root / "research-access/model-access.json").write_text(json.dumps(access))
active.write_text(json.dumps({"run_id": run_id}))
'
docker compose --env-file "$ARENA_ROOT/operator/images.env" -f cases/sc35/campaigns/sc35-first-20261028/deploy/compose.yaml --profile search up -d researcher
ARENA_RESEARCHER_CONTAINER_ID=$(docker compose --env-file "$ARENA_ROOT/operator/images.env" -f cases/sc35/campaigns/sc35-first-20261028/deploy/compose.yaml ps -aq researcher)
ARENA_RESEARCHER_CONTAINER_ID=$(docker inspect --format '{{.Id}}' "$ARENA_RESEARCHER_CONTAINER_ID")
export ARENA_RESEARCHER_CONTAINER_ID
docker run --rm --network none --user 10001:10001 \
  -v "$ARENA_ROOT/operator/launch:$ARENA_ROOT/operator/launch" \
  -e ARENA_ROOT -e ARENA_RESEARCHER_CONTAINER_ID "$ARENA_EVALUATOR_IMAGE" python -c '
import json, os
from pathlib import Path
path = Path(os.environ["ARENA_ROOT"]) / "operator/launch/active.json"
value = json.loads(path.read_text())
value["container_id"] = os.environ["ARENA_RESEARCHER_CONTAINER_ID"]
path.write_text(json.dumps(value))
'
