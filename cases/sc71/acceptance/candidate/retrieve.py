import json
import sys
from pathlib import Path

task = json.load(sys.stdin)
origin = "a" * 64
path = Path(task["assets_dir"]) / "norm" / f"{origin}.md"
print(json.dumps({"results": [{
    "rank": 1,
    "origin_id": origin,
    "path": f"/kb/norm/{origin}.md",
    "text": path.read_text(encoding="utf-8"),
}]}, ensure_ascii=False))
