import json
import sys

payload = json.load(sys.stdin)
reference = payload["example"].get("reference")
if not isinstance(reference, str):
    raise ValueError("exact-match metric requires a string reference")
print(json.dumps({"score": int(payload["answer"] == reference)}))
