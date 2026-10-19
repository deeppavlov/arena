import json
import math
import sys
import urllib.request
from pathlib import Path
from statistics import mean

payload = json.load(sys.stdin)
policy = json.loads(Path(__file__).with_name("policy.json").read_text(encoding="utf-8"))
vote_count = policy["votes"]
if type(vote_count) is not int or vote_count <= 0:
    raise ValueError("votes must be a positive integer")

votes = []
if payload["answer"].strip():
    body = json.dumps(
        {
            "model": payload["model"],
            "messages": [
                {"role": "system", "content": policy["instruction"]},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"example": payload["example"], "answer": payload["answer"]},
                        ensure_ascii=False,
                    ),
                },
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    for _ in range(vote_count):
        request = urllib.request.Request(
            payload["model_base_url"].rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=payload["model_request_timeout"]) as response:
            content = json.load(response)["choices"][0]["message"]["content"]
        score = json.loads(content)["score"]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("judge score must be a finite number between 0 and 1")
        votes.append(score)

print(json.dumps({"score": mean(votes) if votes else 0, "statistics": {"votes": votes}}))
