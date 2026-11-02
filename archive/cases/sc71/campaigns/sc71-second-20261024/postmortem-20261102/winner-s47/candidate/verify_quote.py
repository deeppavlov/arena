import json
import re
import sys
from pathlib import Path

ORIGIN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
KB = Path("/assets/kb/norm")


def normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def main() -> None:
    payload = json.load(sys.stdin)
    arguments = payload.get("arguments") or {}
    origin = str(arguments.get("origin_id", ""))
    quote = arguments.get("quote")
    if not isinstance(quote, str) or not quote.strip():
        print(json.dumps({"found": False, "error": "quote must be a non-empty string",
                          "normalized_chars": 0, "origin_id": origin}, ensure_ascii=False))
        return
    condensed = normalized(quote)
    path = KB / f"{origin}.md"
    if not ORIGIN.fullmatch(origin) or not path.is_file():
        print(json.dumps({"found": False, "error": "document not found or illegal origin_id",
                          "normalized_chars": len(condensed), "origin_id": origin}, ensure_ascii=False))
        return
    haystack = normalized(path.read_text(encoding="utf-8"))
    print(json.dumps({"found": condensed in haystack, "origin_id": origin,
                      "normalized_chars": len(condensed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
