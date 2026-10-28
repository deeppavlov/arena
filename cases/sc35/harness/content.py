import os
import sys
from pathlib import Path


def main() -> int:
    entrypoint = Path("/candidate/solve.py")
    if not entrypoint.is_file():
        print("SC35 candidate solve.py is missing", file=sys.stderr)
        return 1
    os.chdir("/tmp")
    os.execv(sys.executable, [sys.executable, "-E", "-s", "-B", str(entrypoint)])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
