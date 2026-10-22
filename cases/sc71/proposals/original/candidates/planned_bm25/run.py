"""Generate one bounded plan and search without writing to the index."""

import sys

sys.dont_write_bytecode = True

from candidate import cli

if __name__ == "__main__":
    cli("run")
