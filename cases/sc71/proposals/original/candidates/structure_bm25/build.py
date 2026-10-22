"""Build this standalone candidate; the corpus location is explicit."""

import sys

sys.dont_write_bytecode = True

from candidate import cli

if __name__ == "__main__":
    cli("build")
