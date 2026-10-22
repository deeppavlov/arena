"""Build the unchanged lexical baseline inside this standalone package."""

import sys

sys.dont_write_bytecode = True

from candidate import cli

if __name__ == "__main__":
    cli("build")
