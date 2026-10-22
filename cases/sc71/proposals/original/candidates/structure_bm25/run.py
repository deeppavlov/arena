"""Read an immutable index and return retrieved evidence or a grounded answer."""

import sys

sys.dont_write_bytecode = True

from candidate import cli

if __name__ == "__main__":
    cli("run")
