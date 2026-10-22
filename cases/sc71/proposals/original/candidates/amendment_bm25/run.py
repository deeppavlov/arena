"""Run the independent local candidate; stdout is separate from diagnostics."""
import sys

sys.dont_write_bytecode = True

from runtime import main

if __name__ == "__main__":
    main("run")
