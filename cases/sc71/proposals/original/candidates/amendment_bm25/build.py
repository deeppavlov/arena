"""Build an immutable local index or export only the document catalog."""
import sys

sys.dont_write_bytecode = True

from runtime import main

if __name__ == "__main__":
    main("build")
