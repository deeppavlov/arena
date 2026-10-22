"""Standalone arena entry point."""

from pathlib import Path

from sc71 import cli

if __name__ == "__main__":
    cli("build", Path(__file__).resolve().parent)
