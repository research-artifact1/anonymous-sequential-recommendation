"""Convenience entry point for ``trimrec train``."""

from __future__ import annotations

import sys

from trimrec.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
