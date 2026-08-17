"""Convenience entry point for ``trimrec evaluate``."""

from __future__ import annotations

import sys

from trimrec.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
