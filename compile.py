#!/usr/bin/env python3
"""Emmy Hackathon 2026 submission entry point.

Usage:
    python compile.py path/to/kernel.ir.json > out.ptx

Only PTX is written to stdout. Diagnostics go to stderr.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: compile.py path/to/kernel.ir.json", file=sys.stderr)
        return 2

    path = Path(argv[1])
    if not path.is_file():
        print(f"compile.py: no such file: {path}", file=sys.stderr)
        return 2

    try:
        from ptxgen import compile_file

        ptx = compile_file(path)
    except Exception as exc:  # noqa: BLE001 - the harness only needs a clean failure
        if os.environ.get("PTXGEN_DEBUG"):
            traceback.print_exc()
        else:
            print(f"compile.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not ptx.endswith("\n"):
        ptx += "\n"
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
