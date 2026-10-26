#!/usr/bin/env python3
"""Run the submission harness over every open problem and print a summary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "problems"


def main(argv: list[str]) -> int:
    only = set(argv[1:])
    names = sorted(p.name for p in PROBLEMS.iterdir() if p.is_dir())
    if only:
        names = [n for n in names if n in only or any(n.startswith(o) for o in only)]

    rows = []
    failures = 0
    for name in names:
        cmd = [sys.executable, str(ROOT / "submit.py"), name, "--compiler", "compile.py"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0 and "PASS" in out.upper()
        if not ok:
            failures += 1
        rows.append((name, ok, out.strip().splitlines()[-1] if out.strip() else "(no output)"))

    width = max(len(n) for n, _, _ in rows) if rows else 8
    for name, ok, tail in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {tail}")
    print(f"\n{len(rows) - failures}/{len(rows)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
