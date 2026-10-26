#!/usr/bin/env python3
"""Check the numpy reference semantics against every open problem's expected outputs.

    python scripts/verify_semantics.py            # all problems
    python scripts/verify_semantics.py r01 f02    # substring filter
    python scripts/verify_semantics.py -v         # per-kernel details

No GPU and no runner are involved: this exercises ``ptxgen.interp`` only.
If this passes, the interpreter is a trustworthy golden model for the
differential tests in ``scripts/fuzz.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ptxgen.interp import interpret  # noqa: E402
from ptxgen.loader import load_kernel  # noqa: E402

NP = {"f32": np.float32, "f64": np.float64, "f16": np.float16,
      "i32": np.int32, "i64": np.int64, "u32": np.uint32, "u64": np.uint64}


def build_inputs(kernel: dict, produced: dict[str, np.ndarray],
                 rng_cache: dict[int, np.random.Generator]) -> dict[str, np.ndarray]:
    """Mirror submit.py's argument construction, but in numpy."""
    inputs: dict[str, np.ndarray] = {}
    for a in kernel["args"]:
        name = a["name"]
        dtype = np.dtype(NP[a["dtype"]])
        shape = tuple(int(x) for x in a["shape"])
        if a.get("intermediate"):
            inputs[name] = np.asarray(produced[name], dtype=dtype).reshape(shape)
        elif a.get("output"):
            inputs[name] = np.zeros(shape, dtype=dtype)
        elif a.get("scalar"):
            continue
        elif "data" in a:
            inputs[name] = np.asarray(a["data"], dtype=dtype).reshape(shape)
        else:
            seed = int(a.get("seed", 0))
            rng = rng_cache.setdefault(seed, np.random.default_rng(seed))
            inputs[name] = rng.standard_normal(shape).astype(dtype)
    for name in kernel.get("zero_outputs", []):
        if name in inputs:
            inputs[name][...] = 0
    return inputs


def max_abs_err(got: np.ndarray, ref: np.ndarray) -> float:
    if got.size == 0:
        return 0.0
    return float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))


def check_problem(pdir: Path, verbose: bool) -> tuple[bool, list[str]]:
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz"))
    produced: dict[str, np.ndarray] = {}
    rng_cache: dict[int, np.random.Generator] = {}
    notes: list[str] = []
    ok = True

    for kernel in meta["kernels"]:
        ir_path = pdir / f"{kernel['entry']}.ir.json"
        spec = load_kernel(ir_path)
        inputs = build_inputs(kernel, produced, rng_cache)
        t0 = time.perf_counter()
        outputs = interpret(spec, inputs)
        dt = time.perf_counter() - t0
        produced.update(outputs)

        if verbose:
            notes.append(f"  {kernel['entry']}: {dt * 1e3:.1f} ms  "
                         f"outputs={sorted(outputs)}")
        for name, out in outputs.items():
            if name not in expected:
                continue
            ref = expected[name]
            got = np.asarray(out, dtype=np.float32).reshape(ref.shape)
            err = max_abs_err(got, ref)
            good = np.allclose(got, ref, rtol=meta["rtol"], atol=meta["atol"],
                               equal_nan=True)
            notes.append(f"  {kernel['entry']}  {name}: {'PASS' if good else 'FAIL'}  "
                         f"max abs err {err:.3e}  ({dt * 1e3:.1f} ms)")
            ok = ok and good

    return ok, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("filters", nargs="*", help="problem name substrings")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    problems = sorted(p for p in (ROOT / "problems").iterdir() if (p / "meta.json").exists())
    if args.filters:
        problems = [p for p in problems if any(f in p.name for f in args.filters)]
    if not problems:
        print("no matching problems", file=sys.stderr)
        return 2

    failed = 0
    t0 = time.perf_counter()
    for pdir in problems:
        ok, notes = check_problem(pdir, args.verbose)
        print(f"{'PASS' if ok else 'FAIL'}  {pdir.name}")
        if args.verbose or not ok:
            for note in notes:
                print(note)
        failed += 0 if ok else 1
    dt = time.perf_counter() - t0
    print(f"\n{len(problems) - failed}/{len(problems)} problems match expected.npz "
          f"({dt:.1f} s)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
