# Emmy Hackathon 2026 — working notes

## Task

Write a compiler that turns Emmy's **Kernel IR** into **PTX**:

```bash
python compile.py path/to/kernel.ir.json > out.ptx
```

`ptxas` then assembles the PTX and the hosted runner executes it on an RTX 5090
(`sm_120`). PTX is written for `sm_89`; the runner accepts it unchanged.

Hard rules (see `TASK.md`):

- Entry point name must equal the IR's `KernelOp.name`.
- Parameters must follow `meta.json`'s `arg_order` (buffers are `.u64` pointers).
- Must be correct under the launch in `meta.json` (grid / block / smem / zero_outputs).
- The IR → PTX lowering must be ours. Loading with `Graph.from_dict` is allowed;
  shelling out to Emmy's renderer, `emmy compile`, or `nvcc` is not.
- PTX must be ASCII only and start with `.version` / `.target` / `.address_size`.

Judging: correctness by category on a hidden set, then code review, then speed.
MVP = categories 1 (pointwise) and 2 (reductions).

## Environment (WSL, no GPU needed)

A local virtual environment lives at `.venv/` (Python 3.12.14, created with `uv`):

```bash
# create it again if needed
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python emmy-ml==0.3.11 numpy requests

# run anything with it
.venv/bin/python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json
.venv/bin/python submit.py p00_scalar_add --compiler compile.py

# or activate it for a shell
source .venv/bin/activate
```

Never use the system Python: the IR only deserializes through `emmy-ml==0.3.11`
on Python 3.12+. The venv is git-ignored.

Runner health check:

```bash
curl -s http://runner.riftstack.ai:48701/health
```

## Problem shape

15 open problems, 5 categories:

| # | category | problems |
|---|---|---|
| 1 | pointwise | `p00`–`p03` |
| 2 | reductions | `r01`–`r04` |
| 3 | matmul | `m01`, `m02` |
| 4 | fused | `f01`–`f03` |
| 5 | attention-like | `a01`, `a02` |

Each problem folder has the IR (`.ir.json`), a human-readable dump (`.ir.txt`),
Emmy's CUDA reference (`.ref.cu`), expected outputs, and `meta.json`.

Observed IR vocabulary across the open set:

```
Tile, Load, Assign, Accum, Let, Write, Loop, StridedLoop, Cond, Init,
Smem, Sync, WarpShuffle, TreeHalve, CpAsyncCopy/Commit/Wait
```

Expressions: `Var`, `Literal`, `BinaryExpr` (with `Builtin`, `FuncCallExpr`,
`TernaryExpr`, `CastExpr`, `FlatIndex` supported defensively). No `Select`,
`Carry`, `Pre`, `ZeroPrologue`, TMA, or tensor-core nodes in any problem.

Key facts:

- `Tile` maps a flat `blockIdx.x * blockDim.x + threadIdx.x` to `N = product(axis extents)`,
  row-major. `p00` has `N=64` with 256 threads, so the bounds guard is required.
  Reductions have `N == blockDim`; matmul has exact coverage.
- `Accum` starts at the op identity (`0` for add, `-1e30` for max) before its loop;
  nested accumulators are re-seeded on each outer iteration. It is a **mutable
  accumulator**: `name = op(base or name, value)`, and the result is written back
  into `name`'s own register. `base` is only the left operand (softmax uses it).
- `WarpShuffle` is a `shfl.sync.bfly.b32` ladder; `TreeHalve` is a shared-memory
  power-of-two tree with `bar.sync` and a broadcast from slot 0. Both support
  coupled states (softmax max + sum) and restore the reduced state into the
  original registers so they remain correct if nested in a runtime loop.
- Loads/writes may be vector groups (`ld/st.global.v2/v4.f32`); only f32 is
  vectorized, and atomics always go through the scalar path.
- `m01` needs 73,728 B of **dynamic** shared memory and a `cp.async` pipeline;
  `m02` uses static shared memory, split-K tiles, and `red.global.add.f32`.
- Tile rasterization only permutes CTAs — safe to ignore (affects L2 locality only).
- `Loop.unroll` is a hint — loops can be emitted normally (smaller PTX).
- `meta.json` argument order differs from `inputs + outputs` in several problems,
  so it is the source of truth for names, shapes, and dtypes.

## Compiler design

```
compile.py             # argv contract, stdout = PTX only
ptxgen/loader.py       # Graph.from_dict + sibling meta.json
ptxgen/model.py        # buffers, shared-memory layout, arguments
ptxgen/ops.py          # elementwise op metadata (identities, canonical names)
ptxgen/emit.py         # expression + statement lowering, PTX assembly
ptxgen/interp.py       # independent numpy semantics (golden model)
scripts/check_all.py   # run submit.py over every problem
scripts/verify_semantics.py  # CPU-only interpreter vs expected.npz
requirements.txt
```

Pipeline:

1. **Load** the IR with `Graph.from_dict`, find the `KernelOp`, match it to the
   sibling `meta.json` kernel by entry name. Use `arg_order`, arg shapes/dtypes,
   `grid/block/smem_bytes/zero_outputs`. Fall back to IR-derived info if missing.
2. **Buffer map**: `name -> {space, base, shape, dtype, elem_size}`. Global
   params are `.param .u64` + `ld.param` + `cvta.to.global`. Shared buffers are
   static `.shared` declarations, or one `.extern .shared` pool with aligned
   offsets when `smem_bytes > 0` / total > 48 KiB.
3. **Prologue**: `threadIdx.x`, `blockIdx.x`, `blockDim.x`, `lane = tid & 31`,
   `warp = tid >> 5`, Tile axis decode, and the `_gid < N` bounds guard.
4. **Registers**: allocate virtual registers on demand (`%f`, `%r`, `%rd`, `%p`,
   `%fd`, `%h`), emit exact counts at the top after codegen. ptxas allocates.
5. **Addresses**: row-major flatten; `mul.wide.s32` + `add.s64` for global,
   32-bit shared offsets. Vector loads/stores only when alignment is statically
   provable; scalar fallback otherwise.
6. **Statements**: labels/branches for `Cond`/`Loop`/`StridedLoop`; accumulator
   seeding; `activemask.b32` + butterfly for `WarpShuffle`; the tree loop,
   barriers, and broadcast for `TreeHalve`; `cp.async` with lane offsets and
   optional XOR swizzle; static/dynamic `Smem`; `bar.sync` / `bar.warp.sync`.
7. **Ops**: add/sub/mul/div, fma, rcp, sqrt/rsqrt, max/min, abs/neg, relu,
   comparisons, `selp`, casts, `ex2.approx` for exp, `lg2` for log,
   `tanh.approx`, `sin`/`cos`, floor/ceil/trunc/round, and composed
   sigmoid/silu/gelu/gelu_tanh/softplus/erf/expm1/log1p/sinh/cosh, plus
   clamp/clip, pow, fmod, floor_divide, sign, isnan/isinf, logical ops.
   Float literals are emitted as exact `0fXXXXXXXX` bit patterns.

## Hidden-set hardening

`TASK.md` guarantees the hidden set uses only unmarked IR nodes: no tensor cores,
TMA, `Pre`, `Carry`, `Select`, or `ZeroPrologue`. All unmarked nodes are handled:
`Tile`, `Load`, `Assign`, `Accum`, `Let`, `Write`, `Loop`, `StridedLoop`, `Cond`,
`Init`, `Smem`, `Sync`, `WarpShuffle`, `TreeHalve`, `CpAsyncCopy/Commit/Wait`,
`Var`, `Literal`, `BinaryExpr`, `TernaryExpr`, and `Builtin`/`FuncCallExpr`/
`CastExpr`/`FlatIndex` defensively. `Select` and `ZeroPrologue` are implemented
anyway (the latter writes raw 32-bit words with a 4-byte stride).

Hardening beyond the open set:

- f16 loads/stores, `Accum` in f16, f16 `WarpShuffle` (packed through `cvt.u32.u16`)
  and f16 `TreeHalve` (shared memory carries the buffer dtype).
- `sync` with named barriers and no explicit count; `bar.warp.sync`; `warpSize`.
- `cp.async` 4/8/16 B, lane-index decomposition, XOR swizzles, dynamic shared memory.
- Reduction identities for add/mul/max/min, recursive accumulator seeding through
  `Cond`/`Tile` (nested loops seed at their own entry).
- Integer index expressions: `//`, `%`, `&`, `|`, `^`, `<<`, `>>`, comparisons.
- Predicate conversions (`isnan`/`isinf`/logical ops) when a float/int value is needed.
- Atomic writes never use vector stores. Vectorized global access requires f32,
  width 2/4, and statically provable alignment.

Known limitations (outside the guaranteed hidden vocabulary): tensor-core,
TMA/WGMMA, mbarrier, fragment, `Pre`/`Carry`, and fp8 nodes are not lowered;
f64 transcendental ops fall back to f32 precision.

## Build order / status

**Current result: 15/15 open problems pass** via
`.venv/bin/python scripts/check_all.py` (all five categories), and the
independent numpy semantics reproduces all 15 reference outputs via
`.venv/bin/python scripts/verify_semantics.py` (CPU only, ~10 s).

- [x] Explore repo, all problems, Emmy IR classes, runner.
- [x] Create `.venv` and install `emmy-ml==0.3.11`, numpy, requests.
- [x] Verify runner + worked example (`r01`, max err 3.8e-6).
- [x] Validate PTX syntax probes (`cp.async`, `shfl.sync`, `ex2`, `red.global`,
      vector ld/st, shared/global addressing, f16, named barriers, immediates).
- [x] Compiler skeleton + loader + prologue.
- [x] Category 1: `p00`–`p03` (all pass).
- [x] Category 2: `r01`–`r04` (all pass, MVP bar cleared).
- [x] Category 4: `f01`–`f03` (all pass).
- [x] Category 5: `a01`–`a02` (all pass).
- [x] Category 3: `m01`, `m02` (all pass).
- [x] Hidden-set hardening (see below).
- [x] Independent numpy semantics (`ptxgen/interp.py`): 15/15 expected outputs.
- [ ] Differential fuzzer over the unmarked vocabulary.
- [ ] Middle-end (divisibility analysis, canonicalisation, PTX peephole).
- [ ] Demo docs and video script.

## Executable semantics

`ptxgen/interp.py` is a second implementation of the IR, written in numpy,
used as a golden model. It executes a chunk of blocks at a time and represents
a thread as a numpy lane.

- Register file: `dict[str, V(kind, ndarray)]`; every coercion mirrors the PTX
  `cvt` rules (truncating integer casts, f16 round-to-nearest).
- Divergence: `Cond`, loop bounds and the tile guard are boolean masks; every
  register write merges with the mask so lanes that branched around a statement
  keep their old value.
- `Sync` and `cp.async` are no-ops/instant because execution is lockstep; that
  is correct for race-free IR.
- `WarpShuffle` is a lane-index XOR gather within each block; `TreeHalve`
  emulates the shared-memory tree step by step.
- Inactive lanes are clamped and their writes dropped, mirroring the PTX branch;
  an out-of-bounds access on an active lane raises.

Validation: `scripts/verify_semantics.py` rebuilds the same inputs the runner
generates (`seed` → `default_rng(seed).standard_normal`), chains intermediate
buffers between kernels, applies `zero_outputs`, and compares with
`expected.npz` under each problem's tolerance. All 15 match; the zero-tolerance
pointwise problems are bit-exact.

Performance: ~10 s for the whole set (m01 dominates). Blocks are vectorised in
chunks bounded by 256 MiB of shared memory and 2^20 lanes.

## Testing

```bash
# one problem
.venv/bin/python submit.py r01_array_sum --compiler compile.py

# everything (GPU, via the hosted runner)
.venv/bin/python scripts/check_all.py

# CPU-only semantics model vs expected.npz
.venv/bin/python scripts/verify_semantics.py

# inspect generated PTX
.venv/bin/python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json
```

## Gotchas

- Barrier divergence: never wrap a `bar.sync` in a thread guard that can split a
  block. Emmy's own schedules only use barriers when coverage is exact.
- Vector alignment: `ld/st.global.v4.f32` requires 16-byte alignment, `v2`
  requires 8. Fall back to scalar if it cannot be proven.
- Dynamic shared memory: declare `.extern .shared .align 16 .b8 pool[]` and
  address it through `mov.u32`; the runner passes `meta.smem_bytes`.
- `shfl.sync` needs a member mask — emit `activemask.b32` at the shuffle site.
- Float literals must be exact (`0f...`), not decimal approximations.
- PTX comments must be ASCII.
- ptxas rejects f16/f64 registers used with the wrong type; keep separate banks.
