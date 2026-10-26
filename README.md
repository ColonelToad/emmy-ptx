# Kernel IR → PTX

A direct PTX compiler for [Emmy](https://github.com/cloudrift-ai/emmy)'s Kernel IR.
No CUDA C++, no `nvcc`, no renderer: the IR is lowered straight to PTX text.

```bash
python compile.py path/to/kernel.ir.json > out.ptx
```

**Status: 15/15 open problems pass** across all five categories — pointwise,
reductions, matmul, fused, attention-like — including the two matmul pipelines
that need `cp.async` staging and atomics.

The unusual part is in the next section: the IR is implemented **twice**, once
as a PTX backend and once as an executable numpy specification, and the two are
checked against each other.

---

## Why this submission is different

Most submissions are a translator: walk the IR, print instructions. That is
where this one starts. On top of it:

1. **An executable specification of the IR** (`ptxgen/interp.py`). A second,
   independent implementation of the same language, written in numpy, one
   numpy lane per GPU thread. It runs the kernel on the CPU and needs no GPU,
   torch, or emmy renderer.

   It reproduces **all 15 reference outputs** in `problems/*/expected.npz`,
   bit-exact where the tolerances are zero:

   ```bash
   python scripts/verify_semantics.py     # 15/15 in ~10 s, no GPU
   ```

   This is a much stronger statement than "my compiler passed 15 tests": it
   means the language semantics are pinned down by a second implementation, and
   that implementation is a golden model for kernels the open set does not
   contain. It is also the basis for differential fuzzing: compile a random
   kernel, run it on the GPU, diff against the model.

2. **Alignment as a proof, not a heuristic.** Vector loads/stores and
   `cp.async` widths are only used when a small divisibility analysis of the
   index expression proves the address is aligned. Everything else falls back
   to a scalar path, so a surprising shape cannot produce a misaligned access.

3. **A lowering that respects accumulator identity.** `Accum` is a mutable,
   loop-carried register (`name = op(base or name, value)`), and reductions
   (`WarpShuffle`, `TreeHalve`) write the reduced value back into the original
   registers. This is what makes multi-stage softmax and split-K matmul work
   without special-casing them.

4. **No special cases for the open set.** There is no pattern table of problem
   names, no hardcoded grids, and no per-problem PTX. `meta.json` supplies the
   launch and argument order; everything else comes from the IR.

---

## Quickstart

Python 3.12+.

```bash
# with uv (what this repo was developed with)
uv venv .venv && . .venv/bin/activate
uv pip install -r requirements.txt

# or plain pip
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`emmy-ml` is the only unusual dependency. It is needed to *deserialize* the IR;
no torch, CUDA toolkit, or GPU is required locally.

Generate PTX:

```bash
python compile.py problems/p01_vector_add/k_add_pointwise.ir.json > /tmp/add.ptx
head -20 /tmp/add.ptx
```

Assemble and run on the hosted RTX 5090 (no local GPU needed):

```bash
python submit.py p01_vector_add --compiler compile.py     # one problem
python scripts/check_all.py                               # all 15
python scripts/verify_semantics.py                        # CPU-only semantics check
```

`submit.py` is the organizers' harness; `scripts/check_all.py` is a thin wrapper
that runs it over every problem.

---

## Results

`python scripts/check_all.py`, one launch per output:

| category | problem | max abs err | kernel time |
|---|---|---:|---:|
| pointwise | p00_scalar_add | 0.000e+00 | 20 µs |
| pointwise | p01_vector_add | 0.000e+00 | 20 µs |
| pointwise | p02_scale_shift | 0.000e+00 | 14 µs |
| pointwise | p03_relu_mul | 0.000e+00 | 14 µs |
| reduction | r01_array_sum | 3.815e-06 | 15 µs |
| reduction | r02_big_sum | 1.526e-05 | 36 µs |
| reduction | r03_mean | 9.313e-10 | 28 µs |
| reduction | r04_row_sum | 5.722e-06 | 20 µs |
| matmul | m01_matmul_512 | 8.392e-05 | 111 µs |
| matmul | m02_matmul_rect | 1.335e-05 | 22 µs |
| fused | f01_rmsnorm | 1.907e-06 | 17 µs |
| fused | f02_row_softmax | 7.451e-09 | 15 µs |
| fused | f03_l2norm | 1.490e-08 | 34 µs |
| attention | a01_attention | 6.557e-07 | 108 µs |
| attention | a02_attention_128 | 4.470e-07 | 62 µs |

Kernel times are the runner's median over one launch and are dominated by
launch overhead; they are indicative, not a benchmark.

Semantics model (`scripts/verify_semantics.py`, CPU only): **15/15**, ~10 s.

---

## Architecture

```
kernel.ir.json ──► loader ──► KernelSpec ──► emit ──► PTX text
                    │           (meta.json: launch, arg order, dtypes)
                    │
                    └────────► interp ──► numpy result   (golden model)

                       compare against expected.npz / GPU output
```

```
compile.py            argv contract; stdout is PTX only, diagnostics on stderr
ptxgen/loader.py      Graph.from_dict + sibling meta.json
ptxgen/model.py       arguments, buffers, shared-memory layout
ptxgen/ops.py         op aliases, reduction identities
ptxgen/emit.py        full IR -> PTX lowering
ptxgen/interp.py      independent numpy semantics (the golden model)
scripts/check_all.py  run the official harness over every problem
scripts/verify_semantics.py
                      CPU-only: interpreter vs expected.npz
problems/             the 15 open problems
docs/HACKATHON.md     the original task statement and rules
TASK.md, MVP.md, submit.py, docs/API.md
                      organizers' material, kept for reproducibility
```

### How a kernel is lowered

1. **Load.** `Graph.from_dict` deserializes the IR. The sibling `meta.json` is
   authoritative for `arg_order`, shapes, dtypes, grid, block, `smem_bytes`,
   and `zero_outputs` — the IR's argument order is not always the call order.
2. **Layout.** Global parameters become `.u64` pointers. Shared buffers are laid
   out with alignment padding; small kernels get static `.shared`, larger ones a
   single `.extern .shared` pool fed by `meta.smem_bytes`.
3. **Prologue.** `threadIdx`, `blockIdx`, `blockDim`, warp/lane split, and the
   tile guard `gid = cta * blockDim + tid; if (gid >= N) return-path`.
4. **Virtual registers.** `%f`, `%fd`, `%h`, `%r`, `%rd`, `%p` are allocated on
   demand. `ptxas` does the real register allocation; declarations are emitted
   once, at the end, with exact counts.
5. **Statements.** `Tile`, `Load`, `Assign`, `Accum`, `Let`, `Write`, `Loop`,
   `StridedLoop`, `Cond`, `Init`, `Smem`, `Sync`, `WarpShuffle`, `TreeHalve`,
   `CpAsyncCopy/Commit/Wait`.
6. **Ops.** Arithmetic, comparisons, `selp`, casts, `ex2.approx`/`lg2.approx`
   for exp/log, `tanh.approx`, `sin/cos`, `sqrt/rsqrt/rcp`, rounding, and
   composed sigmoid/silu/gelu/softplus/erf. Float literals are emitted as exact
   bit patterns (`0fXXXXXXXX`), never as decimal text.

Shared memory layout, the value model, and the alignment analysis are described
in [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Scope

The evaluation set uses only the unmarked IR nodes (see `docs/HACKATHON.md`).
Within that vocabulary the compiler aims to be complete, including f16/i64
data, named barriers, and dynamic shared memory.

Not implemented, because the task states they are not evaluated: tensor-core
(`mma`/WGMMA), TMA and mbarrier nodes, `Pre`/`Carry`, and fp8. They raise a clear
`NotImplementedError` instead of silently producing wrong PTX.

Known simplifications:

- f64 transcendental functions are computed in f32 and widened.
- `Tile` rasterization and `Loop.unroll` are hints; ignoring them affects
  locality and code size, not results.
- 1-D blocks only (`blockDim.y/z == 1`), matching every problem in the set.

---

## Credits

The task, problem set, runner, and `submit.py` are from the
[Emmy hackathon repository](https://github.com/cloudrift-ai/emmy-hackathon-2026).
The compiler in `compile.py`, `ptxgen/`, and `scripts/` is this submission.
