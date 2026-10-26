# Design notes

This is the reasoning behind the two implementations in `ptxgen/`: the PTX
backend (`emit.py`) and the executable semantics (`interp.py`).

## 1. The contract

The submission is a program, not a set of PTX files:

```bash
python compile.py path/to/kernel.ir.json > out.ptx
```

Constraints that shaped the design:

- stdout is PTX only; diagnostics go to stderr.
- the entry name must be the IR `KernelOp.name`;
- parameters follow `meta.json`'s `arg_order`, which is not always the IR's
  `inputs + outputs` order (`m01`, `m02`, `f01`, `a01`, `a02`, `p02`);
- grid, block, `smem_bytes`, and `zero_outputs` come from `meta.json`;
- PTX must be ASCII and start with `.version`, `.target`, `.address_size`;
- lowering must be original — deserialization through `Graph.from_dict` is the
  only Emmy code used.

The backend targets `.target sm_89` / `.version 8.0`. The evaluation GPU is an
RTX 5090 (`sm_120`), which runs `sm_89` PTX unchanged.

## 2. Two implementations of one language

A compiler is only as trustworthy as its tests. The open set has 15 data points;
the hidden set has different shapes and ops. To get confidence beyond 15
examples, the IR is implemented twice:

| | `ptxgen/emit.py` | `ptxgen/interp.py` |
|---|---|---|
| output | PTX | numpy arrays |
| execution | one GPU thread | one numpy lane per thread |
| used for | submission | golden model, differential testing |
| checked by | `scripts/check_all.py` (GPU) | `scripts/verify_semantics.py` (CPU) |

The interpreter is written from the IR and the task description, not by
translating the backend, and it reproduces **all 15 reference outputs** —
including bit-exact results for the zero-tolerance pointwise problems. That is
the strongest available evidence that the semantics (not just the 15 outputs)
are right, and it gives one definition to diff against when a random kernel
disagrees with the GPU.

## 3. Value model

Every expression evaluates to a `Val(kind, text, imm)` where `kind` is one of
`f32`, `f64`, `f16`, `s32`, `s64`, `pred`. The kind decides the PTX type and
the coercion rules:

- arithmetic promotes `f64 > f32 > s32`; `f16` is widened to `f32` for
  arithmetic and rounded back on assignment (`cvt.rn.f16.f32`);
- casts to integer truncate toward zero (`cvt.rzi`), matching PTX;
- predicate-to-number conversions become `selp` with `1`/`0`;
- literals carry exact bit patterns (`0f3F800000`, `0d3FF0000000000000`) so a
  decimal printer can never change a constant.

Keeping kinds on values rather than recomputing them from the IR is what makes
mixed-precision cases (f16 storage with f32 arithmetic) unambiguous.

## 4. Registers and declarations

Virtual registers are allocated as `%f<n>`, `%fd<n>`, `%h<n>`, `%r<n>`,
`%rd<n>`, `%p<n>` while the body is emitted. Declarations are written at the
end with the exact counts:

```ptx
.reg .pred %p<3>;
.reg .f32 %f<41>;
.reg .b32 %r<12>;
```

Nothing is reused or scheduled by hand. `ptxas` is good at register allocation;
the front end's job is to give it straight-line, side-effect-ordered code.
Emitting declarations after codegen also means no guessing, and adding an op
cannot leave a stale declaration behind.

## 5. Buffers and addressing

- Global parameters are `.param .u64` + `ld.param.u64` + `cvta.to.global.u64`.
- Global addresses are `base + (index * elem_size)` with `mul.wide.s32` and a
  64-bit add.
- Shared memory is addressed with 32-bit offsets, which is what `st.shared`
  expects.

If `meta.smem_bytes > 0`, or the kernel's shared buffers exceed 48 KiB, all
shared buffers go into one `.extern .shared` pool with the same alignment and
packing Emmy uses:

```ptx
.extern .shared .align 16 .b8 __smem_pool[];
```

Otherwise each buffer gets a static `.shared .align N .b8` declaration. This is
what lets `m01` request 73,728 B of dynamic shared memory while `m02` stays
static.

## 6. Accumulators are mutable state

This is the detail that mattered most.

`Accum` is not SSA. It reads as

```
name = op(base_or_name, value)
```

and the result must be written back into `name` itself, because the next
iteration — or a later statement outside the loop — reads that same register.
A backend that treats `Accum` as "compute a fresh SSA value" produces a sum of
only the last iteration. It happens to work for single-iteration tiles and
fails for softmax and matmul, which is exactly how it was found.

The rules implemented in `stmt_accum`:

- seed on first use with the op's identity (`0`, `1`, `-1e30`, `1e30`);
- `base`, when present, is only the left operand (softmax uses it);
- coerce back to the declared accumulator dtype;
- when entering a loop, re-seed accumulators declared in its body, but do not
  descend into nested loops (they seed themselves).

## 7. Reductions

Two instructions are used by Emmy's reductions:

**`WarpShuffle`** is a butterfly `shfl.sync.bfly.b32` ladder. States are paired
(`state`, `state_b`) and combined by ordinary `Assign` statements, which is why
softmax can reduce a max and a sum in one pass. Two details matter:

- `activemask.b32` is used instead of assuming a full warp;
- after the ladder the reduced value is moved back into the original register,
  so a shuffle inside a runtime loop carries correctly across iterations.

**`TreeHalve`** is a shared-memory tree: for `s = length/2 … 1`, threads
`t < s` load `buf[t]` and `buf[t+s]`, combine, and store back, then `bar.sync`.
At the end, slot 0 is broadcast to every thread. The interpreter mirrors this
exactly, including the barrier ordering.

f16 states are packed through `cvt.u32.u16` for the shuffle and stored as `b16`
in shared memory for the tree.

## 8. Async copies

`m01` uses a three-stage `cp.async` pipeline. The lowering supports:

- 4/8/16-byte copies (`ca` for small, `cg` for 16 B);
- an optional `lane_index` that decomposes a copy into `(row, col)` offsets
  within one shared buffer row;
- optional XOR swizzles (`B128`, `B64`, `B32`, with an optional explicit
  shift);
- `commit_group` / `wait_group`, preserved verbatim from the IR.

The interpreter executes each copy immediately. That is correct for this IR
because the pipeline is race-free by construction; commit/wait only constrain
timing, not data dependencies.

## 9. Alignment analysis

Vector access is only legal if the address is aligned. `flat_aligned` computes
the stride of every index component and asks `mod_is_zero(idx, need)` whether
the expression is provably a multiple of `need`. `mod_is_zero` handles
literals, addition/subtraction, multiplication by constants, and shifts.

The rule is conservative: no proof, no vector instruction. That is why the
compiler is correct on odd shapes, and why some access patterns stay scalar.
The natural next step (not needed for correctness) is to replace the syntactic
check with a small divisibility lattice so that more accesses are proven
aligned.

Atomics never vectorise; they always take the scalar `red.global.add.*` path.

## 10. The interpreter

`interp.py` executes one *chunk of blocks at a time*, vectorised across all
threads of those blocks:

- `tid = lane % blockDim.x`, `cta = base + lane // blockDim.x`;
- shared memory is `(chunk, *shape)`, indexed by the lane's block;
- divergence is a boolean mask: `Cond` intersects it, `Loop`/`StridedLoop`
  iterate while any lane is live, all register writes merge with the mask;
- `Sync` is a no-op because execution is lockstep, which is valid for race-free
  IR;
- `cp.async` copies immediately, for the same reason;
- inactive lanes (tile guard, loop guard) are clamped to valid indices and
  their writes are dropped, mirroring the PTX branch;
- the chunk size is bounded by 256 MiB of shared memory and 2²⁰ lanes.

The interpreter is deliberately strict: an out-of-bounds access on an *active*
lane raises instead of silently wrapping. That turns address bugs into test
failures.

## 11. Testing

```bash
python scripts/check_all.py          # GPU: 15/15
python scripts/verify_semantics.py   # CPU: 15/15
```

The two are complementary: the first validates the PTX, the second validates
the meaning. A change that breaks either is visible immediately. The planned
differential fuzzer (`scripts/fuzz.py`) builds on the second by generating
kernels that are not in the open set, compiling them, running them on the GPU,
and diffing against the model.

## 12. Deliberate non-features

The task states that starred IR nodes are not evaluated. Rather than shipping
approximate tensor-core or TMA lowering, these raise
`NotImplementedError` with a clear message: a loud failure is easier to reason
about than quietly wrong PTX. `Select` and `ZeroPrologue` are implemented
anyway because they are cheap and make the backend usable for adjacent IRs.
