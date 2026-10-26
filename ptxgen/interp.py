"""Executable reference semantics for Emmy Kernel IR.

This module is a second, independent implementation of the same IR that
``ptxgen.emit`` lowers to PTX.  It runs the kernel on the CPU with numpy and
is used as a golden model:

* ``scripts/verify_semantics.py`` checks it against the reference outputs of
  every open problem (no GPU involved);
* ``scripts/fuzz.py`` compares it against the PTX that actually ran on the
  GPU, so any disagreement is a compiler bug that the open set did not catch.

The execution model mirrors the IR, not the PTX: one block at a time, all
threads of the block vectorised into numpy arrays.  A thread is a lane of a
boolean mask, so ``Cond`` divergence and bounds guards are just masked
writes.  ``Sync`` is a no-op because the interpreter is lockstep; that is
correct for every race-free IR, which is what the lowering task guarantees.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from emmy.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    FlatIndex,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from emmy.compiler.ir.kernel import (
    Accum,
    Assign,
    Cond,
    Load,
    Loop,
    Select,
    Smem,
    StridedLoop,
    Sync,
    Tile,
    TreeHalve,
    WarpShuffle,
    Write,
)
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait
from emmy.compiler.ir.stmt import Init, Let, ZeroPrologue

from . import ops
from .model import KernelSpec, dtype_info

NP = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    "s32": np.int32,
    "s64": np.int64,
    "pred": np.bool_,
}


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------


@dataclass
class V:
    """A typed per-lane value: one numpy scalar/array element per thread."""

    kind: str
    data: np.ndarray

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=NP[self.kind])


def v(kind: str, data) -> V:
    return V(kind, data)


def kind_of(dtype) -> str:
    name = str(getattr(dtype, "name", dtype))
    return dtype_info(name)[0]


def np_dtype(dtype) -> np.dtype:
    return np.dtype(NP[kind_of(dtype)])


def _is_int_literal(e: Literal) -> bool:
    """Same rule as the PTX lowering: dtype decides, value is the fallback."""
    name = getattr(e.dtype, "name", e.dtype)
    if name in ("int", "i32", "u32", "i64", "u64"):
        return True
    if name in ("float", "f32", "f64", "f16"):
        return False
    return isinstance(e.value, int) and not isinstance(e.value, bool)


def _promote(a: V, b: V) -> str:
    if "f64" in (a.kind, b.kind):
        return "f64"
    if a.kind in ("f32", "f16") or b.kind in ("f32", "f16"):
        return "f32"
    return "s32"


def cv(val: V, kind: str) -> V:
    """Mirror of ``Compiler.coerce`` (PTX ``cvt`` semantics)."""
    if val.kind == kind:
        return val
    data = val.data
    if val.kind == "pred":
        if kind in ("s32", "s64"):
            return V(kind, np.where(data, 1, 0))
        if kind in ("f32", "f64", "f16"):
            return V(kind, np.where(data, 1.0, 0.0))
    if kind == "f32":
        return V("f32", data.astype(np.float32))
    if kind == "f64":
        return V("f64", data.astype(np.float64))
    if kind == "f16":
        # cvt.rn.f16.f32: one rounding to half.
        return V("f16", data.astype(np.float32).astype(np.float16))
    if kind == "s32":
        if val.kind in ("f32", "f64", "f16"):
            # cvt.rzi: truncate toward zero.
            return V("s32", np.trunc(data.astype(np.float64)).astype(np.int32))
        return V("s32", data.astype(np.int32))
    if kind == "s64":
        if val.kind in ("f32", "f64", "f16"):
            return V("s64", np.trunc(data.astype(np.float64)).astype(np.int64))
        return V("s64", data.astype(np.int64))
    raise NotImplementedError(f"coerce {val.kind} -> {kind}")


def as_pred(val: V) -> V:
    if val.kind == "pred":
        return val
    if val.kind == "f16":
        val = cv(val, "f32")
    if val.kind in ("f32", "f64"):
        return V("pred", val.data != 0)
    return V("pred", val.data != 0)


def select(cond: V, a: V, b: V, kind: str) -> V:
    c = as_pred(cond).data
    return V(kind, np.where(c, cv(a, kind).data, cv(b, kind).data))


def _int(val: V) -> np.ndarray:
    if val.kind in ("s32", "s64"):
        return val.data.astype(np.int64)
    if val.kind == "pred":
        return val.data.astype(np.int64)
    return np.trunc(val.data.astype(np.float64)).astype(np.int64)


# ---------------------------------------------------------------------------
# elementwise ops (mirror of Compiler.call_op)
# ---------------------------------------------------------------------------


def _trunc_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.trunc(a / b)
    return np.where(b != 0, q, 0.0)


def arith(name: str, args: list[V]) -> V:
    a, b = args
    kind = _promote(a, b)
    a, b = cv(a, kind), cv(b, kind)
    if kind == "s32":
        if name == "add":
            d = a.data + b.data
        elif name == "subtract":
            d = a.data - b.data
        elif name == "multiply":
            d = a.data * b.data
        elif name == "divide":
            d = _trunc_div(a.data, b.data)
        elif name == "maximum":
            d = np.maximum(a.data, b.data)
        elif name == "minimum":
            d = np.minimum(a.data, b.data)
        else:  # pragma: no cover - guarded by caller
            raise NotImplementedError(name)
        return V("s32", d.astype(np.int32))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if name == "add":
            d = a.data + b.data
        elif name == "subtract":
            d = a.data - b.data
        elif name == "multiply":
            d = a.data * b.data
        elif name == "divide":
            d = a.data / b.data
        elif name == "maximum":
            d = np.where(np.isnan(a.data), b.data, np.where(np.isnan(b.data), a.data,
                                                            np.maximum(a.data, b.data)))
        elif name == "minimum":
            d = np.where(np.isnan(a.data), b.data, np.where(np.isnan(b.data), a.data,
                                                            np.minimum(a.data, b.data)))
        else:  # pragma: no cover
            raise NotImplementedError(name)
    return V(kind, d)


def _erf(x: np.ndarray) -> np.ndarray:
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.fromiter((math.erf(t) for t in flat), dtype=np.float64, count=flat.size)
    return out.reshape(np.shape(x)).astype(np.float32)


def _f32(data) -> np.ndarray:
    return np.asarray(data, dtype=np.float32)


def unary(name: str, args: list[V]) -> V:
    val = args[0]
    if name == "copy":
        return val
    if name in ("negative", "absolute") and val.kind == "s32":
        d = -val.data if name == "negative" else np.abs(val.data)
        return V("s32", d.astype(np.int32))
    if name == "bitcast":
        data = np.ascontiguousarray(val.data)
        if val.kind == "f32":
            return V("s32", data.view(np.int32))
        if val.kind == "s32":
            return V("f32", data.view(np.float32))
        if val.kind == "f64":
            return V("s64", data.view(np.int64))
        if val.kind == "s64":
            return V("f64", data.view(np.float64))
        return val
    x = cv(val, "f32").data
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if name == "negative":
            return V("f32", -x)
        if name == "absolute":
            return V("f32", np.abs(x))
        if name == "square":
            return arith("multiply", [V("f32", x), V("f32", x)])
        if name in ("exp", "exp_fast"):
            return V("f32", _f32(np.exp(x.astype(np.float64))))
        if name == "expm1":
            return V("f32", _f32(np.expm1(x.astype(np.float64))))
        if name == "log":
            return V("f32", _f32(np.log(x.astype(np.float64))))
        if name == "log1p":
            return V("f32", _f32(np.log1p(x.astype(np.float64))))
        if name == "log2":
            return V("f32", _f32(np.log2(x.astype(np.float64))))
        if name == "log10":
            return V("f32", _f32(np.log10(x.astype(np.float64))))
        if name == "sqrt":
            return V("f32", _f32(np.sqrt(x.astype(np.float64))))
        if name == "rsqrt":
            return V("f32", _f32(1.0 / np.sqrt(x.astype(np.float64))))
        if name == "reciprocal":
            return V("f32", _f32(1.0 / x.astype(np.float64)))
        if name == "tanh":
            return V("f32", _f32(np.tanh(x.astype(np.float64))))
        if name == "sin":
            return V("f32", _f32(np.sin(x.astype(np.float64))))
        if name == "cos":
            return V("f32", _f32(np.cos(x.astype(np.float64))))
        if name == "sinh":
            return V("f32", _f32(np.sinh(x.astype(np.float64))))
        if name == "cosh":
            return V("f32", _f32(np.cosh(x.astype(np.float64))))
        if name == "erf":
            return V("f32", _erf(x))
        if name == "sigmoid":
            return V("f32", _f32(1.0 / (1.0 + np.exp(-x.astype(np.float64)))))
        if name == "silu":
            return V("f32", _f32(x.astype(np.float64) /
                                 (1.0 + np.exp(-x.astype(np.float64)))))
        if name == "softplus":
            return V("f32", _f32(np.log1p(np.exp(x.astype(np.float64)))))
        if name in ("gelu", "gelu_tanh"):
            xd = x.astype(np.float64)
            if name == "gelu_tanh":
                inner = math.sqrt(2.0 / math.pi) * (xd + 0.044715 * xd ** 3)
                t = np.tanh(inner)
            else:
                t = _erf(_f32(xd / math.sqrt(2.0))).astype(np.float64)
            return V("f32", _f32(0.5 * xd * (1.0 + t)))
        if name == "relu":
            return V("f32", np.where(x > 0, x, _f32(0.0)))
        if name == "sign":
            return V("f32", np.where(x > 0, 1.0, np.where(x < 0, -1.0, 0.0)))
        if name == "floor":
            return V("f32", _f32(np.floor(x.astype(np.float64))))
        if name == "ceil":
            return V("f32", _f32(np.ceil(x.astype(np.float64))))
        if name == "trunc":
            return V("f32", _f32(np.trunc(x.astype(np.float64))))
        if name == "round":
            return V("f32", _f32(np.rint(x.astype(np.float64))))
        if name == "isnan":
            return V("pred", np.isnan(x))
        if name == "isinf":
            return V("pred", np.isinf(x))
        if name == "logical_not":
            return V("pred", ~as_pred(val).data)
    raise NotImplementedError(f"unsupported unary op {name!r}")


def call_op(name: str, args: list[V]) -> V:
    name = ops.canon(name)
    if name in ("add", "subtract", "multiply", "divide", "maximum", "minimum"):
        return arith(name, args)
    if name == "fma":
        a, b, c = args
        kind = "f64" if "f64" in (a.kind, b.kind, c.kind) else "f32"
        a, b, c = cv(a, kind), cv(b, kind), cv(c, kind)
        return V(kind, a.data * b.data + c.data)
    if name == "where":
        cond, a, b = args
        kind = _promote(a, b)
        return select(cond, a, b, kind)
    if name == "clip":
        a, lo, hi = args
        return arith("minimum", [arith("maximum", [a, lo]), hi])
    if name == "floor_divide":
        a, b = args
        if a.kind in ("f32", "f64", "f16") or b.kind in ("f32", "f64", "f16"):
            return unary("floor", [arith("divide", [a, b])])
        return arith("divide", [a, b])
    if name == "fmod":
        a, b = args
        if a.kind in ("f32", "f64", "f16") or b.kind in ("f32", "f64", "f16"):
            a, b = cv(a, "f32"), cv(b, "f32")
            return V("f32", _f32(np.fmod(a.data.astype(np.float64),
                                        b.data.astype(np.float64))))
        return V("s32", np.fmod(cv(a, "s32").data, cv(b, "s32").data).astype(np.int32))
    if name in ("logical_and", "logical_or"):
        a, b = as_pred(args[0]).data, as_pred(args[1]).data
        return V("pred", np.logical_and(a, b) if name == "logical_and" else np.logical_or(a, b))
    if name == "power":
        a, b = cv(args[0], "f32"), cv(args[1], "f32")
        return V("f32", _f32(np.exp(b.data.astype(np.float64) *
                                    np.log(a.data.astype(np.float64)))))
    if name == "bitcast":
        return unary("bitcast", [args[0]])
    if name in ("remainder", "bitand", "bitor", "bitxor", "shl", "shr"):
        a, b = cv(args[0], "s32").data, cv(args[1], "s32").data
        if name == "remainder":
            return V("s32", np.fmod(a, b).astype(np.int32))
        if name == "bitand":
            return V("s32", np.bitwise_and(a, b))
        if name == "bitor":
            return V("s32", np.bitwise_or(a, b))
        if name == "bitxor":
            return V("s32", np.bitwise_xor(a, b))
        # PTX shifts use the low 5 bits of the shift count.
        sh = np.bitwise_and(b, 31)
        if name == "shl":
            return V("s32", np.left_shift(a, sh))
        return V("s32", np.right_shift(a, sh))  # shr.s32 is arithmetic
    return unary(name, args)


# ---------------------------------------------------------------------------
# interpreter
# ---------------------------------------------------------------------------


class Interpreter:
    def __init__(self, spec: KernelSpec, inputs: dict[str, np.ndarray]):
        self.spec = spec
        bx, by, bz = (int(x) for x in spec.block)
        if by != 1 or bz != 1:
            raise NotImplementedError("interpreter supports 1-D blocks only")
        self.T = bx
        self.nctas = int(np.prod(spec.grid))
        self.cta0 = 0
        self.chunk = 1
        self.nlanes = bx
        self.tid = np.arange(bx, dtype=np.int64)
        self.cta = np.zeros(bx, dtype=np.int64)
        self.env: dict[str, V] = {}
        self.globals: dict[str, np.ndarray] = {}
        self.smem: dict[str, np.ndarray] = {}
        self.smem_schema = self._smem_schema(spec.op.body)
        self.mask: np.ndarray | None = None

        for name, arg in spec.args.items():
            if arg.scalar:
                continue
            data = inputs.get(name)
            if data is None:
                data = np.zeros(arg.shape, dtype=np_dtype(arg.dtype))
            self.globals[name] = np.ascontiguousarray(np.asarray(data, dtype=np_dtype(arg.dtype)))

    # -- setup ------------------------------------------------------------

    @staticmethod
    def _smem_schema(body) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        out: dict[str, tuple[tuple[int, ...], np.dtype]] = {}

        def walk(stmts):
            for st in stmts:
                if isinstance(st, Smem):
                    out[st.name] = (tuple(int(e) for e in st.extents), np_dtype(st.dtype))
                for child in st.nested():
                    walk(child)

        walk(body)
        return out

    def _shape(self, name: str) -> tuple[int, ...]:
        if name in self.smem_schema:
            return self.smem_schema[name][0]
        if name in self.spec.args:
            return self.spec.args[name].shape
        raise KeyError(f"unknown buffer {name!r}")

    def _array(self, name: str) -> np.ndarray:
        if name in self.smem:
            return self.smem[name]
        if name in self.globals:
            return self.globals[name]
        raise KeyError(f"unknown buffer {name!r}")

    # -- masking ----------------------------------------------------------

    @contextmanager
    def masked(self, cond: V | np.ndarray | None):
        old = self.mask
        if cond is not None:
            c = np.broadcast_to(as_pred(cond).data if isinstance(cond, V) else cond,
                                (self.nlanes,)).astype(bool)
            self.mask = c if old is None else (old & c)
        try:
            yield
        finally:
            self.mask = old

    def active(self) -> np.ndarray:
        if self.mask is None:
            return np.ones(self.nlanes, dtype=bool)
        return self.mask

    def set(self, name: str, val: V) -> None:
        """Assign a register value, merging with lanes masked off."""
        m = self.mask
        if m is not None and not m.all():
            old = self.env.get(name)
            if old is None:
                val = V(val.kind, np.where(m, val.data, 0))
            else:
                old = cv(old, val.kind)
                val = V(val.kind, np.where(m, val.data, old.data))
        self.env[name] = val

    # -- expressions ------------------------------------------------------

    def eval_expr(self, e) -> V:
        if isinstance(e, Var):
            if e.name in self.env:
                return self.env[e.name]
            raise KeyError(f"undefined SSA/axis variable {e.name!r}")
        if isinstance(e, Literal):
            if _is_int_literal(e):
                return v("s32", int(e.value))
            return v("f32", np.float32(e.value))
        if isinstance(e, BinaryExpr):
            return self.eval_binary(e)
        if isinstance(e, FuncCallExpr):
            return call_op(e.name, [self.eval_expr(a) for a in e.args])
        if isinstance(e, TernaryExpr):
            cond = as_pred(self.eval_expr(e.cond))
            a = self.eval_expr(e.if_true)
            b = self.eval_expr(e.if_false)
            kind = _promote(a, b)
            return select(cond, a, b, kind)
        if isinstance(e, CastExpr):
            val = self.eval_expr(e.expr)
            return cv(val, kind_of(e.dtype))
        if isinstance(e, FlatIndex):
            return v("s32", self.flat_index(e.index, self._shape(e.buffer)).astype(np.int32))
        if isinstance(e, Builtin):
            return self.eval_builtin(e.name)
        raise NotImplementedError(f"unsupported expression {type(e).__name__}")

    def eval_binary(self, e: BinaryExpr) -> V:
        op = e.op
        if op in ("==", "!=", "<", "<=", ">", ">="):
            a = self.eval_expr(e.left)
            b = self.eval_expr(e.right)
            kind = _promote(a, b)
            a, b = cv(a, kind), cv(b, kind)
            ad, bd = a.data, b.data
            fn = {
                "==": np.equal, "!=": np.not_equal, "<": np.less,
                "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
            }[op]
            with np.errstate(invalid="ignore"):
                return V("pred", fn(ad, bd))
        if op in ("&&", "||"):
            a = as_pred(self.eval_expr(e.left)).data
            b = as_pred(self.eval_expr(e.right)).data
            return V("pred", np.logical_and(a, b) if op == "&&" else np.logical_or(a, b))
        return call_op(op, [self.eval_expr(e.left), self.eval_expr(e.right)])

    def eval_builtin(self, name: str) -> V:
        key = name.replace("blockDim", "block_dim").replace("gridDim", "grid_dim")
        key = key.replace("threadIdx", "thread_idx").replace("blockIdx", "block_idx")
        tid = self.tid
        if key in ("thread_idx.x", "tid"):
            return v("s32", tid)
        if key in ("block_idx.x", "cta", "block_idx"):
            return v("s32", self.cta)
        if key in ("block_dim.x", "ntid"):
            return v("s32", self.T)
        if key in ("lane", "lane_id"):
            return v("s32", tid & 31)
        if key in ("warp", "warp_id"):
            return v("s32", tid >> 5)
        if key in ("warp_size", "warpSize"):
            return v("s32", 32)
        if key == "thread_idx.y":
            return v("s32", 0)
        if key == "thread_idx.z":
            return v("s32", 0)
        if key == "block_idx.y":
            gx = int(self.spec.grid[0])
            return v("s32", (self.cta // gx) % int(self.spec.grid[1]))
        if key == "block_idx.z":
            gx, gy = int(self.spec.grid[0]), int(self.spec.grid[1])
            return v("s32", self.cta // (gx * gy))
        if key == "block_dim.y":
            return v("s32", int(self.spec.block[1]))
        if key == "block_dim.z":
            return v("s32", int(self.spec.block[2]))
        if key in ("grid_dim.x", "grid_dim.y", "grid_dim.z"):
            return v("s32", int(self.spec.grid["xyz".index(key[-1])]))
        raise NotImplementedError(f"unknown builtin {name!r}")

    def lane_flat(self, index, shape) -> np.ndarray:
        """Flat element index, broadcast to one entry per lane."""
        flat = self.flat_index(index, shape)
        return np.broadcast_to(flat, (self.nlanes,))

    def flat_index(self, index, shape) -> np.ndarray:
        n = len(index)
        if n == 0:
            return np.zeros((), dtype=np.int64)
        dims = [int(d) for d in shape]
        while len(dims) < n:
            dims.insert(0, 1)
        dims = dims[-n:]
        strides = [1] * n
        acc = 1
        for i in range(n - 1, -1, -1):
            strides[i] = acc
            acc *= dims[i]
        total = np.zeros((), dtype=np.int64)
        for idx, stride in zip(index, strides):
            term = _int(self.eval_expr(idx))
            total = total + (term * stride if stride != 1 else term)
        return total

    def swizzle(self, flat: np.ndarray, mode: str) -> np.ndarray:
        table = {"B128": (6, 0x7), "B64": (6, 0x3), "B32": (6, 0x1)}
        base, _, shift_txt = str(mode or "NONE").partition("@")
        if base not in table:
            return flat
        shift, mask = table[base]
        if shift_txt:
            shift = int(shift_txt)
        return flat ^ (((flat >> shift) & mask) << 3)

    # -- statements -------------------------------------------------------

    def run_body(self, body) -> None:
        for st in body:
            self.stmt(st)

    def stmt(self, st) -> None:
        if isinstance(st, Tile):
            self.stmt_tile(st)
        elif isinstance(st, Load):
            self.stmt_load(st)
        elif isinstance(st, Assign):
            self.stmt_assign(st)
        elif isinstance(st, Accum):
            self.stmt_accum(st)
        elif isinstance(st, Let):
            self.stmt_let(st)
        elif isinstance(st, Write):
            self.stmt_write(st)
        elif isinstance(st, Loop):
            self.stmt_loop(st)
        elif isinstance(st, StridedLoop):
            self.stmt_strided_loop(st)
        elif isinstance(st, Cond):
            self.stmt_cond(st)
        elif isinstance(st, Init):
            self.stmt_init(st)
        elif isinstance(st, Smem):
            return
        elif isinstance(st, Sync):
            return  # lockstep interpretation: a barrier is already a no-op
        elif isinstance(st, WarpShuffle):
            self.stmt_warp_shuffle(st)
        elif isinstance(st, TreeHalve):
            self.stmt_tree_halve(st)
        elif isinstance(st, CpAsyncCopy):
            self.stmt_cp_async(st)
        elif isinstance(st, (CpAsyncCommit, CpAsyncWait)):
            return  # copies land immediately; ordering is preserved
        elif isinstance(st, Select):
            self.stmt_select(st)
        elif isinstance(st, ZeroPrologue):
            self.stmt_zero_prologue(st)
        else:
            raise NotImplementedError(f"unsupported statement {type(st).__name__}")

    def stmt_tile(self, st: Tile) -> None:
        extents = [int(a.extent.as_static()) for a in st.axes]
        n = 1
        for e in extents:
            n *= e
        gid = self.cta * self.T + self.tid
        active = gid < n
        # Clamp inactive lanes so their speculative reads stay in bounds; all
        # their writes are masked out below, exactly like the PTX branch.
        safe = np.minimum(gid, n - 1) if n else gid
        strides = [1] * len(extents)
        acc = 1
        for i in range(len(extents) - 1, -1, -1):
            strides[i] = acc
            acc *= extents[i]
        saved = {a.name: self.env.get(a.name) for a in st.axes}
        for axis, extent, stride in zip(st.axes, extents, strides):
            if extent == 1:
                self.env[axis.name] = v("s32", 0)
            elif stride == 1:
                self.env[axis.name] = v("s32", safe % extent)
            else:
                self.env[axis.name] = v("s32", (safe // stride) % extent)
        with self.masked(active):
            self.run_body(st.body)
        for axis, old in saved.items():
            if old is None:
                self.env.pop(axis, None)
            else:
                self.env[axis] = old

    def _load_value(self, buf_name: str, flat: np.ndarray, value_kind: str,
                    swizzle: str = "NONE") -> V:
        m = self.active()
        if not m.all():
            flat = np.where(m, flat, 0)
        schema = self.smem_schema.get(buf_name)
        if schema is not None:
            arr = self.smem[buf_name].reshape(self.chunk, -1)
            storage = schema[1]
            flat = self.swizzle(flat, swizzle)
            if np.any(flat >= arr.shape[1]) or np.any(flat < 0):
                raise IndexError(f"shared load out of bounds in {buf_name!r}")
            raw = arr[self.cta - self.cta0, flat]
        else:
            arr = self.globals[buf_name].reshape(-1)
            storage = np_dtype(self.spec.args[buf_name].dtype)
            if np.any(flat >= arr.size) or np.any(flat < 0):
                raise IndexError(f"global load out of bounds in {buf_name!r}")
            raw = arr[flat]
        storage_kind = {
            np.dtype(np.float32): "f32", np.dtype(np.float64): "f64",
            np.dtype(np.float16): "f16", np.dtype(np.int32): "s32",
            np.dtype(np.int64): "s64", np.dtype(np.bool_): "pred",
        }[storage]
        return cv(V(storage_kind, raw), value_kind)

    def stmt_load(self, st: Load) -> None:
        dtype = st.dtype if st.dtype is not None else (
            self.smem_schema[st.input][1] if st.input in self.smem_schema
            else self.spec.args[st.input].dtype)
        value_kind = kind_of(dtype)
        flat = self.lane_flat(st.index, self._shape(st.input))
        m = self.active()
        if not m.all():
            # Inactive lanes are branched around in the PTX; keep them in
            # bounds instead of faulting on a speculatively computed address.
            flat = np.where(m, flat, 0)
        swizzle = getattr(st, "swizzle", "NONE")
        for k, name in enumerate(st.names):
            self.set(name, self._load_value(st.input, flat + k, value_kind, swizzle))

    def stmt_assign(self, st: Assign) -> None:
        res = call_op(st.op.name, [self.env[a] for a in st.args])
        if st.dtype is not None:
            res = cv(res, kind_of(st.dtype))
        self.set(st.name, res)

    def stmt_accum(self, st: Accum) -> None:
        kind = kind_of(st.dtype) if st.dtype is not None else "f32"
        acc = self.env.get(st.name)
        if acc is None or acc.kind != kind:
            acc = self._identity(st.op.name, kind)
        left = self.env[st.base] if st.base is not None else acc
        res = call_op(st.op.name, [left, self.env[st.value]])
        if st.dtype is not None:
            res = cv(res, kind)
        self.set(st.name, res)

    def stmt_let(self, st: Let) -> None:
        val = self.eval_expr(st.value)
        if st.dtype is not None:
            val = cv(val, kind_of(st.dtype))
        self.set(st.name, val)

    def stmt_init(self, st: Init) -> None:
        self.set(st.name, self._identity(st.identity, kind_of(st.dtype)))

    def _identity(self, op_or_value, kind: str) -> V:
        if kind in ("s32", "s64"):
            value = int(op_or_value) if not isinstance(op_or_value, str) else int(ops.identity(op_or_value))
        elif isinstance(op_or_value, str):
            value = np.float32(ops.identity(op_or_value))
        else:
            value = np.float32(op_or_value)
        return V(kind, value)

    def stmt_write(self, st: Write) -> None:
        shared = st.output in self.smem_schema
        if st.output in self.smem:
            arr = self.smem[st.output]
        elif st.output in self.globals:
            arr = self.globals[st.output]
        else:
            raise KeyError(f"write to unknown buffer {st.output!r}")
        dtype = arr.dtype
        buf_kind = kind_of(dtype)
        flat = self.lane_flat(st.index, self._shape(st.output))
        m = self.active()
        for k, name in enumerate(st.values):
            idx = flat + k
            val = cv(self.env[name], buf_kind)
            if shared:
                idx = self.swizzle(idx, getattr(st, "swizzle", "NONE"))
            data = np.asarray(val.data)
            if data.ndim == 0:
                data = np.broadcast_to(data, (self.nlanes,))
            ci = self.cta - self.cta0
            if not m.all():
                idx = idx[m]
                data = data[m]
                ci = ci[m]
            if shared:
                view = self.smem[st.output].reshape(self.chunk, -1)
                if np.any(idx >= view.shape[1]) or np.any(idx < 0):
                    raise IndexError(f"shared store out of bounds in {st.output!r}")
                view[ci, idx] = np.asarray(data, dtype=dtype)
            else:
                off = arr.reshape(-1)
                if np.any(idx >= off.size) or np.any(idx < 0):
                    raise IndexError(f"store out of bounds in {st.output!r}")
                if st.atomic:
                    np.add.at(off, idx, np.asarray(data, dtype=dtype))
                else:
                    off[idx] = np.asarray(data, dtype=dtype)

    # -- control flow -----------------------------------------------------

    def _seed_accums(self, body, enabled: bool) -> None:
        seen: set[str] = set()

        def scan(stmts):
            for st in stmts:
                if isinstance(st, Accum):
                    if st.name in seen:
                        continue
                    seen.add(st.name)
                    if not enabled:
                        continue
                    kind = kind_of(st.dtype) if st.dtype is not None else "f32"
                    self.env[st.name] = self._identity(st.op.name, kind)
                elif isinstance(st, (Loop, StridedLoop)):
                    continue
                else:
                    for child in st.nested():
                        scan(child)

        scan(body)

    def stmt_loop(self, st: Loop) -> None:
        self._seed_accums(st.body, bool(st.seed))
        axis = st.axis.name
        extent = int(st.axis.extent.as_static())
        old = self.env.get(axis)
        var = np.zeros((), dtype=np.int64)
        steps = 0
        while True:
            live = var < extent
            if self.mask is not None:
                live = live & self.mask
            if not np.any(live):
                break
            self.env[axis] = v("s32", var)
            with self.masked(live):
                self.run_body(st.body)
            var = var + 1
            steps += 1
            if steps > 1_000_000:
                raise RuntimeError(f"loop over {axis!r} did not terminate")
        if old is None:
            self.env.pop(axis, None)
        else:
            self.env[axis] = old

    def stmt_strided_loop(self, st: StridedLoop) -> None:
        self._seed_accums(st.body, bool(st.seed))
        axis = st.axis.name
        extent = int(st.axis.extent.as_static())
        start = self.eval_expr(st.start)
        step = self.eval_expr(st.step) if hasattr(st.step, "free_vars") else v("s32", int(st.step))
        bound = self.eval_expr(st.end) if st.end is not None else v("s32", extent)
        old = self.env.get(axis)
        var = _int(start)
        step_i = _int(step)
        bound_i = _int(bound)
        steps = 0
        while True:
            live = var < bound_i
            if self.mask is not None:
                live = live & self.mask
            if not np.any(live):
                break
            self.env[axis] = v("s32", var)
            with self.masked(live):
                self.run_body(st.body)
            var = var + step_i
            steps += 1
            if steps > 1_000_000:
                raise RuntimeError(f"loop over {axis!r} did not terminate")
        if old is None:
            self.env.pop(axis, None)
        else:
            self.env[axis] = old

    def stmt_cond(self, st: Cond) -> None:
        cond = as_pred(self.eval_expr(st.cond))
        with self.masked(cond):
            self.run_body(st.body)
        if st.else_body:
            with self.masked(V("pred", ~cond.data)):
                self.run_body(st.else_body)

    # -- collective ops ---------------------------------------------------

    def stmt_warp_shuffle(self, st: WarpShuffle) -> None:
        m = self.active()
        s = int(st.length) // 2
        while s > 0:
            for state, state_b in zip(st.state, st.state_b):
                cur = self.env[state]
                partner = self.tid ^ s
                gidx = self.cta * self.T + partner
                safe = np.clip(gidx, 0, self.nlanes - 1)
                valid = m[safe] & (partner < self.T)
                gathered = cur.data.reshape(-1)[safe]
                # shfl.sync with an inactive source lane yields the caller's
                # own value; emulate that instead of importing garbage.
                data = np.where(valid, gathered, cur.data)
                self.set(state_b, V(cur.kind, data))
            for assign in st.combine_states:
                self.stmt(assign)
            s //= 2
        # The PTX backend writes the reduced value back into the original
        # registers so a loop-carried shuffle stays correct; env already holds
        # it, nothing to copy.

    def stmt_tree_halve(self, st: TreeHalve) -> None:
        if st.inner is not None:
            raise NotImplementedError("TreeHalve(inner=...) is not supported")
        t = _int(self.env[st.tid_var])
        length = int(st.length)
        m = self.active()
        ci = self.cta - self.cta0
        originals = {name: self.env[name].kind for name in st.state}
        buf_kind = {name: kind_of(self.smem_schema[name][1]) for name in st.bufs}
        views = {name: self.smem[name].reshape(self.chunk, -1) for name in st.bufs}
        s = length // 2
        while s > 0:
            inner = (t < s) & m
            for buf_name, state, state_b in zip(st.bufs, st.state, st.state_b):
                view = views[buf_name]
                size = view.shape[1]
                src = np.clip(t, 0, size - 1)
                off_b = np.clip(t + s, 0, size - 1)
                self.set(state, V(buf_kind[buf_name], view[ci, src]))
                self.set(state_b, V(buf_kind[buf_name], view[ci, off_b]))
            with self.masked(inner):
                for assign in st.combine_states:
                    self.stmt(assign)
            for buf_name, state in zip(st.bufs, st.state):
                view = views[buf_name]
                val = cv(self.env[state], buf_kind[buf_name])
                idx = np.clip(t, 0, view.shape[1] - 1)
                data = np.asarray(val.data)
                if data.ndim == 0:
                    data = np.broadcast_to(data, (self.nlanes,))
                if inner.any():
                    view[ci[inner], idx[inner]] = data[inner]
            s //= 2
        for buf_name, state in zip(st.bufs, st.state):
            view = views[buf_name]
            loaded = V(buf_kind[buf_name], view[ci, 0])
            self.set(state, cv(loaded, originals[state]))

    # -- async / misc -----------------------------------------------------

    def stmt_cp_async(self, st: CpAsyncCopy) -> None:
        if st.smem not in self.smem:
            raise KeyError(f"cp.async destination {st.smem!r} is not a Smem buffer")
        view = self.smem[st.smem].reshape(self.chunk, -1)
        numel = view.shape[1]
        src = self.globals[st.src].reshape(-1)
        dst_flat = self.lane_flat(st.smem_index, self.smem_schema[st.smem][0])
        if st.lane_index is not None:
            cols = int(self.smem_schema[st.smem][0][-1]) if self.smem_schema[st.smem][0] else 1
            lane = _int(self.eval_expr(st.lane_index[0]))
            col = _int(self.eval_expr(st.lane_index[1]))
            dst_flat = dst_flat + lane * cols + col
        dst_flat = self.swizzle(dst_flat, st.swizzle)
        src_flat = self.lane_flat(st.src_index, self._shape(st.src))
        m = self.active()
        ci = self.cta - self.cta0
        if not m.all():
            dst_flat = dst_flat[m]
            src_flat = src_flat[m]
            ci = ci[m]
        itemsize = src.dtype.itemsize
        count = int(st.nbytes) // itemsize
        offsets = np.arange(count, dtype=np.int64)
        dst_idx = dst_flat[:, None] + offsets if np.ndim(dst_flat) else dst_flat + offsets
        src_idx = src_flat[:, None] + offsets if np.ndim(src_flat) else src_flat + offsets
        if np.any(dst_idx >= numel) or np.any(src_idx >= src.size):
            raise IndexError("cp.async out of bounds")
        global_idx = (ci[:, None] * numel + dst_idx) if np.ndim(ci) else (ci * numel + dst_idx)
        view.reshape(-1)[global_idx] = src[src_idx]

    def stmt_select(self, st: Select) -> None:
        branches = list(st.branches)
        kind = self.env[branches[-1].value].kind
        for b in branches[:-1]:
            k = self.env[b.value].kind
            if k in ("f32", "f64") or kind in ("f32", "f64"):
                kind = "f64" if "f64" in (k, kind) else "f32"
            else:
                kind = "s32"
        res = self.env[branches[-1].value]
        for b in reversed(branches[:-1]):
            res = select(as_pred(self.eval_expr(b.select)), self.env[b.value], res, kind)
        self.set(st.name, res)

    def stmt_zero_prologue(self, st: ZeroPrologue) -> None:
        if not np.any(self.cta == 0):
            return
        arr = self.globals[st.dst]
        nbytes = min(int(st.words) * 4, arr.nbytes)
        if nbytes > 0:
            arr.reshape(-1).view(np.uint8)[:nbytes] = 0

    # -- entry point ------------------------------------------------------

    def run(self) -> dict[str, np.ndarray]:
        smem_bytes = 0
        for shape, dtype in self.smem_schema.values():
            n = 1
            for e in shape:
                n *= int(e)
            smem_bytes += n * np.dtype(dtype).itemsize
        # Bound the footprint of the vectorised grid, then run whole chunks of
        # blocks at once: lanes = chunk * blockDim.x.
        budget = 256 * 1024 * 1024
        chunk = max(1, min(self.nctas, budget // max(1, smem_bytes)))
        chunk = min(chunk, max(1, (1 << 20) // self.T))
        for base in range(0, self.nctas, chunk):
            n = min(chunk, self.nctas - base)
            self.chunk = n
            self.cta0 = base
            self.nlanes = n * self.T
            self.tid = np.arange(self.nlanes, dtype=np.int64) % self.T
            self.cta = base + np.arange(self.nlanes, dtype=np.int64) // self.T
            self.env = {}
            self.mask = None
            self.smem = {
                name: np.zeros((n,) + tuple(shape), dtype=dtype)
                for name, (shape, dtype) in self.smem_schema.items()
            }
            self.env["lane"] = v("s32", self.tid & 31)
            self.env["warp"] = v("s32", self.tid >> 5)
            for name in self.spec.arg_order:
                arg = self.spec.args[name]
                if arg.scalar:
                    value = arg.value if arg.value is not None else 0.0
                    self.env[name] = v(kind_of(arg.dtype), value)
            self.run_body(self.spec.op.body)
        return {name: self.globals[name] for name, arg in self.spec.args.items() if arg.output}


def interpret(spec: KernelSpec, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return Interpreter(spec, inputs).run()
