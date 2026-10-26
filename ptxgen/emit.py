"""Kernel IR -> PTX lowering.

The IR is loaded through emmy's own classes (allowed); every line of PTX below
is produced here.  No emmy renderer, nvcc, or external compiler is involved.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

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
from .loader import load_kernel
from .model import KernelSpec, SmemBuf, dtype_info

STATIC_SMEM_CAP = 48 * 1024


# ---------------------------------------------------------------------------
# values and literals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Val:
    kind: str          # f32 / f64 / f16 / s32 / s64 / pred
    text: str          # register name or immediate text
    imm: bool = False


def f32_bits(v: float) -> str:
    bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
    return f"0f{bits:08X}"


def f64_bits(v: float) -> str:
    bits = struct.unpack("<Q", struct.pack("<d", float(v)))[0]
    return f"0d{bits:016X}"


def f_bits(v: float, kind: str) -> str:
    return f64_bits(v) if kind == "f64" else f32_bits(v)


def imm_float(v: Val) -> float:
    if v.kind == "f64":
        return struct.unpack("<d", struct.pack("<Q", int(v.text[2:], 16)))[0]
    return struct.unpack("<f", struct.pack("<I", int(v.text[2:], 16)))[0]


def imm_int(v: Val) -> int:
    if v.kind in ("f32", "f64", "f16"):
        return int(imm_float(v))
    return int(v.text)


def sanitize(name: str) -> str:
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    if not out or out[0].isdigit():
        out = "_" + out
    return out


# ---------------------------------------------------------------------------
# expression helpers
# ---------------------------------------------------------------------------


def _is_int_literal(e) -> bool:
    if not isinstance(e, Literal):
        return False
    name = getattr(e.dtype, "name", e.dtype)
    if name in ("int", "i32", "u32", "i64", "u64"):
        return True
    if name in ("float", "f32", "f64", "f16"):
        return False
    return isinstance(e.value, int) and not isinstance(e.value, bool)


def mod_is_zero(e, w: int) -> bool:
    """True if expression ``e`` is provably a multiple of ``w`` (for alignment)."""
    if w == 1:
        return True
    if isinstance(e, Literal):
        if isinstance(e.value, int) and not isinstance(e.value, bool):
            return e.value % w == 0
        return False
    if isinstance(e, BinaryExpr):
        if e.op in ("+", "-"):
            return mod_is_zero(e.left, w) and mod_is_zero(e.right, w)
        if e.op == "*":
            for lit, other in ((e.left, e.right), (e.right, e.left)):
                if isinstance(lit, Literal) and isinstance(lit.value, int):
                    if lit.value % w == 0:
                        return True
                    sub = w // math.gcd(abs(lit.value), w)
                    if sub > 1 and mod_is_zero(other, sub):
                        return True
            return False
        if e.op in ("<<",):
            if isinstance(e.right, Literal) and isinstance(e.right.value, int):
                shift = int(e.right.value)
                if shift >= 0:
                    return mod_is_zero(e.left, w // math.gcd(1 << shift, w))
            return False
    return False


# ---------------------------------------------------------------------------
# compiler
# ---------------------------------------------------------------------------


class Compiler:
    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self.lines: list[str] = []
        self.decls: list[str] = []

        self.n_f = 0
        self.n_fd = 0
        self.n_h = 0
        self.n_r = 0
        self.n_rd = 0
        self.n_p = 0
        self.n_label = 0

        self.env: dict[str, Val] = {}
        self.gbase: dict[str, str] = {}
        self.sbase: dict[str, str] = {}
        self.smem: dict[str, SmemBuf] = {}
        self.bufs = {}
        self.dynamic_smem = False
        self.pool_symbol = "__smem_pool"
        self.pool_base: str | None = None

        self.tid: str | None = None
        self.cta: str | None = None
        self.ntid: str | None = None

        for name, arg in spec.args.items():
            if not arg.scalar:
                self.bufs[name] = spec.buffer(name)
        self._layout_smem()

    # -- register allocation ------------------------------------------------

    def newf(self) -> str:
        r = f"%f{self.n_f}"
        self.n_f += 1
        return r

    def newfd(self) -> str:
        r = f"%fd{self.n_fd}"
        self.n_fd += 1
        return r

    def newh(self) -> str:
        r = f"%h{self.n_h}"
        self.n_h += 1
        return r

    def newr(self) -> str:
        r = f"%r{self.n_r}"
        self.n_r += 1
        return r

    def newrd(self) -> str:
        r = f"%rd{self.n_rd}"
        self.n_rd += 1
        return r

    def newp(self) -> str:
        r = f"%p{self.n_p}"
        self.n_p += 1
        return r

    def label(self) -> str:
        r = f"__L{self.n_label}"
        self.n_label += 1
        return r

    def emit(self, text: str) -> None:
        self.lines.append("        " + text)

    def put(self, label: str) -> None:
        self.lines.append(f"{label}:")

    # -- typed value helpers ------------------------------------------------

    def alloc(self, kind: str) -> Val:
        if kind == "f32":
            return Val(kind, self.newf())
        if kind == "f64":
            return Val(kind, self.newfd())
        if kind == "f16":
            return Val(kind, self.newh())
        if kind in ("s32", "u32"):
            return Val("s32", self.newr())
        if kind in ("s64", "u64"):
            return Val("s64", self.newrd())
        if kind == "pred":
            return Val(kind, self.newp())
        raise NotImplementedError(f"cannot allocate kind {kind!r}")

    @staticmethod
    def _imm_for(kind: str, value) -> Val:
        """Exact immediate for a seed/identity in the requested value kind."""
        if kind in ("s32", "s64", "u32", "u64"):
            if kind in ("s32", "u32"):
                lo, hi = -2147483648, 2147483647
            else:
                lo, hi = -(2 ** 63), 2 ** 63 - 1
            return Val(kind, str(max(lo, min(hi, int(value)))), True)
        return Val(kind, f_bits(float(value), kind), True)

    def mov(self, dst: Val, src: Val) -> None:
        if dst.text == src.text:
            return
        if src.imm:
            if dst.kind == "f32":
                self.emit(f"mov.f32 {dst.text}, {src.text};")
            elif dst.kind == "f64":
                self.emit(f"mov.f64 {dst.text}, {src.text};")
            elif dst.kind == "f16":
                self.mov(dst, self.to_f16(src))
            elif dst.kind == "s32":
                self.emit(f"mov.s32 {dst.text}, {src.text};")
            elif dst.kind == "s64":
                self.emit(f"mov.s64 {dst.text}, {src.text};")
            elif dst.kind == "pred":
                self.emit(f"mov.pred {dst.text}, {src.text};")
            else:
                raise NotImplementedError(f"mov of immediate into {dst.kind}")
            return
        op = {"f32": "mov.f32", "f64": "mov.f64", "f16": "mov.b16",
              "s32": "mov.b32", "s64": "mov.b64", "pred": "mov.pred"}[dst.kind]
        self.emit(f"{op} {dst.text}, {src.text};")

    def to_f32(self, v: Val) -> Val:
        if v.kind == "f32":
            return v
        if v.kind == "f64":
            d = self.alloc("f32")
            self.emit(f"cvt.rn.f32.f64 {d.text}, {v.text};")
            return d
        if v.kind == "f16":
            d = self.alloc("f32")
            self.emit(f"cvt.f32.f16 {d.text}, {v.text};")
            return d
        if v.kind == "s32":
            if v.imm:
                return Val("f32", f32_bits(float(int(v.text))), True)
            d = self.alloc("f32")
            self.emit(f"cvt.rn.f32.s32 {d.text}, {v.text};")
            return d
        if v.kind == "s64":
            d = self.alloc("f32")
            self.emit(f"cvt.rn.f32.s64 {d.text}, {v.text};")
            return d
        raise NotImplementedError(f"cannot convert {v.kind} to f32")

    def to_f64(self, v: Val) -> Val:
        if v.kind == "f64":
            return v
        if v.kind == "f32":
            if v.imm:
                return Val("f64", f64_bits(imm_float(v)), True)
            d = self.alloc("f64")
            self.emit(f"cvt.rn.f64.f32 {d.text}, {v.text};")
            return d
        if v.kind == "s32":
            d = self.alloc("f64")
            self.emit(f"cvt.rn.f64.s32 {d.text}, {v.text};")
            return d
        raise NotImplementedError(f"cannot convert {v.kind} to f64")

    def to_f16(self, v: Val) -> Val:
        if v.kind == "f16":
            return v
        v = self.to_f32(v)
        if v.imm:
            v = self.as_reg(v)
        d = self.alloc("f16")
        self.emit(f"cvt.rn.f16.f32 {d.text}, {v.text};")
        return d

    def to_s32(self, v: Val) -> Val:
        if v.kind == "s32":
            return v
        if v.kind in ("f32", "f64", "f16"):
            if v.imm:
                return Val("s32", str(int(imm_float(v))), True)
            d = self.alloc("s32")
            src = self.to_f32(v)
            self.emit(f"cvt.rzi.s32.f32 {d.text}, {src.text};")
            return d
        if v.kind == "s64":
            d = self.alloc("s32")
            self.emit(f"cvt.s32.s64 {d.text}, {v.text};")
            return d
        raise NotImplementedError(f"cannot convert {v.kind} to s32")

    def to_s64(self, v: Val) -> Val:
        if v.kind == "s64":
            return v
        if v.kind == "s32":
            if v.imm:
                return Val("s64", v.text, True)
            d = self.alloc("s64")
            self.emit(f"cvt.s64.s32 {d.text}, {v.text};")
            return d
        if v.kind in ("f32", "f64"):
            d = self.alloc("s64")
            src = self.to_f32(v)
            self.emit(f"cvt.rzi.s64.f32 {d.text}, {src.text};")
            return d
        raise NotImplementedError(f"cannot convert {v.kind} to s64")

    def as_pred(self, v: Val) -> Val:
        if v.kind == "pred":
            return v
        if v.kind == "f16":
            v = self.to_f32(v)
        p = self.alloc("pred")
        if v.kind in ("f32", "f64"):
            v = self.to_f32(v) if v.kind == "f32" else v
            suffix = "f32" if v.kind == "f32" else "f64"
            zero = f32_bits(0.0) if suffix == "f32" else f64_bits(0.0)
            self.emit(f"setp.ne.{suffix} {p.text}, {v.text}, {zero};")
        elif v.kind == "s64":
            self.emit(f"setp.ne.s64 {p.text}, {v.text}, 0;")
        else:
            self.emit(f"setp.ne.s32 {p.text}, {v.text}, 0;")
        return p

    def as_reg(self, v: Val) -> Val:
        if not v.imm:
            return v
        d = self.alloc(v.kind)
        self.mov(d, v)
        return d

    # -- buffer / shared-memory layout -------------------------------------

    def _layout_smem(self) -> None:
        smems: list[Smem] = []

        def walk(body):
            for st in body:
                if isinstance(st, Smem):
                    smems.append(st)
                for child in st.nested() if hasattr(st, "nested") else ():
                    walk(child)

        walk(self.spec.op.body)
        total = 0
        for s in smems:
            dtype_name = getattr(s.dtype, "name", s.dtype)
            kind, _ptx, nbytes = dtype_info(str(dtype_name))
            elements = 1
            for e in s.extents:
                elements *= int(e)
            align = max(nbytes, int(s.align or 0))
            if align:
                total = (total + align - 1) // align * align
            buf = SmemBuf(
                name=s.name,
                dtype=str(dtype_name),
                shape=tuple(int(e) for e in s.extents),
                align=align,
                symbol=sanitize(s.name),
                offset=total,
            )
            self.smem[s.name] = buf
            total += elements * nbytes
        self.dynamic_smem = self.spec.smem_bytes > 0 or total > STATIC_SMEM_CAP
        self.smem_total = total

        if self.dynamic_smem:
            self.decls.append(f".extern .shared .align 16 .b8 {self.pool_symbol}[];")
        else:
            for buf in self.smem.values():
                _kind, _ptx, nbytes = dtype_info(buf.dtype)
                elements = 1
                for e in buf.shape:
                    elements *= int(e)
                self.decls.append(
                    f".shared .align {max(buf.align, nbytes)} .b8 {buf.symbol}[{elements * nbytes}];"
                )

    # -- entry point --------------------------------------------------------

    def run(self) -> str:
        self._prologue()
        self.emit_body(self.spec.op.body)
        self.emit("ret;")
        return self._assemble()

    def _prologue(self) -> None:
        params = []
        for i, name in enumerate(self.spec.arg_order):
            arg = self.spec.args[name]
            pname = f"p{i}"
            if arg.scalar:
                kind, ptx, _ = dtype_info(arg.dtype)
                params.append(f"        .param .{ptx} {pname}")
                dst = self.alloc(kind)
                self.emit(f"ld.param.{ptx} {dst.text}, [{pname}];")
                self.env[name] = dst
            else:
                params.append(f"        .param .u64 {pname}")
                raw = self.newrd()
                base = self.newrd()
                self.emit(f"ld.param.u64 {raw}, [{pname}];")
                self.emit(f"cvta.to.global.u64 {base}, {raw};")
                self.gbase[name] = base

        self.params = params

        self.tid = self.newr()
        self.cta = self.newr()
        self.ntid = self.newr()
        self.emit(f"mov.u32 {self.tid}, %tid.x;")
        self.emit(f"mov.u32 {self.cta}, %ctaid.x;")
        self.emit(f"mov.u32 {self.ntid}, %ntid.x;")

        lane = self.newr()
        warp = self.newr()
        self.emit(f"and.b32 {lane}, {self.tid}, 31;")
        self.emit(f"shr.u32 {warp}, {self.tid}, 5;")
        self.env["lane"] = Val("s32", lane)
        self.env["warp"] = Val("s32", warp)

        if self.dynamic_smem:
            self.pool_base = self.newr()
            self.emit(f"mov.u32 {self.pool_base}, {self.pool_symbol};")

        for name, buf in self.smem.items():
            if self.dynamic_smem:
                if buf.offset == 0:
                    self.sbase[name] = self.pool_base
                else:
                    reg = self.newr()
                    self.emit(f"add.u32 {reg}, {self.pool_base}, {buf.offset};")
                    self.sbase[name] = reg
            else:
                reg = self.newr()
                self.emit(f"mov.u32 {reg}, {buf.symbol};")
                self.sbase[name] = reg

    def _assemble(self) -> str:
        out = [
            ".version 8.0",
            ".target sm_89",
            ".address_size 64",
            "",
            "// Generated from Emmy Kernel IR by ptxgen.",
        ]
        out.extend(self.decls)
        out.append("")
        out.append(f".visible .entry {self.spec.entry}(")
        out.append(",\n".join(self.params))
        out.append(")")
        out.append("{")
        if self.n_p:
            out.append(f"        .reg .pred %p<{self.n_p}>;")
        if self.n_f:
            out.append(f"        .reg .f32 %f<{self.n_f}>;")
        if self.n_fd:
            out.append(f"        .reg .f64 %fd<{self.n_fd}>;")
        if self.n_h:
            out.append(f"        .reg .b16 %h<{self.n_h}>;")
        if self.n_r:
            out.append(f"        .reg .b32 %r<{self.n_r}>;")
        if self.n_rd:
            out.append(f"        .reg .b64 %rd<{self.n_rd}>;")
        out.extend(self.lines)
        out.append("}")
        return "\n".join(out) + "\n"

    # -- expressions --------------------------------------------------------

    def eval_expr(self, e) -> Val:
        if isinstance(e, Var):
            if e.name in self.env:
                return self.env[e.name]
            raise KeyError(f"undefined SSA/axis variable {e.name!r}")
        if isinstance(e, Literal):
            if _is_int_literal(e):
                return Val("s32", str(int(e.value)), True)
            return Val("f32", f32_bits(float(e.value)), True)
        if isinstance(e, BinaryExpr):
            return self.eval_binary(e)
        if isinstance(e, FuncCallExpr):
            return self.call_op(e.name, [self.eval_expr(a) for a in e.args])
        if isinstance(e, TernaryExpr):
            return self.eval_ternary(e)
        if isinstance(e, CastExpr):
            return self.eval_cast(e)
        if isinstance(e, FlatIndex):
            buf = self.bufs.get(e.buffer) or self.smem.get(e.buffer)
            if buf is None:
                raise KeyError(f"FlatIndex on unknown buffer {e.buffer!r}")
            return self.flatten(e.index, buf.shape)
        if isinstance(e, Builtin):
            return self.eval_builtin(e.name)
        raise NotImplementedError(f"unsupported expression {type(e).__name__}")

    def eval_builtin(self, name: str) -> Val:
        if name in ("warpSize", "warp_size", "warpSize.x"):
            return Val("s32", "32", True)
        key = name.replace("threadIdx", "thread_idx").replace("blockIdx", "block_idx")
        key = key.replace("blockDim", "block_dim").replace("gridDim", "grid_dim")
        reg = {"thread_idx.x": self.tid, "block_idx.x": self.cta, "block_dim.x": self.ntid}
        if key in reg and reg[key]:
            return Val("s32", reg[key])
        special = {
            "thread_idx.y": "%tid.y", "thread_idx.z": "%tid.z",
            "block_idx.y": "%ctaid.y", "block_idx.z": "%ctaid.z",
            "block_dim.y": "%ntid.y", "block_dim.z": "%ntid.z",
            "grid_dim.x": "%nctaid.x", "grid_dim.y": "%nctaid.y", "grid_dim.z": "%nctaid.z",
            "warp_size": None,
        }
        if key == "warp_size":
            return Val("s32", "32", True)
        if key in special and special[key]:
            r = self.newr()
            self.emit(f"mov.u32 {r}, {special[key]};")
            return Val("s32", r)
        raise NotImplementedError(f"unknown builtin {name!r}")

    def eval_binary(self, e: BinaryExpr) -> Val:
        op = e.op
        if op in ("==", "!=", "<", "<=", ">", ">="):
            a = self.eval_expr(e.left)
            b = self.eval_expr(e.right)
            if "f64" in (a.kind, b.kind):
                a, b = self.to_f64(a), self.to_f64(b)
                suffix = "f64"
            elif a.kind in ("f32", "f16") or b.kind in ("f32", "f16"):
                a, b = self.to_f32(a), self.to_f32(b)
                suffix = "f32"
            else:
                a, b = self.to_s32(a), self.to_s32(b)
                suffix = "s32"
            cmp = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le",
                   ">": "gt", ">=": "ge"}[op]
            p = self.alloc("pred")
            self.emit(f"setp.{cmp}.{suffix} {p.text}, {a.text}, {b.text};")
            return p
        if op in ("&&", "||"):
            a = self.as_pred(self.eval_expr(e.left))
            b = self.as_pred(self.eval_expr(e.right))
            d = self.alloc("pred")
            self.emit(f"{'and' if op == '&&' else 'or'}.pred {d.text}, {a.text}, {b.text};")
            return d
        return self.call_op(op, [self.eval_expr(e.left), self.eval_expr(e.right)])

    def eval_ternary(self, e: TernaryExpr) -> Val:
        cond = self.as_pred(self.eval_expr(e.cond))
        a = self.eval_expr(e.if_true)
        b = self.eval_expr(e.if_false)
        if "f64" in (a.kind, b.kind):
            kind = "f64"
        elif a.kind in ("f32", "f16") or b.kind in ("f32", "f16"):
            kind = "f32"
        else:
            kind = "s32"
        a = self.coerce(a, kind)
        b = self.coerce(b, kind)
        d = self.alloc(kind)
        self.emit(f"selp.{kind if kind != 's32' else 's32'} {d.text}, {a.text}, {b.text}, {cond.text};")
        return d

    def eval_cast(self, e: CastExpr) -> Val:
        v = self.eval_expr(e.expr)
        name = str(getattr(e.dtype, "name", e.dtype))
        if name in ("int", "i32", "u32"):
            return self.to_s32(v)
        if name in ("float", "f32"):
            return self.to_f32(v)
        if name == "f64":
            return self.to_f64(v)
        if name == "f16":
            return self.to_f16(v)
        if name in ("i64", "u64"):
            return self.to_s64(v)
        raise NotImplementedError(f"cast to {name!r}")

    def coerce(self, v: Val, kind: str) -> Val:
        if v.kind == kind:
            return v
        if v.kind == "pred":
            if kind in ("s32", "s64", "u32"):
                d = self.alloc("s32")
                self.emit(f"selp.s32 {d.text}, 1, 0, {v.text};")
                return d if kind == "s32" else self.coerce(d, kind)
            if kind in ("f32", "f64", "f16"):
                d = self.alloc("f32")
                self.emit(f"selp.f32 {d.text}, 0f3F800000, 0f00000000, {v.text};")
                return d if kind == "f32" else self.coerce(d, kind)
        if kind == "f32":
            return self.to_f32(v)
        if kind == "f64":
            return self.to_f64(v)
        if kind == "f16":
            return self.to_f16(v)
        if kind == "s32":
            return self.to_s32(v)
        if kind == "s64":
            return self.to_s64(v)
        raise NotImplementedError(f"coerce {v.kind} -> {kind}")

    # -- op emission --------------------------------------------------------

    def call_op(self, name: str, args: list[Val], dest: Val | None = None) -> Val:
        name = ops.canon(name)
        if name in ("add", "subtract", "multiply", "divide", "maximum", "minimum"):
            return self._arith(name, args, dest)
        if name == "fma":
            return self._fma(args, dest)
        if name == "where":
            return self._where(args)
        if name == "clip":
            a, lo, hi = args
            return self._arith("minimum", [self._arith("maximum", [a, lo]), hi])
        if name == "floor_divide":
            a, b = args
            if a.kind in ("f32", "f64", "f16") or b.kind in ("f32", "f64", "f16"):
                return self._unary("floor", [self._arith("divide", [a, b])])
            return self._arith("divide", [a, b])
        if name == "fmod":
            a, b = args
            if a.kind in ("f32", "f64", "f16") or b.kind in ("f32", "f64", "f16"):
                q = self._arith("divide", [a, b])
                t = self._unary("trunc", [q])
                return self._arith("subtract", [a, self._arith("multiply", [t, b])])
            return self.call_op("remainder", [a, b])
        if name in ("logical_and", "logical_or"):
            pa = self.as_pred(args[0])
            pb = self.as_pred(args[1])
            d = self.alloc("pred")
            self.emit(f"{'and' if name == 'logical_and' else 'or'}.pred {d.text}, {pa.text}, {pb.text};")
            return d
        if name == "power":
            a = self.to_f32(args[0])
            b = self.to_f32(args[1])
            return self._unary("exp", [self._arith("multiply", [b, self._unary("log", [a])])])
        if name == "bitcast":
            return self._bitcast(args[0])
        if name in ("remainder", "bitand", "bitor", "bitxor", "shl", "shr"):
            a = self.to_s32(args[0])
            b = self.to_s32(args[1])
            instr = {
                "remainder": "rem.s32", "bitand": "and.b32", "bitor": "or.b32",
                "bitxor": "xor.b32", "shl": "shl.b32", "shr": "shr.s32",
            }[name]
            d = self.alloc("s32")
            self.emit(f"{instr} {d.text}, {a.text}, {b.text};")
            return d
        return self._unary(name, args, dest)

    def _arith(self, name: str, args: list[Val], dest: Val | None = None) -> Val:
        a, b = args[0], args[1]
        if "f64" in (a.kind, b.kind):
            kind = "f64"
        elif a.kind in ("f32", "f16") or b.kind in ("f32", "f16"):
            kind = "f32"
        else:
            kind = "s32"
        a = self.coerce(a, kind)
        b = self.coerce(b, kind)
        d = dest if (dest is not None and dest.kind == kind) else self.alloc(kind)
        if kind == "s32":
            instr = {"add": "add.s32", "subtract": "sub.s32", "multiply": "mul.lo.s32",
                     "divide": "div.s32", "maximum": "max.s32", "minimum": "min.s32"}[name]
            self.emit(f"{instr} {d.text}, {a.text}, {b.text};")
        else:
            instr = {"add": "add", "subtract": "sub", "multiply": "mul",
                     "divide": "div.rn", "maximum": "max", "minimum": "min"}[name]
            self.emit(f"{instr}.{kind} {d.text}, {a.text}, {b.text};")
        return d

    def _fma(self, args: list[Val], dest: Val | None = None) -> Val:
        a, b, c = args
        kind = "f64" if "f64" in (a.kind, b.kind, c.kind) else "f32"
        a, b, c = self.coerce(a, kind), self.coerce(b, kind), self.coerce(c, kind)
        d = dest if (dest is not None and dest.kind == kind) else self.alloc(kind)
        self.emit(f"fma.rn.{kind} {d.text}, {a.text}, {b.text}, {c.text};")
        return d

    def _where(self, args: list[Val]) -> Val:
        cond = self.as_pred(args[0])
        a, b = args[1], args[2]
        if "f64" in (a.kind, b.kind):
            kind = "f64"
        elif a.kind in ("f32", "f16") or b.kind in ("f32", "f16"):
            kind = "f32"
        else:
            kind = "s32"
        a, b = self.coerce(a, kind), self.coerce(b, kind)
        d = self.alloc(kind)
        self.emit(f"selp.{kind} {d.text}, {a.text}, {b.text}, {cond.text};")
        return d

    def _bitcast(self, v: Val) -> Val:
        if v.kind in ("f32", "s32"):
            d = self.alloc("s32" if v.kind == "f32" else "f32")
            self.emit(f"mov.b32 {d.text}, {v.text};")
            return d
        if v.kind in ("f64", "s64"):
            d = self.alloc("s64" if v.kind == "f64" else "f64")
            self.emit(f"mov.b64 {d.text}, {v.text};")
            return d
        return v

    def _unary(self, name: str, args: list[Val], dest: Val | None = None) -> Val:
        v = args[0]
        if name == "copy":
            return v
        if name in ("negative", "absolute") and v.kind == "s32":
            d = self.alloc("s32")
            self.emit(f"{'neg' if name == 'negative' else 'abs'}.s32 {d.text}, {v.text};")
            return d
        if name == "bitcast":
            return self._bitcast(v)
        v = self.to_f32(v)
        if name == "negative":
            d = self.alloc("f32")
            self.emit(f"neg.f32 {d.text}, {v.text};")
            return d
        if name == "absolute":
            d = self.alloc("f32")
            self.emit(f"abs.f32 {d.text}, {v.text};")
            return d
        if name == "square":
            return self._arith("multiply", [v, v])
        if name == "expm1":
            return self._arith("subtract", [self._unary("exp", [v]),
                                            Val("f32", f32_bits(1.0), True)])
        if name == "log1p":
            return self._unary("log", [self._arith("add", [Val("f32", f32_bits(1.0), True), v])])
        if name == "sinh":
            ep = self._unary("exp", [v])
            en = self._unary("exp", [self._unary("negative", [v])])
            return self._arith("multiply", [Val("f32", f32_bits(0.5), True),
                                            self._arith("subtract", [ep, en])])
        if name == "cosh":
            ep = self._unary("exp", [v])
            en = self._unary("exp", [self._unary("negative", [v])])
            return self._arith("multiply", [Val("f32", f32_bits(0.5), True),
                                            self._arith("add", [ep, en])])
        if name == "sign":
            zero = Val("f32", f32_bits(0.0), True)
            one = Val("f32", f32_bits(1.0), True)
            neg_one = Val("f32", f32_bits(-1.0), True)
            ppos = self.alloc("pred")
            pneg = self.alloc("pred")
            self.emit(f"setp.gt.f32 {ppos.text}, {v.text}, {zero.text};")
            self.emit(f"setp.lt.f32 {pneg.text}, {v.text}, {zero.text};")
            d = self.alloc("f32")
            tmp = self.alloc("f32")
            self.emit(f"selp.f32 {tmp.text}, {one.text}, {zero.text}, {ppos.text};")
            self.emit(f"selp.f32 {d.text}, {neg_one.text}, {tmp.text}, {pneg.text};")
            return d
        if name == "logical_not":
            p = self.as_pred(v)
            d = self.alloc("pred")
            self.emit(f"not.pred {d.text}, {p.text};")
            return d
        if name == "isnan":
            d = self.alloc("pred")
            self.emit(f"setp.nan.f32 {d.text}, {v.text}, {v.text};")
            return d
        if name == "isinf":
            d = self.alloc("pred")
            av = self._unary("absolute", [v])
            self.emit(f"setp.eq.f32 {d.text}, {av.text}, 0f7F800000;")
            return d
        if name == "relu":
            d = self.alloc("f32")
            self.emit(f"max.f32 {d.text}, {v.text}, {f32_bits(0.0)};")
            return d
        if name in ("exp", "exp_fast"):
            t = self._mul_imm(v, math.log2(math.e))
            d = self.alloc("f32")
            self.emit(f"ex2.approx.ftz.f32 {d.text}, {t.text};")
            return d
        if name == "log":
            t = self.alloc("f32")
            self.emit(f"lg2.approx.ftz.f32 {t.text}, {v.text};")
            return self._mul_imm(t, math.log(2.0))
        if name == "log2":
            d = self.alloc("f32")
            self.emit(f"lg2.approx.ftz.f32 {d.text}, {v.text};")
            return d
        if name == "log10":
            t = self.alloc("f32")
            self.emit(f"lg2.approx.ftz.f32 {t.text}, {v.text};")
            return self._mul_imm(t, math.log10(2.0))
        if name == "sqrt":
            d = self.alloc("f32")
            self.emit(f"sqrt.rn.f32 {d.text}, {v.text};")
            return d
        if name == "rsqrt":
            d = self.alloc("f32")
            self.emit(f"rsqrt.approx.f32 {d.text}, {v.text};")
            return d
        if name == "reciprocal":
            d = self.alloc("f32")
            self.emit(f"rcp.rn.f32 {d.text}, {v.text};")
            return d
        if name == "tanh":
            d = self.alloc("f32")
            self.emit(f"tanh.approx.f32 {d.text}, {v.text};")
            return d
        if name == "sigmoid":
            neg = self._unary("negative", [v])
            e = self._unary("exp", [neg])
            one = Val("f32", f32_bits(1.0), True)
            denom = self._arith("add", [one, e])
            return self._unary("reciprocal", [denom])
        if name == "silu":
            s = self._unary("sigmoid", [v])
            return self._arith("multiply", [v, s])
        if name == "softplus":
            e = self._unary("exp", [v])
            one = Val("f32", f32_bits(1.0), True)
            return self._unary("log", [self._arith("add", [one, e])])
        if name in ("gelu", "gelu_tanh", "erf"):
            if name == "erf":
                return self._erf(v)
            half = Val("f32", f32_bits(0.5), True)
            one = Val("f32", f32_bits(1.0), True)
            if name == "gelu_tanh":
                # 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
                c = Val("f32", f32_bits(0.044715), True)
                scale = Val("f32", f32_bits(math.sqrt(2.0 / math.pi)), True)
                x3 = self._arith("multiply", [self._arith("multiply", [v, v]), v])
                inner = self._arith("add", [v, self._arith("multiply", [c, x3])])
                t = self._unary("tanh", [self._arith("multiply", [scale, inner])])
                return self._arith("multiply", [self._arith("multiply", [half, v]),
                                                self._arith("add", [one, t])])
            inv_sqrt2 = Val("f32", f32_bits(1.0 / math.sqrt(2.0)), True)
            arg = self._arith("multiply", [v, inv_sqrt2])
            e = self._erf(arg)
            return self._arith("multiply", [self._arith("multiply", [half, v]),
                                            self._arith("add", [one, e])])
        if name in ("floor", "ceil", "trunc", "round"):
            mod = {"floor": "rmi", "ceil": "rpi", "trunc": "rzi", "round": "rni"}[name]
            d = self.alloc("f32")
            self.emit(f"cvt.{mod}.f32.f32 {d.text}, {v.text};")
            return d
        if name == "sin":
            d = self.alloc("f32")
            self.emit(f"sin.approx.f32 {d.text}, {v.text};")
            return d
        if name == "cos":
            d = self.alloc("f32")
            self.emit(f"cos.approx.f32 {d.text}, {v.text};")
            return d
        if name == "power":
            raise NotImplementedError("power is binary")
        raise NotImplementedError(f"unsupported unary op {name!r}")

    def _mul_imm(self, v: Val, c: float) -> Val:
        d = self.alloc("f32")
        self.emit(f"mul.f32 {d.text}, {v.text}, {f32_bits(c)};")
        return d

    def _erf(self, x: Val) -> Val:
        """Abramowitz & Stegun 7.1.26, |error| < 1.5e-7."""
        p = 0.3275911
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        ax = self._unary("absolute", [x])
        t = self._unary("reciprocal", [self._arith("add", [Val("f32", f32_bits(1.0), True),
                                                          self._mul_imm(ax, p)])])
        poly = Val("f32", f32_bits(a5), True)
        for c in (a4, a3, a2, a1):
            poly = self._arith("add", [self._arith("multiply", [poly, t]), Val("f32", f32_bits(c), True)])
        sq = self._arith("multiply", [x, x])
        neg_sq = self._unary("negative", [sq])
        e = self._unary("exp", [neg_sq])
        prod = self._arith("multiply", [poly, e])
        val = self._arith("subtract", [Val("f32", f32_bits(1.0), True), prod])
        neg_val = self._unary("negative", [val])
        pred = self.alloc("pred")
        self.emit(f"setp.lt.f32 {pred.text}, {x.text}, {f32_bits(0.0)};")
        d = self.alloc("f32")
        self.emit(f"selp.f32 {d.text}, {neg_val.text}, {val.text}, {pred.text};")
        return d

    # -- index / address helpers -------------------------------------------

    def flatten(self, index, shape) -> Val:
        n = len(index)
        if n == 0:
            return Val("s32", "0", True)
        dims = list(int(d) for d in shape)
        while len(dims) < n:
            dims.insert(0, 1)
        dims = dims[-n:]
        strides = [1] * n
        acc = 1
        for i in range(n - 1, -1, -1):
            strides[i] = acc
            acc *= dims[i]
        total: Val | None = None
        for idx, stride in zip(index, strides):
            term = self.eval_expr(idx)
            if stride == 1:
                pass
            elif stride == 0:
                continue
            else:
                term = self._arith("multiply", [term, Val("s32", str(stride), True)])
            if term.imm and int(term.text) == 0:
                continue
            total = term if total is None else self._arith("add", [total, term])
        return total if total is not None else Val("s32", "0", True)

    def flat_aligned(self, index, shape, width: int) -> bool:
        n = len(index)
        if n == 0:
            return True
        dims = list(int(d) for d in shape)
        while len(dims) < n:
            dims.insert(0, 1)
        dims = dims[-n:]
        strides = [1] * n
        acc = 1
        for i in range(n - 1, -1, -1):
            strides[i] = acc
            acc *= dims[i]
        for idx, stride in zip(index, strides):
            need = width // math.gcd(stride, width)
            if need > 1 and not mod_is_zero(idx, need):
                return False
        return True

    def global_addr(self, name: str, flat: Val) -> str:
        if name not in self.gbase:
            raise KeyError(f"no global pointer for buffer {name!r}")
        base = self.gbase[name]
        nbytes = self.bufs[name].elem_bytes
        if flat.imm:
            off = int(flat.text) * nbytes
            if off == 0:
                return f"[{base}]"
            r = self.newr()
            self.emit(f"mov.s32 {r}, {off};")
            rd = self.newrd()
            self.emit(f"mul.wide.s32 {rd}, {r}, 1;")
            out = self.newrd()
            self.emit(f"add.s64 {out}, {base}, {rd};")
            return f"[{out}]"
        r = self.to_s32(flat)
        rd = self.newrd()
        self.emit(f"mul.wide.s32 {rd}, {r.text}, {nbytes};")
        out = self.newrd()
        self.emit(f"add.s64 {out}, {base}, {rd};")
        return f"[{out}]"

    def smem_addr(self, name: str, flat: Val) -> str:
        if name not in self.sbase:
            raise KeyError(f"no shared base for buffer {name!r}")
        base = self.sbase[name]
        nbytes = self.smem[name].elem_bytes
        if flat.imm:
            off = int(flat.text) * nbytes
            return f"[{base}+{off}]" if off else f"[{base}]"
        r = self.to_s32(flat)
        if nbytes != 1:
            scaled = self.newr()
            self.emit(f"mul.lo.s32 {scaled}, {r.text}, {nbytes};")
            r = Val("s32", scaled)
        out = self.newr()
        self.emit(f"add.u32 {out}, {base}, {r.text};")
        return f"[{out}]"

    def apply_swizzle(self, flat: Val, mode: str) -> Val:
        table = {"B128": (6, 0x7), "B64": (6, 0x3), "B32": (6, 0x1)}
        base, _, shift_txt = mode.partition("@")
        if base not in table:
            return flat
        shift, mask = table[base]
        if shift_txt:
            shift = int(shift_txt)
        if flat.imm:
            e = int(flat.text)
            return Val("s32", str(e ^ (((e >> shift) & mask) << 3)), True)
        r = self.to_s32(flat)
        a = self.newr()
        b = self.newr()
        c = self.newr()
        d = self.newr()
        self.emit(f"shr.u32 {a}, {r.text}, {shift};")
        self.emit(f"and.b32 {b}, {a}, {mask};")
        self.emit(f"shl.b32 {c}, {b}, 3;")
        self.emit(f"xor.b32 {d}, {r.text}, {c};")
        return Val("s32", d)

    # -- statements ---------------------------------------------------------

    def emit_body(self, body) -> None:
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
            self.stmt_sync(st)
        elif isinstance(st, WarpShuffle):
            self.stmt_warp_shuffle(st)
        elif isinstance(st, TreeHalve):
            self.stmt_tree_halve(st)
        elif isinstance(st, CpAsyncCopy):
            self.stmt_cp_async(st)
        elif isinstance(st, CpAsyncCommit):
            self.emit("cp.async.commit_group;")
        elif isinstance(st, CpAsyncWait):
            self.emit(f"cp.async.wait_group {int(st.group)};")
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
        gid = self.newr()
        self.emit(f"mad.lo.u32 {gid}, {self.cta}, {self.ntid}, {self.tid};")
        end = self.label()
        p = self.alloc("pred")
        self.emit(f"setp.lt.u32 {p.text}, {gid}, {n};")
        self.emit(f"@!{p.text} bra {end};")
        strides = [1] * len(extents)
        acc = 1
        for i in range(len(extents) - 1, -1, -1):
            strides[i] = acc
            acc *= extents[i]
        for i, (axis, extent) in enumerate(zip(st.axes, extents)):
            var = self.newr()
            if extent == 1:
                self.emit(f"mov.s32 {var}, 0;")
            elif i == 0:
                if strides[i] == 1:
                    self.emit(f"mov.s32 {var}, {gid};")
                else:
                    self.emit(f"div.u32 {var}, {gid}, {strides[i]};")
            elif strides[i] == 1:
                self.emit(f"rem.u32 {var}, {gid}, {extent};")
            else:
                t = self.newr()
                self.emit(f"div.u32 {t}, {gid}, {strides[i]};")
                self.emit(f"rem.u32 {var}, {t}, {extent};")
            self.env[axis.name] = Val("s32", var)
        self.emit_body(st.body)
        self.put(end)

    def stmt_load(self, st: Load) -> None:
        buf = self.bufs.get(st.input) or self.smem.get(st.input)
        if buf is None:
            raise KeyError(f"load from unknown buffer {st.input!r}")
        dtype = str(getattr(st.dtype, "name", st.dtype)) if st.dtype is not None else buf.dtype
        kind, ptx, nbytes = dtype_info(dtype)
        width = len(st.names)
        shared = st.input in self.smem

        vector_ok = (
            width in (2, 4)
            and not shared
            and kind == "f32"
            and nbytes == 4
            and self.flat_aligned(st.index, buf.shape, width)
        )
        if vector_ok:
            flat = self.flatten(st.index, buf.shape)
            addr = self.global_addr(st.input, flat)
            regs = [self.alloc("f32") for _ in range(width)]
            names = ", ".join(r.text for r in regs)
            self.emit(f"ld.global.v{width}.f32 {{{names}}}, {addr};")
            for name, reg in zip(st.names, regs):
                self.env[name] = reg
            return

        for k, name in enumerate(st.names):
            index = st.index
            if width > 1:
                if not index:
                    raise NotImplementedError("vector load without an index")
                index = tuple(index[:-1]) + (BinaryExpr("+", index[-1], Literal(k, "int")),)
            flat = self.flatten(index, buf.shape)
            if shared:
                flat = self.apply_swizzle(flat, getattr(st, "swizzle", "NONE"))
                addr = self.smem_addr(st.input, flat)
                if kind == "f32":
                    d = self.alloc("f32")
                    self.emit(f"ld.shared.f32 {d.text}, {addr};")
                elif kind == "f64":
                    d = self.alloc("f64")
                    self.emit(f"ld.shared.f64 {d.text}, {addr};")
                elif kind == "f16":
                    h = self.alloc("f16")
                    self.emit(f"ld.shared.b16 {h.text}, {addr};")
                    d = self.alloc("f32")
                    self.emit(f"cvt.f32.f16 {d.text}, {h.text};")
                else:
                    d = self.alloc(kind)
                    self.emit(f"ld.shared.{ptx} {d.text}, {addr};")
            else:
                addr = self.global_addr(st.input, flat)
                if kind == "f16":
                    h = self.alloc("f16")
                    self.emit(f"ld.global.b16 {h.text}, {addr};")
                    d = self.alloc("f32")
                    self.emit(f"cvt.f32.f16 {d.text}, {h.text};")
                else:
                    d = self.alloc(kind)
                    self.emit(f"ld.global.{ptx} {d.text}, {addr};")
            self.env[name] = d

    def stmt_assign(self, st: Assign) -> None:
        vals = [self.get_val(a) for a in st.args]
        res = self.call_op(st.op.name, vals)
        if st.dtype is not None:
            res = self.coerce(res, str(st.dtype.name))
        self.env[st.name] = res

    def stmt_accum(self, st: Accum) -> None:
        name = st.name
        kind = str(st.dtype.name) if st.dtype is not None else "f32"
        if kind == "u32":
            kind = "s32"
        elif kind == "u64":
            kind = "s64"
        acc = self.env.get(name)
        if acc is None or acc.kind != kind:
            if kind == "f16":
                acc = self.to_f16(Val("f32", f32_bits(ops.identity(st.op.name)), True))
            else:
                acc = self.alloc(kind)
                self.mov(acc, self._imm_for(kind, ops.identity(st.op.name)))
            self.env[name] = acc
        left = self.get_val(st.base) if st.base is not None else acc
        val = self.get_val(st.value)
        res = self.call_op(st.op.name, [left, val], dest=acc)
        if st.dtype is not None:
            res = self.coerce(res, kind)
        if res is not acc:
            self.mov(acc, res)
        self.env[name] = acc

    def stmt_let(self, st: Let) -> None:
        v = self.eval_expr(st.value)
        if st.dtype is not None:
            v = self.coerce(v, str(st.dtype.name))
        self.env[st.name] = v

    def stmt_init(self, st: Init) -> None:
        kind = str(st.dtype.name)
        if kind == "f16":
            d = self.to_f16(Val("f32", f32_bits(float(st.identity)), True))
        else:
            d = self.alloc(kind)
            self.mov(d, self._imm_for(kind, st.identity))
        self.env[st.name] = d

    def stmt_write(self, st: Write) -> None:
        shared = st.output in self.smem
        buf = self.smem.get(st.output) or self.bufs.get(st.output)
        if buf is None:
            raise KeyError(f"write to unknown buffer {st.output!r}")
        width = len(st.values)
        kind, ptx, nbytes = dtype_info(buf.dtype)

        if shared:
            for k, name in enumerate(st.values):
                index = st.index
                if width > 1:
                    index = tuple(index[:-1]) + (BinaryExpr("+", index[-1], Literal(k, "int")),)
                flat = self.apply_swizzle(self.flatten(index, buf.shape), st.swizzle)
                addr = self.smem_addr(st.output, flat)
                v = self.get_val(name)
                v = self.coerce(v, "f32" if kind == "f16" else kind)
                if kind == "f16":
                    h = self.to_f16(v)
                    self.emit(f"st.shared.b16 {addr}, {h.text};")
                else:
                    self.emit(f"st.shared.{ptx} {addr}, {v.text};")
            return

        if (not st.atomic and width in (2, 4) and kind == "f32" and nbytes == 4
                and self.flat_aligned(st.index, buf.shape, width)):
            flat = self.flatten(st.index, buf.shape)
            addr = self.global_addr(st.output, flat)
            regs = [self.as_reg(self.coerce(self.get_val(n), "f32")) for n in st.values]
            names = ", ".join(r.text for r in regs)
            self.emit(f"st.global.v{width}.f32 {addr}, {{{names}}};")
            return

        for k, name in enumerate(st.values):
            index = st.index
            if width > 1:
                index = tuple(index[:-1]) + (BinaryExpr("+", index[-1], Literal(k, "int")),)
            flat = self.flatten(index, buf.shape)
            addr = self.global_addr(st.output, flat)
            v = self.coerce(self.get_val(name), "f32" if kind == "f16" else kind)
            if st.atomic:
                if kind == "f16":
                    h = self.to_f16(v)
                    self.emit(f"red.global.add.f16 {addr}, {h.text};")
                else:
                    self.emit(f"red.global.add.{ptx} {addr}, {v.text};")
            elif kind == "f16":
                h = self.to_f16(v)
                self.emit(f"st.global.b16 {addr}, {h.text};")
            else:
                self.emit(f"st.global.{ptx} {addr}, {v.text};")

    def _seed_accums(self, body, enabled: bool) -> None:
        seen = set()

        def scan(stmts):
            for st in stmts:
                if isinstance(st, Accum):
                    if st.name in seen:
                        continue
                    seen.add(st.name)
                    if not enabled:
                        continue
                    kind = str(st.dtype.name) if st.dtype is not None else "f32"
                    if kind == "f16":
                        d = self.to_f16(Val("f32", f32_bits(ops.identity(st.op.name)), True))
                    else:
                        d = self.alloc(kind)
                        self.mov(d, self._imm_for(kind, ops.identity(st.op.name)))
                    self.env[st.name] = d
                elif isinstance(st, (Loop, StridedLoop)):
                    # Nested loops seed their own accumulators at their entry.
                    continue
                elif hasattr(st, "nested"):
                    for child in st.nested():
                        scan(child)

        scan(body)

    def stmt_loop(self, st: Loop) -> None:
        self._seed_accums(st.body, bool(st.seed))
        axis = st.axis.name
        extent = int(st.axis.extent.as_static())
        old = self.env.get(axis)
        var = self.newr()
        self.emit(f"mov.s32 {var}, 0;")
        self.env[axis] = Val("s32", var)
        top = self.label()
        end = self.label()
        self.put(top)
        p = self.alloc("pred")
        self.emit(f"setp.ge.s32 {p.text}, {var}, {extent};")
        self.emit(f"@{p.text} bra {end};")
        self.emit_body(st.body)
        self.emit(f"add.s32 {var}, {var}, 1;")
        self.emit(f"bra {top};")
        self.put(end)
        if old is None:
            self.env.pop(axis, None)
        else:
            self.env[axis] = old

    def stmt_strided_loop(self, st: StridedLoop) -> None:
        self._seed_accums(st.body, bool(st.seed))
        axis = st.axis.name
        extent = int(st.axis.extent.as_static())
        start = self.eval_expr(st.start)
        step = self.eval_expr(st.step) if hasattr(st.step, "free_vars") else Val("s32", str(st.step), True)
        bound = self.eval_expr(st.end) if st.end is not None else Val("s32", str(extent), True)
        old = self.env.get(axis)
        var = self.newr()
        self.emit(f"mov.s32 {var}, {start.text};")
        self.env[axis] = Val("s32", var)
        top = self.label()
        end = self.label()
        self.put(top)
        p = self.alloc("pred")
        self.emit(f"setp.ge.s32 {p.text}, {var}, {bound.text};")
        self.emit(f"@{p.text} bra {end};")
        self.emit_body(st.body)
        self.emit(f"add.s32 {var}, {var}, {step.text};")
        self.emit(f"bra {top};")
        self.put(end)
        if old is None:
            self.env.pop(axis, None)
        else:
            self.env[axis] = old

    def stmt_cond(self, st: Cond) -> None:
        cond = self.as_pred(self.eval_expr(st.cond))
        other = self.label()
        end = self.label()
        self.emit(f"@!{cond.text} bra {other};")
        self.emit_body(st.body)
        if st.else_body:
            self.emit(f"bra {end};")
            self.put(other)
            self.emit_body(st.else_body)
            self.put(end)
        else:
            self.put(other)

    def stmt_sync(self, st: Sync) -> None:
        if st.warp:
            self.emit("bar.warp.sync 0xffffffff;")
        elif st.barrier_id == 0:
            self.emit("bar.sync 0;")
        else:
            if st.count is None:
                self.emit(f"bar.sync {int(st.barrier_id)};")
            else:
                self.emit(f"bar.sync {int(st.barrier_id)}, {int(st.count)};")

    def stmt_warp_shuffle(self, st: WarpShuffle) -> None:
        mask = self.newr()
        self.emit(f"activemask.b32 {mask};")
        originals = {state: self.get_val(state) for state in st.state}
        s = int(st.length) // 2
        while s > 0:
            for state, state_b in zip(st.state, st.state_b):
                v = self.get_val(state)
                if v.kind == "f32":
                    raw = self.newr()
                    self.emit(f"mov.b32 {raw}, {v.text};")
                elif v.kind in ("s32", "u32"):
                    raw = self.newr()
                    self.emit(f"mov.b32 {raw}, {v.text};")
                elif v.kind == "f16":
                    raw = self.newr()
                    self.emit(f"cvt.u32.u16 {raw}, {v.text};")
                else:
                    raise NotImplementedError(f"WarpShuffle does not support {v.kind} state")
                sh = self.newr()
                self.emit(f"shfl.sync.bfly.b32 {sh}, {raw}, {s}, 0x1f, {mask};")
                if v.kind == "f16":
                    fb = self.newh()
                    self.emit(f"cvt.u16.u32 {fb}, {sh};")
                    self.env[state_b] = Val("f16", fb)
                elif v.kind in ("s32", "u32"):
                    fb = self.newr()
                    self.emit(f"mov.b32 {fb}, {sh};")
                    self.env[state_b] = Val("s32", fb)
                else:
                    fb = self.newf()
                    self.emit(f"mov.b32 {fb}, {sh};")
                    self.env[state_b] = Val("f32", fb)
            for assign in st.combine_states:
                self.stmt(assign)
            s //= 2
        # Keep the reduced state in the same physical registers so that a
        # WarpShuffle inside a runtime loop carries correctly across iterations.
        for state, reg in originals.items():
            self.mov(reg, self.coerce(self.env[state], reg.kind))
            self.env[state] = reg

    def stmt_tree_halve(self, st: TreeHalve) -> None:
        if st.inner is not None:
            raise NotImplementedError("TreeHalve(inner=...) is not supported")
        t = self.get_val(st.tid_var)
        length = int(st.length)
        if st.barrier_id == 0:
            barrier = "bar.sync 0;"
        elif st.barrier_count is None:
            barrier = f"bar.sync {int(st.barrier_id)};"
        else:
            barrier = f"bar.sync {int(st.barrier_id)}, {int(st.barrier_count)};"
        originals = {state: self.get_val(state) for state in st.state}
        s = length // 2
        while s > 0:
            p = self.alloc("pred")
            self.emit(f"setp.lt.s32 {p.text}, {t.text}, {s};")
            skip = self.label()
            self.emit(f"@!{p.text} bra {skip};")
            for buf_name, state, state_b in zip(st.bufs, st.state, st.state_b):
                buf_kind = self.smem[buf_name].kind
                off = self._offset(t, 0)
                addr = self.smem_addr(buf_name, off)
                loaded = self.alloc(buf_kind)
                self.emit(f"ld.shared.{self._ptx_type(buf_kind)} {loaded.text}, {addr};")
                self.env[state] = loaded
                off_b = self._offset(t, s)
                addr_b = self.smem_addr(buf_name, off_b)
                loaded_b = self.alloc(buf_kind)
                self.emit(f"ld.shared.{self._ptx_type(buf_kind)} {loaded_b.text}, {addr_b};")
                self.env[state_b] = loaded_b
            for assign in st.combine_states:
                self.stmt(assign)
            for buf_name, state in zip(st.bufs, st.state):
                off = self._offset(t, 0)
                addr = self.smem_addr(buf_name, off)
                v = self.coerce(self.get_val(state), self.smem[buf_name].kind)
                self.emit(f"st.shared.{self._ptx_type(v.kind)} {addr}, {v.text};")
            self.put(skip)
            self.emit(barrier)
            s //= 2
        for buf_name, state in zip(st.bufs, st.state):
            addr = self.smem_addr(buf_name, Val("s32", "0", True))
            kind = self.smem[buf_name].kind
            loaded = self.alloc(kind)
            self.emit(f"ld.shared.{self._ptx_type(kind)} {loaded.text}, {addr};")
            reg = originals[state]
            self.mov(reg, self.coerce(loaded, reg.kind))
            self.env[state] = reg

    def _offset(self, base: Val, add: int) -> Val:
        if add == 0:
            return base
        return self._arith("add", [base, Val("s32", str(add), True)])

    @staticmethod
    def _ptx_type(kind: str) -> str:
        return {"f32": "f32", "f64": "f64", "f16": "b16", "s32": "s32", "s64": "s64"}[kind]

    def stmt_cp_async(self, st: CpAsyncCopy) -> None:
        buf = self.smem.get(st.smem)
        if buf is None:
            raise KeyError(f"cp.async destination {st.smem!r} is not a Smem buffer")
        dst_flat = self.flatten(st.smem_index, buf.shape)
        if st.lane_index is not None:
            cols = int(buf.shape[-1]) if buf.shape else 1
            lane = self.eval_expr(st.lane_index[0])
            col = self.eval_expr(st.lane_index[1])
            lane_flat = self._arith("add", [self._arith("multiply", [lane, Val("s32", str(cols), True)]), col])
            dst_flat = self._arith("add", [dst_flat, lane_flat])
        dst_flat = self.apply_swizzle(dst_flat, st.swizzle)
        dst_addr = self.smem_addr(st.smem, dst_flat)
        src = self.bufs.get(st.src)
        if src is None:
            raise KeyError(f"cp.async source {st.src!r} is not a global buffer")
        src_flat = self.flatten(st.src_index, src.shape)
        src_addr = self.global_addr(st.src, src_flat)
        nbytes = int(st.nbytes)
        qual = "cg" if nbytes == 16 else "ca"
        self.emit(f"cp.async.{qual}.shared.global {dst_addr}, {src_addr}, {nbytes};")

    def stmt_select(self, st: Select) -> None:
        branches = list(st.branches)
        last = self.get_val(branches[-1].value)
        kind = last.kind
        for b in branches[:-1]:
            v = self.get_val(b.value)
            if v.kind in ("f32", "f64") or kind in ("f32", "f64"):
                kind = "f64" if "f64" in (v.kind, kind) else "f32"
            else:
                kind = "s32"
        res = self.coerce(last, kind)
        for b in reversed(branches[:-1]):
            cond = self.as_pred(self.eval_expr(b.select))
            v = self.coerce(self.get_val(b.value), kind)
            d = self.alloc(kind)
            self.emit(f"selp.{kind} {d.text}, {v.text}, {res.text}, {cond.text};")
            res = d
        self.env[st.name] = res

    def stmt_zero_prologue(self, st: ZeroPrologue) -> None:
        skip = self.label()
        p = self.alloc("pred")
        self.emit(f"setp.ne.s32 {p.text}, {self.cta}, 0;")
        self.emit(f"@{p.text} bra {skip};")
        i = self.newr()
        top = self.label()
        end = self.label()
        self.emit(f"mov.s32 {i}, {self.tid};")
        self.put(top)
        self.emit(f"setp.ge.s32 {p.text}, {i}, {int(st.words)};")
        self.emit(f"@{p.text} bra {end};")
        # ZeroPrologue counts raw 32-bit words, independent of the buffer dtype.
        base = self.gbase[st.dst]
        off = self.newrd()
        addr = self.newrd()
        self.emit(f"mul.wide.s32 {off}, {i}, 4;")
        self.emit(f"add.s64 {addr}, {base}, {off};")
        self.emit(f"st.global.u32 [{addr}], 0;")
        self.emit(f"add.s32 {i}, {i}, {self.ntid};")
        self.emit(f"bra {top};")
        self.put(end)
        self.put(skip)

    # -- misc ---------------------------------------------------------------

    def get_val(self, name: str) -> Val:
        if name in self.env:
            return self.env[name]
        raise KeyError(f"undefined SSA value {name!r}")


# ---------------------------------------------------------------------------


def compile_kernel(spec: KernelSpec) -> str:
    return Compiler(spec).run()


def compile_file(path) -> str:
    return compile_kernel(load_kernel(path))
