"""Kernel-level model: arguments, buffers, and static/dynamic shared memory."""

from __future__ import annotations

from dataclasses import dataclass, field


DTYPE = {
    # emmy name -> (value kind, ptx scalar type, element bytes)
    "f32": ("f32", "f32", 4),
    "f64": ("f64", "f64", 8),
    "f16": ("f16", "b16", 2),
    "i32": ("s32", "s32", 4),
    "u32": ("s32", "u32", 4),
    "i64": ("s64", "s64", 8),
    "u64": ("s64", "u64", 8),
    "bool": ("pred", "pred", 1),
    # CUDA spellings that can reach shared-memory declarations
    "float": ("f32", "f32", 4),
    "double": ("f64", "f64", 8),
    "half": ("f16", "b16", 2),
    "__half": ("f16", "b16", 2),
    "int": ("s32", "s32", 4),
    "long": ("s64", "s64", 8),
    "unsigned long long": ("u64", "u64", 8),
    "float32": ("f32", "f32", 4),
    "float64": ("f64", "f64", 8),
    "float16": ("f16", "b16", 2),
    "int32": ("s32", "s32", 4),
    "int64": ("s64", "s64", 8),
}


def dtype_info(name: str):
    key = str(name)
    if key not in DTYPE:
        raise NotImplementedError(f"unsupported dtype {key!r}")
    return DTYPE[key]


@dataclass
class ArgSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    scalar: bool = False
    value: float | int | None = None
    output: bool = False


@dataclass
class Buffer:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def elem_bytes(self) -> int:
        return dtype_info(self.dtype)[2]

    @property
    def kind(self) -> str:
        return dtype_info(self.dtype)[0]


@dataclass
class SmemBuf(Buffer):
    align: int = 0
    symbol: str = ""
    offset: int = 0


@dataclass
class KernelSpec:
    op: object                      # emmy KernelOp
    entry: str
    arg_order: list[str]
    args: dict[str, ArgSpec]
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    smem_bytes: int = 0
    zero_outputs: list[str] = field(default_factory=list)

    def buffer(self, name: str) -> Buffer:
        arg = self.args.get(name)
        if arg is None:
            raise KeyError(f"buffer {name!r} is not in meta.json arg_order")
        return Buffer(name=arg.name, dtype=arg.dtype, shape=arg.shape)
