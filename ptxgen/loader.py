"""Load a Kernel IR json plus its sibling meta.json into a KernelSpec."""

from __future__ import annotations

import json
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel import KernelOp

from .model import ArgSpec, KernelSpec


def _dim_int(dim) -> int:
    if hasattr(dim, "as_static"):
        return int(dim.as_static())
    return int(dim)


def _shape_of(tensor) -> tuple[int, ...]:
    return tuple(_dim_int(d) for d in tensor.shape)


def _graph_tensor(g: Graph, name: str):
    for node in g.nodes.values():
        for t in node.outputs:
            if t.name == name:
                return t
    return None


def load_kernel(path: str | Path) -> KernelSpec:
    path = Path(path)
    data = json.loads(path.read_text())
    graph = Graph.from_dict(data)

    kernels = [n.op for n in graph.nodes.values() if isinstance(n.op, KernelOp)]
    if not kernels:
        raise ValueError(f"{path}: no KernelOp node in the graph")
    op = kernels[0]
    entry = op.name or ""

    meta = None
    kmeta = None
    meta_path = path.parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        candidates = meta.get("kernels", [])
        for k in candidates:
            if k.get("entry") == entry:
                kmeta = k
                break
        if kmeta is None and len(candidates) == 1:
            kmeta = candidates[0]

    args: dict[str, ArgSpec] = {}
    arg_order: list[str] = []

    if kmeta is not None:
        arg_order = list(kmeta.get("arg_order") or [])
        for a in kmeta.get("args", []):
            name = a["name"]
            args[name] = ArgSpec(
                name=name,
                dtype=a.get("dtype", "f32"),
                shape=tuple(int(x) for x in (a.get("shape") or ())),
                scalar=bool(a.get("scalar", False)),
                value=a.get("value", (a.get("data") or [None])[0] if a.get("data") else None),
                output=bool(a.get("output", False)),
            )
        grid = tuple(int(x) for x in (kmeta.get("grid") or (1, 1, 1)))
        block = tuple(int(x) for x in (kmeta.get("block") or (1, 1, 1)))
        smem_bytes = int(kmeta.get("smem_bytes") or 0)
        zero_outputs = list(kmeta.get("zero_outputs") or [])
    else:
        # No meta.json: derive the signature from the graph. The IR node's inputs
        # followed by outputs is Emmy's natural order (the shipped set happens to
        # agree except where the pipeline reordered them, which is exactly why
        # meta.json is the contract).
        arg_order = list(op.inputs) + list(op.outputs)
        for name in arg_order:
            t = _graph_tensor(graph, name)
            if t is None:
                args[name] = ArgSpec(name=name, dtype="f32", shape=())
            else:
                args[name] = ArgSpec(
                    name=name,
                    dtype=t.dtype.name,
                    shape=_shape_of(t),
                    output=name in op.outputs,
                )
        grid, block, smem_bytes, zero_outputs = (1, 1, 1), (1, 1, 1), 0, []

    if not entry:
        raise ValueError(f"{path}: KernelOp has no entry name")
    if not arg_order:
        arg_order = list(args)
    # Defensively fill argument metadata for names that appear in arg_order but
    # not in meta's args list (ZeroPrologue destinations are the known case).
    for name in arg_order:
        if name not in args:
            t = _graph_tensor(graph, name)
            if t is not None:
                args[name] = ArgSpec(name=name, dtype=t.dtype.name, shape=_shape_of(t),
                                     output=name in op.outputs)
            else:
                args[name] = ArgSpec(name=name, dtype="f32", shape=(),
                                     output=name in op.outputs)

    return KernelSpec(
        op=op,
        entry=entry,
        arg_order=arg_order,
        args=args,
        grid=grid,
        block=block,
        smem_bytes=smem_bytes,
        zero_outputs=zero_outputs,
    )
