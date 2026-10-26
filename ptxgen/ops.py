"""Elementwise op metadata: canonical names, arities, and reduce identities."""

from __future__ import annotations

# Emmy spells a handful of ops one way and numpy another.
ALIASES = {
    "sum": "add",
    "prod": "multiply",
    "abs": "absolute",
    "neg": "negative",
    "pad": "copy",
    "positive": "copy",
    "pow": "power",
    "clamp": "clip",
    "__rshift__": "shr",
    "bitwise_right_shift": "shr",
    "right_shift": "shr",
    # C-style operators from BinaryExpr
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide",
    "//": "floor_divide",
    "%": "remainder",
    "&": "bitand",
    "|": "bitor",
    "^": "bitxor",
    "<<": "shl",
    ">>": "shr",
}

# Neutral element used to seed an Accum / reduce.
IDENTITY = {
    "add": 0.0,
    "sum": 0.0,
    "multiply": 1.0,
    "prod": 1.0,
    "maximum": -1.0e30,
    "amax": -1.0e30,
    "minimum": 1.0e30,
}

BINARY = {
    "add", "subtract", "multiply", "divide", "maximum", "minimum",
    "power", "fma", "where", "clip", "floor_divide", "remainder",
    "bitand", "bitor", "bitxor", "shl", "shr", "fmod",
    "logical_and", "logical_or",
}

UNARY = {
    "copy", "negative", "absolute", "relu", "exp", "exp_fast", "expm1", "log",
    "log1p", "log2", "log10", "sqrt", "rsqrt", "reciprocal", "tanh", "sigmoid",
    "silu", "softplus", "gelu", "gelu_tanh", "erf", "floor", "ceil",
    "trunc", "round", "sin", "cos", "sinh", "cosh", "bitcast", "square",
    "sign", "logical_not", "isnan", "isinf",
}


def canon(name: str) -> str:
    return ALIASES.get(name, name)


def identity(name: str) -> float:
    name = canon(name)
    if name not in IDENTITY:
        raise NotImplementedError(f"reduce op {name!r} has no known identity")
    return IDENTITY[name]
