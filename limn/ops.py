"""Graph primitives: the closed op set and the Node every lazy Tensor wraps.

A limn computation is a DAG of Nodes. Tensor methods only ever build Nodes; nothing in the
frontend computes. A device (device.py) walks the DAG when .realize() is called. This op set
is closed on purpose: everything user-facing (sub, div, matmul, softmax, ...) is composed
from these in tensor.py, so a backend only ever has to implement what is listed here.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


@dataclass(frozen=True)
class DType:
    name: str
    itemsize: int

    def __repr__(self) -> str:
        return self.name


float64 = DType("float64", 8)  # a working precision above float32: every backend computes it natively, at its own width
float32 = DType("float32", 4)
float16 = DType("float16", 2)  # a storage width: half the bytes to move, and every backend widens it to compute
int32 = DType("int32", 4)
int16 = DType("int16", 2)  # the narrow ints are exact storage: arithmetic wraps modulo 2**width, the same on every device
int8 = DType("int8", 1)
DTYPES = (float64, float32, float16, int32, int16, int8)
FLOATS = (float64, float32, float16)
INTS = (int32, int16, int8)


def promote(a: DType, b: DType) -> DType:
    """The dtype an op on these two produces. Within a family the wider side wins; an int meeting
    a float joins the floats, at float32 or wider, since ints never carry gradients and an int32
    does not fit in a float16."""
    if a == b:
        return a
    if (a in FLOATS) == (b in FLOATS):
        return max(a, b, key=lambda dtype: dtype.itemsize)
    wider = a if a in FLOATS else b
    return max(wider, float32, key=lambda dtype: dtype.itemsize)


def accumulate_in(dtype: DType) -> DType:
    """What a reduce keeps its running total in: wider than float16, the dtype itself otherwise.

    A float16 total loses its low bits within a few hundred elements. The answer rounds back to
    float16 once at the end, and every device owes this rule, the numpy reference included.
    The int dtypes stay at their own width: addition wraps modulo 2**width, so a wider total
    truncated at the end would land on the same bits anyway.
    """
    return float32 if dtype == float16 else dtype


class Op(Enum):
    # sources
    BUFFER = auto()  # realized bytes on a device; arg is the device buffer
    CONST = auto()  # python scalar, shape (); arg is the value
    # movement: presents its single src through a different layout; arg is a View
    VIEW = auto()
    # elementwise unary
    NEG = auto()
    EXP = auto()
    LOG = auto()
    SQRT = auto()
    RECIP = auto()
    CAST = auto()  # arg is the target DType
    # elementwise binary (srcs already share shape and dtype; tensor.py broadcasts first)
    ADD = auto()
    MUL = auto()
    CMPLT = auto()  # a < b as 0/1 in the src dtype; not differentiable
    # elementwise ternary
    WHERE = auto()  # (cond, a, b): a where cond != 0 else b
    # reduce; arg is a tuple of axes, reduced dims are kept as size 1
    SUM = auto()
    MAX = auto()
    # indexed row access on a 2D table, by an int32 index per row; the only data-dependent addressing
    GATHER = auto()  # (table, indices): table's rows picked by indices, shape indices.shape + table.shape[1:]
    SCATTER = auto()  # (indices, values): values summed into the named rows of a zeroed table; GATHER's gradient
    # barriers
    CONTIGUOUS = auto()  # copy src into canonical row-major layout
    ASSIGN = auto()  # (target BUFFER node, value): overwrite target's bytes with value


@dataclass(eq=False)  # eq=False: nodes hash by identity, so a BUFFER node stays itself as its bytes change
class Node:
    op: Op
    srcs: tuple[Node, ...]
    dtype: DType
    shape: tuple[int, ...]
    arg: Any = None

    def __repr__(self) -> str:
        return f"Node({self.op.name}, shape={self.shape}, dtype={self.dtype})"


def topological[T](sinks: Sequence[T], srcs: Callable[[T], Sequence[T]] = operator.attrgetter("srcs")) -> list[T]:
    """Everything reachable from sinks, sources first; srcs is the edge to walk, Node.srcs unless
    told otherwise (backward() walks Tensor parents with it). Iterative so deep graphs can't blow
    the stack."""
    order: list[T] = []
    visited: set[T] = set()
    stack: list[tuple[T, bool]] = [(n, False) for n in reversed(sinks)]
    while stack:
        node, children_done = stack.pop()
        if children_done:
            order.append(node)
        elif node not in visited:
            visited.add(node)
            stack.append((node, True))
            stack.extend((src, False) for src in reversed(srcs(node)))
    return order
