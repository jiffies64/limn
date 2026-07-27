"""Graph primitives: the closed op set and the Node every lazy Tensor wraps.

A limn computation is a DAG of Nodes. Tensor methods only ever build Nodes; nothing in the
frontend computes. A device (device.py) walks the DAG when .realize() is called. This op set
is closed on purpose: everything user-facing (sub, div, matmul, softmax, ...) is composed
from these in tensor.py, so a backend only ever has to implement what is listed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


@dataclass(frozen=True)
class DType:
    name: str
    itemsize: int

    def __repr__(self) -> str:
        return self.name


float32 = DType("float32", 4)
int32 = DType("int32", 4)
DTYPES = (float32, int32)


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


def topological(sinks: list[Node]) -> list[Node]:
    """All nodes reachable from sinks, sources first. Iterative so deep graphs can't blow the stack."""
    order: list[Node] = []
    visited: set[Node] = set()
    stack: list[tuple[Node, bool]] = [(n, False) for n in reversed(sinks)]
    while stack:
        node, children_done = stack.pop()
        if children_done:
            order.append(node)
        elif node not in visited:
            visited.add(node)
            stack.append((node, True))
            stack.extend((src, False) for src in reversed(node.srcs))
    return order
