"""Scheduler: cut the op DAG into kernels.

A kernel is one loop nest that writes one buffer. Elementwise work fuses into whatever kernel
consumes it; a short list of nodes has to land in a buffer of its own, and those are the cuts
between kernels:

    CONTIGUOUS, ASSIGN   a copy and an in-place write already mean "put this in a buffer"
    SUM, MAX             a reduce stores what it accumulated
    GATHER, SCATTER      a table read at a computed row, and the write that undoes it
    CUSTOM               the device supplies the whole kernel; nothing schedules inside it
    an addressed src     a VIEW's buffer and a GATHER's table are indexed, so they have to be bytes
    every sink           the caller asked for these bytes, unless it aliases a buffer already

BUFFER nodes are already realized, so they are kernel inputs rather than kernels. CONST never
gets a buffer: it lowers to a literal inside each kernel that uses it.

Not every VIEW is work. One that reads its source straight through, row-major and unmasked and
all of it, is a reshape that moves no data, so it can share the buffer underneath rather than
copy into a new one: realized() peels those off, and a sink that is one costs no kernel at all.
That is how a caller finds a sink's bytes, and it is the last step of matmul, of any reduce with
keepdim=False, and of reshaping something already realized.

Two things still cost more than they look like they should. An elementwise node feeding two
kernels is recomputed in both, the usual trade here (recompute beats a round trip to memory, but
a widely shared subtree gets emitted more than once). And a VIEW that does move data, or one
whose source is still an expression, forces that source into a buffer: fusing through it would
mean re-expressing every view underneath in the consumer's shape, and falling back to a cut
whenever View.reshape says the layout cannot survive it.

A VIEW that only expands could be seen through, since the source's dims all land on loop variables
the consumer already has. It is cut anyway: the source would be re-evaluated once per point of the
expanded dims, and a softmax feeding a matmul that way spends more on repeated exp calls than its
buffer costs in bandwidth.

A schedule is a plan, and ordering it is not the whole story for ASSIGN: an executor still owes
the transaction rule from device.py, where every sink is computed before any assign commits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from limn.ops import Node, Op, topological

CUT_OPS = (Op.CONTIGUOUS, Op.ASSIGN, Op.SUM, Op.MAX, Op.GATHER, Op.SCATTER, Op.CUSTOM)


@dataclass(frozen=True)
class Kernel:
    """One loop nest's worth of work: `ast` is what it computes, over `inputs`, into `target`'s buffer."""

    ast: Node
    body: tuple[Node, ...]  # nodes computed inside the kernel, sources first
    inputs: tuple[Node, ...]  # realized nodes it loads from, in load order

    @property
    def target(self) -> Node:
        """The node whose buffer this kernel writes: an ASSIGN writes its target, everything else itself."""
        return self.ast.srcs[0] if self.ast.op is Op.ASSIGN else self.ast


def addressed(node: Node) -> Node | None:
    """The src this node indexes into rather than reads: a VIEW's buffer, a GATHER's table.

    Addressing one means its bytes have to be laid out, so it is realized and loaded from rather
    than fused. An ASSIGN's first src is not one of these: it is written, not addressed.
    """
    return node.srcs[0] if node.op in (Op.VIEW, Op.GATHER) else None


def reads(node: Node) -> tuple[Node, ...]:
    """The srcs whose *value* this node consumes.

    What a node addresses is not read, and an ASSIGN's first src is where it stores, so neither
    counts. Both still show up as inputs; they just are not fusable. Dropping the first src covers
    every case, a VIEW having only the one.
    """
    if node.op in (Op.VIEW, Op.ASSIGN, Op.GATHER):
        return node.srcs[1:]
    return node.srcs


def is_alias(node: Node) -> bool:
    """True when this VIEW only reinterprets its source's buffer: the same bytes, in the same order."""
    return node.op is Op.VIEW and node.arg.is_contiguous and node.arg.numel == math.prod(node.srcs[0].shape)


def realized(node: Node) -> Node:
    """The node whose buffer holds this node's bytes: itself, unless it aliases someone else's.

    An ASSIGN resolves the same way, to the buffer it overwrites. It still gets a kernel of its
    own (it is in CUT_OPS); this only says where to read the result afterwards.
    """
    while is_alias(node) or node.op is Op.ASSIGN:
        node = node.srcs[0]
    return node


def boundaries(sinks: list[Node], order: list[Node]) -> set[Node]:
    """Every node that has to end up in a buffer of its own; order is topological(sinks)."""
    cuts: set[Node] = set()
    for node in order:
        if node.op in CUT_OPS:
            cuts.add(node)
        if (src := addressed(node)) is not None and src.op not in (Op.BUFFER, Op.CONST):
            cuts.add(src)
    cuts.update(home for sink in sinks if (home := realized(sink)).op is not Op.BUFFER)
    return cuts


def fuse(root: Node, cuts: set[Node]) -> Kernel:
    """Walk down from a cut until the next cut, splitting the kernel into computed nodes and inputs."""
    body: list[Node] = []
    inputs: list[Node] = []
    seen: set[Node] = set()
    stack: list[tuple[Node, bool]] = [(root, False)]
    while stack:
        node, children_done = stack.pop()
        if children_done:
            body.append(node)
        elif node not in seen:
            seen.add(node)
            if node is not root and (node in cuts or node.op is Op.BUFFER):
                inputs.append(node)
            else:
                stack.append((node, True))
                stack.extend((src, False) for src in reversed(reads(node)))
    for node in body:  # what a node addresses is not in reads(), so the walk above never saw it
        if (src := addressed(node)) is not None and src.op is not Op.CONST and src not in seen:
            seen.add(src)
            inputs.append(src)
    return Kernel(root, tuple(body), tuple(inputs))


def schedule(sinks: list[Node], order: list[Node] | None = None) -> list[Kernel]:
    """The kernels needed to realize these sinks, in dependency order; order is topological(sinks)."""
    if order is None:
        order = topological(sinks)
    cuts = boundaries(sinks, order)
    return [fuse(node, cuts) for node in order if node in cuts]
