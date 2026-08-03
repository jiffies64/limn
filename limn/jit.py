"""The executor shared by every backend that compiles the scheduled IR.

Scheduling, lowering and compiling cost more than running the kernels do, so execution is
planned once per graph *structure* and cached: graph_key hashes each node's op, dtype, shape
and arg plus the wiring between them, and two graphs with the same key run the same compiled
program against different buffers. A training loop rebuilds an identical graph every step;
the first step pays for everything, every later step goes straight to the kernel calls.

execute() owes the transaction rule from device.py: every sink is computed before any ASSIGN
commits its write. Each call gets a freshly allocated output that no kernel also reads, which
is what lets a renderer mark the output pointer restrict, and what lets an assign compute its
value before the target is touched.

A backend provides the rest: runners() turns lowered nests into callables, out_alloc and
commit move bytes on its memory, prepare() admits buffers that live elsewhere (a host array
handed to a device backend), finish() is its end-of-batch barrier.

Plans, and the compiled programs behind them, live for the process and are never evicted:
the intended workload is fixed shapes, where the caches converge after the first step. A
workload with unboundedly many distinct graph structures (say, ragged sequence lengths)
grows them without bound; bucket or pad shapes instead.

limn.capture sits on top: plan caching removes the compile from steps after the first, but
every step still rebuilds its graph in Python and walks it to find the cached plan. A
captured step skips that too, replaying the recorded kernel calls against fresh argument
buffers. This module only exposes the hook (record) and the transaction (run()) capture
drives.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from limn.codegen import LoopNest, lower_all
from limn.device import Buffer
from limn.ops import Node, Op, topological
from limn.schedule import realized, schedule

Runner = Callable[[list[Buffer], Buffer], None]


def nbytes(node: Node) -> int:
    return node.dtype.itemsize * math.prod(node.shape)


def graph_key(order: list[Node], position: dict[Node, int], sinks: list[Node]) -> tuple:
    """This graph's structure, as a hashable key: two graphs with the same key lower to the same code.

    Plans are cached on what lowering depends on: each node's op, dtype, shape and arg, the
    wiring between them as positions in topological order, and which positions are the sinks.
    A BUFFER's bytes are the one thing lowering never looks at, only its shape and dtype, so
    its arg is left out; that is what lets the same plan serve every step of a training loop.
    A CONST's arg stays in, because it is baked into the source as a literal.
    """
    at = position.__getitem__
    parts = tuple(
        (node.op, node.dtype, node.shape, None if node.op is Op.BUFFER else node.arg, tuple(map(at, node.srcs))) for node in order
    )
    return parts, tuple(map(at, sinks))


@dataclass(frozen=True, slots=True)
class Call:
    """One compiled kernel invocation, with its buffers named by position in topological order."""

    fn: Runner
    inputs: tuple[int, ...]
    output: int  # the position whose value this call produces (the ASSIGN node itself for assigns)
    out_nbytes: int
    zero_fill: bool  # a scatter adds into its output and reaches only the rows its indices name
    assign_target: int | None  # the BUFFER position an assign commits to, once every call has run


@dataclass(frozen=True, slots=True)
class Plan:
    calls: tuple[Call, ...]
    buffers: tuple[int, ...]  # the BUFFER positions, whose bytes come from this step's graph
    sinks: tuple[int, ...]  # realized() resolved to positions, so alias chains are walked once
    positions: int  # how many nodes the graph had: the size of a run's buffer table


@dataclass(frozen=True, slots=True)
class Recording:
    """One execute() as a replay needs it: the plan and the live buffers.

    `bound` holds each BUFFER position's bytes as the node carried them, before prepare() may
    have migrated a copy; that is the same buffer a commit would target. `outs` is what the
    call handed back for its sinks, which is how capture ties a returned tensor to a position.
    """

    plan: Plan
    bound: tuple[Buffer, ...]  # one per plan.buffers entry
    outs: tuple[Buffer, ...]  # one per plan.sinks entry


class CompiledDevice:
    """Plan-cached execution over backend hooks; alloc/copyin/copyout still come from Device."""

    def __init__(self) -> None:
        self.plans: dict[tuple, Plan] = {}
        self.record: list[Recording] | None = None  # set by capture while it observes a call

    # ---- what a backend implements ----

    def runners(self, nests: list[LoopNest]) -> list[Runner]:
        """Compile these nests, always at least one, and return one callable per nest, in order."""
        raise NotImplementedError

    def out_alloc(self, nb: int, zero: bool) -> Buffer:
        """A fresh output buffer, zeroed when asked for."""
        raise NotImplementedError

    def commit(self, target: Buffer, value: Buffer) -> None:
        """Overwrite an assign's target buffer with the computed value, after the whole batch ran."""
        raise NotImplementedError

    def has_custom(self, name: str) -> bool:
        """Whether this device supplies a kernel for the named CUSTOM op; the frontend falls
        back to composing the op from primitives when it does not."""
        return False

    def custom_runner(self, node: Node) -> Runner:
        """A callable for a CUSTOM kernel, which never reaches the loop-nest IR. Backends that
        register the node's op name supply it; the rest refuse the node at plan time."""
        raise NotImplementedError(f"{type(self).__name__} has no kernel for custom op {node.arg[0]!r}")

    def prepare(self, buf: Buffer) -> Buffer:
        """Admit a BUFFER node's bytes; a backend whose memory is elsewhere migrates them here."""
        return buf

    def finish(self) -> None:
        """Block until every launched kernel is done; execute() calls it once per batch."""

    # ---- shared execution ----

    def execute(self, sinks: list[Node]) -> list[Buffer]:
        order = topological(sinks)
        position = {node: p for p, node in enumerate(order)}
        key = graph_key(order, position, sinks)
        plan = self.plans.get(key)
        if plan is None:
            plan = self.plans[key] = self.plan_of(order, position, sinks)
        sources = {p: order[p].arg for p in plan.buffers}
        outs = self.run(plan, sources)
        if self.record is not None:
            self.record.append(Recording(plan, tuple(sources.values()), tuple(outs)))
        return outs

    def run(self, plan: Plan, sources: Mapping[int, Buffer]) -> list[Buffer]:
        """One transaction over a plan: every kernel, then the deferred commits, then the barrier.

        sources holds each BUFFER position's bytes as its node carries them. The commit goes to
        that buffer (prepare() may have handed the kernels a migrated copy), and the position is
        repointed at the fresh value so sinks read what was just written rather than a copy the
        commit never touched. execute() and capture.replay both come through here, so the
        transaction rule has one owner.
        """
        bufs: list[Buffer] = [None] * plan.positions
        for p, buf in sources.items():
            bufs[p] = self.prepare(buf)
        deferred: list[tuple[int, Buffer]] = []
        for call in plan.calls:
            out = self.out_alloc(call.out_nbytes, call.zero_fill)
            call.fn([bufs[p] for p in call.inputs], out)
            bufs[call.output] = out
            if call.assign_target is not None:
                deferred.append((call.assign_target, out))
        for p, value in deferred:
            self.commit(sources[p], value)
            bufs[p] = value
        self.finish()
        return [bufs[p] for p in plan.sinks]

    def plan_of(self, order: list[Node], position: dict[Node, int], sinks: list[Node]) -> Plan:
        """Lower, compile and wire up these sinks' kernels by position instead of by node.

        A CUSTOM kernel is handed to the device whole and never touches the loop-nest IR; the
        rest lower and compile together. Calls come back in kernel order either way, so a
        custom kernel sits in the plan exactly where the schedule put it.
        """
        kernels = schedule(sinks, order)
        nests = lower_all(sinks, order, kernels)
        fns = iter(self.runners(nests) if nests else [])  # a graph of pure views schedules no kernels
        calls = tuple(
            Call(
                fn=self.custom_runner(kernel.ast) if kernel.ast.op is Op.CUSTOM else next(fns),
                inputs=tuple(position[node] for node in kernel.inputs),
                output=position[kernel.ast],
                out_nbytes=nbytes(kernel.target),
                zero_fill=kernel.ast.op is Op.SCATTER,
                assign_target=position[kernel.target] if kernel.ast.op is Op.ASSIGN else None,
            )
            for kernel in kernels
        )
        return Plan(
            calls,
            tuple(p for p, node in enumerate(order) if node.op is Op.BUFFER),
            tuple(position[realized(sink)] for sink in sinks),
            len(order),
        )
