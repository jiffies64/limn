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

capture sits on top: plan caching removes the compile from steps after the first, but every
step still rebuilds its graph in Python and walks it to find the cached plan. A captured
step skips that too, replaying the recorded kernel calls against fresh argument buffers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from limn import device
from limn.codegen import LoopNest, lower_all
from limn.device import Buffer
from limn.ops import Node, Op, topological
from limn.schedule import realized

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
            self.record.append(Recording(plan, tuple(sources[p] for p in plan.buffers), tuple(outs)))
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
        """Lower, compile and wire up these sinks' kernels by position instead of by node."""
        nests = lower_all(sinks, order)
        fns = self.runners(nests) if nests else []  # a graph of pure views schedules no kernels
        calls = tuple(
            Call(
                fn=fn,
                inputs=tuple(position[node] for node in nest.kernel.inputs),
                output=position[nest.kernel.ast],
                out_nbytes=nbytes(nest.kernel.target),
                zero_fill=nest.kernel.ast.op is Op.SCATTER,
                assign_target=position[nest.kernel.target] if nest.kernel.ast.op is Op.ASSIGN else None,
            )
            for nest, fn in zip(nests, fns, strict=True)
        )
        return Plan(
            calls,
            tuple(p for p, node in enumerate(order) if node.op is Op.BUFFER),
            tuple(position[realized(sink)] for sink in sinks),
            len(order),
        )


@dataclass(frozen=True, slots=True)
class Step:
    """One recorded execute(), rewired for replay: where every one of its buffers comes from."""

    plan: Plan
    args_at: tuple[tuple[int, int], ...]  # (position, index into the call's arguments)
    outs_at: tuple[tuple[int, int, int], ...]  # (position, earlier step, index into that step's outs)
    held_at: tuple[tuple[int, Buffer], ...]  # (position, the buffer captured there)


class capture:
    """Record what a step function makes the device do, then replay that without running it.

    Wraps a function whose tensor arguments are the only thing that changes call to call (the
    data batch of a training step), while everything else it touches lives in buffers updated
    in place (parameters, optimizer state). The first two calls run the function normally and
    must build the same graph; the second one is recorded. Every later call skips the function,
    and with it all graph building, autograd bookkeeping and plan lookup: the recorded kernels
    run against the new arguments' buffers, assigns commit to the same state buffers, and a
    value the function realized mid-call and computed on feeds the later kernels that replay's
    fresh bytes. Each replay returns its own tensors over its own results, so readings kept
    from different replays do not alias; a returned in-place target (a parameter) is the
    parameter itself, as it is on a plain call.

    Everything else is baked in at record time. A python value that varies between calls (a
    learning rate schedule, a step counter) freezes at its recorded value; keep such state in
    tensors the graph advances, as AdamW keeps beta**t. The function must realize whatever it
    returns (pass a loss to Optimizer.step, or call realize()), and takes only tensor
    arguments, all of the recorded shapes and dtypes.

    Because the function never runs again, whatever its last real call left behind stays put.
    Gradients are the case that matters: end a training step with zero_grad() so the recorded
    call's gradient graphs do not sit on the parameters for the life of the capture.

    On a device that interprets graphs instead of compiling plans (numpy), there is nothing to
    replay and the function simply runs every call.
    """

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        self.first: tuple | None = None  # the first call's wiring, held until the second confirms it
        self.steps: list[Step] | None = None
        self.argspec: tuple = ()
        self.result: Any = None  # the recorded return structure; replays rebuild it over their own outs
        self.result_at: dict[int, tuple[int, int]] = {}  # id of a returned tensor -> (step, out index)

    def __call__(self, *args: Any) -> Any:
        dev = device.active()
        if not isinstance(dev, CompiledDevice):
            return self.fn(*args)
        if self.steps is not None:
            return self.replay(dev, args)
        return self.observe(dev, args)

    def observe(self, dev: CompiledDevice, args: tuple) -> Any:
        from limn.tensor import Tensor

        for t in args:
            if not isinstance(t, Tensor) or t.node.op is not Op.BUFFER:
                raise ValueError("capture takes realized, buffer-backed tensors as arguments")
        if dev.record is not None:
            raise ValueError("capture does not nest")
        record: list[Recording] = []
        dev.record = record
        try:
            out = self.fn(*args)
        finally:
            dev.record = None
        if not record:
            raise ValueError("the captured function realized nothing, so there is no work to replay")
        wired, produced = self.wired(record, args)
        signature = tuple((id(rec.plan), args_at, outs_at) for rec, (args_at, outs_at, _) in zip(record, wired))
        if self.first is None:
            # the first call settles one-time work (first realizes, lazy state); the second is
            # compared against it, so a graph that changes call to call is refused, not replayed
            self.first = signature
            return out
        if signature != self.first:
            raise ValueError("the captured function built a different graph on its second call; capture needs a fixed graph")
        tensors = self.returned(out)
        self.steps = [Step(rec.plan, *wiring) for rec, wiring in zip(record, wired)]
        self.argspec = tuple((t.shape, t.dtype) for t in args)
        # a returned tensor whose buffer no recording produced is an in-place target (an
        # assign's), refreshed by every replay's commits; it is handed back as itself
        self.result_at = {id(t): spot for t in tensors if (spot := produced.get(id(t.node.arg))) is not None}
        for t in tensors:
            # replays only read shape and dtype off these; the autograd record would pin the
            # observed call's whole graph (every intermediate tensor and closure) for the
            # capture's life
            t.requires_grad, t.parents, t.grad_fn = False, (), None
        self.result = out
        return out

    def wired(self, record: list[Recording], args: tuple) -> tuple[list[tuple], dict[int, tuple[int, int]]]:
        """Where each recording's buffers come from, by identity: the call's arguments, an
        earlier recording's outs (a value the function realized mid-call and computed on), or
        bytes that only update in place and can be captured as they are. Also returns the
        finished producer map, which is how returned tensors find their outs."""
        owner: dict[int, int] = {}
        for i, t in enumerate(args):
            owner.setdefault(id(t.node.arg), i)
        produced: dict[int, tuple[int, int]] = {}
        wired = []
        for k, rec in enumerate(record):
            args_at, outs_at, held_at = [], [], []
            for p, buf in zip(rec.plan.buffers, rec.bound):
                if id(buf) in owner:
                    args_at.append((p, owner[id(buf)]))
                elif id(buf) in produced:
                    outs_at.append((p, *produced[id(buf)]))
                else:
                    held_at.append((p, buf))
            wired.append((tuple(args_at), tuple(outs_at), tuple(held_at)))
            produced.update({id(b): (k, i) for i, b in enumerate(rec.outs)})  # a later producer wins
        return wired, produced

    def returned(self, out: Any) -> list:
        """The tensors the function handed back, each realized so replays have bytes to serve."""
        from limn.tensor import Tensor

        tensors = [out] if out is not None and not isinstance(out, (tuple, list)) else list(out or [])
        for t in tensors:
            if not isinstance(t, Tensor):
                raise ValueError(f"a captured function may return tensors only, got {type(t).__name__}")
            if t.node.op is not Op.BUFFER:
                raise ValueError("a captured function must realize what it returns; pass it to Optimizer.step or realize()")
        return tensors

    def replay(self, dev: CompiledDevice, args: tuple) -> Any:
        from limn.tensor import Tensor

        assert self.steps is not None
        for t in args:
            if not isinstance(t, Tensor) or t.node.op is not Op.BUFFER:
                raise ValueError("capture takes realized, buffer-backed tensors as arguments")
        spec = tuple((t.shape, t.dtype) for t in args)
        if spec != self.argspec:
            raise ValueError(f"capture recorded arguments {self.argspec}, this call passed {spec}")
        history: list[list[Buffer]] = []
        for step in self.steps:
            sources: dict[int, Buffer] = {p: args[i].node.arg for p, i in step.args_at}
            sources.update((p, history[j][i]) for p, j, i in step.outs_at)
            sources.update(step.held_at)
            history.append(dev.run(step.plan, sources))

        def remade(t: Any) -> Any:
            spot = self.result_at.get(id(t))
            if spot is None:
                return t
            j, i = spot
            return Tensor.from_node(Node(Op.BUFFER, (), t.node.dtype, t.node.shape, history[j][i]))

        if isinstance(self.result, (tuple, list)):
            return type(self.result)(remade(t) for t in self.result)
        return remade(self.result)
