"""Record what a training step makes the device do once, then replay it without running the step.

Plan caching (jit.py) removes the compile from steps after the first, but every step still
rebuilds its graph in Python and walks it to find the cached plan. A captured step skips that
too, replaying the recorded kernel calls against fresh argument buffers. This module is the
user-facing wrapper over the recording hook CompiledDevice exposes; the device side of the
mechanism (Recording, run()) lives in jit.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from limn import device
from limn.device import Buffer
from limn.jit import CompiledDevice, Plan, Recording
from limn.ops import Node, Op
from limn.schedule import realized
from limn.tensor import Tensor


@dataclass(frozen=True, slots=True)
class Step:
    """One recorded execute(), rewired for replay: where every one of its buffers comes from."""

    plan: Plan
    args_at: tuple[tuple[int, int], ...]  # (position, index into the call's arguments)
    outs_at: tuple[tuple[int, int, int], ...]  # (position, earlier step, index into that step's outs)
    held: dict[int, Buffer]  # position -> the buffer captured there, bytes that only update in place


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
    learning rate schedule, a step counter) freezes at its recorded value, and so does a buffer
    the function creates fresh each call; keep such state in tensors the graph advances, as
    AdamW keeps beta**t. The function must realize whatever it returns (pass a loss to
    Optimizer.step, or call realize()), and takes only tensor arguments, all of the recorded
    shapes and dtypes. The recording binds to the active device: replays refuse another one,
    since they would hand its kernels a stranger's buffers.

    Because the function never runs again, whatever its last real call left behind stays put.
    Gradients are the case that matters: end a training step with zero_grad() so the recorded
    call's gradient graphs do not sit on the parameters for the life of the capture.

    On a device that interprets graphs instead of compiling plans (numpy), there is nothing to
    replay and the function simply runs every call.
    """

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        self.dev: CompiledDevice | None = None  # the device the recording is bound to
        self.first: tuple | None = None  # the first call's wiring, held until the second confirms it
        self.steps: list[Step] | None = None
        self.argspec: tuple = ()
        self.result: Any = None  # the recorded return structure; replays rebuild it over their own outs
        self.result_at: dict[int, tuple[int, int]] = {}  # id of a returned tensor -> (step, out index)

    def __call__(self, *args: Any) -> Any:
        dev = device.active()
        if not isinstance(dev, CompiledDevice):
            return self.fn(*args)
        for t in args:
            if not isinstance(t, Tensor) or t.node.op is not Op.BUFFER:
                raise ValueError("capture takes realized, buffer-backed tensors as arguments")
        if self.steps is not None:
            return self.replay(dev, args)
        return self.observe(dev, args)

    def observe(self, dev: CompiledDevice, args: tuple) -> Any:
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
        steps, produced = self.wired(record, args)
        signature = tuple((id(step.plan), step.args_at, step.outs_at) for step in steps)
        if self.first is None:
            # the first call settles one-time work (first realizes, lazy state); the second is
            # compared against it, so a graph that changes call to call is refused, not replayed
            self.first = signature
            return out
        if signature != self.first:
            raise ValueError("the captured function built a different graph on its second call; capture needs a fixed graph")
        tensors = self.returned(out)
        # a returned tensor whose buffer no recording produced is an in-place target (an
        # assign's), refreshed by every replay's commits; it is handed back as itself
        self.result_at = {id(t): spot for t in tensors if (spot := produced.get(id(realized(t.node).arg))) is not None}
        for t in tensors:
            # severing parents and grad_fn unpins the observed call's whole graph (every
            # intermediate tensor and closure). requires_grad stays: a returned in-place
            # target is a live parameter that must keep taking gradients
            t.parents, t.grad_fn = (), None
        self.dev = dev
        self.steps = steps
        self.argspec = tuple((t.shape, t.dtype) for t in args)
        self.result = out
        return out

    def wired(self, record: list[Recording], args: tuple) -> tuple[list[Step], dict[int, tuple[int, int]]]:
        """Where each recording's buffers come from, by identity: the call's arguments, an
        earlier recording's outs (a value the function realized mid-call and computed on), or
        bytes that only update in place and can be captured as they are. Also returns the
        finished producer map, which is how returned tensors find their outs."""
        owner: dict[int, int] = {}
        for i, t in enumerate(args):
            owner.setdefault(id(t.node.arg), i)
        produced: dict[int, tuple[int, int]] = {}
        steps: list[Step] = []
        for k, rec in enumerate(record):
            args_at, outs_at, held = [], [], {}
            for p, buf in zip(rec.plan.buffers, rec.bound):
                if id(buf) in owner:
                    args_at.append((p, owner[id(buf)]))
                elif id(buf) in produced:
                    outs_at.append((p, *produced[id(buf)]))
                else:
                    held[p] = buf
            steps.append(Step(rec.plan, tuple(args_at), tuple(outs_at), held))
            produced.update({id(b): (k, i) for i, b in enumerate(rec.outs)})  # a later producer wins
        return steps, produced

    def returned(self, out: Any) -> list:
        """The tensors the function handed back, each realized so replays have bytes to serve."""
        tensors = [out] if out is not None and not isinstance(out, (tuple, list)) else list(out or [])
        for t in tensors:
            if not isinstance(t, Tensor):
                raise ValueError(f"a captured function may return tensors only, got {type(t).__name__}")
            if realized(t.node).op is not Op.BUFFER:
                raise ValueError("a captured function must realize what it returns; pass it to Optimizer.step or realize()")
        return tensors

    def replay(self, dev: CompiledDevice, args: tuple) -> Any:
        assert self.steps is not None
        if dev is not self.dev:
            raise ValueError("a capture replays only on the device that recorded it; set_device has replaced that one")
        if dev.record is not None:
            raise ValueError("capture does not nest")
        spec = tuple((t.shape, t.dtype) for t in args)
        if spec != self.argspec:
            raise ValueError(f"capture recorded arguments {self.argspec}, this call passed {spec}")
        history: list[list[Buffer]] = []
        for step in self.steps:
            sources = dict(step.held)  # one copy for the big in-place part; the few args and outs overlay it
            for p, i in step.args_at:
                sources[p] = args[i].node.arg
            for p, j, i in step.outs_at:
                sources[p] = history[j][i]
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
