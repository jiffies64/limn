"""Benchmark fused attention against its composed form on the cuda device, forward and backward.

Each case captures one call and times replays, so the numbers are the kernels alone: no
graph building, no plan lookup. The backward table also prints the intermediates each step
has to hold, which is where the fused pair earns its keep: the composed backward builds the
whole t by t three times over, so past a point it does not run at all rather than running
slowly. Run this after changing the kernels and write the numbers down wherever the change
is described; the tree keeps none of them.
"""

import time

import numpy as np

from limn import Tensor, capture, grad, realize, set_device
from limn.jit import nbytes
from limn.schedule import schedule
from limn.tensor import composed_attention

HEADS, HD = 6, 32
CASES = {256: 20, 1024: 10, 4096: 3}  # sequence length -> timed replays
TRIALS = 3  # the GPU may be contended; the min of a few trials is the stable number
# Rows per batch, so every length runs the same number of tokens and the same number of blocks.
# Without it a short sequence would be timed on a handful of blocks, which says more about how
# empty the GPU was left than about the kernels; a transformer trades batch for context the same way.
TOKENS = 8192


def time_step(step, args, reps: int) -> float:
    """Milliseconds per call, the best of a few trials. Two runs settle the capture and one
    warms the replay before each timed pass, so contention shows up as a slower trial, not
    as a cold first call."""
    replayed = capture(step)

    def once() -> float:
        for _ in range(3):
            replayed(*args)
        start = time.perf_counter()
        for _ in range(reps):
            replayed(*args)
        return (time.perf_counter() - start) / reps * 1e3

    return min(once() for _ in range(TRIALS))


def attention(fused: bool, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return q.attention(k, v, causal=True) if fused else composed_attention(q, k, v, causal=True, scale=HD**-0.5)


def forward_step(fused: bool):
    return lambda q, k, v: attention(fused, q, k, v).realize()


def train_step(fused: bool):
    def step(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, ...]:
        out = attention(fused, q, k, v)
        grads = grad((out * out).sum(), [q, k, v])
        realize(*grads)
        return tuple(grads)

    return step


def working_bytes(fused: bool, args: tuple[Tensor, ...]) -> int:
    """What a training step's scheduled kernels write: one buffer per cut, so this is the
    intermediates it has to hold. Measured on a graph nothing has realized yet, since a
    realized sink is bytes already and schedules no kernel at all."""
    out = attention(fused, *args)
    grads = grad((out * out).sum(), list(args))
    return sum(nbytes(kernel.target) for kernel in schedule([g.node for g in grads]))


def main() -> None:
    set_device("cuda")
    rng = np.random.default_rng(0)

    def inputs(t: int, requires_grad: bool) -> tuple[Tensor, ...]:
        shape = (max(1, TOKENS // t), HEADS, t, HD)
        return tuple(Tensor(rng.standard_normal(shape).astype(np.float32), requires_grad=requires_grad) for _ in range(3))

    print(f"forward, {TOKENS} tokens per case")
    for t, reps in CASES.items():
        args = inputs(t, False)
        fused, composed = (time_step(forward_step(f), args, reps) for f in (True, False))
        print(f"B={TOKENS // t:3d} T={t:5d}  fused {fused:9.3f} ms   composed {composed:9.3f} ms   x{composed / fused:6.2f}")

    print("\nforward + backward, with the intermediates the step holds")
    for t, reps in CASES.items():
        args = inputs(t, True)
        held = tuple(working_bytes(f, args) / 2**20 for f in (True, False))
        try:
            fused, composed = (time_step(train_step(f), args, reps) for f in (True, False))
        except RuntimeError as e:  # the composed form runs out of card before it runs out of speed
            print(f"B={TOKENS // t:3d} T={t:5d}  fused {held[0]:8.1f} MB   composed {held[1]:8.1f} MB, and did not run: {e}")
            continue
        print(
            f"B={TOKENS // t:3d} T={t:5d}  fused {fused:9.3f} ms ({held[0]:8.1f} MB)"
            f"   composed {composed:9.3f} ms ({held[1]:8.1f} MB)   x{composed / fused:6.2f}"
        )


if __name__ == "__main__":
    main()
