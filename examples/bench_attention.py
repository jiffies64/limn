"""Benchmark fused attention against its composed form on the cuda device.

Each case captures one call and times replays, so the numbers are the kernels alone: no
graph building, no plan lookup. Run it after changing the kernel and write the numbers
down wherever the change is described; the tree keeps none of them.
"""

import time

import numpy as np

from limn import Tensor, capture, set_device
from limn.tensor import composed_attention

HEADS, HD = 6, 32
CASES = {256: 50, 1024: 20, 4096: 5}  # sequence length -> timed replays
TRIALS = 3  # the GPU may be contended; the min of a few trials is the stable number


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


def fused_step(q, k, v):
    return q.attention(k, v, causal=True).realize()


def composed_step(q, k, v):
    return composed_attention(q, k, v, causal=True, scale=HD**-0.5).realize()


def main() -> None:
    set_device("cuda")
    rng = np.random.default_rng(0)
    for t, reps in CASES.items():
        shape = (HEADS, t, HD)
        q = Tensor(rng.standard_normal(shape).astype(np.float32))
        k = Tensor(rng.standard_normal(shape).astype(np.float32))
        v = Tensor(rng.standard_normal(shape).astype(np.float32))
        fused = time_step(fused_step, (q, k, v), reps)
        composed = time_step(composed_step, (q, k, v), reps)
        print(f"T={t:5d}  fused {fused:9.3f} ms   composed {composed:9.3f} ms   x{composed / fused:6.2f}")


if __name__ == "__main__":
    main()
