"""Fused scaled dot-product attention: the numpy reference a CUSTOM "sdpa" node runs.

The math is ordinary attention, q @ k' scaled, softmax along the key axis, weighted values.
The shape of the computation is flash's: the softmax runs in blocks of keys, and each query
row carries a running max, a running denominator, and a running weighted sum, so no
t_q by t_k intermediate ever exists.

The bookkeeping is one line of algebra. When a block raises a row's running max from m to
m', everything accumulated under m was exponentiated against the wrong baseline, and the
factor exp(m - m') converts it. That leaves the recurrence equal to plain softmax up to
rounding, at O(t_q) working memory instead of O(t_q * t_k).

Running it as a script checks the recurrence against plain softmax; the registry entry the
numpy device uses is `kernel` at the bottom.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

HALF_FLOATS = (np.float16, np.dtype(ml_dtypes.bfloat16))  # storage widths: the recurrence runs wider


def sdpa(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    causal: bool,
    scale: float,
    block_size: int = 16,
) -> np.ndarray:
    """softmax(q @ k' * scale) @ v along the key axis, folded one block of keys at a time.

    The inputs share their leading dims; q is (..., t_q, hd), k is (..., t_k, hd),
    v is (..., t_k, hd_v). causal wants t_q == t_k and hides keys to the right of each
    query row. half-width float inputs widen to float32 for the recurrence and round back
    once at the end, the same rule every reduce in limn owes.
    """
    if q.ndim < 2 or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError(f"sdpa wants 2D+ inputs of equal rank, got {q.shape}, {k.shape}, {v.shape}")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2] or k.shape[-2] != v.shape[-2]:
        raise ValueError(f"sdpa shape mismatch: {q.shape}, {k.shape}, {v.shape}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"sdpa: head dims differ, {q.shape[-1]} and {k.shape[-1]}")
    if causal and q.shape[-2] != k.shape[-2]:
        raise ValueError(f"causal sdpa needs square keys, got {q.shape[-2]} queries and {k.shape[-2]} keys")

    work = np.dtype(np.float32) if q.dtype in HALF_FLOATS else q.dtype
    Q, K, V = q.astype(work, copy=False), k.astype(work, copy=False), v.astype(work, copy=False)
    batch, t_q, t_k = q.shape[:-2], q.shape[-2], k.shape[-2]

    out = np.zeros(batch + (t_q, v.shape[-1]), dtype=work)
    running_max = np.full(batch + (t_q,), -np.inf, dtype=work)
    running_sum = np.zeros(batch + (t_q,), dtype=work)
    rows = np.arange(t_q)[:, None]  # the mask's trailing dims broadcast over any batch
    for j0 in range(0, t_k, block_size):
        j1 = min(j0 + block_size, t_k)
        scores = Q @ K[..., j0:j1, :].swapaxes(-1, -2) * scale
        if causal:
            scores = np.where(np.arange(j0, j1) <= rows, scores, -np.inf)
        block_max = scores.max(axis=-1)
        new_max = np.maximum(running_max, block_max)
        rescale = np.exp(running_max - new_max)
        weights = np.exp(scores - new_max[..., None])
        running_sum = running_sum * rescale + weights.sum(axis=-1)
        out = out * rescale[..., None] + weights @ V[..., j0:j1, :]
        running_max = new_max
    return (out / running_sum[..., None]).astype(q.dtype)


def kernel(srcs: list[np.ndarray], arg: tuple) -> np.ndarray:
    """The CUSTOM ("sdpa", causal, scale) node, as the numpy device runs it."""
    _, causal, scale = arg
    return sdpa(srcs[0], srcs[1], srcs[2], causal=causal, scale=scale)


# ---- checker: run as a script ----


def _plain(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, causal: bool, scale: float) -> np.ndarray:
    """Attention the obvious way, for the checker to diff against: builds the whole t_q by t_k."""
    scores = q @ k.swapaxes(-1, -2) * scale
    if causal:
        scores = np.where(np.arange(k.shape[-2]) <= np.arange(q.shape[-2])[:, None], scores, -np.inf)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    return (p / p.sum(axis=-1, keepdims=True)) @ v


def check() -> None:
    rng = np.random.default_rng(0)

    def rand(dtype, *shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(dtype)

    f32, f64, f16 = np.float32, np.float64, np.float16
    bf16 = np.dtype(ml_dtypes.bfloat16)
    q = rand(f32, 2, 3, 128, 32)
    k = rand(f32, 2, 3, 128, 32)
    v = rand(f32, 2, 3, 128, 48)
    rect_q, rect_k, rect_v = rand(f32, 64, 200, 16), rand(f32, 64, 160, 16), rand(f32, 64, 160, 48)
    tail = rand(f32, 100, 32)
    hot_q = rand(f32, 256, 64) * 100.0
    cases = [
        (q, k, v, True, 1e-5, "f32 causal batched (2,3,128,32)"),
        (q, k, v, False, 1e-5, "f32 full batched (2,3,128,32)"),
        (rect_q, rect_k, rect_v, False, 1e-5, "f32 rectangular 64x160"),
        (tail, tail, rand(f32, 100, 32), False, 1e-5, "f32 100 keys, undivided tail"),
        (hot_q, rand(f32, 256, 64), rand(f32, 256, 64), True, 1e-4, "f32 causal, Q x 100"),
        (rand(f64, 2, 64, 32), rand(f64, 2, 64, 32), rand(f64, 2, 64, 16), True, 1e-12, "f64 causal"),
        (rand(f16, 2, 3, 128, 32), rand(f16, 2, 3, 128, 32), rand(f16, 2, 3, 128, 48), True, 2e-3, "f16 causal batched"),
        (rand(bf16, 2, 3, 128, 32), rand(bf16, 2, 3, 128, 32), rand(bf16, 2, 3, 128, 48), True, 2e-2, "bf16 causal batched"),
    ]
    for cq, ck, cv, causal, tol, name in cases:
        # half-width floats are compared widened: the check is the recurrence, not the storage rounding
        wide = np.float32 if cq.dtype in HALF_FLOATS else cq.dtype
        scale = cq.shape[-1] ** -0.5
        expected = _plain(cq.astype(wide), ck.astype(wide), cv.astype(wide), causal=causal, scale=scale)
        worst, worst_block = 0.0, 0
        for block in (1, 16, ck.shape[-2], ck.shape[-2] + 7):
            got = sdpa(cq, ck, cv, causal=causal, scale=scale, block_size=block)
            assert got.dtype == cq.dtype, f"{name}: result dtype {got.dtype}, want {cq.dtype}"
            diff = float(np.abs(got.astype(wide) - expected).max())
            worst, worst_block = (diff, block) if diff > worst else (worst, worst_block)
        verdict = "ok " if worst <= tol else "FAIL"
        print(f"{verdict} {name:36s} blocks 1,16,{ck.shape[-2]},{ck.shape[-2] + 7}  max diff {worst:.2e} (at {worst_block})")
        assert worst <= tol, f"{name}: max diff {worst:.2e} exceeds {tol:g}"


if __name__ == "__main__":
    check()
