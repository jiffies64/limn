"""Fused scaled dot-product attention: the numpy reference the CUSTOM "sdpa" nodes run.

The math is ordinary attention, q @ k' scaled, softmax along the key axis, weighted values.
The shape of the computation is flash's: the softmax runs in blocks of keys, and each query
row carries a running max, a running denominator, and a running weighted sum, so no
t_q by t_k intermediate ever exists, forward or backward.

The bookkeeping is one line of algebra. When a block raises a row's running max from m to
m', everything accumulated under m was exponentiated against the wrong baseline, and the
factor exp(m - m') converts it. That leaves the recurrence equal to plain softmax up to
rounding, at O(t_q) working memory instead of O(t_q * t_k).

The forward hands back one number per query row besides the output: L, the row's logsumexp,
which is m + log(l) once the recurrence has run. That is the whole of what the backward needs
to rebuild a probability without the forward having kept one, since p_ij = exp(s_ij - L_i)
exactly. Recomputing p from L in blocks is what keeps the backward the same shape as the
forward; deriving it from the output instead would need the t_q by t_k the forward refused to
build.

The backward is two passes over that recomputed p, because the two halves want opposite loop
orders. dQ accumulates per query row over all keys, dK and dV accumulate per key row over all
queries, and each row of each pass belongs to exactly one accumulator, so nothing is added
twice and no ordering is left to chance. dK and dV come out of one pass rather than two: they
share p, and splitting them would recompute it. The other input both passes need is
D_i = sum_d dO_id * O_id, the row dot of the output and its gradient, which is ordinary
elementwise work the caller supplies.

Running it as a script checks the recurrence against plain softmax and the two backward passes
against a textbook one that does materialize the t_q by t_k; the registry the numpy device
uses is KERNELS at the bottom.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import ml_dtypes
import numpy as np

HALF_FLOATS = (np.float16, np.dtype(ml_dtypes.bfloat16))  # storage widths: the recurrence runs wider

SDPA = "sdpa"  # the CUSTOM names; every device's registry and every emitter answers to these
SDPA_BWD_Q = "sdpa_bwd_q"
SDPA_BWD_KV = "sdpa_bwd_kv"


def _widened(*arrays: np.ndarray) -> Iterator[np.ndarray]:
    """The arrays at the width the recurrence runs: float32 for a half, their own otherwise."""
    work = np.dtype(np.float32) if arrays[0].dtype in HALF_FLOATS else arrays[0].dtype
    return (a.astype(work, copy=False) for a in arrays)


def _blocks(n: int, block_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + block_size, n)) for start in range(0, n, block_size)]


def _validate(q: np.ndarray, k: np.ndarray, v: np.ndarray, key_mask: np.ndarray | None, causal: bool) -> None:
    if q.ndim < 2 or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError(f"sdpa wants 2D+ inputs of equal rank, got {q.shape}, {k.shape}, {v.shape}")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2] or k.shape[-2] != v.shape[-2]:
        raise ValueError(f"sdpa shape mismatch: {q.shape}, {k.shape}, {v.shape}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"sdpa: head dims differ, {q.shape[-1]} and {k.shape[-1]}")
    if causal and q.shape[-2] != k.shape[-2]:
        raise ValueError(f"causal sdpa needs square keys, got {q.shape[-2]} queries and {k.shape[-2]} keys")
    if key_mask is not None and key_mask.shape != q.shape[:-2] + k.shape[-2:-1]:
        raise ValueError(f"sdpa key mask must be {q.shape[:-2] + k.shape[-2:-1]}, got {key_mask.shape}")


def _keep(key_mask: np.ndarray | None, cols: slice) -> np.ndarray | None:
    """A block of the key mask, shaped to broadcast over a block of scores: one flag per key,
    the same for every query row, so it needs a length-1 query axis wedged in."""
    return None if key_mask is None else key_mask[..., None, cols]


def _scores(
    a: np.ndarray, b: np.ndarray, rows: range, cols: range, *, causal: bool, scale: float, keep: np.ndarray | None = None
) -> np.ndarray:
    """The scaled scores of the `rows` queries against the `cols` keys, -inf where causal hides
    a key: the block every pass starts from. The causal mask comes from the two index ranges,
    since a block knows where it sits on each axis, and it lands on the trailing dims only, so it
    broadcasts over any batch. `keep` hides a key from every query row instead of from the ones
    left of it, and the two compose."""
    scores = a @ b.swapaxes(-1, -2) * scale
    if causal:
        scores = np.where(np.asarray(cols) <= np.asarray(rows)[:, None], scores, -np.inf)
    return scores if keep is None else np.where(keep, scores, -np.inf)


def sdpa(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    key_mask: np.ndarray | None = None,
    *,
    causal: bool,
    scale: float,
    block_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """softmax(q @ k' * scale) @ v along the key axis, folded one block of keys at a time.

    The inputs share their leading dims; q is (..., t_q, hd), k is (..., t_k, hd),
    v is (..., t_k, hd_v). causal wants t_q == t_k and hides keys to the right of each
    query row. key_mask is (..., t_k), one flag per key and the same for every query row,
    and a falsy one hides that key. half-width float inputs widen to float32 for the
    recurrence and round back once at the end, the same rule every reduce in limn owes.

    Returns the output and L, the per-row logsumexp of the masked scaled scores, as
    (..., t_q, 1) at the recurrence's own width: the row statistic the backward rebuilds
    probabilities from. A row left with no key at all is the caller's error: it comes back
    NaN, as it does from torch, and its L is -inf.
    """
    _validate(q, k, v, key_mask, causal)
    Q, K, V = _widened(q, k, v)
    work = Q.dtype
    batch, t_q, t_k = q.shape[:-2], q.shape[-2], k.shape[-2]

    out = np.zeros(batch + (t_q, v.shape[-1]), dtype=work)
    running_max = np.full(batch + (t_q,), -np.inf, dtype=work)
    running_sum = np.zeros(batch + (t_q,), dtype=work)
    for j0, j1 in _blocks(t_k, block_size):
        scores = _scores(
            Q, K[..., j0:j1, :], range(t_q), range(j0, j1), causal=causal, scale=scale, keep=_keep(key_mask, slice(j0, j1))
        )
        new_max = np.maximum(running_max, scores.max(axis=-1))
        # a row whose keys have all been hidden so far has no baseline to convert against, and
        # -inf minus -inf is NaN rather than the nothing it should be. Standing that row's
        # baseline at 0 leaves both its factors exp(-inf), which is the zero it has accumulated.
        baseline = np.where(np.isneginf(new_max), work.type(0), new_max)
        rescale = np.exp(running_max - baseline)
        weights = np.exp(scores - baseline[..., None])
        running_sum = running_sum * rescale + weights.sum(axis=-1)
        out = out * rescale[..., None] + weights @ V[..., j0:j1, :]
        running_max = new_max
    lse = (running_max + np.log(running_sum))[..., None]
    return (out / running_sum[..., None]).astype(q.dtype), lse


def sdpa_backward_q(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    do: np.ndarray,
    lse: np.ndarray,
    delta: np.ndarray,
    key_mask: np.ndarray | None = None,
    *,
    causal: bool,
    scale: float,
    block_size: int = 16,
) -> np.ndarray:
    """dQ, one block of keys at a time: every query row's total is its own, so the pass runs
    over query rows and folds keys into them.

    lse and delta are the forward's per-row logsumexp and the row dot sum_d dO_id * O_id, both
    (..., t_q, 1), which is the shape that broadcasts over a block of keys.
    """
    _validate(q, k, v, key_mask, causal)
    Q, K, V, DO, L, D = _widened(q, k, v, do, lse, delta)
    t_q, t_k = q.shape[-2], k.shape[-2]
    dq = np.zeros(Q.shape, dtype=Q.dtype)
    for j0, j1 in _blocks(t_k, block_size):
        keep = _keep(key_mask, slice(j0, j1))
        scores = _scores(Q, K[..., j0:j1, :], range(t_q), range(j0, j1), causal=causal, scale=scale, keep=keep)
        p = np.exp(scores - L)  # exactly zero where the mask sent the score to -inf
        dp = DO @ V[..., j0:j1, :].swapaxes(-1, -2)
        ds = p * (dp - D) * scale
        dq += ds @ K[..., j0:j1, :]
    return dq.astype(q.dtype)


def sdpa_backward_kv(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    do: np.ndarray,
    lse: np.ndarray,
    delta: np.ndarray,
    key_mask: np.ndarray | None = None,
    *,
    causal: bool,
    scale: float,
    block_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """dK and dV out of one pass over blocks of queries, since both fold the same recomputed p.

    A hidden key contributes to no query row, so its p is zero the whole way down and its dK
    and dV rows come out zero, which is what a gradient that reaches nothing should be."""
    _validate(q, k, v, key_mask, causal)
    Q, K, V, DO, L, D = _widened(q, k, v, do, lse, delta)
    t_q, t_k = q.shape[-2], k.shape[-2]
    dk = np.zeros(K.shape, dtype=K.dtype)
    dv = np.zeros(V.shape, dtype=V.dtype)
    for i0, i1 in _blocks(t_q, block_size):
        Qi, DOi = Q[..., i0:i1, :], DO[..., i0:i1, :]
        scores = _scores(Qi, K, range(i0, i1), range(t_k), causal=causal, scale=scale, keep=_keep(key_mask, slice(None)))
        p = np.exp(scores - L[..., i0:i1, :])
        dv += p.swapaxes(-1, -2) @ DOi
        dp = DOi @ V.swapaxes(-1, -2)
        ds = p * (dp - D[..., i0:i1, :]) * scale
        dk += ds.swapaxes(-1, -2) @ Qi
    return dk.astype(k.dtype), dv.astype(v.dtype)


# ---- the registry: what the numpy device runs for each CUSTOM name ----


def _kernel(fn: Callable[..., Any]) -> Callable[[list[np.ndarray], Any], tuple[np.ndarray, ...]]:
    """One registry entry: the CUSTOM node's params unpacked into the keywords every kernel here
    takes, and the lone answer of a one-output kernel read as the tuple the device expects."""

    def run(srcs: list[np.ndarray], arg) -> tuple[np.ndarray, ...]:
        causal, scale = arg.params
        out = fn(*srcs, causal=causal, scale=scale)
        return out if isinstance(out, tuple) else (out,)

    return run


KERNELS = {name: _kernel(fn) for name, fn in ((SDPA, sdpa), (SDPA_BWD_Q, sdpa_backward_q), (SDPA_BWD_KV, sdpa_backward_kv))}


# ---- checker: run as a script ----


def _plain_probs(q: np.ndarray, k: np.ndarray, *, causal: bool, scale: float, key_mask=None) -> np.ndarray:
    """The probabilities the obvious way, for the checker to diff against: the whole t_q by t_k
    at once, softmaxed in one shot rather than folded block by block."""
    keep = _keep(key_mask, slice(None))
    scores = _scores(q, k, range(q.shape[-2]), range(k.shape[-2]), causal=causal, scale=scale, keep=keep)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def _plain(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, causal: bool, scale: float, key_mask=None) -> np.ndarray:
    """Attention the obvious way: the answer the recurrence has to land on."""
    return _plain_probs(q, k, causal=causal, scale=scale, key_mask=key_mask) @ v


def _plain_backward(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, do: np.ndarray, *, causal: bool, scale: float, key_mask=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The textbook backward, materialized: the form Tensor.attention used to differentiate."""
    p = _plain_probs(q, k, causal=causal, scale=scale, key_mask=key_mask)
    dp = do @ v.swapaxes(-1, -2)
    ds = p * (dp - (dp * p).sum(axis=-1, keepdims=True)) * scale
    return ds @ k, ds.swapaxes(-1, -2) @ q, p.swapaxes(-1, -2) @ do


def _worst(got: np.ndarray, want: np.ndarray) -> float:
    """How far apart two answers are, measured against the size of the one being checked.

    A gradient of q scaled by 100 is itself scaled by 100, and holding it to the absolute
    tolerance an order-1 output meets would be asking float32 for digits it does not have.
    Dividing by the answer's own magnitude (never sharpening below 1) puts every case on the
    one tolerance.
    """
    return float(np.abs(got - want).max() / max(1.0, float(np.abs(want).max())))


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
    pad = np.arange(128) < 77  # a padded batch: the keys past 77 are not there
    holes = rng.random((2, 3, 128)) > 0.3  # hidden keys need not be a suffix
    step_q, step_k, step_v = rand(f32, 4, 1, 32), rand(f32, 4, 96, 32), rand(f32, 4, 96, 48)
    cases = [
        (q, k, v, None, True, 1e-5, "f32 causal batched (2,3,128,32)"),
        (q, k, v, None, False, 1e-5, "f32 full batched (2,3,128,32)"),
        (rect_q, rect_k, rect_v, None, False, 1e-5, "f32 rectangular 64x160"),
        (tail, tail, rand(f32, 100, 32), None, False, 1e-5, "f32 100 keys, undivided tail"),
        (hot_q, rand(f32, 256, 64), rand(f32, 256, 64), None, True, 1e-4, "f32 causal, Q x 100"),
        (rand(f64, 2, 64, 32), rand(f64, 2, 64, 32), rand(f64, 2, 64, 16), None, True, 1e-12, "f64 causal"),
        (rand(f16, 2, 3, 128, 32), rand(f16, 2, 3, 128, 32), rand(f16, 2, 3, 128, 48), None, True, 2e-3, "f16 causal batched"),
        (rand(bf16, 2, 3, 128, 32), rand(bf16, 2, 3, 128, 32), rand(bf16, 2, 3, 128, 48), None, True, 2e-2, "bf16 causal"),
        (q, k, v, np.broadcast_to(pad, (2, 3, 128)), True, 1e-5, "f32 causal, 77 keys of 128"),
        (q, k, v, holes.astype(f32), False, 1e-5, "f32 full, keys hidden at random"),
        (step_q, step_k, step_v, np.arange(96) < 61, False, 1e-5, "f32 one query row, 61 keys of 96"),
    ]
    for cq, ck, cv, mask, causal, tol, name in cases:
        # half-width floats are compared widened: the check is the recurrence, not the storage rounding
        wide = np.float32 if cq.dtype in HALF_FLOATS else cq.dtype
        scale = cq.shape[-1] ** -0.5
        wq, wk, wv = cq.astype(wide), ck.astype(wide), cv.astype(wide)
        do = rng.standard_normal(cq.shape[:-1] + cv.shape[-1:]).astype(wide)  # the output's shape
        mask = None if mask is None else np.broadcast_to(mask, cq.shape[:-2] + ck.shape[-2:-1])
        expected = _plain(wq, wk, wv, causal=causal, scale=scale, key_mask=mask)
        wanted = _plain_backward(wq, wk, wv, do, causal=causal, scale=scale, key_mask=mask)
        worst, worst_block = 0.0, 0
        for block in (1, 16, ck.shape[-2], ck.shape[-2] + 7):
            got, lse = sdpa(cq, ck, cv, mask, causal=causal, scale=scale, block_size=block)
            assert got.dtype == cq.dtype, f"{name}: result dtype {got.dtype}, want {cq.dtype}"
            assert lse.shape == cq.shape[:-1] + (1,), f"{name}: lse shape {lse.shape}"
            delta = (do * got.astype(wide)).sum(axis=-1, keepdims=True)
            args = (cq, ck, cv, do.astype(cq.dtype), lse, delta, mask)
            grads = (
                sdpa_backward_q(*args, causal=causal, scale=scale, block_size=block),
                *sdpa_backward_kv(*args, causal=causal, scale=scale, block_size=block),
            )
            diffs = [_worst(got.astype(wide), expected)]
            diffs += [_worst(g.astype(wide), w) for g, w in zip(grads, wanted, strict=True)]
            diff = max(diffs)
            worst, worst_block = (diff, block) if diff > worst else (worst, worst_block)
        verdict = "ok " if worst <= tol else "FAIL"
        print(f"{verdict} {name:36s} blocks 1,16,{ck.shape[-2]},{ck.shape[-2] + 7}  max diff {worst:.2e} (at {worst_block})")
        assert worst <= tol, f"{name}: max diff {worst:.2e} exceeds {tol:g}"


if __name__ == "__main__":
    check()
