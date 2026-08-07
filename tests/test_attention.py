"""Attention: the CUSTOM seam. The forward diffs against the composed form on the numpy
device, the backward against the composed form and torch, and a device that registers no
kernel gets the composed form back."""

import gc
import weakref
from functools import reduce
from operator import add

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import check, cudev, needs_cc, needs_cuda

from limn import Tensor, grad, set_device, set_seed
from limn.nn import Linear, parameters
from limn.ops import Custom, DType, Op, bfloat16, float16, float32, topological
from limn.optim import AdamW
from limn.tensor import composed_attention

rng = np.random.default_rng(3)


def rand(*shape: int) -> Tensor:
    return Tensor(rng.standard_normal(shape).astype(np.float32))


# (q, k, v, causal): batched, plain 2D, rectangular keys, and batch dims that broadcast
CASES = [
    ((2, 3, 64, 16), (2, 3, 64, 16), (2, 3, 64, 24), True),
    ((64, 32), (64, 32), (64, 48), False),
    ((2, 40, 60, 8), (2, 40, 60, 8), (2, 40, 60, 8), False),
    ((1, 3, 16, 8), (4, 3, 16, 8), (4, 1, 16, 12), True),
]


def test_forward_matches_composed():
    for q_shape, k_shape, v_shape, causal in CASES:
        q, k, v = rand(*q_shape), rand(*k_shape), rand(*v_shape)
        got = q.attention(k, v, causal=causal)
        want = composed_attention(q, k, v, causal=causal, scale=q_shape[-1] ** -0.5)
        np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype,rtol", [(float16, 1e-2), (bfloat16, 4e-2)], ids=str)
def test_forward_matches_composed_in_a_half_width(dtype, rtol):
    q, k, v = rand(2, 3, 64, 16).cast(dtype), rand(2, 3, 64, 16).cast(dtype), rand(2, 3, 64, 24).cast(dtype)
    got = q.attention(k, v, causal=True)
    want = composed_attention(q, k, v, causal=True, scale=16**-0.5)
    assert got.dtype == q.dtype
    # compared widened: the fused recurrence accumulates in float32, the composed path in halves
    np.testing.assert_allclose(got.numpy().astype(np.float32), want.numpy().astype(np.float32), rtol=rtol, atol=rtol)


def test_numpy_builds_a_custom_node():
    q, k, v = rand(2, 8, 16), rand(2, 8, 16), rand(2, 8, 16)
    t = q.attention(k, v, causal=True)
    assert t.node.op is Op.CUSTOM
    assert t.node.arg == Custom("sdpa", (True, 16**-0.5), ((float32, (2, 8, 16)), (float32, (2, 8, 1))), index=0)


def test_validation():
    q, k, v = rand(4, 8, 16), rand(4, 8, 16), rand(4, 8, 16)
    with pytest.raises(ValueError):
        q.attention(k[:, :6], v[:, :6], causal=True)  # causal wants square keys
    with pytest.raises(ValueError):
        q.attention(rand(4, 8, 12), v)  # head dims differ
    with pytest.raises(ValueError):
        q.attention(k, rand(4, 7, 16))  # key counts differ
    with pytest.raises(ValueError):
        Tensor(np.arange(12, dtype=np.int32).reshape(4, 3)).attention(k, v)  # ints carry no attention
    with pytest.raises(ValueError):
        q[0, 0].attention(k, v)  # 1D tensors
    with pytest.raises(ValueError):
        q.attention(k, v, key_mask=rand(4, 7))  # a flag per key, and there are 8
    with pytest.raises(ValueError):
        q.attention(k, v, key_mask=rand(3, 8))  # leading dims that do not broadcast


def keys_live(shape: tuple[int, ...], live: int) -> Tensor:
    """A mask over the last axis keeping the first `live` keys, the shape padding arrives in."""
    return Tensor(np.broadcast_to(np.arange(shape[-1]) < live, shape).astype(np.float32))


def _weighted_grads(fused: bool, q_shape, k_shape, v_shape, causal: bool, data, key_mask=None) -> list[np.ndarray]:
    """d(w . attention(q, k, v))/d(q, k, v), fused or composed. The random weighting is what
    makes the incoming gradient uneven, so a backward that ignored it would show."""
    q, k, v = (Tensor(d, requires_grad=True) for d in data[:3])
    scale = q_shape[-1] ** -0.5
    kw = {"causal": causal, "key_mask": key_mask}
    att = q.attention(k, v, **kw) if fused else composed_attention(q, k, v, scale=scale, **kw)
    loss = (att * Tensor(data[3])).sum()
    return [g.numpy() for g in grad(loss, [q, k, v])]


def test_a_key_mask_equals_slicing_the_keys():
    """The oracle that needs no second implementation: hiding a suffix of the keys gives exactly
    what passing only the keys before it gives, gradients included."""
    t, hd, hd_v, live = 24, 8, 12, 15
    data = [rng.standard_normal(s).astype(np.float32) for s in ((2, t, hd), (2, t, hd), (2, t, hd_v), (2, t, hd_v))]

    def grads(mask: Tensor | None, keys: int) -> list[np.ndarray]:
        q, k, v = (Tensor(d[:, :n], requires_grad=True) for d, n in zip(data[:3], (t, keys, keys), strict=True))
        return [g.numpy() for g in grad((q.attention(k, v, key_mask=mask) * Tensor(data[3])).sum(), [q, k, v])]

    for got, want in zip(grads(keys_live((2, t), live), t), grads(None, live), strict=True):
        np.testing.assert_allclose(got[:, : want.shape[1]], want, rtol=1e-5, atol=1e-5)  # dQ is whole, dK and dV are cut


def test_a_hidden_key_takes_no_gradient():
    """A key no query row reads is a key nothing flows back to, so its dK and dV rows are zero
    rather than merely small: the fused passes never visit it at all."""
    t, live = 20, 12
    data = [rng.standard_normal((2, t, 8)).astype(np.float32) for _ in range(4)]
    _, dk, dv = _weighted_grads(True, (2, t, 8), (2, t, 8), (2, t, 8), False, data, keys_live((2, t), live))
    np.testing.assert_array_equal(dk[:, live:], 0.0)
    np.testing.assert_array_equal(dv[:, live:], 0.0)


def test_a_key_mask_matches_the_composed_form():
    """Both maskings at once, so the fused kernels and the composed fallback have to agree on
    how a causal mask and a key mask compose."""
    shape, live = (2, 3, 32, 8), 21
    data = [rng.standard_normal(shape).astype(np.float32) for _ in range(4)]
    args = (shape, shape, shape, True, data, keys_live(shape[:-1], live))
    for got, want in zip(*(_weighted_grads(f, *args) for f in (True, False)), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("q_shape,k_shape,v_shape,causal", CASES, ids=str)
def test_backward_matches_the_composed_backward(q_shape, k_shape, v_shape, causal):
    """The same shapes the forward is held to: batched, plain 2D, rectangular keys, and batch
    dims that broadcast, where a leaf's gradient sums over the dim it was expanded along."""
    out_shape = np.broadcast_shapes(q_shape[:-2], k_shape[:-2], v_shape[:-2]) + (q_shape[-2], v_shape[-1])
    data = [rng.standard_normal(s).astype(np.float32) for s in (q_shape, k_shape, v_shape, out_shape)]
    for got, want in zip(*(_weighted_grads(f, q_shape, k_shape, v_shape, causal, data) for f in (True, False))):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_the_backward_never_builds_a_t_by_t(causal):
    """The point of the whole exercise: the composed backward materializes (t_q, t_k) three
    times over, and the fused one must not put a single node of that shape in the graph."""
    t, hd, hd_v = 48, 8, 12
    q, k, v = (
        Tensor(rng.standard_normal(s).astype(np.float32), requires_grad=True) for s in ((2, 3, t, hd),) * 2 + ((2, 3, t, hd_v),)
    )
    weight = Tensor(rng.standard_normal((2, 3, t, hd_v)).astype(np.float32))
    grads = grad((q.attention(k, v, causal=causal) * weight).sum(), [q, k, v])
    square = [n for n in topological([g.node for g in grads]) if n.shape[-2:] == (t, t)]
    assert not square, f"the backward graph holds {len(square)} nodes of shape (..., {t}, {t})"


def test_the_output_dies_with_its_last_reference():
    """The fused backward reads the output back, so it must not hold the tensor that holds it:
    a cycle there leaves every buffer under the graph to a collector pass. gc off is what makes
    the check the reference count and not the collector."""
    gc.disable()
    try:
        q, k, v = (Tensor(rng.standard_normal((2, 8, 4)).astype(np.float32), requires_grad=True) for _ in range(3))
        out = q.attention(k, v, causal=True)
        ref = weakref.ref(out)
        del out
        assert ref() is None
    finally:
        gc.enable()


def _attention_case(causal: bool):
    qd, kd, vd = (rng.standard_normal(s).astype(np.float32) for s in ((2, 3, 32, 8), (2, 3, 32, 8), (2, 3, 32, 12)))
    q, k, v = (Tensor(d, requires_grad=True) for d in (qd, kd, vd))
    out = q.attention(k, v, causal=causal)
    (out * out).sum().backward()
    tq = torch.tensor(qd, requires_grad=True)
    tk = torch.tensor(kd, requires_grad=True)
    tv = torch.tensor(vd, requires_grad=True)
    tout = F.scaled_dot_product_attention(tq, tk, tv, is_causal=causal)
    (tout * tout).sum().backward()
    return (q, k, v), (tq, tk, tv)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_backward_matches_torch(causal):
    limn_tensors, torch_tensors = _attention_case(causal)
    for got, want in zip(limn_tensors, torch_tensors, strict=True):
        assert got.grad is not None and want.grad is not None
        np.testing.assert_allclose(got.grad.numpy(), want.grad.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_grad_of_grad_matches_torch(causal):
    # torch's fused CPU kernel has no second backward, so hold it to the math backend
    from torch.nn.attention import SDPBackend, sdpa_kernel

    qd, kd, vd = (rng.standard_normal(s).astype(np.float32) for s in ((2, 3, 16, 8),) * 3)
    q, k, v = (Tensor(d, requires_grad=True) for d in (qd, kd, vd))
    loss = (q.attention(k, v, causal=causal) ** 2).sum()
    first = grad(loss, [q, k, v], create_graph=True)
    second = grad(reduce(add, (g.sum() for g in first)), [q, k, v])
    tq = torch.tensor(qd, requires_grad=True)
    tk = torch.tensor(kd, requires_grad=True)
    tv = torch.tensor(vd, requires_grad=True)
    with sdpa_kernel(SDPBackend.MATH):
        tloss = (F.scaled_dot_product_attention(tq, tk, tv, is_causal=causal) ** 2).sum()
        tfirst = torch.autograd.grad(tloss, (tq, tk, tv), create_graph=True)
        tsecond = torch.autograd.grad(reduce(add, (t.sum() for t in tfirst)), (tq, tk, tv))
    for got, want in zip(second, tsecond, strict=True):
        np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=1e-3, atol=1e-3)


class TinyAttention:
    def __init__(self, fused: bool):
        self.fused = fused
        self.wq, self.wk, self.wv, self.wo = Linear(8, 8), Linear(8, 8), Linear(8, 8), Linear(8, 8)

    def __call__(self, x: Tensor) -> Tensor:
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        att = q.attention(k, v, causal=True) if self.fused else composed_attention(q, k, v, causal=True, scale=8**-0.5)
        return self.wo(att)


def test_tiny_train_matches_composed():
    def run(fused: bool) -> list[float]:
        set_seed(0)
        model = TinyAttention(fused)
        optimizer = AdamW(parameters(model), lr=1e-2)
        data = np.random.default_rng(1)
        losses = []
        for _ in range(10):
            x = Tensor(data.standard_normal((4, 8, 8)).astype(np.float32))
            y = Tensor(data.standard_normal((4, 8, 8)).astype(np.float32))
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            losses.append(float(loss.item()))
            loss.backward()
            optimizer.step()
        return losses

    np.testing.assert_allclose(run(True), run(False), rtol=1e-4, atol=1e-5)


def _cuda_custom_graphs(causal: bool, dtype: DType = float32):
    qd, kd, vd = (rng.standard_normal(s).astype(np.float32) for s in ((2, 3, 64, 16), (2, 3, 64, 16), (2, 3, 64, 24)))
    q, k, v = (Tensor(d, dtype=dtype, requires_grad=True) for d in (qd, kd, vd))
    return q.attention(k, v, causal=causal), (q, k, v), (qd, kd, vd)


@needs_cuda
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("dtype,tol", [(float32, 1e-5), (float16, 2e-3), (bfloat16, 1e-2)], ids=str)
def test_cuda_matches_the_numpy_custom(causal, dtype, tol):
    out, _, _ = _cuda_custom_graphs(causal, dtype)
    check(cudev, out, rtol=tol, atol=tol)


@needs_cuda
def test_cuda_custom_is_deterministic():
    out, _, _ = _cuda_custom_graphs(True)
    assert cudev is not None
    first = cudev.copyout(cudev.execute([out.node])[0])
    second = cudev.copyout(cudev.execute([out.node])[0])
    np.testing.assert_array_equal(first, second)


@needs_cuda
def test_cuda_backward_runs_and_matches_numpy():
    out, leaves, _ = _cuda_custom_graphs(True)
    (out * out).sum().backward()
    for leaf in leaves:
        assert leaf.grad is not None
        check(cudev, leaf.grad, rtol=1e-4, atol=1e-4)


def _cuda_backward_grads(q_shape, k_shape, v_shape, causal: bool, dtype: DType = float32, key_mask=None):
    out_shape = np.broadcast_shapes(q_shape[:-2], k_shape[:-2], v_shape[:-2]) + (q_shape[-2], v_shape[-1])
    data = [rng.standard_normal(s).astype(np.float32) for s in (q_shape, k_shape, v_shape, out_shape)]
    q, k, v = (Tensor(d, dtype=dtype, requires_grad=True) for d in data[:3])
    weighted = q.attention(k, v, causal=causal, key_mask=key_mask) * Tensor(data[3], dtype=dtype)
    return grad(weighted.sum(), [q, k, v])


@needs_cuda
def test_cuda_two_attentions_on_one_input():
    """Two calls over the same q, k and v build CUSTOM nodes a plan cannot tell apart by name,
    params and srcs, which is all it merges siblings by; they still owe four separate buffers."""
    q, k, v = rand(2, 3, 32, 8), rand(2, 3, 32, 8), rand(2, 3, 32, 12)
    check(cudev, q.attention(k, v, causal=True) + q.attention(k, v, causal=True))


@needs_cuda
def test_cuda_backward_is_deterministic():
    """One thread owns each row's total in both passes, so no float lands in an atomic and two
    runs of the same kernels give back the same bits."""
    assert cudev is not None
    for g in _cuda_backward_grads((2, 3, 64, 16), (2, 3, 64, 16), (2, 3, 64, 24), causal=True):
        first = cudev.copyout(cudev.execute([g.node])[0])
        second = cudev.copyout(cudev.execute([g.node])[0])
        np.testing.assert_array_equal(first, second)


@needs_cuda
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("dtype,tol", [(float32, 1e-4), (float16, 4e-3), (bfloat16, 3e-2)], ids=str)
def test_cuda_backward_matches_the_numpy_custom(causal, dtype, tol):
    for g in _cuda_backward_grads((2, 3, 64, 16), (2, 3, 64, 16), (2, 3, 64, 24), causal, dtype):
        check(cudev, g, rtol=tol, atol=tol)


@needs_cuda
@pytest.mark.parametrize(
    "q_shape,k_shape,v_shape,causal",
    [
        ((2, 2, 100, 16), (2, 2, 100, 16), (2, 2, 100, 24), True),  # 3 tiles of keys and a remainder
        ((2, 2, 100, 16), (2, 2, 100, 16), (2, 2, 100, 24), False),
        ((300, 16), (300, 16), (300, 24), True),  # rows and keys past one block's chunk
        ((40, 8), (72, 8), (72, 12), False),  # rectangular: the two passes cover different counts
    ],
    ids=["ragged causal", "ragged full", "several chunks", "rectangular"],
)
def test_cuda_backward_covers_the_ragged_shapes(q_shape, k_shape, v_shape, causal):
    """The tails the two passes have to agree on, since one walks queries and the other keys."""
    for g in _cuda_backward_grads(q_shape, k_shape, v_shape, causal):
        check(cudev, g, rtol=1e-4, atol=1e-4)


@needs_cuda
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_cuda_handles_a_ragged_key_tail(causal):
    """100 keys is 3 tiles and a remainder, so the zero-filled tail of the last tile runs."""
    q, k, v = rand(2, 2, 100, 16), rand(2, 2, 100, 16), rand(2, 2, 100, 24)
    check(cudev, q.attention(k, v, causal=causal))


@needs_cuda
def test_cuda_rows_beyond_one_block_chunk():
    """300 rows over more than one block: the last block's tail threads sit past t_q, and under
    causal the earlier blocks stop streaming tiles at their own diagonal."""
    q, k, v = rand(300, 16), rand(300, 16), rand(300, 24)
    check(cudev, q.attention(k, v, causal=True))


def test_ir_prints_over_a_custom_kernel():
    """The custom kernel never lowers, so the printable IR shows everything but it."""
    from limn.codegen import ir

    q, k, v = rand(2, 8, 16), rand(2, 8, 16), rand(2, 8, 16)
    text = ir(q.attention(k, v, causal=True))
    assert text and "CUSTOM" not in text


@needs_cuda
def test_capture_replays_the_custom_step():
    from limn import capture

    set_device("cuda")
    qd, kd, vd = (rng.standard_normal(s).astype(np.float32) for s in ((2, 3, 32, 16),) * 2 + ((2, 3, 32, 24),))
    q, k, v = (Tensor(d) for d in (qd, kd, vd))

    def step(q, k, v):
        return q.attention(k, v, causal=True).realize()

    direct = step(q, k, v).numpy()
    replayed = capture(step)
    for got in [replayed(q, k, v).numpy() for _ in range(3)]:  # observe twice, then a real replay
        np.testing.assert_array_equal(got, direct)


@needs_cc
def test_unregistered_device_falls_back_to_composed():
    shapes = ((2, 3, 32, 8), (2, 3, 32, 8), (2, 3, 32, 12))
    data = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    set_device("c")
    q, k, v = (Tensor(d) for d in data)
    t = q.attention(k, v, causal=True, key_mask=keys_live((2, 3, 32), 20))
    assert Op.CUSTOM not in {n.op for n in topological([t.node])}
    got = t.numpy()
    set_device("numpy")
    q, k, v = (Tensor(d) for d in data)
    want = composed_attention(q, k, v, causal=True, scale=8**-0.5, key_mask=keys_live((2, 3, 32), 20))
    np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-5)


@needs_cuda
@pytest.mark.parametrize(
    "q_shape,k_shape,causal",
    [((2, 100, 16), (2, 100, 16), True), ((4, 1, 32), (4, 96, 32), False)],
    ids=["ragged causal", "one query row against a partly filled cache"],
)
def test_cuda_key_mask_matches_the_numpy_custom(q_shape, k_shape, causal):
    """The two shapes a key mask actually arrives in: a padded batch, and the decode step of a
    kv cache, where one query row reads the slots filled so far and nothing past them."""
    mask = keys_live(k_shape[:-1], k_shape[-2] * 2 // 3)
    for g in _cuda_backward_grads(q_shape, k_shape, k_shape, causal, key_mask=mask):
        check(cudev, g, rtol=1e-4, atol=1e-4)
