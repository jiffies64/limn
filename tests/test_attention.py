"""Attention: the CUSTOM seam. The forward diffs against the composed form on the numpy
device, the backward against the composed form and torch, and a device that registers no
kernel gets the composed form back."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import needs_cc

from limn import Tensor, grad, set_device, set_seed
from limn.nn import Linear, parameters
from limn.ops import Op, topological
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


def test_forward_matches_composed_float16():
    q, k, v = rand(2, 3, 64, 16).half(), rand(2, 3, 64, 16).half(), rand(2, 3, 64, 24).half()
    got = q.attention(k, v, causal=True)
    want = composed_attention(q, k, v, causal=True, scale=16**-0.5)
    assert got.dtype == q.dtype
    # compared widened: the fused recurrence accumulates in float32, the composed path in halves
    np.testing.assert_allclose(got.numpy().astype(np.float32), want.numpy().astype(np.float32), rtol=1e-2, atol=1e-2)


def test_numpy_builds_a_custom_node():
    q, k, v = rand(2, 8, 16), rand(2, 8, 16), rand(2, 8, 16)
    t = q.attention(k, v, causal=True)
    assert t.node.op is Op.CUSTOM
    assert t.node.arg == ("sdpa", True, 16**-0.5)


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
        np.testing.assert_allclose(got.grad.numpy(), want.grad.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_grad_of_grad_matches_torch(causal):
    # torch's fused CPU kernel has no second backward, so hold it to the math backend
    from torch.nn.attention import SDPBackend, sdpa_kernel

    qd, kd, vd = (rng.standard_normal(s).astype(np.float32) for s in ((2, 3, 16, 8),) * 3)
    q, k, v = (Tensor(d, requires_grad=True) for d in (qd, kd, vd))
    loss = (q.attention(k, v, causal=causal) ** 2).sum()
    first = grad(loss, [q, k, v], create_graph=True)
    second = grad(sum(g.sum() for g in first), [q, k, v])
    tq = torch.tensor(qd, requires_grad=True)
    tk = torch.tensor(kd, requires_grad=True)
    tv = torch.tensor(vd, requires_grad=True)
    with sdpa_kernel(SDPBackend.MATH):
        tloss = (F.scaled_dot_product_attention(tq, tk, tv, is_causal=causal) ** 2).sum()
        tfirst = torch.autograd.grad(tloss, (tq, tk, tv), create_graph=True)
        tsecond = torch.autograd.grad(sum(t.sum() for t in tfirst), (tq, tk, tv))
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


@needs_cc
def test_unregistered_device_falls_back_to_composed():
    shapes = ((2, 3, 32, 8), (2, 3, 32, 8), (2, 3, 32, 12))
    data = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    set_device("c")
    try:
        q, k, v = (Tensor(d) for d in data)
        t = q.attention(k, v, causal=True)
        assert Op.CUSTOM not in {n.op for n in topological([t.node])}
        got = t.numpy()
    finally:
        set_device("numpy")
    q, k, v = (Tensor(d) for d in data)
    want = composed_attention(q, k, v, causal=True, scale=8**-0.5)
    np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-5)
