"""Attention: the CUSTOM seam. The forward diffs against the composed form on the numpy
device, and a device that registers no kernel gets the composed form back."""

import numpy as np
import pytest

from conftest import needs_cc

from limn import Tensor, set_device
from limn.ops import Op, topological
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
