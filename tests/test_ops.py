"""Tensor-level op tests against numpy: elementwise, broadcasting, reduces, matmul, movement, errors."""

import numpy as np
import pytest

from limn import Tensor, float32, int32
from limn.ops import Op

rng = np.random.default_rng(7)


def randf(*shape: int) -> np.ndarray:
    return rng.uniform(-2, 2, shape).astype(np.float32)


def check(limn_out: Tensor, expected: np.ndarray) -> None:
    got = limn_out.numpy()
    assert got.shape == tuple(expected.shape)
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_ops_are_lazy():
    a = Tensor(randf(2, 3))
    b = a + a * 2.0
    assert b.node.op is Op.ADD  # building did not compute


def test_elementwise_and_broadcasting():
    a, b = randf(2, 3), randf(3)
    c = randf(2, 1)
    check(Tensor(a) + Tensor(b), a + b)
    check(Tensor(a) * Tensor(c), a * c)
    check(Tensor(a) - Tensor(b), a - b)
    check(Tensor(a) / (Tensor(b) * Tensor(b) + 0.5), a / (b * b + 0.5))
    check(-Tensor(a), -a)
    check(Tensor(a) * 3.0 + 1.0, a * 3 + 1)
    check(2.0 - Tensor(a), 2 - a)
    check(1.0 / (Tensor(a) * Tensor(a) + 1.0), 1 / (a * a + 1))


def test_unary_math():
    a = randf(3, 4)
    positive = a * a + 0.5
    check(Tensor(a).exp(), np.exp(a))
    check(Tensor(positive).log(), np.log(positive))
    check(Tensor(positive).sqrt(), np.sqrt(positive))
    check(Tensor(positive).reciprocal(), 1 / positive)
    check(Tensor(a) ** 3, a**3)
    check(Tensor(a) ** 2.0, a**2)
    check(Tensor(positive) ** 0.5, np.sqrt(positive))
    with pytest.raises(ValueError, match="pow"):
        _ = Tensor(a) ** 2.5


def test_comparisons_and_where():
    a, b = randf(2, 3), randf(2, 3)
    check(Tensor(a) < Tensor(b), (a < b).astype(np.float32))
    check(Tensor(a) <= Tensor(a), np.ones_like(a))
    check(Tensor(a).eq(Tensor(a)), np.ones_like(a))
    check(Tensor(a).eq(Tensor(b)), (a == b).astype(np.float32))
    check((Tensor(a) < Tensor(b)).where(Tensor(a), Tensor(b)), np.where(a < b, a, b))
    check((Tensor(a) > 0).where(1.0, -1.0), np.where(a > 0, 1.0, -1.0).astype(np.float32))
    check(Tensor(a).maximum(Tensor(b)), np.maximum(a, b))
    check(Tensor(a).minimum(0.0), np.minimum(a, 0))
    check(Tensor(a).relu(), np.maximum(a, 0))


def test_reduces():
    a = randf(2, 3, 4)
    check(Tensor(a).sum(), np.asarray(a.sum()))
    check(Tensor(a).sum(axis=1), a.sum(axis=1))
    check(Tensor(a).sum(axis=(0, 2), keepdim=True), a.sum(axis=(0, 2), keepdims=True))
    check(Tensor(a).sum(axis=-1), a.sum(axis=-1))
    check(Tensor(a).max(), np.asarray(a.max()))
    check(Tensor(a).max(axis=0), a.max(axis=0))
    check(Tensor(a).max(axis=(1, 2), keepdim=True), a.max(axis=(1, 2), keepdims=True))
    check(Tensor(a).mean(axis=1), a.mean(axis=1))
    check(Tensor(a).mean(), np.asarray(a.mean()))


def test_int_reduces_and_promotion():
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert Tensor(a).dtype == int32
    check(Tensor(a).sum(axis=0), a.sum(axis=0))
    out = Tensor(a) + Tensor(0.5)
    assert out.dtype == float32
    check(out, a + 0.5)
    check(Tensor(a).float() * 2.0, a * 2.0)


def test_cast():
    a = np.array([1.7, -2.3, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(Tensor(a).int().numpy(), a.astype(np.int32))
    b = np.array([1, -2, 3], dtype=np.int32)
    np.testing.assert_array_equal(Tensor(b).float().numpy(), b.astype(np.float32))


def test_matmul():
    a, b = randf(4, 5), randf(5, 3)
    check(Tensor(a) @ Tensor(b), a @ b)
    a3, b3 = randf(2, 4, 5), randf(2, 5, 3)
    check(Tensor(a3) @ Tensor(b3), a3 @ b3)
    check(Tensor(a3) @ Tensor(b), a3 @ b)  # batch broadcast
    a4, b4 = randf(2, 3, 4, 5), randf(2, 3, 5, 6)
    check(Tensor(a4) @ Tensor(b4), a4 @ b4)


def test_movement():
    a = randf(2, 3, 4)
    check(Tensor(a).reshape(6, 4), a.reshape(6, 4))
    check(Tensor(a).reshape(-1), a.reshape(-1))
    check(Tensor(a).permute(2, 0, 1), a.transpose(2, 0, 1))
    check(Tensor(a).transpose(), a.swapaxes(-2, -1))
    check(Tensor(a).pad(((1, 0), (0, 2), (1, 1))), np.pad(a, ((1, 0), (0, 2), (1, 1))))
    check(Tensor(a).shrink(((0, 1), (1, 3), (0, 4))), a[0:1, 1:3, 0:4])
    check(Tensor(a).flatten(), a.reshape(-1))
    check(Tensor(a[0, 0]).reshape(1, 1, 4).expand(2, 3, 4), np.broadcast_to(a[0, 0], (2, 3, 4)))


def test_reshape_of_noncontiguous_inserts_copy():
    a = randf(3, 4)
    check(Tensor(a).transpose().reshape(12), a.T.reshape(12))
    padded = Tensor(a).pad(((1, 0), (0, 0)))
    check(padded.reshape(16), np.pad(a, ((1, 0), (0, 0))).reshape(16))


def test_movement_chain_matches_numpy():
    a = randf(2, 3)
    out = Tensor(a).pad(((1, 1), (2, 0))).permute(1, 0).shrink(((1, 4), (0, 3))).reshape(3, 3, 1)
    expected = np.pad(a, ((1, 1), (2, 0))).T[1:4, 0:3].reshape(3, 3, 1)
    check(out, expected)


def test_assign_updates_buffer():
    t = Tensor(np.zeros((2, 2), dtype=np.float32))
    t.assign(Tensor(np.ones((2, 2), dtype=np.float32)) * 3.0)
    assert t.node.op is Op.ASSIGN
    t.realize()
    assert t.node.op is Op.BUFFER
    np.testing.assert_array_equal(t.numpy(), np.full((2, 2), 3.0, dtype=np.float32))
    doubled = (t * 2.0).numpy()  # graphs built after the assign see the new bytes
    np.testing.assert_array_equal(doubled, np.full((2, 2), 6.0, dtype=np.float32))


def test_an_assign_committed_by_another_graph_is_not_applied_twice():
    """Any graph containing the assign commits it, so the tensor must stop pointing at it."""
    t = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    t.assign(t + 1.0)
    np.testing.assert_array_equal((t * 10.0).numpy(), np.array([20.0, 30.0], dtype=np.float32))
    assert t.node.op is Op.BUFFER
    np.testing.assert_array_equal(t.numpy(), np.array([2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(t.numpy(), np.array([2.0, 3.0], dtype=np.float32))


def test_shape_errors():
    a, b = Tensor(randf(2, 3)), Tensor(randf(4, 5))
    with pytest.raises(ValueError, match="ADD"):
        _ = a + b
    with pytest.raises(ValueError, match="matmul"):
        _ = a @ b
    with pytest.raises(ValueError, match="reshape"):
        a.reshape(7)
    with pytest.raises(ValueError, match="expand"):
        a.expand(2, 5)
    with pytest.raises(ValueError, match="axis"):
        a.sum(axis=5)
    with pytest.raises(ValueError):
        Tensor(randf(2, 2), dtype=int32, requires_grad=True)
    with pytest.raises(ValueError):
        a.assign(Tensor(randf(3, 2)))


def test_scalar_tensors():
    t = Tensor(2.5)
    assert t.shape == () and t.dtype == float32
    assert t.item() == pytest.approx(2.5)
    check(t + 1.0, np.asarray(np.float32(3.5)))


def test_gather_rows_picks_table_rows():
    table = randf(5, 3)
    indices = np.array([[4, 0], [2, 2]], dtype=np.int32)
    got = Tensor(table).gather_rows(Tensor(indices))
    assert got.shape == (2, 2, 3)
    check(got, table[indices])


def test_gather_rows_backward_sums_repeated_indices():
    table = Tensor(np.arange(8, dtype=np.float32).reshape(4, 2), requires_grad=True)
    indices = Tensor(np.array([0, 2, 0], dtype=np.int32))
    weights = Tensor(np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32))
    (table.gather_rows(indices) * weights).sum().backward()
    assert table.grad is not None
    # row 0 was picked twice, so it collects both weights; rows nobody picked stay zero
    check(table.grad, np.array([[4.0, 40.0], [0.0, 0.0], [2.0, 20.0], [0.0, 0.0]], dtype=np.float32))


def test_gather_rows_rejects_bad_tables_and_indices():
    with pytest.raises(ValueError, match="2D table"):
        Tensor(randf(2, 3, 4)).gather_rows(Tensor(np.array([0], dtype=np.int32)))
    with pytest.raises(ValueError, match="int32 indices"):
        Tensor(randf(5, 3)).gather_rows(Tensor(randf(2)))
