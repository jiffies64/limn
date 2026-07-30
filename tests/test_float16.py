"""float16: the dtype rules, and the cuda device against the numpy reference.

float16 is a storage width. Every device widens it to compute, and a reduce keeps its running
total in float32, so the tolerances here are float16's own rounding (eps is about 1e-3) rather
than the 1e-5 the float32 corpus is diffed at.
"""

import numpy as np
import pytest
from conftest import GRAPHS, check, cudev, needs_cc, needs_cuda, randf

from limn import Tensor, set_device
from limn.backend_c import CDevice
from limn.device import NUMPY_DTYPES
from limn.ops import float16, float32, int32, promote


def halves(*shape: int, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Values in [0, scale), which is where float16 keeps enough precision to diff at 2e-3.

    Two operands of one test need different seeds. Drawn from the shape alone they come out
    identical, and a graph over a == b stops telling a + b from 2a.
    """
    return (np.random.default_rng((seed, *shape)).random(shape).astype(np.float32) * scale).astype(np.float16)


def check_cuda(t: Tensor, rtol: float = 2e-3) -> None:
    check(cudev, t, rtol=rtol, atol=1e-3)


# ---- dtype rules, which hold on every device ----


def test_promotion_keeps_half_and_widens_against_float32_and_int32():
    assert promote(float16, float16) == float16
    assert promote(float16, float32) == float32
    assert promote(float16, int32) == float32


def test_arithmetic_between_halves_stays_half():
    a, b = Tensor(halves(4, 3), dtype=float16), Tensor(halves(4, 3, seed=1), dtype=float16)
    assert (a + b).dtype == float16
    assert (a * b).dtype == float16
    assert (a - b).dtype == float16
    assert (a / b).dtype == float16
    assert a.softmax(-1).dtype == float16
    assert (a @ b.transpose()).dtype == float16


def test_dividing_into_a_half_promotes_the_same_way_round_as_dividing_by_one():
    """A reflected divide has to ask promote too, or the same literal widens on one side only."""
    a = Tensor(halves(4, 3), dtype=float16)
    assert (a / 2.0).dtype == float16
    assert (2.0 / a).dtype == float16


def test_mixing_a_half_with_a_float32_widens_to_float32():
    a = Tensor(halves(4, 3), dtype=float16)
    b = Tensor(randf(4, 3))
    assert (a + b).dtype == float32
    assert (a * 2.0).dtype == float16  # a python scalar takes the tensor's dtype, it does not widen


def test_half_round_trips_through_the_numpy_device():
    data = halves(5, 7)
    np.testing.assert_array_equal(Tensor(data, dtype=float16).numpy(), data)


def test_a_half_reduce_totals_in_float32():
    """Summed in float16 the tail of this vector would vanish into the running total."""
    data = np.concatenate([np.array([2048.0], dtype=np.float16), np.ones(4096, dtype=np.float16)])
    got = float(Tensor(data, dtype=float16).sum().numpy())
    assert got == pytest.approx(2048.0 + 4096.0, rel=1e-3)


def test_half_carries_gradients():
    x = Tensor(halves(4, 5), dtype=float16, requires_grad=True)
    (x * x).sum().backward()
    assert x.grad is not None and x.grad.dtype == float16
    np.testing.assert_allclose(np.asarray(x.grad.numpy(), dtype=np.float32), 2 * np.asarray(x.numpy(), np.float32), rtol=2e-3)


def test_a_cast_between_float_dtypes_carries_gradients():
    """Mixed precision is float32 weights met by float16 activations, so the cast between cannot detach."""
    w = Tensor(randf(4, 3), requires_grad=True)
    (w.half() * Tensor(halves(4, 3), dtype=float16)).sum().float().backward()
    assert w.grad is not None, "the cast to float16 detached the parameter from its gradient"
    assert w.grad.dtype == float32, "a gradient has to come back at the width of the leaf it lands on"


def test_a_cast_to_int32_still_detaches():
    """An int carries no gradient, so a graph that goes through one reaches no leaf."""
    x = Tensor(halves(4, 3), dtype=float16, requires_grad=True)
    assert not (x * 4.0).int().half().requires_grad


def test_mixing_a_half_parameter_with_float32_leaves_it_trainable():
    """broadcast_pair inserts the widening cast itself, which is where the gradient used to stop."""
    p = Tensor(halves(4, 3), dtype=float16, requires_grad=True)
    other = Tensor(randf(4, 3))
    (p + other).sum().backward()
    assert p.grad is not None and p.grad.dtype == float16


def test_an_optimizer_does_not_silently_skip_a_half_parameter():
    """AdamW passes over a parameter whose grad is None, so a detached cast would freeze it in place."""
    from limn.optim import AdamW

    p = Tensor(halves(4, 3), dtype=float16, requires_grad=True)
    before = p.numpy().copy()
    opt = AdamW([p], lr=0.1)
    opt.zero_grad()
    (p.float() * 3.0).sum().backward()
    assert p.grad is not None
    opt.step()
    assert not np.array_equal(p.numpy(), before), "the parameter never moved"


@needs_cc
def test_the_c_device_says_it_has_no_half_rather_than_emitting_nonsense():
    a = Tensor(halves(4, 3), dtype=float16)
    with pytest.raises(NotImplementedError, match="float16"):
        CDevice().execute([(a + a).node])


@needs_cc
def test_the_c_device_catches_a_half_it_only_computes_in():
    """A CAST fuses, so this nest reads and writes float32 and is float16 in the middle."""
    x, y = Tensor(randf(4, 3)), Tensor(randf(4, 3))
    with pytest.raises(NotImplementedError, match="float16"):
        CDevice().execute([((x.half() * 2.0).float() + y).node])


# ---- the cuda device against the numpy reference ----


@needs_cuda
@pytest.mark.parametrize("name", list(GRAPHS))
def test_cuda_half_matches_numpy_device(name):
    a, b = Tensor(halves(3, 4), dtype=float16), Tensor(halves(3, 4, seed=1), dtype=float16)
    check_cuda(GRAPHS[name](a, b))


@needs_cuda
@pytest.mark.parametrize("shape", [(64, 96, 128), (192, 192, 192), (129, 40, 65)])
def test_cuda_half_matmul_matches_numpy_device(shape):
    """Big enough to be tiled, and one shape no tile divides, so the tails are covered too.

    Small on m*n*k for the reason test_backend_cuda.py spells out: the oracle materialises it.
    """
    m, k, n = shape
    a, b = Tensor(halves(m, k), dtype=float16), Tensor(halves(k, n, seed=1), dtype=float16)
    check_cuda(a @ b, rtol=2e-2)
    check_cuda(a @ Tensor(halves(n, k, seed=2), dtype=float16).transpose(), rtol=2e-2)


@needs_cuda
def test_cuda_half_split_reduce_matches_numpy_device():
    """Few cells over a long reduce, so it runs as two kernels through a float32 scratch."""
    check_cuda(Tensor(halves(1 << 18, scale=0.01), dtype=float16).sum(), rtol=2e-2)


@needs_cuda
def test_cuda_half_vectorised_reduce_matches_numpy_device():
    a = Tensor(halves(2048, 192), dtype=float16)
    check_cuda(a.sum(-1))
    check_cuda(a.max(-1))
    check_cuda(a.softmax(-1))


@needs_cuda
def test_cuda_half_gradients_match_numpy_device():
    x = Tensor(halves(128, 64), dtype=float16, requires_grad=True)
    w = Tensor(halves(64, 32, seed=1), dtype=float16, requires_grad=True)
    (x @ w).relu().sum().backward()
    check_cuda(x.grad, rtol=2e-2)
    check_cuda(w.grad, rtol=2e-2)


@needs_cuda
def test_cuda_half_scatter_says_it_is_unsupported():
    """A float16 atomic add needs an architecture emission cannot see, so it must not guess."""
    set_device("cuda")
    table = Tensor(halves(6, 4), dtype=float16, requires_grad=True)
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))
    (table.gather_rows(indices) * Tensor(halves(2, 3, 4, seed=1), dtype=float16)).sum().backward()
    with pytest.raises(NotImplementedError, match="float16"):
        table.grad.realize()


@needs_cuda
def test_a_half_buffer_is_half_the_bytes():
    a = Tensor(halves(64, 64), dtype=float16)
    assert a.dtype.itemsize == 2
    assert NUMPY_DTYPES[a.dtype].itemsize == 2
    assert a.numpy().nbytes == 64 * 64 * 2
    assert int32.itemsize == 4
