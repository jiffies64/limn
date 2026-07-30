"""float64: the dtype rules, and the compiled devices against the numpy reference.

float64 is a working precision rather than a storage width: every device computes it natively,
so the corpus diffs at 1e-12, seven digits past the float32 suite, and the width survives
gradients and optimizer state instead of rounding through float32 somewhere on the way.
"""

import numpy as np
import pytest
from conftest import GRAPHS, cdev, check, cudev, needs_cc, needs_cuda

from limn import Tensor, set_device
from limn.ops import float16, float32, float64, int8, int32, promote
from limn.optim import AdamW


def doubles(*shape: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng((seed, *shape)).uniform(-2, 2, shape)


def check64(dev, t: Tensor) -> None:
    check(dev, t, rtol=1e-12, atol=1e-12)


# ---- dtype rules, which hold on every device ----


def test_promotion_widens_everything_to_double():
    assert promote(float64, float32) == float64
    assert promote(float64, float16) == float64
    assert promote(float64, int32) == float64
    assert promote(float64, int8) == float64


def test_arithmetic_between_doubles_stays_double():
    a, b = Tensor(doubles(4, 3), dtype=float64), Tensor(doubles(4, 3, seed=1), dtype=float64)
    assert (a + b).dtype == float64
    assert (a * 2.0).dtype == float64  # a python scalar takes the tensor's dtype
    assert (a / 2).dtype == float64
    assert a.softmax(-1).dtype == float64
    assert (a @ b.transpose()).dtype == float64


def test_double_holds_what_float32_loses():
    tail = 2**-40  # far below float32's 24-bit mantissa, well within float64's 53
    assert (Tensor([1 + tail], dtype=float64) - 1.0).item() * 2**40 == pytest.approx(1.0)
    assert (Tensor([1 + tail], dtype=float32) - 1.0).item() == 0.0


def test_a_double_reduce_keeps_its_own_width():
    data = np.full(1 << 20, 1 + 2**-30)
    got = Tensor(data, dtype=float64).sum().item()
    assert got == pytest.approx(float(data.sum()), rel=1e-15)


def test_double_carries_gradients_at_its_width():
    x = Tensor(doubles(4, 5), dtype=float64, requires_grad=True)
    (x * x).sum().backward()
    assert x.grad is not None and x.grad.dtype == float64
    np.testing.assert_allclose(x.grad.numpy(), 2 * x.numpy(), rtol=1e-15)


def test_a_cast_between_float_dtypes_carries_gradients():
    w = Tensor(doubles(4, 3), dtype=float32, requires_grad=True)
    (w.double() * Tensor(doubles(4, 3, seed=1), dtype=float64)).sum().backward()
    assert w.grad is not None, "the cast to float64 detached the parameter from its gradient"
    assert w.grad.dtype == float32, "a gradient has to come back at the width of the leaf it lands on"


def test_a_cast_through_an_int_still_detaches():
    x = Tensor(doubles(4, 3), dtype=float64, requires_grad=True)
    assert not (x * 4.0).int().double().requires_grad


def test_optimizer_state_keeps_a_double_parameter_wide():
    p = Tensor(doubles(4, 3), dtype=float64, requires_grad=True)
    before = p.numpy().copy()
    opt = AdamW([p], lr=0.1)
    opt.zero_grad()
    (p * 3.0).sum().backward()
    opt.step()
    assert opt.m[0].dtype == float64 and opt.v[0].dtype == float64
    assert p.dtype == float64 and not np.array_equal(p.numpy(), before)


def test_optimizer_state_still_floors_at_float32():
    """The state rule is promote(param, float32): float64 keeps its width, float16 does not shrink it."""
    p = Tensor(doubles(4, 3).astype(np.float16), dtype=float16, requires_grad=True)
    assert AdamW([p]).m[0].dtype == float32


def test_a_numpy_float64_array_still_defaults_to_float32():
    """The wide float is opt-in, like the narrow dtypes: defaults stay at the working widths."""
    assert Tensor(doubles(4, 3)).dtype == float32


def test_double_round_trips_and_is_eight_bytes():
    data = doubles(5, 7)
    t = Tensor(data, dtype=float64)
    np.testing.assert_array_equal(t.numpy(), data)
    assert float64.itemsize == 8
    assert t.numpy().nbytes == 5 * 7 * 8


# ---- the compiled devices against the numpy reference ----


@needs_cc
@pytest.mark.parametrize("name", list(GRAPHS))
def test_c_double_matches_numpy_device(name):
    a, b = Tensor(doubles(3, 4), dtype=float64), Tensor(doubles(3, 4, seed=1), dtype=float64)
    check64(cdev, GRAPHS[name](a, b))


@needs_cuda
@pytest.mark.parametrize("name", list(GRAPHS))
def test_cuda_double_matches_numpy_device(name):
    a, b = Tensor(doubles(3, 4), dtype=float64), Tensor(doubles(3, 4, seed=1), dtype=float64)
    check64(cudev, GRAPHS[name](a, b))


@needs_cuda
@pytest.mark.parametrize("shape", [(64, 96, 128), (192, 192, 192), (129, 40, 65)])
def test_cuda_double_matmul_matches_numpy_device(shape):
    """Big enough to be tiled, and one shape no tile divides, so the tails are covered too."""
    m, k, n = shape
    a, b = Tensor(doubles(m, k), dtype=float64), Tensor(doubles(k, n, seed=1), dtype=float64)
    check64(cudev, a @ b)
    check64(cudev, a @ Tensor(doubles(n, k, seed=2), dtype=float64).transpose())


@needs_cuda
def test_cuda_double_split_reduce_matches_numpy_device():
    """Few cells over a long reduce runs as two kernels; the partials stay double throughout."""
    check64(cudev, Tensor(doubles(1 << 18), dtype=float64).sum())


@needs_cuda
def test_cuda_double_scatter_adds_atomically():
    """GATHER's gradient collides threads on repeated rows, which lands on atomicAdd in double."""
    indices_data = np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32)
    weights_data = doubles(2, 3, 4, seed=1)
    set_device("cuda")
    table = Tensor(doubles(6, 4), dtype=float64, requires_grad=True)
    indices = Tensor(indices_data)
    (table.gather_rows(indices) * Tensor(weights_data, dtype=float64)).sum().backward()
    got = table.grad.numpy()
    expected = np.zeros((6, 4))
    np.add.at(expected, indices_data, weights_data)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
