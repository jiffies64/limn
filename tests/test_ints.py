"""int8 and int16: the dtype rules, and the compiled devices against the numpy reference.

The narrow ints are exact storage, not approximations: arithmetic wraps modulo 2**width, the
same on every device, so everything here is diffed for equality rather than to a tolerance.
"""

import numpy as np
import pytest

from limn import Tensor
from limn.backend_c import CDevice, has_cc
from limn.backend_cuda import CudaDevice, has_cuda
from limn.device import NUMPY_DTYPES
from limn.ops import DType, float16, float32, int8, int16, int32, promote
from limn.tensor import scatter_rows

cdev = CDevice() if has_cc() else None
cudev = CudaDevice() if has_cuda() else None
needs_cc = pytest.mark.skipif(not has_cc(), reason="no C compiler found")
needs_cuda = pytest.mark.skipif(not has_cuda(), reason="no CUDA driver, device, or NVRTC found")

NARROW = {int8: np.int8, int16: np.int16}


def ints(*shape: int, dtype: DType = int8, seed: int = 0) -> np.ndarray:
    """Values wide enough that sums and products actually wrap, drawn per (seed, shape)."""
    return np.random.default_rng((seed, *shape)).integers(-100, 100, shape, dtype=NARROW[dtype])


def check(dev, t: Tensor) -> None:
    expected = t.numpy()
    bufs = dev.execute([t.node])
    got = dev.copyout(bufs[0]).view(NUMPY_DTYPES[t.dtype]).reshape(t.shape)
    np.testing.assert_array_equal(got, expected)


GRAPHS = {
    "elementwise": lambda a, b: (a + b) * 2 - a,
    "wraparound": lambda a, b: a * b * 3 + 100,
    "compare and pick": lambda a, b: (a < b).where(a, b),
    "relu": lambda a, b: (a * b).relu(),
    "sum an axis": lambda a, b: (a + b).sum(axis=1),
    "sum everything": lambda a, b: (a * b).sum(),
    "max keepdim": lambda a, b: (a * b).max(axis=0, keepdim=True),
    "matmul": lambda a, b: a @ b.transpose(),
    "cast round trip": lambda a, b: (a.int() * 300 + b.int()).cast(a.dtype),
    "movement then reduce": lambda a, b: (a.transpose() * 2).sum(axis=0),
}


# ---- dtype rules, which hold on every device ----


def test_promotion_takes_the_wider_int_and_floats_win_at_float32_or_wider():
    assert promote(int8, int8) == int8
    assert promote(int8, int16) == int16
    assert promote(int16, int32) == int32
    assert promote(int8, float16) == float32
    assert promote(int16, float32) == float32


def test_arithmetic_between_narrow_ints_stays_narrow():
    a, b = Tensor(ints(4, 3), dtype=int8), Tensor(ints(4, 3, seed=1), dtype=int8)
    assert (a + b).dtype == int8
    assert (a * b).dtype == int8
    assert (a - b).dtype == int8
    assert (a @ b.transpose()).dtype == int8
    assert (a < b).dtype == int8
    assert a.sum().dtype == int8


def test_a_python_scalar_takes_the_tensor_dtype_and_a_float_scalar_widens():
    a = Tensor(ints(4, 3, dtype=int16), dtype=int16)
    assert (a + 3).dtype == int16
    assert (a * 2).dtype == int16
    assert (a + 0.5).dtype == float32
    assert (a / 2).dtype == float32  # division is a reciprocal, so it is float whatever comes in


def test_narrow_arithmetic_wraps_exactly_like_numpy():
    data, other = ints(5, 7), ints(5, 7, seed=1)
    got = (Tensor(data, dtype=int8) * Tensor(other, dtype=int8) + 100).numpy()
    np.testing.assert_array_equal(got, data * other + np.int8(100))


def test_a_narrow_sum_wraps_at_its_own_width():
    assert Tensor(np.ones(300, np.int8), dtype=int8).sum().item() == 44  # 300 mod 256
    assert Tensor(np.full(5000, 100, np.int16), dtype=int16).sum().item() == np.array(500000).astype(np.int16).item()


def test_narrow_ints_round_trip_and_are_narrow_in_memory():
    data = ints(5, 7)
    t = Tensor(data, dtype=int8)
    np.testing.assert_array_equal(t.numpy(), data)
    assert t.numpy().nbytes == 5 * 7
    assert int8.itemsize == 1 and int16.itemsize == 2


def test_a_numpy_int8_array_still_defaults_to_int32():
    """Narrow storage is opt-in, like float16: defaults stay at the working widths."""
    assert Tensor(ints(4, 3)).dtype == int32


def test_requires_grad_still_needs_a_float_dtype():
    with pytest.raises(ValueError, match="requires_grad"):
        Tensor(ints(4, 3), dtype=int8, requires_grad=True)


def test_gather_reads_a_narrow_table_and_scatter_sums_into_one():
    table = Tensor(ints(6, 4), dtype=int8)
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))
    np.testing.assert_array_equal(table.gather_rows(indices).numpy(), table.numpy()[indices.numpy()])
    values = Tensor(ints(2, 3, 4, seed=1), dtype=int8)
    expected = np.zeros((6, 4), np.int8)
    np.add.at(expected, indices.numpy(), values.numpy())
    np.testing.assert_array_equal(scatter_rows(values, indices, (6, 4)).numpy(), expected)


# ---- the compiled devices against the numpy reference ----


@needs_cc
@pytest.mark.parametrize("dtype", [int8, int16], ids=str)
@pytest.mark.parametrize("name", list(GRAPHS))
def test_c_narrow_matches_numpy_device(name, dtype):
    a, b = Tensor(ints(3, 4, dtype=dtype), dtype=dtype), Tensor(ints(3, 4, dtype=dtype, seed=1), dtype=dtype)
    check(cdev, GRAPHS[name](a, b))


@needs_cc
def test_c_narrow_scatter_matches_numpy_device():
    values = Tensor(ints(2, 3, 4), dtype=int8)
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))
    check(cdev, scatter_rows(values, indices, (6, 4)))


@needs_cuda
@pytest.mark.parametrize("dtype", [int8, int16], ids=str)
@pytest.mark.parametrize("name", list(GRAPHS))
def test_cuda_narrow_matches_numpy_device(name, dtype):
    a, b = Tensor(ints(3, 4, dtype=dtype), dtype=dtype), Tensor(ints(3, 4, dtype=dtype, seed=1), dtype=dtype)
    check(cudev, GRAPHS[name](a, b))


@needs_cuda
def test_cuda_narrow_split_reduce_matches_numpy_device():
    """Few cells over a long reduce runs as two kernels; modulo addition makes it exact anyway."""
    check(cudev, Tensor(ints(1 << 18), dtype=int8).sum())


@needs_cuda
def test_cuda_narrow_matmul_is_exact_where_it_tiles():
    """Big enough to stage through shared memory, and one shape no tile divides."""
    for m, k, n in [(64, 96, 128), (129, 40, 65)]:
        a = Tensor(ints(m, k, dtype=int16), dtype=int16)
        b = Tensor(ints(k, n, dtype=int16, seed=1), dtype=int16)
        check(cudev, a @ b)


@needs_cuda
@pytest.mark.parametrize("dtype", [int8, int16], ids=str)
def test_cuda_narrow_scatter_says_it_is_unsupported(dtype):
    """atomicAdd has no 8- or 16-bit overload, so the emitter declines rather than racing."""
    values = Tensor(ints(2, 3, 4, dtype=dtype), dtype=dtype)
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))
    with pytest.raises(NotImplementedError, match=str(dtype)):
        cudev.execute([scatter_rows(values, indices, (6, 4)).node])
