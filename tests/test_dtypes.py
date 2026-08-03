"""The dtypes past the defaults: their rules, and the compiled devices against the numpy reference.

float16 is a storage width, so every device widens it to compute and a reduce totals in float32;
its tolerances are float16's own rounding rather than the 1e-5 the float32 corpus is diffed at.
bfloat16 is the other storage width: the same rules, float32's range, and a coarser mantissa, so
its tolerances are eight times wider still. float64 is a working precision every device computes
natively, seven digits tighter. int8 and int16 are exact storage whose arithmetic wraps modulo
2**width identically everywhere, so they are diffed for equality. What differs per dtype is the
table below; the tests it feeds are shared.
"""

from functools import partial
from typing import Any, Callable, NamedTuple

import ml_dtypes
import numpy as np
import pytest
from conftest import BACKENDS, GRAPHS, check, cdev, cudev, needs_cc, needs_cuda, randf

from limn import Tensor, set_device
from limn.device import NUMPY_DTYPES
from limn.ops import DType, INTS, bfloat16, float16, float32, float64, int8, int16, int32, promote
from limn.optim import AdamW
from limn.tensor import scatter_rows

NARROW = {int8: np.int8, int16: np.int16}
BFLOAT16 = np.dtype(ml_dtypes.bfloat16)


def halves(*shape: int, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Values in [0, scale), which is where float16 keeps enough precision to diff at 2e-3.

    Two operands of one test need different seeds. Drawn from the shape alone they come out
    identical, and a graph over a == b stops telling a + b from 2a.
    """
    return (np.random.default_rng((seed, *shape)).random(shape).astype(np.float32) * scale).astype(np.float16)


def bfloats(*shape: int, seed: int = 0) -> np.ndarray:
    """Values in [0, 1): bfloat16's 7 mantissa bits resolve them, and its range never clips them."""
    return np.random.default_rng((seed, *shape)).random(shape).astype(np.float32).astype(BFLOAT16)


def doubles(*shape: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng((seed, *shape)).uniform(-2, 2, shape)


def ints(*shape: int, dtype: DType = int8, seed: int = 0) -> np.ndarray:
    """Values wide enough that sums and products actually wrap, drawn per (seed, shape)."""
    return np.random.default_rng((seed, *shape)).integers(-100, 100, shape, dtype=NARROW[dtype])


INT_GRAPHS = {  # the corpus without the ops an int has no answer for: exp, log, sqrt, divide
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


class Spec(NamedTuple):
    gen: Callable[..., np.ndarray]  # values for this dtype, drawn per (seed, shape)
    graphs: dict  # the corpus it can run
    devices: tuple[str, ...]  # the compiled devices that have this dtype at all
    tol: dict[str, Any]  # what check() is given for a corpus-sized graph
    big_tol: dict[str, Any]  # and for a long reduce or a tiled matmul, where the error has room to grow
    long_gen: Callable[..., np.ndarray] | None = None  # values small enough that a 2**18 reduce still fits


EXACT = {"exact": True}
BOTH = ("c", "cuda")
SPECS = {
    float16: Spec(
        halves, GRAPHS, ("cuda",), {"rtol": 2e-3, "atol": 1e-3}, {"rtol": 2e-2, "atol": 1e-3}, partial(halves, scale=0.01)
    ),
    # bfloat16's eps is 2**-8 where float16's is 2**-10, so both tolerances are eight times wider;
    # the c device declines it as it declines float16, so cuda is its one compiled home
    bfloat16: Spec(bfloats, GRAPHS, ("cuda",), {"rtol": 1e-2, "atol": 1e-2}, {"rtol": 4e-2, "atol": 1e-2}),
    float64: Spec(doubles, GRAPHS, BOTH, {"rtol": 1e-12, "atol": 1e-12}, {"rtol": 1e-12, "atol": 1e-12}),
    int8: Spec(partial(ints, dtype=int8), INT_GRAPHS, BOTH, EXACT, EXACT),
    int16: Spec(partial(ints, dtype=int16), INT_GRAPHS, BOTH, EXACT, EXACT),
}

CORPUS = [
    pytest.param(BACKENDS[name].shared, dtype, graph, id=f"{name}-{dtype}-{graph}", marks=BACKENDS[name].mark)
    for dtype, spec in SPECS.items()
    for name in spec.devices
    for graph in spec.graphs
]
ON_CUDA = [pytest.param(dtype, id=str(dtype), marks=needs_cuda) for dtype, spec in SPECS.items() if "cuda" in spec.devices]
TILED = [pytest.param(dtype, id=str(dtype), marks=needs_cuda) for dtype in (float16, bfloat16, float64, int16)]


def tensor(dtype: DType, *shape: int, seed: int = 0, requires_grad: bool = False) -> Tensor:
    return Tensor(SPECS[dtype].gen(*shape, seed=seed), dtype=dtype, requires_grad=requires_grad)


# ---- dtype rules, which hold on every device ----


def test_promotion_takes_the_wider_side_and_a_float_beats_an_int():
    assert promote(float16, float16) == float16 and promote(bfloat16, bfloat16) == bfloat16 and promote(int8, int8) == int8
    assert promote(float16, float32) == float32 and promote(int8, int16) == int16 and promote(int16, int32) == int32
    assert promote(float64, float32) == float64 and promote(float64, float16) == float64
    assert promote(float16, int32) == float32 and promote(int8, float16) == float32 and promote(int16, float32) == float32
    assert promote(float64, int32) == float64 and promote(float64, int8) == float64


def test_the_two_half_width_floats_meet_at_float32():
    """float16 keeps mantissa and bfloat16 keeps range, so neither can hold the other's numbers."""
    assert promote(float16, bfloat16) == float32 and promote(bfloat16, float16) == float32
    assert promote(bfloat16, float32) == float32 and promote(bfloat16, float64) == float64
    assert promote(bfloat16, int8) == float32 and promote(int32, bfloat16) == float32


@pytest.mark.parametrize("dtype", list(SPECS), ids=str)
def test_arithmetic_between_two_of_this_dtype_stays_in_it(dtype):
    a, b = tensor(dtype, 4, 3), tensor(dtype, 4, 3, seed=1)
    for out in (a + b, a * b, a - b, a @ b.transpose()):
        assert out.dtype == dtype


@pytest.mark.parametrize("dtype", list(SPECS), ids=str)
def test_a_python_scalar_takes_the_tensor_dtype_and_only_an_int_widens(dtype):
    """A reflected divide has to ask promote too, or the same literal widens on one side only."""
    a = tensor(dtype, 4, 3)
    assert (a + 3).dtype == dtype and (a * 2).dtype == dtype
    mixed = float32 if dtype in INTS else dtype  # a float scalar joins the floats where an int cannot follow
    assert (a + 0.5).dtype == mixed
    assert (a / 2).dtype == mixed and (2.0 / a).dtype == mixed  # a divide is a reciprocal, so it is never an int


@pytest.mark.parametrize("dtype", list(SPECS), ids=str)
def test_mixing_with_a_float32_tensor_promotes(dtype):
    assert (tensor(dtype, 4, 3) + Tensor(randf(4, 3))).dtype == promote(dtype, float32)


@pytest.mark.parametrize("dtype", list(SPECS), ids=str)
def test_round_trips_through_the_numpy_device_at_its_own_width(dtype):
    data = SPECS[dtype].gen(5, 7)
    t = Tensor(data, dtype=dtype)
    np.testing.assert_array_equal(t.numpy(), data)
    assert NUMPY_DTYPES[dtype].itemsize == dtype.itemsize
    assert t.numpy().nbytes == 5 * 7 * dtype.itemsize


@pytest.mark.parametrize(
    "dtype,default", [(float16, float32), (bfloat16, float32), (float64, float32), (int8, int32), (int16, int32)], ids=str
)
def test_a_numpy_array_of_this_dtype_still_defaults_to_the_working_width(dtype, default):
    """The other widths are opt-in: what the array holds does not change what Tensor picks."""
    assert Tensor(SPECS[dtype].gen(4, 3)).dtype == default


@pytest.mark.parametrize("dtype,rtol", [(float16, 2e-3), (bfloat16, 2e-2), (float64, 1e-15)], ids=str)
def test_a_float_dtype_carries_gradients_at_its_own_width(dtype, rtol):
    x = tensor(dtype, 4, 5, requires_grad=True)
    (x * x).sum().backward()
    assert x.grad is not None and x.grad.dtype == dtype
    np.testing.assert_allclose(x.grad.numpy().astype(np.float64), 2 * x.numpy().astype(np.float64), rtol=rtol)


@pytest.mark.parametrize("dtype", [float16, bfloat16, float64], ids=str)
def test_a_cast_between_float_dtypes_carries_gradients(dtype):
    """Mixed precision is float32 weights met by another width, so the cast between cannot detach."""
    w = Tensor(randf(4, 3), requires_grad=True)
    (w.cast(dtype) * tensor(dtype, 4, 3, seed=1)).sum().backward()
    assert w.grad is not None, "the cast detached the parameter from its gradient"
    assert w.grad.dtype == float32, "a gradient has to come back at the width of the leaf it lands on"


@pytest.mark.parametrize("dtype", [float16, bfloat16, float64], ids=str)
def test_a_cast_through_an_int_still_detaches(dtype):
    """An int carries no gradient, so a graph that goes through one reaches no leaf."""
    assert not (tensor(dtype, 4, 3, requires_grad=True) * 4.0).int().cast(dtype).requires_grad


@pytest.mark.parametrize("dtype", [float16, bfloat16, float64], ids=str)
def test_mixing_a_parameter_with_float32_leaves_it_trainable(dtype):
    """broadcast_pair inserts the widening cast itself, which is where the gradient used to stop."""
    p = tensor(dtype, 4, 3, requires_grad=True)
    (p + Tensor(randf(4, 3))).sum().backward()
    assert p.grad is not None and p.grad.dtype == dtype


@pytest.mark.parametrize("dtype,state", [(float16, float32), (bfloat16, float32), (float64, float64)], ids=str)
def test_an_optimizer_moves_a_parameter_of_this_dtype_and_keeps_its_state_wide(dtype, state):
    """State is promote(param, float32): a half does not shrink it, a double keeps its width. AdamW
    passes over a parameter whose grad is None, so a detached cast would freeze it in place."""
    p = tensor(dtype, 4, 3, requires_grad=True)
    before = p.numpy().copy()
    opt = AdamW([p], lr=0.1)
    assert opt.m[0].dtype == state and opt.v[0].dtype == state
    opt.zero_grad()
    (p.float() * 3.0).sum().backward()
    assert p.grad is not None
    opt.step()
    assert p.dtype == dtype and not np.array_equal(p.numpy(), before), "the parameter never moved"


# ---- what one dtype alone has to get right ----


def test_a_half_reduce_totals_in_float32():
    """Summed in float16 the tail of this vector would vanish into the running total."""
    data = np.concatenate([np.array([2048.0], dtype=np.float16), np.ones(4096, dtype=np.float16)])
    assert float(Tensor(data, dtype=float16).sum().numpy()) == pytest.approx(2048.0 + 4096.0, rel=1e-3)


def test_a_bfloat16_reduce_totals_in_float32():
    """bf16's 7-bit mantissa stalls even sooner: at 256 the ulp is 2, so 0.5 cannot join a bf16 total."""
    data = np.concatenate([np.array([256.0], dtype=BFLOAT16), np.full(4096, 0.5, dtype=BFLOAT16)])
    assert float(Tensor(data, dtype=bfloat16).sum().numpy()) == pytest.approx(256.0 + 2048.0, rel=1e-2)


def test_double_holds_what_float32_loses():
    tail = 2**-40  # far below float32's 24-bit mantissa, well within float64's 53
    assert (Tensor([1 + tail], dtype=float64) - 1.0).item() * 2**40 == pytest.approx(1.0)
    assert (Tensor([1 + tail], dtype=float32) - 1.0).item() == 0.0


def test_a_double_reduce_keeps_its_own_width():
    data = np.full(1 << 20, 1 + 2**-30)
    assert Tensor(data, dtype=float64).sum().item() == pytest.approx(float(data.sum()), rel=1e-15)


def test_narrow_arithmetic_wraps_exactly_like_numpy():
    data, other = ints(5, 7), ints(5, 7, seed=1)
    got = (Tensor(data, dtype=int8) * Tensor(other, dtype=int8) + 100).numpy()
    np.testing.assert_array_equal(got, data * other + np.int8(100))


def test_a_narrow_sum_wraps_at_its_own_width():
    assert Tensor(np.ones(300, np.int8), dtype=int8).sum().item() == 44  # 300 mod 256
    assert Tensor(np.full(5000, 100, np.int16), dtype=int16).sum().item() == np.array(500000).astype(np.int16).item()


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


@needs_cc
@pytest.mark.parametrize("dtype", [float16, bfloat16], ids=str)
def test_the_c_device_says_it_has_no_half_width_float_rather_than_emitting_nonsense(dtype):
    a = tensor(dtype, 4, 3)
    assert cdev is not None
    with pytest.raises(NotImplementedError, match=str(dtype)):
        cdev.execute([(a + a).node])


@needs_cc
@pytest.mark.parametrize("dtype", [float16, bfloat16], ids=str)
def test_the_c_device_catches_a_half_width_float_it_only_computes_in(dtype):
    """A CAST fuses, so this nest reads and writes float32 and is half-width in the middle."""
    x, y = Tensor(randf(4, 3)), Tensor(randf(4, 3))
    assert cdev is not None
    with pytest.raises(NotImplementedError, match=str(dtype)):
        cdev.execute([((x.cast(dtype) * 2.0).float() + y).node])


# ---- the compiled devices against the numpy reference ----


@pytest.mark.parametrize("dev,dtype,graph", CORPUS)
def test_the_corpus_matches_the_numpy_device(dev, dtype, graph):
    spec = SPECS[dtype]
    a, b = tensor(dtype, 3, 4), tensor(dtype, 3, 4, seed=1)
    check(dev, spec.graphs[graph](a, b), **spec.tol)


@pytest.mark.parametrize("dtype", ON_CUDA)
def test_a_long_reduce_that_splits_matches_the_numpy_device(dtype):
    """Few cells over a long reduce runs as two kernels, through a scratch at the accumulator's width."""
    spec = SPECS[dtype]
    data = (spec.long_gen or spec.gen)(1 << 18)
    check(cudev, Tensor(data, dtype=dtype).sum(), **spec.big_tol)


@pytest.mark.parametrize("m,k,n", [(64, 96, 128), (192, 192, 192), (129, 40, 65)])
@pytest.mark.parametrize("dtype", TILED)
def test_a_matmul_big_enough_to_tile_matches_the_numpy_device(dtype, m, k, n):
    """Big enough to be tiled, and one shape no tile divides, so the tails are covered too.

    Small on m*n*k for the reason test_backend_cuda.py spells out: the oracle materialises it.
    """
    tol = SPECS[dtype].big_tol
    a = tensor(dtype, m, k)
    check(cudev, a @ tensor(dtype, k, n, seed=1), **tol)
    check(cudev, a @ tensor(dtype, n, k, seed=2).transpose(), **tol)


@needs_cuda
@pytest.mark.parametrize("dtype", [float16, bfloat16], ids=str)
def test_a_half_width_reduce_reads_four_elements_at_a_time(dtype):
    a = tensor(dtype, 2048, 192)
    for t in (a.sum(-1), a.max(-1), a.softmax(-1)):
        check(cudev, t, **SPECS[dtype].tol)


@needs_cuda
@pytest.mark.parametrize("dtype", [float16, bfloat16], ids=str)
def test_cuda_half_width_gradients_match_numpy_device(dtype):
    x = tensor(dtype, 128, 64, requires_grad=True)
    w = tensor(dtype, 64, 32, seed=1, requires_grad=True)
    (x @ w).relu().sum().backward()
    assert x.grad is not None and w.grad is not None
    check(cudev, x.grad, **SPECS[dtype].big_tol)
    check(cudev, w.grad, **SPECS[dtype].big_tol)


@needs_cuda
def test_cuda_scatter_adds_doubles_atomically():
    """GATHER's gradient collides threads on repeated rows, which lands on atomicAdd in double."""
    indices_data = np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32)
    weights = doubles(2, 3, 4, seed=1)
    set_device("cuda")
    try:
        table = tensor(float64, 6, 4, requires_grad=True)
        (table.gather_rows(Tensor(indices_data)) * Tensor(weights, dtype=float64)).sum().backward()
        assert table.grad is not None
        got = table.grad.numpy()
    finally:
        set_device("numpy")
    expected = np.zeros((6, 4))
    np.add.at(expected, indices_data, weights)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@needs_cuda
@pytest.mark.parametrize("dtype", [float16, bfloat16, int8, int16], ids=str)
def test_cuda_says_which_dtypes_it_cannot_scatter(dtype):
    """atomicAdd has no 8- or 16-bit overload, and a half-width one needs an architecture emission cannot see."""
    values = tensor(dtype, 2, 3, 4)
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))
    assert cudev is not None
    with pytest.raises(NotImplementedError, match=str(dtype)):
        cudev.execute([scatter_rows(values, indices, (6, 4)).node])


@needs_cc
def test_the_c_device_scatters_narrow_ints():
    values = Tensor(ints(2, 3, 4), dtype=int8)
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))
    check(cdev, scatter_rows(values, indices, (6, 4)), exact=True)
