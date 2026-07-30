"""Shared test helpers: the devices, the check that diffs them against the numpy reference,
and the graph corpus every backend and the lowered IR are run through."""

import numpy as np
import pytest

from limn.backend_c import CDevice, has_cc
from limn.backend_cuda import CudaDevice, has_cuda
from limn.device import NUMPY_DTYPES
from limn.tensor import Tensor

rng = np.random.default_rng(7)

cdev = CDevice() if has_cc() else None
cudev = CudaDevice() if has_cuda() else None
needs_cc = pytest.mark.skipif(not has_cc(), reason="no C compiler found")
needs_cuda = pytest.mark.skipif(not has_cuda(), reason="no CUDA driver, device, or NVRTC found")


def check(dev, t: Tensor, rtol: float = 1e-5, atol: float = 1e-5, exact: bool = False) -> None:
    """One tensor realized on `dev`, diffed against the numpy reference.

    exact is for the dtypes whose arithmetic is bit-defined on every device, like the ints
    wrapping modulo 2**width. float16 results are compared as float32, since comparing in half
    rounds the comparison itself.

    The device under test runs first: .numpy() realizes, and realize retires the sink to a
    buffer, so the other order would hand `dev` the reference's bytes and diff nothing.
    """
    got = dev.copyout(dev.execute([t.node])[0]).view(NUMPY_DTYPES[t.dtype]).reshape(t.shape)
    expected = t.numpy()
    if exact:
        np.testing.assert_array_equal(got, expected)
        return
    if got.dtype == np.float16:
        got, expected = got.astype(np.float32), expected.astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


def randf(*shape: int) -> np.ndarray:
    return rng.uniform(-2, 2, shape).astype(np.float32)


GRAPHS = {
    "elementwise": lambda a, b: (a + b) * 2.0 - a,
    "relu": lambda a, b: (a * b).relu() + 1.0,
    "unary math": lambda a, b: (a * a + 1.0).log().exp().sqrt(),
    "divide": lambda a, b: a / (b * b + 0.5),
    "where": lambda a, b: (a < b).where(a, b),
    "cast round trip": lambda a, b: (a * 4.0).int().float() + 0.5,
    "sum an axis": lambda a, b: (a + b).sum(axis=1),
    "sum everything": lambda a, b: (a * b).sum(),
    "max keepdim": lambda a, b: (a * b).max(axis=0, keepdim=True),
    "mean": lambda a, b: (a + b).mean(axis=1),
    "matmul": lambda a, b: a @ b.transpose(),
    "softmax": lambda a, b: (a + b).softmax(axis=1),
    "log softmax": lambda a, b: (a * b).log_softmax(axis=0),
    "padded reduce": lambda a, b: a.pad(((1, 1), (0, 2))).sum(axis=1),
    "movement then reduce": lambda a, b: (a.transpose() * 2.0).sum(axis=0),
    "reduce then elementwise": lambda a, b: (a.sum(axis=1, keepdim=True) * b).relu(),
}
