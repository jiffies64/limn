"""Device-level behaviour: cache invalidation around assigns, and selecting a backend."""

import numpy as np
import pytest

from limn import Tensor, device, realize, set_device, set_seed
from limn.backend_c import has_cc
from limn.ops import Op


@pytest.fixture(autouse=True)
def numpy_device():
    """Every test here may switch devices; none of them may leak that to the next file."""
    set_device("numpy")
    yield
    set_device("numpy")


def test_assign_invalidates_results_cached_before_it():
    """An interior value the interpreter cached must not survive an assign to what it read.

    `doubled` stays an unrealized expression throughout: each read realizes a fresh sink built
    on it. A sink realized *before* the assign keeps its bytes instead, which is
    test_realize_retires_a_computed_sink_to_its_bytes.
    """
    t = Tensor(np.ones((2, 2), dtype=np.float32))
    doubled = t * 2.0
    np.testing.assert_array_equal((doubled + 1.0).numpy(), np.full((2, 2), 3.0, dtype=np.float32))
    t.assign(t + 10.0).realize()
    np.testing.assert_array_equal((doubled + 1.0).numpy(), np.full((2, 2), 23.0, dtype=np.float32))


def test_reads_in_the_assign_batch_still_see_pre_assign_bytes():
    t = Tensor(np.ones((2, 2), dtype=np.float32))
    before = t * 10.0
    t.assign(t + 100.0)
    buffers = device.active().execute([before.node, t.node])
    got_before, got_after = (b.view(np.float32).reshape(2, 2) for b in buffers)
    np.testing.assert_array_equal(got_before, np.full((2, 2), 10.0, dtype=np.float32))
    np.testing.assert_array_equal(got_after, np.full((2, 2), 101.0, dtype=np.float32))


def test_an_assign_whose_value_is_a_plain_buffer_still_reads_pre_assign_bytes():
    """The value of one assign may be a buffer another assign in the same batch overwrites."""
    a = Tensor(np.array([1.0], dtype=np.float32))
    b = Tensor(np.array([0.0], dtype=np.float32))
    a_before = a.detach()
    a.assign(a_before * 0.0 + 7.0)
    b.assign(a_before)  # b takes a's *old* value, whichever order the two commit in
    realize(a, b)
    np.testing.assert_array_equal(a.numpy(), np.array([7.0], dtype=np.float32))
    np.testing.assert_array_equal(b.numpy(), np.array([1.0], dtype=np.float32))


def test_realize_retires_a_computed_sink_to_its_bytes():
    a = Tensor(np.array([2.0], dtype=np.float32))
    b = a + 1.0
    b.realize()
    assert b.node.op is Op.BUFFER
    a.assign(Tensor(np.array([10.0], dtype=np.float32))).realize()
    np.testing.assert_array_equal(b.numpy(), np.array([3.0], dtype=np.float32))  # realized bytes, not a recompute


def test_realize_keeps_an_alias_of_a_buffer_as_a_view():
    a = Tensor(np.arange(4, dtype=np.float32))
    r = a.reshape(2, 2)
    r.realize()
    assert r.node.op is Op.VIEW  # no kernel ran and no second buffer node now covers a's bytes
    np.testing.assert_array_equal(r.numpy(), np.arange(4, dtype=np.float32).reshape(2, 2))


def test_set_device_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown device"):
        set_device("tpu")
    assert type(device.active()).__name__ == "NumpyDevice"


@pytest.mark.skipif(not has_cc(), reason="no C compiler found")
def test_set_device_switches_the_backend_a_graph_runs_on():
    a = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    b = Tensor(np.ones((3, 2), dtype=np.float32))
    expected = np.arange(6, dtype=np.float32).reshape(2, 3) @ np.ones((3, 2), dtype=np.float32)

    set_device("c")
    assert type(device.active()).__name__ == "CDevice"
    np.testing.assert_allclose((a @ b).numpy(), expected, atol=1e-6)

    set_device("numpy")
    assert type(device.active()).__name__ == "NumpyDevice"
    np.testing.assert_allclose((a @ b).numpy(), expected, atol=1e-6)


@pytest.mark.skipif(not has_cc(), reason="no C compiler found")
def test_an_optimizer_step_runs_on_the_c_device():
    from limn.nn import Linear, parameters
    from limn.optim import SGD

    set_device("c")
    set_seed(3)  # whether five momentum steps end below the start depends on the init, so own it
    layer = Linear(4, 3)
    x = Tensor(np.random.default_rng(3).random((8, 4)).astype(np.float32))
    losses = []
    opt = SGD(parameters(layer), lr=0.05, momentum=0.9)
    for _ in range(5):
        opt.zero_grad()
        loss = (layer(x) ** 2).sum()
        loss.backward()
        losses.append(loss.item())
        opt.step()
    assert losses[-1] < losses[0]
