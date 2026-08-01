"""What every compiled backend owes, run once per backend: the graph corpus, assigns and their
deferral, gather, and the plan cache. All of it is diffed against the numpy reference.

What belongs to one backend stays in its own file: cc's loop splitting in test_backend_c.py,
cuda's tiles, pool and atomics in test_backend_cuda.py.
"""

import numpy as np
import pytest
from conftest import COMPILED, GRAPHS, check, randf, read

from limn import Tensor, device, set_device, set_seed
from limn.nn import Linear, parameters
from limn.optim import SGD, AdamW

pytestmark = pytest.mark.parametrize("backend", COMPILED)


@pytest.fixture
def on_backend(backend):
    """The active device, for the tests that drive the whole stack (realize, optimizers) onto it."""
    set_device(backend.name)
    yield device.active()
    set_device("numpy")


@pytest.mark.parametrize("name", list(GRAPHS))
def test_the_corpus_matches_the_numpy_device(backend, name):
    a, b = Tensor(randf(3, 4)), Tensor(randf(3, 4))
    check(backend.shared, GRAPHS[name](a, b))


def test_matmul_4x5_5x3(backend):
    check(backend.shared, Tensor(randf(4, 5)) @ Tensor(randf(5, 3)))


def test_backward_pass(backend):
    x = Tensor(randf(4, 5), requires_grad=True)
    w = Tensor(randf(5, 3), requires_grad=True)
    loss = (x @ w).relu().sum()
    loss.backward()
    assert x.grad is not None and w.grad is not None
    for t in (loss, w.grad, x.grad):
        check(backend.shared, t)


def test_multi_kernel_chain(backend):
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    check(backend.shared, (a + b).reshape(6) * 2.0)


def test_contiguous_copy(backend):
    check(backend.shared, Tensor(randf(3, 4)).transpose().reshape(12))


def test_shared_subgraph(backend):
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    shared = (a + b).sum(axis=1, keepdim=True)
    check(backend.shared, shared * 2.0)
    check(backend.shared, shared * 3.0)


def test_gather_rows_forward_and_backward(backend):
    table = Tensor(randf(6, 4), requires_grad=True)
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))  # repeats, so the scatter accumulates
    gathered = table.gather_rows(indices)
    check(backend.shared, gathered)
    (gathered * Tensor(randf(2, 3, 4))).sum().backward()
    assert table.grad is not None
    check(backend.shared, table.grad)


def test_assign_deferred(backend):
    dev = backend.shared
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    expected = p.numpy() * 2.0 + 1.0
    p.assign(p * 2.0 + 1.0)
    np.testing.assert_allclose(read(dev, dev.execute([p.node])[0], p), expected, atol=1e-6)


def test_assign_deferral_reads_pre_assign_bytes(backend):
    """A graph built before the assign, realized in the same batch, sees old values."""
    dev = backend.shared
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    old_value = p * 10.0
    p.assign(p + 100.0)
    old_buf, new_buf = dev.execute([old_value.node, p.node])
    np.testing.assert_allclose(read(dev, old_buf, old_value), [[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose(read(dev, new_buf, p), [[101.0, 102.0], [103.0, 104.0]])


def test_assign_consumed_as_value(backend):
    """An ASSIGN node read by a later kernel yields the new value, not the old buffer."""
    dev = backend.shared
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    p.assign(p * 2.0)
    consumer = p + 1.0
    np.testing.assert_allclose(read(dev, dev.execute([consumer.node])[0], consumer), [[3.0, 5.0], [7.0, 9.0]])


def test_a_repeated_graph_reuses_the_plan_and_reads_fresh_bytes(backend):
    dev = backend.make()
    for x in (randf(3, 4), randf(3, 4)):
        check(dev, (Tensor(x) * 2.0).sum(axis=1))
    assert len(dev.plans) == 1


def test_a_changed_constant_is_a_different_plan(backend):
    dev = backend.make()
    x = randf(2, 3)
    for scale in (2.0, 3.0):
        check(dev, Tensor(x) * scale)
    assert len(dev.plans) == 2


def test_a_repeated_assign_commits_through_the_cached_plan(on_backend):
    p = Tensor(np.ones(4, dtype=np.float32))
    for _ in range(3):
        p.assign(p * 2.0)
        p.realize()
    np.testing.assert_allclose(p.numpy(), np.full(4, 8.0, dtype=np.float32))
    assert len(on_backend.plans) == 2  # one plan for the assign step, one for numpy()'s read


def test_adamw_compiles_one_program_for_all_steps(backend, on_backend):
    """AdamW's bias correction changes every step, and a literal would rehash the source with it."""
    p = Tensor(np.ones((4, 4), dtype=np.float32), requires_grad=True)
    opt = AdamW([p], lr=0.1)
    backend.cache.clear()
    for _ in range(3):
        p.grad = Tensor(np.ones((4, 4), dtype=np.float32))
        opt.step()
    assert len(backend.cache) == 1, f"{len(backend.cache)} programs compiled for 3 steps"


def test_an_optimizer_step_trains(on_backend):
    set_seed(3)  # whether five momentum steps end below the start depends on the init, so own it
    layer = Linear(4, 3)
    x = Tensor(np.random.default_rng(3).random((8, 4)).astype(np.float32))
    opt = SGD(parameters(layer), lr=0.05, momentum=0.9)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = (layer(x) ** 2).sum()
        loss.backward()
        losses.append(loss.item())
        opt.step()
    assert losses[-1] < losses[0]
