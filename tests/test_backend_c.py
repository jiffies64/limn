"""C backend tests: every graph the numpy device can run, the C backend must match."""

import os

import numpy as np
import pytest
from conftest import GRAPHS, randf

import limn.backend_c as backend_c
from limn import Tensor, set_device
from limn.backend_c import PARALLEL_MIN, CDevice, cache, emit_c, has_cc, openmp, team_size
from limn.codegen import lower_all
from limn.device import NUMPY_DTYPES

pytestmark = pytest.mark.skipif(not has_cc(), reason="no C compiler found")

needs_openmp = pytest.mark.skipif(not (has_cc() and openmp()), reason="cc has no working OpenMP runtime")

cdev = CDevice()


def check_c(t: Tensor, dev: CDevice = cdev) -> None:
    expected = t.numpy()
    bufs = dev.execute([t.node])
    got = bufs[0].view(NUMPY_DTYPES[t.dtype]).reshape(t.shape)
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name", list(GRAPHS))
def test_c_matches_numpy_device(name):
    a, b = Tensor(randf(3, 4)), Tensor(randf(3, 4))
    check_c(GRAPHS[name](a, b))


def test_matmul_4x5_5x3():
    a, b = Tensor(randf(4, 5)), Tensor(randf(5, 3))
    check_c(a @ b)


def test_backward_pass():
    x = Tensor(randf(4, 5), requires_grad=True)
    w = Tensor(randf(5, 3), requires_grad=True)
    loss = (x @ w).relu().sum()
    loss.backward()
    assert x.grad is not None and w.grad is not None
    check_c(loss)
    check_c(w.grad)
    check_c(x.grad)


def test_assign_deferred():
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    expected = p.numpy() * 2.0 + 1.0
    p.assign(p * 2.0 + 1.0)
    bufs = cdev.execute([p.node])
    got = bufs[0].view(NUMPY_DTYPES[p.dtype]).reshape(p.shape)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_assign_deferral_reads_pre_assign_bytes():
    """A graph built before the assign, realized in the same batch, sees old values."""
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    old_value = p * 10.0
    p.assign(p + 100.0)
    bufs = cdev.execute([old_value.node, p.node])
    got_old = bufs[0].view(NUMPY_DTYPES[old_value.dtype]).reshape(old_value.shape)
    got_new = bufs[1].view(NUMPY_DTYPES[p.dtype]).reshape(p.shape)
    np.testing.assert_allclose(got_old, np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
    np.testing.assert_allclose(got_new, np.array([[101.0, 102.0], [103.0, 104.0]], dtype=np.float32))


def test_assign_consumed_as_value():
    """An ASSIGN node read by a later kernel yields the new value, not the old buffer."""
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    p.assign(p * 2.0)
    consumer = p + 1.0
    bufs = cdev.execute([consumer.node])
    got = bufs[0].view(NUMPY_DTYPES[consumer.dtype]).reshape(consumer.shape)
    np.testing.assert_allclose(got, np.array([[3.0, 5.0], [7.0, 9.0]], dtype=np.float32))


def test_sgd_momentum_step():
    """SGD with momentum builds assign-then-read-through graphs (optim.py's v.assign, g = v)."""
    from limn.optim import SGD

    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    p.grad = Tensor(np.ones((2, 2), dtype=np.float32))
    opt = SGD([p], lr=0.1, momentum=0.9)

    expected_p = p.numpy() - 0.1 * np.ones((2, 2), dtype=np.float32)
    opt.step()

    bufs = cdev.execute([p.node])
    got = bufs[0].view(NUMPY_DTYPES[p.dtype]).reshape(p.shape)
    np.testing.assert_allclose(got, expected_p, atol=1e-6)

    p.grad = Tensor(np.ones((2, 2), dtype=np.float32))
    expected_p2 = p.numpy() - 0.1 * (0.9 * np.ones((2, 2)) + np.ones((2, 2)))
    opt.step()
    bufs = cdev.execute([p.node])
    got2 = bufs[0].view(NUMPY_DTYPES[p.dtype]).reshape(p.shape)
    np.testing.assert_allclose(got2, expected_p2, atol=1e-6)


def test_multi_kernel_chain():
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    check_c((a + b).reshape(6) * 2.0)


def test_contiguous_copy():
    a = Tensor(randf(3, 4))
    check_c(a.transpose().reshape(12))


def test_shared_subgraph():
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    shared = (a + b).sum(axis=1, keepdim=True)
    check_c(shared * 2.0)
    check_c(shared * 3.0)


def test_gather_rows_forward_and_backward():
    table = Tensor(randf(6, 4), requires_grad=True)
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))  # repeats, so the scatter accumulates
    gathered = table.gather_rows(indices)
    check_c(gathered)
    (gathered * Tensor(randf(2, 3, 4))).sum().backward()
    assert table.grad is not None
    check_c(table.grad)


def test_a_repeated_graph_reuses_the_plan_and_reads_fresh_bytes():
    dev = CDevice()
    for x in (randf(3, 4), randf(3, 4)):
        check_c((Tensor(x) * 2.0).sum(axis=1), dev)
    assert len(dev.plans) == 1


def test_a_changed_constant_is_a_different_plan():
    dev = CDevice()
    x = randf(2, 3)
    for scale in (2.0, 3.0):
        check_c(Tensor(x) * scale, dev)
    assert len(dev.plans) == 2


def test_a_repeated_assign_commits_through_the_cached_plan():
    from limn import device

    set_device("c")
    try:
        active = device.active()
        assert isinstance(active, CDevice)
        p = Tensor(np.ones(4, dtype=np.float32))
        for _ in range(3):
            p.assign(p * 2.0)
            p.realize()
        np.testing.assert_allclose(p.numpy(), np.full(4, 8.0, dtype=np.float32))
        assert len(active.plans) == 2  # one plan for the assign step, one for numpy()'s read
    finally:
        set_device("numpy")


# ---- threading: which loops get a team, and that having one changes no numbers ----


def emitted(t: Tensor) -> str:
    return emit_c(lower_all([t.node]))


def threaded_loops(source: str) -> list[str]:
    """The loop variable each `omp parallel for` in this source governs."""
    lines = source.splitlines()
    return [lines[k + 1].split()[2] for k, line in enumerate(lines) if "omp parallel for" in line]


@needs_openmp
def test_a_big_nest_is_threaded_over_its_output_dims():
    n = 256  # 256*256 points, comfortably over PARALLEL_MIN
    a, b = Tensor(randf(n, n)), Tensor(randf(n, n))
    assert threaded_loops(emitted((a + b) * 2.0)) == ["i0"]
    # a matmul runs two nests: the identity fill, then the fold. Both thread over the output rows.
    assert threaded_loops(emitted(a @ b)) == ["i0", "i0"]


@needs_openmp
def test_a_short_outer_dim_collapses_until_there_is_work_for_every_thread():
    rows = Tensor(randf(2, PARALLEL_MIN))
    source = emitted(rows * 2.0)
    assert threaded_loops(source) == ["i0"]
    assert f"collapse(2) num_threads({team_size()})" in source, source


@needs_openmp
def test_a_small_nest_runs_serial():
    a, b = Tensor(randf(3, 4)), Tensor(randf(3, 4))
    assert threaded_loops(emitted((a + b) * 2.0)) == []


@needs_openmp
@pytest.mark.parametrize("name", list(GRAPHS))
def test_no_reduce_axis_is_ever_threaded(name):
    """Threading a loop that carries a running total would regroup the folds; only i-loops qualify."""
    a, b = Tensor(randf(256, 256)), Tensor(randf(256, 256))
    assert all(var.startswith("i") for var in threaded_loops(emitted(GRAPHS[name](a, b))))


@needs_openmp
def test_a_scatter_nest_runs_serial():
    """Two rows of the values can land on the same row of the table, so its adds stay ordered."""
    table = Tensor(randf(64, 256), requires_grad=True)
    indices = Tensor(np.zeros((64, 64), dtype=np.int32))
    (table.gather_rows(indices) * Tensor(randf(64, 64, 256))).sum().backward()
    assert table.grad is not None
    source = emitted(table.grad)
    assert "+=" in source and threaded_loops(source) == []


THREADED_GRAPHS = {
    "elementwise": lambda a, b: (a + b) * 2.0 - a,
    "register reduce": lambda a, b: (a * b).sum(axis=1),  # reduce axis innermost: folds in a register
    "output-folded reduce": lambda a, b: a @ b,  # reduce axis not innermost: folds into the output
    "dot reduce": lambda a, b: a @ b.transpose(),
    "softmax": lambda a, b: (a + b).softmax(axis=1),
}


@needs_openmp
@pytest.mark.parametrize("name", list(THREADED_GRAPHS))
def test_threading_changes_no_bits(name, monkeypatch):
    """Every output cell is one thread's from start to finish, so the answer is the serial one exactly.

    Both builds get the same flags; only the pragmas differ, so any difference is threading's.
    """
    a, b = Tensor(randf(256, 256)), Tensor(randf(256, 256))
    graph = THREADED_GRAPHS[name]
    assert threaded_loops(emitted(graph(a, b))), "nothing was threaded, so this proves nothing"

    threaded = CDevice().execute([graph(a, b).node])[0]
    monkeypatch.setattr(backend_c, "openmp", lambda: False)
    serial = CDevice().execute([graph(a, b).node])[0]
    np.testing.assert_array_equal(threaded, serial)


@pytest.mark.skipif("OMP_NUM_THREADS" in os.environ, reason="the environment chose the team size")
def test_the_default_team_never_outgrows_the_processors_this_process_may_use():
    assert 1 <= team_size() <= (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())


def test_adamw_compiles_one_program_for_all_steps():
    """AdamW's bias correction changes every step, and a literal would rehash the source with it."""
    from limn.optim import AdamW

    set_device("c")
    try:
        p = Tensor(np.ones((4, 4), dtype=np.float32), requires_grad=True)
        opt = AdamW([p], lr=0.1)
        cache.clear()
        for _ in range(3):
            p.grad = Tensor(np.ones((4, 4), dtype=np.float32))
            opt.step()
        assert len(cache) == 1, f"{len(cache)} programs compiled for 3 steps"
    finally:
        set_device("numpy")
