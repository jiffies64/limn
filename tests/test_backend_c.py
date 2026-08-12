"""What is particular to the C backend: convolutions through cc, the loop splitting it wants, and
the thread team its loops are handed to.

Everything a compiled backend owes in general is in test_compiled_devices.py.
"""

import math
import os
import re
from unittest import mock

import numpy as np
import pytest
from conftest import GRAPHS, cdev, check, randf, read

import limn.backend_c as backend_c
from limn import Tensor, set_seed
from limn.backend_c import PARALLEL_MIN, CDevice, emit_c, has_cc, openmp, team_size
from limn.codegen import lower_all
from limn.nn import Conv2d
from limn.ops import Op

pytestmark = pytest.mark.skipif(not has_cc(), reason="no C compiler found")


def test_sgd_momentum_step():
    """SGD with momentum builds assign-then-read-through graphs (optim.py's v.assign, g = v)."""
    from limn.optim import SGD

    dev = CDevice()
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9)
    for velocity in (1.0, 1.9):  # the second step carries the first step's gradient through momentum
        p.grad = Tensor(np.ones((2, 2), dtype=np.float32))
        expected = p.numpy() - 0.1 * velocity
        opt.step()
        np.testing.assert_allclose(read(dev, dev.execute([p.node])[0], p), expected, atol=1e-6)


CONVS = {  # padding is what puts a mask on the innermost loop, which is what the split is for
    "3x3 padded": lambda: Conv2d(3, 4, 3, padding=1),
    "3x3 unpadded": lambda: Conv2d(3, 4, 3),
    "5x5 padded, both ends of the inner loop cut": lambda: Conv2d(2, 3, 5, padding=2),
    "strided and padded": lambda: Conv2d(4, 4, 3, stride=2, padding=1),
    "dilated, grouped, 'same'": lambda: Conv2d(4, 4, 3, padding="same", dilation=2, groups=4),
}


@pytest.mark.parametrize("name", list(CONVS))
def test_conv_matches_the_numpy_device(name):
    set_seed(0)
    layer = CONVS[name]()
    x = Tensor(randf(2, layer.in_channels, 9, 8), requires_grad=True)
    check(cdev, layer(x))
    (layer(x) * layer(x)).sum().backward()
    assert x.grad is not None and layer.weight.grad is not None
    check(cdev, x.grad)
    check(cdev, layer.weight.grad)


@pytest.mark.parametrize("name", list(CONVS))
def test_splitting_the_innermost_loop_changes_no_result(name):
    """The split is a speedup, so it owes bit-identical output, not merely output within a tolerance."""
    layer = CONVS[name]()
    x = Tensor(randf(2, layer.in_channels, 9, 8))
    out = layer(x)
    with mock.patch.object(backend_c, "split_masked", tuple):
        expected = CDevice().execute([out.node])[0]
    np.testing.assert_array_equal(CDevice().execute([out.node])[0], expected)


FOR = re.compile(r"for \(int (\w+) = \d+; \1 < \d+; \1\+\+\) \{$")


def innermost_bodies(source: str) -> list[tuple[str, list[str]]]:
    """Each innermost for loop in the emitted C, as (loop variable, the lines inside it)."""
    lines = source.splitlines()
    bodies = []
    for i, line in enumerate(lines):
        if not (opened := FOR.search(line.strip())):
            continue
        depth, body = 0, []
        for inner in lines[i + 1 :]:
            depth += inner.count("{") - inner.count("}")
            if depth < 0:
                break
            body.append(inner)
        if not any(FOR.search(inner.strip()) for inner in body):
            bodies.append((opened.group(1), body))
    return bodies


def test_a_padded_conv_emits_no_guard_on_its_innermost_loop():
    """What the split buys: cc vectorises the innermost loop, and a guard on its variable stops it."""
    x = Tensor(randf(2, 3, 9, 8))
    taps = [n for n in lower_all([Conv2d(3, 4, 3, padding=1)(x).node]) if n.kernel.ast.op is Op.SUM]
    assert len(taps) == 9, "a 3x3 conv is nine taps, six of them masked on the innermost dim"

    with mock.patch.object(backend_c, "split_masked", tuple):
        before = innermost_bodies(emit_c(taps))
    guarded = [var for var, body in before if any(re.search(rf"\b{var} [<>]", line) for line in body)]
    assert guarded, "the unsplit conv should be the thing this test is about"

    for var, body in innermost_bodies(emit_c(taps)):
        for line in body:
            assert not re.search(rf"\b{var} [<>]", line), f"{var} is still tested inside its own loop: {line.strip()}"


# ---- threading: which loops get a team, and that having one changes no numbers ----

needs_openmp = pytest.mark.skipif(not (has_cc() and openmp()), reason="cc has no working OpenMP runtime")


def emitted(t: Tensor) -> str:
    return emit_c(lower_all([t.node]))


def threaded_loops(source: str) -> list[str]:
    """The loop variable each `omp parallel for` in this source governs."""
    lines = source.splitlines()
    opened = [FOR.search(lines[k + 1].strip()) for k, line in enumerate(lines) if "omp parallel for" in line]
    return [loop.group(1) for loop in opened if loop is not None]


@needs_openmp
def test_a_big_nest_is_threaded_over_its_output_dims():
    a, b = Tensor(randf(256, 256)), Tensor(randf(256, 256))
    assert threaded_loops(emitted((a + b) * 2.0)) == ["i0"]
    # a matmul is two chains: the fill with the reduce identity, then the fold. Both thread over rows.
    assert threaded_loops(emitted(a @ b)) == ["i0", "i0"]


@needs_openmp
def test_a_short_outer_dim_collapses_until_there_is_work_for_every_thread():
    source = emitted(Tensor(randf(2, PARALLEL_MIN)) * 2.0)
    assert threaded_loops(source) == ["i0"]
    assert f"collapse(2) num_threads({team_size()})" in source, source


@needs_openmp
def test_a_team_collapses_no_further_than_the_loops_are_perfectly_nested():
    """split_masked leaves the innermost loop as siblings, and a collapse clause may not span those."""
    padded = Tensor(randf(2, 4, 4096)).pad(((0, 0), (0, 0), (1, 1))) * 2.0
    source = emitted(padded)
    assert "collapse(2)" in source and "collapse(3)" not in source, source
    check(cdev, padded)  # collapsing past the split would not compile at all


@needs_openmp
def test_a_small_nest_runs_serial():
    a, b = Tensor(randf(3, 4)), Tensor(randf(3, 4))
    assert threaded_loops(emitted((a + b) * 2.0)) == []


@needs_openmp
@pytest.mark.parametrize("name", list(GRAPHS))
def test_no_reduce_axis_is_ever_threaded(name):
    """Splitting a loop that carries a running total would regroup the folds; only i-loops qualify."""
    a, b = Tensor(randf(256, 256)), Tensor(randf(256, 256))
    assert all(var.startswith("i") for var in threaded_loops(emitted(GRAPHS[name](a, b))))


@needs_openmp
def test_a_scatter_nest_runs_serial():
    """Two rows of the values can land on one row of the table, so its adds stay in the serial order."""
    table = Tensor(randf(64, 256), requires_grad=True)
    (table.gather_rows(Tensor(np.zeros((64, 64), dtype=np.int32))) * Tensor(randf(64, 64, 256))).sum().backward()
    assert table.grad is not None
    nests = [nest for nest in lower_all([table.grad.node]) if nest.kernel.ast.op is Op.SCATTER]
    assert len(nests) == 1 and math.prod(nests[0].space) >= PARALLEL_MIN, "else it stays serial for its size"
    source = emit_c(nests)
    assert "+=" in source and threaded_loops(source) == []


@needs_openmp
@pytest.mark.parametrize("name", list(GRAPHS))
def test_threading_changes_no_bits(name):
    """Every output cell is one thread's from start to finish, so the answer is the serial one exactly.

    Held to the bits rather than to a tolerance, which is what makes this a test of threading and
    not of the FMA contraction -march=native brings: both builds get the same flags and differ only
    in the pragmas. A team folding into cells it does not own alone would race rather than merely
    round differently, so the miss would be orders of magnitude wider than either.

    test_compiled_devices runs this corpus against the numpy device at (3, 4), which is under
    PARALLEL_MIN and so threads nothing; this is the size at which it threads at all.
    """
    a, b = Tensor(randf(256, 128)), Tensor(randf(256, 128))
    threaded = CDevice().execute([GRAPHS[name](a, b).node])[0]
    with mock.patch.object(backend_c, "openmp", lambda: False):
        serial = CDevice().execute([GRAPHS[name](a, b).node])[0]
    np.testing.assert_array_equal(threaded, serial)


@pytest.mark.skipif("OMP_NUM_THREADS" in os.environ, reason="the environment chose the team size")
def test_the_default_team_never_outgrows_the_processors_this_process_may_use():
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    assert 1 <= team_size() <= usable
