"""C backend tests: every graph the numpy device can run, the C backend must match."""

import re
from unittest import mock

import numpy as np
import pytest
from conftest import GRAPHS, cdev, check, randf

import limn.backend_c as backend_c
from limn import Tensor, set_device, set_seed
from limn.backend_c import CDevice, cache, emit_c, has_cc
from limn.codegen import lower_all
from limn.device import NUMPY_DTYPES
from limn.nn import Conv2d
from limn.ops import Op

pytestmark = pytest.mark.skipif(not has_cc(), reason="no C compiler found")


def check_c(t: Tensor, dev: CDevice | None = None) -> None:
    check(dev or cdev, t)


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
    bufs = CDevice().execute([p.node])
    got = bufs[0].view(NUMPY_DTYPES[p.dtype]).reshape(p.shape)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_assign_deferral_reads_pre_assign_bytes():
    """A graph built before the assign, realized in the same batch, sees old values."""
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    old_value = p * 10.0
    p.assign(p + 100.0)
    bufs = CDevice().execute([old_value.node, p.node])
    got_old = bufs[0].view(NUMPY_DTYPES[old_value.dtype]).reshape(old_value.shape)
    got_new = bufs[1].view(NUMPY_DTYPES[p.dtype]).reshape(p.shape)
    np.testing.assert_allclose(got_old, np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
    np.testing.assert_allclose(got_new, np.array([[101.0, 102.0], [103.0, 104.0]], dtype=np.float32))


def test_assign_consumed_as_value():
    """An ASSIGN node read by a later kernel yields the new value, not the old buffer."""
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    p.assign(p * 2.0)
    consumer = p + 1.0
    bufs = CDevice().execute([consumer.node])
    got = bufs[0].view(NUMPY_DTYPES[consumer.dtype]).reshape(consumer.shape)
    np.testing.assert_allclose(got, np.array([[3.0, 5.0], [7.0, 9.0]], dtype=np.float32))


def test_sgd_momentum_step():
    """SGD with momentum builds assign-then-read-through graphs (optim.py's v.assign, g = v)."""
    from limn.optim import SGD

    dev = CDevice()
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    p.grad = Tensor(np.ones((2, 2), dtype=np.float32))
    opt = SGD([p], lr=0.1, momentum=0.9)

    expected_p = p.numpy() - 0.1 * np.ones((2, 2), dtype=np.float32)
    opt.step()

    bufs = dev.execute([p.node])
    got = bufs[0].view(NUMPY_DTYPES[p.dtype]).reshape(p.shape)
    np.testing.assert_allclose(got, expected_p, atol=1e-6)

    p.grad = Tensor(np.ones((2, 2), dtype=np.float32))
    expected_p2 = p.numpy() - 0.1 * (0.9 * np.ones((2, 2)) + np.ones((2, 2)))
    opt.step()
    bufs = dev.execute([p.node])
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
    check_c(layer(x))
    (layer(x) * layer(x)).sum().backward()
    assert x.grad is not None and layer.weight.grad is not None
    check_c(x.grad)
    check_c(layer.weight.grad)


@pytest.mark.parametrize("name", list(CONVS))
def test_splitting_the_innermost_loop_changes_no_result(name):
    """The split is a speedup, so it owes bit-identical output, not merely output within a tolerance."""
    layer = CONVS[name]()
    x = Tensor(randf(2, layer.in_channels, 9, 8))
    out = layer(x)
    plain = CDevice()
    with mock.patch.object(backend_c, "split_masked", tuple):
        expected = plain.execute([out.node])[0]
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
    active = device.active()
    assert isinstance(active, CDevice)
    p = Tensor(np.ones(4, dtype=np.float32))
    for _ in range(3):
        p.assign(p * 2.0)
        p.realize()
    np.testing.assert_allclose(p.numpy(), np.full(4, 8.0, dtype=np.float32))
    assert len(active.plans) == 2  # one plan for the assign step, one for numpy()'s read


def test_adamw_compiles_one_program_for_all_steps():
    """AdamW's bias correction changes every step, and a literal would rehash the source with it."""
    from limn.optim import AdamW

    set_device("c")
    p = Tensor(np.ones((4, 4), dtype=np.float32), requires_grad=True)
    opt = AdamW([p], lr=0.1)
    cache.clear()
    for _ in range(3):
        p.grad = Tensor(np.ones((4, 4), dtype=np.float32))
        opt.step()
    assert len(cache) == 1, f"{len(cache)} programs compiled for 3 steps"
