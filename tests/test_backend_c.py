"""What is particular to the C backend: convolutions through cc, and the loop splitting it wants.

Everything a compiled backend owes in general is in test_compiled_devices.py.
"""

import re
from unittest import mock

import numpy as np
import pytest
from conftest import cdev, check, randf, read

import limn.backend_c as backend_c
from limn import Tensor, set_seed
from limn.backend_c import CDevice, emit_c, has_cc
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
