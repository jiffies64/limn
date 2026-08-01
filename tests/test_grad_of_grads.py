"""Second derivatives against torch: create_graph keeps a gradient differentiable, grad() reads it.

Everything here runs on the numpy device; a second-order graph is ordinary primitives, so the
backends see nothing new. Inputs follow the fuzzer's safe-range idiom (log and friends get
x*x + 0.5) so both frameworks stay away from singularities.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import randf

from limn import Tensor, grad
from limn.tensor import scatter_rows

SECOND = [
    ("cube", lambda x: (x * x * x).sum(), lambda x: (x * x * x).sum()),
    ("exp", lambda x: (x * 0.5).exp().sum(), lambda x: (x * 0.5).exp().sum()),
    ("log", lambda x: (x * x + 0.5).log().sum(), lambda x: (x * x + 0.5).log().sum()),
    ("sqrt", lambda x: (x * x + 0.5).sqrt().sum(), lambda x: (x * x + 0.5).sqrt().sum()),
    ("recip", lambda x: (x * x + 0.5).reciprocal().sum(), lambda x: (1 / (x * x + 0.5)).sum()),
    ("relu", lambda x: (x.relu() * x * x).sum(), lambda x: (x.relu() * x * x).sum()),
    ("max", lambda x: (x.max(1) ** 2).sum(), lambda x: (x.amax(1) ** 2).sum()),
    ("softmax", lambda x: (x.softmax(-1) ** 2).sum(), lambda x: (x.softmax(-1) ** 2).sum()),
    ("matmul", lambda x: ((x.transpose() @ x) ** 2).sum(), lambda x: ((x.transpose(-2, -1) @ x) ** 2).sum()),
    ("pad", lambda x: (x.pad(((1, 0), (0, 1))) ** 3).sum(), lambda x: (F.pad(x, (0, 1, 1, 0)) ** 3).sum()),
]


@pytest.mark.parametrize("limn_f,torch_f", [pytest.param(lf, tf, id=name) for name, lf, tf in SECOND])
def test_second_derivative_matches_torch(limn_f, torch_f):
    data = randf(3, 4)
    x = Tensor(data, requires_grad=True)
    (g,) = grad(limn_f(x), [x], create_graph=True)
    (h,) = grad(g.sum(), [x])
    t = torch.tensor(data, requires_grad=True)
    (tg,) = torch.autograd.grad(torch_f(t), t, create_graph=True)
    (th,) = torch.autograd.grad(tg.sum(), t)
    np.testing.assert_allclose(h.numpy(), th.numpy(), rtol=1e-4, atol=1e-4)


def test_third_derivative_of_x_to_the_fourth():
    data = randf(5)
    x = Tensor(data, requires_grad=True)
    (g,) = grad((x**4).sum(), [x], create_graph=True)
    (h,) = grad(g.sum(), [x], create_graph=True)
    (third,) = grad(h.sum(), [x])
    np.testing.assert_allclose(third.numpy(), 24 * data, rtol=1e-4, atol=1e-4)


def test_mlp_hessian_vector_product_matches_torch():
    w1d, w2d, xd, v1d, v2d = randf(4, 8), randf(8, 2), randf(5, 4), randf(4, 8), randf(8, 2)
    w1, w2 = Tensor(w1d, requires_grad=True), Tensor(w2d, requires_grad=True)
    loss = (((Tensor(xd) @ w1).relu() @ w2) ** 2).sum()
    g1, g2 = grad(loss, [w1, w2], create_graph=True)
    h1, h2 = grad((g1 * Tensor(v1d)).sum() + (g2 * Tensor(v2d)).sum(), [w1, w2])
    tw1, tw2 = torch.tensor(w1d, requires_grad=True), torch.tensor(w2d, requires_grad=True)
    tloss = (((torch.tensor(xd) @ tw1).relu() @ tw2) ** 2).sum()
    tg1, tg2 = torch.autograd.grad(tloss, (tw1, tw2), create_graph=True)
    th1, th2 = torch.autograd.grad((tg1 * torch.tensor(v1d)).sum() + (tg2 * torch.tensor(v2d)).sum(), (tw1, tw2))
    np.testing.assert_allclose(h1.numpy(), th1.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(h2.numpy(), th2.numpy(), rtol=1e-4, atol=1e-4)


def test_gather_second_order_matches_torch():
    td, idx = randf(4, 3), np.array([0, 2, 2, 1])
    table = Tensor(td, requires_grad=True)
    (g,) = grad((table[Tensor(idx)] ** 3).sum(), [table], create_graph=True)
    (h,) = grad((g * g).sum(), [table])
    tt = torch.tensor(td, requires_grad=True)
    (tg,) = torch.autograd.grad((tt[torch.tensor(idx)] ** 3).sum(), tt, create_graph=True)
    (th,) = torch.autograd.grad((tg * tg).sum(), tt)
    np.testing.assert_allclose(h.numpy(), th.numpy(), rtol=1e-4, atol=1e-4)


def test_scatter_rows_gradient_gathers():
    vals, weights = randf(4, 3), randf(3, 3)
    values = Tensor(vals, requires_grad=True)
    table = scatter_rows(values, Tensor(np.array([0, 2, 2, 1])), (3, 3))
    (g,) = grad((table * Tensor(weights)).sum(), [values])
    np.testing.assert_allclose(g.numpy(), weights[[0, 2, 2, 1]], rtol=1e-4, atol=1e-4)


def test_backward_create_graph_makes_grad_differentiable():
    data = randf(3)
    w = Tensor(data, requires_grad=True)
    (w * w * w).sum().backward(create_graph=True)
    assert w.grad is not None and w.grad.requires_grad
    (h,) = grad(w.grad.sum(), [w])
    np.testing.assert_allclose(h.numpy(), 6 * data, rtol=1e-4, atol=1e-4)


def test_plain_backward_leaves_grad_unrecorded():
    w = Tensor(randf(3), requires_grad=True)
    (w * w).sum().backward()
    assert w.grad is not None and not w.grad.requires_grad and w.grad.parents == ()


def test_grad_reads_without_writing():
    data = randf(3)
    w = Tensor(data, requires_grad=True)
    (g,) = grad((w * w).sum(), [w])
    assert w.grad is None
    np.testing.assert_allclose(g.numpy(), 2 * data, rtol=1e-4, atol=1e-4)


def test_grad_rejects_what_it_cannot_differentiate():
    x = Tensor(randf(2, 2), requires_grad=True)
    with pytest.raises(ValueError, match="scalar"):
        grad(x * x, [x])
    with pytest.raises(ValueError, match="requires_grad"):
        grad(Tensor(randf(3)).sum(), [x])
    with pytest.raises(ValueError, match=r"inputs\[1\]"):
        grad((x * x).sum(), [x, Tensor(randf(3), requires_grad=True)])
