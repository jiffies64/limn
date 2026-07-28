"""SGD, AdamW and Muon trajectories over several steps must match torch.optim exactly.

Muon is compared under adjust_lr_fn="match_rms_adamw": limn implements that scaling, and its
accumulator momentum differs from torch's EMA by a factor Newton-Schulz normalizes away.
torch orthogonalizes in bfloat16 where limn stays in float32, so the torch comparison runs at
a tolerance sized to bfloat16 noise and catches algorithm-level drift; the float32 numpy
replica below it pins the exact arithmetic.
"""

import math

import numpy as np
import pytest
import torch

from limn import Tensor, no_grad
from limn.optim import SGD, AdamW, Muon

rng = np.random.default_rng(11)


def make_params(*shapes: tuple[int, ...]) -> tuple[list[Tensor], list[torch.Tensor]]:
    datas = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    return ([Tensor(d.copy(), requires_grad=True) for d in datas], [torch.tensor(d.copy(), requires_grad=True) for d in datas])


def loss_pair(lparams: list[Tensor], tparams: list[torch.Tensor], x: np.ndarray) -> tuple[Tensor, torch.Tensor]:
    lx, tx = Tensor(x), torch.tensor(x)
    lloss = ((lx @ lparams[0]) * lparams[1]).relu().sum()
    tloss = ((tx @ tparams[0]) * tparams[1]).relu().sum()
    return lloss, tloss


def run_steps(lopt, topt, lparams, tparams, steps: int = 8) -> None:
    for step in range(steps):
        x = rng.standard_normal((6, 4)).astype(np.float32)
        lopt.zero_grad()
        topt.zero_grad()
        lloss, tloss = loss_pair(lparams, tparams, x)
        lloss.backward()
        tloss.backward()
        lopt.step()
        topt.step()
        for i, (lp, tp) in enumerate(zip(lparams, tparams)):
            np.testing.assert_allclose(
                lp.numpy(), tp.detach().numpy(), atol=1e-5, rtol=1e-5, err_msg=f"param {i} diverged at step {step}"
            )


def test_sgd_plain():
    lparams, tparams = make_params((4, 3), (3,))
    run_steps(SGD(lparams, lr=0.05), torch.optim.SGD(tparams, lr=0.05), lparams, tparams)


def test_sgd_momentum():
    lparams, tparams = make_params((4, 3), (3,))
    run_steps(SGD(lparams, lr=0.05, momentum=0.9), torch.optim.SGD(tparams, lr=0.05, momentum=0.9), lparams, tparams)


def test_adamw_defaults():
    lparams, tparams = make_params((4, 3), (3,))
    run_steps(AdamW(lparams), torch.optim.AdamW(tparams), lparams, tparams)


def test_adamw_custom_hyperparameters():
    lparams, tparams = make_params((4, 3), (3,))
    lopt = AdamW(lparams, lr=3e-3, betas=(0.85, 0.99), eps=1e-6, weight_decay=0.1)
    topt = torch.optim.AdamW(tparams, lr=3e-3, betas=(0.85, 0.99), eps=1e-6, weight_decay=0.1)
    run_steps(lopt, topt, lparams, tparams)


def test_unused_param_is_skipped():
    lparams, tparams = make_params((4, 3), (3,), (5, 5))
    lopt, topt = AdamW(lparams, lr=1e-2), torch.optim.AdamW(tparams, lr=1e-2)
    x = rng.standard_normal((6, 4)).astype(np.float32)
    lloss, tloss = loss_pair(lparams, tparams, x)  # never touches params[2]
    lloss.backward()
    tloss.backward()
    assert lparams[2].grad is None and tparams[2].grad is None
    lopt.step()
    topt.step()
    for lp, tp in zip(lparams, tparams):
        np.testing.assert_allclose(lp.numpy(), tp.detach().numpy(), atol=1e-6, rtol=1e-6)


def test_optimizer_requires_grad_params():
    with pytest.raises(ValueError):
        SGD([Tensor(np.ones((2, 2), dtype=np.float32))], lr=0.1)


def test_adamw_updates_builds_without_committing():
    """updates() is inspection: nothing lands on the device until step(), which still matches torch."""
    lparams, tparams = make_params((4, 3), (3,))
    lopt, topt = AdamW(lparams, lr=1e-2), torch.optim.AdamW(tparams, lr=1e-2)
    x = rng.standard_normal((6, 4)).astype(np.float32)
    lloss, tloss = loss_pair(lparams, tparams, x)
    lloss.backward()
    tloss.backward()
    with no_grad():
        lopt.updates()  # built and discarded; beta**t and the moments must not move
    lopt.step()
    topt.step()
    for lp, tp in zip(lparams, tparams):
        np.testing.assert_allclose(lp.numpy(), tp.detach().numpy(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_matches_torch(nesterov):
    lparams, tparams = make_params((4, 6), (6, 3))  # one wide, one tall, so both orientations transpose
    lopt = Muon(lparams, lr=0.02, momentum=0.9, weight_decay=0.1, nesterov=nesterov)
    topt = torch.optim.Muon(tparams, lr=0.02, momentum=0.9, weight_decay=0.1, nesterov=nesterov, adjust_lr_fn="match_rms_adamw")
    for step in range(6):
        x = rng.standard_normal((5, 4)).astype(np.float32)
        lopt.zero_grad()
        topt.zero_grad()
        lloss = ((Tensor(x) @ lparams[0]).relu() @ lparams[1]).sum()
        tloss = ((torch.tensor(x) @ tparams[0]).relu() @ tparams[1]).sum()
        lloss.backward()
        tloss.backward()
        lopt.step()
        topt.step()
        for i, (lp, tp) in enumerate(zip(lparams, tparams)):
            # torch's bfloat16 Newton-Schulz perturbs the direction by ~1e-4; an algorithmic
            # divergence (scaling, decay, momentum) would be two orders larger
            np.testing.assert_allclose(
                lp.numpy(), tp.detach().numpy(), atol=2e-3, rtol=2e-3, err_msg=f"param {i} diverged at step {step}"
            )


def muon_reference_step(param, grad, buf, lr, momentum, wd, nesterov, ns_steps=5, eps=1e-7):
    buf = momentum * buf + grad
    direction = grad + momentum * buf if nesterov else buf
    x = direction.T if direction.shape[0] > direction.shape[1] else direction
    x = x / (np.sqrt((x * x).sum()) + eps)
    a, b, c = Muon.NS_COEFFS
    for _ in range(ns_steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    ortho = x.T if direction.shape[0] > direction.shape[1] else x
    return param * (1 - lr * wd) - lr * 0.2 * math.sqrt(max(param.shape)) * ortho, buf


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_trajectory_matches_the_float32_replica(nesterov):
    datas = [rng.standard_normal(s).astype(np.float32) for s in ((4, 6), (6, 3))]
    params = [Tensor(d.copy(), requires_grad=True) for d in datas]
    opt = Muon(params, lr=0.02, momentum=0.9, weight_decay=0.1, nesterov=nesterov)
    expected = [d.copy() for d in datas]
    bufs = [np.zeros_like(d) for d in datas]
    for step in range(5):
        x = rng.standard_normal((5, 4)).astype(np.float32)
        opt.zero_grad()
        loss = ((Tensor(x) @ params[0]).relu() @ params[1]).sum()
        loss.backward()
        grads = [p.grad.numpy() for p in params if p.grad is not None]
        assert len(grads) == len(params)
        opt.step()
        for i, (p, g) in enumerate(zip(params, grads)):
            expected[i], bufs[i] = muon_reference_step(expected[i], g, bufs[i], 0.02, 0.9, 0.1, nesterov)
            np.testing.assert_allclose(p.numpy(), expected[i], atol=1e-5, rtol=1e-5, err_msg=f"param {i} diverged at step {step}")


@pytest.mark.parametrize("shape", [(8, 4), (4, 8)])
def test_newton_schulz_lands_singular_values_near_one(shape):
    opt = Muon([Tensor(np.zeros((2, 2), dtype=np.float32), requires_grad=True)])
    ortho = opt.newton_schulz(Tensor(rng.standard_normal(shape).astype(np.float32))).numpy()
    singular = np.linalg.svd(ortho, compute_uv=False)
    assert 0.6 < singular.min() and singular.max() < 1.2  # the 5-step quintic's published band is (0.68, 1.13)


def test_muon_rejects_non_2d_parameters():
    with pytest.raises(ValueError, match="2D"):
        Muon([Tensor(np.ones(3, dtype=np.float32), requires_grad=True)])


def test_step_realizes_extras_in_the_same_batch():
    """A loss passed to step() reads the pre-step parameters, like everything in the batch."""
    lparams, _ = make_params((4, 3), (3,))
    before = [p.numpy().copy() for p in lparams]
    opt = SGD(lparams, lr=0.5)
    loss = (lparams[0] * lparams[1].reshape(1, 3)).sum()
    loss.backward()
    opt.step(loss)
    expected = (before[0] * before[1].reshape(1, 3)).sum()
    np.testing.assert_allclose(loss.item(), expected, rtol=1e-6)
    assert not np.allclose(lparams[0].numpy(), before[0])  # the updates themselves still committed
