"""SGD and AdamW trajectories over several steps must match torch.optim exactly."""

import numpy as np
import pytest
import torch

from limn import Tensor, no_grad
from limn.optim import SGD, AdamW

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
