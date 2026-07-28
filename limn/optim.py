"""Optimizers. Updates are ASSIGN graphs, committed in one realize() batch per step.

step() builds every parameter's update expression against the pre-step values (the device
defers ASSIGN writes until the whole batch is computed), so update order can't matter.
Semantics match torch.optim exactly; test_optim.py holds them to it.

State is float32 whatever the parameter is, since a moment accumulated in float16 loses the
small updates it exists to carry. The update promotes with it, so a narrower parameter rounds
once on the way back into its own dtype.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from limn.tensor import Tensor, no_grad, realize


class Optimizer:
    def __init__(self, params: Iterable[Tensor]):
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("optimizer got no parameters with requires_grad=True")

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        with no_grad():
            updated = self.updates()
        realize(*updated)

    def updates(self) -> list[Tensor]:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float, momentum: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [Tensor.zeros(p.shape) for p in self.params] if momentum else [None] * len(self.params)

    def updates(self) -> list[Tensor]:
        updated: list[Tensor] = []
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            g = p.grad
            if v is not None:
                updated.append(v.assign(self.momentum * v + g))
                g = v
            updated.append(p.assign((p - self.lr * g).cast(p.dtype)))
        return updated


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [Tensor.zeros(p.shape) for p in self.params]
        self.v = [Tensor.zeros(p.shape) for p in self.params]
        self.t = 0

    def step(self) -> None:
        """Advance the step count here, so that inspecting updates() leaves optimizer state alone."""
        self.t += 1
        super().step()

    def bias_correction(self, beta: float) -> Tensor:
        """1 - beta**t, as a buffer rather than a literal.

        This is the one number in the update that changes every step. A literal would change the
        emitted source with it, so every step would hash differently and pay a full compile; as
        bytes the source is identical and compiles once.
        """
        return Tensor(np.array([1 - beta**self.t], dtype=np.float32))

    def updates(self) -> list[Tensor]:
        updated: list[Tensor] = []
        corrections = (self.bias_correction(self.beta1), self.bias_correction(self.beta2))
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            new_m = self.beta1 * m + (1 - self.beta1) * g
            new_v = self.beta2 * v + (1 - self.beta2) * g * g
            m_hat = new_m / corrections[0]
            v_hat = new_v / corrections[1]
            update = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p  # decoupled decay, torch AdamW
            updated += [m.assign(new_m), v.assign(new_v), p.assign((p - self.lr * update).cast(p.dtype))]
        return updated
