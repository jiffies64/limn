"""Optimizers. Updates are ASSIGN graphs, committed in one realize() batch per step.

step() builds every parameter's update expression against the pre-step values (the device
defers ASSIGN writes until the whole batch is computed), so update order can't matter.
Semantics match torch.optim exactly; test_optim.py holds them to it.
"""

from __future__ import annotations

import math
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
            updated.append(p.assign(p - self.lr * g))
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
            updated += [m.assign(new_m), v.assign(new_v), p.assign(p - self.lr * update)]
        return updated


class Muon(Optimizer):
    """Orthogonalized momentum, for 2D parameters only.

    Route embeddings, output heads, biases and norm gains to AdamW; Newton-Schulz
    is only defined on matrices.
    """

    NS_COEFFS = (3.4445, -4.7750, 2.0315)  # Jordan et al., tuned for 5 steps

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ):
        super().__init__(params)
        if any(len(p.shape) != 2 for p in self.params):
            raise ValueError("Muon takes 2D parameters only; reshape convs, send the rest to AdamW")
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.eps = eps
        self.buf = [Tensor.zeros(p.shape) for p in self.params]

    def newton_schulz(self, mat: Tensor) -> Tensor:
        """Approximate the matrix sign function: drives every singular value toward 1.

        Cheaper than an SVD by a mile, and the approximation being loose doesn't hurt.
        Iterates the odd polynomial aX + bX^3 + cX^5, written in matrix form.
        """
        a, b, c = self.NS_COEFFS
        transposed = mat.shape[0] > mat.shape[1]  # iterate on the wide orientation
        if transposed:
            mat = mat.transpose()

        mat = mat / ((mat**2).sum().sqrt() + self.eps)  # singular values into [0, 1]

        for _ in range(self.ns_steps):
            gram = mat @ mat.transpose()
            poly = b * gram + c * (gram @ gram)
            mat = a * mat + poly @ mat

        return mat.transpose() if transposed else mat

    def updates(self) -> list[Tensor]:
        updated: list[Tensor] = []
        for p, buf in zip(self.params, self.buf):
            if p.grad is None:
                continue
            new_buf = self.momentum * buf + p.grad
            direction = p.grad + self.momentum * new_buf if self.nesterov else new_buf

            ortho = self.newton_schulz(direction)

            # Shape-scaled step, so matrices of different sizes move by a comparable
            # amount and one lr works for the whole model (Moonlight, arXiv 2502.16982).
            rows, cols = p.shape
            scaled_lr = self.lr * 0.2 * math.sqrt(max(rows, cols))

            decayed = p - self.lr * self.weight_decay * p  # decoupled, base lr
            updated += [buf.assign(new_buf), p.assign(decayed - scaled_lr * ortho)]
        return updated
