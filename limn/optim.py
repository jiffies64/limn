"""Optimizers. Updates are ASSIGN graphs, committed in one realize() batch per step.

step() builds every parameter's update expression against the pre-step values (the device
defers ASSIGN writes until the whole batch is computed), so update order can't matter.
Semantics match torch.optim exactly; test_optim.py holds them to it.

State is the parameter's dtype, never narrower than float32: a moment accumulated in float16
loses the small updates it exists to carry, and a float64 parameter would forfeit its width
to float32 state. The update promotes with the state, so a narrower parameter rounds once on
the way back into its own dtype.

Everything that changes from one step to the next lives in device buffers, AdamW's beta**t
included. A step therefore builds the same graph every time, which is what lets one compiled
plan serve the whole run and a captured step (limn.capture) replay with no host bookkeeping.
"""

from __future__ import annotations

from collections.abc import Iterable

from limn.ops import DType, float32, promote
from limn.tensor import Tensor, no_grad, realize


def state_like(p: Tensor) -> Tensor:
    return Tensor.zeros(p.shape, dtype=promote(p.dtype, float32))


class Optimizer:
    def __init__(self, params: Iterable[Tensor]):
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("optimizer got no parameters with requires_grad=True")

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self, *also: Tensor) -> None:
        """Commit this step's updates in one batch.

        Tensors passed in realize in that same batch: a loss handed here shares the forward
        pass with the gradients, instead of recomputing it when it is read afterwards, and it
        reads the pre-step parameters like every other value in the batch.
        """
        with no_grad():
            updates = self.updates()
        realize(*[target.assign(value) for target, value in updates], *also)

    def updates(self) -> list[tuple[Tensor, Tensor]]:
        """(target, new value) for every tensor this step writes; building them commits nothing."""
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float, momentum: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [state_like(p) for p in self.params] if momentum else [None] * len(self.params)

    def updates(self) -> list[tuple[Tensor, Tensor]]:
        updates: list[tuple[Tensor, Tensor]] = []
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            g = p.grad
            if v is not None:
                g = self.momentum * v + g
                updates.append((v, g))
            updates.append((p, (p - self.lr * g).cast(p.dtype)))
        return updates


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
        self.m = [state_like(p) for p in self.params]
        self.v = [state_like(p) for p in self.params]
        # beta**t, held on the device and advanced by one multiply inside each step's graph. As a
        # literal it would change the emitted source every step and pay a compile each time; as a
        # host-side buffer it would be the one value a replayed step could not advance. One pair
        # per state dtype, so a float64 step is not scaled through a float32 rounding.
        self.powers: dict[DType, tuple[Tensor, Tensor]] = {
            dtype: (Tensor.ones((1,), dtype=dtype), Tensor.ones((1,), dtype=dtype)) for dtype in {m.dtype for m in self.m}
        }

    def updates(self) -> list[tuple[Tensor, Tensor]]:
        updates: list[tuple[Tensor, Tensor]] = []
        scales: dict[DType, tuple[Tensor, Tensor]] = {}
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            if m.dtype not in scales:
                pow1, pow2 = self.powers[m.dtype]
                new_pow1, new_pow2 = pow1 * self.beta1, pow2 * self.beta2
                updates += [(pow1, new_pow1), (pow2, new_pow2)]
                # one reciprocal node per dtype: every parameter multiplies by the same buffer,
                # where a division per parameter would cut a kernel for each
                scales[m.dtype] = ((1 - new_pow1).reciprocal(), (1 - new_pow2).reciprocal())
            inv1, inv2 = scales[m.dtype]
            g = p.grad
            new_m = self.beta1 * m + (1 - self.beta1) * g
            new_v = self.beta2 * v + (1 - self.beta2) * g * g
            m_hat = new_m * inv1
            v_hat = new_v * inv2
            update = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p  # decoupled decay, torch AdamW
            updates += [(m, new_m), (v, new_v), (p, (p - self.lr * update).cast(p.dtype))]
        return updates
