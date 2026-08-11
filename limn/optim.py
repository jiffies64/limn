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

import math
from collections.abc import Iterable, Mapping, Sequence

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

    def slots(self) -> dict[str, Sequence[Tensor | None]]:
        """The per-parameter state this optimizer carries between steps, by slot name, one entry
        per parameter in order. None is a parameter with nothing in that slot, as SGD without
        momentum has."""
        return {}

    def state_dict(self, named: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Everything this optimizer must keep across a stop, as "opt.<parameter name>.<slot>".

        Keyed by the parameter's name rather than by its position, so a model whose layers are
        built in another order still resumes onto the right state. The names come in as an
        argument because an optimizer is given parameters, not a module: pass nn.named_parameters
        of whatever the parameters came from.

        The tensors are the live state buffers, so the map serves both directions: save_file reads
        them, and load_into assigns the file straight back into them.
        """
        names = {id(p): name for name, p in named.items()}
        unnamed = [p for p in self.params if id(p) not in names]
        if unnamed:
            raise ValueError(f"{len(unnamed)} of {len(self.params)} parameters are not in the given names")
        return {
            f"opt.{names[id(p)]}.{slot}": state
            for slot, states in self.slots().items()
            for p, state in zip(self.params, states)
            if state is not None
        }


class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float, momentum: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [state_like(p) for p in self.params] if momentum else [None] * len(self.params)

    def slots(self) -> dict[str, Sequence[Tensor | None]]:
        return {"momentum": self.velocity}

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

    def slots(self) -> dict[str, Sequence[Tensor | None]]:
        return {"m": self.m, "v": self.v}

    def state_dict(self, named: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """The moments, and beta**t alongside them: it belongs to the step count rather than to
        any one parameter, and a resume that left it at 1 would bias-correct a warm run as if it
        had just started."""
        powers = {f"opt.{dtype}.beta{i + 1}_t": t for dtype, pair in self.powers.items() for i, t in enumerate(pair)}
        return super().state_dict(named) | powers

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


class Muon(Optimizer):
    """Orthogonalized momentum, for 2D parameters only (Jordan et al.'s Muon).

    Route embeddings, output heads, biases and norm gains to AdamW; Newton-Schulz is only
    defined on matrices, and the shape-scaled step assumes hidden-layer geometry.

    The buffer keeps the original accumulator convention, buf = momentum * buf + grad, where
    torch.optim.Muon keeps an EMA. The two buffers differ by a factor of 1 - momentum, which
    the Frobenius normalization inside Newton-Schulz cancels, so under torch's
    match_rms_adamw scaling the steps agree to the bfloat16 torch orthogonalizes in.
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
        self.buf = [state_like(p) for p in self.params]

    def slots(self) -> dict[str, Sequence[Tensor | None]]:
        return {"momentum": self.buf}

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

    def updates(self) -> list[tuple[Tensor, Tensor]]:
        updates: list[tuple[Tensor, Tensor]] = []
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
            updates += [(buf, new_buf), (p, (decayed - scaled_lr * ortho).cast(p.dtype))]
        return updates
