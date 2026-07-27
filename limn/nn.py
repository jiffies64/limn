"""Neural network layers, composed purely from Tensor ops.

Layers are plain classes holding parameter Tensors; parameters() collects them for an
optimizer by walking attributes. Weight layouts match torch (Linear stores (out, in)) so
tests can copy state across frameworks without transposing.
"""

from __future__ import annotations

import math

from limn.tensor import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        bound = 1 / math.sqrt(in_features)
        self.weight = Tensor.uniform((out_features, in_features), -bound, bound, requires_grad=True)
        self.bias = Tensor.uniform((out_features,), -bound, bound, requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose()
        return out if self.bias is None else out + self.bias


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = Tensor.ones((dim,), requires_grad=True)
        self.bias = Tensor.zeros((dim,), requires_grad=True)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        centered = x - x.mean(axis=-1, keepdim=True)
        variance = (centered * centered).mean(axis=-1, keepdim=True)  # biased, like torch.nn.LayerNorm
        return centered / (variance + self.eps).sqrt() * self.weight + self.bias


class Embedding:
    def __init__(self, vocab_size: int, dim: int):
        self.weight = Tensor.randn((vocab_size, dim), requires_grad=True)

    def __call__(self, indices: Tensor) -> Tensor:
        """Rows of weight selected by int32 indices (...,) -> (..., dim)."""
        return self.weight.gather_rows(indices)


def parameters(module: object) -> list[Tensor]:
    """Every requires_grad Tensor reachable from module's attributes, depth-first, deduplicated.

    Containers are tracked by identity too, so a module that holds a reference to itself, or two
    that hold each other, is walked once rather than forever.
    """
    found: list[Tensor] = []
    seen: set[int] = set()

    def walk(obj: object) -> None:
        if isinstance(obj, Tensor):
            if obj.requires_grad and id(obj) not in seen:
                seen.add(id(obj))
                found.append(obj)
            return
        if id(obj) in seen:
            return
        if isinstance(obj, (list, tuple)):
            children = obj
        elif isinstance(obj, dict):
            children = obj.values()
        elif hasattr(obj, "__dict__"):
            children = vars(obj).values()
        else:
            return
        seen.add(id(obj))
        for value in children:
            walk(value)

    walk(module)
    return found
