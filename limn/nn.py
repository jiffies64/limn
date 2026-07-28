"""Neural network layers, composed purely from Tensor ops.

Layers are plain classes holding parameter Tensors; parameters() collects them for an
optimizer by walking attributes. Weight layouts match torch (Linear stores (out, in)) so
tests can copy state across frameworks without transposing.
"""

from __future__ import annotations

import itertools
import math

from limn.tensor import Tensor, realize


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


class BatchNorm:
    """Normalize (N, C, ...) over every axis but the channels, like torch.nn.BatchNorm2d.

    Training normalizes by the batch statistics and commits them into the running estimates
    in place; eval normalizes by the running estimates. Toggle self.training by hand.
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.weight = Tensor.ones((dim,), requires_grad=True)
        self.bias = Tensor.zeros((dim,), requires_grad=True)
        self.running_mean = Tensor.zeros((dim,))
        self.running_var = Tensor.ones((dim,))
        self.eps = eps
        self.momentum = momentum
        self.training = True

    def __call__(self, x: Tensor) -> Tensor:
        # x is (N, C) or (N, C, H, W); reduce everything except C
        axes = tuple(i for i in range(x.ndim) if i != 1)
        shape = tuple(x.shape[1] if i == 1 else 1 for i in range(x.ndim))

        if self.training:
            mean = x.mean(axis=axes, keepdim=True)
            centered = x - mean
            variance = (centered * centered).mean(axis=axes, keepdim=True)  # biased, like torch's normalization
            count = x.numel // x.shape[1]
            unbiased = variance.detach().reshape(-1) * (count / (count - 1))  # the running estimate is unbiased, like torch
            # committed in place; rebinding the attributes instead would grow an unrealized graph every step
            realize(
                self.running_mean.assign((1 - self.momentum) * self.running_mean + self.momentum * mean.detach().reshape(-1)),
                self.running_var.assign((1 - self.momentum) * self.running_var + self.momentum * unbiased),
            )
        else:
            mean = self.running_mean.reshape(*shape)
            centered = x - mean
            variance = self.running_var.reshape(*shape)

        return centered / (variance + self.eps).sqrt() * self.weight.reshape(*shape) + self.bias.reshape(*shape)


def as_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    """Spatial arguments are either one int for both dims or an (h, w) pair, like torch."""
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"{name} takes an int or an (h, w) pair, got {value!r}")
    return (value[0], value[1])


class Conv2d:
    """Cross-correlation over (N, C, H, W), composed from pad/shrink/reshape and one broadcast
    multiply-reduce per kernel tap.

    Weight layout is torch's (out_channels, in_channels // groups, kh, kw), and so is the init:
    uniform over 1/sqrt(fan_in). padding is an int, an (h, w) pair, or 'same'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        if groups < 1 or in_channels % groups or out_channels % groups:
            raise ValueError(f"groups={groups} must divide in_channels={in_channels} and out_channels={out_channels}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = as_pair(kernel_size, "kernel_size")
        self.stride = as_pair(stride, "stride")
        self.dilation = as_pair(dilation, "dilation")
        self.groups = groups
        if isinstance(padding, str):
            if padding.lower() != "same":
                raise ValueError(f"invalid padding {padding!r}, the only string padding is 'same'")
            if self.stride != (1, 1):
                raise ValueError("padding='same' is not supported for strided convolutions")
            # hold the output size: dilation*(k-1) split across the sides, the odd one going after, like torch
            spans = (d * (k - 1) for d, k in zip(self.dilation, self.kernel_size))
            self.padding = tuple((span // 2, span - span // 2) for span in spans)
        else:
            self.padding = tuple((p, p) for p in as_pair(padding, "padding"))
        bound = 1 / math.sqrt(in_channels // groups * math.prod(self.kernel_size))
        shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight = Tensor.uniform(shape, -bound, bound, requires_grad=True)
        self.bias = Tensor.uniform((out_channels,), -bound, bound, requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Conv2d takes (N, C, H, W) input, got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Conv2d over {self.in_channels} channels got input {x.shape}")
        batch = x.shape[0]
        # per spatial axis: the output size, and pads reaching exactly the last index any window reads
        # (with stride > 1 the far pad can be less than requested, since a trailing remainder is never read)
        geometry = []
        for size, (before, after), k, s, d in zip(x.shape[2:], self.padding, self.kernel_size, self.stride, self.dilation):
            out = (size + before + after - d * (k - 1) - 1) // s + 1
            geometry.append((out, before, d * (k - 1) + s * out - size - before))
        (out_h, top, bottom), (out_w, left, right) = geometry
        if out_h < 1 or out_w < 1:
            raise ValueError(f"Conv2d kernel {self.kernel_size} does not fit input {x.shape} with padding {self.padding}")

        kh, kw = self.kernel_size
        dilation_h, dilation_w = self.dilation
        stride_h, stride_w = self.stride
        in_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        # split channels into (groups, in_per_group) while x is still contiguous, so every step below stays
        # a view; merging dims of a padded slice instead would force a full copy per kernel tap
        padded = x.reshape(batch, self.groups, in_per_group, *x.shape[2:]).pad(
            ((0, 0), (0, 0), (0, 0), (top, bottom), (left, right))
        )
        weight = self.weight.reshape(self.groups, out_per_group, in_per_group, kh, kw)

        total = None
        for i, j in itertools.product(range(kh), range(kw)):
            start_h, start_w = i * dilation_h, j * dilation_w
            window = ((start_h, start_h + stride_h * out_h), (start_w, start_w + stride_w * out_w))
            tap = padded.shrink(((0, batch), (0, self.groups), (0, in_per_group), *window))
            # split each spatial axis into (out, stride) and keep index 0 of the stride; a no-op at stride 1
            tap = tap.reshape(batch, self.groups, in_per_group, out_h, stride_h, out_w, stride_w)
            tap = tap.shrink(((0, batch), (0, self.groups), (0, in_per_group), (0, out_h), (0, 1), (0, out_w), (0, 1)))
            tap = tap.reshape(batch, self.groups, 1, in_per_group, out_h, out_w)
            tap_weight = weight.shrink(((0, self.groups), (0, out_per_group), (0, in_per_group), (i, i + 1), (j, j + 1)))
            tap_weight = tap_weight.reshape(1, self.groups, out_per_group, in_per_group, 1, 1)
            # a matmul over in_per_group, spelled as multiply+sum so the spatial dims never merge
            term = (tap_weight * tap).sum(axis=3)
            total = term if total is None else total + term

        assert total is not None  # kernel dims are at least 1, so the loop always runs
        out = total.reshape(batch, self.out_channels, out_h, out_w)
        return out if self.bias is None else out + self.bias.reshape(1, self.out_channels, 1, 1)


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
