"""Neural network layers, composed purely from Tensor ops.

Layers are plain classes holding parameter Tensors; parameters() collects them for an
optimizer by walking attributes. Weight layouts match torch (Linear stores (out, in)) so
tests can copy state across frameworks without transposing.
"""

from __future__ import annotations

import itertools
import math

from limn.tensor import Tensor, no_grad


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
    """Batch normalization over (N, C, *spatial): statistics shared across every axis but channels.

    Training normalizes by this batch's mean and biased variance, and folds them into
    running_mean / running_var through one ASSIGN each. An assign only commits when the step
    realizes it, and it refuses a target that is not a buffer, so the training loop must
    realize the stats with the step's other sinks — realize(loss, bn.running_mean,
    bn.running_var), the transaction serialize.load_into builds — or the second forward
    raises. The transaction rule then guarantees the step read pre-update stats.

    The stat semantics are torch's: momentum weights the new observation, and running_var
    stores the unbiased variance (times n/(n-1)) the biased normalization did not use.
    training is a plain Python flag on the layer: eval normalizes by the running stats and
    queues no assign, so it cannot mutate state. A captured training step replays the assigns
    into the same buffers, but the flag is a Python value and freezes at record time.
    """

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1):
        self.weight = Tensor.ones((channels,), requires_grad=True)
        self.bias = Tensor.zeros((channels,), requires_grad=True)
        self.running_mean = Tensor.zeros((channels,))
        self.running_var = Tensor.ones((channels,))
        self.eps = eps
        self.momentum = momentum
        self.training = True

    def __call__(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(f"BatchNorm needs (batch, channels, ...), got input {x.shape}")
        axes = (0, *range(2, x.ndim))  # every axis but channels
        stats = (1, x.shape[1], *(1,) * (x.ndim - 2))
        gain, shift = self.weight.reshape(*stats), self.bias.reshape(*stats)
        if not self.training:
            mean, var = self.running_mean.reshape(*stats), self.running_var.reshape(*stats)
            return (x - mean) / (var + self.eps).sqrt() * gain + shift
        count = math.prod(x.shape[d] for d in axes)
        if count == 1:
            raise ValueError(f"BatchNorm needs more than one value per channel when training, got input {x.shape}")
        mean = x.mean(axis=axes, keepdim=True)
        centered = x - mean
        variance = (centered * centered).mean(axis=axes, keepdim=True)  # biased, like torch's normalization
        with no_grad():  # the stat update records no graph, so it costs no gradient and replays in place
            unbiased = variance * (count / (count - 1))  # torch's running var: unbiased, never the biased one
            self.running_mean.assign(self.running_mean * (1 - self.momentum) + mean.reshape(x.shape[1]) * self.momentum)
            self.running_var.assign(self.running_var * (1 - self.momentum) + unbiased.reshape(x.shape[1]) * self.momentum)
        return centered / (variance + self.eps).sqrt() * gain + shift


def as_dims(value: int | tuple[int, ...], n: int, name: str) -> tuple[int, ...]:
    """Spatial arguments are either one int for every dim or an n-tuple, like torch."""
    if isinstance(value, int):
        return (value,) * n
    if len(value) != n:
        raise ValueError(f"{name} takes an int or {n}-tuple, got {value!r}")
    return tuple(int(v) for v in value)


class Conv:
    """Cross-correlation over (batch, channels, *spatial), composed from pad/shrink/reshape and one
    broadcast multiply-reduce per kernel tap; Conv1d and Conv2d fix the number of spatial dims.

    Weight layout is torch's (out_channels, in_channels // groups, *kernel_size), and so is the
    init: uniform over 1/sqrt(fan_in). padding is an int, one int per spatial dim, or 'same'.
    """

    spatial: int  # set by the subclasses

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] | str = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        if groups < 1 or in_channels % groups or out_channels % groups:
            raise ValueError(f"groups={groups} must divide in_channels={in_channels} and out_channels={out_channels}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = as_dims(kernel_size, self.spatial, "kernel_size")
        self.stride = as_dims(stride, self.spatial, "stride")
        self.dilation = as_dims(dilation, self.spatial, "dilation")
        self.groups = groups
        if any(v < 1 for v in self.kernel_size + self.stride + self.dilation):
            raise ValueError(
                f"kernel_size={self.kernel_size}, stride={self.stride} and dilation={self.dilation} must be positive"
            )
        if isinstance(padding, str):
            if padding.lower() != "same":
                raise ValueError(f"invalid padding {padding!r}, the only string padding is 'same'")
            if self.stride != (1,) * self.spatial:
                raise ValueError("padding='same' is not supported for strided convolutions")
            # hold the output size: dilation*(k-1) split across the sides, the odd one going after, like torch
            spans = (d * (k - 1) for d, k in zip(self.dilation, self.kernel_size))
            self.padding = tuple((span // 2, span - span // 2) for span in spans)
        else:
            self.padding = tuple((p, p) for p in as_dims(padding, self.spatial, "padding"))
        bound = 1 / math.sqrt(in_channels // groups * math.prod(self.kernel_size))
        shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight = Tensor.uniform(shape, -bound, bound, requires_grad=True)
        self.bias = Tensor.uniform((out_channels,), -bound, bound, requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        name = type(self).__name__
        if x.ndim != self.spatial + 2:
            raise ValueError(f"{name} takes (batch, channels) plus {self.spatial} spatial dims, got input {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"{name} over {self.in_channels} channels got input {x.shape}")
        batch = x.shape[0]
        # per spatial axis: the output size, and pads sized so every tap's window below is exactly
        # stride*out long (up to stride-1 zeros past the requested pad; the stride shrink drops them)
        geometry = []
        for size, (before, after), k, s, d in zip(x.shape[2:], self.padding, self.kernel_size, self.stride, self.dilation):
            out = (size + before + after - d * (k - 1) - 1) // s + 1
            geometry.append((out, before, d * (k - 1) + s * out - size - before))
        outs = tuple(out for out, _, _ in geometry)
        if any(out < 1 for out in outs):
            raise ValueError(f"{name} kernel {self.kernel_size} does not fit input {x.shape} with padding {self.padding}")

        in_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        # split channels into (groups, in_per_group) while x is still contiguous, so every step below stays
        # a view; merging dims of a padded slice instead would force a full copy per kernel tap
        padded = x.reshape(batch, self.groups, in_per_group, *x.shape[2:]).pad(
            ((0, 0), (0, 0), (0, 0), *((before, after) for _, before, after in geometry))
        )
        weight = self.weight.reshape(self.groups, out_per_group, in_per_group, *self.kernel_size)

        total = None
        for taps in itertools.product(*(range(k) for k in self.kernel_size)):
            starts = tuple(t * d for t, d in zip(taps, self.dilation))
            windows = tuple((start, start + s * out) for start, s, out in zip(starts, self.stride, outs))
            tap = padded.shrink(((0, batch), (0, self.groups), (0, in_per_group), *windows))
            # split each spatial axis into (out, stride) and keep index 0 of the stride; a no-op at stride 1
            tap = tap.reshape(batch, self.groups, in_per_group, *itertools.chain.from_iterable(zip(outs, self.stride)))
            keep = itertools.chain.from_iterable(((0, out), (0, 1)) for out in outs)
            tap = tap.shrink(((0, batch), (0, self.groups), (0, in_per_group), *keep))
            tap = tap.reshape(batch, self.groups, 1, in_per_group, *outs)
            tap_weight = weight.shrink(((0, self.groups), (0, out_per_group), (0, in_per_group), *((t, t + 1) for t in taps)))
            tap_weight = tap_weight.reshape(1, self.groups, out_per_group, in_per_group, *(1,) * self.spatial)
            # a matmul over in_per_group, spelled as multiply+sum so the spatial dims never merge
            term = (tap_weight * tap).sum(axis=3)
            total = term if total is None else total + term

        assert total is not None  # kernel dims are at least 1, so the loop always runs
        out = total.reshape(batch, self.out_channels, *outs)
        return out if self.bias is None else out + self.bias.reshape(1, self.out_channels, *(1,) * self.spatial)


class Conv1d(Conv):
    spatial = 1


class Conv2d(Conv):
    spatial = 2


class Embedding:
    def __init__(self, vocab_size: int, dim: int):
        self.weight = Tensor.randn((vocab_size, dim), requires_grad=True)

    def __call__(self, indices: Tensor) -> Tensor:
        """Rows of weight selected by int32 indices (...,) -> (..., dim)."""
        return self.weight.gather_rows(indices)


def named_parameters(module: object) -> list[tuple[str, Tensor]]:
    """Every requires_grad Tensor reachable from module's attributes, depth-first, deduplicated,
    under the dotted path the walk took to reach it: "blocks.0.attn.w.weight".

    The path is an attribute name per module, an index per list or tuple, and a key per dict, so
    it is stable across runs while a position in parameters() is stable only across a fixed
    build order. That is what a checkpoint keys on, and the whole reason this walk carries a name.

    Containers are tracked by identity too, so a module that holds a reference to itself, or two
    that hold each other, is walked once rather than forever. A tensor two paths reach is one
    parameter, named for the first path there.
    """
    found: list[tuple[str, Tensor]] = []
    seen: set[int] = set()

    def walk(obj: object, path: str) -> None:
        if isinstance(obj, Tensor):
            if obj.requires_grad and id(obj) not in seen:
                seen.add(id(obj))
                found.append((path, obj))
            return
        if id(obj) in seen:
            return
        if isinstance(obj, (list, tuple)):
            children = enumerate(obj)
        elif isinstance(obj, dict):
            children = obj.items()
        elif hasattr(obj, "__dict__"):
            children = vars(obj).items()
        else:
            return
        seen.add(id(obj))
        for key, value in children:
            walk(value, f"{path}.{key}" if path else str(key))

    walk(module, "")
    return found


def parameters(module: object) -> list[Tensor]:
    """The parameters named_parameters finds, without their names: what an optimizer takes.

    One walk under both, so a container type either walk learns about, the other has already.
    """
    return [p for _, p in named_parameters(module)]
