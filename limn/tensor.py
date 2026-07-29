"""Tensor: the user-facing API. Lazy graph building, broadcasting, composed ops, autograd.

Every method here only builds Nodes (ops.py); nothing computes until .realize() or .numpy()
hands the graph to the active device. Autograd lives at this layer: each op-created Tensor
remembers its parent Tensors (`parents`) and a closure (`grad_fn`) that turns the output
gradient into parent gradients, themselves lazy Tensors. backward() walks that record in
reverse topological order. Composed ops (matmul, softmax, ...) get their gradients for free
from the pieces. Everything in this module is fair game for the rest of limn; the public
surface is whatever __init__.py re-exports.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import EllipsisType
from typing import SupportsIndex

import numpy as np

from limn import device
from limn.ops import DTYPES, FLOATS, DType, Node, Op, float16, float32, int32, promote, topological
from limn.view import View

grad_enabled: bool = True
rng: np.random.Generator = np.random.default_rng(0)


def set_seed(seed: int) -> None:
    global rng
    rng = np.random.default_rng(seed)


@contextmanager
def no_grad() -> Iterator[None]:
    global grad_enabled
    previous, grad_enabled = grad_enabled, False
    try:
        yield
    finally:
        grad_enabled = previous


BackwardFn = Callable[["Tensor"], tuple["Tensor | None", ...]]
type Index = SupportsIndex | slice | EllipsisType | Tensor | None


class Tensor:
    def __init__(self, data: np.ndarray | list | float | int, dtype: DType | None = None, requires_grad: bool = False):
        array = np.asarray(data)
        if dtype is None:
            dtype = int32 if array.dtype.kind in "iub" else float32
        if dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {dtype}; limn has {[str(d) for d in DTYPES]}")
        array = array.astype(device.NUMPY_DTYPES[dtype])
        if array.size == 0:
            raise ValueError(f"zero-size tensors are not supported (shape {array.shape})")
        dev = device.active()
        buf = dev.alloc(array.nbytes)
        dev.copyin(buf, array)
        self.node: Node = Node(Op.BUFFER, (), dtype, array.shape, buf)
        if requires_grad and dtype not in FLOATS:
            raise ValueError(f"requires_grad needs a float dtype, got {dtype}")
        self.requires_grad: bool = requires_grad
        self.grad: Tensor | None = None
        self.parents: tuple[Tensor, ...] = ()
        self.grad_fn: BackwardFn | None = None

    @staticmethod
    def from_node(node: Node, parents: tuple[Tensor, ...] = (), grad_fn: BackwardFn | None = None) -> Tensor:
        """Wrap an existing graph node; the autograd record is kept only if a parent needs it."""
        t = Tensor.__new__(Tensor)
        t.node = node
        t.requires_grad = grad_enabled and any(p.requires_grad for p in parents)
        t.grad = None
        t.parents = parents if t.requires_grad else ()
        t.grad_fn = grad_fn if t.requires_grad else None
        return t

    @staticmethod
    def const(value: float | int, dtype: DType) -> Tensor:
        """A scalar CONST node: lives in the graph, no buffer behind it."""
        return Tensor.from_node(Node(Op.CONST, (), dtype, (), value))

    # ---- creation helpers (host-side numpy, loaded as buffers) ----

    @staticmethod
    def full(shape: Sequence[int], value: float | int, dtype: DType = float32, requires_grad: bool = False) -> Tensor:
        return Tensor(np.full(shape, value), dtype=dtype, requires_grad=requires_grad)

    @staticmethod
    def zeros(shape: Sequence[int], dtype: DType = float32, requires_grad: bool = False) -> Tensor:
        return Tensor.full(shape, 0, dtype, requires_grad)

    @staticmethod
    def ones(shape: Sequence[int], dtype: DType = float32, requires_grad: bool = False) -> Tensor:
        return Tensor.full(shape, 1, dtype, requires_grad)

    @staticmethod
    def arange(n: int, dtype: DType = int32) -> Tensor:
        return Tensor(np.arange(n), dtype=dtype)

    @staticmethod
    def randn(shape: Sequence[int], std: float = 1.0, requires_grad: bool = False, dtype: DType = float32) -> Tensor:
        return Tensor(rng.standard_normal(shape) * std, dtype=dtype, requires_grad=requires_grad)

    @staticmethod
    def uniform(shape: Sequence[int], low: float, high: float, requires_grad: bool = False, dtype: DType = float32) -> Tensor:
        return Tensor(rng.uniform(low, high, shape), dtype=dtype, requires_grad=requires_grad)

    # ---- basic properties ----

    @property
    def shape(self) -> tuple[int, ...]:
        return self.node.shape

    @property
    def dtype(self) -> DType:
        return self.node.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, op={self.node.op.name}, requires_grad={self.requires_grad})"

    # ---- realization ----

    def realize(self) -> Tensor:
        realize(self)
        return self

    def numpy(self) -> np.ndarray:
        buf = realize(self)[0]
        return device.active().copyout(buf).view(device.NUMPY_DTYPES[self.dtype]).reshape(self.shape)

    def item(self) -> float | int:
        return self.numpy().item()

    def detach(self) -> Tensor:
        return Tensor.from_node(self.node)

    def assign(self, value: Tensor) -> Tensor:
        """Queue an in-place overwrite of this tensor's buffer; committed on realize()."""
        if self.node.op is not Op.BUFFER:
            raise ValueError(f"assign target must be a realized buffer, this tensor is {self.node.op.name}")
        if value.shape != self.shape or value.dtype != self.dtype:
            raise ValueError(f"assign mismatch: target {self.shape} {self.dtype}, value {value.shape} {value.dtype}")
        self.node = Node(Op.ASSIGN, (self.node, value.node), self.dtype, self.shape)
        return self

    # ---- movement ops; gradients are the inverse movement ----

    def view(self) -> View:
        """This tensor's View: the composed one if it is a VIEW node, identity otherwise."""
        return self.node.arg if self.node.op is Op.VIEW else View.contiguous(self.shape)

    def moved(self, new_view: View, grad_fn: BackwardFn) -> Tensor:
        """Rewrap the base node with a composed View; grad_fn is the inverse movement."""
        base = self.node.srcs[0] if self.node.op is Op.VIEW else self.node
        node = Node(Op.VIEW, (base,), self.dtype, new_view.shape, new_view)
        return Tensor.from_node(node, (self,), grad_fn)

    def reshape(self, *shape: int) -> Tensor:
        new_shape = resolve_shape(as_ints(shape), self.numel)
        if math.prod(new_shape) != self.numel:
            raise ValueError(f"reshape: cannot view {self.shape} as {new_shape}")
        old_shape = self.shape

        def backward(g: Tensor) -> tuple[Tensor]:
            return (g.reshape(*old_shape),)

        new_view = self.view().reshape(new_shape)
        if new_view is not None:
            return self.moved(new_view, backward)
        # layout can't be reindexed in place: copy to contiguous, then view that
        copy = Node(Op.CONTIGUOUS, (self.node,), self.dtype, self.shape)
        node = Node(Op.VIEW, (copy,), self.dtype, new_shape, View.contiguous(new_shape))
        return Tensor.from_node(node, (self,), backward)

    def permute(self, *order: int) -> Tensor:
        axes = tuple(wrap_axis(d, self.ndim) for d in as_ints(order))
        inverse = tuple(axes.index(d) for d in range(self.ndim))
        return self.moved(self.view().permute(axes), lambda g: (g.permute(*inverse),))

    def transpose(self, dim0: int = -2, dim1: int = -1) -> Tensor:
        order = list(range(self.ndim))
        d0, d1 = wrap_axis(dim0, self.ndim), wrap_axis(dim1, self.ndim)
        order[d0], order[d1] = order[d1], order[d0]
        return self.permute(*order)

    def expand(self, *shape: int) -> Tensor:
        sizes = as_ints(shape)
        if len(sizes) != self.ndim:
            raise ValueError(f"expand: {sizes} has {len(sizes)} dims, tensor {self.shape} has {self.ndim}")
        target = tuple(old if new == -1 else new for old, new in zip(self.shape, sizes))
        grown = tuple(d for d, (old, new) in enumerate(zip(self.shape, target)) if old != new)
        return self.moved(self.view().expand(target), lambda g: (g.sum(axis=grown, keepdim=True),))

    def pad(self, padding: Sequence[tuple[int, int]]) -> Tensor:
        pads = tuple((int(b), int(a)) for b, a in padding)
        bounds = tuple((before, before + size) for size, (before, _) in zip(self.shape, pads))
        return self.moved(self.view().pad(pads), lambda g: (g.shrink(bounds),))

    def shrink(self, bounds: Sequence[tuple[int, int]]) -> Tensor:
        cuts = tuple((int(lo), int(hi)) for lo, hi in bounds)
        padding = tuple((lo, size - hi) for size, (lo, hi) in zip(self.shape, cuts))
        return self.moved(self.view().shrink(cuts), lambda g: (g.pad(padding),))

    def flatten(self) -> Tensor:
        return self.reshape(-1)

    # ---- elementwise ops ----

    def cast(self, dtype: DType) -> Tensor:
        """Convert dtype. Between float dtypes this carries gradients; to or from an int it detaches.

        broadcast_pair inserts these itself wherever a float16 meets a float32, so detaching would
        strip requires_grad off a parameter and leave the optimizer skipping it. An int carries no
        gradient, so those casts still end the chain.
        """
        if dtype == self.dtype:
            return self
        if dtype not in DTYPES:
            raise ValueError(f"cast: unsupported dtype {dtype}")
        node = Node(Op.CAST, (self.node,), dtype, self.shape, dtype)
        if self.dtype not in FLOATS or dtype not in FLOATS:
            return Tensor.from_node(node)
        source = self.dtype  # back at the source's width, so a leaf's .grad matches the leaf
        return Tensor.from_node(node, (self,), lambda g: (g.cast(source),))

    def float(self) -> Tensor:
        return self.cast(float32)

    def int(self) -> Tensor:
        return self.cast(int32)

    def half(self) -> Tensor:
        return self.cast(float16)

    def unary(self, op: Op, grad_fn: BackwardFn) -> Tensor:
        if op in (Op.EXP, Op.LOG, Op.SQRT, Op.RECIP) and self.dtype not in FLOATS:
            raise ValueError(f"{op.name} requires a float dtype, got {self.dtype}")
        node = Node(op, (self.node,), self.dtype, self.shape)
        return Tensor.from_node(node, (self,), grad_fn)

    def __neg__(self) -> Tensor:
        return self.unary(Op.NEG, lambda g: (-g,))

    def exp(self) -> Tensor:
        out = self.unary(Op.EXP, lambda g: (g * out,))
        return out

    def log(self) -> Tensor:
        return self.unary(Op.LOG, lambda g: (g * self.reciprocal(),))

    def sqrt(self) -> Tensor:
        out = self.unary(Op.SQRT, lambda g: (g * (out * 2.0).reciprocal(),))
        return out

    def reciprocal(self) -> Tensor:
        out = self.unary(Op.RECIP, lambda g: (-g * out * out,))
        return out

    def binary(self, other: Tensor | float | int, op: Op, make_grad_fn: Callable[[Tensor, Tensor], BackwardFn]) -> Tensor:
        a, b = broadcast_pair(self, other, op.name)
        node = Node(op, (a.node, b.node), a.dtype, a.shape)
        return Tensor.from_node(node, (a, b), make_grad_fn(a, b))

    def __add__(self, other: Tensor | float | int) -> Tensor:
        return self.binary(other, Op.ADD, lambda a, b: lambda g: (g, g))

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        return self.binary(other, Op.MUL, lambda a, b: lambda g: (g * b, g * a))

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        return self + (-as_tensor(other, self))

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        other = as_tensor(other, self)
        wider = promote(self.dtype, other.dtype)
        if wider not in FLOATS:  # int over int still divides as floats, since a reciprocal has to
            wider = float32
        return self.cast(wider) * other.cast(wider).reciprocal()

    def __radd__(self, other: float | int) -> Tensor:
        return self + other

    def __rmul__(self, other: float | int) -> Tensor:
        return self * other

    def __rsub__(self, other: float | int) -> Tensor:
        return -self + other

    def __rtruediv__(self, other: float | int) -> Tensor:
        return as_tensor(other, self) / self

    def __pow__(self, exponent: int | float) -> Tensor:
        if exponent == 0.5:
            return self.sqrt()
        if float(exponent).is_integer() and exponent >= 1:
            out = self
            for _ in range(int(exponent) - 1):
                out = out * self
            return out
        raise ValueError(f"pow supports positive integer exponents and 0.5, got {exponent}; limn has no POW primitive")

    def __lt__(self, other: Tensor | float | int) -> Tensor:
        a, b = broadcast_pair(self, other, "CMPLT")  # not differentiable: never gets an autograd record
        return Tensor.from_node(Node(Op.CMPLT, (a.node, b.node), a.dtype, a.shape))

    def __gt__(self, other: Tensor | float | int) -> Tensor:
        return as_tensor(other, self) < self

    def __le__(self, other: Tensor | float | int) -> Tensor:
        return 1 - (self > other)

    def __ge__(self, other: Tensor | float | int) -> Tensor:
        return as_tensor(other, self) <= self

    def eq(self, other: Tensor | float | int) -> Tensor:
        """Elementwise equality as 0/1, composed as 1 - (a<b) - (b<a)."""
        return 1 - (self < other) - (self > other)

    def where(self, if_true: Tensor | float | int, if_false: Tensor | float | int) -> Tensor:
        """self is the condition (nonzero picks if_true). Gradient flows to the branches only."""
        x, y = as_tensor(if_true, self), as_tensor(if_false, self)
        if x.dtype != y.dtype:  # same promotion broadcast_pair does, applied across three operands
            wider = promote(x.dtype, y.dtype)
            x, y = x.cast(wider), y.cast(wider)
        shape = broadcast_shape(broadcast_shape(x.shape, y.shape, "WHERE"), self.shape, "WHERE")
        x, y = broadcast_to(x, shape, "WHERE"), broadcast_to(y, shape, "WHERE")
        cond = broadcast_to(self.detach(), shape, "WHERE")
        node = Node(Op.WHERE, (cond.node, x.node, y.node), x.dtype, shape)

        def backward(g: Tensor) -> tuple[Tensor, Tensor]:
            return cond.where(g, zero(g)), cond.where(zero(g), g)

        return Tensor.from_node(node, (x, y), backward)

    def maximum(self, other: Tensor | float | int) -> Tensor:
        other = as_tensor(other, self)
        return (self < other).where(other, self)

    def minimum(self, other: Tensor | float | int) -> Tensor:
        other = as_tensor(other, self)
        return (other < self).where(other, self)

    def relu(self) -> Tensor:
        return (self > 0).where(self, zero(self))

    # ---- reduces (primitive keeps reduced dims as size 1; keepdim=False reshapes after) ----

    def reduce(
        self,
        op: Op,
        axis: int | Sequence[int] | None,
        keepdim: bool,
        make_grad_fn: Callable[[Tensor], BackwardFn],
    ) -> Tensor:
        """make_grad_fn receives the kept-dims output, which most reduce gradients need."""
        axes = canon_axes(axis, self.ndim)
        if not axes:
            return self
        kept_shape = tuple(1 if d in axes else size for d, size in enumerate(self.shape))
        kept = Tensor.from_node(Node(op, (self.node,), self.dtype, kept_shape, axes), (self,))
        if kept.requires_grad:
            kept.grad_fn = make_grad_fn(kept)
        if keepdim:
            return kept
        return kept.reshape(*(size for d, size in enumerate(self.shape) if d not in axes))

    def sum(self, axis: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        return self.reduce(Op.SUM, axis, keepdim, lambda kept: lambda g: (broadcast_to(g, self.shape, "SUM"),))

    def max(self, axis: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        axes = canon_axes(axis, self.ndim)

        def make_grad_fn(kept: Tensor) -> BackwardFn:
            def backward(g: Tensor) -> tuple[Tensor]:
                hit = self.eq(broadcast_to(kept.detach(), self.shape, "MAX"))
                share = hit / hit.sum(axis=axes, keepdim=True)  # ties split the gradient evenly, like torch.amax
                return (broadcast_to(g, self.shape, "MAX") * share,)

            return backward

        return self.reduce(Op.MAX, axis, keepdim, make_grad_fn)

    def mean(self, axis: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        axes = canon_axes(axis, self.ndim)
        count = math.prod(self.shape[d] for d in axes) if axes else 1
        return self.sum(axis, keepdim) / float(count)

    # ---- indexed access ----

    def __getitem__(self, index: Index | tuple[Index, ...]) -> Tensor:
        """numpy-style indexing, built out of shrink and reshape: `x[0, 1:5]`, `...`, `None`.

        A step other than 1 needs a stride no View op produces, and an empty slice a zero-size
        tensor limn does not have; both raise. An int32 Tensor index stands alone and gathers rows.
        Out of range is IndexError, as numpy raises.
        """
        keys: list[Index] = list(index) if isinstance(index, tuple) else [index]
        if len(keys) == 1 and isinstance(keys[0], Tensor):
            return self.gather_rows(keys[0])
        if any(isinstance(k, Tensor) for k in keys):
            raise ValueError(f"index {index}: a Tensor index selects rows and cannot be combined with others")
        ellipses = [d for d, k in enumerate(keys) if k is Ellipsis]
        if len(ellipses) > 1:
            raise ValueError(f"index {index}: at most one Ellipsis")
        indexed = sum(k is not None and k is not Ellipsis for k in keys)
        if indexed > self.ndim:
            raise IndexError(f"index {index}: too many indices, {indexed} for {self.ndim} dims")
        fill = ellipses[0] if ellipses else len(keys)  # an absent Ellipsis is an implicit trailing one
        keys[fill : fill + 1] = [slice(None)] * (self.ndim - indexed)
        bounds: list[tuple[int, int]] = []
        shape: list[int] = []
        for key in keys:
            if key is None:
                shape.append(1)
                continue
            dim = len(bounds)
            size = self.shape[dim]
            if isinstance(key, slice):
                start, stop, step = key.indices(size)
                if step != 1:
                    raise ValueError(f"index {index}: only step-1 slices are supported")
                if start >= stop:
                    raise ValueError(f"index {index}: empty slice on dim {dim}, limn has no zero-size tensors")
                bounds.append((start, stop))
                shape.append(stop - start)
                continue
            if isinstance(key, bool):
                raise ValueError(f"index {index}: a bool index is a numpy mask, which limn does not have")
            if not isinstance(key, SupportsIndex):
                raise ValueError(f"index {index}: {type(key).__name__} is not a valid index")
            pos = operator.index(key)
            start = pos + size if pos < 0 else pos
            if not 0 <= start < size:
                raise IndexError(f"index {index}: {pos} is out of range for dim {dim} of size {size}")
            bounds.append((start, start + 1))
        return self.shrink(bounds).reshape(*shape)

    def __iter__(self) -> Iterator[Tensor]:
        """Rows along dim 0, so `for row in t` and `q, k, v = qkv` mean what they do in numpy."""
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        return (self[i] for i in range(self.shape[0]))

    def gather_rows(self, indices: Tensor) -> Tensor:
        """Rows of this 2D table picked by int32 indices: (V, D) read at (...) gives (..., D).

        Indices are trusted: limn does not bounds-check them, so an out-of-range one reads whatever
        the buffer holds there. The gradient scatters back, summing the rows an index repeats.
        """
        if self.ndim != 2:
            raise ValueError(f"gather_rows needs a 2D table, got {self.shape}")
        if indices.dtype != int32:
            raise ValueError(f"gather_rows needs int32 indices, got {indices.dtype}")
        node = Node(Op.GATHER, (self.node, indices.node), self.dtype, indices.shape + self.shape[1:])
        table_shape = self.shape

        def backward(g: Tensor) -> tuple[Tensor]:
            return (scatter_rows(g, indices, table_shape),)

        return Tensor.from_node(node, (self,), backward)

    # ---- composed ops ----

    def matmul(self, other: Tensor) -> Tensor:
        a, b = self, other
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError(f"matmul needs 2D+ tensors, got {a.shape} @ {b.shape}")
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(f"matmul: inner dims differ, {a.shape} @ {b.shape}")
        batch = broadcast_shape(a.shape[:-2], b.shape[:-2], "matmul")
        m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
        a = broadcast_to(a, batch + a.shape[-2:], "matmul").reshape(*batch, m, 1, k).expand(*batch, m, n, k)
        b = broadcast_to(b, batch + b.shape[-2:], "matmul").transpose(-2, -1).reshape(*batch, 1, n, k).expand(*batch, m, n, k)
        return (a * b).sum(axis=-1)

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    def softmax(self, axis: int = -1) -> Tensor:
        shifted = (self - self.max(axis, keepdim=True).detach()).exp()
        return shifted / shifted.sum(axis, keepdim=True)

    def log_softmax(self, axis: int = -1) -> Tensor:
        shifted = self - self.max(axis, keepdim=True).detach()
        return shifted - shifted.exp().sum(axis, keepdim=True).log()

    # ---- autograd ----

    def backward(self) -> None:
        if self.numel != 1:
            raise ValueError(f"backward() needs a scalar, got shape {self.shape}")
        if self.dtype not in FLOATS or not self.requires_grad:
            raise ValueError("backward() needs a float tensor with requires_grad=True")
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            t, parents_done = stack.pop()
            if parents_done:
                order.append(t)
            elif id(t) not in visited:
                visited.add(id(t))
                stack.append((t, True))
                stack.extend((p, False) for p in reversed(t.parents) if p.requires_grad)
        grads: dict[int, Tensor] = {id(self): Tensor.const(1.0, self.dtype).reshape(*self.shape)}
        with no_grad():
            for t in reversed(order):
                g = grads.pop(id(t), None)
                if g is None:
                    continue
                if t.grad_fn is None:  # a leaf: accumulate into .grad
                    t.grad = g if t.grad is None else t.grad + g
                    continue
                for parent, pg in zip(t.parents, t.grad_fn(g), strict=True):
                    if pg is None or not parent.requires_grad:
                        continue
                    prior = grads.get(id(parent))
                    grads[id(parent)] = pg if prior is None else prior + pg


# ---- module-level helpers ----


def realize(*tensors: Tensor) -> list[device.Buffer]:
    """Execute the graphs of all given tensors in one batch (shared subgraphs compute once)."""
    sinks = [t.node for t in tensors]
    buffers = device.active().execute(sinks)
    # every reachable ASSIGN has committed, so each one becomes the buffer it just wrote. Retiring
    # the node rather than the tensor covers holders that were not passed here; running it a second
    # time would apply the write again.
    for node in topological(sinks):
        if node.op is Op.ASSIGN:
            node.op, node.srcs, node.arg = Op.BUFFER, (), node.srcs[0].arg
    return buffers


def scatter_rows(values: Tensor, indices: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Sum `values`' rows into a zeroed table of `shape` at `indices`: the gradient of gather_rows.

    Module-level rather than a Tensor method because the result's shape comes from an argument
    instead of a receiver. A public scatter-add wants the opposite (rows added into a table that
    already exists), and composes as `table + scatter_rows(...)` rather than exposing this.

    It carries no autograd record, like every gradient limn builds: backward() runs under no_grad,
    so there is no grad-of-grad here or anywhere else. Turning that on would give this parents of
    (values,) and a gradient of gather_rows(g, indices).
    """
    if values.shape != indices.shape + shape[1:]:
        raise ValueError(f"scatter_rows: values {values.shape} do not match indices {indices.shape} into {shape}")
    return Tensor.from_node(Node(Op.SCATTER, (indices.node, values.node), values.dtype, shape))


def as_ints(shape: Sequence[int] | tuple) -> tuple[int, ...]:
    """Accept both f(2, 3) and f((2, 3)) call styles."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(int(s) for s in shape[0])
    return tuple(int(s) for s in shape)


def resolve_shape(shape: tuple[int, ...], numel: int) -> tuple[int, ...]:
    if shape.count(-1) > 1:
        raise ValueError(f"reshape: more than one -1 in {shape}")
    if -1 in shape:
        rest = math.prod(s for s in shape if s != -1)
        shape = tuple(numel // rest if s == -1 else s for s in shape)
    if any(s < 1 for s in shape):
        raise ValueError(f"reshape: invalid shape {shape}")
    return shape


def wrap_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} out of range for {ndim} dims")
    return axis % ndim


def canon_axes(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    wrapped = tuple(sorted(wrap_axis(a, ndim) for a in axes))
    if len(set(wrapped)) != len(wrapped):
        raise ValueError(f"duplicate axes in {axes}")
    return wrapped


def as_tensor(x: Tensor | float | int, like: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x
    # a python scalar takes the tensor's dtype rather than widening it; only a float scalar
    # against an int tensor brings its own, and joins the floats where the int cannot follow
    dtype = like.dtype if like.dtype in FLOATS or isinstance(x, int) else float32
    return Tensor.const(x, dtype)


def zero(like: Tensor) -> Tensor:
    return Tensor.const(0, like.dtype)


def broadcast_shape(s1: tuple[int, ...], s2: tuple[int, ...], opname: str) -> tuple[int, ...]:
    ndim = max(len(s1), len(s2))
    a = (1,) * (ndim - len(s1)) + s1
    b = (1,) * (ndim - len(s2)) + s2
    for d1, d2 in zip(a, b):
        if d1 != d2 and 1 not in (d1, d2):
            raise ValueError(f"{opname}: cannot broadcast {s1} with {s2}")
    return tuple(max(d1, d2) for d1, d2 in zip(a, b))


def broadcast_to(t: Tensor, shape: tuple[int, ...], opname: str) -> Tensor:
    if t.shape == shape:
        return t
    if broadcast_shape(t.shape, shape, opname) != shape:
        raise ValueError(f"{opname}: cannot broadcast {t.shape} to {shape}")
    lifted = t.reshape(*((1,) * (len(shape) - t.ndim) + t.shape)) if t.ndim != len(shape) else t
    return lifted.expand(*shape)


def broadcast_pair(a: Tensor, b: Tensor | float | int, opname: str) -> tuple[Tensor, Tensor]:
    b = as_tensor(b, a)
    if a.dtype != b.dtype:
        wider = promote(a.dtype, b.dtype)
        a, b = a.cast(wider), b.cast(wider)
    shape = broadcast_shape(a.shape, b.shape, opname)
    return broadcast_to(a, shape, opname), broadcast_to(b, shape, opname)
