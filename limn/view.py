"""View: how a tensor's logical indices map onto a flat buffer.

A View is (shape, strides, offset, mask). Logical index (i0, i1, ...) reads flat position
offset + sum(i_d * stride_d). A stride of 0 repeats data along that dim (expand); the mask
marks the valid [lo, hi) range per dim, and reads outside it produce 0 (pad). Movement ops
are View -> View functions, so chained movements collapse into a single View and cost
nothing at runtime. reshape returns None when the new shape can't be expressed over the
same buffer without a copy; the caller (tensor.py) inserts a CONTIGUOUS node instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def canonical_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major strides, with size-1 dims given stride 0 so layouts compare cleanly."""
    strides: list[int] = []
    acc = 1
    for size in reversed(shape):
        strides.append(0 if size == 1 else acc)
        acc *= size
    return tuple(reversed(strides))


def normalize_mask(shape: tuple[int, ...], mask: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...] | None:
    return None if all(m == (0, size) for m, size in zip(mask, shape)) else mask


@dataclass(frozen=True)
class View:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int
    mask: tuple[tuple[int, int], ...] | None  # per-dim valid range [lo, hi); None means fully valid

    def __post_init__(self) -> None:
        """A size-1 dim is only ever indexed at 0, so its stride is dead weight; zero it here and
        every layout compares against canonical_strides on the same footing."""
        strides = tuple(0 if size == 1 else stride for size, stride in zip(self.shape, self.strides))
        if strides != self.strides:
            object.__setattr__(self, "strides", strides)

    @staticmethod
    def contiguous(shape: tuple[int, ...]) -> View:
        return View(tuple(shape), canonical_strides(shape), 0, None)

    @property
    def is_dense(self) -> bool:
        """Reads its whole extent row-major, possibly starting partway in: reshapeable in place."""
        return self.mask is None and self.strides == canonical_strides(self.shape)

    @property
    def is_contiguous(self) -> bool:
        """Dense *and* starting at 0, so it covers a source buffer rather than a slab of one."""
        return self.offset == 0 and self.is_dense

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    def permute(self, order: tuple[int, ...]) -> View:
        if sorted(order) != list(range(len(self.shape))):
            raise ValueError(f"permute order {order} is not a permutation of {len(self.shape)} dims")
        mask = tuple(self.mask[d] for d in order) if self.mask is not None else None
        return View(tuple(self.shape[d] for d in order), tuple(self.strides[d] for d in order), self.offset, mask)

    def expand(self, shape: tuple[int, ...]) -> View:
        if len(shape) != len(self.shape) or any(old not in (1, new) for old, new in zip(self.shape, shape)):
            raise ValueError(f"cannot expand {self.shape} to {shape}: only size-1 dims may grow")
        strides = tuple(s if old == new else 0 for s, old, new in zip(self.strides, self.shape, shape))
        mask = None
        if self.mask is not None:
            # a size-1 dim holds one element: if it was valid, every repeat is valid, else none are
            mask = tuple(
                m if old == new else ((0, new) if m == (0, 1) else (0, 0)) for m, old, new in zip(self.mask, self.shape, shape)
            )
        return View(tuple(shape), strides, self.offset, mask)

    def shrink(self, bounds: tuple[tuple[int, int], ...]) -> View:
        ok = len(bounds) == len(self.shape) and all(0 <= lo < hi <= size for (lo, hi), size in zip(bounds, self.shape))
        if not ok:
            raise ValueError(f"cannot shrink shape {self.shape} to bounds {bounds}")
        offset = self.offset + sum(lo * stride for (lo, _), stride in zip(bounds, self.strides))
        shape = tuple(hi - lo for lo, hi in bounds)
        mask = None
        if self.mask is not None:
            mask = normalize_mask(
                shape, tuple((max(mlo - lo, 0), min(mhi - lo, hi - lo)) for (lo, hi), (mlo, mhi) in zip(bounds, self.mask))
            )
        return View(shape, self.strides, offset, mask)

    def pad(self, padding: tuple[tuple[int, int], ...]) -> View:
        if len(padding) != len(self.shape) or any(before < 0 or after < 0 for before, after in padding):
            raise ValueError(f"cannot pad shape {self.shape} with {padding}")
        offset = self.offset - sum(before * stride for (before, _), stride in zip(padding, self.strides))
        shape = tuple(size + before + after for size, (before, after) in zip(self.shape, padding))
        old_mask = self.mask if self.mask is not None else tuple((0, size) for size in self.shape)
        mask = tuple((mlo + before, mhi + before) for (mlo, mhi), (before, _) in zip(old_mask, padding))
        return View(shape, self.strides, offset, normalize_mask(shape, mask))

    def reshape(self, shape: tuple[int, ...]) -> View | None:
        """The reshaped View, or None if this layout can't be reshaped without a copy."""
        if math.prod(shape) != self.numel:
            raise ValueError(f"cannot reshape {self.shape} to {shape}: element counts differ")
        if shape == self.shape:
            return self
        if self.is_dense:
            return View(tuple(shape), canonical_strides(shape), self.offset, None)
        # inserting or removing size-1 dims never moves data; anything else on this layout needs a copy
        old_mask = self.mask if self.mask is not None else tuple((0, size) for size in self.shape)
        kept = [(stride, m) for size, stride, m in zip(self.shape, self.strides, old_mask) if size != 1]
        kept_sizes = [size for size in self.shape if size != 1]
        dropped_ok = all(m == (0, 1) for size, m in zip(self.shape, old_mask) if size == 1)
        if kept_sizes != [size for size in shape if size != 1] or not dropped_ok:
            return None
        strides: list[int] = []
        mask: list[tuple[int, int]] = []
        it = iter(kept)
        for size in shape:
            new_stride, new_m = (0, (0, 1)) if size == 1 else next(it)
            strides.append(new_stride)
            mask.append(new_m)
        return View(tuple(shape), tuple(strides), self.offset, normalize_mask(tuple(shape), tuple(mask)))

    def materialize(self, flat: np.ndarray) -> np.ndarray:
        """Gather this view's elements out of a flat buffer. Reference implementation for all backends."""
        index = np.full(self.shape, self.offset, dtype=np.int64)
        valid = np.ones(self.shape, dtype=bool)
        for d, (size, stride) in enumerate(zip(self.shape, self.strides)):
            axis = np.arange(size).reshape((1,) * d + (size,) + (1,) * (len(self.shape) - d - 1))
            index += axis * stride
            if self.mask is not None:
                lo, hi = self.mask[d]
                valid &= (axis >= lo) & (axis < hi)
        if self.mask is None:  # every index is in bounds, so gather straight out of the buffer
            return flat[index]
        # masked-out positions may index out of bounds (pad reads past the buffer): clip, then zero them
        gathered = flat[np.clip(index, 0, flat.size - 1)]
        return np.where(valid, gathered, np.zeros((), dtype=flat.dtype))
