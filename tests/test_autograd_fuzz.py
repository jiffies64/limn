"""Seeded fuzzer: random expression DAGs run forward and backward in limn and torch.

Each case grows a pool of mirrored (limn, torch) value pairs from a few leaf tensors, applies
random unary / binary / movement / reduce / matmul ops, sums the last value to a scalar loss,
and backpropagates in both frameworks. Values and every leaf gradient must agree to 1e-4.
On failure the printed reproducer has the seed, the expression string, and all leaf shapes.

Inputs are kept in safe ranges by construction: log / sqrt / reciprocal / division are always
applied to (x*x + 0.5), and exp to (x * 0.5), mirrored identically in torch.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from limn import Tensor

BASE_SEED = 1337
CASES = 300
ATOL = RTOL = 1e-4

SHAPES = [(3,), (4, 3), (2, 3, 4), (2, 1, 3), (5, 1), (1, 4), (2, 2, 2, 3), (6,), (3, 5)]


@dataclass
class Pair:
    """One mirrored value: the same expression held by both frameworks, plus its history."""

    limn: Tensor
    torch: torch.Tensor
    desc: str


def make_leaf(rng: np.random.Generator, shape: tuple[int, ...], name: str) -> Pair:
    data = rng.uniform(-2, 2, shape).astype(np.float32)
    lt = Tensor(data.copy(), requires_grad=True)
    tt = torch.tensor(data.copy(), requires_grad=True)
    return Pair(lt, tt, f"{name}{list(shape)}")


UNARY = [
    ("neg", lambda lx: -lx, lambda tx: -tx),
    ("square", lambda lx: lx * lx, lambda tx: tx * tx),
    ("exp", lambda lx: (lx * 0.5).exp(), lambda tx: (tx * 0.5).exp()),
    ("log", lambda lx: (lx * lx + 0.5).log(), lambda tx: (tx * tx + 0.5).log()),
    ("sqrt", lambda lx: (lx * lx + 0.5).sqrt(), lambda tx: (tx * tx + 0.5).sqrt()),
    ("recip", lambda lx: (lx * lx + 0.5).reciprocal(), lambda tx: 1 / (tx * tx + 0.5)),
    ("relu", lambda lx: lx.relu(), lambda tx: tx.relu()),
    ("softmax", lambda lx: lx.softmax(-1), lambda tx: tx.softmax(-1)),
]

BINARY = [
    ("add", lambda la, lb: la + lb, lambda ta, tb: ta + tb),
    ("sub", lambda la, lb: la - lb, lambda ta, tb: ta - tb),
    ("mul", lambda la, lb: la * lb, lambda ta, tb: ta * tb),
    ("div", lambda la, lb: la / (lb * lb + 0.5), lambda ta, tb: ta / (tb * tb + 0.5)),
    ("wheregt", lambda la, lb: (la > lb).where(la, lb * 0.5), lambda ta, tb: torch.where(ta > tb, ta, tb * 0.5)),
]


def apply_movement(rng: np.random.Generator, pair: Pair) -> Pair:
    lv, tv = pair.limn, pair.torch
    kind = rng.choice(["reshape", "permute", "pad", "shrink", "transpose"])
    if kind == "reshape":
        flat = int(np.prod(lv.shape))
        divisors = [d for d in range(1, flat + 1) if flat % d == 0]
        first = int(rng.choice(divisors))
        shape: tuple[int, ...] = (first, flat // first)
        return Pair(lv.reshape(*shape), tv.reshape(*shape), f"reshape{list(shape)}({pair.desc})")
    if kind == "permute" and lv.ndim >= 2:
        order = tuple(rng.permutation(lv.ndim).tolist())
        return Pair(lv.permute(*order), tv.permute(*order), f"permute{list(order)}({pair.desc})")
    if kind == "transpose" and lv.ndim >= 2:
        return Pair(lv.transpose(), tv.transpose(-2, -1), f"transpose({pair.desc})")
    if kind == "pad":
        pads = tuple((int(rng.integers(0, 2)), int(rng.integers(0, 2))) for _ in lv.shape)
        torch_pads = [p for pair_ in reversed(pads) for p in pair_]  # F.pad orders last dim first
        return Pair(lv.pad(pads), F.pad(tv, torch_pads), f"pad{list(pads)}({pair.desc})")
    if kind == "shrink" and all(s >= 2 for s in lv.shape):
        bounds = tuple((0, int(rng.integers(1, s))) if rng.random() < 0.5 else (int(rng.integers(0, s - 1)), s) for s in lv.shape)
        slices = tuple(slice(lo, hi) for lo, hi in bounds)
        return Pair(lv.shrink(bounds), tv[slices], f"shrink{list(bounds)}({pair.desc})")
    return pair


def apply_reduce(rng: np.random.Generator, pair: Pair) -> Pair:
    lv, tv = pair.limn, pair.torch
    axis = int(rng.integers(0, lv.ndim)) if lv.ndim else 0
    keepdim = bool(rng.random() < 0.5)
    if lv.ndim == 0:
        return pair
    kind = rng.choice(["sum", "max", "mean"])
    if kind == "sum":
        return Pair(lv.sum(axis, keepdim=keepdim), tv.sum(axis, keepdim=keepdim), f"sum[{axis}]({pair.desc})")
    if kind == "max":
        return Pair(lv.max(axis, keepdim=keepdim), tv.amax(axis, keepdim=keepdim), f"max[{axis}]({pair.desc})")
    return Pair(lv.mean(axis, keepdim=keepdim), tv.mean(axis, keepdim=keepdim), f"mean[{axis}]({pair.desc})")


def run_case(seed: int) -> None:
    rng = np.random.default_rng(seed)
    leaves = [make_leaf(rng, SHAPES[int(rng.integers(0, len(SHAPES)))], f"x{i}") for i in range(int(rng.integers(2, 5)))]
    pool = list(leaves)
    for _ in range(int(rng.integers(3, 9))):
        roll = rng.random()
        pair = pool[int(rng.integers(0, len(pool)))]
        if roll < 0.30:
            name, lf, tf = UNARY[int(rng.integers(0, len(UNARY)))]
            pool.append(Pair(lf(pair.limn), tf(pair.torch), f"{name}({pair.desc})"))
        elif roll < 0.60:
            other = pool[int(rng.integers(0, len(pool)))]
            try:
                np.broadcast_shapes(pair.limn.shape, other.limn.shape)
            except ValueError:
                other = pair
            name, lf, tf = BINARY[int(rng.integers(0, len(BINARY)))]
            pool.append(Pair(lf(pair.limn, other.limn), tf(pair.torch, other.torch), f"{name}({pair.desc}, {other.desc})"))
        elif roll < 0.80:
            pool.append(apply_movement(rng, pair))
        elif roll < 0.93 or pair.limn.ndim < 2:
            pool.append(apply_reduce(rng, pair))
        else:
            n = int(rng.integers(1, 5))
            rhs = make_leaf(rng, pair.limn.shape[:-2] + (pair.limn.shape[-1], n), f"w{len(leaves)}")
            leaves.append(rhs)
            pool.append(Pair(pair.limn @ rhs.limn, pair.torch @ rhs.torch, f"matmul({pair.desc}, {rhs.desc})"))

    final = pool[-1]
    loss_l, loss_t = final.limn.sum(), final.torch.sum()
    context = f"\nseed={seed}\nexpression: sum({final.desc})\nleaves: {[(leaf.desc, leaf.limn.shape) for leaf in leaves]}"
    np.testing.assert_allclose(
        loss_l.numpy(), loss_t.detach().numpy(), atol=ATOL, rtol=RTOL, err_msg="forward value mismatch" + context
    )
    loss_l.backward()
    loss_t.backward()
    for leaf in leaves:
        lg, tg = leaf.limn.grad, leaf.torch.grad
        assert (lg is None) == (tg is None), f"grad presence differs for {leaf.desc}" + context
        if lg is not None and tg is not None:
            np.testing.assert_allclose(
                lg.numpy(), tg.numpy(), atol=ATOL, rtol=RTOL, err_msg=f"gradient mismatch for leaf {leaf.desc}" + context
            )


@pytest.mark.parametrize("chunk", range(10))
def test_fuzz_autograd(chunk: int):
    for i in range(CASES // 10):
        run_case(BASE_SEED + chunk * (CASES // 10) + i)
