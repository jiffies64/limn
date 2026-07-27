"""Shared test helpers: the graph corpus every backend and the lowered IR are diffed against."""

import numpy as np

rng = np.random.default_rng(7)


def randf(*shape: int) -> np.ndarray:
    return rng.uniform(-2, 2, shape).astype(np.float32)


GRAPHS = {
    "elementwise": lambda a, b: (a + b) * 2.0 - a,
    "relu": lambda a, b: (a * b).relu() + 1.0,
    "unary math": lambda a, b: (a * a + 1.0).log().exp().sqrt(),
    "divide": lambda a, b: a / (b * b + 0.5),
    "where": lambda a, b: (a < b).where(a, b),
    "cast round trip": lambda a, b: (a * 4.0).int().float() + 0.5,
    "sum an axis": lambda a, b: (a + b).sum(axis=1),
    "sum everything": lambda a, b: (a * b).sum(),
    "max keepdim": lambda a, b: (a * b).max(axis=0, keepdim=True),
    "mean": lambda a, b: (a + b).mean(axis=1),
    "matmul": lambda a, b: a @ b.transpose(),
    "softmax": lambda a, b: (a + b).softmax(axis=1),
    "log softmax": lambda a, b: (a * b).log_softmax(axis=0),
    "padded reduce": lambda a, b: a.pad(((1, 1), (0, 2))).sum(axis=1),
    "movement then reduce": lambda a, b: (a.transpose() * 2.0).sum(axis=0),
    "reduce then elementwise": lambda a, b: (a.sum(axis=1, keepdim=True) * b).relu(),
}
