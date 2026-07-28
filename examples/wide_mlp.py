"""Head-to-head: Muon vs AdamW memorizing random data with a wide MLP.

Runs on the c device by default (pass a device name to override: numpy, c, cuda).
The numpy device materializes matmul's expanded operands, and Newton-Schulz at
512x512 makes that gigabytes per step; don't run this example there.
"""

import sys

import numpy as np
from limn import Tensor, set_device, set_seed
from limn.nn import Linear, parameters
from limn.optim import AdamW, Muon


class WideMLP:
    def __init__(self):
        self.fc1 = Linear(128, 512)
        self.fc2 = Linear(512, 512)
        self.fc3 = Linear(512, 128)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc3(self.fc2(self.fc1(x).relu()).relu())


def split(params):
    """Muon takes the matrices, AdamW takes biases and anything else 1D."""
    return (
        [p for p in params if len(p.shape) == 2],
        [p for p in params if len(p.shape) != 2],
    )


def train(make_optimizers, x, y, steps=200):
    set_seed(0)
    model = WideMLP()
    opts = make_optimizers(parameters(model))
    losses = []
    for _ in range(steps):
        for o in opts:
            o.zero_grad()
        error = model(x) - y
        loss = (error * error).mean()
        losses.append(float(loss.item()))
        loss.backward()
        for o in opts:
            o.step()
    return losses


def main() -> None:
    set_device(sys.argv[1] if len(sys.argv) > 1 else "c")
    rng = np.random.default_rng(0)
    # random inputs, random targets: nothing to learn but the mapping itself
    x = Tensor(rng.normal(size=(512, 128)).astype(np.float32))
    y = Tensor(rng.normal(size=(512, 128)).astype(np.float32))

    adam_only = train(lambda ps: [AdamW(ps, lr=3e-3)], x, y)

    def muon_hybrid(ps):
        mats, rest = split(ps)
        return [Muon(mats, lr=0.02), AdamW(rest, lr=3e-3)]

    hybrid = train(muon_hybrid, x, y)

    for step in (0, 25, 50, 100, 199):
        print(f"step {step:3d}   adamw {adam_only[step]:.5f}   muon {hybrid[step]:.5f}")


if __name__ == "__main__":
    main()
