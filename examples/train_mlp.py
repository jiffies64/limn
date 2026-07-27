"""Train a small MLP on synthetic data with AdamW. The loss must visibly decrease."""

import numpy as np

from limn import Tensor, set_seed
from limn.nn import Linear, parameters
from limn.optim import AdamW


class MLP:
    def __init__(self):
        self.fc1 = Linear(2, 32)
        self.fc2 = Linear(32, 32)
        self.fc3 = Linear(32, 1)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc3(self.fc2(self.fc1(x).relu()).relu())


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-1, 1, (256, 2)).astype(np.float32)
    labels = np.sin(3 * inputs[:, :1]) * np.cos(2 * inputs[:, 1:])  # a smooth 2D surface to regress

    model = MLP()
    optimizer = AdamW(parameters(model), lr=1e-2)
    x, y = Tensor(inputs), Tensor(labels)

    first_loss = loss_value = 0.0
    for step in range(300):
        optimizer.zero_grad()
        error = model(x) - y
        loss = (error * error).mean()
        loss_value = float(loss.item())
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = loss_value
        if step % 50 == 0 or step == 299:
            print(f"step {step:3d}  loss {loss_value:.5f}")

    assert loss_value < 0.05 * first_loss, "loss did not decrease"
    print(f"loss went {first_loss:.5f} -> {loss_value:.5f}")


if __name__ == "__main__":
    main()
