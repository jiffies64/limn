"""Train a small MLP on synthetic data with AdamW. The loss must visibly decrease."""

import numpy as np

from limn import Tensor, set_seed
from limn.nn import Linear, parameters
from limn.ops import Node, Op, topological
from limn.optim import AdamW
from limn.schedule import boundaries, realized, schedule


class MLP:
    def __init__(self):
        self.fc1 = Linear(2, 32)
        self.fc2 = Linear(32, 32)
        self.fc3 = Linear(32, 1)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc3(self.fc2(self.fc1(x).relu()).relu())


CUT_REASON = {
    Op.SUM: "reduce: stores what it accumulated",
    Op.MAX: "reduce: stores what it accumulated",
    Op.CONTIGUOUS: "copy: lays out contiguous bytes",
    Op.ASSIGN: "in-place write to a buffer",
    Op.GATHER: "indexed table read",
    Op.SCATTER: "indexed scatter write",
    Op.CUSTOM: "custom kernel: the device supplies it whole",
}


def cut_reasons(sinks: list[Node]) -> dict[Node, str]:
    """Why each cut node needs its own buffer: boundaries() decides, this only names the rule."""
    homes = {realized(sink) for sink in sinks}
    return {
        node: CUT_REASON.get(node.op)
        or ("requested as an output" if node in homes else "a view or gather indexes its layout, so the bytes must exist")
        for node in boundaries(sinks, topological(sinks))
    }


def report_fusion(title: str, *tensors: Tensor) -> None:
    """Print each kernel: why it is its own kernel, what fused inside it, and what it reads from memory."""
    sinks = [t.node for t in tensors]
    kernels = schedule(sinks)
    reasons = cut_reasons(sinks)
    producer = {k.ast: i for i, k in enumerate(kernels)}
    print(f"=== {title}: {len(kernels)} kernels ===")
    for i, k in enumerate(kernels):
        root = k.ast
        fused = [n.op.name for n in k.body if n is not root and n.op is not Op.CONST]
        print(f"\nk{i}  {root.op.name}{list(root.shape)}  <- its own kernel: {reasons[root]}")
        print(
            f"    fused inside ({len(fused)} ops ride free): {' '.join(fused) if fused else 'nothing, just the ' + root.op.name}"
        )
        reads = [f"{_describe(inp, producer)} {inp.dtype}{list(inp.shape)}" for inp in k.inputs]
        print(f"    reads from memory ({len(reads)}): {', '.join(reads)}")
    print()


def _describe(node: Node, producer: dict[Node, int]) -> str:
    if node.op is Op.BUFFER:
        return "input"
    if (j := producer.get(node)) is not None:
        return f"k{j}->"
    return "buf"


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-1, 1, (256, 2)).astype(np.float32)
    labels = np.sin(3 * inputs[:, :1]) * np.cos(2 * inputs[:, 1:])  # a smooth 2D surface to regress

    model = MLP()
    params = parameters(model)
    optimizer = AdamW(params, lr=1e-2)
    x, y = Tensor(inputs), Tensor(labels)

    first_loss = loss_value = 0.0
    for step in range(300):
        optimizer.zero_grad()
        error = model(x) - y
        loss = (error * error).mean()
        loss_value = float(loss.item())
        loss.backward()
        if step == 0:
            report_fusion("forward: loss", loss)
            report_fusion("backward: gradients", *[p.grad for p in params if p.grad is not None])
        optimizer.step()
        if step == 0:
            first_loss = loss_value
        if step % 50 == 0 or step == 299:
            print(f"step {step:3d}  loss {loss_value:.5f}")

    assert loss_value < 0.05 * first_loss, "loss did not decrease"
    print(f"loss went {first_loss:.5f} -> {loss_value:.5f}")


if __name__ == "__main__":
    main()
