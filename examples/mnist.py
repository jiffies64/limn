"""The convnet from tinygrad's beautiful_mnist example, trained on MNIST with limn.

Runs on the c device by default (pass a device name to override: numpy, c, cuda).
Stride-2 convolutions stand in for max pooling, which limn does not have; the spatial
sizes in the comments track a 28x28 input down to the 64*3*3 features the head reads.
"""

import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np

from limn import Tensor, set_device, set_seed
from limn.nn import BatchNorm, Conv2d, Linear, parameters
from limn.optim import AdamW

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def load_mnist(cache: Path = Path(__file__).parent / "data"):
    cache.mkdir(exist_ok=True)
    arrays = []
    for name in FILES:
        path = cache / name
        if not path.exists():
            urllib.request.urlretrieve(MNIST_URL + name, path)
        offset = 16 if "images" in name else 8  # idx header: magic + dims
        arrays.append(np.frombuffer(gzip.open(path).read(), dtype=np.uint8, offset=offset))
    images = [a.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0 for a in arrays[::2]]
    labels = [a.astype(np.int32) for a in arrays[1::2]]
    return images[0], labels[0], images[1], labels[1]


class BeautifulMnist:
    def __init__(self):
        self.conv1 = Conv2d(1, 32, 5)  # 28 -> 24
        self.conv2 = Conv2d(32, 32, 5, stride=2)  # 24 -> 10
        self.norm1 = BatchNorm(32)
        self.conv3 = Conv2d(32, 64, 3)  # 10 -> 8
        self.conv4 = Conv2d(64, 64, 3, stride=2)  # 8 -> 3
        self.norm2 = BatchNorm(64)
        self.head = Linear(64 * 3 * 3, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.norm1(self.conv2(self.conv1(x).relu()).relu())
        x = self.norm2(self.conv4(self.conv3(x).relu()).relu())
        return self.head(x.reshape(x.shape[0], -1))

    def set_training(self, training: bool) -> None:
        self.norm1.training = self.norm2.training = training


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy of (N, V) float logits against (N,) int32 class indices."""
    log_probs = logits.log_softmax(-1)
    onehot = targets.reshape(-1, 1).eq(Tensor.arange(logits.shape[-1])).float()
    return -(onehot * log_probs).sum() / logits.shape[0]


def accuracy(model: BeautifulMnist, images: np.ndarray, labels: np.ndarray, batch: int = 500) -> float:
    model.set_training(False)
    correct = 0
    for i in range(0, len(images), batch):
        predictions = model(Tensor(images[i : i + batch])).numpy().argmax(axis=1)
        correct += int((predictions == labels[i : i + batch]).sum())
    model.set_training(True)
    return 100.0 * correct / len(images)


def main() -> None:
    set_device(sys.argv[1] if len(sys.argv) > 1 else "c")
    set_seed(0)
    x_train, y_train, x_test, y_test = load_mnist()

    model = BeautifulMnist()
    opt = AdamW(parameters(model), lr=1e-3)
    rng = np.random.default_rng(0)

    for step in range(70):
        opt.zero_grad()
        batch = rng.integers(0, len(x_train), size=512)
        loss = cross_entropy(model(Tensor(x_train[batch])), Tensor(y_train[batch]))
        loss.backward()
        opt.step()
        if step % 10 == 9:
            # a 2000-image slice keeps the running eval cheap; the full test set is scored once at the end
            test_acc = accuracy(model, x_test[:2000], y_test[:2000])
            print(f"step {step + 1:3d}   loss {loss.item():.4f}   test accuracy {test_acc:.2f}%")

    print(f"final accuracy on the full test set: {accuracy(model, x_test, y_test):.2f}%")


if __name__ == "__main__":
    main()
