"""Train a byte-level GPT on TinyStories and make it tell one.

The default run is small (50M tokens): enough for word-correct baby English. --full trains
500M tokens overnight. Both log loss and throughput, checkpoint as they go, resume with
--resume, and finish by sampling a story. The dataset (about 2 GB of text) downloads on
first use into examples/data/.
"""

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np

from limn import Tensor, capture, no_grad, realize, set_device, set_seed
from limn.nn import Embedding, LayerNorm, Linear, parameters
from limn.optim import AdamW

DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
DATA_DIR = Path(__file__).parent / "data"

VOCAB, CTX, DIM, LAYERS, HEADS = 256, 256, 192, 6, 6


class Block:
    def __init__(self):
        self.ln1, self.ln2 = LayerNorm(DIM), LayerNorm(DIM)
        self.qkv = Linear(DIM, 3 * DIM)
        self.proj = Linear(DIM, DIM)
        self.up, self.down = Linear(DIM, 4 * DIM), Linear(4 * DIM, DIM)

    def __call__(self, x: Tensor, mask: Tensor) -> Tensor:
        b, t, c = x.shape
        hd = c // HEADS
        qkv = self.qkv(self.ln1(x)).reshape(b, t, 3, HEADS, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        scores = (q @ k.transpose(-2, -1)) * (1.0 / hd**0.5)
        att = mask.where(scores, -1e9).softmax(-1)
        x = x + self.proj((att @ v).permute(0, 2, 1, 3).reshape(b, t, c))
        return x + self.down(self.up(self.ln2(x)).relu())


class GPT:
    def __init__(self):
        self.tok = Embedding(VOCAB, DIM)
        self.pos = Embedding(CTX, DIM)
        self.blocks = [Block() for _ in range(LAYERS)]
        self.ln = LayerNorm(DIM)
        self.head = Linear(DIM, VOCAB, bias=False)

    def __call__(self, tokens: Tensor, mask: Tensor) -> Tensor:
        x = self.tok(tokens) + self.pos(Tensor.arange(CTX))
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln(x))


def causal_mask() -> Tensor:
    rows = Tensor.arange(CTX).reshape(CTX, 1)
    cols = Tensor.arange(CTX).reshape(1, CTX)
    return (cols <= rows).reshape(1, 1, CTX, CTX)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    logp = logits.log_softmax(-1)
    onehot = Tensor.arange(VOCAB).reshape(1, 1, VOCAB).eq(targets.reshape(*targets.shape, 1))
    return -(logp * onehot).sum() / float(targets.numel)


def load_data() -> np.ndarray:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "tinystories.txt"
    if not path.exists():
        print(f"downloading TinyStories (about 2 GB) to {path} ...", flush=True)
        urllib.request.urlretrieve(DATA_URL, path)
    # mapped, not read: a batch touches 32 scattered rows of 257 bytes, so the pages it needs are
    # evictable file cache instead of 2 GB of resident memory
    return np.memmap(path, dtype=np.uint8, mode="r")


def batch_of(data: np.ndarray, batch: int, rng: np.random.Generator) -> tuple[Tensor, Tensor]:
    offsets = rng.integers(0, len(data) - CTX - 1, size=batch)
    chunk = data[offsets[:, None] + np.arange(CTX + 1)[None, :]].astype(np.int32)
    return Tensor(chunk[:, :-1]), Tensor(chunk[:, 1:])


def save_checkpoint(path: Path, params: list[Tensor]) -> None:
    np.savez(path, *[p.numpy() for p in params])  # arr_0..arr_N, in parameters() order


def load_checkpoint(path: Path, params: list[Tensor]) -> None:
    stored = np.load(path)
    assert len(stored.files) == len(params), f"checkpoint has {len(stored.files)} tensors, model has {len(params)}"
    realize(*[p.assign(Tensor(stored[f"arr_{i}"])) for i, p in enumerate(params)])


def sample(model: GPT, mask: Tensor, n: int, temperature: float, rng: np.random.Generator) -> str:
    window = np.full(CTX, ord("\n"), dtype=np.int32)
    prompt = b"Once upon a time"
    window[-len(prompt) :] = np.frombuffer(prompt, dtype=np.uint8)
    out = bytearray(prompt)
    with no_grad():
        for _ in range(n):
            # only the last position predicts, so drop the rest on the device rather than copy it back
            logits = model(Tensor(window.reshape(1, CTX)), mask)[0, -1].numpy()
            weights = np.exp((logits - logits.max()) / temperature)
            token = int(rng.choice(VOCAB, p=weights / weights.sum()))
            out.append(token)
            window[:-1] = window[1:]
            window[-1] = token
    return out.decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="train 500M tokens instead of the default 50M")
    parser.add_argument("--tokens", type=int, default=None, help="override the token budget")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="continue from the checkpoint")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--sample-bytes", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    set_device(args.device)
    set_seed(0)
    rng = np.random.default_rng(0)
    data = load_data()

    model = GPT()
    params = parameters(model)
    checkpoint = DATA_DIR / "stories_checkpoint.npz"
    if args.resume and checkpoint.exists():
        load_checkpoint(checkpoint, params)
        print(f"resumed from {checkpoint}")
    opt = AdamW(params, lr=args.lr)
    mask = causal_mask().realize()  # realized once; unrealized it would be recomputed inside every step

    budget = args.tokens if args.tokens is not None else (500_000_000 if args.full else 50_000_000)
    tokens_per_step = args.batch * CTX
    steps = max(1, budget // tokens_per_step)
    print(f"{sum(p.numel for p in params) / 1e6:.2f}M params, {steps} steps of {tokens_per_step} tokens on {args.device}")

    @capture
    def train_step(x: Tensor, y: Tensor) -> Tensor:
        opt.zero_grad()
        loss = cross_entropy(model(x, mask), y)
        loss.backward()
        opt.step(loss)  # the loss realizes with the updates, so logging it costs no second forward
        opt.zero_grad()  # once recorded the function never runs again; leave no gradient graphs behind
        return loss

    start = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = batch_of(data, args.batch, rng)
        loss = train_step(x, y)
        if step % args.log_every == 0 or step == steps:
            done = step * tokens_per_step
            rate = done / (time.perf_counter() - start)
            eta = (steps - step) * tokens_per_step / rate
            print(f"step {step:6d}/{steps}  loss {loss.item():.4f}  {rate:8.0f} tok/s  eta {eta / 3600:.2f}h", flush=True)
        if step % args.checkpoint_every == 0 or step == steps:
            save_checkpoint(checkpoint, params)

    print(f"\ndone in {(time.perf_counter() - start) / 3600:.2f}h; a story:\n")
    print(sample(model, mask, args.sample_bytes, args.temperature, rng))


if __name__ == "__main__":
    main()
