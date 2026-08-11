"""Train a byte-level GPT on TinyStories and make it tell one.

The default run is small (50M tokens): enough for word-correct baby English. --full trains
500M tokens overnight. Both log loss and throughput, checkpoint as they go, resume with
--resume, and finish by sampling a story. The checkpoint is one safetensors file holding the
weights and the optimizer state, so a resumed run continues the trajectory the uninterrupted
one would have taken. The dataset (about 2 GB of text) downloads on first use into
examples/data/.
"""

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np

from limn import Tensor, capture, no_grad, realize, set_device, set_seed
from limn.nn import Embedding, LayerNorm, Linear, named_parameters
from limn.optim import AdamW
from limn.serialize import load_into, save_file

DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
DATA_DIR = Path(__file__).parent / "data"

VOCAB, CTX, DIM, LAYERS, HEADS = 256, 256, 192, 6, 6


class Block:
    def __init__(self):
        self.ln1, self.ln2 = LayerNorm(DIM), LayerNorm(DIM)
        self.qkv = Linear(DIM, 3 * DIM)
        self.proj = Linear(DIM, DIM)
        self.up, self.down = Linear(DIM, 4 * DIM), Linear(4 * DIM, DIM)

    def __call__(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        hd = c // HEADS
        qkv = self.qkv(self.ln1(x)).reshape(b, t, 3, HEADS, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        att = q.attention(k, v, causal=True)
        x = x + self.proj(att.permute(0, 2, 1, 3).reshape(b, t, c))
        return x + self.down(self.up(self.ln2(x)).relu())

    def step(self, x: Tensor, kc: Tensor, vc: Tensor, pos: Tensor) -> Tensor:
        """One position against the kv caches: its k and v land in slot pos, and the query row
        reads the slots filled so far. Attention reads the updated value rather than the cache
        buffer, since assigns commit only after every sink computes."""
        b, _, c = x.shape
        hd = c // HEADS
        q, k, v = self.qkv(self.ln1(x)).reshape(b, 1, 3, HEADS, hd).permute(2, 0, 3, 1, 4)
        slot = Tensor.arange(CTX).reshape(CTX, 1).eq(pos)
        new_k, new_v = slot.where(k, kc), slot.where(v, vc)
        att = q.attention(new_k, new_v, key_mask=Tensor.arange(CTX) <= pos)
        kc.assign(new_k)
        vc.assign(new_v)
        x = x + self.proj(att.permute(0, 2, 1, 3).reshape(b, 1, c))
        return x + self.down(self.up(self.ln2(x)).relu())


class GPT:
    def __init__(self):
        self.tok = Embedding(VOCAB, DIM)
        self.pos = Embedding(CTX, DIM)
        self.blocks = [Block() for _ in range(LAYERS)]
        self.ln = LayerNorm(DIM)
        self.head = Linear(DIM, VOCAB, bias=False)

    def __call__(self, tokens: Tensor) -> Tensor:
        x = self.tok(tokens) + self.pos(Tensor.arange(CTX))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))

    def step(self, token: Tensor, pos: Tensor, caches: list[list[Tensor]]) -> Tensor:
        """Next-byte logits for one position, reading and advancing the per-block kv caches."""
        x = self.tok(token) + self.pos(pos)
        for block, (kc, vc) in zip(self.blocks, caches):
            x = block.step(x, kc, vc, pos)
        return self.head(self.ln(x))


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


def sample(model: GPT, n: int, temperature: float, rng: np.random.Generator) -> str:
    """One byte at a time against the kv caches. Prefill and decode are the same captured step,
    so each byte costs one recorded plan over a single position instead of a full-window
    forward. When the caches fill, the newest half of the text refeeds at fresh positions: the
    forgetting the old sliding window did every byte, once per half context instead."""
    caches = [[Tensor.zeros((1, HEADS, CTX, DIM // HEADS)).realize() for _ in "kv"] for _ in model.blocks]

    @capture
    def step(token: Tensor, pos: Tensor) -> Tensor:
        logits = model.step(token, pos, caches)
        realize(logits, *[c for pair in caches for c in pair])
        return logits

    prompt = b"Once upon a time"
    out = bytearray(prompt)
    fed = pos = 0  # bytes of out already in the caches, and the slot the next one lands in
    with no_grad():
        while len(out) < len(prompt) + n:
            if pos == CTX:
                fed, pos = len(out) - CTX // 2, 0
            logits = step(Tensor(np.array([[out[fed]]], dtype=np.int32)), Tensor(np.array([pos], dtype=np.int32)))
            fed, pos = fed + 1, pos + 1
            if fed == len(out):  # caught up: these logits predict a byte nothing has seen
                row = logits.numpy()[0, 0]
                weights = np.exp((row - row.max()) / temperature)
                out.append(int(rng.choice(VOCAB, p=weights / weights.sum())))
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
    named = dict(named_parameters(model))
    opt = AdamW(named.values(), lr=args.lr)
    state = {**named, **opt.state_dict(named)}  # live buffers, so the one map both saves and resumes
    checkpoint = DATA_DIR / "stories_checkpoint.safetensors"
    if args.resume and checkpoint.exists():
        load_into(state, checkpoint)
        print(f"resumed from {checkpoint}")

    budget = args.tokens if args.tokens is not None else (500_000_000 if args.full else 50_000_000)
    tokens_per_step = args.batch * CTX
    steps = max(1, budget // tokens_per_step)
    print(f"{sum(p.numel for p in named.values()) / 1e6:.2f}M params, {steps} steps of {tokens_per_step} tokens on {args.device}")

    @capture
    def train_step(x: Tensor, y: Tensor) -> Tensor:
        opt.zero_grad()
        loss = cross_entropy(model(x), y)
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
            save_file(state, checkpoint)

    print(f"\ndone in {(time.perf_counter() - start) / 3600:.2f}h; a story:\n")
    print(sample(model, args.sample_bytes, args.temperature, rng))


if __name__ == "__main__":
    main()
