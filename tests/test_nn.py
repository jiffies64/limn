"""A 2-layer transformer (embeddings, causal attention, layernorm, relu MLP, cross-entropy)
built from limn layers, forward+backward checked against an identical torch model."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import randf

from limn import Tensor, set_seed
from limn.nn import Conv1d, Conv2d, Embedding, LayerNorm, Linear, parameters

BATCH, SEQ, VOCAB, DIM, HEADS, LAYERS = 2, 6, 19, 16, 4, 2


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy of (N, V) float logits against (N,) int32 class indices."""
    log_probs = logits.log_softmax(-1)
    onehot = targets.reshape(-1, 1).eq(Tensor.arange(logits.shape[-1])).float()
    return -(onehot * log_probs).sum() / logits.shape[0]


class Attention:
    def __init__(self, dim: int, heads: int):
        self.qkv = Linear(dim, 3 * dim)
        self.proj = Linear(dim, dim)
        self.heads = heads

    def __call__(self, x: Tensor) -> Tensor:
        batch, seq, dim = x.shape
        head_dim = dim // self.heads
        qkv = self.qkv(x)
        q, k, v = (
            qkv[:, :, i * dim : (i + 1) * dim].reshape(batch, seq, self.heads, head_dim).permute(0, 2, 1, 3) for i in range(3)
        )
        scores = (q @ k.transpose()) * head_dim**-0.5
        rows, cols = Tensor.arange(seq).reshape(seq, 1), Tensor.arange(seq).reshape(1, seq)
        causal = (cols <= rows).reshape(1, 1, seq, seq)
        weights = causal.where(scores, float("-inf")).softmax(-1)
        return self.proj((weights @ v).permute(0, 2, 1, 3).reshape(batch, seq, dim))


class Block:
    def __init__(self, dim: int, heads: int):
        self.ln1 = LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ln2 = LayerNorm(dim)
        self.fc1 = Linear(dim, 4 * dim)
        self.fc2 = Linear(4 * dim, dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(self.fc1(self.ln2(x)).relu())


class Transformer:
    def __init__(self):
        self.emb = Embedding(VOCAB, DIM)
        self.pos = Embedding(SEQ, DIM)
        self.blocks = [Block(DIM, HEADS) for _ in range(LAYERS)]
        self.ln_f = LayerNorm(DIM)
        self.head = Linear(DIM, VOCAB)

    def __call__(self, idx: Tensor) -> Tensor:
        x = self.emb(idx) + self.pos(Tensor.arange(idx.shape[-1]))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


class TorchBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = torch.nn.LayerNorm(DIM), torch.nn.LayerNorm(DIM)
        self.qkv, self.proj = torch.nn.Linear(DIM, 3 * DIM), torch.nn.Linear(DIM, DIM)
        self.fc1, self.fc2 = torch.nn.Linear(DIM, 4 * DIM), torch.nn.Linear(4 * DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        head_dim = dim // HEADS
        q, k, v = (z.view(batch, seq, HEADS, head_dim).transpose(1, 2) for z in self.qkv(self.ln1(x)).split(dim, dim=2))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(torch.tril(torch.ones(seq, seq)) == 0, float("-inf"))
        x = x + self.proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, seq, dim))
        return x + self.fc2(F.relu(self.fc1(self.ln2(x))))


class TorchTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.pos = torch.nn.Embedding(VOCAB, DIM), torch.nn.Embedding(SEQ, DIM)
        self.blocks = torch.nn.ModuleList(TorchBlock() for _ in range(LAYERS))
        self.ln_f, self.head = torch.nn.LayerNorm(DIM), torch.nn.Linear(DIM, VOCAB)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[-1]))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def param_pairs(lmodel: Transformer, tmodel: TorchTransformer) -> list[tuple[str, Tensor, torch.Tensor]]:
    pairs = [("emb", lmodel.emb.weight, tmodel.emb.weight), ("pos", lmodel.pos.weight, tmodel.pos.weight)]
    for i, (lb, tb) in enumerate(zip(lmodel.blocks, tmodel.blocks)):
        for name in ("ln1", "ln2", "qkv", "proj", "fc1", "fc2"):
            llayer = getattr(lb.attn, name) if name in ("qkv", "proj") else getattr(lb, name)
            tlayer = getattr(tb, name)
            pairs.append((f"block{i}.{name}.weight", llayer.weight, tlayer.weight))
            pairs.append((f"block{i}.{name}.bias", llayer.bias, tlayer.bias))
    pairs += [
        ("ln_f.weight", lmodel.ln_f.weight, tmodel.ln_f.weight),
        ("ln_f.bias", lmodel.ln_f.bias, tmodel.ln_f.bias),
        ("head.weight", lmodel.head.weight, tmodel.head.weight),
        ("head.bias", lmodel.head.bias, tmodel.head.bias),
    ]
    return pairs


def test_transformer_matches_torch():
    set_seed(0)
    torch.manual_seed(0)
    lmodel, tmodel = Transformer(), TorchTransformer()
    pairs = param_pairs(lmodel, tmodel)
    assert len(parameters(lmodel)) == len(pairs)
    for _, lparam, tparam in pairs:
        tparam.data = torch.tensor(lparam.numpy())

    rng = np.random.default_rng(3)
    idx = rng.integers(0, VOCAB, (BATCH, SEQ)).astype(np.int32)
    targets = rng.integers(0, VOCAB, (BATCH, SEQ)).astype(np.int32)

    logits = lmodel(Tensor(idx))
    loss = cross_entropy(logits.reshape(BATCH * SEQ, VOCAB), Tensor(targets).reshape(BATCH * SEQ))
    tlogits = tmodel(torch.tensor(idx, dtype=torch.long))
    tloss = F.cross_entropy(tlogits.reshape(BATCH * SEQ, VOCAB), torch.tensor(targets, dtype=torch.long).reshape(-1))

    np.testing.assert_allclose(logits.numpy(), tlogits.detach().numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(loss.numpy(), tloss.detach().numpy(), atol=1e-4, rtol=1e-4)

    loss.backward()
    tloss.backward()
    for name, lparam, tparam in pairs:
        assert lparam.grad is not None and tparam.grad is not None, f"missing gradient for {name}"
        np.testing.assert_allclose(
            lparam.grad.numpy(), tparam.grad.numpy(), atol=1e-4, rtol=1e-4, err_msg=f"gradient mismatch for {name}"
        )


CONV_CASES = [
    # layer, spatial input, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
    (Conv1d, (11,), 3, 4, 3, 1, 0, 1, 1, True),
    (Conv1d, (11,), 2, 5, 1, 1, 0, 1, 1, False),
    (Conv1d, (10,), 4, 6, 4, 3, 2, 1, 2, True),  # strided: the far pad grows past the requested one
    (Conv1d, (9,), 4, 4, 3, 1, "same", 2, 4, True),  # depthwise, dilated, output size held
    (Conv1d, (8,), 3, 3, 2, 1, "same", 1, 1, False),  # even kernel: 'same' pads asymmetrically
    (Conv2d, (9, 8), 3, 4, 3, 1, 0, 1, 1, True),
    (Conv2d, (9, 8), 3, 4, 3, 1, 1, 1, 1, False),
    (Conv2d, (9, 8), 2, 5, 1, 1, 0, 1, 1, True),
    (Conv2d, (9, 8), 4, 6, (3, 2), 2, (2, 1), 1, 2, True),
    (Conv2d, (9, 8), 4, 4, 3, 1, "same", 2, 4, True),  # depthwise, dilated, output size held
    (Conv2d, (9, 8), 3, 3, (4, 2), 1, "same", 1, 1, False),  # even kernel: 'same' pads asymmetrically
    (Conv2d, (9, 8), 6, 9, (2, 3), (2, 3), (1, 2), (2, 1), 3, True),
]

TORCH_CONV = {Conv1d: F.conv1d, Conv2d: F.conv2d}


@pytest.mark.parametrize(
    "layer_cls, spatial, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias", CONV_CASES
)
def test_conv_matches_torch(layer_cls, spatial, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    set_seed(0)
    layer = layer_cls(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    assert len(parameters(layer)) == (2 if bias else 1)

    x = randf(2, in_channels, *spatial)
    lx = Tensor(x, requires_grad=True)
    tx = torch.tensor(x, requires_grad=True)
    tweight = torch.tensor(layer.weight.numpy(), requires_grad=True)
    tbias = torch.tensor(layer.bias.numpy(), requires_grad=True) if layer.bias is not None else None

    out = layer(lx)
    tout = TORCH_CONV[layer_cls](tx, tweight, tbias, stride, padding, dilation, groups)
    assert out.shape == tuple(tout.shape)
    np.testing.assert_allclose(out.numpy(), tout.detach().numpy(), atol=1e-4, rtol=1e-4)

    (out * out).sum().backward()  # a weighted sum, so every output position reaches the gradients
    (tout * tout).sum().backward()
    pairs = [("input", lx, tx), ("weight", layer.weight, tweight)]
    if layer.bias is not None and tbias is not None:
        pairs.append(("bias", layer.bias, tbias))
    for name, lparam, tparam in pairs:
        assert lparam.grad is not None and tparam.grad is not None, f"missing gradient for {name}"
        np.testing.assert_allclose(
            lparam.grad.numpy(), tparam.grad.numpy(), atol=1e-4, rtol=1e-4, err_msg=f"gradient mismatch for {name}"
        )


def test_conv_rejects_bad_configuration():
    with pytest.raises(ValueError, match="groups"):
        Conv2d(4, 6, 3, groups=4)
    with pytest.raises(ValueError, match="padding"):
        Conv1d(4, 6, 3, padding="valid")
    with pytest.raises(ValueError, match="same"):
        Conv2d(4, 6, 3, stride=2, padding="same")
    with pytest.raises(ValueError, match="kernel_size"):
        Conv1d(4, 6, (3, 3))
    with pytest.raises(ValueError, match="positive"):
        Conv2d(4, 6, 0)


def test_conv_rejects_bad_input():
    with pytest.raises(ValueError, match="does not fit"):
        Conv2d(3, 4, 5)(Tensor.zeros((1, 3, 4, 4)))
    with pytest.raises(ValueError, match="channels"):
        Conv1d(3, 4, 3)(Tensor.zeros((1, 5, 8)))
    with pytest.raises(ValueError, match="spatial"):
        Conv1d(3, 4, 3)(Tensor.zeros((1, 3, 8, 8)))


def test_parameters_finds_layers_held_in_a_dict():
    class Model:
        def __init__(self):
            self.stem = Linear(3, 2)
            self.heads = {"a": Linear(2, 4), "b": Linear(2, 4)}

    assert len(parameters(Model())) == 6


def test_parameters_walks_reference_cycles_once():
    class Block:
        def __init__(self, layer: Linear):
            self.layer = layer
            self.peer: Block | None = None

    first, second = Block(Linear(3, 2)), Block(Linear(2, 4))
    first.peer, second.peer = second, first  # a cycle between two blocks
    found = parameters(first)
    assert len(found) == 4  # two weights and two biases, each seen exactly once
    assert len({id(p) for p in found}) == 4

    solo = Block(Linear(3, 2))
    solo.peer = solo
    assert len(parameters(solo)) == 2
