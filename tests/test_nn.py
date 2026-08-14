"""A 2-layer transformer (embeddings, causal attention, layernorm, relu MLP, cross-entropy)
built from limn layers, forward+backward checked against an identical torch model."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import COMPILED, check, randf

from limn import Tensor, realize, set_device, set_seed
from limn.capture import capture
from limn.nn import Conv1d, Conv2d, Dropout, Embedding, LayerNorm, Linear, named_parameters, parameters
from limn.ops import Op

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
    layers = []
    for i, (lb, tb) in enumerate(zip(lmodel.blocks, tmodel.blocks)):
        for name in ("ln1", "ln2", "qkv", "proj", "fc1", "fc2"):
            llayer = getattr(lb.attn, name) if name in ("qkv", "proj") else getattr(lb, name)
            layers.append((f"block{i}.{name}", llayer, getattr(tb, name)))
    layers += [("ln_f", lmodel.ln_f, tmodel.ln_f), ("head", lmodel.head, tmodel.head)]
    for label, llayer, tlayer in layers:
        pairs.append((f"{label}.weight", llayer.weight, tlayer.weight))
        pairs.append((f"{label}.bias", llayer.bias, tlayer.bias))
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


def test_named_parameters_names_the_path_it_walked():
    class Model:
        def __init__(self):
            self.stem = Linear(3, 2)
            self.heads = {"a": Linear(2, 4)}
            self.blocks = [LayerNorm(2)]
            self.tied = self.stem.weight  # a second path to a tensor already found

    model = Model()
    named = named_parameters(model)
    assert [name for name, _ in named] == [
        "stem.weight",
        "stem.bias",
        "heads.a.weight",
        "heads.a.bias",
        "blocks.0.weight",
        "blocks.0.bias",
    ]
    assert [id(p) for _, p in named] == [id(p) for p in parameters(model)]  # one walk, so one order


def test_parameters_finds_layers_held_in_a_dict():
    class Model:
        def __init__(self):
            self.stem = Linear(3, 2)
            self.heads = {"a": Linear(2, 4), "b": Linear(2, 4)}

    assert len(parameters(Model())) == 6


def test_parameters_walks_reference_cycles_once():
    class Cell:
        def __init__(self, layer: Linear):
            self.layer = layer
            self.peer: Cell | None = None

    first, second = Cell(Linear(3, 2)), Cell(Linear(2, 4))
    first.peer, second.peer = second, first  # a cycle between two cells
    found = parameters(first)
    assert len(found) == 4  # two weights and two biases, each seen exactly once
    assert len({id(p) for p in found}) == 4

    solo = Cell(Linear(3, 2))
    solo.peer = solo
    assert len(parameters(solo)) == 2


def test_dropout_statistics():
    """About p of the elements zero, and the survivors are exactly x/(1-p)."""
    set_seed(0)
    d = Dropout(0.5)
    x = Tensor(np.ones((200, 200), dtype=np.float32))
    out = d(x)
    realize(out, d.key)
    kept = out.numpy() != 0
    assert 0.49 < kept.mean() < 0.51
    np.testing.assert_array_equal(out.numpy()[kept], np.full(kept.sum(), 2.0, dtype=np.float32))


def test_dropout_backward_uses_the_forward_mask():
    """Gradients are zero exactly where the mask dropped and 1/(1-p) where it kept.

    The backward graph recomputes the mask from the key's buffer, so the key's assign must
    commit in the same realize batch as the gradients; an assign committed earlier would
    recompute a different mask and break this exact equality.
    """
    set_seed(0)
    d = Dropout(0.5)
    x = Tensor(np.arange(1, 25, dtype=np.float32).reshape(4, 6), requires_grad=True)
    out = d(x)
    out.sum().backward()
    assert x.grad is not None
    realize(x.grad, out, d.key)  # one batch: mask and gradient read pre-assign key bytes
    kept = out.numpy() != 0
    np.testing.assert_array_equal(x.grad.numpy(), np.where(kept, 2.0, 0.0).astype(np.float32))


def test_dropout_key_advances_once_per_realized_step():
    """Each realized step commits one step-word increment and a fresh mask, not each forward."""
    set_seed(0)
    d = Dropout(0.5)
    x = Tensor(np.ones((64,), dtype=np.float32))
    seed = int(d.key.numpy()[0])
    masks = []
    for step in range(3):
        out = d(x)
        realize(out, d.key)
        key = d.key.numpy()
        assert (int(key[0]), int(key[1])) == (seed, step + 1)
        masks.append(out.numpy())
    assert not np.array_equal(masks[0], masks[1])  # threefry: consecutive steps hash different masks


def test_dropout_second_forward_before_realize_raises():
    d = Dropout(0.5)
    x = Tensor(np.ones((8,), dtype=np.float32))
    d(x)
    with pytest.raises(ValueError, match="realized buffer"):
        d(x)  # the key's assign is still pending, like any unrealized assign target


def test_dropout_eval_returns_x_untouched_and_leaves_the_key():
    set_seed(0)
    d = Dropout(0.5)
    x = Tensor(randf(4, 5))
    realize(d(x), d.key)
    key = d.key.numpy().copy()
    d.training = False
    np.testing.assert_array_equal(d(x).numpy(), x.numpy())  # eval: x itself, not a scaled copy
    assert d.key.node.op is Op.BUFFER  # eval queued no assign
    np.testing.assert_array_equal(d.key.numpy(), key)


def test_dropout_rejects_bad_probability():
    with pytest.raises(ValueError, match="probability"):
        Dropout(1.0)
    with pytest.raises(ValueError, match="probability"):
        Dropout(-0.1)


@pytest.mark.parametrize("backend", COMPILED)
def test_dropout_forward_on_compiled_backends(backend):
    set_seed(0)
    d = Dropout(0.5)
    check(backend.shared, d(Tensor(randf(8, 12))))


@pytest.mark.parametrize("backend", COMPILED)
def test_captured_dropout_steps_replay_fresh_masks(backend):
    """The payoff: a captured train step replays with fresh masks, which host-side dropout cannot do."""
    set_device(backend.name)
    set_seed(0)
    d = Dropout(0.5)
    x = Tensor(np.ones((64,), dtype=np.float32))

    def step(batch: Tensor) -> Tensor:
        out = d(batch)
        realize(out, d.key)  # the key assign commits in the step's batch
        return out

    recorded = capture(step)
    recorded(x)  # the first two calls observe and settle the recording
    recorded(x)
    first = recorded(x).numpy()  # replays read the key their predecessor's assign left
    second = recorded(x).numpy()
    assert not np.array_equal(first, second)
