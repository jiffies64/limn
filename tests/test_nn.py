"""Layers built from limn, held to torch: a 2-layer transformer (embeddings, causal attention,
layernorm, relu MLP, cross-entropy) forward+backward against an identical torch model, convs
against F.conv, and BatchNorm's forward, backward and running stats against BatchNorm2d."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import COMPILED, check, randf

from limn import Tensor, realize, set_seed
from limn.nn import BatchNorm, Conv1d, Conv2d, Embedding, LayerNorm, Linear, named_parameters, parameters
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


def batchnorm_pair(channels: int) -> tuple[BatchNorm, torch.nn.BatchNorm2d]:
    """Same init on both sides: weight 1, bias 0, running mean 0 and var 1, momentum 0.1."""
    bn, tbn = BatchNorm(channels), torch.nn.BatchNorm2d(channels)
    assert tbn.weight is not None and tbn.bias is not None
    tbn.weight.data = torch.tensor(bn.weight.numpy())
    tbn.bias.data = torch.tensor(bn.bias.numpy())
    return bn, tbn


def test_batchnorm_matches_torch():
    """Train forward, backward, and the first stats update, against torch.nn.BatchNorm2d."""
    set_seed(0)
    bn, tbn = batchnorm_pair(3)
    x = randf(4, 3, 5, 6)
    lx, tx = Tensor(x, requires_grad=True), torch.tensor(x, requires_grad=True)
    out, tout = bn(lx), tbn(tx)
    realize(out, bn.running_mean, bn.running_var)  # the stats assigns commit with the step
    np.testing.assert_allclose(out.numpy(), tout.detach().numpy(), atol=1e-4, rtol=1e-4)

    (out * out).sum().backward()
    (tout * tout).sum().backward()
    for name, lgrad, tgrad in [
        ("input", lx.grad, tx.grad),
        ("weight", bn.weight.grad, tbn.weight.grad),
        ("bias", bn.bias.grad, tbn.bias.grad),
    ]:
        assert lgrad is not None and tgrad is not None, f"missing gradient for {name}"
        np.testing.assert_allclose(lgrad.numpy(), tgrad.numpy(), atol=1e-4, rtol=1e-4, err_msg=f"gradient mismatch for {name}")
    assert tbn.running_mean is not None and tbn.running_var is not None
    np.testing.assert_allclose(bn.running_mean.numpy(), tbn.running_mean.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(bn.running_var.numpy(), tbn.running_var.numpy(), atol=1e-5, rtol=1e-5)


def test_batchnorm_running_stats_track_torch_across_training():
    """The deliverable: a hundred train steps leave the stats torch-exact, and eval reads them."""
    bn, tbn = batchnorm_pair(3)
    rng = np.random.default_rng(5)
    for step in range(100):
        x = rng.standard_normal((4, 3, 4, 4)).astype(np.float32)
        out = bn(Tensor(x))
        tbn(torch.tensor(x))
        realize(out, bn.running_mean, bn.running_var)
        assert tbn.running_mean is not None and tbn.running_var is not None
        np.testing.assert_allclose(
            bn.running_mean.numpy(), tbn.running_mean.numpy(), atol=1e-5, rtol=1e-5, err_msg=f"mean diverged at step {step}"
        )
        np.testing.assert_allclose(
            bn.running_var.numpy(), tbn.running_var.numpy(), atol=1e-5, rtol=1e-5, err_msg=f"var diverged at step {step}"
        )
    bn.training = False
    tbn.eval()
    x = rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
    np.testing.assert_allclose(bn(Tensor(x)).numpy(), tbn(torch.tensor(x)).detach().numpy(), atol=1e-5, rtol=1e-5)


def test_batchnorm_small_batch_exposes_the_unbiased_factor():
    """Two values per channel: the biased/unbiased factor is 2, not a rounding blur."""
    bn, tbn = batchnorm_pair(2)
    x = randf(2, 2, 1, 1)
    out, tout = bn(Tensor(x)), tbn(torch.tensor(x))
    realize(out, bn.running_mean, bn.running_var)
    np.testing.assert_allclose(out.numpy(), tout.detach().numpy(), atol=1e-5, rtol=1e-5)
    assert tbn.running_var is not None
    np.testing.assert_allclose(bn.running_var.numpy(), tbn.running_var.numpy(), atol=1e-5, rtol=1e-5)


def test_batchnorm_eval_leaves_the_stats_bit_identical():
    bn, _ = batchnorm_pair(3)
    out = bn(Tensor(randf(4, 3, 4, 4)))
    realize(out, bn.running_mean, bn.running_var)
    mean_before, var_before = bn.running_mean.numpy().copy(), bn.running_var.numpy().copy()
    bn.training = False
    bn(Tensor(randf(4, 3, 4, 4))).numpy()
    assert bn.running_mean.node.op is Op.BUFFER and bn.running_var.node.op is Op.BUFFER  # eval built no assign
    np.testing.assert_array_equal(bn.running_mean.numpy(), mean_before)
    np.testing.assert_array_equal(bn.running_var.numpy(), var_before)


def test_batchnorm_stats_advance_once_per_realized_step():
    """forward -> realize -> forward -> realize: each realize commits exactly one update."""
    bn = BatchNorm(2)
    rng = np.random.default_rng(6)
    expected_mean, expected_var = np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
    for _ in range(2):
        x = rng.standard_normal((3, 2, 4, 4)).astype(np.float32)
        out = bn(Tensor(x))  # a second forward before the first realize would raise
        realize(out, bn.running_mean, bn.running_var)
        expected_mean = 0.9 * expected_mean + 0.1 * x.mean(axis=(0, 2, 3))
        expected_var = 0.9 * expected_var + 0.1 * x.var(axis=(0, 2, 3)) * (48 / 47)  # unbiased, 48 values per channel
        np.testing.assert_allclose(bn.running_mean.numpy(), expected_mean, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(bn.running_var.numpy(), expected_var, atol=1e-5, rtol=1e-5)


def test_batchnorm_second_forward_before_realize_raises():
    bn = BatchNorm(3)
    bn(Tensor(randf(4, 3, 4, 4)))
    with pytest.raises(ValueError, match="realized buffer"):
        bn(Tensor(randf(4, 3, 4, 4)))


@pytest.mark.parametrize("backend", COMPILED)
def test_batchnorm_forward_on_compiled_backends(backend):
    set_seed(0)
    bn = BatchNorm(3)
    check(backend.shared, bn(Tensor(randf(2, 3, 7))))  # (batch, channels, length): axes reduce over 0 and 2
