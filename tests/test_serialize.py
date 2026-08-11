"""The checkpoint format, held to the safetensors library rather than to itself: a round trip
through limn alone would pass with a wrong header length or a swapped dtype tag."""

import json

import numpy as np
import pytest
import safetensors.torch
import torch
from conftest import needs_cc, needs_cuda, randf
from safetensors import safe_open

from limn import Tensor, capture, realize, set_device, set_seed
from limn.nn import Embedding, LayerNorm, Linear, named_parameters
from limn.ops import DTYPES as LIMN_DTYPES
from limn.optim import SGD, AdamW
from limn.serialize import DTYPES, NAMES, load_file, load_into, load_metadata, save_file

TORCH_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
}
VALUES = np.array([[1, -2, 3], [-4, 5, -6]])  # exact in all seven dtypes, so every comparison below is exact
VOCAB, DIM = 11, 8


def as_float(array) -> np.ndarray:
    """A limn or torch array at a width that holds any of the seven exactly, so one comparison serves all."""
    return array.to(torch.float64).numpy() if isinstance(array, torch.Tensor) else array.astype(np.float64)


def test_the_dtype_map_covers_every_limn_dtype():
    assert set(NAMES) == set(LIMN_DTYPES) and set(DTYPES) == set(TORCH_DTYPES)


@pytest.mark.parametrize("name", list(DTYPES))
def test_round_trip_against_the_safetensors_library(tmp_path, name):
    ours, theirs = tmp_path / "ours.safetensors", tmp_path / "theirs.safetensors"
    dtype = DTYPES[name]
    t = Tensor(VALUES, dtype=dtype)

    save_file({"x": t}, ours)
    read = safetensors.torch.load_file(ours)["x"]
    assert read.dtype is TORCH_DTYPES[name] and tuple(read.shape) == t.shape
    np.testing.assert_array_equal(as_float(read), as_float(t.numpy()))

    safetensors.torch.save_file({"x": torch.tensor(VALUES).to(TORCH_DTYPES[name])}, theirs)
    back = load_file(theirs)["x"]
    assert back.dtype is dtype and back.shape == t.shape
    np.testing.assert_array_equal(as_float(back.numpy()), as_float(t.numpy()))


def test_a_view_saves_the_bytes_it_presents(tmp_path):
    path = tmp_path / "view.safetensors"
    t = Tensor(randf(4, 3))
    save_file({"x": t.transpose()}, path)  # .numpy() folds the View, so the file is row-major (3, 4)
    read = safetensors.torch.load_file(path)["x"]
    np.testing.assert_array_equal(read.numpy(), t.numpy().T)


def test_metadata_round_trips_as_strings(tmp_path):
    path = tmp_path / "meta.safetensors"
    hyper = {"dim": DIM, "vocab": VOCAB}
    save_file({"x": Tensor(VALUES)}, path, metadata={"hyper": json.dumps(hyper)})
    assert json.loads(load_metadata(path)["hyper"]) == hyper
    with safe_open(path, framework="pt") as f:
        assert f.metadata() == load_metadata(path)

    save_file({"x": Tensor(VALUES)}, path)
    assert load_metadata(path) == {}
    unstringed: dict = {"dim": DIM}
    with pytest.raises(ValueError, match="strings only"):
        save_file({"x": Tensor(VALUES)}, path, metadata=unstringed)


class Small:
    def __init__(self):
        self.emb = Embedding(VOCAB, DIM)
        self.fc = Linear(DIM, DIM)
        self.ln = LayerNorm(DIM)

    def __call__(self, idx: Tensor) -> Tensor:
        return self.ln(self.fc(self.emb(idx)))


class TorchSmall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(VOCAB, DIM)
        self.fc = torch.nn.Linear(DIM, DIM)
        self.ln = torch.nn.LayerNorm(DIM)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.ln(self.fc(self.emb(idx)))


def test_a_torch_checkpoint_lands_in_the_matching_layers(tmp_path):
    """Both directions, unaltered: the walk's names are torch's state_dict keys and the layouts
    already match, so neither side renames a tensor or transposes one."""
    set_seed(0)
    torch.manual_seed(0)
    model, tmodel = Small(), TorchSmall()
    named = dict(named_parameters(model))
    assert set(named) == set(tmodel.state_dict())

    idx = np.arange(VOCAB - 1).reshape(2, 5).astype(np.int32)
    lidx, tidx = Tensor(idx), torch.tensor(idx, dtype=torch.long)
    theirs, ours = tmp_path / "torch.safetensors", tmp_path / "limn.safetensors"

    safetensors.torch.save_file(tmodel.state_dict(), theirs)
    load_into(named, theirs)
    np.testing.assert_allclose(model(lidx).numpy(), tmodel(tidx).detach().numpy(), atol=1e-6)

    set_seed(1)
    model = Small()  # fresh weights, so the file below is the only thing that can carry them across
    save_file(dict(named_parameters(model)), ours)
    tmodel.load_state_dict(safetensors.torch.load_file(ours))
    np.testing.assert_allclose(model(lidx).numpy(), tmodel(tidx).detach().numpy(), atol=1e-6)


def test_load_into_writes_the_buffers_every_reader_already_holds(tmp_path):
    path = tmp_path / "w.safetensors"
    layer = Linear(3, 2)
    named = dict(named_parameters(layer))
    saved = {name: t.numpy().copy() for name, t in named.items()}
    buffers = {name: t.node.arg for name, t in named.items()}
    save_file(named, path)

    realize(*[t.assign(t.detach() + 1.0) for t in named.values()])
    load_into(named, path)
    for name, t in named.items():
        np.testing.assert_array_equal(t.numpy(), saved[name])
        assert t.node.arg is buffers[name]  # an assign, so nothing was handed a new buffer
        assert t.requires_grad


def test_load_into_reads_only_what_it_was_asked_for(tmp_path):
    path = tmp_path / "pair.safetensors"
    a = Tensor(randf(2, 3))
    save_file({"a": a, "b": Tensor(randf(2, 3))}, path)
    only_a = Tensor.zeros((2, 3))
    load_into({"a": only_a}, path)  # the file holds b as well, which nothing here names
    np.testing.assert_array_equal(only_a.numpy(), a.numpy())


def test_load_into_refuses_what_it_cannot_land(tmp_path):
    path = tmp_path / "w.safetensors"
    save_file({"w": Tensor(randf(2, 3))}, path)
    with pytest.raises(ValueError, match="no entry"):
        load_into({"missing": Tensor.zeros((2, 3))}, path)
    with pytest.raises(ValueError, match="which is"):
        load_into({"w": Tensor.zeros((3, 2))}, path)
    with pytest.raises(ValueError, match="which is"):
        load_into({"w": Tensor.zeros((2, 3), dtype=DTYPES["F16"])}, path)


def steps(layer: Linear, opt: AdamW, batches: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
    losses = []
    for x, y in batches:
        opt.zero_grad()
        err = layer(Tensor(x)) - Tensor(y)
        loss = (err * err).mean()
        loss.backward()
        opt.step(loss)
        losses.append(loss.item())
    return losses


def test_a_run_resumes_out_of_one_file(tmp_path):
    """Parameters and optimizer state in the same file: what follows the load is the trajectory
    the uninterrupted run took, which the parameters on their own do not reproduce."""
    path = tmp_path / "run.safetensors"
    data = [(randf(6, 4), randf(6, 3)) for _ in range(8)]

    def fresh(seed: int) -> tuple[Linear, dict[str, Tensor], AdamW]:
        set_seed(seed)
        layer = Linear(4, 3)
        named = dict(named_parameters(layer))
        return layer, named, AdamW(list(named.values()), lr=1e-2)

    layer, named, opt = fresh(0)
    steps(layer, opt, data[:4])
    save_file({**named, **opt.state_dict(named)}, path)
    straight = steps(layer, opt, data[4:])

    layer, named, opt = fresh(1)  # a different init, so only the file can carry the run across
    load_into({**named, **opt.state_dict(named)}, path)
    assert steps(layer, opt, data[4:]) == straight

    layer, named, opt = fresh(1)
    load_into(named, path)  # the weights alone: the moments and beta**t stay cold, and it shows
    assert steps(layer, opt, data[4:]) != straight


@needs_cuda
def test_round_trip_on_cuda(tmp_path):
    path = tmp_path / "cuda.safetensors"
    set_device("cuda")
    t = Tensor(randf(4, 3))
    save_file({"x": t}, path)  # .numpy() pulls it off the device
    values = t.numpy().copy()
    from_cuda = load_file(path)["x"].numpy()  # written into device memory, read back out
    set_device("numpy")
    np.testing.assert_array_equal(from_cuda, values)
    np.testing.assert_array_equal(load_file(path)["x"].numpy(), values)  # the same file on the reference device


@needs_cc
def test_a_replay_reads_the_restored_buffers(tmp_path):
    """A capture holds the buffers it recorded, so a load that rebound the parameters would
    replay the weights it was meant to replace."""
    path = tmp_path / "step.safetensors"
    set_device("c")
    set_seed(5)
    layer = Linear(4, 3)
    named = dict(named_parameters(layer))
    opt = SGD(list(named.values()), lr=0.1)

    @capture
    def step(x: Tensor, y: Tensor) -> Tensor:
        opt.zero_grad()
        err = layer(x) - y
        loss = (err * err).mean()
        loss.backward()
        opt.step(loss)
        opt.zero_grad()
        return loss

    data = [(Tensor(randf(8, 4)), Tensor(randf(8, 3))) for _ in range(4)]
    for x, y in data:  # the first two calls observe, the rest replay
        step(x, y)
    save_file(named, path)
    moved = [step(x, y).item() for x, y in data]
    load_into(named, path)
    # the same replays over the restored bytes: same kernels, same buffers, so equality is exact
    assert [step(x, y).item() for x, y in data] == moved
