"""Checkpoints as safetensors: an 8-byte length, that many bytes of json, then the tensor bytes.

The format earns its place by being the one every other framework already reads. The header
names each tensor's dtype, shape and byte range, the ranges are plain row-major bytes, and
nothing in the file executes. limn's layouts match torch's, so a file written here loads there
untransposed and one torch wrote loads here.

Both sides are numpy plus the bfloat16 ml_dtypes registers, so a checkpoint costs no dependency.
Writing realizes: .numpy() runs the graph, folds the View to contiguous bytes, and pulls a cuda
tensor back to the host. Reading maps the file and copies each range into a buffer on the active
device, so where a checkpoint lands is set by set_device rather than by the file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from limn.device import NUMPY_DTYPES
from limn.ops import DType, bfloat16, float16, float32, float64, int8, int16, int32
from limn.tensor import Tensor, realize

# total over limn's seven dtypes in both directions: everything a tensor can hold has a name in
# the file, and every name limn writes reads back as the dtype it was
DTYPES: dict[str, DType] = {
    "F64": float64,
    "F32": float32,
    "F16": float16,
    "BF16": bfloat16,
    "I32": int32,
    "I16": int16,
    "I8": int8,
}
NAMES: dict[DType, str] = {dtype: name for name, dtype in DTYPES.items()}


def save_file(tensors: Mapping[str, Tensor], path: str | Path, metadata: Mapping[str, str] | None = None) -> None:
    """Write these tensors to path, in the order given.

    metadata rides in the header under __metadata__ and holds strings only, which is the format's
    rule and not a limitation worth working around: hyperparameters go in as json, and a counter a
    step advances stays a tensor so the step can keep advancing it.
    """
    if metadata is not None and any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("safetensors metadata holds strings only; json-encode anything else")
    arrays = {name: t.numpy() for name, t in tensors.items()}
    header: dict = {} if metadata is None else {"__metadata__": dict(metadata)}
    offset = 0
    for name, t in tensors.items():
        header[name] = {"dtype": NAMES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + arrays[name].nbytes]}
        offset += arrays[name].nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # padded, so the bytes start 8-byte aligned and a reader can map them in place
    with open(path, "wb") as f:
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
        for array in arrays.values():
            f.write(array.tobytes())


def load_file(path: str | Path) -> dict[str, Tensor]:
    """Every tensor in the file, each one buffer on the active device.

    The file is mapped rather than read, so only the ranges actually copied into buffers are ever
    paged in, and the whole of it is never resident at once.
    """
    blob = np.memmap(path, dtype=np.uint8, mode="r")
    header, start = _header(blob)
    out: dict[str, Tensor] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if entry["dtype"] not in DTYPES:
            raise ValueError(f"{name}: dtype {entry['dtype']} is not one of limn's {tuple(DTYPES)}")
        dtype = DTYPES[entry["dtype"]]
        begin, end = entry["data_offsets"]
        raw = blob[start + begin : start + end].view(NUMPY_DTYPES[dtype])
        out[name] = Tensor(raw.reshape(entry["shape"]), dtype=dtype)
    return out


def load_metadata(path: str | Path) -> dict[str, str]:
    """The header's __metadata__ strings, without reading a byte of the tensors."""
    return _header(np.memmap(path, dtype=np.uint8, mode="r"))[0].get("__metadata__", {})


def load_into(tensors: Mapping[str, Tensor], path: str | Path) -> None:
    """Write the file's values into these tensors' own buffers, committed in one realize().

    An ASSIGN rather than a rebind, and the difference is not style. Rebinding gives the module
    fresh buffers, while a limn.capture recording holds the buffers it saw, so a captured step
    would go on replaying the weights the load meant to replace. Assigning writes where every
    reader is already looking, and one realize() makes the whole checkpoint land at once.

    The tensors handed in are what gets read, so a file may hold more than this map names: a model
    loads out of a file that carries the optimizer state too. Every name here must be in the file
    at the same shape and dtype, since a checkpoint that silently half-loads is worse than one
    that refuses.
    """
    stored = load_file(path)
    for name, t in tensors.items():
        have = stored.get(name)
        if have is None or have.shape != t.shape or have.dtype != t.dtype:
            held = "no entry" if have is None else f"{have.shape} {have.dtype}"
            raise ValueError(f"{path} has {held} for {name!r}, which is {t.shape} {t.dtype}")
    realize(*[t.assign(stored[name]) for name, t in tensors.items()])


def _header(blob: np.ndarray) -> tuple[dict, int]:
    """The parsed json header, and the offset its byte ranges count from."""
    length = int.from_bytes(blob[:8].tobytes(), "little")
    return json.loads(blob[8 : 8 + length].tobytes()), 8 + length
