"""The device interface and the numpy reference interpreter.

A Device owns raw byte buffers and knows how to execute op graphs against them. Nothing
above this line of the stack (tensor.py, nn.py, optim.py) may know which device it is on.
NumpyDevice is the permanent reference implementation: it walks the graph topologically,
dispatching each Node to one numpy call. Every future backend gets diffed against it.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from limn.ops import DType, Node, Op, accumulate_in, float16, float32, float64, int8, int16, int32
from limn.sdpa import kernel as sdpa_kernel
from limn.view import View

type Buffer = Any

NUMPY_DTYPES: dict[DType, np.dtype] = {
    float64: np.dtype(np.float64),
    float32: np.dtype(np.float32),
    float16: np.dtype(np.float16),
    int32: np.dtype(np.int32),
    int16: np.dtype(np.int16),
    int8: np.dtype(np.int8),
}


class Device(Protocol):
    """What every backend must provide. Buffers are opaque to callers; dtype and shape live on the Node.

    execute() is one transaction: every sink's value is computed before any ASSIGN commits its
    write, so reads in a batch see pre-assign bytes and update order cannot matter (optim.py
    depends on this). When execute returns, each ASSIGN's target buffer holds the new bytes.
    """

    def alloc(self, nbytes: int) -> Buffer: ...
    def copyin(self, buf: Buffer, array: np.ndarray) -> None: ...
    def copyout(self, buf: Buffer) -> np.ndarray: ...
    def execute(self, sinks: list[Node]) -> list[Buffer]: ...
    def has_custom(self, name: str) -> bool: ...


class HostDevice:
    """Buffers as host bytes: flat uint8 arrays. The host backends share these three primitives."""

    def alloc(self, nbytes: int) -> Buffer:
        return np.zeros(nbytes, dtype=np.uint8)

    def copyin(self, buf: Buffer, array: np.ndarray) -> None:
        buf[:] = np.ascontiguousarray(array).view(np.uint8).reshape(-1)

    def copyout(self, buf: Buffer) -> np.ndarray:
        return buf.copy()


class NumpyDevice(HostDevice):
    """Reference interpreter: one numpy call per node, memoized over the DAG.

    Results are cached per Node in a weak map keyed by node identity, so shared subgraphs (a
    forward pass referenced by many gradients) compute once. BUFFER nodes are never cached:
    they always read the live bytes.

    Any cached value may have read bytes that a later ASSIGN overwrites, and a cache holding
    numbers that no longer follow from the buffers is worse than no cache at all, so
    committing an assign drops all of it. That costs little in practice: a training loop
    builds fresh Nodes every step, so the entries thrown away were already dead.
    """

    def __init__(self) -> None:
        self.cache: weakref.WeakKeyDictionary[Node, np.ndarray] = weakref.WeakKeyDictionary()
        # the CUSTOM kernels this device interprets, by the name carried in the node's arg;
        # the reference for every backend's kernel of the same name
        self.custom: dict[str, Callable[[list[np.ndarray], tuple], np.ndarray]] = {"sdpa": sdpa_kernel}

    def has_custom(self, name: str) -> bool:
        return name in self.custom

    def execute(self, sinks: list[Node]) -> list[Buffer]:
        """Compute every sink; return one buffer of result bytes per sink.

        ASSIGN writes are deferred until every sink's value is computed, so all reads in the
        batch see pre-assign bytes (an optimizer step reads old params, then commits).
        """
        from limn.ops import topological

        results = {node: self.compute(node) for node in topological(sinks)}
        assigned = [node for node in results if node.op is Op.ASSIGN]
        # a value can be a live view of a buffer this batch overwrites (BUFFER reads hand back the
        # bytes themselves), so take every value before writing any of them
        writes = [(node.srcs[0].arg, results[node].copy()) for node in assigned]
        for buf, value in writes:
            self.copyin(buf, value)
        if assigned:  # every cached value is suspect now: it may have read bytes just overwritten
            self.cache.clear()
        out: list[Buffer] = []
        for sink in sinks:
            if sink.op is Op.BUFFER:
                out.append(sink.arg)
            elif sink.op is Op.ASSIGN:
                out.append(sink.srcs[0].arg)
            else:
                buf = self.alloc(results[sink].nbytes)
                self.copyin(buf, results[sink])
                out.append(buf)
        return out

    def compute(self, node: Node) -> np.ndarray:
        """One numpy call for one node. Called in topo order, so srcs are already in the cache."""
        if node.op is Op.BUFFER:
            return node.arg.view(NUMPY_DTYPES[node.dtype]).reshape(node.shape)
        cached = self.cache.get(node)
        if cached is not None:
            return cached
        srcs = [self.compute(s) for s in node.srcs]
        match node.op:
            case Op.CONST:
                result = np.full((), node.arg, dtype=NUMPY_DTYPES[node.dtype])
            case Op.VIEW:
                view: View = node.arg
                result = view.materialize(srcs[0].reshape(-1))
            case Op.NEG:
                result = -srcs[0]
            case Op.EXP:
                result = np.exp(srcs[0])
            case Op.LOG:
                result = np.log(srcs[0])
            case Op.SQRT:
                result = np.sqrt(srcs[0])
            case Op.RECIP:
                result = 1 / srcs[0]
            case Op.CAST:
                result = srcs[0].astype(NUMPY_DTYPES[node.dtype])
            case Op.ADD:
                result = srcs[0] + srcs[1]
            case Op.MUL:
                result = srcs[0] * srcs[1]
            case Op.CMPLT:
                result = (srcs[0] < srcs[1]).astype(NUMPY_DTYPES[node.dtype])
            case Op.WHERE:
                result = np.where(srcs[0] != 0, srcs[1], srcs[2])
            case Op.SUM:  # the running total is wider than float16; the astype below rounds it back
                result = srcs[0].sum(axis=node.arg, keepdims=True, dtype=NUMPY_DTYPES[accumulate_in(node.dtype)])
            case Op.MAX:
                result = srcs[0].max(axis=node.arg, keepdims=True)
            case Op.GATHER:
                result = srcs[0][srcs[1]]
            case Op.SCATTER:
                result = np.zeros(node.shape, dtype=NUMPY_DTYPES[node.dtype])
                np.add.at(result, srcs[0], srcs[1])  # buffered, so repeated indices accumulate
            case Op.CONTIGUOUS:
                result = np.ascontiguousarray(srcs[0])
            case Op.CUSTOM:
                name = node.arg[0]
                kernel = self.custom.get(name)
                if kernel is None:
                    raise NotImplementedError(f"the numpy device has no rule for custom op {name!r}")
                result = kernel(srcs, node.arg)
            case Op.ASSIGN:
                result = srcs[1]
            case _:
                raise NotImplementedError(f"NumpyDevice has no rule for {node.op}")
        result = result.astype(NUMPY_DTYPES[node.dtype], copy=False)
        assert result.shape == node.shape, f"{node.op.name}: interpreter produced {result.shape}, graph says {node.shape}"
        self.cache[node] = result
        return result


def _numpy_device() -> Device:
    return NumpyDevice()


def _c_device() -> Device:
    from limn.backend_c import CDevice, has_cc

    if not has_cc():
        raise RuntimeError("the 'c' device needs a C compiler (cc) on PATH")
    return CDevice()


def _cuda_device() -> Device:
    from limn.backend_cuda import CudaDevice, cuda_unavailable

    reason = cuda_unavailable()
    if reason is not None:
        raise RuntimeError(f"the 'cuda' device is unavailable: {reason}")
    return CudaDevice()


DEVICES: dict[str, Callable[[], Device]] = {"numpy": _numpy_device, "c": _c_device, "cuda": _cuda_device}

active_device: Device = NumpyDevice()


def active() -> Device:
    return active_device


def set_device(name: str) -> None:
    """Point every graph built from here on at a different backend.

    The host devices share bytes, so tensors move freely between them; the cuda device uploads
    any host buffer it is handed and writes assigns back, but its own buffers live in device
    memory and only it can read them, so tensors created under cuda stay there. Backends are
    imported inside their factories because they are written against this module, and each is
    checked for its toolchain now instead of failing later inside a subprocess.
    """
    global active_device
    if name not in DEVICES:
        raise ValueError(f"unknown device {name!r}; limn has {tuple(DEVICES)}")
    active_device = DEVICES[name]()
