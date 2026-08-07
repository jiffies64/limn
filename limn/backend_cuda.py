"""CUDA device: bind the driver and NVRTC through ctypes, compile the emitted kernels, launch them.

Everything is reached through ctypes, so there is no build step and no pinned CUDA version.
libcuda ships with the display driver and NVRTC comes from a CUDA toolkit or from the
nvidia-cuda-nvrtc wheel, whichever loads first. Kernels compile to PTX for the newest
architecture this NVRTC supports that does not exceed the device's; PTX for an older
architecture still JITs onto a newer GPU, so an old toolkit serves a new card, and a driver
error names the one combination that cannot work (a toolkit newer than the driver).

cuda_emit.py renders the kernels; what is left here is the memory and the launching. Launches
are grid-stride loops over at most GRID blocks of BLOCK threads, so any size runs a bounded
launch, and execute() synchronizes per batch, which keeps the queue short and pins a failure to
the batch that launched it.

A host numpy buffer handed to this device (a tensor created while a host device was active)
is uploaded per batch by prepare(), and an assign whose target is host gets the new bytes
copied back. The reverse is not supported: buffers this device allocates live in device
memory, and only this device knows how to read them.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import glob
import hashlib
import importlib.util
from collections.abc import Callable

import numpy as np

from limn.codegen import LoopNest
from limn.cuda_emit import (
    BLOCK,
    SDPA_BLOCK,
    SDPA_KERNELS,
    emit_cuda,
    outer_extent,
    part_name,
    split_partials,
    split_scratch,
    tile_count,
    tiled,
)
from limn.device import Buffer
from limn.jit import CompiledDevice, Runner
from limn.ops import Node

GRID = 4096  # most blocks per launch; the grid-stride loop covers whatever is left

CUdeviceptr = ctypes.c_uint64

DRIVER = {
    "cuInit": [ctypes.c_uint],
    "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
    "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    "cuDeviceGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
    "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
    "cuCtxSetCurrent": [ctypes.c_void_p],
    "cuCtxSynchronize": [],
    "cuModuleLoadData": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p],
    "cuModuleGetFunction": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
    "cuLaunchKernel": [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3,
    "cuMemAlloc": [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t],
    "cuMemFree": [CUdeviceptr],
    "cuMemcpyHtoD": [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t],
    "cuMemcpyDtoH": [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t],
    "cuMemcpyDtoD": [CUdeviceptr, CUdeviceptr, ctypes.c_size_t],
    "cuMemsetD8": [CUdeviceptr, ctypes.c_ubyte, ctypes.c_size_t],
    "cuGetErrorString": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
}

# For these six, the unsuffixed symbol is the pre-CUDA-3.2 32-bit ABI kept for old binaries and
# the _v2 is the modern one with the signatures bound above. Nothing else may resolve _v2-first:
# newer drivers export _v2 variants of *context* functions too (cuCtxSynchronize_v2 takes an
# explicit CUcontext), and binding one of those against the plain signature passes garbage.
V2_ABI = frozenset({"cuMemAlloc", "cuMemFree", "cuMemcpyHtoD", "cuMemcpyDtoH", "cuMemcpyDtoD", "cuMemsetD8"})

NVRTC = {
    "nvrtcCreateProgram": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    + [ctypes.c_void_p] * 2,
    "nvrtcCompileProgram": [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    "nvrtcGetProgramLogSize": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)],
    "nvrtcGetProgramLog": [ctypes.c_void_p, ctypes.c_char_p],
    "nvrtcGetPTXSize": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)],
    "nvrtcGetPTX": [ctypes.c_void_p, ctypes.c_char_p],
    "nvrtcDestroyProgram": [ctypes.POINTER(ctypes.c_void_p)],
}


# present only in NVRTC >= 11.2; bound when the library has them, and pick_arch copes when not
NVRTC_OPTIONAL = {
    "nvrtcGetNumSupportedArchs": [ctypes.POINTER(ctypes.c_int)],
    "nvrtcGetSupportedArchs": [ctypes.POINTER(ctypes.c_int)],
}


class Lib:
    """Bound functions of one shared library; every call returns a status int checked by check()."""

    def __init__(self, lib: ctypes.CDLL, signatures: dict[str, list], versioned: frozenset[str], optional: dict[str, list] = {}):
        for name, argtypes in (signatures | optional).items():
            fn = None
            for candidate in (name + "_v2", name) if name in versioned else (name,):
                try:
                    fn = getattr(lib, candidate)
                    break
                except AttributeError:
                    continue
            if fn is None:
                if name in optional:
                    continue
                raise RuntimeError(f"missing symbol {name}")
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
            setattr(self, name, fn)

    def __getattr__(self, name: str) -> Callable[..., int]:
        raise AttributeError(name)  # bound names are instance attributes; this types the rest


def _load(
    paths: list[str | None], signatures: dict[str, list], versioned: frozenset[str], optional: dict[str, list] = {}
) -> Lib | None:
    for path in paths:
        if not path:
            continue
        try:
            return Lib(ctypes.CDLL(path), signatures, versioned, optional)
        except (OSError, RuntimeError):
            continue
    return None


@functools.cache
def driver() -> Lib | None:
    paths = [ctypes.util.find_library("cuda"), "libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1"]
    return _load(paths, DRIVER, versioned=V2_ABI)


@functools.cache
def nvrtc() -> Lib | None:
    paths = [ctypes.util.find_library("nvrtc"), "libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"]
    paths += sorted(glob.glob("/usr/local/cuda*/lib64/libnvrtc.so.*"), reverse=True)
    paths += sorted(glob.glob("/usr/local/cuda*/targets/*/lib/libnvrtc.so.*"), reverse=True)
    spec = importlib.util.find_spec("nvidia")  # the nvidia-cuda-nvrtc wheel, when installed
    if spec is not None:
        for location in spec.submodule_search_locations or []:
            paths += sorted(glob.glob(f"{location}/*/lib/libnvrtc.so*"), reverse=True)
    return _load(paths, NVRTC, versioned=frozenset(), optional=NVRTC_OPTIONAL)


def check(result: int, doing: str) -> None:
    if result == 0:
        return
    api = driver()
    assert api is not None, "a driver call produced this result"
    message = ctypes.c_char_p()
    api.cuGetErrorString(result, ctypes.byref(message))  # leaves message NULL for a code it does not know
    detail = message.value.decode() if message.value else f"error {result}"
    raise RuntimeError(f"cuda: {doing}: {detail}")


@functools.cache
def cuda_unavailable() -> str | None:
    """None when the cuda device can run here, else the reason it cannot."""
    api = driver()
    if api is None:
        return "libcuda not found (it ships with the NVIDIA driver)"
    if api.cuInit(0) != 0:
        return "cuInit failed (no usable NVIDIA driver?)"
    count = ctypes.c_int()
    if api.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value == 0:
        return "no CUDA device found"
    if nvrtc() is None:
        return "libnvrtc not found (install a CUDA toolkit, or the nvidia-cuda-nvrtc wheel: uv sync --extra cuda)"
    return None


def has_cuda() -> bool:
    return cuda_unavailable() is None


# ---- compilation ----


cache: dict[str, dict[str, ctypes.c_void_p]] = {}


def compile_cuda(source: str, arch: int, kernel_names: list[str]) -> dict[str, ctypes.c_void_p]:
    """PTX-compile this source for the arch and resolve the named kernels to function handles."""
    key = hashlib.sha256(f"compute_{arch}:{source}".encode()).hexdigest()
    if key in cache:
        return cache[key]
    api, nv = driver(), nvrtc()
    assert api is not None and nv is not None
    prog = ctypes.c_void_p()
    if nv.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), b"limn.cu", 0, None, None) != 0:
        raise RuntimeError("nvrtc: could not create a program")
    options = (ctypes.c_char_p * 1)(f"--gpu-architecture=compute_{arch}".encode())
    failed = nv.nvrtcCompileProgram(prog, 1, options) != 0
    if failed:
        size = ctypes.c_size_t()
        nv.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        nv.nvrtcGetProgramLog(prog, log)
        nv.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"nvrtc failed:\n{log.value.decode()}")
    size = ctypes.c_size_t()
    nv.nvrtcGetPTXSize(prog, ctypes.byref(size))
    ptx = ctypes.create_string_buffer(size.value)
    nv.nvrtcGetPTX(prog, ptx)
    nv.nvrtcDestroyProgram(ctypes.byref(prog))

    module = ctypes.c_void_p()
    check(api.cuModuleLoadData(ctypes.byref(module), ptx), "loading PTX (a driver older than the toolkit cannot JIT its PTX)")
    functions: dict[str, ctypes.c_void_p] = {}
    for name in kernel_names:
        fn = ctypes.c_void_p()
        check(api.cuModuleGetFunction(ctypes.byref(fn), module, name.encode()), f"resolving kernel {name}")
        functions[name] = fn
    cache[key] = functions  # the module handle is dropped: it stays loaded in the context for the process, like the plans
    return functions


# ---- the device ----


class CudaBuffer:
    """Bytes in device memory, recycled through the owning device's pool when the last reference drops.

    cuMemAlloc and cuMemFree are slow enough to dominate a training step, which retires one
    output buffer per kernel, so freed pointers go back to a per-size free list instead of
    the driver. The pool therefore holds device memory at its high-water mark per size;
    trim() gives it back, and allocation failure trims and retries before giving up.
    """

    __slots__ = ("ptr", "nbytes", "pool")

    def __init__(self, ptr: int, nbytes: int, pool: dict[int, list[int]]):
        self.ptr = ptr
        self.nbytes = nbytes
        self.pool = pool

    def __del__(self) -> None:
        try:
            self.pool.setdefault(self.nbytes, []).append(self.ptr)
        except Exception:  # interpreter teardown can have unloaded anything by now
            pass


def pick_arch(cc: int, nv: Lib) -> int:
    """The newest architecture this NVRTC supports that does not exceed the device's capability.

    nvrtcGetSupportedArchs is missing before CUDA 11.2; there the device's own capability is
    the only sensible guess. A device older than everything NVRTC supports gets the oldest
    target NVRTC has, and the driver rejects the PTX with a clear error if even that is too new.
    """
    if not hasattr(nv, "nvrtcGetSupportedArchs"):
        return cc
    count = ctypes.c_int()
    if nv.nvrtcGetNumSupportedArchs(ctypes.byref(count)) != 0 or count.value == 0:
        return cc
    archs = (ctypes.c_int * count.value)()
    if nv.nvrtcGetSupportedArchs(archs) != 0:
        return cc
    fits = [a for a in archs if a <= cc]
    return max(fits) if fits else min(archs)


class CudaDevice(CompiledDevice):
    """Runs op graphs on an NVIDIA GPU; construct through set_device("cuda"), which checks first."""

    def __init__(self) -> None:
        super().__init__()
        api = driver()
        nv = nvrtc()
        assert api is not None and nv is not None, "construct via set_device('cuda'), which explains what is missing"
        self.api = api
        check(api.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        check(api.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        major, minor = ctypes.c_int(), ctypes.c_int()
        check(api.cuDeviceGetAttribute(ctypes.byref(major), 75, dev), "querying compute capability")
        check(api.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev), "querying compute capability")
        self.arch = pick_arch(major.value * 10 + minor.value, nv)
        self.pool: dict[int, list[int]] = {}
        context = ctypes.c_void_p()
        check(api.cuDevicePrimaryCtxRetain(ctypes.byref(context), dev), "retaining the primary context")
        check(api.cuCtxSetCurrent(context), "making the context current")
        for name in SDPA_KERNELS:
            self.custom[name] = self._sdpa_runner

    # ---- Device protocol: raw byte buffers ----

    def _alloc(self, nb: int) -> CudaBuffer:
        recycled = self.pool.get(nb)
        if recycled:
            return CudaBuffer(recycled.pop(), nb, self.pool)
        ptr = CUdeviceptr()
        result = self.api.cuMemAlloc(ctypes.byref(ptr), nb)
        if result != 0 and self.pool:
            self.trim()  # the pool may be hoarding what this allocation needs
            result = self.api.cuMemAlloc(ctypes.byref(ptr), nb)
        check(result, f"allocating {nb} bytes")
        return CudaBuffer(ptr.value, nb, self.pool)

    def trim(self) -> None:
        """Return every pooled buffer to the driver."""
        for pointers in self.pool.values():
            for ptr in pointers:
                self.api.cuMemFree(ptr)
        self.pool.clear()

    def alloc(self, nbytes: int) -> Buffer:
        return self.out_alloc(nbytes, zero=True)  # HostDevice zeroes too

    def copyin(self, buf: Buffer, array: np.ndarray) -> None:
        data = np.ascontiguousarray(array)
        check(self.api.cuMemcpyHtoD(buf.ptr, data.ctypes.data, data.nbytes), "uploading bytes")

    def copyout(self, buf: Buffer) -> np.ndarray:
        out = np.empty(buf.nbytes, dtype=np.uint8)
        check(self.api.cuMemcpyDtoH(out.ctypes.data, buf.ptr, buf.nbytes), "downloading bytes")
        return out

    # ---- CompiledDevice hooks ----

    def prepare(self, buf: Buffer) -> Buffer:
        if isinstance(buf, np.ndarray):  # a tensor created while a host device was active
            device_buf = self._alloc(buf.nbytes)
            self.copyin(device_buf, buf)
            return device_buf
        return buf

    def out_alloc(self, nb: int, zero: bool) -> Buffer:
        buf = self._alloc(nb)
        if zero:
            check(self.api.cuMemsetD8(buf.ptr, 0, nb), "zeroing an allocation")
        return buf

    def commit(self, target: Buffer, value: Buffer) -> None:
        if isinstance(target, np.ndarray):  # an assign to a tensor whose bytes live on the host
            check(self.api.cuMemcpyDtoH(target.ctypes.data, value.ptr, target.nbytes), "committing an assign to host")
        else:
            check(self.api.cuMemcpyDtoD(target.ptr, value.ptr, value.nbytes), "committing an assign")

    def finish(self) -> None:
        check(self.api.cuCtxSynchronize(), "waiting for the batch")

    def _sdpa_runner(self, node: Node) -> Runner:
        """A fused-attention kernel is compiled here and launched like any other, so a plan
        or a capture treats its call exactly like a lowered nest's. The three of them differ
        only in their source and in how many blocks cover the rows they hand out."""
        name = node.arg.name
        emit, grid = SDPA_KERNELS[name]
        functions = compile_cuda(emit(node), self.arch, [name])
        # one block per (batch, chunk of rows), one thread per row; the kernel decodes blockIdx
        # the same way, and its rows-per-block is the narrower SDPA_BLOCK, not a nest's BLOCK
        return self._launcher(functions[name], grid(node), len(node.srcs), len(node.arg.outs), SDPA_BLOCK)

    def runners(self, nests: list[LoopNest]) -> list[Runner]:
        splits = [split_partials(nest) for nest in nests]
        names = [name for nest, p in zip(nests, splits, strict=True) for name in ([part_name(nest)] if p else []) + [nest.name]]
        functions = compile_cuda(emit_cuda(nests), self.arch, names)
        out: list[Runner] = []
        for nest, partials in zip(nests, splits, strict=True):
            cells = outer_extent(nest)
            nin = len(nest.kernel.inputs)
            if partials:
                part = self._launcher(functions[part_name(nest)], self._grid(cells * partials), nin, 1)
                final = self._launcher(functions[nest.name], self._grid(cells), 1, 1)
                out.append(self._split_runner(part, final, split_scratch(nest, partials)))
            elif plan := tiled(nest):
                # a tiled kernel's unit of work is a tile, not a cell, and its block is LANES^2 threads
                out.append(self._launcher(functions[nest.name], min(tile_count(*plan), GRID), nin, 1))
            else:
                out.append(self._launcher(functions[nest.name], self._grid(cells), nin, 1))
        return out

    @staticmethod
    def _grid(threads: int) -> int:
        return min((threads + BLOCK - 1) // BLOCK, GRID)

    def _split_runner(self, part: Runner, final: Runner, partials_nbytes: int) -> Runner:
        # one scratch buffer serves every call: the stream orders each final-read before the next
        # part-write. Taken on the first call, so a plan that is compiled but never run holds none.
        scratch: list[Buffer] = []

        def run(inputs: list[Buffer], outs: list[Buffer]) -> None:
            if not scratch:
                scratch.append(self._alloc(partials_nbytes))
            part(inputs, scratch)  # the one scratch buffer is the partial stage's output and the fold's input
            final(scratch, outs)

        return run

    def _launcher(self, fn: ctypes.c_void_p, grid: int, nin: int, nout: int, block: int = BLOCK) -> Runner:
        launch = self.api.cuLaunchKernel
        # the param arrays are built once and refilled per call, which is safe because
        # cuLaunchKernel copies the pointed-to arguments before it returns
        nargs = nin + nout
        ptrs = (CUdeviceptr * nargs)()
        size = ctypes.sizeof(CUdeviceptr)
        params = (ctypes.c_void_p * nargs)(*[ctypes.addressof(ptrs) + k * size for k in range(nargs)])

        def run(inputs: list[Buffer], outs: list[Buffer]) -> None:
            for k, buf in enumerate(inputs):
                ptrs[k] = buf.ptr
            for k, buf in enumerate(outs):
                ptrs[nin + k] = buf.ptr
            check(launch(fn, grid, 1, 1, block, 1, 1, 0, None, params, None), "launching a kernel")

        return run
