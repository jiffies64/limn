"""CUDA backend: emit CUDA C from the loop nest IR, compile with NVRTC, launch via the driver API.

Everything is reached through ctypes, so there is no build step and no pinned CUDA version.
libcuda ships with the display driver and NVRTC comes from a CUDA toolkit or from the
nvidia-cuda-nvrtc wheel, whichever loads first. Kernels compile to PTX for the newest
architecture this NVRTC supports that does not exceed the device's; PTX for an older
architecture still JITs onto a newer GPU, so an old toolkit serves a new card, and a driver
error names the one combination that cannot work (a toolkit newer than the driver).

The thread mapping: one thread per point of the non-reduce dims, reduce loops run sequentially
inside it. Every output cell then belongs to exactly one thread, so STORE and ACCUM need no
atomics, and a cell folds its elements in ascending reduce order, matching the C backend
exactly. The innermost non-reduce dim (picked by loop_order for being stride-1 in the most
buffers) becomes the fastest-varying thread index, which is what makes warp loads coalesce.
SCATTER is the one racing write, since two values can name the same row, so it adds with
atomicAdd. A large reduce over few output cells is the exception to one-thread-per-cell: it
runs as two kernels, strided partial accumulators and then their fold, with a fixed grouping
that keeps it deterministic but matches the host backends to rounding rather than bit for bit.

Launches are grid-stride loops over at most GRID blocks of BLOCK threads, so any size runs a
bounded launch, and execute() synchronizes per batch, which keeps the queue short and pins a
failure to the batch that launched it.

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
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from limn.backend_c import C_TYPE, arith_c, c_literal, fold_c, guard
from limn.codegen import REDUCES, Instr, LoopNest, Opcode, reduce_axes
from limn.device import Buffer
from limn.jit import CompiledDevice, Runner

BLOCK = 256
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
    message = ctypes.c_char_p()
    api = driver()
    if api is not None:
        api.cuGetErrorString(result, ctypes.byref(message))
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


# ---- emission: a second rendering of the Instr stream, like emit_function in backend_c ----

PRELUDE = """\
typedef int int32_t;
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fc00000))
#endif
"""


def outer_extent(nest: LoopNest) -> int:
    """How many points the non-reduce dims span: one thread per point."""
    axes = reduce_axes(nest)
    return math.prod(size for d, size in enumerate(nest.space) if d not in axes)


def reduce_extent(nest: LoopNest) -> int:
    axes = reduce_axes(nest)
    return math.prod(nest.space[d] for d in axes)


SPLIT_MIN = 4096  # a reduce at least this long, over fewer than this many cells, gets a second stage


def split_partials(nest: LoopNest) -> int:
    """How many partial accumulators a split reduce uses per cell; 0 when the nest is not split.

    Only the register form splits: there each output cell is one thread, so few cells over a
    long reduce leaves the GPU idle while single threads walk millions of elements. The ACCUM
    form already spreads its work across the non-reduce dims. Partials are capped so a shorter
    reduce still hands every partial a real chunk.
    """
    if not any(instr.opcode is Opcode.ACC for instr in nest.instrs):
        return 0
    r = reduce_extent(nest)
    if r < SPLIT_MIN or outer_extent(nest) >= SPLIT_MIN:
        return 0
    return min(4096, r // 64)


def value_line(instr: Instr, indent: str) -> str:
    """One value-producing instruction as a C declaration; shared by both kernel forms."""
    match instr.opcode:
        case Opcode.CONST:
            value, valid = instr.arg
            pre, post = guard(valid, instr.value_type)
            return f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = {pre}{c_literal(value, instr.value_type)}{post};"
        case Opcode.LOAD:
            buf, index, valid = instr.arg
            pre, post = guard(valid, instr.value_type)
            return f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = {pre}{buf}[{index.render()}]{post};"
        case Opcode.ARITH:
            return f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = {arith_c(instr.arg, list(instr.srcs), instr.value_type)};"
        case Opcode.CAST:
            return f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = ({C_TYPE[instr.value_type]}){instr.srcs[0]};"
        case _:
            raise NotImplementedError(f"{instr.opcode} does not define a value")


def emit_split(nest: LoopNest, partials: int) -> str:
    """A register reduce as two kernels: strided partials per output cell, then their fold.

    Partial p of a cell accumulates elements p, p + partials, p + 2*partials, ... so warp
    loads stay coalesced and the grouping is fixed, which is what makes the result
    deterministic; it is grouped differently from the sequential nest, so it agrees with the
    host backends to rounding rather than bit for bit (int folds stay exact, addition being
    associative modulo 2**32).
    """
    kernel = nest.kernel
    root = kernel.ast
    fold = REDUCES[root.op].fold
    ctype = C_TYPE[root.dtype]
    reduce_vars = {f"r{d}" for d in reduce_axes(nest)}
    outer: list[tuple[str, int]] = []
    loops: list[tuple[str, int]] = []
    for instr in nest.instrs:
        if instr.opcode is Opcode.LOOP and all(var != instr.dest for var, _ in outer + loops):
            (loops if instr.dest in reduce_vars else outer).append((instr.dest, instr.arg))
    total = reduce_extent(nest)
    first = "blockIdx.x * (long long)blockDim.x + threadIdx.x"
    stride = "(long long)gridDim.x * blockDim.x"

    params = [f"const {C_TYPE[node.dtype]}* __restrict__ in{k}" for k, node in enumerate(kernel.inputs)]
    lines = [f'extern "C" __global__ void {nest.name}_part({", ".join(params + [f"{ctype}* __restrict__ out"])}) {{']
    lines.append(f"  for (long long gid = {first}; gid < {outer_extent(nest) * partials}LL; gid += {stride}) {{")
    lines.append(f"    const int p = (int)(gid % {partials});")
    lines.append(f"    long long t = gid / {partials};")
    for var, bound in reversed(outer[1:]):
        lines.append(f"    const int {var} = (int)(t % {bound}); t /= {bound};")
    if outer:
        lines.append(f"    const int {outer[0][0]} = (int)t;")
    for instr in nest.instrs:
        match instr.opcode:
            case Opcode.LOOP | Opcode.ENDLOOP:
                continue
            case Opcode.ACC:
                lines.append(f"    {ctype} {instr.dest} = {c_literal(instr.arg, instr.value_type)};")
                lines.append(f"    for (long long j = p; j < {total}LL; j += {partials}) {{")
                lines.append("      long long u = j;")
                for var, bound in reversed(loops[1:]):
                    lines.append(f"      const int {var} = (int)(u % {bound}); u /= {bound};")
                lines.append(f"      const int {loops[0][0]} = (int)u;")
            case Opcode.UPDATE:
                lines.append("      " + fold_c(instr.arg, instr.dest, instr.srcs[0]))
            case Opcode.STORE:
                lines.append("    }")
                lines.append(f"    out[(gid / {partials}) * {partials} + p] = acc;")
            case _:
                lines.append(value_line(instr, "      "))
    lines.append("  }")
    lines.append("}")

    lines.append("")
    lines.append(f'extern "C" __global__ void {nest.name}(const {ctype}* __restrict__ in0, {ctype}* __restrict__ out) {{')
    lines.append(f"  for (long long gid = {first}; gid < {outer_extent(nest)}LL; gid += {stride}) {{")
    lines.append(f"    {ctype} acc = {c_literal(REDUCES[root.op].identity[root.dtype], root.dtype)};")
    lines.append(f"    for (int p = 0; p < {partials}; p++) {{")
    lines.append("      " + fold_c(fold, "acc", f"in0[gid * {partials} + p]"))
    lines.append("    }")
    lines.append("    out[gid] = acc;")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def emit_kernel(nest: LoopNest) -> str:
    kernel = nest.kernel
    reduce_vars = {f"r{d}" for d in reduce_axes(nest)}
    outer: list[tuple[str, int]] = []  # non-reduce loop vars in nesting order, outermost first
    for instr in nest.instrs:
        if instr.opcode is Opcode.LOOP and instr.dest not in reduce_vars and all(var != instr.dest for var, _ in outer):
            outer.append((instr.dest, instr.arg))

    params = [f"const {C_TYPE[node.dtype]}* __restrict__ in{k}" for k, node in enumerate(kernel.inputs)]
    params.append(f"{C_TYPE[kernel.target.dtype]}* __restrict__ out")
    # restrict on the inputs is safe even when two of them are the same buffer (a @ a.transpose()):
    # it only promises that nothing *written* is reachable through another name, and only out is
    # written, to memory out_alloc freshly allocated
    lines = [f'extern "C" __global__ void {nest.name}({", ".join(params)}) {{']
    stride = "(long long)gridDim.x * blockDim.x"
    first = "blockIdx.x * (long long)blockDim.x + threadIdx.x"
    lines.append(f"  for (long long gid = {first}; gid < {outer_extent(nest)}LL; gid += {stride}) {{")

    # bind the non-reduce dims from gid, the innermost one varying fastest: neighbouring threads
    # then touch neighbouring elements of whatever loop_order found to be stride-1, and coalesce
    if len(outer) == 1:
        lines.append(f"    const int {outer[0][0]} = (int)gid;")
    elif outer:
        lines.append("    long long t = gid;")
        for var, bound in reversed(outer[1:]):
            lines.append(f"    const int {var} = (int)(t % {bound}); t /= {bound};")
        lines.append(f"    const int {outer[0][0]} = (int)t;")

    depth = 2
    for instr in nest.instrs:
        if instr.opcode is Opcode.LOOP:
            if instr.dest in reduce_vars:
                lines.append("  " * depth + f"for (int {instr.dest} = 0; {instr.dest} < {instr.arg}; {instr.dest}++) {{")
                depth += 1
            continue
        if instr.opcode is Opcode.ENDLOOP:
            if instr.arg in reduce_vars:
                depth -= 1
                lines.append("  " * depth + "}")
            continue
        indent = "  " * depth
        match instr.opcode:
            case Opcode.ACC:
                lines.append(f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = {c_literal(instr.arg, instr.value_type)};")
            case Opcode.CONST | Opcode.LOAD | Opcode.ARITH | Opcode.CAST:
                lines.append(value_line(instr, indent))
            case Opcode.UPDATE:
                lines.append(indent + fold_c(instr.arg, instr.dest, instr.srcs[0]))
            case Opcode.STORE:
                buf, index = instr.arg
                lines.append(f"{indent}{buf}[{index.render()}] = {instr.srcs[0]};")
            case Opcode.ACCUM:
                # no atomic: the cell's index uses only thread-bound vars, so one thread owns it
                buf, index, fold = instr.arg
                lines.append(indent + fold_c(fold, f"{buf}[{index.render()}]", instr.srcs[0]))
            case Opcode.GATHER:
                buf, index = instr.arg
                lines.append(f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = {buf}[{index.render()}];")
            case Opcode.SCATTER:
                buf, index = instr.arg  # threads collide wherever indices repeat a row
                lines.append(f"{indent}atomicAdd(&{buf}[{index.render()}], {instr.srcs[0]});")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def emit_cuda(nests: list[LoopNest]) -> str:
    kernels = [emit_split(nest, p) if (p := split_partials(nest)) else emit_kernel(nest) for nest in nests]
    return "\n".join([PRELUDE] + [kernel + "\n" for kernel in kernels])


# ---- compilation ----


@dataclass(frozen=True)
class Program:
    module: ctypes.c_void_p
    functions: dict[str, ctypes.c_void_p]


cache: dict[str, Program] = {}


def compile_cuda(source: str, arch: int, kernel_names: list[str]) -> Program:
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
    program = Program(module, functions)
    cache[key] = program
    return program


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
        buf = self._alloc(nbytes)
        check(self.api.cuMemsetD8(buf.ptr, 0, nbytes), "zeroing an allocation")  # HostDevice zeroes too
        return buf

    def copyin(self, buf: Buffer, array: np.ndarray) -> None:
        flat = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        check(self.api.cuMemcpyHtoD(buf.ptr, flat.ctypes.data, flat.nbytes), "uploading bytes")

    def copyout(self, buf: Buffer) -> np.ndarray:
        out = np.empty(buf.nbytes, dtype=np.uint8)
        check(self.api.cuMemcpyDtoH(out.ctypes.data, buf.ptr, buf.nbytes), "downloading bytes")
        return out

    # ---- CompiledDevice hooks ----

    def prepare(self, buf: Buffer) -> Buffer:
        if isinstance(buf, np.ndarray):  # a tensor created while a host device was active
            device_buf = self._alloc(buf.nbytes)
            check(self.api.cuMemcpyHtoD(device_buf.ptr, buf.ctypes.data, buf.nbytes), "uploading a host buffer")
            return device_buf
        return buf

    def out_alloc(self, nb: int, zero: bool) -> Buffer:
        buf = self._alloc(nb)
        if zero:
            check(self.api.cuMemsetD8(buf.ptr, 0, nb), "zeroing a scatter output")
        return buf

    def commit(self, target: Buffer, value: Buffer) -> None:
        if isinstance(target, np.ndarray):  # an assign to a tensor whose bytes live on the host
            check(self.api.cuMemcpyDtoH(target.ctypes.data, value.ptr, target.nbytes), "committing an assign to host")
        else:
            check(self.api.cuMemcpyDtoD(target.ptr, value.ptr, value.nbytes), "committing an assign")

    def finish(self) -> None:
        check(self.api.cuCtxSynchronize(), "waiting for the batch")

    def runners(self, nests: list[LoopNest]) -> list[Runner]:
        if not nests:
            return []
        names = [name for nest in nests for name in ([nest.name + "_part"] if split_partials(nest) else []) + [nest.name]]
        program = compile_cuda(emit_cuda(nests), self.arch, names)
        out: list[Runner] = []
        for nest in nests:
            outer = outer_extent(nest)
            partials = split_partials(nest)
            final = self._launcher(program.functions[nest.name], self._grid(outer))
            if partials:
                part = self._launcher(program.functions[nest.name + "_part"], self._grid(outer * partials))
                out.append(self._split_runner(part, final, outer * partials * nest.kernel.target.dtype.itemsize))
            else:
                out.append(final)
        return out

    @staticmethod
    def _grid(threads: int) -> int:
        return min((threads + BLOCK - 1) // BLOCK, GRID)

    def _split_runner(self, part: Runner, final: Runner, partials_nbytes: int) -> Runner:
        def run(inputs: list[Buffer], out: Buffer) -> None:
            partials = self._alloc(partials_nbytes)
            part(inputs, partials)
            final([partials], out)

        return run

    def _launcher(self, fn: ctypes.c_void_p, grid: int) -> Runner:
        def run(inputs: list[Buffer], out: Buffer) -> None:
            ptrs = (CUdeviceptr * (len(inputs) + 1))(*[b.ptr for b in inputs], out.ptr)
            params = (ctypes.c_void_p * len(ptrs))(*[ctypes.addressof(ptrs) + i * 8 for i in range(len(ptrs))])
            check(self.api.cuLaunchKernel(fn, grid, 1, 1, BLOCK, 1, 1, 0, None, params, None), "launching a kernel")

        return run
