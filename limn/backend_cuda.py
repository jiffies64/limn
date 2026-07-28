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

One thread per cell is the wrong shape for a matmul, which re-reads a whole row and a whole
column per cell and gets no reuse out of either. A nest that reads as a matmul (matmul_shape
decides, and a Conv2d tap reads as one) instead gives a block a tile of the output, staging both
operands through shared memory a step of the reduce axis at a time and holding a patch of cells
per thread in registers. It stays bit-identical to the kernel it replaces, since the reduce axis
is still walked in order: what changes is where the operands are read from, not what is summed.

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from limn.backend_c import C_TYPE, arith_c, c_literal, fold_c, guard
from limn.codegen import REDUCES, Index, Instr, LoopNest, Opcode, reduce_axes
from limn.device import Buffer
from limn.jit import CompiledDevice, Runner
from limn.ops import float32

BLOCK = 256
GRID = 4096  # most blocks per launch; the grid-stride loop covers whatever is left

LANES = 16  # threads across a tile in each direction, so one block is LANES * LANES = BLOCK threads
TILE_K = 16  # how far down the reduce axis one staged step reaches
TILE_MIN = 1 << 16  # a matmul with fewer multiply-adds than this is not worth staging for
TILE_PAD = 4  # slack on a staged row, so a column of it spreads over banks instead of piling on one
VECTOR = {float32: "float4"}  # the four-wide load type, for the dtypes that have one

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


@dataclass(frozen=True)
class Matmul:
    """A reduce nest read as a matmul: two sides of loop variables that meet only in the reduce axis.

    Every kernel that multiplies two things and sums has this shape, whatever the tensor API called
    it. What makes it one is how the loads are indexed: each side's loads move with their own loop
    variables and with the reduce variable, and with nothing the other side reads. That is exactly
    the condition for a staged tile to be reused, since a tile of one side is then the same for
    every point of the other. `a @ b` sends one variable to each side; a Conv2d tap sends the
    output channels one way and the batch and spatial dims the other, over the input channels.

    Variables both sides read (a batched matmul's batch dim, a grouped convolution's groups) belong
    to neither: they select which matmul, so each of their points gets its own tiles. Sides can hold
    several variables, which is why the emitter works in fused row and column indices and unpacks
    them back into loop variables wherever it needs an address.
    """

    rows: tuple[str, ...]  # the loop vars only one side's loads read; their extents multiply to M
    cols: tuple[str, ...]  # the other side's, multiplying to N, and holding the output's fastest dim
    batch: tuple[str, ...]  # vars both sides read or neither does: one matmul per point of them
    depth: str  # the reduce variable, whose extent is K
    sizes: Mapping[str, int]
    staged: tuple[Instr, ...]  # the loads to stage in shared memory, in emission order
    body: tuple[Instr, ...]  # every value instruction the fold consumes, including those loads
    fold: Instr  # the UPDATE or ACCUM that folds the body into the running total
    out: Index

    def extent(self, group: Sequence[str]) -> int:
        return math.prod(self.sizes[var] for var in group)

    def on_cols(self, load: Instr) -> bool:
        return bool(index_vars(load) & set(self.cols))


def loop_bounds(nest: LoopNest) -> dict[str, int]:
    """Every loop variable of this nest and its bound, named the way lower() names them."""
    axes = reduce_axes(nest)
    return {f"{'r' if d in axes else 'i'}{d}": size for d, size in enumerate(nest.space)}


def index_vars(load: Instr) -> frozenset[str]:
    """The loop variables a load's address and its mask depend on."""
    _, index, valid = load.arg
    return frozenset(var for var, _ in index.terms) | frozenset(var for var, *_ in valid.bounds)


def matmul_shape(nest: LoopNest) -> Matmul | None:
    """Read this nest as a matmul, or None when it is not one.

    Deliberately strict: everything it turns down still runs, as the one-thread-per-cell kernel
    below. A load that does not move with the reduce axis, a masked constant, a second reduce axis
    and an indexed read all bail out, because each would have to be re-read or re-derived per
    output cell and none of them appears in a matmul.
    """
    axes = reduce_axes(nest)
    if len(axes) != 1:
        return None
    sizes = loop_bounds(nest)
    depth = f"r{axes[0]}"
    folds = [instr for instr in nest.instrs if instr.opcode in (Opcode.UPDATE, Opcode.ACCUM)]
    if len(folds) != 1:
        return None
    fold = folds[0]

    defined = {instr.dest: instr for instr in nest.instrs if instr.opcode not in (Opcode.LOOP, Opcode.ENDLOOP)}
    needed: set[str] = set()
    stack = [fold.srcs[0]]
    while stack:  # the fold's dependency cone, which is the whole body of a fused kernel
        dest = stack.pop()
        if dest not in needed and dest in defined:
            needed.add(dest)
            stack.extend(defined[dest].srcs)
    body = tuple(instr for instr in nest.instrs if instr.dest in needed)
    if any(instr.opcode not in (Opcode.CONST, Opcode.LOAD, Opcode.ARITH, Opcode.CAST) for instr in body):
        return None
    if any(instr.opcode is Opcode.CONST and instr.arg[1].bounds for instr in body):
        return None

    loads = [instr for instr in body if instr.opcode is Opcode.LOAD]
    staged = [load for load in loads if depth in index_vars(load)]
    if len(staged) != len(loads) or len(staged) < 2:
        return None
    everywhere = frozenset.intersection(*(index_vars(load) for load in staged))
    reads = frozenset.union(*(index_vars(load) for load in staged))
    shared = {var for var in sizes if var != depth and (var in everywhere or var not in reads)}
    sides: list[frozenset[str]] = []
    for load in staged:
        side = index_vars(load) - {depth} - shared
        if side and side not in sides:
            sides.append(side)
    if len(sides) != 2 or sides[0] & sides[1]:
        return None

    def dims(group: frozenset[str] | set[str]) -> tuple[str, ...]:
        return tuple(sorted(group, key=lambda var: int(var[1:])))

    # the side holding the nest's innermost dim owns the output's stride-1 axis, so it is the one to
    # spread across a warp: call it the columns, and the store coalesces
    rows, cols = sorted(sides, key=lambda side: max(int(var[1:]) for var in side))
    if fold.opcode is Opcode.ACCUM:
        out = fold.arg[1]
    else:
        stores = [instr for instr in nest.instrs if instr.opcode is Opcode.STORE and instr.srcs[0] == fold.dest]
        if len(stores) != 1:
            return None
        out = stores[0].arg[1]
    return Matmul(dims(rows), dims(cols), dims(shared), depth, sizes, tuple(staged), body, fold, out)


@dataclass(frozen=True)
class TileSpec:
    """One block's share of the output: `rows` by `cols` cells, held in registers across LANES^2 threads."""

    rows: int
    cols: int

    @property
    def row_regs(self) -> int:
        return self.rows // LANES

    @property
    def col_regs(self) -> int:
        return self.cols // LANES


def tiled(nest: LoopNest) -> tuple[Matmul, TileSpec] | None:
    """How to tile this nest, or None to leave it to the one-thread-per-cell kernel.

    A tile is square-ish and never wider than the extent it covers, so a short side does not hand
    most of a block nothing to do. Below TILE_K deep there is too little reuse to pay for staging,
    and a small matmul is over before the extra instructions matter.
    """
    mm = matmul_shape(nest)
    if mm is None:
        return None
    m, n, k = mm.extent(mm.rows), mm.extent(mm.cols), mm.sizes[mm.depth]
    if k < TILE_K or m * n * k < TILE_MIN:
        return None
    sides = [max((tile for tile in (4 * LANES, 2 * LANES, LANES) if tile <= extent), default=0) for extent in (m, n)]
    return (mm, TileSpec(*sides)) if all(sides) else None


def tile_count(mm: Matmul, spec: TileSpec) -> int:
    """How many output tiles the whole nest covers: one block's worth of work each."""
    return mm.extent(mm.batch) * -(-mm.extent(mm.rows) // spec.rows) * -(-mm.extent(mm.cols) // spec.cols)


def unpack(group: Sequence[str], sizes: Mapping[str, int], fused: str, indent: str, tmp: str) -> list[str]:
    """Bind a side's loop variables from one fused index, its innermost dim varying fastest."""
    if not group:
        return []
    if len(group) == 1:
        return [f"{indent}const int {group[0]} = {fused};"]
    lines = [f"{indent}int {tmp} = {fused};"]
    for var in reversed(group[1:]):
        lines.append(f"{indent}const int {var} = {tmp} % {sizes[var]}; {tmp} /= {sizes[var]};")
    return lines + [f"{indent}const int {group[0]} = {tmp};"]


def emit_tiled(nest: LoopNest, mm: Matmul, spec: TileSpec) -> str:
    """A matmul as one tile per block, staged through shared memory and accumulated in registers.

    The naive kernel gives one output cell to one thread, which then reads a whole row and a whole
    column out of global memory: every cell of a row re-reads the same operand, and the reduce axis
    is walked at whatever stride the operand's layout has. Here a block stakes out a rows-by-cols
    tile of the output, and walks the reduce axis TILE_K at a time. Each step stages both sides'
    slabs into shared memory once, and every thread then reads its share of them back out for
    row_regs * col_regs cells at a time, so an element fetched from memory is used by a whole tile
    rather than by one cell.

    The reduce axis is still walked in ascending order, one step after another, and the arithmetic
    per element is the same instruction on the same two values, so a tiled nest returns exactly the
    bits the untiled one returns. Tiling moves memory, not numbers.

    The staging pass chooses its thread order per operand from that operand's own strides, so
    whichever axis is contiguous in memory is the one neighbouring threads walk, and the reads
    coalesce whether the operand is stored k-major (a @ b.T) or tile-major (a @ b). Tails are
    compile-time knowledge here, since every extent is a constant: a bound check is emitted only
    for the dim whose extent the tile does not divide.
    """
    kernel = nest.kernel
    root = kernel.ast
    m, n, k = mm.extent(mm.rows), mm.extent(mm.cols), mm.sizes[mm.depth]
    tm, tn = spec.row_regs, spec.col_regs
    col_blocks = -(-n // spec.cols)
    row_blocks = -(-m // spec.rows)
    acc_type = C_TYPE[root.dtype]

    params = [f"const {C_TYPE[node.dtype]}* __restrict__ in{i}" for i, node in enumerate(kernel.inputs)]
    params.append(f"{C_TYPE[kernel.target.dtype]}* __restrict__ out")
    lines = [f'extern "C" __global__ void {nest.name}({", ".join(params)}) {{']
    for load in mm.staged:
        width = spec.cols if mm.on_cols(load) else spec.rows
        # aligned so a thread can take its four cells of a row as one vector read
        lines.append(f"  __shared__ __align__(16) {C_TYPE[load.value_type]} {load.dest}_s[{TILE_K * (width + TILE_PAD)}];")
    lines.append(f"  const int lane_m = threadIdx.x / {LANES};")
    lines.append(f"  const int lane_n = threadIdx.x % {LANES};")
    lines.append(f"  for (long long tile = blockIdx.x; tile < {tile_count(mm, spec)}LL; tile += gridDim.x) {{")
    lines.append("    long long slab = tile;")
    lines.append(f"    const int col_block = (int)(slab % {col_blocks}) * {spec.cols}; slab /= {col_blocks};")
    lines.append(f"    const int row_block = (int)(slab % {row_blocks}) * {spec.rows}; slab /= {row_blocks};")
    lines += unpack(mm.batch, mm.sizes, "(int)slab", "    ", "batch")
    lines.append(f"    {acc_type} acc[{tm}][{tn}];")
    lines.append("    #pragma unroll")
    lines.append(f"    for (int i = 0; i < {tm}; i++)")
    lines.append("      #pragma unroll")
    lines.append(
        f"      for (int j = 0; j < {tn}; j++) acc[i][j] = {c_literal(REDUCES[root.op].identity[root.dtype], root.dtype)};"
    )

    lines.append(f"    for (int k0 = 0; k0 < {k}; k0 += {TILE_K}) {{")
    for load in mm.staged:
        lines += stage_lines(mm, spec, load, m, n, k, "      ")
    lines.append("      __syncthreads();")
    depth_left = f"{TILE_K}" if k % TILE_K == 0 else f"min({TILE_K}, {k} - k0)"
    lines.append("      #pragma unroll")
    lines.append(f"      for (int kk = 0; kk < {depth_left}; kk++) {{")
    for load in mm.staged:
        cols_side = mm.on_cols(load)
        width, regs = (spec.cols, tn) if cols_side else (spec.rows, tm)
        lane, ctype = ("lane_n" if cols_side else "lane_m"), C_TYPE[load.value_type]
        cell = f"{load.dest}_s[kk * {width + TILE_PAD} + {lane} * {regs}]"
        lines.append(f"        {ctype} {load.dest}_r[{regs}];")
        if (vector := VECTOR.get(load.value_type)) and regs == 4:
            # the four cells are adjacent and the tile is aligned, so they come back as one read
            lines.append(f"        *({vector}*){load.dest}_r = *(const {vector}*)&{cell};")
        else:
            lines.append("        #pragma unroll")
            lines.append(
                f"        for (int e = 0; e < {regs}; e++) {load.dest}_r[e] = {load.dest}_s[kk * {width + TILE_PAD} + {lane} * {regs} + e];"
            )
    lines.append("        #pragma unroll")
    lines.append(f"        for (int i = 0; i < {tm}; i++) {{")
    lines.append("          #pragma unroll")
    lines.append(f"          for (int j = 0; j < {tn}; j++) {{")
    for instr in mm.body:
        if instr in mm.staged:
            lines.append(
                f"            const {C_TYPE[instr.value_type]} {instr.dest} = {instr.dest}_r[{'j' if mm.on_cols(instr) else 'i'}];"
            )
        else:
            lines.append(value_line(instr, "            "))
    lines.append("            " + fold_c(REDUCES[root.op].fold, "acc[i][j]", mm.fold.srcs[0]))
    lines.append("          }")
    lines.append("        }")
    lines.append("      }")
    lines.append("      __syncthreads();")
    lines.append("    }")

    lines.append("    #pragma unroll")
    lines.append(f"    for (int i = 0; i < {tm}; i++) {{")
    lines.append(f"      const int row = row_block + lane_m * {tm} + i;")
    lines += unpack(mm.rows, mm.sizes, "row", "      ", "row_left")
    lines.append("      #pragma unroll")
    lines.append(f"      for (int j = 0; j < {tn}; j++) {{")
    lines.append(f"        const int col = col_block + lane_n * {tn} + j;")
    lines += unpack(mm.cols, mm.sizes, "col", "        ", "col_left")
    checks = ([f"row < {m}"] if m % spec.rows else []) + ([f"col < {n}"] if n % spec.cols else [])
    guarded = f"if ({' && '.join(checks)}) " if checks else ""
    lines.append(f"        {guarded}out[{mm.out.render()}] = acc[i][j];")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def contiguous_in(index: Index, var: str) -> bool:
    return next((coeff for name, coeff in index.terms if name == var), 0) == 1


def vector_width(load: Instr, fast: str, extent: int, width: int) -> int:
    """Four when a thread may take four cells of this operand as one read, else one.

    Four cells are one read when they are four adjacent addresses that this thread wants anyway,
    which takes a stride-1 axis to walk along and 16-byte alignment to start from. Extents are
    constants here, so both are decided now rather than guarded at run time: the axis's extent has
    to be a multiple of four for a read never to straddle its end, every other term of the address
    has to move in multiples of four, and the block has to divide the slab into whole vectors so no
    thread is left holding part of one. A mask disqualifies the operand outright, since the four
    cells would not agree on whether they are inside it.
    """
    _, index, valid = load.arg
    if load.value_type not in VECTOR or valid.bounds or not fast:
        return 1
    if extent % 4 or (width * TILE_K // 4) % BLOCK:
        return 1
    if index.const % 4 or any(coeff % 4 for name, coeff in index.terms if name != fast):
        return 1
    return 4


def stage_lines(mm: Matmul, spec: TileSpec, load: Instr, m: int, n: int, k: int, indent: str) -> list[str]:
    """One operand's slab, read into shared memory by the whole block.

    Which of the two axes neighbouring threads walk is the operand's decision, not the tile's: the
    one it is contiguous in. Getting that backwards is what makes the naive kernel slow on
    `a @ b.T`, where the fastest thread index strides a whole row. That axis is also the one a
    thread reads four-wide when vector_width allows it, and the two cases part company on the way
    into shared memory: four cells along the tile stay adjacent there and go back as one write,
    four along the reduce axis land a padded row apart and go back as four.
    """
    buf, index, valid = load.arg
    cols_side = mm.on_cols(load)
    width = spec.cols if cols_side else spec.rows
    pitch = width + TILE_PAD
    group = mm.cols if cols_side else mm.rows
    block, extent = ("col_block", n) if cols_side else ("row_block", m)
    depth_fastest = contiguous_in(index, mm.depth)
    if depth_fastest:
        fast, reach = mm.depth, k
    elif len(group) == 1 and contiguous_in(index, group[0]):
        fast, reach = group[0], extent
    else:
        fast, reach = "", 0
    wide = vector_width(load, fast, reach, width)

    lines = [f"{indent}{{"]
    lines.append(f"{indent}  #pragma unroll")
    lines.append(f"{indent}  for (int step = 0; step < {width * TILE_K // (BLOCK * wide)}; step++) {{")
    lines.append(f"{indent}    const int slot = (threadIdx.x + step * {BLOCK}) * {wide};")
    if depth_fastest:
        lines.append(f"{indent}    const int depth_at = slot % {TILE_K}, tile_at = slot / {TILE_K};")
    else:
        lines.append(f"{indent}    const int tile_at = slot % {width}, depth_at = slot / {width};")
    lines.append(f"{indent}    const int {mm.depth} = k0 + depth_at;")
    lines.append(f"{indent}    const int fused = {block} + tile_at;")
    lines += unpack(group, mm.sizes, "fused", f"{indent}    ", "tile_left")
    checks = ([f"fused < {extent}"] if extent % width else []) + ([f"{mm.depth} < {k}"] if k % TILE_K else [])
    checks += [valid.render()] if valid.bounds else []
    zero, cell = c_literal(0, load.value_type), f"{load.dest}_s[depth_at * {pitch} + tile_at]"
    if wide == 1:
        read = f"{buf}[{index.render()}]"
        lines.append(
            f"{indent}    {cell} = {' && '.join(checks)} ? {read} : {zero};" if checks else f"{indent}    {cell} = {read};"
        )
    else:
        vector = VECTOR[load.value_type]
        taken = f"*(const {vector}*)&{buf}[{index.render()}]"
        if checks:  # out of range the whole vector is out, since its axis's extent is a multiple of four
            lines.append(f"{indent}    {vector} wide = {{{', '.join([zero] * 4)}}};")
            lines.append(f"{indent}    if ({' && '.join(checks)}) wide = {taken};")
        else:
            lines.append(f"{indent}    const {vector} wide = {taken};")
        if depth_fastest:  # the four cells are a padded row apart in shared memory
            for step, field in enumerate("xyzw"):
                lines.append(f"{indent}    {load.dest}_s[(depth_at + {step}) * {pitch} + tile_at] = wide.{field};")
        else:
            lines.append(f"{indent}    *({vector}*)&{cell} = wide;")
    lines.append(f"{indent}  }}")
    lines.append(f"{indent}}}")
    return lines


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


def emit_one(nest: LoopNest) -> str:
    """The kernel form this nest gets: a split reduce, a tiled matmul, or one thread per output cell."""
    if partials := split_partials(nest):
        return emit_split(nest, partials)
    if plan := tiled(nest):
        return emit_tiled(nest, *plan)
    return emit_kernel(nest)


def emit_cuda(nests: list[LoopNest]) -> str:
    return "\n".join([PRELUDE] + [emit_one(nest) + "\n" for nest in nests])


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
            if plan := (None if partials else tiled(nest)):
                # a tiled kernel's unit of work is a tile, not a cell, and its block is LANES^2 threads
                out.append(self._launcher(program.functions[nest.name], min(tile_count(*plan), GRID)))
                continue
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
