"""Render the loop nest IR as CUDA C: a second rendering of the Instr stream, like backend_c.

Three kernel shapes come out of here, and emit_one picks between them per nest.

One thread per point of the non-reduce dims, with the reduce loops sequential inside it, is the
default. Every output cell then belongs to exactly one thread, so STORE and ACCUM need no
atomics, and a cell folds its elements in ascending reduce order, matching the C backend
exactly. The innermost non-reduce dim (picked by loop_order for being stride-1 in the most
buffers) becomes the fastest-varying thread index, which is what makes warp loads coalesce.
SCATTER is the one racing write, since two values can name the same row, so it adds with
atomicAdd.

A long reduce over few output cells would leave the GPU idle while single threads walk millions
of elements, so it runs as two kernels instead: strided partial accumulators and then their
fold. The grouping is fixed, which keeps it deterministic, but it is not the sequential nest's
grouping, so that shape agrees with the host backends to rounding rather than bit for bit.

A matmul gets a block per output tile, both operands staged through shared memory. That one is
bit-identical to the kernel it replaces, since the reduce axis is still walked in order: what
changes is where the operands are read from, not what is summed.

Launches are grid-stride loops over a bounded number of blocks of BLOCK threads, so any size
runs in one launch; backend_cuda.py owns the grid and the launching.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from limn.backend_c import C_TYPE, c_literal, fold_c, value_c
from limn.codegen import REDUCES, Index, Instr, LoopNest, Opcode
from limn.ops import DType, Op, accumulate_in, float16, float32, int8, int16

BLOCK = 256  # threads per block; a tiled kernel needs exactly LANES * LANES of them

LANES = 16  # threads across a tile in each direction, so one block is LANES * LANES = BLOCK threads
TILE_K = 8  # how far down the reduce axis one staged step reaches
TILE_REGS = (8, 4, 2, 1)  # cells one thread may hold in each direction, widest first
TILE_MIN = 1 << 16  # a matmul with fewer multiply-adds than this is not worth staging for
TILE_PAD = 4  # slack on a staged row, so a column of it spreads over banks instead of piling on one
UNROLL = 4  # vector steps of a reduce to unroll, so a thread has several loads in flight at once
# How each dtype is spelled in memory, and while it is a value. Rounding every float16
# intermediate instead of just the stores costs two conversions per multiply-add in a matmul.
CUDA_TYPE = C_TYPE | {float16: "half_t"}
CUDA_VALUE = C_TYPE | {float16: "float"}
VECTOR = {float32: "float4", float16: "half4_t"}  # the four-wide load type, for the dtypes that have one

PRELUDE = """\
typedef int int32_t;
typedef short int16_t;
typedef signed char int8_t;
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fc00000))
#endif

// atomicAdd on double arrives at sm_60; below that, the CAS loop the programming guide gives,
// which adds in double all the same and only retries when another thread got there first.
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
  unsigned long long* cell = (unsigned long long*)address;
  unsigned long long old = *cell, seen;
  do {
    seen = old;
    old = atomicCAS(cell, seen, __double_as_longlong(val + __longlong_as_double(seen)));
  } while (seen != old);
  return __longlong_as_double(old);
}
#endif

// float16 without cuda_fp16.h, which NVRTC has no include path for and which would put a
// toolkit back in the requirements. Both conversions are single PTX instructions every
// architecture has. The default constructor stays trivial, so __shared__ arrays need no
// initialisation.
struct __align__(2) half_t {
  unsigned short bits;
  half_t() = default;
  __device__ half_t(float f) { asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(f)); }
  __device__ operator float() const {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
  }
};
struct __align__(8) half4_t {
  half_t x, y, z, w;
};
"""


def value_cuda(instr: Instr, indent: str) -> str:
    """One value-defining instruction, spelled for this device: computed as a value, stored as one."""
    return value_c(instr, indent, "", CUDA_VALUE, CUDA_TYPE)


def reduce_axes(nest: LoopNest) -> tuple[int, ...]:
    root = nest.kernel.ast
    return root.arg if root.op in REDUCES else ()


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

    One thread per cell is all the parallelism a reduce has, so few cells over a long reduce leave
    the GPU idle. Splitting the axis buys threads at the price of regrouping the fold, so it is
    worth it only there, and partials are capped so each still gets a real chunk.
    """
    if not reduce_axes(nest):
        return 0
    r = reduce_extent(nest)
    if r < SPLIT_MIN or outer_extent(nest) >= SPLIT_MIN:
        return 0
    return min(4096, r // 64)


@dataclass(frozen=True)
class Fold:
    """A reduce nest as a thread runs it: bind the non-reduce dims, fold the rest into a register.

    lower() emits two shapes, a register total (ACC, UPDATE, STORE) and one folded into the output
    buffer (ACCUM). That distinction is a host one: here the non-reduce dims are the thread index,
    so the reduce axes are always innermost from inside a thread. Reading both into this one costs
    ACCUM its global read-modify-write per element, and the fold order is unchanged.
    """

    dtype: DType
    identity: float | int
    fold: Op  # the arith op folding one element into the total
    body: tuple[Instr, ...]  # the value instructions inside the reduce loops, in order
    value: str  # the name among them that the fold consumes
    out: Index  # where the finished total goes in the output buffer


def fold_form(nest: LoopNest) -> Fold:
    """Read either lowered shape of a reduce nest as one register fold."""
    root = nest.kernel.ast
    identity = REDUCES[root.op].identity[root.dtype]
    body: list[Instr] = []
    for instr in nest.instrs:
        match instr.opcode:
            case Opcode.LOOP | Opcode.ENDLOOP | Opcode.ACC:
                continue
            case Opcode.UPDATE:  # the register shape: what came before it is the fold's body
                store = next(i for i in nest.instrs if i.opcode is Opcode.STORE and i.srcs[0] == instr.dest)
                return Fold(root.dtype, identity, instr.arg, tuple(body), instr.srcs[0], store.arg[1])
            case Opcode.ACCUM:
                # the cell is indexed by non-reduce dims only (the target's reduce dims are size 1,
                # so they carry no stride), which is what makes it one thread's to keep in a register
                _, index, fold = instr.arg
                return Fold(root.dtype, identity, fold, tuple(body), instr.srcs[0], index)
            case Opcode.STORE:  # the ACCUM shape fills the output with the identity first; drop that
                body.clear()
            case _:
                body.append(instr)
    raise AssertionError(f"{nest.name} reduces but folds nothing")


def contiguous_in(index: Index, var: str) -> bool:
    return next((coeff for name, coeff in index.terms if name == var), 0) == 1


def four_wide(load: Instr, fast: str, extent: int) -> bool:
    """Whether a thread may take four cells of this load at once along `fast`.

    A warp walking one element at a time spends a transaction on four bytes. Four at a time is the
    same four out of one transaction, which takes stride 1 along the axis, whole fours to cover its
    extent so a read never straddles the end, and a 16-byte start: every other term of the address
    moving in fours. A mask disqualifies the load, since the four need not agree on being inside it.
    """
    _, index, valid = load.arg
    if load.value_type not in VECTOR or valid.bounds or extent % 4 or not contiguous_in(index, fast):
        return False
    return index.const % 4 == 0 and all(coeff % 4 == 0 for name, coeff in index.terms if name != fast)


def vector_loads(folded: Fold, var: str, bound: int) -> tuple[Instr, ...]:
    """The body's loads a thread may take four elements of at once along `var`; empty for none.

    The rest of the body is emitted once for all four, so anything else that moves with `var`
    means no vector rather than a wrong one.
    """
    if any(instr.opcode not in (Opcode.CONST, Opcode.LOAD, Opcode.ARITH, Opcode.CAST) for instr in folded.body):
        return ()
    wide: list[Instr] = []
    for instr in folded.body:
        if instr.opcode is Opcode.CONST:
            if any(name == var for name, *_ in instr.arg[1].bounds):
                return ()
        elif instr.opcode is Opcode.LOAD:
            if var not in index_vars(instr):
                continue  # the same element for all four, so one rendering already serves them
            if not four_wide(instr, var, bound):
                return ()
            wide.append(instr)
    return tuple(wide)


def fold_lines(folded: Fold, reduce_loops: list[tuple[str, int]], indent: str) -> list[str]:
    """A register total, the reduce loops around it, and one store when they are done."""
    lines = [f"{indent}{CUDA_VALUE[accumulate_in(folded.dtype)]} acc = {c_literal(folded.identity, folded.dtype)};"]
    for depth, (var, bound) in enumerate(reduce_loops[:-1]):
        lines.append(f"{indent}{'  ' * depth}for (int {var} = 0; {var} < {bound}; {var}++) {{")

    var, bound = reduce_loops[-1]
    wide = vector_loads(folded, var, bound)
    at = indent + "  " * (len(reduce_loops) - 1)
    if wide:
        # the running total is a dependency chain, so a thread has one load in flight and waits on
        # it; unrolling gives the scheduler several to issue before the first one is needed
        lines.append(f"{at}#pragma unroll {UNROLL}")
    lines.append(f"{at}for (int {var} = 0; {var} < {bound}; {var} += {4 if wide else 1}) {{")
    for load in wide:
        buf, index, _ = load.arg
        vector = VECTOR[load.value_type]
        lines.append(f"{at}  const {vector} {load.dest}_w = *(const {vector}*)&{buf}[{index.render()}];")
    body = at + ("    " if wide else "  ")
    for field in "xyzw" if wide else "x":
        # one scope per element, so the four renderings of the body may reuse their value names
        lines += [f"{at}  {{"] if wide else []
        for instr in folded.body:
            if instr in wide:
                lines.append(f"{body}const {CUDA_VALUE[instr.value_type]} {instr.dest} = {instr.dest}_w.{field};")
            else:
                lines.append(value_cuda(instr, body))
        lines.append(body + fold_c(folded.fold, "acc", folded.value))
        lines += [f"{at}  }}"] if wide else []
    lines.append(f"{at}}}")

    lines += [f"{indent}{'  ' * depth}}}" for depth in reversed(range(len(reduce_loops) - 1))]
    lines.append(f"{indent}out[{folded.out.render()}] = acc;")
    return lines


def nest_loops(nest: LoopNest) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """The nest's loops as (variable, bound) lists in nesting order: non-reduce, then reduce.

    An ACCUM-form nest opens its non-reduce loops twice (the identity fill, then the fold),
    so each variable counts at first sight only.
    """
    reduce_vars = {f"r{d}" for d in reduce_axes(nest)}
    outer: list[tuple[str, int]] = []
    reduce_loops: list[tuple[str, int]] = []
    seen: set[str] = set()
    for instr in nest.instrs:
        if instr.opcode is Opcode.LOOP and instr.dest not in seen:
            seen.add(instr.dest)
            (reduce_loops if instr.dest in reduce_vars else outer).append((instr.dest, instr.arg))
    return outer, reduce_loops


def bind_vars(loops: Sequence[tuple[str, int]], flat: str, tmp: str, indent: str, wide: bool = True) -> list[str]:
    """Recover loop variables from a flat index, the last one varying fastest.

    wide is for an index that does not fit an int, which a grid-stride gid need not: the scratch
    is long long and each variable is narrowed on the way out of it.
    """
    if not loops:
        return []
    cast, scratch = ("(int)", "long long") if wide else ("", "int")
    if len(loops) == 1:
        return [f"{indent}const int {loops[0][0]} = {cast}({flat});"]
    lines = [f"{indent}{scratch} {tmp} = {flat};"]
    for var, bound in reversed(loops[1:]):
        lines.append(f"{indent}const int {var} = {cast}({tmp} % {bound}); {tmp} /= {bound};")
    return lines + [f"{indent}const int {loops[0][0]} = {cast}({tmp});"]


def grid_stride(extent: int) -> str:
    """Open the launch loop: each thread starts at its global id and strides by the whole grid."""
    first = "blockIdx.x * (long long)blockDim.x + threadIdx.x"
    return f"  for (long long gid = {first}; gid < {extent}LL; gid += (long long)gridDim.x * blockDim.x) {{"


def kernel_sig(name: str, in_dtypes: list[DType], out_dtype: DType) -> str:
    """The signature line every kernel shares: const restrict inputs, one restrict output.

    restrict on the inputs is safe even when two of them are the same buffer (a @
    a.transpose()): it only promises that nothing *written* is reachable through another
    name, and only out is written, to memory out_alloc freshly allocated.
    """
    params = [f"const {CUDA_TYPE[dtype]}* __restrict__ in{k}" for k, dtype in enumerate(in_dtypes)]
    params.append(f"{CUDA_TYPE[out_dtype]}* __restrict__ out")
    return f'extern "C" __global__ void {name}({", ".join(params)}) {{'


def part_name(nest: LoopNest) -> str:
    """The partial-stage kernel's name; emission and symbol lookup must agree on it."""
    return nest.name + "_part"


def split_scratch(nest: LoopNest, partials: int) -> int:
    """Bytes the two stages of a split reduce pass between them: one running total per partial."""
    return outer_extent(nest) * partials * accumulate_in(nest.kernel.ast.dtype).itemsize


def emit_split(nest: LoopNest, partials: int) -> str:
    """A register reduce as two kernels: strided partials per output cell, then their fold.

    Partial p of a cell accumulates elements p, p + partials, p + 2*partials, ... so warp
    loads stay coalesced and the grouping is fixed, which is what makes the result
    deterministic; it is grouped differently from the sequential nest, so it agrees with the
    host backends to rounding rather than bit for bit (int folds stay exact, addition being
    associative modulo 2**width).

    The partials are running totals, so they move at the accumulator's width, not the output's;
    split_scratch sizes the buffer backend_cuda allocates for them.
    """
    kernel = nest.kernel
    outer, loops = nest_loops(nest)
    folded = fold_form(nest)
    wide = accumulate_in(folded.dtype)
    ctype = CUDA_VALUE[wide]
    cells = outer_extent(nest)

    start = f"    {ctype} acc = {c_literal(folded.identity, folded.dtype)};"

    lines = [kernel_sig(part_name(nest), [node.dtype for node in kernel.inputs], wide), grid_stride(cells * partials)]
    # the cell is the fast half of gid, so neighbouring threads hold neighbouring cells and read
    # the same element of each: whatever stride-1 axis the cells span, a warp still walks it
    lines.append(f"    const int p = (int)(gid / {cells});")
    lines += bind_vars(outer, f"gid % {cells}", "t", "    ")
    lines += [start, f"    for (long long j = p; j < {reduce_extent(nest)}LL; j += {partials}) {{"]
    lines += bind_vars(loops, "j", "u", "      ")
    lines += [value_cuda(instr, "      ") for instr in folded.body]
    # gid is p * cells + cell, which is the layout the fold kernel below reads
    lines += ["      " + fold_c(folded.fold, "acc", folded.value), "    }", "    out[gid] = acc;", "  }", "}"]

    lines += ["", kernel_sig(nest.name, [wide], folded.dtype), grid_stride(cells)]
    lines += bind_vars(outer, "gid", "t", "    ")
    lines += [start, f"    for (int p = 0; p < {partials}; p++) {{"]
    lines += ["      " + fold_c(folded.fold, "acc", f"in0[p * {cells} + gid]"), "    }"]
    lines += [f"    out[{folded.out.render()}] = acc;", "  }", "}"]
    return "\n".join(lines)


# ---- matmuls: a block per output tile, both operands staged through shared memory ----


@dataclass(frozen=True)
class Matmul:
    """A reduce nest read as a matmul: two sides of loop variables that meet only in the reduce axis.

    What makes a nest one is how its loads are indexed: each side moves with its own variables and
    the reduce variable, and with nothing the other side reads. That is the condition for a staged
    tile to be reused, since a tile of one side is then the same for every point of the other.
    Variables both sides read, or neither does, select which matmul instead of indexing into one.
    A side can hold several, which is why the emitter works in fused indices and unpacks them.
    """

    rows: tuple[str, ...]  # the loop vars only one side's loads read; their extents multiply to M
    cols: tuple[str, ...]  # the other side's, multiplying to N, and holding the output's fastest dim
    batch: tuple[str, ...]  # vars both sides read or neither does: one matmul per point of them
    depth: str  # the reduce variable, whose extent is K
    sizes: Mapping[str, int]
    staged: tuple[Instr, ...]  # the loads to stage in shared memory, in emission order
    fold: Fold  # the body the tile computes, and where its totals go

    def extent(self, group: Sequence[str]) -> int:
        return math.prod(self.sizes[var] for var in group)

    def loops(self, group: Sequence[str]) -> list[tuple[str, int]]:
        """One side's variables as bind_vars takes them: (variable, extent), outermost first."""
        return [(var, self.sizes[var]) for var in group]

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
    below. A load that does not move with the reduce axis, a masked constant and an indexed read
    all bail out, because each would have to be re-read or re-derived per output cell and none of
    them appears in a matmul.
    """
    axes = reduce_axes(nest)
    if len(axes) != 1:
        return None
    folded = fold_form(nest)
    depth = f"r{axes[0]}"
    sizes = loop_bounds(nest)
    if any(instr.opcode not in (Opcode.CONST, Opcode.LOAD, Opcode.ARITH, Opcode.CAST) for instr in folded.body):
        return None
    if any(instr.opcode is Opcode.CONST and instr.arg[1].bounds for instr in folded.body):
        return None

    loads = [instr for instr in folded.body if instr.opcode is Opcode.LOAD]
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

    # the side holding the nest's innermost dim owns the output's stride-1 axis, so it is the one
    # to spread across a warp: call it the columns, and the store coalesces
    rows, cols = sorted(sides, key=lambda side: max(int(var[1:]) for var in side))
    return Matmul(dims(rows), dims(cols), dims(shared), depth, sizes, tuple(staged), folded)


@dataclass(frozen=True)
class TileSpec:
    """One block's share of the output: `rows` by `cols` cells, held in registers across LANES^2 threads."""

    rows: int
    cols: int

    def width(self, cols_side: bool) -> int:
        """How wide the tile is on the side this operand feeds."""
        return self.cols if cols_side else self.rows

    def regs(self, cols_side: bool) -> int:
        """Cells one thread holds along that side: the width spread over LANES threads."""
        return self.width(cols_side) // LANES


def stages_whole(width: int) -> bool:
    """Whether the block covers a width-by-TILE_K slab exactly, which stage_lines assumes.

    The staging loop's trip count is a truncating division, so a slab under one pass of the block
    stages nothing at all and the fold reads shared memory nobody wrote. Widths are picked against
    this rather than clamped up to it, so a short side falls back to the untiled kernel.
    """
    return (width * TILE_K) % BLOCK == 0


def tiled(nest: LoopNest) -> tuple[Matmul, TileSpec] | None:
    """How to tile this nest, or None to leave it to the one-thread-per-cell kernel.

    A tile reaches neither past the extent it covers nor under what the block stages in one piece.
    Below TILE_K deep there is too little reuse to pay for staging, and a small matmul is over
    before the extra instructions matter.
    """
    mm = matmul_shape(nest)
    if mm is None:
        return None
    m, n, k = mm.extent(mm.rows), mm.extent(mm.cols), mm.sizes[mm.depth]
    if k < TILE_K or m * n * k < TILE_MIN:
        return None
    widths = [regs * LANES for regs in TILE_REGS if stages_whole(regs * LANES)]
    sides = [max((width for width in widths if width <= extent), default=0) for extent in (m, n)]
    return (mm, TileSpec(*sides)) if all(sides) else None


def blocks(mm: Matmul, spec: TileSpec) -> tuple[int, int]:
    """Tiles down the rows and across the columns. The launch bound and the in-kernel decode of a
    block's tile are the same two numbers, so they are counted once."""
    return -(-mm.extent(mm.rows) // spec.rows), -(-mm.extent(mm.cols) // spec.cols)


def tile_count(mm: Matmul, spec: TileSpec) -> int:
    """How many output tiles the whole nest covers: one block's worth of work each."""
    row_blocks, col_blocks = blocks(mm, spec)
    return mm.extent(mm.batch) * row_blocks * col_blocks


def vector_width(load: Instr, fast: str, extent: int, width: int) -> int:
    """Four when a thread may take four cells of this operand as one read, else one.

    Past the load's own alignment, the block has to divide the slab into whole vectors, or the
    staging loop would not cover it.
    """
    if not fast or (width * TILE_K // 4) % BLOCK:
        return 1
    return 4 if four_wide(load, fast, extent) else 1


def stage_lines(mm: Matmul, spec: TileSpec, load: Instr, indent: str) -> list[str]:
    """One operand's slab, read into shared memory by the whole block.

    Which axis neighbouring threads walk is the operand's decision, not the tile's: the one it is
    contiguous in. Getting that backwards is what makes the untiled kernel slow on `a @ b.T`. The
    same axis is read four-wide where vector_width allows, and the two cases part on the way into
    shared memory: four along the tile stay adjacent, four along the reduce axis land a row apart.
    """
    buf, index, valid = load.arg
    cols_side = mm.on_cols(load)
    width = spec.width(cols_side)
    pitch = width + TILE_PAD
    group = mm.cols if cols_side else mm.rows
    block, extent = ("col_block", mm.extent(mm.cols)) if cols_side else ("row_block", mm.extent(mm.rows))
    k = mm.sizes[mm.depth]
    depth_fastest = contiguous_in(index, mm.depth)
    if depth_fastest:
        fast, reach = mm.depth, k
    elif len(group) == 1 and contiguous_in(index, group[0]):
        fast, reach = group[0], extent
    else:
        fast, reach = "", 0
    wide = vector_width(load, fast, reach, width)

    lines = [f"{indent}{{", f"{indent}  #pragma unroll"]
    lines.append(f"{indent}  for (int step = 0; step < {width * TILE_K // (BLOCK * wide)}; step++) {{")
    lines.append(f"{indent}    const int slot = (threadIdx.x + step * {BLOCK}) * {wide};")
    if depth_fastest:
        lines.append(f"{indent}    const int depth_at = slot % {TILE_K}, tile_at = slot / {TILE_K};")
    else:
        lines.append(f"{indent}    const int tile_at = slot % {width}, depth_at = slot / {width};")
    lines += [f"{indent}    const int {mm.depth} = k0 + depth_at;", f"{indent}    const int fused = {block} + tile_at;"]
    lines += bind_vars(mm.loops(group), "fused", "tile_left", f"{indent}    ", wide=False)
    checks = ([f"fused < {extent}"] if extent % width else []) + ([f"{mm.depth} < {k}"] if k % TILE_K else [])
    checks += [valid.render()] if valid.bounds else []
    zero = f"({CUDA_TYPE[load.value_type]}){c_literal(0, load.value_type)}"  # same arm-type rule as guard()
    cell = f"{load.dest}_s[depth_at * {pitch} + tile_at]"
    if wide == 1:
        read = f"{buf}[{index.render()}]"
        guarded = f"{' && '.join(checks)} ? {read} : {zero}" if checks else read
        lines.append(f"{indent}    {cell} = {guarded};")
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
    return lines + [f"{indent}  }}", f"{indent}}}"]


def register_lines(mm: Matmul, spec: TileSpec, load: Instr, indent: str) -> list[str]:
    """One row of an operand's staged slab into a thread's registers, four at a time where it can."""
    cols_side = mm.on_cols(load)
    regs, ctype = spec.regs(cols_side), CUDA_TYPE[load.value_type]
    lane = "lane_n" if cols_side else "lane_m"
    base = f"{load.dest}_s[kk * {spec.width(cols_side) + TILE_PAD} + {lane} * {regs}"
    # aligned to the vector below, which a plain array of the element type would not be
    lines = [f"{indent}__align__(16) {ctype} {load.dest}_r[{regs}];"]
    if (vector := VECTOR.get(load.value_type)) and regs % 4 == 0:
        # a thread's cells are adjacent, and every term of the index is a multiple of four,
        # so the slab being 16-byte aligned makes each four of them one read
        for at in range(regs // 4):
            lines.append(f"{indent}*(({vector}*){load.dest}_r + {at}) = *(const {vector}*)&{base} + {4 * at}];")
    else:
        lines += [f"{indent}#pragma unroll", f"{indent}for (int e = 0; e < {regs}; e++) {load.dest}_r[e] = {base} + e];"]
    return lines


def cell_lines(mm: Matmul, indent: str) -> list[str]:
    """The fused body for one output cell, each staged operand read out of its registers."""
    folded = mm.fold
    lines = []
    for instr in folded.body:
        if instr in mm.staged:
            at = "j" if mm.on_cols(instr) else "i"
            lines.append(f"{indent}const {CUDA_VALUE[instr.value_type]} {instr.dest} = {instr.dest}_r[{at}];")
        else:
            lines.append(value_cuda(instr, indent))
    lines.append(indent + fold_c(folded.fold, "acc[i][j]", folded.value))
    return lines


def store_lines(mm: Matmul, spec: TileSpec, indent: str) -> list[str]:
    """The tile's totals out to the output, bound-checked only where a tile does not divide."""
    m, n = mm.extent(mm.rows), mm.extent(mm.cols)
    tm, tn = spec.regs(False), spec.regs(True)
    checks = ([f"row < {m}"] if m % spec.rows else []) + ([f"col < {n}"] if n % spec.cols else [])
    guard = f"if ({' && '.join(checks)}) " if checks else ""
    lines = [f"{indent}#pragma unroll", f"{indent}for (int i = 0; i < {tm}; i++) {{"]
    lines.append(f"{indent}  const int row = row_block + lane_m * {tm} + i;")
    lines += bind_vars(mm.loops(mm.rows), "row", "row_left", f"{indent}  ", wide=False)
    lines += [f"{indent}  #pragma unroll", f"{indent}  for (int j = 0; j < {tn}; j++) {{"]
    lines.append(f"{indent}    const int col = col_block + lane_n * {tn} + j;")
    lines += bind_vars(mm.loops(mm.cols), "col", "col_left", f"{indent}    ", wide=False)
    lines.append(f"{indent}    {guard}out[{mm.fold.out.render()}] = acc[i][j];")
    return lines + [f"{indent}  }}", f"{indent}}}"]


def emit_tiled(nest: LoopNest, mm: Matmul, spec: TileSpec) -> str:
    """A matmul as one tile per block, staged through shared memory and accumulated in registers.

    The untiled kernel gives one cell to one thread, which reads a whole row and column out of
    global memory. Here a block takes a rows-by-cols tile and walks the reduce axis TILE_K at a
    time, staging both slabs once per step, so an element fetched serves a tile rather than a cell.

    The reduce axis is still walked in ascending order over the same two values per element, so a
    tiled nest returns the bits the untiled one returns: tiling moves memory, not numbers. Tails
    are compile-time knowledge, so a bound check is emitted only where a tile does not divide.
    """
    kernel = nest.kernel
    folded = mm.fold
    k = mm.sizes[mm.depth]
    tm, tn = spec.regs(False), spec.regs(True)
    row_blocks, col_blocks = blocks(mm, spec)
    reach = TILE_K if k % TILE_K == 0 else f"min({TILE_K}, {k} - k0)"

    lines = [kernel_sig(nest.name, [node.dtype for node in kernel.inputs], kernel.target.dtype)]
    for load in mm.staged:
        # aligned so a thread can take its four cells of a row as one vector read
        held = TILE_K * (spec.width(mm.on_cols(load)) + TILE_PAD)
        lines.append(f"  __shared__ __align__(16) {CUDA_TYPE[load.value_type]} {load.dest}_s[{held}];")
    lines += [f"  const int lane_m = threadIdx.x / {LANES};", f"  const int lane_n = threadIdx.x % {LANES};"]
    lines.append(f"  for (long long tile = blockIdx.x; tile < {tile_count(mm, spec)}LL; tile += gridDim.x) {{")
    lines.append("    long long slab = tile;")
    lines.append(f"    const int col_block = (int)(slab % {col_blocks}) * {spec.cols}; slab /= {col_blocks};")
    lines.append(f"    const int row_block = (int)(slab % {row_blocks}) * {spec.rows}; slab /= {row_blocks};")
    lines += bind_vars(mm.loops(mm.batch), "(int)slab", "batch_left", "    ", wide=False)
    lines.append(f"    {CUDA_VALUE[accumulate_in(folded.dtype)]} acc[{tm}][{tn}];")
    lines += ["    #pragma unroll", f"    for (int i = 0; i < {tm}; i++)", "      #pragma unroll"]
    lines.append(f"      for (int j = 0; j < {tn}; j++) acc[i][j] = {c_literal(folded.identity, folded.dtype)};")

    lines.append(f"    for (int k0 = 0; k0 < {k}; k0 += {TILE_K}) {{")
    for load in mm.staged:
        lines += stage_lines(mm, spec, load, "      ")
    lines += ["      __syncthreads();", "      #pragma unroll"]
    lines.append(f"      for (int kk = 0; kk < {reach}; kk++) {{")
    for load in mm.staged:
        lines += register_lines(mm, spec, load, "        ")
    lines.append("        #pragma unroll")
    lines += [
        f"        for (int i = 0; i < {tm}; i++) {{",
        "          #pragma unroll",
        f"          for (int j = 0; j < {tn}; j++) {{",
    ]
    lines += cell_lines(mm, "            ")
    lines += ["          }", "        }", "      }", "      __syncthreads();", "    }"]
    lines += store_lines(mm, spec, "    ")
    return "\n".join(lines + ["  }", "}"])


def write_line(instr: Instr, indent: str) -> str:
    """One instruction of an elementwise nest: a value, or the write that ends it."""
    match instr.opcode:
        case Opcode.STORE:
            buf, index = instr.arg
            return f"{indent}{buf}[{index.render()}] = {instr.srcs[0]};"
        case Opcode.SCATTER:
            buf, index = instr.arg  # threads collide wherever indices repeat a row
            if instr.value_type in (int8, int16):
                # atomicAdd has no 8- or 16-bit overload on any architecture
                raise NotImplementedError(f"the cuda device cannot scatter {instr.value_type}; cast the table to int32")
            if instr.value_type == float16:
                # a float16 atomic add needs either cuda_fp16.h or a PTX instruction that only
                # sm_70 and later have, and emission does not know the architecture
                raise NotImplementedError("the cuda device cannot scatter float16; cast the table to float32")
            return f"{indent}atomicAdd(&{buf}[{index.render()}], {instr.srcs[0]});"
        case _:
            return value_cuda(instr, indent)


def emit_kernel(nest: LoopNest) -> str:
    kernel = nest.kernel
    outer, reduce_loops = nest_loops(nest)

    lines = [kernel_sig(nest.name, [node.dtype for node in kernel.inputs], kernel.target.dtype), grid_stride(outer_extent(nest))]
    # bind the non-reduce dims from gid, the innermost one varying fastest: neighbouring threads
    # then touch neighbouring elements of whatever loop_order found to be stride-1, and coalesce
    lines += bind_vars(outer, "gid", "t", "    ")
    if reduce_loops:
        lines += fold_lines(fold_form(nest), reduce_loops, "    ")
    else:
        control = (Opcode.LOOP, Opcode.ENDLOOP)  # every loop of an elementwise nest is a thread
        lines += [write_line(instr, "    ") for instr in nest.instrs if instr.opcode not in control]
    return "\n".join(lines + ["  }", "}"])


def emit_one(nest: LoopNest) -> str:
    """The kernel form this nest gets: a split reduce, a tiled matmul, or one thread per output cell."""
    if partials := split_partials(nest):
        return emit_split(nest, partials)
    if plan := tiled(nest):
        return emit_tiled(nest, *plan)
    return emit_kernel(nest)


def emit_cuda(nests: list[LoopNest]) -> str:
    return "\n".join([PRELUDE] + [emit_one(nest) + "\n" for nest in nests])


# ---- fused attention: a CUSTOM kernel the schedule never lowers ----

SDPA_KERNEL = "sdpa"
SDPA_TILE_K = 32  # keys staged through shared memory per step


def sdpa_blocks(node) -> int:
    """How many blocks the kernel launches: one per (batch, chunk of query rows).

    Blocks never straddle a batch, because a block's shared K/V tile belongs to exactly one
    batch; the launcher and the kernel's blockIdx decoding must agree on this count.
    """
    q = node.srcs[0]
    return math.prod(q.shape[:-2]) * math.ceil(q.shape[-2] / BLOCK)


def emit_sdpa(node) -> str:
    """One hand-written kernel for a CUSTOM ("sdpa", causal, scale) node: one thread per query
    row keeps the online softmax recurrence in registers while K and V stream through shared
    memory a tile at a time, so no t_q by t_k buffer exists and a tile is read from global
    once per block rather than once per row.

    A block covers BLOCK consecutive query rows of a single batch. Every thread helps load
    each tile and meets at the same barriers, and threads whose row is past the end of t_q
    simply skip the per-row work. A causal block also stops streaming tiles at its own last
    row's diagonal: every key past it is masked for every row the block holds, and the bound
    depends only on blockIdx, so the barriers stay uniform.
    """
    _, causal, scale = node.arg
    q, k, v = node.srcs
    dtype = node.dtype
    comp = CUDA_VALUE[accumulate_in(dtype)]  # the width the recurrence runs at
    mem = CUDA_TYPE[dtype]  # the width the buffers hold
    exp, fmax = ("expf", "fmaxf") if comp == "float" else ("exp", "fmax")
    zero = c_literal(0.0, accumulate_in(dtype))
    hd, t_q, t_k, hd_v = q.shape[-1], q.shape[-2], k.shape[-2], v.shape[-1]
    n_chunks = math.ceil(t_q / BLOCK)
    tile = SDPA_TILE_K
    scale_lit = c_literal(scale, accumulate_in(dtype))
    keys_end = "i + 1" if causal else f"{t_k}LL"
    block_keys = f"(row_chunk + 1) * {BLOCK} < {t_k} ? (row_chunk + 1) * {BLOCK} : {t_k}LL" if causal else f"{t_k}LL"

    def load_tile(name: str, base: str, width: int) -> list[str]:
        """One cooperative copy of a key/value tile into shared memory, zero-filled past t_k."""
        return [
            f"    for (int idx = (int)threadIdx.x; idx < {tile} * {width}; idx += blockDim.x) {{",
            f"      int lj = idx / {width}, d = idx - lj * {width};",
            "      long long j = j0 + lj;",
            f"      {name}[lj][d] = j < {t_k} ? {base}[j * {width} + d] : {mem}({zero});",
            "    }",
        ]

    lines = [
        kernel_sig(SDPA_KERNEL, [q.dtype, k.dtype, v.dtype], dtype),
        f"  __shared__ {mem} K_tile[{tile}][{hd}];",
        f"  __shared__ {mem} V_tile[{tile}][{hd_v}];",
        f"  long long b = blockIdx.x / {n_chunks};",
        f"  long long row_chunk = blockIdx.x - b * {n_chunks};",
        f"  int i = (int)(row_chunk * {BLOCK} + threadIdx.x);",
        f"  int valid = i < {t_q};",
        f"  const {mem}* k_base = in1 + b * {t_k} * {hd};",
        f"  const {mem}* v_base = in2 + b * {t_k} * {hd_v};",
        f"  {comp} q_local[{hd}];",
        f"  {comp} acc[{hd_v}];",
        f"  {comp} m = -INFINITY;",
        f"  {comp} l = {zero};",
        f"  for (int d = 0; d < {hd_v}; d++) acc[d] = {zero};",
        "  if (valid) {",
        f"    const {mem}* q_row = in0 + (b * {t_q} + i) * {hd};",
        f"    for (int d = 0; d < {hd}; d++) q_local[d] = ({comp})q_row[d];",
        "  }",
        f"  long long keys_end = valid ? ({keys_end}) : 0LL;",
        f"  long long block_keys = {block_keys};",
        f"  for (long long j0 = 0; j0 < block_keys; j0 += {tile}) {{",
        *load_tile("K_tile", "k_base", hd),
        *load_tile("V_tile", "v_base", hd_v),
        "    __syncthreads();",
        f"    long long tile_end = j0 + {tile} < keys_end ? j0 + {tile} : keys_end;",
        "    for (long long j = j0; j < tile_end; j++) {",
        "      int lj = (int)(j - j0);",
        f"      {comp} s = {zero};",
        f"      for (int d = 0; d < {hd}; d++) s += q_local[d] * ({comp})K_tile[lj][d];",
        f"      s *= {scale_lit};",
        f"      {comp} m_new = {fmax}(m, s);",
        f"      {comp} p = {exp}(s - m_new);",
        f"      {comp} alpha = {exp}(m - m_new);",
        "      l = l * alpha + p;",
        f"      for (int d = 0; d < {hd_v}; d++) acc[d] = acc[d] * alpha + p * ({comp})V_tile[lj][d];",
        "      m = m_new;",
        "    }",
        "    __syncthreads();",
        "  }",
        "  if (valid) {",
        f"    {mem}* o_row = out + (b * {t_q} + i) * {hd_v};",
        f"    for (int d = 0; d < {hd_v}; d++) o_row[d] = ({mem})(acc[d] / l);",
        "  }",
        "}",
    ]
    return PRELUDE + "\n" + "\n".join(lines) + "\n"
