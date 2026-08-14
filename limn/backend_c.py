"""C backend: emit C from the loop nest IR, compile with cc, execute via ctypes.

Each scheduled kernel becomes one C function taking void* pointers (one per input, one for
output), cast to the right type inside. The emitter is a second rendering of the same Instr
stream that render() in codegen.py prints; correctness is proven by diffing against NumpyDevice.

-march=native gets a nest the host's vector width, which is one core's worth of speed; the cores
come from OpenMP. A nest's leading non-reduce loops go to a thread team, so the threads divide
the output cells between them and each cell is still computed start to finish by one thread,
folding in the order the serial nest folds. That leaves threading with no numerical consequence:
the same nest emitted without the pragmas gives back the same bits, not merely the same answer to
a tolerance, which is what test_backend_c.py holds it to. Whether a cc has a usable OpenMP runtime
is not knowable up front, so it is probed, and the pragmas are simply not emitted when the probe
fails.
"""

from __future__ import annotations

import atexit
import ctypes
import functools
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from limn.codegen import Instr, LoopNest, Opcode, Valid, loop_range, reduce_axes, split_masked
from limn.device import NUMPY_DTYPES, Buffer, HostDevice
from limn.jit import CompiledDevice, Runner
from limn.ops import DType, FLOATS, HALF_FLOATS, INTS, Op, float32, float64, int8, int16, int32

C_TYPE = {float64: "double", float32: "float", int32: "int32_t", int16: "int16_t", int8: "int8_t"}

PARALLEL_MIN = 1 << 15  # a nest smaller than this stays serial: waking a team costs more than it does
CHUNKS_PER_THREAD = 4  # slabs of the parallel space to give each thread, so no core holds up the rest

OPENMP_PROBE = """\
#include <omp.h>
int probe(void) {
  int total = 0;
  #pragma omp parallel for reduction(+ : total)
  for (int i = 0; i < 8; i++) total += i;
  return total + omp_get_max_threads();
}
"""

cache: dict[str, ctypes.CDLL] = {}
tmpdirs: list[Path] = []


def has_cc() -> bool:
    return shutil.which("cc") is not None


def cc_builds(flags: tuple[str, ...], source: str = "") -> bool:
    """Whether this machine's cc compiles *and links* a probe with these flags.

    Linking is the point. A driver can accept a flag and then fail at the link for want of a
    runtime library, which is exactly what clang does with -fopenmp where there is no libomp.
    """
    with tempfile.TemporaryDirectory(prefix="limn_probe_") as tmp:
        src = Path(tmp) / "probe.c"
        src.write_text(source)
        probe = ["cc", *flags, "-shared", "-fPIC", "-o", str(Path(tmp) / "probe.so"), str(src)]
        return subprocess.run(probe, capture_output=True, text=True).returncode == 0


@functools.cache
def cc_flags() -> tuple[str, ...]:
    """Optimisation flags for this machine's cc.

    A reordered nest only vectorises if the compiler may target the vector width this CPU actually
    has; plain -O3 compiles for baseline x86-64, which stops at SSE2. Not every cc takes
    -march=native (clang on arm64 rejects it), and not every one has an OpenMP runtime to link
    against, so probe rather than assume; openmp() is how the emitter asks whether that one
    survived.

    Targeting the host also lets the compiler contract a multiply and an add into one FMA, so a
    float result can differ in the last bit from the numpy device's, and between two machines.
    test_backend_c.py diffs at 1e-5, which absorbs that. -ffp-contract=off would buy the bit back
    at most of the speed.
    """
    flags = ("-O3",) + (("-march=native",) if cc_builds(("-march=native",)) else ())
    return flags + (("-fopenmp",) if cc_builds(("-fopenmp",), OPENMP_PROBE) else ())


def openmp() -> bool:
    return "-fopenmp" in cc_flags()


@functools.cache
def team_size() -> int:
    """How many threads a kernel's team gets: what OMP_NUM_THREADS says, else one per physical core.

    libgomp's own default is one per logical processor, and its idle threads spin. Between two
    kernels the whole team is idle while the executor finds and calls the next one, so on an SMT
    machine both siblings of every core sit there spinning and the thread doing the useful work has
    to share a core with one of them. Siblings share a core's vector units and its L1 anyway, so
    what the second adds to a kernel already streaming memory is small.

    sysfs is the only place the topology is written down; without it (a non-Linux host) the logical
    count is the best guess available. Either way the affinity mask is the ceiling, so a run pinned
    to two cores asks for two threads. The answer is fixed on first use, since it is compiled into
    the kernels as a num_threads clause and the collapse decision is taken against it.
    """
    asked = os.environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()  # a list sets one count per nesting level
    if asked.isdigit() and int(asked) > 0:
        return int(asked)
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    cores = {p.read_text() for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology/thread_siblings_list")}
    return min(usable, len(cores)) if cores else usable


def collapse_depth(bounds: Sequence[int]) -> int:
    """How many of a chain's leading loops to fuse into one parallel iteration space; 0 for none.

    Static scheduling cuts the space into one contiguous slab per thread, so a loop with fewer
    iterations than there are threads leaves cores idle, and one with only a couple each leaves
    them waiting on whichever thread drew the slowest core. Fusing the next loop in multiplies the
    slabs there are to divide up. Fusing is not free, since the fused index costs a division per
    slab, so it stops as soon as there is enough work to go round, which for most nests is at the
    first loop.
    """
    target = CHUNKS_PER_THREAD * team_size()
    extent, depth = 1, 0
    for bound in bounds:
        if extent >= target:
            break
        extent *= bound
        depth += 1
    return depth if extent > 1 else 0


def parallel_loops(nest: LoopNest, instrs: Sequence[Instr]) -> dict[int, int]:
    """Which loops open a thread team: the index of one in `instrs` -> how many loops it fuses.

    A nest is one or two top-level chains of loops (a reduce that folds into its output fills it
    with the identity in a chain of its own first). The leading non-reduce loops of a chain are the
    ones worth threading: the output's index is affine in exactly those variables, so two of their
    iterations never name the same cell, and everything an iteration carries is declared inside it.

    Where a chain starts on a reduce axis it stays serial. That axis carries a running total, in a
    register or in the output cell, that every later iteration folds into; splitting it needs
    partial accumulators, which regroup the additions and stop the answer being the serial one bit
    for bit. It costs the nests that reduce nearly all of what they read: a full reduce, and one
    whose single surviving dim loop_order moved innermost for being the stride-1 one.

    A SCATTER stays serial too. It adds into whichever rows its indices name, so two iterations can
    collide; an atomic add would fix the race and leave the sum's order up to thread timing, which
    is a bad trade for the host backend the other ones are diffed against.

    `instrs` is the stream the emitter renders rather than nest.instrs, since a team is opened by
    position. That stream is split_masked's, which cuts innermost loops into siblings, and a
    collapse clause may only span loops that are perfectly nested. Two things keep it to those: the
    run has to open back to back in the stream, and it is cut back to the depth above any sibling.
    """
    if not openmp() or math.prod(nest.space) < PARALLEL_MIN:
        return {}
    if any(instr.opcode is Opcode.SCATTER for instr in instrs):
        return {}
    reduce_vars = {f"r{d}" for d in reduce_axes(nest)}
    teams: dict[int, int] = {}
    depth, start, deepest, lead = 0, 0, 0, []
    for k, instr in enumerate(instrs):
        if instr.opcode is Opcode.LOOP:
            if depth == 0:
                start, deepest, lead = k, 0, []
            if depth < deepest:  # a second loop at this depth, so the chain is no longer one loop wide below it
                del lead[depth:]
            deepest = max(deepest, depth + 1)
            lo, hi = loop_range(instr)
            if len(lead) == depth == k - start and instr.dest not in reduce_vars:  # the leading run is unbroken
                lead.append(hi - lo)
            depth += 1
        elif instr.opcode is Opcode.ENDLOOP:
            depth -= 1
            if depth == 0 and (collapse := collapse_depth(lead)):
                teams[start] = collapse
    return teams


def c_literal(value: float | int, dtype: DType) -> str:
    """A scalar as a C literal of this dtype, including the awkward ones (infinities, NAN, int min).

    A half-width literal is the float it rounds to: backends compute in float, so this costs nothing.
    """
    if dtype in HALF_FLOATS:
        value = float(NUMPY_DTYPES[dtype].type(value))
    if dtype in FLOATS:
        value = float(value)
        if math.isinf(value):
            return "INFINITY" if value > 0 else "-INFINITY"
        if math.isnan(value):
            return "NAN"
        return str(value) if dtype == float64 else f"{value}f"  # a bare float literal is already C's double
    return "(-2147483647 - 1)" if value == -(2**31) else str(value)


def guard(valid: Valid, dtype: DType, types: dict[DType, str]) -> tuple[str, str]:
    """Wrap a read so it yields zero wherever the mask is off.

    Both arms carry the value's type. Left bare, a half read against a float zero converts either
    way, and the compiler stops on the ambiguity rather than choosing.
    """
    if not valid.bounds:
        return "", ""
    return f"({valid.render()}) ? ({types[dtype]})", f" : ({types[dtype]}){c_literal(0, dtype)}"


def fold_c(op: Op, dest: str, src: str) -> str:
    """Fold src into dest: a register for UPDATE, a buffer cell for ACCUM.

    MAX keeps dest when dest is NaN and takes src otherwise, so a NaN anywhere in a reduce survives
    it, which is what the numpy device does. Repeating dest is safe because it is either a plain
    name or a buffer cell, neither of which has a side effect to evaluate twice.
    """
    if op is Op.ADD:
        return f"{dest} = {dest} + {src};"
    return f"{dest} = ({dest} > {src} || {dest} != {dest}) ? {dest} : {src};"


# One arith op as C, over its operands ({0}, {1}, ...), the dtype spelled as a value ({t}), and
# the suffix that picks the float-width math function ({f}); unsuffixed, those are C's double ones.
ARITH_C = {
    Op.NEG: "-{0}",
    Op.EXP: "exp{f}({0})",
    Op.LOG: "log{f}({0})",
    Op.SQRT: "sqrt{f}({0})",
    Op.RECIP: "1.0{f} / {0}",
    Op.ADD: "{0} + {1}",
    Op.MUL: "{0} * {1}",
    Op.CMPLT: "({t})({0} < {1})",
    Op.XOR: "{0} ^ {1}",
    Op.SHL: "({t})((unsigned int){0} << (unsigned int){1})",
    Op.SHR: "({t})((unsigned int){0} >> (unsigned int){1})",
    Op.WHERE: "{0} != 0 ? {1} : {2}",
}


def arith_c(op: Op, srcs: list[str], dtype: DType, types: dict[DType, str]) -> str:
    if op is Op.ADD and dtype in INTS:  # ints wrap like numpy; C's signed + overflows into undefined
        return f"({types[dtype]})((unsigned int){srcs[0]} + (unsigned int){srcs[1]})"
    if op not in ARITH_C:
        raise NotImplementedError(f"no C lowering for {op}")
    return ARITH_C[op].format(*srcs, f="" if dtype == float64 else "f", t=types[dtype])


def value_c(instr: Instr, indent: str, prefix: str, types: dict[DType, str], store: dict[DType, str]) -> str:
    """One value-defining instruction as a C declaration, shared by both compiled backends.

    prefix is what the caller puts before buffer names: "_" for this backend's typed casts of the
    void* params, nothing for CUDA's typed params. types spells a dtype as a value, store spells it
    in memory; they differ only for the half-width floats on cuda, which compute as float. A CAST
    goes through the stored spelling, since casting to a half width means rounding to it.
    """
    decl = f"{indent}{types[instr.value_type]} {instr.dest} = "
    match instr.opcode:
        case Opcode.ACC:
            return f"{decl}{c_literal(instr.arg, instr.value_type)};"
        case Opcode.CONST:
            value, valid = instr.arg
            pre, post = guard(valid, instr.value_type, types)
            return f"{decl}{pre}{c_literal(value, instr.value_type)}{post};"
        case Opcode.LOAD:
            buf, index, valid = instr.arg
            pre, post = guard(valid, instr.value_type, types)
            return f"{decl}{pre}{prefix}{buf}[{index.render()}]{post};"
        case Opcode.ARITH:
            return f"{decl}{arith_c(instr.arg, list(instr.srcs), instr.value_type, types)};"
        case Opcode.CAST:
            return f"{decl}({store[instr.value_type]}){instr.srcs[0]};"
        case Opcode.GATHER:
            buf, index = instr.arg
            return f"{decl}{prefix}{buf}[{index.render()}];"
        case _:
            raise NotImplementedError(f"{instr.opcode} does not define a value")


def emit_function(nest: LoopNest) -> str:
    kernel = nest.kernel
    lines: list[str] = []
    params = [f"void* in{k}" for k in range(len(kernel.inputs))] + ["void* out"]
    lines.append(f"void {nest.name}({', '.join(params)}) {{")
    for k, node in enumerate(kernel.inputs):
        lines.append(f"  {C_TYPE[node.dtype]}* _in{k} = ({C_TYPE[node.dtype]}*)in{k};")
    # execute() hands every kernel a freshly allocated output, and an assign writes that rather than
    # the buffer it overwrites, so the output aliases no input. Without saying so the compiler must
    # assume a store to out could land in in0, and reloads every operand around an accumulating
    # loop. Inputs stay unqualified: two of them can be the same buffer (a @ a.transpose()).
    out_type = C_TYPE[kernel.target.dtype]
    lines.append(f"  {out_type}* restrict _out = ({out_type}*)out;")
    depth = 1
    # cc vectorises the innermost loop or it vectorises nothing, and a pad's mask left riding on that
    # loop's variable is what stops it; split_masked turns those checks into loop bounds instead
    instrs = split_masked(nest.instrs)
    teams = parallel_loops(nest, instrs)
    for k, instr in enumerate(instrs):
        indent = "  " * (depth + 1)
        match instr.opcode:
            case Opcode.LOOP:
                if (collapse := teams.get(k)) is not None:
                    fuse = f" collapse({collapse})" if collapse > 1 else ""
                    lines.append(f"{indent}#pragma omp parallel for{fuse} num_threads({team_size()})")
                lo, hi = loop_range(instr)
                lines.append(f"{indent}for (int {instr.dest} = {lo}; {instr.dest} < {hi}; {instr.dest}++) {{")
                depth += 1
            case Opcode.ENDLOOP:
                depth -= 1
                lines.append(f"{'  ' * (depth + 1)}}}")
            case Opcode.ACC | Opcode.CONST | Opcode.LOAD | Opcode.ARITH | Opcode.CAST | Opcode.GATHER:
                lines.append(value_c(instr, indent, "_", C_TYPE, C_TYPE))
            case Opcode.UPDATE:
                lines.append(indent + fold_c(instr.arg, instr.dest, instr.srcs[0]))
            case Opcode.STORE:
                buf, index = instr.arg
                lines.append(f"{indent}_{buf}[{index.render()}] = {instr.srcs[0]};")
            case Opcode.ACCUM:
                buf, index, fold = instr.arg
                lines.append(indent + fold_c(fold, f"_{buf}[{index.render()}]", instr.srcs[0]))
            case Opcode.SCATTER:
                buf, index = instr.arg
                # two iterations can name the same row, which is why parallel_loops leaves this nest serial
                lines.append(f"{indent}_{buf}[{index.render()}] += {instr.srcs[0]};")
    lines.append("}")
    return "\n".join(lines)


def unsupported_dtype(nest: LoopNest) -> DType | None:
    """A dtype in this nest that C_TYPE has no name for, if any.

    Every instruction, not just the buffers: a CAST fuses, so a nest whose inputs and target are
    all float32 can still compute an intermediate in float16.
    """
    dtypes = [node.dtype for node in nest.kernel.inputs] + [nest.kernel.target.dtype]
    dtypes += [instr.dtype for instr in nest.instrs if instr.dtype is not None]
    return next((dtype for dtype in dtypes if dtype not in C_TYPE), None)


def emit_c(nests: list[LoopNest]) -> str:
    parts = ["#include <math.h>", "#include <stdint.h>", ""]
    for nest in nests:
        if (dtype := unsupported_dtype(nest)) is not None:
            raise NotImplementedError(f"the c device has no {dtype}; run this graph on the numpy or cuda device")
        parts.append(emit_function(nest))
        parts.append("")
    return "\n".join(parts)


def compile_c(source: str) -> ctypes.CDLL:
    key = hashlib.sha256(source.encode()).hexdigest()
    if key in cache:
        return cache[key]
    tmpdir = Path(tempfile.mkdtemp(prefix="limn_c_"))
    tmpdirs.append(tmpdir)
    src_path = tmpdir / "kernel.c"
    lib_path = tmpdir / "kernel.so"
    src_path.write_text(source)
    result = subprocess.run(
        ["cc", *cc_flags(), "-shared", "-fPIC", "-o", str(lib_path), str(src_path), "-lm"],
        capture_output=True,
        text=True,
    )
    src_path.unlink()
    if result.returncode != 0:
        raise RuntimeError(f"cc failed:\n{result.stderr}")
    lib = ctypes.CDLL(str(lib_path))
    cache[key] = lib
    return lib


def cleanup() -> None:
    for d in tmpdirs:
        shutil.rmtree(d, ignore_errors=True)
    tmpdirs.clear()


atexit.register(cleanup)


class CDevice(CompiledDevice, HostDevice):
    """Runs op graphs as C: renders the loop-nest IR, compiles with cc, calls through ctypes.

    Planning, caching and the assign transaction come from CompiledDevice; buffers are host
    bytes from HostDevice. Identical source (a training loop's repeated step) reuses its
    shared library by hash.
    """

    def runners(self, nests: list[LoopNest]) -> list[Runner]:
        lib = compile_c(emit_c(nests))
        runners: list[Runner] = []
        for nest in nests:
            fn = getattr(lib, nest.name)
            fn.argtypes = [ctypes.c_void_p] * (len(nest.kernel.inputs) + 1)
            fn.restype = None
            runners.append(lambda inputs, outs, fn=fn: fn(*[b.ctypes.data for b in inputs], outs[0].ctypes.data))
        return runners

    def out_alloc(self, nb: int, zero: bool) -> Buffer:
        return (np.zeros if zero else np.empty)(nb, dtype=np.uint8)

    def commit(self, target: Buffer, value: Buffer) -> None:
        target[:] = value
