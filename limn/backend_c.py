"""C backend: emit C from the loop nest IR, compile with cc, execute via ctypes.

Each scheduled kernel becomes one C function taking void* pointers (one per input, one for
output), cast to the right type inside. The emitter is a second rendering of the same Instr
stream that render() in codegen.py prints; correctness is proven by diffing against NumpyDevice.

-march=native gets the nest the host's vector width, which is one core's worth of speed; the
cores come from OpenMP. A nest's leading non-reduce loops go to a thread team, so the threads
divide the output cells between them and each cell is still computed start to finish by one
thread, folding in the order the serial nest folds. That makes threading free of numerical
consequence: the same nest emitted without the pragmas produces the same bits, not merely the
same answer to a tolerance, which is what test_backend_c.py holds it to. Whether a cc has a
usable OpenMP runtime is not knowable up front, so it is probed, and the pragmas are simply not
emitted when the probe fails. OMP_NUM_THREADS still chooses the team size.
"""

from __future__ import annotations

import atexit
import ctypes
import functools
import glob
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from limn.codegen import LoopNest, Opcode, Valid, reduce_axes
from limn.device import Buffer, HostDevice
from limn.jit import CompiledDevice, Runner
from limn.ops import DType, Op, float32, int32

C_TYPE = {float32: "float", int32: "int32_t"}

PARALLEL_MIN = 1 << 15  # a nest smaller than this runs serial: waking a thread team costs more than it saves
CHUNKS_PER_THREAD = 4  # how many slabs of the parallel iteration space each thread should get to pick from

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
    """Whether this machine's cc compiles and links a probe with these flags.

    The probe goes all the way to a shared object rather than stopping at the preprocessor,
    because a driver can accept a flag and then fail at the link for want of a runtime library:
    clang takes -fopenmp on a machine with no libomp to link against.
    """
    with tempfile.TemporaryDirectory(prefix="limn_probe_") as tmp:
        src = Path(tmp) / "probe.c"
        src.write_text(source)
        result = subprocess.run(
            ["cc", *flags, "-shared", "-fPIC", "-o", str(Path(tmp) / "probe.so"), str(src)], capture_output=True, text=True
        )
    return result.returncode == 0


@functools.cache
def cc_flags() -> tuple[str, ...]:
    """Optimisation flags for this machine's cc.

    A reordered nest only vectorises if the compiler may target the vector width this CPU actually
    has; plain -O3 compiles for baseline x86-64, which stops at SSE2. Not every cc takes
    -march=native (clang on arm64 rejects it), so probe rather than assume and fall back to -O3.
    -fopenmp is probed the same way, and the emitter asks through openmp() whether it survived.

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
    kernels the whole team is idle while Python builds the next call, so on an SMT machine both
    siblings of every core sit there spinning and the thread doing the useful work has to share a
    core with one of them: measured on this laptop, a training step at one thread per logical
    processor runs several times slower than at one thread, while one per core is the fastest
    setting there is. Siblings share a core's vector units and its L1 anyway, so what the second
    one adds to a kernel already streaming memory is small.

    sysfs is the only place the topology is written down; without it (a non-Linux host) the logical
    count is the best guess available. Either way the affinity mask is the ceiling, so a run pinned
    to two cores asks for two threads. The answer is fixed on first use, since it is compiled into
    the kernels as a num_threads clause and the collapse decision is taken against it.
    """
    requested = os.environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()  # a list sets one count per nesting level
    if requested.isdigit() and int(requested) > 0:
        return int(requested)
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    cores = {Path(p).read_text() for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list")}
    return min(usable, len(cores)) if cores else usable


def collapse_depth(bounds: list[int]) -> int:
    """How many of a chain's leading loops to fuse into one parallel iteration space; 0 for none.

    Static scheduling cuts the space into one contiguous slab per thread, so a loop with fewer
    iterations than there are threads leaves cores idle, and one with only a couple of iterations
    each leaves them waiting on whichever thread drew the slowest core (not every core on a modern
    laptop runs at the same speed). Fusing the next loop in multiplies the slabs to divide up.
    Fusing is not free, since the fused index costs a division per slab, so it stops as soon as
    there is enough work to go round, which for most nests is at the first loop.
    """
    target = CHUNKS_PER_THREAD * team_size()
    extent, depth = 1, 0
    for bound in bounds:
        if extent >= target:
            break
        extent *= bound
        depth += 1
    return depth if extent > 1 else 0


def c_literal(value: float | int, dtype: DType) -> str:
    """A scalar as a C literal of this dtype, including the awkward ones (infinities, NAN, int min)."""
    if dtype == float32:
        value = float(value)
        if math.isinf(value):
            return "INFINITY" if value > 0 else "-INFINITY"
        if math.isnan(value):
            return "NAN"
        return f"{value}f"
    return "(-2147483647 - 1)" if value == -(2**31) else str(value)


def guard(valid: Valid, dtype: DType) -> tuple[str, str]:
    if not valid.bounds:
        return "", ""
    return f"({valid.render()}) ? ", f" : {c_literal(0, dtype)}"


def fold_c(op: Op, dest: str, src: str) -> str:
    """Fold src into dest: a register for UPDATE, a buffer cell for ACCUM.

    MAX keeps dest when dest is NaN and takes src otherwise, so a NaN anywhere in a reduce survives
    it, which is what the numpy device does. Repeating dest is safe because it is either a plain
    name or a buffer cell, neither of which has a side effect to evaluate twice.
    """
    if op is Op.ADD:
        return f"{dest} = {dest} + {src};"
    return f"{dest} = ({dest} > {src} || {dest} != {dest}) ? {dest} : {src};"


def arith_c(op: Op, srcs: list[str], dtype: DType) -> str:
    match op:
        case Op.NEG:
            return f"-{srcs[0]}"
        case Op.EXP:
            return f"expf({srcs[0]})"
        case Op.LOG:
            return f"logf({srcs[0]})"
        case Op.SQRT:
            return f"sqrtf({srcs[0]})"
        case Op.RECIP:
            return f"1.0f / {srcs[0]}"
        case Op.ADD:
            return f"{srcs[0]} + {srcs[1]}"
        case Op.MUL:
            return f"{srcs[0]} * {srcs[1]}"
        case Op.CMPLT:
            return f"({C_TYPE[dtype]})({srcs[0]} < {srcs[1]})"
        case Op.WHERE:
            return f"{srcs[0]} != 0 ? {srcs[1]} : {srcs[2]}"
        case _:
            raise NotImplementedError(f"no C lowering for {op}")


def parallel_loops(nest: LoopNest) -> dict[int, int]:
    """Which loops go to a thread team: the instruction index of a top-level loop -> loops to collapse.

    A nest has one or two top-level loop groups (a reduce that folds into the output fills it with
    the identity in a group of its own first), and each group is a single chain of loops. The
    leading non-reduce loops of a chain are the ones worth threading: the output's index is affine
    in exactly those variables, so two iterations of them never name the same cell, and everything
    the iteration carries is declared inside it. collapse_depth says how many of them to fuse, so
    that a nest whose outermost dim is short still fills the machine.

    Where the chain starts with a reduce axis, it stays serial. That axis carries a running total,
    in a register or in the output cell, that every later iteration depends on; splitting it needs
    partial accumulators, which regroup the additions and stop the answer being the serial one bit
    for bit. It costs the nests that reduce nearly everything they read: a full reduce, and a
    reduce whose one surviving dim loop_order moved innermost for being the stride-1 one.

    A SCATTER stays serial too. It adds into whichever rows its indices name, so two iterations can
    collide; an atomic add would fix the race and leave the sum's order up to thread timing, which
    is a bad trade for the host backend the other ones are diffed against.
    """
    if not openmp() or any(instr.opcode is Opcode.SCATTER for instr in nest.instrs):
        return {}
    reduce_vars = {f"r{d}" for d in reduce_axes(nest)}
    threaded: dict[int, int] = {}
    depth = start = work = 0
    lead: list[int] = []  # the bounds of the leading non-reduce loops of the group being read
    for k, instr in enumerate(nest.instrs):
        if instr.opcode is Opcode.LOOP:
            if depth == 0:
                start, work, lead = k, 1, []
            work *= instr.arg
            if len(lead) == depth and instr.dest not in reduce_vars:  # the leading run is still unbroken
                lead.append(instr.arg)
            depth += 1
        elif instr.opcode is Opcode.ENDLOOP:
            depth -= 1
            if depth == 0 and work >= PARALLEL_MIN and (collapse := collapse_depth(lead)):
                threaded[start] = collapse
    return threaded


def emit_function(nest: LoopNest) -> str:
    kernel = nest.kernel
    threaded = parallel_loops(nest)
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
    for k, instr in enumerate(nest.instrs):
        indent = "  " * (depth + 1)
        match instr.opcode:
            case Opcode.LOOP:
                if (collapse := threaded.get(k)) is not None:
                    fuse = f" collapse({collapse})" if collapse > 1 else ""
                    lines.append(f"{indent}#pragma omp parallel for{fuse} num_threads({team_size()})")
                lines.append(f"{indent}for (int {instr.dest} = 0; {instr.dest} < {instr.arg}; {instr.dest}++) {{")
                depth += 1
            case Opcode.ENDLOOP:
                depth -= 1
                lines.append(f"{'  ' * (depth + 1)}}}")
            case Opcode.ACC:
                lines.append(f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = {c_literal(instr.arg, instr.value_type)};")
            case Opcode.CONST:
                value, valid = instr.arg
                pre, post = guard(valid, instr.value_type)
                lines.append(
                    f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = {pre}{c_literal(value, instr.value_type)}{post};"
                )
            case Opcode.LOAD:
                buf, index, valid = instr.arg
                pre, post = guard(valid, instr.value_type)
                lines.append(f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = {pre}_{buf}[{index.render()}]{post};")
            case Opcode.ARITH:
                expr = arith_c(instr.arg, list(instr.srcs), instr.value_type)
                lines.append(f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = {expr};")
            case Opcode.CAST:
                lines.append(f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = ({C_TYPE[instr.value_type]}){instr.srcs[0]};")
            case Opcode.UPDATE:
                lines.append(indent + fold_c(instr.arg, instr.dest, instr.srcs[0]))
            case Opcode.STORE:
                buf, index = instr.arg
                lines.append(f"{indent}_{buf}[{index.render()}] = {instr.srcs[0]};")
            case Opcode.ACCUM:
                buf, index, fold = instr.arg
                lines.append(indent + fold_c(fold, f"_{buf}[{index.render()}]", instr.srcs[0]))
            case Opcode.GATHER:
                buf, index = instr.arg
                lines.append(f"{indent}{C_TYPE[instr.value_type]} {instr.dest} = _{buf}[{index.render()}];")
            case Opcode.SCATTER:
                buf, index = instr.arg
                # two iterations can name the same row, which is why parallel_loops leaves this nest serial
                lines.append(f"{indent}_{buf}[{index.render()}] += {instr.srcs[0]};")
    lines.append("}")
    return "\n".join(lines)


def emit_c(nests: list[LoopNest]) -> str:
    parts = ["#include <math.h>", "#include <stdint.h>", ""]
    for nest in nests:
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
        if not nests:
            return []
        lib = compile_c(emit_c(nests))
        runners: list[Runner] = []
        for nest in nests:
            fn = getattr(lib, nest.name)
            fn.argtypes = [ctypes.c_void_p] * (len(nest.kernel.inputs) + 1)
            fn.restype = None
            runners.append(lambda inputs, out, fn=fn: fn(*[b.ctypes.data for b in inputs], out.ctypes.data))
        return runners

    def out_alloc(self, nb: int, zero: bool) -> Buffer:
        return (np.zeros if zero else np.empty)(nb, dtype=np.uint8)

    def commit(self, target: Buffer, value: Buffer) -> None:
        target[:] = value
