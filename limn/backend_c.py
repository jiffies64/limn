"""C backend: emit C from the loop nest IR, compile with cc, execute via ctypes.

Each scheduled kernel becomes one C function taking void* pointers (one per input, one for
output), cast to the right type inside. The emitter is a second rendering of the same Instr
stream that render() in codegen.py prints; correctness is proven by diffing against NumpyDevice.
"""

from __future__ import annotations

import atexit
import ctypes
import functools
import hashlib
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from limn.codegen import Instr, LoopNest, Opcode, Valid
from limn.device import Buffer, HostDevice
from limn.jit import CompiledDevice, Runner
from limn.ops import DType, FLOATS, Op, float16, float32, int8, int16, int32

C_TYPE = {float32: "float", int32: "int32_t", int16: "int16_t", int8: "int8_t"}

cache: dict[str, ctypes.CDLL] = {}
tmpdirs: list[Path] = []


def has_cc() -> bool:
    return shutil.which("cc") is not None


@functools.cache
def cc_flags() -> tuple[str, ...]:
    """Optimisation flags for this machine's cc.

    A reordered nest only vectorises if the compiler may target the vector width this CPU actually
    has; plain -O3 compiles for baseline x86-64, which stops at SSE2. Not every cc takes
    -march=native (clang on arm64 rejects it), so probe rather than assume and fall back to -O3.

    Targeting the host also lets the compiler contract a multiply and an add into one FMA, so a
    float result can differ in the last bit from the numpy device's, and between two machines.
    test_backend_c.py diffs at 1e-5, which absorbs that. -ffp-contract=off would buy the bit back
    at most of the speed.
    """
    probe = subprocess.run(["cc", "-march=native", "-E", "-x", "c", "-"], input="", capture_output=True, text=True)
    return ("-O3", "-march=native") if probe.returncode == 0 else ("-O3",)


def c_literal(value: float | int, dtype: DType) -> str:
    """A scalar as a C literal of this dtype, including the awkward ones (infinities, NAN, int min).

    A float16 literal is the float it rounds to: backends compute in float, so this costs nothing.
    """
    if dtype == float16:
        value = float(np.float16(value))
    if dtype in FLOATS:
        value = float(value)
        if math.isinf(value):
            return "INFINITY" if value > 0 else "-INFINITY"
        if math.isnan(value):
            return "NAN"
        return f"{value}f"
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


def arith_c(op: Op, srcs: list[str], dtype: DType, types: dict[DType, str]) -> str:
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
            return f"({types[dtype]})({srcs[0]} < {srcs[1]})"
        case Op.WHERE:
            return f"{srcs[0]} != 0 ? {srcs[1]} : {srcs[2]}"
        case _:
            raise NotImplementedError(f"no C lowering for {op}")


def value_c(instr: Instr, indent: str, prefix: str, types: dict[DType, str], store: dict[DType, str]) -> str:
    """One value-defining instruction as a C declaration, shared by both compiled backends.

    prefix is what the caller puts before buffer names: "_" for this backend's typed casts of the
    void* params, nothing for CUDA's typed params. types spells a dtype as a value, store spells it
    in memory; they differ only for float16 on cuda, which computes as float. A CAST goes through
    the stored spelling, since casting to float16 means rounding to it.
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
    for instr in nest.instrs:
        indent = "  " * (depth + 1)
        match instr.opcode:
            case Opcode.LOOP:
                lines.append(f"{indent}for (int {instr.dest} = 0; {instr.dest} < {instr.arg}; {instr.dest}++) {{")
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
                # two iterations can name the same row, so a threaded backend owes this an atomic add
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
            runners.append(lambda inputs, out, fn=fn: fn(*[b.ctypes.data for b in inputs], out.ctypes.data))
        return runners

    def out_alloc(self, nb: int, zero: bool) -> Buffer:
        return (np.zeros if zero else np.empty)(nb, dtype=np.uint8)

    def commit(self, target: Buffer, value: Buffer) -> None:
        target[:] = value
