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

from limn.codegen import LoopNest, Opcode, Valid
from limn.device import Buffer, HostDevice
from limn.jit import CompiledDevice, Runner
from limn.ops import DType, Op, float32, int32

C_TYPE = {float32: "float", int32: "int32_t"}

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
                # two iterations can name the same row, so a threaded backend owes this an atomic add
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
