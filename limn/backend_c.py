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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from limn.codegen import LoopNest, Opcode, Valid, lower_all
from limn.device import Buffer, HostDevice
from limn.ops import DType, Node, Op, float32, int32, topological
from limn.schedule import realized

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


def nbytes(node: Node) -> int:
    return node.dtype.itemsize * math.prod(node.shape)


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


def graph_key(order: list[Node], position: dict[Node, int], sinks: list[Node]) -> tuple:
    """This graph's structure, as a hashable key: two graphs with the same key lower to the same C.

    A training loop rebuilds an identical graph every step, and scheduling plus emission cost more
    than running the kernels do, so plans are cached on what they depend on: each node's op, dtype,
    shape and arg, the wiring between them as positions in topological order, and which positions
    are the sinks. A BUFFER's bytes are the one thing lowering never looks at, only its shape and
    dtype, so its arg is left out; that is what lets the same plan serve every step. A CONST's arg
    stays in, because it is baked into the source as a literal.
    """
    at = position.__getitem__
    parts = tuple(
        (node.op, node.dtype, node.shape, None if node.op is Op.BUFFER else node.arg, tuple(map(at, node.srcs))) for node in order
    )
    return parts, tuple(map(at, sinks))


@dataclass(frozen=True, slots=True)
class Call:
    """One compiled kernel invocation, with its buffers named by position in topological order."""

    fn: Callable[..., None]  # the ctypes function, argtypes already set
    inputs: tuple[int, ...]
    output: int  # the position whose value this call produces (the ASSIGN node itself for assigns)
    out_nbytes: int
    zero_fill: bool  # a scatter adds into its output and reaches only the rows its indices name
    assign_target: int | None  # the BUFFER position an assign commits to, once every call has run


@dataclass(frozen=True, slots=True)
class Plan:
    calls: tuple[Call, ...]
    buffers: tuple[int, ...]  # the BUFFER positions, whose bytes come from this step's graph
    sinks: tuple[int, ...]  # realized() resolved to positions, so alias chains are walked once


class CDevice(HostDevice):
    """Executes op graphs by compiling the scheduled IR to C. Host-memory only for now.

    Plans are cached per graph structure (graph_key), so a repeated graph skips scheduling,
    emission and the source hash, and goes straight to calling the compiled kernels with this
    graph's buffers.
    """

    def __init__(self) -> None:
        self.plans: dict[tuple, Plan] = {}

    def execute(self, sinks: list[Node]) -> list[Buffer]:
        order = topological(sinks)
        position = {node: p for p, node in enumerate(order)}
        key = graph_key(order, position, sinks)
        plan = self.plans.get(key)
        if plan is None:
            plan = self.plans[key] = plan_of(order, position, sinks)

        bufs: list[Buffer] = [None] * len(order)
        for p in plan.buffers:
            bufs[p] = order[p].arg

        deferred: list[tuple[Buffer, Buffer]] = []
        for call in plan.calls:
            # a fresh output per call, never a buffer the kernel also reads, is what lets
            # emit_function mark _out restrict; an assign relies on it too, and commits below
            fill = np.zeros if call.zero_fill else np.empty
            out_buf = fill(call.out_nbytes, dtype=np.uint8)
            call.fn(*[bufs[p].ctypes.data for p in call.inputs], out_buf.ctypes.data)
            bufs[call.output] = out_buf
            if call.assign_target is not None:
                deferred.append((order[call.assign_target].arg, out_buf))

        for target_buf, src_buf in deferred:
            target_buf[:] = src_buf

        return [bufs[p] for p in plan.sinks]


def plan_of(order: list[Node], position: dict[Node, int], sinks: list[Node]) -> Plan:
    """Lower, emit and compile these sinks' kernels, wired up by position instead of by node."""
    nests = lower_all(sinks, order)
    lib = compile_c(emit_c(nests)) if nests else None
    calls = []
    for nest in nests:
        kernel = nest.kernel
        fn = getattr(lib, nest.name)
        fn.argtypes = [ctypes.c_void_p] * (len(kernel.inputs) + 1)
        fn.restype = None
        calls.append(
            Call(
                fn=fn,
                inputs=tuple(position[node] for node in kernel.inputs),
                output=position[kernel.ast],
                out_nbytes=nbytes(kernel.target),
                zero_fill=kernel.ast.op is Op.SCATTER,
                assign_target=position[kernel.target] if kernel.ast.op is Op.ASSIGN else None,
            )
        )
    return Plan(
        tuple(calls),
        tuple(p for p, node in enumerate(order) if node.op is Op.BUFFER),
        tuple(position[realized(sink)] for sink in sinks),
    )
