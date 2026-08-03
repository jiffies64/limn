"""Codegen: lower each scheduled kernel into a printable loop nest.

A kernel becomes one nest over an iteration space: the output shape for elementwise work, the
pre-reduce shape for a reduce, whose axes become inner loops with an accumulator wrapped around
them. Every fused node turns into one instruction, so a node used twice inside a kernel is
computed once. Nothing is hoisted out of the reduce loops and nothing is shared between nests.

Which dim ends up innermost is a decision, not a given: loop_order picks it from the strides the
kernel reads at, since that is what decides how the nest walks memory. Putting a reduce axis
anywhere but innermost costs the register accumulator, so those nests keep their running totals
in the output buffer and fold into it, which is what ACCUM is.

Nodes are keyed by the loop variables they are read at, not just by identity, because a GATHER's
indices span the leading dims of a nest whose values span all of them.

Views turn into arithmetic here, which is the whole point of the layer. A VIEW's index into its
source buffer is affine in the loop variables (offset + sum(i_d * stride_d)) and its mask is a
conjunction of range checks on those same variables, so both are exact, both render, and both
evaluate: Index.at and Valid.at are what the tests check against View.materialize.

Masks are the one thing a backend may want restructured rather than rendered. A check on the
innermost loop's own variable guards a load per element, which is what stops a C compiler
vectorising; split_masked turns those checks into loop bounds instead, leaving the iterations and
their order alone. It is a transform on the instruction stream, not part of the lowering, because
whether it helps is a property of the backend: cc wants it, a GPU thread does not.

A loop nest is a plan. Nothing executes it, so nothing here allocates, compiles, or runs.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, NamedTuple

from limn.ops import DType, Node, Op, float16, float32, float64, int8, int16, int32
from limn.schedule import Kernel, addressed, realized, schedule
from limn.view import View, canonical_strides

if TYPE_CHECKING:
    from limn.tensor import Tensor

ARITH_OPS = (Op.NEG, Op.EXP, Op.LOG, Op.SQRT, Op.RECIP, Op.ADD, Op.MUL, Op.CMPLT, Op.WHERE)


class Reduce(NamedTuple):
    fold: Op  # the arith op folding each element into the accumulator
    identity: dict[DType, float | int]  # what the accumulator starts at, so folding the first element is exact


REDUCES = {
    Op.SUM: Reduce(Op.ADD, {float64: 0.0, float32: 0.0, float16: 0.0, int32: 0, int16: 0, int8: 0}),
    Op.MAX: Reduce(
        Op.MAX,
        {
            float64: float("-inf"),
            float32: float("-inf"),
            float16: float("-inf"),
            int32: -(2**31),
            int16: -(2**15),
            int8: -(2**7),
        },
    ),
}


@dataclass(frozen=True)
class Index:
    """An affine index into a flat buffer: const + sum(coeff * var)."""

    const: int
    terms: tuple[tuple[str, int], ...] = ()

    def render(self) -> str:
        if not self.terms:
            return str(self.const)
        text = " + ".join(var if coeff == 1 else f"{var}*{coeff}" for var, coeff in self.terms)
        if self.const:
            text += f" + {self.const}" if self.const > 0 else f" - {-self.const}"
        return text

    def at(self, loops: Mapping[str, int]) -> int:
        return self.const + sum(coeff * loops[var] for var, coeff in self.terms)


@dataclass(frozen=True)
class Valid:
    """A mask: lo <= var < hi for each padded dim, over a dim of that size.

    Carrying the size lets rendering drop the half of a check that covers the whole dim, which is
    most of them: a pad on one side only constrains one side.
    """

    bounds: tuple[tuple[str, int, int, int], ...] = ()  # (var, lo, hi, size)

    @property
    def never(self) -> bool:
        return any(lo >= hi for _, lo, hi, _ in self.bounds)

    def render(self) -> str:
        checks = []
        for var, lo, hi, size in self.bounds:
            if lo > 0:
                checks.append(f"{var} >= {lo}")
            if hi < size:
                checks.append(f"{var} < {hi}")
        return " && ".join(checks)

    def at(self, loops: Mapping[str, int]) -> bool:
        return all(lo <= loops[var] < hi for var, lo, hi, _ in self.bounds)

    def given(self, var: str, lo: int, hi: int) -> Valid:
        """This mask, knowing `var` only takes values in [lo, hi).

        A check that range satisfies outright is dropped, which is the point: what is left says
        nothing about `var`. A check it fails outright leaves an empty bound, so `never` holds and
        the whole read is a constant zero. A check it straddles has to stay.
        """
        bounds: list[tuple[str, int, int, int]] = []
        for bound in self.bounds:
            other, blo, bhi, size = bound
            if other != var:
                bounds.append(bound)
            elif bhi <= lo or hi <= blo:
                return Valid(((var, 0, 0, size),))
            elif not (blo <= lo and hi <= bhi):
                bounds.append(bound)
        return Valid(tuple(bounds))


def index_of(view: View, loop_vars: Sequence[str]) -> tuple[Index, Valid]:
    """A view's read, as index arithmetic over the loop variables: one variable per dim."""
    assert len(view.shape) == len(loop_vars), f"view {view.shape} does not match loops {tuple(loop_vars)}"
    terms = tuple((var, stride) for var, stride in zip(loop_vars, view.strides) if stride != 0)
    if view.mask is None:
        return Index(view.offset, terms), Valid()
    bounds = tuple((var, lo, hi, size) for var, (lo, hi), size in zip(loop_vars, view.mask, view.shape) if (lo, hi) != (0, size))
    return Index(view.offset, terms), Valid(bounds)


def contiguous_index(shape: tuple[int, ...], loop_vars: Sequence[str]) -> Index:
    """Row-major index into a buffer of this shape, read at the loop variables."""
    assert len(shape) == len(loop_vars), f"shape {shape} does not match loops {tuple(loop_vars)}"
    return Index(0, tuple((var, stride) for var, stride in zip(loop_vars, canonical_strides(shape)) if stride != 0))


class Opcode(Enum):
    LOOP = auto()  # dest is the loop variable, arg its bound, or the (lo, hi) a split left behind
    ENDLOOP = auto()  # arg is the loop variable it closes
    CONST = auto()  # arg is (value, Valid); zero wherever the check fails
    LOAD = auto()  # arg is (buffer, Index, Valid); zero wherever the check fails
    ARITH = auto()  # arg is the graph Op, srcs are its operands
    CAST = auto()  # dtype is the target
    ACC = auto()  # dest is the accumulator, arg its reduce identity
    UPDATE = auto()  # dest is the accumulator, arg the reduce Op folding srcs[0] into it
    STORE = auto()  # arg is (buffer, Index)
    ACCUM = auto()  # a STORE that folds into what the cell already holds; arg is (buffer, Index, reduce Op)
    GATHER = auto()  # a LOAD whose Index names a computed row; arg is (buffer, Index)
    SCATTER = auto()  # a STORE that adds rather than overwrites, at a computed row; arg is (buffer, Index)


@dataclass(frozen=True)
class Instr:
    """One instruction, shaped like a Node: a kind, what it defines, what it reads, one arg."""

    opcode: Opcode
    dest: str = ""
    dtype: DType | None = None
    srcs: tuple[str, ...] = ()
    arg: Any = None

    @property
    def value_type(self) -> DType:
        """The dtype of the value this instruction defines.

        LOOP and ENDLOOP are control flow and define no value, which is the whole reason dtype is
        optional; asking them for one is a bug in the caller, not a case to handle.
        """
        assert self.dtype is not None, f"{self.opcode.name} defines no typed value"
        return self.dtype


@dataclass(frozen=True)
class LoopNest:
    """A lowered kernel: the loops to run, and the instructions inside them."""

    name: str
    kernel: Kernel
    space: tuple[int, ...]  # the iteration space: one loop per dim
    instrs: tuple[Instr, ...]


def accesses(kernel: Kernel, ndim: int) -> list[tuple[int, ...]]:
    """The stride vector of every buffer this reduce kernel touches, one entry per loop dim.

    A VIEW carries its own strides. Anything else read straight out of a buffer is contiguous, as
    is the output. What a node addresses rather than reads is reached through that node's own
    strides, so it is not counted twice.

    Only a reduce kernel is asked: there every buffer spans the whole iteration space, which is
    what makes one stride per loop dim a complete answer. A GATHER's indices do not, and the
    assert below is what says so.
    """
    indexed = {src for node in kernel.body if (src := addressed(node)) is not None}
    strides = [canonical_strides(kernel.target.shape)]
    strides += [node.arg.strides for node in kernel.body if node.op is Op.VIEW and node.srcs[0].op is not Op.CONST]
    strides += [canonical_strides(node.shape) for node in kernel.inputs if node not in indexed]
    odd = sorted({len(s) for s in strides} - {ndim})
    assert not odd, f"a {ndim}-dim nest reads {odd}-dim buffers"
    return strides


def loop_order(kernel: Kernel, ndim: int, axes: tuple[int, ...]) -> tuple[int, ...]:
    """The dims' nesting, outermost first. Default is every non-reduce dim, then the reduce axes.

    The innermost loop decides how the nest walks memory. A dim that is stride-1 in an operand
    reads that operand element by neighbouring element, so each cache line arrives whole and the
    loop vectorises; a dim that is stride-1 in nothing pulls a fresh line per element and
    vectorises into nothing. A matmul's reduce axis is the second kind: it strides its right-hand
    operand by a whole row, so shape order alone would walk that operand a cache line at a time.

    So score each dim by how many of the kernel's buffers it is stride-1 in, and move the winner
    innermost. Only a strict win moves anything, because a reduce axis that stays innermost gets
    to accumulate in a register, which is worth a tie.
    """
    if not axes:  # nothing to weigh: shape order already puts a contiguous dim innermost
        return tuple(range(ndim))
    default = tuple(d for d in range(ndim) if d not in axes) + axes
    strides = accesses(kernel, ndim)
    score = [sum(1 for s in strides if s[d] == 1) for d in range(ndim)]
    best = score.index(max(score))
    if best in axes or score[best] <= score[default[-1]]:
        return default
    return tuple(d for d in default if d != best) + (best,)


def lower(kernel: Kernel, name: str) -> LoopNest:
    """Lay one kernel out as loops, values, and a store."""
    root = kernel.ast
    axes: tuple[int, ...] = root.arg if root.op in REDUCES else ()
    if root.op is Op.SCATTER:  # runs over the values being scattered, not over the table they land in
        space = root.srcs[1].shape
    else:
        space = root.srcs[0].shape if axes else root.shape
    loop_vars = tuple(f"{'r' if d in axes else 'i'}{d}" for d in range(len(space)))
    buffers = {node: f"in{k}" for k, node in enumerate(kernel.inputs)}
    instrs: list[Instr] = []
    values: dict[tuple[Node, tuple[str, ...]], str] = {}
    names = (f"v{k}" for k in itertools.count())

    def emit(opcode: Opcode, dtype: DType, srcs: tuple[str, ...] = (), arg: Any = None) -> str:
        dest = next(names)
        instrs.append(Instr(opcode, dest, dtype, srcs, arg))
        return dest

    def value_of(node: Node, at: tuple[str, ...]) -> str:
        """The name holding this node's value at these loop variables, one per dim of its shape."""
        if (node, at) not in values:
            values[node, at] = load(node, at) if node in buffers else lower_node(node, at)
        return values[node, at]

    def load(node: Node, at: tuple[str, ...]) -> str:
        return emit(Opcode.LOAD, node.dtype, arg=(buffers[node], contiguous_index(node.shape, at), Valid()))

    def lower_node(node: Node, at: tuple[str, ...]) -> str:
        match node.op:
            case Op.CONST:
                return emit(Opcode.CONST, node.dtype, arg=(node.arg, Valid()))
            case Op.VIEW:
                src = node.srcs[0]
                index, valid = index_of(node.arg, at)
                if valid.never:  # every element is masked out, so the view is a constant zero
                    return emit(Opcode.CONST, node.dtype, arg=(0, Valid()))
                if src.op is Op.CONST:
                    return emit(Opcode.CONST, node.dtype, arg=(src.arg, valid))
                return emit(Opcode.LOAD, node.dtype, arg=(buffers[src], index, valid))
            case Op.CAST:
                return emit(Opcode.CAST, node.dtype, srcs=(value_of(node.srcs[0], at),))
            case op if op in ARITH_OPS:
                return emit(Opcode.ARITH, node.dtype, srcs=tuple(value_of(src, at) for src in node.srcs), arg=op)
            case _:
                raise NotImplementedError(f"codegen has no rule for {node.op} inside a kernel")

    def row_address(table_shape: tuple[int, ...], index_src: Node) -> Index:
        """Where one element of a table lives: the row `index_src` names, at the column being looped.

        The row is a computed value rather than a loop variable, which Index does not distinguish:
        both are names that resolve to an integer wherever the nest runs. Passing the table's own
        shape is what checks it is 2D, since contiguous_index wants one variable per dim.
        """
        row = value_of(index_src, loop_vars[: len(index_src.shape)])
        return contiguous_index(table_shape, (row, loop_vars[-1]))

    def open_loops(dims: Sequence[int]) -> None:
        for d in dims:
            instrs.append(Instr(Opcode.LOOP, loop_vars[d], arg=space[d]))

    def close_loops(dims: Sequence[int]) -> None:
        for d in reversed(dims):
            instrs.append(Instr(Opcode.ENDLOOP, arg=loop_vars[d]))

    order = loop_order(kernel, len(space), axes)
    outer = tuple(d for d in order if d not in axes)

    if axes and order[-1] not in axes:
        # A reduce axis that is not innermost has loops inside it, so every point of those loops is
        # a separate running total and one register cannot hold them. They live in the output
        # instead: fill it with the reduce identity, then fold each element into the cell it belongs
        # to. A cell still sees its elements in ascending reduce order, which is what keeps this
        # bit-identical to the register form rather than merely close.
        out_index = contiguous_index(kernel.target.shape, loop_vars)
        open_loops(outer)
        identity = emit(Opcode.CONST, root.dtype, arg=(REDUCES[root.op].identity[root.dtype], Valid()))
        instrs.append(Instr(Opcode.STORE, dtype=root.dtype, srcs=(identity,), arg=("out", out_index)))
        close_loops(outer)
        open_loops(order)
        folded = value_of(root.srcs[0], loop_vars)
        instrs.append(Instr(Opcode.ACCUM, dtype=root.dtype, srcs=(folded,), arg=("out", out_index, REDUCES[root.op].fold)))
        close_loops(order)
        return LoopNest(name, kernel, space, tuple(instrs))

    open_loops(outer)
    if axes:
        instrs.append(Instr(Opcode.ACC, "acc", root.dtype, arg=REDUCES[root.op].identity[root.dtype]))
        open_loops(axes)
        # the operand is lowered here, after the reduce loops are open, so its loads sit inside them
        folded = value_of(root.srcs[0], loop_vars)
        instrs.append(Instr(Opcode.UPDATE, "acc", root.dtype, (folded,), REDUCES[root.op].fold))
        close_loops(axes)
        result = "acc"
    elif root.op in (Op.ASSIGN, Op.SCATTER):
        result = value_of(root.srcs[1], loop_vars)
    elif root.op is Op.CONTIGUOUS:
        result = value_of(root.srcs[0], loop_vars)
    elif root.op is Op.GATHER:
        table = root.srcs[0]
        result = emit(Opcode.GATHER, root.dtype, arg=(buffers[table], row_address(table.shape, root.srcs[1])))
    else:
        result = value_of(root, loop_vars)

    if root.op is Op.SCATTER:  # adds into a row the loop variables do not name, so it is the write
        write = Instr(Opcode.SCATTER, dtype=root.dtype, srcs=(result,), arg=("out", row_address(root.shape, root.srcs[0])))
    else:
        write = Instr(
            Opcode.STORE, dtype=root.dtype, srcs=(result,), arg=("out", contiguous_index(kernel.target.shape, loop_vars))
        )
    instrs.append(write)
    close_loops(outer)
    return LoopNest(name, kernel, space, tuple(instrs))


def lower_all(sinks: list[Node], order: list[Node] | None = None, kernels: list[Kernel] | None = None) -> list[LoopNest]:
    """Lower every scheduled kernel except the CUSTOM ones, which the device supplies whole:
    the skip lives here so ir() and every executor agree on what lowers."""
    kernels = schedule(sinks, order) if kernels is None else kernels
    return [lower(kernel, f"k{k}") for k, kernel in enumerate(k for k in kernels if k.ast.op is not Op.CUSTOM)]


MAX_PIECES = 4  # a mask cut finer than this is not worth a copy of the body per piece


def mask_of(instr: Instr) -> Valid:
    """The mask this instruction reads under; the ones that carry none read under an empty mask."""
    return instr.arg[-1] if instr.opcode in (Opcode.CONST, Opcode.LOAD) else Valid()


def loop_range(instr: Instr) -> tuple[int, int]:
    """A LOOP's half-open range: arg is a plain bound, or the (lo, hi) a split left behind."""
    return instr.arg if isinstance(instr.arg, tuple) else (0, instr.arg)


def matching_end(instrs: Sequence[Instr], start: int) -> int:
    """Where the LOOP opened at `start` is closed."""
    depth = 0
    for j in range(start, len(instrs)):
        depth += (instrs[j].opcode is Opcode.LOOP) - (instrs[j].opcode is Opcode.ENDLOOP)
        if depth == 0:
            return j
    raise AssertionError("unbalanced loops in the nest")


def resolved(instr: Instr, var: str, lo: int, hi: int) -> Instr:
    """This instruction with its mask settled against `var` in [lo, hi)."""
    valid = mask_of(instr).given(var, lo, hi)
    if valid.never:  # no value in this range is in bounds, so the read is a literal zero
        return Instr(Opcode.CONST, instr.dest, instr.dtype, arg=(0, Valid()))
    if instr.opcode in (Opcode.CONST, Opcode.LOAD):  # both keep their Valid last in arg
        return replace(instr, arg=(*instr.arg[:-1], valid))
    return instr


def cut(body: Sequence[Instr], var: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """[lo, hi) cut wherever the body's checks on `var` begin or end, so each piece settles them all."""
    edges = {lo, hi}
    edges |= {edge for instr in body for v, blo, bhi, _ in mask_of(instr).bounds if v == var for edge in (blo, bhi)}
    ordered = sorted(edge for edge in edges if lo <= edge <= hi)
    return list(zip(ordered, ordered[1:]))


def split_innermost(loop: Instr, body: Sequence[Instr], endloop: Instr) -> list[Instr]:
    """One innermost loop as the pieces its body's checks cut it into; one piece means nothing to split."""
    pieces = cut(body, loop.dest, *loop_range(loop))
    if not 2 <= len(pieces) <= MAX_PIECES:
        return [loop, *body, endloop]
    out: list[Instr] = []
    for lo, hi in pieces:
        piece = Instr(Opcode.LOOP, loop.dest, arg=(lo, hi))
        out += [piece, *(resolved(instr, loop.dest, lo, hi) for instr in body), endloop]
    return out


def split_masked(instrs: Sequence[Instr]) -> tuple[Instr, ...]:
    """Peel the innermost loop apart where a mask cuts it, so no check rides on that variable.

    A pad's mask is a range check on a loop variable, and the innermost loop re-tests it once per
    element. On a guarded *load* that is expensive out of proportion to the check: a C compiler
    turns it into a masked load and then gives up vectorising the loop around it, so the six taps
    of a padded 3x3 conv whose mask touches the innermost dim run scalar while the three whose
    mask does not run on vectors.

    But the check is loop structure, not data. Cutting the loop at the mask's edges leaves every
    piece wholly inside it or wholly outside it, so the check folds away in all of them: inside it
    is nothing, outside it the read is a literal zero. Same iterations, in the same order, so this
    is bit-identical rather than merely close.

    Only the innermost loop is worth this. It is the one that decides whether the nest vectorises,
    and splitting an outer loop would copy its whole subtree per piece to buy a few percent.
    """
    out: list[Instr] = []
    i = 0
    while i < len(instrs):
        instr = instrs[i]
        if instr.opcode is not Opcode.LOOP:
            out.append(instr)
            i += 1
            continue
        end = matching_end(instrs, i)
        body, endloop = instrs[i + 1 : end], instrs[end]
        if any(inner.opcode is Opcode.LOOP for inner in body):
            out += [instr, *split_masked(body), endloop]
        else:
            out += split_innermost(instr, body, endloop)
        i = end + 1
    return tuple(out)


def guard(valid: Valid) -> str:
    return f" if {valid.render()} else 0" if valid.bounds else ""


def render_instr(instr: Instr) -> str:
    match instr.opcode:
        case Opcode.LOOP:
            lo, hi = loop_range(instr)
            return f"LOOP    {instr.dest} < {hi}" if lo == 0 else f"LOOP    {lo} <= {instr.dest} < {hi}"
        case Opcode.ENDLOOP:
            return f"ENDLOOP {instr.arg}"
        case Opcode.ACC:
            return f"ACC     {instr.dest} = {instr.arg} : {instr.dtype}"
        case Opcode.CONST:
            value, valid = instr.arg
            return f"CONST   {instr.dest} = {value}{guard(valid)} : {instr.dtype}"
        case Opcode.LOAD:
            buf, index, valid = instr.arg
            return f"LOAD    {instr.dest} = {buf}[{index.render()}]{guard(valid)} : {instr.dtype}"
        case Opcode.ARITH:
            return f"ARITH   {instr.dest} = {instr.arg.name} {', '.join(instr.srcs)} : {instr.dtype}"
        case Opcode.CAST:
            return f"CAST    {instr.dest} = {instr.srcs[0]} : {instr.dtype}"
        case Opcode.UPDATE:
            return f"UPDATE  {instr.dest} = {instr.arg.name} {instr.dest}, {instr.srcs[0]}"
        case Opcode.STORE:
            buf, index = instr.arg
            return f"STORE   {buf}[{index.render()}] = {instr.srcs[0]}"
        case Opcode.ACCUM:
            buf, index, fold = instr.arg
            cell = f"{buf}[{index.render()}]"
            return f"ACCUM   {cell} = {fold.name} {cell}, {instr.srcs[0]}"
        case Opcode.GATHER:
            buf, index = instr.arg
            return f"GATHER  {instr.dest} = {buf}[{index.render()}] : {instr.dtype}"
        case Opcode.SCATTER:
            buf, index = instr.arg
            return f"SCATTER {buf}[{index.render()}] += {instr.srcs[0]}"


def render(nests: Sequence[LoopNest], sinks: Sequence[Node] = ()) -> str:
    """The whole schedule as text: a header and a loop nest per kernel.

    Buffers are numbered across the schedule in order of first use, so a value one nest writes and
    a later one reads carries the same name, and an ASSIGN that reads its own target shows that
    buffer on both sides. Given the sinks, the last lines say which buffer holds each one, which is
    worth printing because a sink that only reshapes its source has no kernel of its own.
    """
    ids: dict[Node, str] = {}

    def buf_id(node: Node) -> str:
        return ids.setdefault(node, f"buf{len(ids)}")

    lines: list[str] = []
    for nest in nests:
        kernel = nest.kernel
        lines.append(f"{nest.name}  {kernel.ast.op.name}  loop{list(nest.space)}")
        for k, node in enumerate(kernel.inputs):
            lines.append(f"    in{k} = {buf_id(node)}  {node.dtype}{list(node.shape)}")
        lines.append(f"    out = {buf_id(kernel.target)}  {kernel.target.dtype}{list(kernel.target.shape)}")
        depth = 1
        for instr in nest.instrs:
            depth -= instr.opcode is Opcode.ENDLOOP
            lines.append("  " * depth + render_instr(instr))
            depth += instr.opcode is Opcode.LOOP
        lines.append("")
    for k, sink in enumerate(sinks):
        lines.append(f"sink {k} = {buf_id(realized(sink))}  {sink.dtype}{list(sink.shape)}")
    return "\n".join(lines)


def ir(*tensors: Tensor) -> str:
    """The scheduled IR for these tensors' graphs, as one printable block."""
    sinks = [t.node for t in tensors]
    return render(lower_all(sinks), sinks)
