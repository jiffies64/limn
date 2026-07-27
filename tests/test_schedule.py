"""Scheduler and codegen tests: where the graph gets cut, and whether the IR means what it says.

limn does not execute this IR, so the correctness tests bring their own interpreter. run_ir walks
the instructions with a dict of loop variables and writes real buffers, and its results are diffed
against NumpyDevice on the same graph, so a mistake in index arithmetic, masks, loop order, or
reduce identities shows up as a wrong number rather than as prose that reads fine.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from conftest import GRAPHS, randf

from limn import Tensor
from limn.codegen import Instr, LoopNest, Opcode, Valid, ir, lower_all
from limn.device import NUMPY_DTYPES
from limn.ops import DType, Node, Op, int32
from limn.schedule import Kernel, is_alias, realized, schedule

# ---- an interpreter for the IR, so the lowering can be checked against the numpy device ----


def scalar(dtype: DType, value: Any) -> Any:
    return NUMPY_DTYPES[dtype].type(value)


def arith(op: Op, srcs: list[Any]) -> Any:
    match op:
        case Op.NEG:
            return -srcs[0]
        case Op.EXP:
            return np.exp(srcs[0])
        case Op.LOG:
            return np.log(srcs[0])
        case Op.SQRT:
            return np.sqrt(srcs[0])
        case Op.RECIP:
            return 1 / srcs[0]
        case Op.ADD:
            return srcs[0] + srcs[1]
        case Op.MUL:
            return srcs[0] * srcs[1]
        case Op.CMPLT:
            return srcs[0] < srcs[1]
        case Op.WHERE:
            return srcs[1] if srcs[0] != 0 else srcs[2]
        case _:
            raise AssertionError(f"{op} should not appear as an arithmetic instruction")


def matching_end(instrs: Sequence[Instr], start: int) -> int:
    depth = 0
    for j in range(start, len(instrs)):
        depth += (instrs[j].opcode is Opcode.LOOP) - (instrs[j].opcode is Opcode.ENDLOOP)
        if depth == 0:
            return j
    raise AssertionError("unbalanced loops in the nest")


def run_block(
    instrs: Sequence[Instr], start: int, loops: dict[str, int], bufs: dict[str, np.ndarray], env: dict[str, Any]
) -> None:
    i = start
    while i < len(instrs):
        instr = instrs[i]
        match instr.opcode:
            case Opcode.ENDLOOP:
                return
            case Opcode.LOOP:
                end = matching_end(instrs, i)
                for value in range(instr.arg):
                    loops[instr.dest] = value
                    run_block(instrs, i + 1, loops, bufs, env)
                i = end + 1
                continue
            case Opcode.ACC:
                env[instr.dest] = scalar(instr.value_type, instr.arg)
            case Opcode.CONST:
                value, valid = instr.arg
                env[instr.dest] = scalar(instr.value_type, value if valid.at(loops) else 0)
            case Opcode.LOAD:
                buf, index, valid = instr.arg
                env[instr.dest] = scalar(instr.value_type, bufs[buf][index.at(loops)] if valid.at(loops) else 0)
            case Opcode.CAST:
                env[instr.dest] = scalar(instr.value_type, env[instr.srcs[0]])
            case Opcode.ARITH:
                env[instr.dest] = scalar(instr.value_type, arith(instr.arg, [env[src] for src in instr.srcs]))
            case Opcode.UPDATE:
                folded = env[instr.dest] + env[instr.srcs[0]] if instr.arg is Op.ADD else max(env[instr.dest], env[instr.srcs[0]])
                env[instr.dest] = scalar(instr.value_type, folded)
            case Opcode.STORE:
                buf, index = instr.arg
                bufs[buf][index.at(loops)] = env[instr.srcs[0]]
            case Opcode.ACCUM:
                buf, index, fold = instr.arg
                at = index.at(loops)
                held, new = bufs[buf][at], env[instr.srcs[0]]
                bufs[buf][at] = held + new if fold is Op.ADD else max(held, new)
            case Opcode.GATHER:  # the row is a value, so the index resolves against both scopes
                buf, index = instr.arg
                env[instr.dest] = scalar(instr.value_type, bufs[buf][index.at(loops | env)])
            case Opcode.SCATTER:
                buf, index = instr.arg
                bufs[buf][index.at(loops | env)] += env[instr.srcs[0]]
        i += 1


def blank(kernel: Kernel) -> np.ndarray:
    """The buffer a nest starts from: zeros for a scatter, which adds into them, and a value that
    cannot be a right answer for everything else, so a cell no instruction writes fails loudly."""
    dtype = NUMPY_DTYPES[kernel.target.dtype]
    fill = 0 if kernel.ast.op is Op.SCATTER else (np.nan if dtype.kind == "f" else np.iinfo(dtype).min)
    return np.full(math.prod(kernel.target.shape), fill, dtype=dtype)


def run_ir(nests: Sequence[LoopNest]) -> dict[Node, np.ndarray]:
    """Interpret a whole schedule, returning the flat buffer each nest wrote."""
    produced: dict[Node, np.ndarray] = {}
    for nest in nests:
        kernel = nest.kernel
        bufs = {f"in{k}": flat_buffer(node, produced) for k, node in enumerate(kernel.inputs)}
        bufs["out"] = blank(kernel)
        run_block(nest.instrs, 0, {}, bufs, {})
        produced[kernel.target] = bufs["out"]
    return produced


def flat_buffer(node: Node, produced: dict[Node, np.ndarray]) -> np.ndarray:
    return node.arg.view(NUMPY_DTYPES[node.dtype]) if node.op is Op.BUFFER else produced[node]


def check_ir(t: Tensor) -> None:
    """What the IR computes for this tensor, against what the numpy device computes."""
    assert t.node.op is not Op.ASSIGN, "run_ir does not implement the deferred-commit rule assigns need"
    produced = run_ir(lower_all([t.node]))
    got = flat_buffer(realized(t.node), produced).reshape(t.shape)
    np.testing.assert_allclose(got, t.numpy(), atol=1e-5, rtol=1e-5)


# ---- where the graph gets cut ----


def test_elementwise_chain_is_one_kernel():
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    kernels = schedule([((a + b) * 2.0).relu().node])
    assert len(kernels) == 1
    assert set(kernels[0].inputs) == {a.node, b.node}  # the constants stayed literals


def test_a_node_used_twice_in_a_kernel_is_computed_once():
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    product = a * b
    nest = lower_all([(product + product).node])[0]
    assert sum(1 for i in nest.instrs if i.opcode is Opcode.ARITH and i.arg is Op.MUL) == 1


def test_reduce_ends_a_kernel():
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    kernels = schedule([(a + b).sum(axis=1, keepdim=True).node])
    assert [k.ast.op for k in kernels] == [Op.SUM]
    assert kernels[0].body[-1].op is Op.SUM  # the add fused into the reduce


def test_matmul_is_one_reduce_kernel():
    a, b = Tensor(randf(4, 5)), Tensor(randf(5, 3))
    matmul = a @ b
    nests = lower_all([matmul.node])
    assert [n.kernel.ast.op for n in nests] == [Op.SUM]
    assert nests[0].space == (4, 3, 5)  # the reduce axis is part of the iteration space
    assert nests[0].kernel.inputs == (a.node, b.node)
    assert realized(matmul.node) is nests[0].kernel.target  # dropping the kept dim is free


def test_view_forces_a_computed_source_to_realize():
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    kernels = schedule([((a + b).reshape(6) * 2.0).node])
    assert [k.ast.op for k in kernels] == [Op.ADD, Op.MUL]
    assert kernels[1].inputs == (kernels[0].ast,)


def test_contiguous_is_its_own_kernel():
    a = Tensor(randf(3, 4))
    kernels = schedule([a.transpose().reshape(12).node])
    assert [k.ast.op for k in kernels] == [Op.CONTIGUOUS]
    assert kernels[0].inputs == (a.node,)  # the transpose fused in as index arithmetic


# ---- views that only reinterpret a buffer ----


def test_reshaping_a_realized_tensor_costs_nothing():
    a = Tensor(randf(2, 3))
    reshaped = a.reshape(6)
    assert is_alias(reshaped.node) and realized(reshaped.node) is a.node
    assert schedule([reshaped.node]) == []


def test_reshape_at_the_end_of_a_graph_aliases_the_buffer_under_it():
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    summed = (a + b).sum(axis=1)  # keepdim=False reshapes (2, 1) down to (2,)
    kernels = schedule([summed.node])
    assert [k.ast.op for k in kernels] == [Op.SUM]
    assert realized(summed.node) is kernels[0].target


def test_reshaping_a_dense_slab_costs_no_copy_kernel():
    a = Tensor(randf(8, 4))
    sliced = a.shrink(((2, 6), (0, 4))).reshape(16) * 2.0
    assert [k.ast.op for k in schedule([sliced.node])] == [Op.MUL]  # the slice fused in as index arithmetic
    np.testing.assert_allclose(sliced.numpy(), a.numpy()[2:6].reshape(16) * 2.0, atol=1e-6)


def test_a_view_that_moves_data_still_copies():
    a = Tensor(randf(3, 4))
    for moved in (a.transpose(), a.pad(((1, 0), (0, 0))), a.shrink(((0, 2), (0, 4)))):
        assert not is_alias(moved.node)
        assert [k.ast.op for k in schedule([moved.node])] == [Op.VIEW]


def test_sink_line_names_the_buffer_holding_each_result():
    a, b = Tensor(randf(4, 5)), Tensor(randf(5, 3))
    text = ir(a @ b)
    assert text.rstrip().endswith("sink 0 = buf2  float32[4, 3]")
    assert "out = buf2  float32[4, 3, 1]" in text  # the same buffer, seen through the kept dim


def test_sink_line_follows_an_assign_to_the_buffer_it_overwrites():
    p = Tensor(np.zeros((2, 2), dtype=np.float32))
    buffer = p.node
    p.assign(p * 2.0 + 1.0)
    assert realized(p.node) is buffer
    text = ir(p)
    assert "out = buf0  float32[2, 2]" in text  # the assign writes the buffer it also reads
    assert text.rstrip().endswith("sink 0 = buf0  float32[2, 2]")


def test_assign_writes_the_buffer_it_reads():
    p = Tensor(np.zeros((2, 2), dtype=np.float32))
    buffer = p.node
    p.assign(p * 2.0 + 1.0)
    kernels = schedule([p.node])
    assert len(kernels) == 1
    assert kernels[0].ast.op is Op.ASSIGN
    assert kernels[0].target is buffer and buffer in kernels[0].inputs


def test_constants_never_get_a_buffer():
    a = Tensor(randf(2, 3))
    kernels = schedule([((a * 2.0 + 1.0).relu()).node])
    assert all(k.target.op is not Op.CONST for k in kernels)
    assert all(node.op is not Op.CONST for k in kernels for node in k.inputs)


def test_kernels_come_in_dependency_order():
    x, w = Tensor(randf(4, 5), requires_grad=True), Tensor(randf(5, 3), requires_grad=True)
    loss = (x @ w).relu().sum()
    loss.backward()
    assert x.grad is not None and w.grad is not None
    kernels = schedule([loss.node, x.grad.node, w.grad.node])
    assert len(kernels) > 3
    produced: set[Node] = set()
    for kernel in kernels:
        for node in kernel.inputs:
            assert node.op is Op.BUFFER or node in produced, f"{kernel.ast} reads a buffer nobody wrote yet"
        produced.add(kernel.target)


def test_shared_subgraph_is_scheduled_once():
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    shared = (a + b).sum(axis=1, keepdim=True)
    kernels = schedule([(shared * 2.0).node, (shared * 3.0).node])
    assert sum(1 for k in kernels if k.ast.op is Op.SUM) == 1


# ---- what the lowering emits ----


def loop_groups(instrs: Sequence[Instr]) -> list[list[str]]:
    """The loop variables of each top-level nest, outermost first, asserting the loops balance."""
    groups: list[list[str]] = []
    open_vars: list[str] = []
    for instr in instrs:
        if instr.opcode is Opcode.LOOP:
            if not open_vars:
                groups.append([])
            groups[-1].append(instr.dest)
            open_vars.append(instr.dest)
        elif instr.opcode is Opcode.ENDLOOP:
            assert open_vars and open_vars.pop() == instr.arg, f"{instr.arg} closes out of order"
    assert not open_vars, f"never closed: {open_vars}"
    return groups


def prologue_identity(nest: LoopNest) -> list[Instr]:
    """The CONST a folding nest fills its output with, which is where its identity lives."""
    stored = {i.srcs[0] for i in nest.instrs if i.opcode is Opcode.STORE}
    return [i for i in nest.instrs if i.opcode is Opcode.CONST and i.dest in stored]


def test_a_folding_reduce_fills_its_output_with_the_identity():
    a = Tensor(randf(2, 3))  # reducing axis 0 leaves the stride-1 dim to move innermost, so all three fold
    assert [i.arg for i in prologue_identity(lower_all([a.sum(axis=0).node])[0])] == [(0.0, Valid())]
    assert [i.arg for i in prologue_identity(lower_all([a.max(axis=0).node])[0])] == [(float("-inf"), Valid())]
    ints = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int32))
    inits = prologue_identity(lower_all([ints.max(axis=0).node])[0])
    assert [i.arg for i in inits] == [(-(2**31), Valid())] and inits[0].dtype == int32


def test_a_reduce_over_the_last_axis_keeps_its_accumulator_in_a_register():
    a = Tensor(randf(4, 5))  # already stride-1 innermost, so nothing to gain by moving
    for reduced, identity in ((a.sum(axis=1), 0.0), (a.max(axis=1), float("-inf"))):
        nest = lower_all([reduced.node])[0]
        assert loop_groups(nest.instrs) == [["i0", "r1"]]
        assert [i.opcode for i in nest.instrs if i.opcode in (Opcode.ACC, Opcode.ACCUM)] == [Opcode.ACC]
        assert [i.arg for i in nest.instrs if i.opcode is Opcode.ACC] == [identity]


def test_matmul_moves_its_reduce_axis_out_of_the_innermost_loop():
    a, b = Tensor(randf(4, 5)), Tensor(randf(5, 3))
    nest = lower_all([(a @ b).node])[0]
    # i1 is stride-1 in both the right-hand operand and the output; r2 is stride-1 in neither
    assert loop_groups(nest.instrs) == [["i0", "i1"], ["i0", "r2", "i1"]]
    assert [i.opcode for i in nest.instrs if i.opcode in (Opcode.ACC, Opcode.ACCUM)] == [Opcode.ACCUM]


def test_an_elementwise_kernel_keeps_the_shape_order():
    a, b = Tensor(randf(2, 3, 4)), Tensor(randf(2, 3, 4))
    nest = lower_all([(a * b + 1.0).node])[0]
    assert loop_groups(nest.instrs) == [["i0", "i1", "i2"]]


def test_render_shows_the_nest_and_reuses_buffer_names():
    text = ir(Tensor(randf(4, 5)).sum(axis=1))
    assert "ACC     acc = 0.0 : float32" in text
    assert "UPDATE  acc = ADD acc," in text
    folded = ir(Tensor(randf(4, 5)) @ Tensor(randf(5, 3)))
    assert "ACCUM   out[i0*3 + i1] = ADD out[i0*3 + i1], " in folded
    chained = ir((Tensor(randf(2, 3)) + Tensor(randf(2, 3))).reshape(6) * 2.0)
    assert "out = buf2" in chained and "in0 = buf2" in chained  # one nest writes it, the next reads it


MOVEMENTS = {
    "transpose": lambda t: t.permute(2, 0, 1),
    "pad": lambda t: t.pad(((1, 2), (0, 1), (0, 0))),
    "shrink": lambda t: t.shrink(((0, 1), (1, 3), (2, 4))),
    "expand": lambda t: t.shrink(((0, 1), (0, 3), (0, 4))).expand(2, 3, 4),
    "pad then permute then shrink": lambda t: t.pad(((1, 1), (2, 0), (0, 0))).permute(1, 0, 2).shrink(((1, 4), (0, 3), (1, 3))),
    "pad an expanded dim": lambda t: t.shrink(((0, 1), (0, 3), (0, 4))).expand(2, 3, 4).pad(((0, 0), (1, 1), (0, 0))),
}


@pytest.mark.parametrize("name", list(MOVEMENTS))
def test_load_index_matches_view_materialize(name):
    flat = np.arange(24, dtype=np.float32)
    moved = MOVEMENTS[name](Tensor(flat.reshape(2, 3, 4)))
    view = moved.view()
    loads = [i for i in lower_all([moved.node])[0].instrs if i.opcode is Opcode.LOAD]
    assert len(loads) == 1
    _, index, valid = loads[0].arg
    got = np.zeros(view.shape, dtype=np.float32)
    for position in np.ndindex(*view.shape):
        loops = {f"i{d}": p for d, p in enumerate(position)}
        if valid.at(loops):
            got[position] = flat[index.at(loops)]
    np.testing.assert_array_equal(got, view.materialize(flat))


# ---- what the lowering means, against the numpy device ----


@pytest.mark.parametrize("name", list(GRAPHS))
def test_ir_matches_the_numpy_device(name):
    a, b = Tensor(randf(3, 4)), Tensor(randf(3, 4))
    check_ir(GRAPHS[name](a, b))


def test_ir_matches_the_numpy_device_for_a_backward_pass():
    x = Tensor(randf(4, 5), requires_grad=True)
    w = Tensor(randf(5, 3), requires_grad=True)
    loss = (x @ w).relu().sum()
    loss.backward()
    assert x.grad is not None and w.grad is not None
    check_ir(loss)
    check_ir(w.grad)
    check_ir(x.grad)


def test_kernel_target_is_the_assign_target():
    p = Tensor(np.zeros((2, 2), dtype=np.float32))
    buffer = p.node
    p.assign(Tensor(np.ones((2, 2), dtype=np.float32)) * 3.0)
    kernel: Kernel = schedule([p.node])[0]
    nest = lower_all([p.node])[0]
    assert kernel.target is buffer
    stores = [i for i in nest.instrs if i.opcode is Opcode.STORE]
    assert len(stores) == 1 and stores[0].arg[0] == "out"


# ---- indexed access ----


def test_expand_of_an_expression_still_realizes_it():
    """Recomputing an expanded source once per expanded point costs more than its buffer saves."""
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    widened = (a + b).reshape(2, 3, 1).expand(2, 3, 4)
    kernels = schedule([(widened * 2.0).node])
    assert [k.ast.op for k in kernels] == [Op.ADD, Op.MUL]
    check_ir(widened * 2.0)


def test_gather_is_its_own_kernel_over_the_result_shape():
    table = Tensor(randf(6, 4))
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))
    nests = lower_all([table.gather_rows(indices).node])
    assert [n.kernel.ast.op for n in nests] == [Op.GATHER]
    assert nests[0].space == (2, 3, 4)
    assert nests[0].kernel.inputs == (indices.node, table.node)  # the table is addressed, so it lands after


def test_scatter_runs_over_the_values_not_the_table():
    table = Tensor(randf(6, 4), requires_grad=True)
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))
    (table.gather_rows(indices) * Tensor(randf(2, 3, 4))).sum().backward()
    assert table.grad is not None
    nest = [n for n in lower_all([table.grad.node]) if n.kernel.ast.op is Op.SCATTER][0]
    assert nest.space == (2, 3, 4)  # the incoming gradient's shape, not the (6, 4) table's
    assert nest.kernel.target.shape == (6, 4)
    assert sum(1 for i in nest.instrs if i.opcode is Opcode.STORE) == 0  # the scatter is the write


def test_ir_matches_the_numpy_device_for_gather_and_its_gradient():
    table = Tensor(randf(6, 4), requires_grad=True)
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))
    gathered = table.gather_rows(indices)
    check_ir(gathered)
    (gathered * Tensor(randf(2, 3, 4))).sum().backward()
    assert table.grad is not None
    check_ir(table.grad)
