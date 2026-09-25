"""CUDA emission: which kernel shape a nest gets, that the shape it gets is self-consistent, and
that NVRTC accepts what comes out.

Emission is pure text, so none of this needs a GPU or a driver. That is the point: the tiling
decision has invariants a machine without a card can still hold it to, and NVRTC compiles to
PTX without one too, so CI installs it (uv sync --extra cuda) and a broken emitter fails there
instead of on the next GPU run. What those kernels compute is checked separately, against the
numpy device, in test_backend_cuda.py.
"""

import re

import numpy as np
import pytest

from limn import Tensor
from limn.backend_cuda import nvrtc, pick_arch, ptx
from limn.codegen import LoopNest, lower_all
from limn.cuda_emit import BLOCK, SDPA_KERNELS, TILE_K, emit_cuda, emit_one, split_partials, stages_whole, tile_count, tiled
from limn.ops import DTYPES, FLOATS, DType, Op, float32, float64, int32
from limn.schedule import schedule
from limn.tensor import scatter_rows


def matmul_nest(m: int, k: int, n: int, transposed: bool = False) -> LoopNest:
    """The single kernel a plain (m, k) @ (k, n) lowers to."""
    a = Tensor.zeros((m, k))
    b = Tensor.zeros((n, k) if transposed else (k, n))
    out = a @ (b.transpose() if transposed else b)
    nests = lower_all([out.node])
    assert len(nests) == 1, f"expected one kernel, got {len(nests)}"
    return nests[0]


def staging_passes(source: str) -> list[tuple[int, int]]:
    """Every staging loop as (trip count, cells a thread takes per trip), in emission order."""
    trips = [int(bound) for bound in re.findall(r"for \(int step = 0; step < (\d+); step\+\+\)", source)]
    cells = [int(wide) for wide in re.findall(rf"const int slot = \(threadIdx\.x \+ step \* {BLOCK}\) \* (\d+);", source)]
    assert len(trips) == len(cells), f"{len(trips)} staging loops but {len(cells)} slot bindings"
    return list(zip(trips, cells, strict=True))


# ---- the staging floor: a tile the block cannot fill in whole passes is not a tile ----


def test_a_width_the_block_cannot_stage_whole_is_rejected():
    """BLOCK // TILE_K wide is the floor; under it the staging loop would not run at all."""
    assert not stages_whole(BLOCK // TILE_K // 2)
    assert stages_whole(BLOCK // TILE_K)
    assert stages_whole(BLOCK // TILE_K * 2)


def test_a_side_too_short_to_stage_falls_back_to_the_untiled_kernel():
    """A batch of 16 through a Linear: the rows side is under the floor, so nothing is tiled."""
    nest = matmul_nest(16, 192, 768, transposed=True)
    assert tiled(nest) is None
    assert not staging_passes(emit_one(nest))


@pytest.mark.parametrize("extent", range(8, 40))
def test_every_tile_a_short_side_wins_stages_whole_slabs(extent):
    """Sweep the window where a side is short enough for the narrowest tiles to come into play."""
    plan = tiled(matmul_nest(extent, 256, 512, transposed=True))
    if plan is not None:
        assert stages_whole(plan[1].rows) and stages_whole(plan[1].cols)


@pytest.mark.parametrize("m,k,n", [(16, 192, 768), (24, 256, 512), (31, 512, 64), (16, 256, 16), (64, 64, 20)])
def test_a_short_side_never_emits_a_staging_loop_that_does_not_run(m, k, n):
    """A zero-trip staging loop leaves the shared slab unwritten and the fold reads what it held."""
    for transposed in (False, True):
        assert all(trips > 0 for trips, _ in staging_passes(emit_one(matmul_nest(m, k, n, transposed))))


@pytest.mark.parametrize("m,k,n", [(512, 512, 512), (8192, 192, 768), (129, 40, 65), (256, 33, 256), (64, 96, 128)])
def test_a_staged_slab_is_covered_exactly_by_the_block(m, k, n):
    """Trip count times the cells a block takes per trip has to be the slab: no gap, no overlap."""
    for transposed in (False, True):
        nest = matmul_nest(m, k, n, transposed)
        plan = tiled(nest)
        assert plan is not None, f"{m}x{k}x{n} transposed={transposed} should tile"
        mm, spec = plan
        passes = staging_passes(emit_one(nest))
        assert len(passes) == len(mm.staged)
        for load, (trips, cells) in zip(mm.staged, passes, strict=True):
            assert trips * BLOCK * cells == spec.width(mm.on_cols(load)) * TILE_K


# ---- the tiling still has to reach every output cell ----


@pytest.mark.parametrize("m,k,n", [(512, 512, 512), (129, 40, 65), (8192, 192, 768), (33, 64, 4096)])
def test_the_tiles_cover_every_output_cell(m, k, n):
    nest = matmul_nest(m, k, n, transposed=True)
    plan = tiled(nest)
    assert plan is not None, f"{m}x{k}x{n} should tile"
    mm, spec = plan
    assert -(-m // spec.rows) * spec.rows >= m
    assert -(-n // spec.cols) * spec.cols >= n
    assert tile_count(mm, spec) == -(-m // spec.rows) * -(-n // spec.cols) * mm.extent(mm.batch)


def test_a_shallow_matmul_is_left_untiled():
    """Under TILE_K deep there is not enough reuse down the reduce axis to pay for staging."""
    assert tiled(matmul_nest(512, TILE_K - 1, 512)) is None


def test_a_split_reduce_beats_a_tile_where_both_would_take_the_nest():
    """Few output cells over a very long reduce: emit_one asks for the split first, and should."""
    nest = matmul_nest(32, 8192, 32, transposed=True)
    assert split_partials(nest) and tiled(nest) is not None
    assert "_part" in emit_one(nest)


# ---- NVRTC: every kernel form, at every dtype, has to compile ----

SCATTERS = (float32, float64, int32)  # the dtypes atomicAdd has an overload for that NVRTC can reach


def workload(dtype: DType) -> list[Tensor]:
    """A graph per kernel form at this dtype: elementwise, casts, reduces, the split reduce and the
    tiled matmul, then the atomic scatter where the device has one, and for a float, fused
    attention and the gradients of everything. The tensors are zeros because nothing here runs."""
    leaves: list[Tensor] = []

    def t(*shape: int) -> Tensor:
        leaves.append(Tensor.zeros(shape, dtype, requires_grad=dtype in FLOATS))
        return leaves[-1]

    a, b, long = t(3, 4), t(3, 4), t(1 << 18)
    outs = [
        (a + b) * 2 - a,
        (a < b).where(a, b).relu(),
        (a * b).max(axis=0, keepdim=True),
        (a.transpose() * 2).sum(axis=0),
        a.pad(((1, 1), (0, 2))).sum(axis=1),
        a.cast(float32),
        Tensor.zeros((3, 4)).cast(dtype),
        t(2048, 192).sum(-1),  # four-wide loads down the reduce
        long.sum(),
        long.max(),
        long.pad(((3, 5),)).sum(),
        t(129, 40) @ t(40, 65),
        t(129, 40) @ t(65, 40).transpose(),
        t(4, 64, 64) @ t(4, 64, 128),
        (t(128, 64) * 2) @ (t(64, 128) + 1),
    ]
    if dtype in SCATTERS:
        outs.append(scatter_rows(t(2, 3, 4), Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32)), (6, 4)))
    if dtype in FLOATS:
        outs += [a / (b * b + 0.5), (a * a + 1.0).log().exp().sqrt(), (a + b).softmax(axis=1), (a * b).log_softmax(axis=0)]
        q, k, v = t(2, 64, 16), t(2, 64, 16), t(2, 64, 24)
        outs += [q.attention(k, v, causal=True), q.attention(k, v, key_mask=Tensor.ones((2, 64), dtype))]
        sum((out.cast(float32).sum() for out in outs), Tensor.zeros(())).backward()
        outs += [leaf.grad for leaf in leaves if leaf.grad is not None]
    return outs


def sources(outs: list[Tensor]) -> set[str]:
    """What the cuda device hands NVRTC for these tensors: the lowered nests as one batch, as
    runners() compiles them, and each fused-attention kernel on its own."""
    sinks = [out.node for out in outs]
    kernels = schedule(sinks)
    custom = {SDPA_KERNELS[kernel.ast.arg.name][0](kernel.ast) for kernel in kernels if kernel.ast.op is Op.CUSTOM}
    return custom | {emit_cuda(lower_all(sinks, kernels=kernels))}


@pytest.mark.skipif(nvrtc() is None, reason="no NVRTC found (uv sync --extra cuda)")
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_every_kernel_form_compiles(dtype):
    """At the oldest architecture this NVRTC has, where pick_arch puts a device older than all of
    them, and where an instruction only newer cards have would go missing."""
    nv = nvrtc()
    assert nv is not None
    for source in sources(workload(dtype)):
        ptx(source, pick_arch(0, nv))
