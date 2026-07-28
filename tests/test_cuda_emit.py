"""CUDA emission: which kernel shape a nest gets, and that the shape it gets is self-consistent.

Emission is pure text, so none of this needs a GPU, a driver, or NVRTC. That is the point: the
tiling decision has invariants a machine without a card can still hold it to. What those kernels
compute is checked separately, against the numpy device, in test_backend_cuda.py.
"""

import re

import pytest

from limn import Tensor, set_device
from limn.codegen import LoopNest, lower_all
from limn.cuda_emit import BLOCK, TILE_K, emit_one, split_partials, stages_whole, tile_count, tiled


@pytest.fixture(autouse=True)
def on_the_numpy_device():
    """Nothing here runs a kernel, so the buffers behind these shapes may as well stay on the host."""
    set_device("numpy")
    yield
    set_device("numpy")


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
