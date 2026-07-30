"""CUDA backend tests: every graph the numpy device can run, the cuda device must match."""

import numpy as np
import pytest
from conftest import GRAPHS, check, cudev, randf

from limn import Tensor, set_device
from limn.backend_cuda import BLOCK, GRID, CudaDevice, cache, has_cuda
from limn.device import NUMPY_DTYPES

pytestmark = pytest.mark.skipif(not has_cuda(), reason="no CUDA driver, device, or NVRTC found")


def check_cuda(t: Tensor, dev: CudaDevice | None = None, tol: float = 1e-5) -> None:
    """Diff against the numpy device. tol has to grow with how long the graph's reduces are.

    numpy sums pairwise and a kernel sums straight down its reduce axis, so the two drift apart
    by roughly the length of the reduce in units of float32's epsilon. At the corpus shapes that
    is nothing; over a couple of hundred elements it is not, and a cell that lands near zero by
    cancellation shows it as a large relative difference over a tiny absolute one.
    """
    check(dev or cudev, t, rtol=tol, atol=tol)


@pytest.mark.parametrize("name", list(GRAPHS))
def test_cuda_matches_numpy_device(name):
    a, b = Tensor(randf(3, 4)), Tensor(randf(3, 4))
    check_cuda(GRAPHS[name](a, b))


def test_matmul_4x5_5x3():
    a, b = Tensor(randf(4, 5)), Tensor(randf(5, 3))
    check_cuda(a @ b)


@pytest.mark.parametrize("name", list(GRAPHS))
def test_corpus_gradients_match_numpy_device(name):
    a = Tensor(randf(3, 4), requires_grad=True)
    b = Tensor(randf(3, 4), requires_grad=True)
    loss = GRAPHS[name](a, b).sum()
    if not loss.requires_grad:
        assert name == "cast round trip"  # casting detaches; every other graph must reach a leaf
        return
    loss.backward()
    grads = [g for g in (a.grad, b.grad) if g is not None]
    assert grads
    for grad in grads:
        check_cuda(grad)


def test_batched_matmul_forward_and_backward():
    a = Tensor(randf(2, 3, 4, 5), requires_grad=True)
    b = Tensor(randf(5, 6), requires_grad=True)  # broadcast against the batch dims
    out = a @ b
    check_cuda(out)
    out.sum().backward()
    check_cuda(a.grad)
    check_cuda(b.grad)


def test_backward_pass():
    x = Tensor(randf(4, 5), requires_grad=True)
    w = Tensor(randf(5, 3), requires_grad=True)
    loss = (x @ w).relu().sum()
    loss.backward()
    assert x.grad is not None and w.grad is not None
    check_cuda(loss)
    check_cuda(w.grad)
    check_cuda(x.grad)


# ---- the tiled matmul: big enough shapes that emit_tiled takes them, unlike the corpus above ----

# Every shape here is kept small on m*n*k, not on m*n. The oracle is the numpy device, which does
# not fuse, so a matmul there materialises the whole (m, n, k) broadcast: an int64 index array of
# that shape, both operands gathered through it, and their product. That is around 20 bytes per
# m*n*k, so a shape a GPU shrugs at (8192x192x768 is 1.2e9 points, over 20 GB) takes the host down
# long before the kernel is reached. Tiling needs m*n*k >= TILE_MIN and both sides >= BLOCK/TILE_K,
# which these clear with room to spare.


@pytest.mark.parametrize(
    "m,k,n",
    [
        (128, 128, 128),  # the widest tile on both sides, every extent dividing it
        (129, 40, 65),  # no tile divides any side, so every tail check is live
        (64, 96, 160),  # a middle tile against the widest one, with a tail on the columns
        (33, 64, 160),  # the narrowest tile the block can stage, one row of tail over it
        (96, 33, 96),  # both sides land between tiles, and k leaves a remainder against TILE_K
        (16, 192, 256),  # rows too short to stage, so this one must fall back and still be right
    ],
)
@pytest.mark.parametrize("transposed", [False, True])
def test_a_matmul_big_enough_to_tile_matches_the_numpy_device(m, k, n, transposed):
    a = Tensor(randf(m, k))
    b = Tensor(randf(n, k)) if transposed else Tensor(randf(k, n))
    check_cuda(a @ (b.transpose() if transposed else b), tol=1e-4)


def test_a_tiled_matmul_with_elementwise_work_fused_in_matches_the_numpy_device():
    """matmul_shape reads through a fused body, so the tile has to compute it per cell."""
    a, b = Tensor(randf(128, 64)), Tensor(randf(64, 128))
    check_cuda((a * 2.0) @ (b + 1.0), tol=1e-4)


def test_a_batched_matmul_big_enough_to_tile_matches_the_numpy_device():
    """Batch dims belong to neither side: each point of them gets its own tiles."""
    a, b = Tensor(randf(4, 64, 64)), Tensor(randf(4, 64, 128))
    check_cuda(a @ b, tol=1e-4)


def test_a_tiled_matmul_backward_matches_the_numpy_device():
    """Both gradients are matmuls too, so this covers three tiled nests, not one."""
    x = Tensor(randf(128, 96), requires_grad=True)
    w = Tensor(randf(96, 128), requires_grad=True)
    (x @ w).relu().sum().backward()
    check_cuda(x.grad, tol=1e-4)
    check_cuda(w.grad, tol=1e-4)


def test_grid_stride_covers_more_elements_than_one_launch_has_threads():
    n = 2 * GRID * BLOCK + 3  # forces every thread around the stride loop at least twice
    a, b = Tensor(randf(n)), Tensor(randf(n))
    check_cuda(a * 2.0 + b)


def test_int32_elementwise():
    a = Tensor(np.arange(-6, 6, dtype=np.int32).reshape(3, 4))
    check_cuda(a + 1)
    check_cuda((a > 0).where(a, -a))


def test_assign_deferred():
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    expected = p.numpy() * 2.0 + 1.0
    p.assign(p * 2.0 + 1.0)
    bufs = cudev.execute([p.node])
    got = cudev.copyout(bufs[0]).view(NUMPY_DTYPES[p.dtype]).reshape(p.shape)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_assign_to_a_host_tensor_writes_the_host_bytes_back():
    """These tensors live on the numpy device; the assign must land in their host buffer."""
    p = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    p.assign(p * 10.0)
    cudev.execute([p.node])
    host = p.node.srcs[0].arg.view(np.float32)
    np.testing.assert_allclose(host, np.array([10.0, 20.0, 30.0], dtype=np.float32))


def test_assign_deferral_reads_pre_assign_bytes():
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    old_value = p * 10.0
    p.assign(p + 100.0)
    bufs = cudev.execute([old_value.node, p.node])
    got_old = cudev.copyout(bufs[0]).view(NUMPY_DTYPES[old_value.dtype]).reshape(old_value.shape)
    got_new = cudev.copyout(bufs[1]).view(NUMPY_DTYPES[p.dtype]).reshape(p.shape)
    np.testing.assert_allclose(got_old, np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
    np.testing.assert_allclose(got_new, np.array([[101.0, 102.0], [103.0, 104.0]], dtype=np.float32))


def test_assign_consumed_as_value():
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    p.assign(p * 2.0)
    consumer = p + 1.0
    bufs = cudev.execute([consumer.node])
    got = cudev.copyout(bufs[0]).view(NUMPY_DTYPES[consumer.dtype]).reshape(consumer.shape)
    np.testing.assert_allclose(got, np.array([[3.0, 5.0], [7.0, 9.0]], dtype=np.float32))


def test_multi_kernel_chain():
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    check_cuda((a + b).reshape(6) * 2.0)


def test_contiguous_copy():
    a = Tensor(randf(3, 4))
    check_cuda(a.transpose().reshape(12))


def test_shared_subgraph():
    a, b = Tensor(randf(2, 3)), Tensor(randf(2, 3))
    shared = (a + b).sum(axis=1, keepdim=True)
    check_cuda(shared * 2.0)
    check_cuda(shared * 3.0)


def test_gather_rows_forward_and_backward():
    table = Tensor(randf(6, 4), requires_grad=True)
    indices = Tensor(np.array([[5, 1, 1], [0, 3, 5]], dtype=np.int32))  # repeats, so the scatter accumulates
    gathered = table.gather_rows(indices)
    check_cuda(gathered)
    (gathered * Tensor(randf(2, 3, 4))).sum().backward()
    assert table.grad is not None
    check_cuda(table.grad)


def test_scatter_collisions_hit_one_row_atomically():
    """512 gathered rows all name row 2, so 512 threads add into the same cells."""
    table = Tensor(randf(4, 8), requires_grad=True)
    indices = Tensor(np.full(512, 2, dtype=np.int32))
    (table.gather_rows(indices) * Tensor(randf(512, 8))).sum().backward()
    expected = table.grad.numpy()
    bufs = cudev.execute([table.grad.node])
    got = cudev.copyout(bufs[0]).view(np.float32).reshape(4, 8)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)  # adds land in atomic order, not loop order


def test_a_large_full_sum_splits_and_matches_numpy():
    data = randf(1 << 22)
    total = Tensor(data).sum()
    expected = total.numpy()
    got = cudev.copyout(cudev.execute([total.node])[0]).view(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-4)  # split grouping reassociates the float adds


def test_a_large_full_max_is_exact():
    biggest = Tensor(randf(1 << 22)).max()
    expected = biggest.numpy()
    got = cudev.copyout(cudev.execute([biggest.node])[0]).view(np.float32)
    np.testing.assert_array_equal(got, expected)


def test_a_large_int_sum_is_exact():
    values = np.arange(1 << 20, dtype=np.int32)
    total = Tensor(values).sum()
    got = cudev.copyout(cudev.execute([total.node])[0]).view(np.int32)
    np.testing.assert_array_equal(got, values.sum(dtype=np.int32))  # int folds associate modulo 2**32


def test_a_split_reduce_is_deterministic():
    total = Tensor(randf(1 << 20)).sum()
    once = cudev.copyout(cudev.execute([total.node])[0])
    again = cudev.copyout(cudev.execute([total.node])[0])
    np.testing.assert_array_equal(once, again)


def test_a_split_reduce_with_a_fused_and_masked_body():
    x = Tensor(randf(100_000))
    r = (x * 2.0 + 1.0).pad(((3, 5),)).sum()
    expected = r.numpy()
    got = cudev.copyout(cudev.execute([r.node])[0]).view(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-4)


def test_the_pool_recycles_dropped_buffers():
    dev = CudaDevice()
    first = dev._alloc(1024)
    ptr = first.ptr
    del first
    second = dev._alloc(1024)
    assert second.ptr == ptr
    del second
    dev.trim()
    assert not dev.pool
    assert dev._alloc(1024).ptr != 0


def test_a_repeated_graph_reuses_the_plan_and_reads_fresh_bytes():
    dev = CudaDevice()
    for x in (randf(3, 4), randf(3, 4)):
        check_cuda((Tensor(x) * 2.0).sum(axis=1), dev)
    assert len(dev.plans) == 1


def test_a_changed_constant_is_a_different_plan():
    dev = CudaDevice()
    x = randf(2, 3)
    for scale in (2.0, 3.0):
        check_cuda(Tensor(x) * scale, dev)
    assert len(dev.plans) == 2


def test_a_repeated_assign_commits_through_the_cached_plan():
    from limn import device

    set_device("cuda")
    active = device.active()
    assert isinstance(active, CudaDevice)
    p = Tensor(np.ones(4, dtype=np.float32))
    for _ in range(3):
        p.assign(p * 2.0)
        p.realize()
    np.testing.assert_allclose(p.numpy(), np.full(4, 8.0, dtype=np.float32))
    assert len(active.plans) == 2  # one plan for the assign step, one for numpy()'s read


def test_adamw_compiles_one_program_for_all_steps():
    from limn.optim import AdamW

    set_device("cuda")
    p = Tensor(np.ones((4, 4), dtype=np.float32), requires_grad=True)
    opt = AdamW([p], lr=0.1)
    cache.clear()
    for _ in range(3):
        p.grad = Tensor(np.ones((4, 4), dtype=np.float32))
        opt.step()
    assert len(cache) == 1, f"{len(cache)} programs compiled for 3 steps"


def test_an_optimizer_step_trains_on_cuda():
    from limn.nn import Linear, parameters
    from limn.optim import SGD

    set_device("cuda")
    layer = Linear(4, 3)
    x = Tensor(np.random.default_rng(3).random((8, 4)).astype(np.float32))
    losses = []
    opt = SGD(parameters(layer), lr=0.05, momentum=0.9)
    for _ in range(5):
        opt.zero_grad()
        loss = (layer(x) ** 2).sum()
        loss.backward()
        losses.append(loss.item())
        opt.step()
    assert losses[-1] < losses[0]
