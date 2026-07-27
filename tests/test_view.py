"""Direct unit tests of View composition, checked against equivalent numpy op chains."""

import numpy as np
import pytest

from limn.view import View


def arange(*shape: int) -> np.ndarray:
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


def check(view: View, base: np.ndarray, expected: np.ndarray) -> None:
    got = view.materialize(base.reshape(-1))
    assert got.shape == expected.shape, f"shape {got.shape} != {expected.shape}"
    np.testing.assert_array_equal(got, expected)


def test_contiguous_identity():
    base = arange(2, 3)
    view = View.contiguous((2, 3))
    assert view.is_contiguous
    check(view, base, base)


def test_permute():
    base = arange(2, 3, 4)
    check(View.contiguous((2, 3, 4)).permute((2, 0, 1)), base, base.transpose(2, 0, 1))


def test_permute_of_expand():
    base = arange(3, 1)
    view = View.contiguous((3, 1)).expand((3, 5)).permute((1, 0))
    check(view, base, np.broadcast_to(base, (3, 5)).T)


def test_shrink_of_pad():
    base = arange(2, 3)
    view = View.contiguous((2, 3)).pad(((1, 1), (2, 0))).shrink(((0, 3), (1, 4)))
    padded = np.pad(base, ((1, 1), (2, 0)))
    check(view, base, padded[0:3, 1:4])


def test_pad_then_shrink_back_recovers_original():
    base = arange(2, 3)
    view = View.contiguous((2, 3)).pad(((1, 1), (2, 0))).shrink(((1, 3), (2, 5)))
    assert view.mask is None  # fully-valid masks normalize away
    check(view, base, base)


def test_pad_of_permute():
    base = arange(2, 3)
    view = View.contiguous((2, 3)).permute((1, 0)).pad(((0, 2), (1, 0)))
    check(view, base, np.pad(base.T, ((0, 2), (1, 0))))


def test_expand_of_padded_size1_dim():
    base = arange(1, 3)
    view = View.contiguous((1, 3)).pad(((0, 0), (1, 0))).expand((4, 4))
    check(view, base, np.broadcast_to(np.pad(base, ((0, 0), (1, 0))), (4, 4)))


def test_reshape_of_contiguous():
    base = arange(2, 3, 4)
    view = View.contiguous((2, 3, 4)).reshape((6, 4))
    assert view is not None and view.is_contiguous
    check(view, base, base.reshape(6, 4))


def test_reshape_adds_and_removes_unit_dims_on_permuted_view():
    base = arange(2, 3)
    permuted = View.contiguous((2, 3)).permute((1, 0))
    view = permuted.reshape((3, 1, 2))
    assert view is not None
    check(view, base, base.T.reshape(3, 1, 2))
    back = view.reshape((3, 2))
    assert back is not None
    check(back, base, base.T)


def test_size_1_dims_never_keep_a_live_stride():
    base = arange(4, 4)
    view = View.contiguous((4, 4)).shrink(((0, 1), (0, 4)))
    assert view.strides == (0, 1) and view.is_contiguous  # one row, read straight through
    reshaped = view.reshape((2, 2))
    assert reshaped is not None
    check(reshaped, base, base[0:1].reshape(2, 2))


def test_reshape_of_a_dense_slab_starting_partway_in():
    base = arange(4, 4)
    view = View.contiguous((4, 4)).shrink(((1, 3), (0, 4)))
    assert view.is_dense and not view.is_contiguous  # dense, but not from the start of the buffer
    reshaped = view.reshape((8,))
    assert reshaped is not None and reshaped.offset == 4
    check(reshaped, base, base[1:3].reshape(8))


def test_reshape_of_noncontiguous_needs_copy():
    transposed = View.contiguous((2, 3)).permute((1, 0))
    assert transposed.reshape((6,)) is None
    padded = View.contiguous((2, 3)).pad(((1, 0), (0, 0)))
    assert padded.reshape((9,)) is None


def test_reshape_of_expanded_needs_copy():
    view = View.contiguous((1, 3)).expand((4, 3))
    assert view.reshape((12,)) is None


def test_shrink_of_permute_offset():
    base = arange(4, 5)
    view = View.contiguous((4, 5)).permute((1, 0)).shrink(((2, 5), (1, 3)))
    check(view, base, base.T[2:5, 1:3])


def test_chain_pad_permute_shrink_expand():
    base = arange(2, 1)
    view = View.contiguous((2, 1)).pad(((1, 0), (0, 0))).permute((1, 0)).shrink(((0, 1), (1, 3))).expand((5, 2))
    expected = np.broadcast_to(np.pad(base, ((1, 0), (0, 0))).T[0:1, 1:3], (5, 2))
    check(view, base, expected)


def test_flip_like_negative_slice_not_supported_shapes_still_ok():
    base = arange(3, 4)
    view = View.contiguous((3, 4)).shrink(((1, 3), (0, 4))).pad(((0, 0), (2, 2)))
    check(view, base, np.pad(base[1:3], ((0, 0), (2, 2))))


def test_errors():
    view = View.contiguous((2, 3))
    with pytest.raises(ValueError):
        view.permute((0, 0))
    with pytest.raises(ValueError):
        view.expand((4, 3))
    with pytest.raises(ValueError):
        view.shrink(((0, 3), (0, 3)))
    with pytest.raises(ValueError):
        view.pad(((-1, 0), (0, 0)))
    with pytest.raises(ValueError):
        view.reshape((7,))
