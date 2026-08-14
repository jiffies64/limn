"""In-graph RNG and the Dropout layer built on it.

The oracle is Random123: threefry2x32-20 is a published spec, and the known-answer vectors
below are copied from its tests/kat_vectors (DEShawResearch/random123). A wrong hash still
looks perfectly random, so only fixed vectors catch it; the statistics and mask tests only
mean anything once the hash is pinned.
"""

import numpy as np
import pytest
from conftest import COMPILED, check, randf

from limn import Tensor, int32
from limn.tensor import _threefry2x32

# threefry2x32, R=20, from Random123's tests/kat_vectors: (counter, key) -> output, as uint32 words
KAT_VECTORS = [
    ((0x00000000, 0x00000000), (0x00000000, 0x00000000), (0x6B200159, 0x99BA4EFE)),
    ((0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 0xFFFFFFFF), (0x1CB996FC, 0xBB002BE7)),
    ((0x243F6A88, 0x85A308D3), (0x13198A2E, 0x03707344), (0xC4923A9C, 0x483DF7A0)),
]


def as_words(words: tuple[int, ...]) -> np.ndarray:
    """The kat file spells its words in hex; int32 holds the same bits."""
    return np.array(words, dtype=np.uint32).view(np.int32)


def test_threefry_matches_the_random123_kat_vectors():
    for counter, key, expected in KAT_VECTORS:
        h0, h1 = _threefry2x32(
            Tensor(as_words(counter[:1])), Tensor(as_words(counter[1:])), Tensor(as_words(key[:1])), Tensor(as_words(key[1:]))
        )
        np.testing.assert_array_equal(np.array([h0.item(), h1.item()], dtype=np.int32), as_words(expected))


@pytest.mark.parametrize("backend", COMPILED)
def test_threefry_is_bit_exact_on_compiled_backends(backend):
    """The composed hash through every backend, diffed for equality: this is what stress-tests
    the unsigned-wrapping templates, since threefry's additions overflow constantly."""
    n = 4096
    k0, k1 = Tensor(as_words((0x13198A2E,))), Tensor(as_words((0x03707344,)))
    h0, h1 = _threefry2x32(Tensor.arange(n), Tensor.const(0, int32), k0, k1)
    check(backend.shared, h0, exact=True)
    check(backend.shared, h1, exact=True)


def test_numpy_int32_addition_wraps():
    """The numpy reference wraps silently modulo 2**32; the C template's unsigned round trip owes it."""
    np.testing.assert_array_equal(np.array([2**31 - 1], np.int32) + np.int32(1), np.array([-(2**31)], np.int32))


BIT_VALUES = np.array([0x00000000, 0x00000001, 0xFFFFFFFF, 0x12345678, 0x80000000, 0x55555555], dtype=np.uint32)


def test_bit_ops_match_numpy():
    a, b = Tensor(BIT_VALUES.view(np.int32)), Tensor(np.array([0, 1, 1, 4, 31, 13], dtype=np.int32))
    np.testing.assert_array_equal((a ^ b).numpy(), BIT_VALUES.view(np.int32) ^ b.numpy())
    np.testing.assert_array_equal((a << b).numpy(), (BIT_VALUES << b.numpy().view(np.uint32)).view(np.int32))
    np.testing.assert_array_equal((a >> b).numpy(), (BIT_VALUES >> b.numpy().view(np.uint32)).view(np.int32))


def test_shifts_are_logical():
    """A signed >> would repeat the sign bit; SHR shifts in zeros on the uint32 representation."""
    np.testing.assert_array_equal((Tensor(np.array([-1], dtype=np.int32)) >> 1).numpy(), np.array([0x7FFFFFFF], np.int32))
    np.testing.assert_array_equal((Tensor(np.array([-(2**31)], dtype=np.int32)) >> 31).numpy(), np.array([1], np.int32))
    np.testing.assert_array_equal((Tensor(np.array([0x40000000], dtype=np.int32)) << 1).numpy(), np.array([-(2**31)], np.int32))


def test_bit_ops_refuse_float_dtypes():
    for op in (lambda t: t ^ t, lambda t: t << 1, lambda t: t >> 1):
        with pytest.raises(ValueError, match="int32"):
            op(Tensor(randf(3)))


@pytest.mark.parametrize("backend", COMPILED)
def test_bit_ops_and_wrapping_addition_are_bit_exact_on_compiled_backends(backend):
    a, b = Tensor(BIT_VALUES.view(np.int32)), Tensor(np.array([0, 1, 1, 4, 31, 13], dtype=np.int32))
    check(backend.shared, a ^ b, exact=True)
    check(backend.shared, a << b, exact=True)
    check(backend.shared, a >> b, exact=True)
    overflow = Tensor(np.array([2**31 - 1, -(2**31), 123456789], dtype=np.int32))
    check(backend.shared, overflow + overflow + 1, exact=True)  # signed overflow is UB; the template wraps
