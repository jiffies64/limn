"""In-graph RNG and the Dropout layer built on it.

The oracle is Random123: threefry2x32-20 is a published spec, and the known-answer vectors
below are copied from its tests/kat_vectors (DEShawResearch/random123). A wrong hash still
looks perfectly random, so only fixed vectors catch it; the statistics and mask tests only
mean anything once the hash is pinned.
"""

import numpy as np
import pytest
from conftest import COMPILED, check, randf

from limn import Tensor, capture, int32, realize, set_device, set_seed
from limn.nn import Dropout
from limn.ops import Op
from limn.optim import SGD
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


# ---- the Dropout layer ----


def test_dropout_mask_and_gradient_agree():
    """The Phase-4 regression: forward, backward and the key's assign realize in one batch, so
    the gradients are diffed against the mask the forward actually used. A key that commits
    before the gradients are realized recomputes a different mask under them and breaks this."""
    drop = Dropout(0.5)
    x = Tensor(randf(6, 7), requires_grad=True)
    out = drop(x)
    out.sum().backward()
    assert x.grad is not None
    realize(out, x.grad, drop.key)
    out_np, grad_np = out.numpy(), x.grad.numpy()
    np.testing.assert_array_equal(grad_np == 0, out_np == 0)
    np.testing.assert_array_equal(grad_np[out_np != 0], np.float32(1 / (1 - 0.5)))


def test_dropout_statistics():
    drop = Dropout(0.3)
    out = drop(Tensor(np.ones((256, 256), dtype=np.float32))).numpy()
    assert (out == 0).mean() == pytest.approx(0.3, abs=0.01)
    np.testing.assert_array_equal(out[out != 0], np.float32(1 / (1 - 0.3)))  # survivors are x/(1-p) exactly


def test_dropout_key_advances_once_per_realized_step():
    drop = Dropout(0.5)
    start = drop.key.numpy().copy()
    x = Tensor(randf(4, 4))
    for step in range(2):
        out = drop(x)  # a second forward before the realize below would raise
        realize(out, drop.key)
        np.testing.assert_array_equal(drop.key.numpy(), start + np.array([0, step + 1], dtype=np.int32))


def test_dropout_second_forward_before_realize_raises():
    drop = Dropout(0.5)
    drop(Tensor(randf(4, 4)))
    with pytest.raises(ValueError, match="realized buffer"):
        drop(Tensor(randf(4, 4)))


def test_dropout_eval_leaves_the_key_bit_identical():
    drop = Dropout(0.5)
    realize(drop(Tensor(randf(4, 4))), drop.key)
    before = drop.key.numpy().copy()
    drop.training = False
    data = randf(5, 5)
    out = drop(Tensor(data))
    assert drop.key.node.op is Op.BUFFER  # eval queued no assign
    np.testing.assert_array_equal(drop.key.numpy(), before)
    np.testing.assert_array_equal(out.numpy(), data)  # eval returns x untouched


def test_two_dropout_layers_get_independent_streams():
    a, b = Dropout(0.5), Dropout(0.5)
    assert not np.array_equal(a.key.numpy(), b.key.numpy())  # seeds drawn separately at construction
    x = Tensor(np.ones((16, 16), dtype=np.float32))
    assert not np.array_equal(a(x).numpy() == 0, b(x).numpy() == 0)


def test_dropout_rejects_bad_probability_and_dtype():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        Dropout(1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        Dropout(-0.1)
    with pytest.raises(ValueError, match="float dtype"):
        Dropout(0.5)(Tensor(np.ones((2, 2), dtype=np.int32)))


@pytest.mark.parametrize("backend", COMPILED)
def test_dropout_forward_on_compiled_backends(backend):
    set_seed(0)
    drop = Dropout(0.5)
    check(backend.shared, drop(Tensor(randf(4, 16))))


@pytest.mark.parametrize("backend", COMPILED)
def test_a_captured_step_replays_advance_the_mask(backend):
    """The payoff: a replay runs the recorded kernels against the key buffer its own commits
    advance, so two replays draw different masks — the one thing a host-side dropout cannot
    do, since host state does not move when the step is replayed."""
    set_device(backend.name)
    drop = Dropout(0.5)
    w = Tensor(randf(8, 8), requires_grad=True)
    opt = SGD([w], lr=0.01)

    def step(x: Tensor) -> tuple[Tensor, Tensor]:
        y = drop(x @ w)
        loss = (y * y).sum()
        loss.backward()
        opt.step(loss, drop.key, y)
        opt.zero_grad()
        return loss, y

    x = Tensor(randf(8, 8))
    captured = capture(step)
    captured(x)  # first call settles one-time work
    captured(x)  # second runs and is recorded
    _, y1 = captured(x)  # every later call is a replay
    _, y2 = captured(x)
    assert not np.array_equal(y1.numpy() == 0, y2.numpy() == 0)
