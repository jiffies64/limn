"""capture records a step's device work once and replays it; the c device proves the replays."""

import numpy as np
import pytest
from conftest import needs_cc, randf

from limn import Tensor, capture, set_device, set_seed
from limn.nn import Linear, parameters
from limn.optim import SGD, AdamW


def batches(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(randf(8, 4), randf(8, 3)) for _ in range(n)]


def run(data: list[tuple[np.ndarray, np.ndarray]], captured: bool, counter: dict | None = None) -> list[float]:
    counter = counter if counter is not None else {"ran": 0}
    set_seed(5)
    layer = Linear(4, 3)
    opt = AdamW(parameters(layer), lr=1e-2)

    def step(x: Tensor, y: Tensor) -> Tensor:
        counter["ran"] += 1
        opt.zero_grad()
        err = layer(x) - y
        loss = (err * err).mean()
        loss.backward()
        opt.step(loss)
        return loss

    fn = capture(step) if captured else step
    return [fn(Tensor(x), Tensor(y)).item() for x, y in data]


@needs_cc
def test_captured_steps_match_plain_steps_bit_for_bit():
    set_device("c")
    data = batches(6)
    plain = run(data, captured=False)
    counter = {"ran": 0}
    replayed = run(data, captured=True, counter=counter)
    assert replayed == plain  # same kernels against the same bytes, so equality is exact
    assert counter["ran"] == 2  # observed twice, replayed four times


@needs_cc
def test_a_mid_call_realize_feeds_later_kernels_fresh_bytes():
    """A value realized partway through the call re-enters later plans as a buffer input; a
    replay must serve that replay's bytes there, not the recorded call's."""
    set_device("c")
    w = Tensor(randf(4))

    def step(x: Tensor) -> Tensor:
        h = (x * 2.0).realize()
        return (h + w).sum().realize()

    data = [randf(4) for _ in range(5)]
    plain = [step(Tensor(d)).item() for d in data]
    fn = capture(step)
    assert [fn(Tensor(d)).item() for d in data] == plain


@needs_cc
def test_each_replay_returns_its_own_tensor():
    set_device("c")

    def step(x: Tensor) -> Tensor:
        return (x + 1.0).sum().realize()

    fn = capture(step)
    fn(Tensor(randf(4)))
    fn(Tensor(randf(4)))
    third = fn(Tensor(np.full(4, 1.0, dtype=np.float32)))
    fourth = fn(Tensor(np.full(4, 2.0, dtype=np.float32)))
    assert third is not fourth
    assert (third.item(), fourth.item()) == (8.0, 12.0)  # the earlier reading survives the later replay


@needs_cc
def test_replay_rejects_an_unrealized_argument():
    set_device("c")

    def step(x: Tensor) -> Tensor:
        return (x * 2.0).sum().realize()

    fn = capture(step)
    fn(Tensor(randf(4)))
    fn(Tensor(randf(4)))
    with pytest.raises(ValueError, match="buffer-backed"):
        fn(Tensor(randf(4)) + 1.0)


@needs_cc
def test_replay_rejects_arguments_of_another_shape():
    set_device("c")

    def step(x: Tensor) -> Tensor:
        return (x * 2.0).sum().realize()

    fn = capture(step)
    fn(Tensor(randf(4)))
    fn(Tensor(randf(4)))
    with pytest.raises(ValueError, match="recorded arguments"):
        fn(Tensor(randf(3)))


@needs_cc
def test_a_graph_that_changes_between_calls_is_refused():
    set_device("c")
    calls = {"n": 0}

    def step(x: Tensor) -> Tensor:
        calls["n"] += 1
        return (x * float(calls["n"])).sum().realize()  # the literal changes, so the graph does too

    fn = capture(step)
    fn(Tensor(randf(4)))
    with pytest.raises(ValueError, match="different graph"):
        fn(Tensor(randf(4)))


@needs_cc
def test_a_returned_tensor_must_be_realized():
    set_device("c")

    def step(x: Tensor) -> Tensor:
        doubled = x * 2.0
        (x + 1.0).realize()  # some realized work, but not the tensor handed back
        return doubled

    fn = capture(step)
    fn(Tensor(randf(4)))
    with pytest.raises(ValueError, match="must realize what it returns"):
        fn(Tensor(randf(4)))


@needs_cc
def test_a_recorded_capture_refuses_to_replay_inside_an_observation():
    """Replaying skips execute(), so an observing outer capture would freeze the inner result
    as held bytes; refusing loudly beats replaying a constant."""
    set_device("c")
    inner = capture(lambda x: (x * 2.0).sum().realize())
    inner(Tensor(randf(4)))
    inner(Tensor(randf(4)))

    outer = capture(lambda x: (inner(x) + 1.0).realize())
    with pytest.raises(ValueError, match="does not nest"):
        outer(Tensor(randf(4)))


@needs_cc
def test_a_returned_parameter_keeps_its_grad_requirement():
    set_device("c")
    w = Tensor(randf(3), requires_grad=True)
    opt = SGD([w], lr=0.1)

    def step(x: Tensor) -> tuple[Tensor, Tensor]:
        opt.zero_grad()
        loss = (x * w).sum()
        loss.backward()
        opt.step(loss)
        opt.zero_grad()
        return loss, w

    fn = capture(step)
    fn(Tensor(randf(3)))
    _, back = fn(Tensor(randf(3)))  # the recorded call severs the returned tensors' graphs
    assert back is w
    assert w.requires_grad  # a returned in-place target is a live parameter, not a reading


@needs_cc
def test_a_returned_realized_alias_is_accepted():
    """realize() keeps a full alias as a VIEW; returning one is still a realized return."""
    set_device("c")

    def step(x: Tensor) -> Tensor:
        return (x * 2.0).realize().reshape(2, 2).realize()

    fn = capture(step)
    fn(Tensor(randf(4)))
    fn(Tensor(randf(4)))
    d = randf(4)
    np.testing.assert_array_equal(fn(Tensor(d)).numpy(), (d * 2.0).reshape(2, 2))


@needs_cc
def test_replay_is_bound_to_the_device_that_recorded_it():
    set_device("c")

    def step(x: Tensor) -> Tensor:
        return (x * 2.0).sum().realize()

    fn = capture(step)
    fn(Tensor(randf(4)))
    fn(Tensor(randf(4)))
    set_device("c")  # a fresh instance of the same backend, not the recording's device
    with pytest.raises(ValueError, match="device that recorded"):
        fn(Tensor(randf(4)))


def test_on_an_interpreting_device_the_function_just_runs():
    data = batches(4)
    counter = {"ran": 0}
    losses = run(data, captured=True, counter=counter)
    assert counter["ran"] == 4  # numpy has no plans to replay, so every call runs the function
    assert losses == run(data, captured=False)
