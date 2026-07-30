"""capture records a step's device work once and replays it; the c device proves the replays."""

import numpy as np
import pytest
from conftest import needs_cc, randf

from limn import Tensor, capture, set_device, set_seed
from limn.nn import Linear, parameters
from limn.optim import AdamW


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


def test_on_an_interpreting_device_the_function_just_runs():
    data = batches(4)
    counter = {"ran": 0}
    losses = run(data, captured=True, counter=counter)
    assert counter["ran"] == 4  # numpy has no plans to replay, so every call runs the function
    assert losses == run(data, captured=False)
