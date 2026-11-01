import math
import random

import pytest

from micrograd.engine import Value
from micrograd.nn import MLP


def test_matches_reference_micrograd_values():
    # This expression and its expected values come from Karpathy's micrograd
    # test suite, where they were checked against PyTorch.
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()

    assert g.data == pytest.approx(24.7041, abs=1e-4)
    assert a.grad == pytest.approx(138.8338, abs=1e-4)
    assert b.grad == pytest.approx(645.5773, abs=1e-4)


EXPRESSIONS = {
    "add_mul": lambda x, y: x * y + x + 3,
    "sub_div": lambda x, y: (x - y) / (x * x + 1),
    "reused_node": lambda x, y: x * x * x + x * y + y * y,
    "right_hand_ops": lambda x, y: 2 - x + 3 * y + 1 / (y * y + 1),
    "power": lambda x, y: (x * x + 1) ** 1.5 + y**3,
    "tanh": lambda x, y: (x * y).tanh() + x.tanh(),
    "relu": lambda x, y: (x + y).relu() * x,
}


@pytest.mark.parametrize("name", EXPRESSIONS)
@pytest.mark.parametrize("x0, y0", [(0.7, 1.3), (-1.2, 0.4), (2.1, -0.6)])
def test_gradients_match_finite_differences(name, x0, y0):
    fn = EXPRESSIONS[name]
    x, y = Value(x0), Value(y0)
    fn(x, y).backward()

    h = 1e-6
    dx = (fn(Value(x0 + h), Value(y0)).data - fn(Value(x0 - h), Value(y0)).data) / (2 * h)
    dy = (fn(Value(x0), Value(y0 + h)).data - fn(Value(x0), Value(y0 - h)).data) / (2 * h)
    assert x.grad == pytest.approx(dx, abs=1e-5)
    assert y.grad == pytest.approx(dy, abs=1e-5)


def test_node_used_twice_accumulates_gradient():
    a = Value(3.0)
    (a + a).backward()
    assert a.grad == 2.0


def test_relu_gradient_is_zero_when_inactive():
    a = Value(-1.0)
    a.relu().backward()
    assert a.grad == 0.0


def test_power_of_zero_exponent_at_zero():
    a = Value(0.0)
    out = a**0
    out.backward()
    assert out.data == 1.0
    assert a.grad == 0.0


def test_deep_graph_does_not_hit_recursion_limit():
    a = Value(1.0)
    out = a
    for _ in range(20_000):
        out = out + 1
    out.backward()
    assert a.grad == 1.0


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_mlp_parameter_gradients_match_finite_differences(activation):
    model = MLP(2, [4, 1], seed=1, activation=activation)
    inputs = [0.3, -0.8]

    model.zero_grad()
    model(inputs).backward()

    rng = random.Random(0)
    h = 1e-6
    for p in rng.sample(model.parameters(), 8):
        original = p.data
        p.data = original + h
        high = model(inputs).data
        p.data = original - h
        low = model(inputs).data
        p.data = original
        assert p.grad == pytest.approx((high - low) / (2 * h), abs=1e-5)


def test_mlp_parameter_count():
    # (2 weights + bias) * 4 neurons + (4 weights + bias) * 1 neuron
    assert len(MLP(2, [4, 1]).parameters()) == 17


def test_same_seed_gives_same_model():
    first = [p.data for p in MLP(2, [4, 1], seed=7).parameters()]
    second = [p.data for p in MLP(2, [4, 1], seed=7).parameters()]
    assert first == second
    assert math.isfinite(sum(first))
