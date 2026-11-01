"""A small neural network library built on Value."""

import random

from micrograd.engine import Value

ACTIVATIONS = {"relu": Value.relu, "tanh": Value.tanh}


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        # Gradients accumulate, so clear them before each backward pass.
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, n_inputs, rng, activation=None):
        self.weights = [Value(rng.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(0.0)
        self.activation = ACTIVATIONS[activation] if activation else None

    def __call__(self, inputs):
        total = sum((w * x for w, x in zip(self.weights, inputs)), self.bias)
        return self.activation(total) if self.activation else total

    def parameters(self):
        return [*self.weights, self.bias]


class Layer(Module):
    def __init__(self, n_inputs, n_outputs, rng, activation=None):
        self.neurons = [Neuron(n_inputs, rng, activation) for _ in range(n_outputs)]

    def __call__(self, inputs):
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    """A feed-forward network whose final layer is linear, so it outputs raw scores."""

    def __init__(self, n_inputs, layer_sizes, seed=0, activation="relu"):
        rng = random.Random(seed)
        sizes = [n_inputs, *layer_sizes]
        last = len(layer_sizes) - 1
        self.layers = [
            Layer(sizes[i], sizes[i + 1], rng, None if i == last else activation)
            for i in range(len(layer_sizes))
        ]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs[0] if len(inputs) == 1 else inputs

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
