"""A scalar autograd engine.

Every math operation creates a new Value that remembers its inputs and how to
pass gradients back to them. Together these form a graph, and backward() walks
it from output to inputs, applying the chain rule at each node.
"""

import math


class Value:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0  # d(output)/d(this value), filled in by backward()
        self._backward = lambda: None
        self._prev = _children
        self._op = _op  # the operation that produced this node, for debugging

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # Use += because a node can feed into several others, and each
            # path contributes to its gradient.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("only constant int/float exponents are supported")
        out = Value(math.pow(self.data, exponent), (self,), f"**{exponent}")

        def _backward():
            if exponent != 0:  # x**0 is constant, and 0**-1 would raise
                self.grad += exponent * math.pow(self.data, exponent - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Compute d(self)/d(node) for every node this value depends on."""
        # Each node must run after all the nodes that use it, so sort the graph
        # topologically. This is iterative because a deep graph would hit
        # Python's recursion limit.
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((child, False) for child in node._prev)

        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    # Everything below is built from the primitives above.
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
