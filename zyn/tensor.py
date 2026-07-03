from __future__ import annotations
from zyn.backend import xp as np
from zyn.backend import fdtype
from zyn.backend import scatter_add


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(self, data, _children=(), requires_grad=True):
        self.data = np.asarray(data, dtype=fdtype)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape})"

    def add(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def mul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            a, b = self.data, other.data
            ga = out.grad @ np.swapaxes(b, -1, -2)
            gb = np.swapaxes(a, -1, -2) @ out.grad
            self.grad += _unbroadcast(ga, a.shape)
            other.grad += _unbroadcast(gb, b.shape)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, axis1=-2, axis2=-1):
        out = Tensor(np.swapaxes(self.data, axis1, axis2), (self,))

        def _backward():
            self.grad += np.swapaxes(out.grad, axis1, axis2)
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, (self,))

        def _backward():
            dot = (out.grad * s).sum(axis=axis, keepdims=True)
            self.grad += s * (out.grad - dot)
        out._backward = _backward
        return out

    def masked_fill(self, mask, value):
        mask = np.asarray(mask, dtype=bool)
        out = Tensor(np.where(mask, value, self.data), (self,))

        def _backward():
            self.grad += _unbroadcast(np.where(mask, 0.0, out.grad), self.data.shape)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,))

        def _backward():
            self.grad += (self.data > 0).astype(fdtype) * out.grad
        out._backward = _backward
        return out

    def gelu(self):
        c = np.sqrt(2.0 / np.pi)
        x = self.data
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out = Tensor(0.5 * x * (1.0 + t), (self,))

        def _backward():
            d_inner = c * (1.0 + 3.0 * 0.044715 * x ** 2)
            dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
            self.grad += dx * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,))

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.data.shape).copy()
        out._backward = _backward
        return out

    def gather(self, indices, dim: int = 0):
        if dim != 0:
            raise NotImplementedError("gather only supports dim=0 (row lookup)")
        indices = np.asarray(indices, dtype=np.int64)
        out = Tensor(np.take(self.data, indices, axis=0), (self,))

        def _backward():
            scatter_add(self.grad, indices, out.grad)
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                if child not in visited:
                    stack.append((child, False))

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __matmul__(self, other):
        return self.matmul(other)
