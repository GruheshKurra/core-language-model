from .backend import xp, to_device


def _reduce_grad(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape)):
        if shape[i] == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = to_device(data)
        self.grad = xp.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op or 'leaf'})"

    def add(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "add")

        def _backward():
            self.grad = self.grad + _reduce_grad(out.grad, self.data.shape)
            other.grad = other.grad + _reduce_grad(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def mul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "mul")

        def _backward():
            self.grad = self.grad + _reduce_grad(out.grad * other.data, self.data.shape)
            other.grad = other.grad + _reduce_grad(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            ga = out.grad @ xp.swapaxes(other.data, -1, -2)
            gb = xp.swapaxes(self.data, -1, -2) @ out.grad
            self.grad = self.grad + _reduce_grad(ga, self.data.shape)
            other.grad = other.grad + _reduce_grad(gb, other.data.shape)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            self.grad = self.grad + out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Tensor(xp.transpose(self.data, axes), (self,), "transpose")

        def _backward():
            if axes is None:
                self.grad = self.grad + xp.transpose(out.grad)
            else:
                inv = [0] * len(axes)
                for i, a in enumerate(axes):
                    inv[a] = i
                self.grad = self.grad + xp.transpose(out.grad, tuple(inv))

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                ax = axis if isinstance(axis, tuple) else (axis,)
                for a in sorted(a % self.data.ndim for a in ax):
                    g = xp.expand_dims(g, a)
            self.grad = self.grad + xp.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out

    def gather(self, index):
        idx = index.data if isinstance(index, Tensor) else to_device(index)
        idx = idx.astype(xp.int64)
        out = Tensor(self.data[idx], (self,), "gather")

        def _backward():
            grad = xp.zeros_like(self.data)
            xp.add.at(grad, idx, out.grad)
            self.grad = self.grad + grad

        out._backward = _backward
        return out

    def neg(self):
        return self.mul(-1.0)

    def sub(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.add(other.neg())

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __matmul__(self, other):
        return self.matmul(other)

    def __neg__(self):
        return self.neg()

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return self.neg().add(other)

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = xp.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                v.grad = xp.zeros_like(v.data)
                for child in v._prev:
                    build(child)

        build(self)
