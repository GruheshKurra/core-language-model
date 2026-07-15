from .backend import xp
from .tensor import Tensor


def exp(t):
    out = Tensor(xp.exp(t.data), (t,), "exp")

    def _backward():
        t.grad = t.grad + out.grad * out.data

    out._backward = _backward
    return out


def log(t):
    out = Tensor(xp.log(t.data), (t,), "log")

    def _backward():
        t.grad = t.grad + out.grad / t.data

    out._backward = _backward
    return out


def rsqrt(t):
    r = 1.0 / xp.sqrt(t.data)
    out = Tensor(r, (t,), "rsqrt")

    def _backward():
        t.grad = t.grad + out.grad * (-0.5) * (r ** 3)

    out._backward = _backward
    return out


def silu(t):
    s = 1.0 / (1.0 + xp.exp(-t.data))
    out = Tensor(t.data * s, (t,), "silu")

    def _backward():
        t.grad = t.grad + out.grad * (s * (1.0 + t.data * (1.0 - s)))

    out._backward = _backward
    return out


def gelu(t):
    x = t.data
    k = xp.sqrt(2.0 / xp.pi)
    a = 0.044715
    u = k * (x + a * x ** 3)
    tanh_u = xp.tanh(u)
    out = Tensor(0.5 * x * (1.0 + tanh_u), (t,), "gelu")

    def _backward():
        du = k * (1.0 + 3.0 * a * x ** 2)
        dg = 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u ** 2) * du
        t.grad = t.grad + out.grad * dg

    out._backward = _backward
    return out


def softmax(t, axis=-1):
    z = t.data - xp.max(t.data, axis=axis, keepdims=True)
    e = xp.exp(z)
    s = e / xp.sum(e, axis=axis, keepdims=True)
    out = Tensor(s, (t,), "softmax")

    def _backward():
        dot = xp.sum(out.grad * s, axis=axis, keepdims=True)
        t.grad = t.grad + s * (out.grad - dot)

    out._backward = _backward
    return out
