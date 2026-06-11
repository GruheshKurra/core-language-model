import numpy as np

from zyn.tensor import Tensor
from zyn.gradcheck import gradcheck


def analytic(x, build):
    t = Tensor(x)
    out = build(t)
    out.backward()
    return t.grad


def test_add():
    x = np.random.randn(3, 4)
    b = np.random.randn(4)
    f = lambda v: np.sum(v + b)
    g = analytic(x, lambda t: (t + Tensor(b)).sum())
    assert gradcheck(f, x, g)


def test_mul():
    x = np.random.randn(3, 4)
    w = np.random.randn(3, 4)
    f = lambda v: np.sum(v * w)
    g = analytic(x, lambda t: (t * Tensor(w)).sum())
    assert gradcheck(f, x, g)


def test_matmul():
    x = np.random.randn(2, 3)
    W = np.random.randn(3, 5)
    f = lambda v: np.sum(v @ W)
    g = analytic(x, lambda t: (t @ Tensor(W)).sum())
    assert gradcheck(f, x, g)


def test_relu():
    x = np.random.randn(4, 4) + 0.5
    f = lambda v: np.sum(np.maximum(0.0, v))
    g = analytic(x, lambda t: t.relu().sum())
    assert gradcheck(f, x, g)


def test_exp():
    x = np.random.randn(3, 3) * 0.5
    f = lambda v: np.sum(np.exp(v))
    g = analytic(x, lambda t: t.exp().sum())
    assert gradcheck(f, x, g)


def test_log():
    x = np.random.rand(3, 3) + 0.5
    f = lambda v: np.sum(np.log(v))
    g = analytic(x, lambda t: t.log().sum())
    assert gradcheck(f, x, g)


def test_chain():
    x = np.random.randn(2, 3) * 0.3
    W = np.random.randn(3, 4)
    f = lambda v: np.sum(np.log(np.exp(np.maximum(0.0, v @ W)) + 1.0))
    g = analytic(x, lambda t: (t @ Tensor(W)).relu().exp().add(1.0).log().sum())
    assert gradcheck(f, x, g)


def test_gather():
    rng = np.random.default_rng(3)
    weight = rng.normal(size=(5, 3))
    indices = np.array([[0, 1], [2, 4]])
    f = lambda w: np.sum(np.take(w, indices, axis=0))
    g = analytic(weight, lambda t: t.gather(indices).sum())
    assert gradcheck(f, weight, g)
