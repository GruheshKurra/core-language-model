import numpy as np

from zyn.gradcheck import gradcheck
from zyn.layernorm import LayerNorm
from zyn.tensor import Tensor


def _reference(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    xhat = xc / np.sqrt(var + eps)
    return gamma * xhat + beta


def test_forward_normalizes():
    rng = np.random.default_rng(0)
    ln = LayerNorm(d_model=8)
    x = rng.normal(size=(2, 4, 8)) * 5.0 + 3.0
    out = ln(Tensor(x)).data
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-3)


def test_forward_matches_reference():
    rng = np.random.default_rng(1)
    d = 6
    ln = LayerNorm(d_model=d)
    ln.gamma.data = rng.normal(size=d)
    ln.beta.data = rng.normal(size=d)
    x = rng.normal(size=(3, d))
    out = ln(Tensor(x)).data
    assert np.allclose(out, _reference(x, ln.gamma.data, ln.beta.data))


def test_dim_guard():
    ln = LayerNorm(d_model=8)
    try:
        ln(Tensor(np.zeros((2, 4, 5))))
        assert False
    except ValueError:
        pass


def test_gradcheck_x():
    rng = np.random.default_rng(2)
    d = 5
    ln = LayerNorm(d_model=d)
    ln.gamma.data = rng.normal(size=d)
    ln.beta.data = rng.normal(size=d)
    x = rng.normal(size=(2, d))
    w = rng.normal(size=(2, d))

    def f(xv):
        return float((_reference(xv.reshape(2, d), ln.gamma.data, ln.beta.data) * w).sum())

    t = Tensor(x.copy())
    ln(t).mul(Tensor(w)).sum().backward()
    assert gradcheck(f, x.reshape(-1), t.grad.reshape(-1))


def test_gradcheck_gamma():
    rng = np.random.default_rng(3)
    d = 5
    ln = LayerNorm(d_model=d)
    g0 = rng.normal(size=d)
    ln.gamma.data = g0.copy()
    x = rng.normal(size=(4, d))
    w = rng.normal(size=(4, d))

    def f(gv):
        return float((_reference(x, gv, ln.beta.data) * w).sum())

    ln(Tensor(x.copy())).mul(Tensor(w)).sum().backward()
    assert gradcheck(f, g0.copy(), ln.gamma.grad)


def test_gradcheck_beta():
    rng = np.random.default_rng(4)
    d = 5
    ln = LayerNorm(d_model=d)
    b0 = rng.normal(size=d)
    ln.beta.data = b0.copy()
    x = rng.normal(size=(4, d))
    w = rng.normal(size=(4, d))

    def f(bv):
        return float((_reference(x, ln.gamma.data, bv) * w).sum())

    ln(Tensor(x.copy())).mul(Tensor(w)).sum().backward()
    assert gradcheck(f, b0.copy(), ln.beta.grad)
