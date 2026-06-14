import numpy as np

from zyn.gradcheck import gradcheck
from zyn.mlp import MLP
from zyn.tensor import Tensor


def _gelu(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def test_gelu_forward_matches_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4))
    out = Tensor(x).gelu().data
    assert np.allclose(out, _gelu(x))


def test_gelu_gradcheck():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 5))
    w = rng.normal(size=(2, 5))

    def f(xv):
        return float((_gelu(xv.reshape(2, 5)) * w).sum())

    t = Tensor(x.copy())
    t.gelu().mul(Tensor(w)).sum().backward()
    assert gradcheck(f, x.reshape(-1), t.grad.reshape(-1))


def test_shape_and_default_ff():
    mlp = MLP(d_model=8)
    assert mlp.d_ff == 32
    out = mlp(Tensor(np.random.randn(2, 5, 8)))
    assert out.shape == (2, 5, 8)


def test_dim_guard():
    mlp = MLP(d_model=8)
    try:
        mlp(Tensor(np.zeros((2, 4, 7))))
        assert False
    except ValueError:
        pass


def _ref_forward(x, W1, b1, W2, b2):
    return _gelu(x @ W1 + b1) @ W2 + b2


def test_param_gradcheck_W1():
    rng = np.random.default_rng(2)
    d, ff = 4, 8
    mlp = MLP(d_model=d, d_ff=ff)
    base = {
        "W1": rng.normal(size=(d, ff)),
        "b1": rng.normal(size=ff),
        "W2": rng.normal(size=(ff, d)),
        "b2": rng.normal(size=d),
    }
    for k, v in base.items():
        setattr(mlp, k, Tensor(v.copy()))
    x = rng.normal(size=(1, 3, d))
    g = rng.normal(size=(1, 3, d))

    def f(W1v):
        return float((_ref_forward(x, W1v.reshape(d, ff), base["b1"], base["W2"], base["b2"]) * g).sum())

    mlp(Tensor(x.copy())).mul(Tensor(g)).sum().backward()
    assert gradcheck(f, base["W1"].reshape(-1), mlp.W1.grad.reshape(-1), tol=1e-5)


def test_param_gradcheck_b2():
    rng = np.random.default_rng(3)
    d, ff = 4, 8
    mlp = MLP(d_model=d, d_ff=ff)
    base = {
        "W1": rng.normal(size=(d, ff)),
        "b1": rng.normal(size=ff),
        "W2": rng.normal(size=(ff, d)),
        "b2": rng.normal(size=d),
    }
    for k, v in base.items():
        setattr(mlp, k, Tensor(v.copy()))
    x = rng.normal(size=(1, 3, d))
    g = rng.normal(size=(1, 3, d))

    def f(b2v):
        return float((_ref_forward(x, base["W1"], base["b1"], base["W2"], b2v) * g).sum())

    mlp(Tensor(x.copy())).mul(Tensor(g)).sum().backward()
    assert gradcheck(f, base["b2"].copy(), mlp.b2.grad)
