import numpy as np

from zyn.attention import MultiHeadAttention, SelfAttention
from zyn.gradcheck import gradcheck
from zyn.tensor import Tensor


def test_reshape_gradcheck():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 6))
    w = rng.normal(size=(2, 2, 3))

    def f(x):
        return float((x.reshape(2, 2, 3) * w).sum())

    t = Tensor(a.copy())
    t.reshape(2, 2, 3).mul(Tensor(w)).sum().backward()
    assert gradcheck(f, a.reshape(-1), t.grad.reshape(-1))


def test_divisibility_guard():
    try:
        MultiHeadAttention(d_model=10, n_head=3)
        assert False
    except ValueError:
        pass


def test_shape():
    mha = MultiHeadAttention(d_model=12, n_head=4)
    x = Tensor(np.random.randn(2, 5, 12))
    out = mha(x)
    assert out.shape == (2, 5, 12)


def test_single_head_matches_selfattention():
    rng = np.random.default_rng(1)
    d = 6
    x = rng.normal(size=(2, 4, d))

    sa = SelfAttention(d_model=d)
    mha = MultiHeadAttention(d_model=d, n_head=1)
    for name in ("W_q", "W_k", "W_v", "W_o"):
        w = rng.normal(size=(d, d))
        setattr(sa, name, Tensor(w.copy()))
        setattr(mha, name, Tensor(w.copy()))

    out_sa = sa(Tensor(x.copy())).data
    out_mha = mha(Tensor(x.copy())).data
    assert np.allclose(out_sa, out_mha)


def test_causality():
    rng = np.random.default_rng(2)
    mha = MultiHeadAttention(d_model=8, n_head=2)
    x = rng.normal(size=(1, 4, 8))

    out_full = mha(Tensor(x.copy())).data
    x_mod = x.copy()
    x_mod[0, 2] += 5.0
    out_mod = mha(Tensor(x_mod)).data

    assert np.allclose(out_full[0, :2], out_mod[0, :2])
    assert not np.allclose(out_full[0, 2], out_mod[0, 2])


def test_param_gradcheck():
    rng = np.random.default_rng(3)
    d, h = 4, 2
    x = rng.normal(size=(1, 3, d))
    target = rng.normal(size=(1, 3, d))

    base = {n: rng.normal(size=(d, d)) for n in ("W_q", "W_k", "W_v", "W_o")}

    def build(Wq):
        m = MultiHeadAttention(d_model=d, n_head=h)
        m.W_q = Tensor(Wq.reshape(d, d))
        m.W_k = Tensor(base["W_k"].copy())
        m.W_v = Tensor(base["W_v"].copy())
        m.W_o = Tensor(base["W_o"].copy())
        return m

    def f(Wq):
        out = build(Wq)(Tensor(x.copy()))
        return float(((out.data - target) ** 2).sum())

    m = build(base["W_q"].copy())
    out = m(Tensor(x.copy()))
    err = out.add(Tensor(-target))
    err.mul(err).sum().backward()
    assert gradcheck(f, base["W_q"].reshape(-1), m.W_q.grad.reshape(-1), tol=1e-5)
