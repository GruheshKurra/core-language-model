import numpy as np

from zyn.attention import SelfAttention
from zyn.gradcheck import gradcheck
from zyn.tensor import Tensor


def test_batched_matmul_gradcheck():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 3, 4))
    b = rng.normal(size=(4, 5))

    def f(x):
        return float((x.reshape(2, 3, 4) @ b).sum())

    t = Tensor(a.copy())
    t.matmul(Tensor(b)).sum().backward()
    assert gradcheck(f, a.reshape(-1), t.grad.reshape(-1))


def test_transpose_gradcheck():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2, 3, 4))
    target = rng.normal(size=(2, 4, 3))

    def f(x):
        return float((np.swapaxes(x.reshape(2, 3, 4), -2, -1) * target).sum())

    t = Tensor(a.copy())
    (t.transpose().mul(Tensor(target))).sum().backward()
    assert gradcheck(f, a.reshape(-1), t.grad.reshape(-1))


def test_softmax_forward_and_gradcheck():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(2, 5))
    s = Tensor(a).softmax(axis=-1).data
    assert np.allclose(s.sum(axis=-1), 1.0)
    assert np.all(s > 0)

    w = rng.normal(size=(2, 5))

    def f(x):
        z = x.reshape(2, 5) - x.reshape(2, 5).max(axis=-1, keepdims=True)
        e = np.exp(z)
        sm = e / e.sum(axis=-1, keepdims=True)
        return float((sm * w).sum())

    t = Tensor(a.copy())
    t.softmax(axis=-1).mul(Tensor(w)).sum().backward()
    assert gradcheck(f, a.reshape(-1), t.grad.reshape(-1))


def test_masked_fill_forward_and_grad():
    a = np.arange(9, dtype=np.float64).reshape(3, 3)
    mask = np.triu(np.ones((3, 3), dtype=bool), k=1)
    t = Tensor(a.copy())
    out = t.masked_fill(mask, -1e9)
    assert np.allclose(out.data[mask], -1e9)
    assert np.allclose(out.data[~mask], a[~mask])
    out.sum().backward()
    assert np.allclose(t.grad[mask], 0.0)
    assert np.allclose(t.grad[~mask], 1.0)


def test_attention_shape():
    attn = SelfAttention(d_model=8)
    x = Tensor(np.random.randn(2, 5, 8))
    out = attn(x)
    assert out.shape == (2, 5, 8)


def test_causality():
    rng = np.random.default_rng(3)
    attn = SelfAttention(d_model=6)
    B, T, d = 1, 4, 6
    x = rng.normal(size=(B, T, d))

    out_full = attn(Tensor(x.copy())).data

    x_mod = x.copy()
    x_mod[0, 2] += 5.0
    out_mod = attn(Tensor(x_mod)).data

    assert np.allclose(out_full[0, :2], out_mod[0, :2])
    assert not np.allclose(out_full[0, 2], out_mod[0, 2])


def test_attention_param_gradcheck():
    rng = np.random.default_rng(4)
    attn = SelfAttention(d_model=4)
    x = rng.normal(size=(1, 3, 4))
    target = rng.normal(size=(1, 3, 4))

    attn.W_q = Tensor(rng.normal(size=(4, 4)))
    attn.W_k = Tensor(rng.normal(size=(4, 4)))
    attn.W_v = Tensor(rng.normal(size=(4, 4)))
    attn.W_o = Tensor(rng.normal(size=(4, 4)))

    def loss_from(Wq):
        a = SelfAttention(d_model=4)
        a.W_q = Tensor(Wq.reshape(4, 4))
        a.W_k = Tensor(attn.W_k.data.copy())
        a.W_v = Tensor(attn.W_v.data.copy())
        a.W_o = Tensor(attn.W_o.data.copy())
        out = a(Tensor(x.copy()))
        return out, a

    def f(Wq):
        out, _ = loss_from(Wq)
        return float(((out.data - target) ** 2).sum())

    out, a = loss_from(attn.W_q.data.copy())
    err = out.add(Tensor(-target))
    err.mul(err).sum().backward()
    assert gradcheck(f, attn.W_q.data.copy().reshape(-1), a.W_q.grad.reshape(-1), tol=1e-5)
