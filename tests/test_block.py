import numpy as np

from zyn.block import TransformerBlock
from zyn.gradcheck import gradcheck
from zyn.tensor import Tensor


def test_shape():
    blk = TransformerBlock(d_model=12, n_head=4)
    x = Tensor(np.random.randn(2, 5, 12))
    out = blk(x)
    assert out.shape == (2, 5, 12)


def test_residual_identity_when_sublayers_zero():
    rng = np.random.default_rng(0)
    blk = TransformerBlock(d_model=8, n_head=2)
    blk.attn.W_o.data[:] = 0.0
    blk.mlp.W2.data[:] = 0.0
    blk.mlp.b2.data[:] = 0.0
    x = rng.normal(size=(2, 4, 8))
    out = blk(Tensor(x)).data
    assert np.allclose(out, x)


def test_parameters_count():
    blk = TransformerBlock(d_model=8, n_head=2)
    params = blk.parameters()
    assert len(params) == 2 + 4 + 2 + 4


def test_causality():
    rng = np.random.default_rng(1)
    blk = TransformerBlock(d_model=8, n_head=2)
    x = rng.normal(size=(1, 4, 8))
    out_full = blk(Tensor(x.copy())).data
    x_mod = x.copy()
    x_mod[0, 2] += 5.0
    out_mod = blk(Tensor(x_mod)).data
    assert np.allclose(out_full[0, :2], out_mod[0, :2])
    assert not np.allclose(out_full[0, 2], out_mod[0, 2])


def test_input_gradcheck():
    rng = np.random.default_rng(2)
    d, h = 4, 2
    blk = TransformerBlock(d_model=d, n_head=h)
    x = rng.normal(size=(1, 3, d))
    g = rng.normal(size=(1, 3, d))

    def f(xv):
        return float((blk(Tensor(xv.reshape(1, 3, d))).data * g).sum())

    t = Tensor(x.copy())
    blk(t).mul(Tensor(g)).sum().backward()
    assert gradcheck(f, x.reshape(-1), t.grad.reshape(-1), tol=1e-5)
