import numpy as np
import pytest

from zyn.embedding import Embedding
from zyn.gradcheck import gradcheck
from zyn.positional import PositionalEmbedding
from zyn.tensor import Tensor


def test_shape():
    pos = PositionalEmbedding(max_seq=64, d_model=16)
    out = pos(10)
    assert out.shape == (10, 16)


def test_forward_picks_first_t_rows():
    rng = np.random.default_rng(0)
    pos = PositionalEmbedding(max_seq=8, d_model=4)
    pos.weight.data = rng.normal(size=(8, 4))
    out = pos(5).data
    assert np.allclose(out, pos.weight.data[:5])


def test_max_seq_guard():
    pos = PositionalEmbedding(max_seq=4, d_model=3)
    with pytest.raises(ValueError):
        pos(5)


def test_broadcast_add_to_tokens():
    B, T, d = 2, 3, 5
    tok = Embedding(vocab_size=10, d_model=d)
    pos = PositionalEmbedding(max_seq=8, d_model=d)
    x = np.array([[0, 1, 2], [3, 4, 5]])
    h = tok(x) + pos(T)
    assert h.shape == (B, T, d)
    expected = tok.weight.data[x] + pos.weight.data[:T]
    assert np.allclose(h.data, expected)


def test_gradcheck():
    rng = np.random.default_rng(2)
    weight = rng.normal(size=(6, 4))
    seq_len = 4

    def f(w):
        return float(np.take(w, np.arange(seq_len), axis=0).sum())

    t = Tensor(weight.copy())
    t.gather(np.arange(seq_len), dim=0).sum().backward()
    assert gradcheck(f, weight, t.grad)


def test_grad_accumulates_only_used_positions():
    pos = PositionalEmbedding(max_seq=8, d_model=3)
    pos(3).sum().backward()
    assert np.allclose(pos.weight.grad[:3], 1.0)
    assert np.allclose(pos.weight.grad[3:], 0.0)
