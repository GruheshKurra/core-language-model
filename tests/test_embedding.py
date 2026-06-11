import numpy as np

from zyn.embedding import Embedding
from zyn.gradcheck import gradcheck
from zyn.tensor import Tensor


def analytic(weight, indices, build):
    t = Tensor(weight.copy())
    out = build(t, indices)
    out.backward()
    return t.grad


def test_gather_forward():
    weight = np.arange(12, dtype=np.float64).reshape(4, 3)
    indices = np.array([[0, 2], [1, 0]])
    out = Tensor(weight).gather(indices).data
    expected = np.take(weight, indices, axis=0)
    assert np.allclose(out, expected)


def test_gather_gradcheck():
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(6, 4))
    indices = np.array([[0, 2, 1], [3, 0, 2]])

    def f(w):
        return float(np.take(w, indices, axis=0).sum())

    g = analytic(weight, indices, lambda t, idx: t.gather(idx).sum())
    assert gradcheck(f, weight, g)


def test_embedding_shape():
    emb = Embedding(vocab_size=100, d_model=32)
    x = np.zeros((4, 16), dtype=np.int64)
    out = emb(x)
    assert out.shape == (4, 16, 32)


def test_embedding_forward():
    rng = np.random.default_rng(1)
    vocab_size, d_model = 8, 5
    emb = Embedding(vocab_size, d_model, std=1.0)
    emb.weight.data = rng.normal(size=(vocab_size, d_model))

    indices = np.array([[0, 3], [7, 1]])
    out = emb(indices).data
    expected = np.take(emb.weight.data, indices, axis=0)
    assert np.allclose(out, expected)


def test_embedding_gradcheck():
    rng = np.random.default_rng(2)
    indices = np.array([[0, 2, 1], [3, 0, 2]])
    weight = rng.normal(size=(6, 4))

    def f(w):
        return float(np.take(w, indices, axis=0).sum())

    g = analytic(weight, indices, lambda t, idx: t.gather(idx).sum())
    assert gradcheck(f, weight, g)


def test_embedding_duplicate_indices_accumulate_grad():
    weight = np.eye(3, dtype=np.float64)
    indices = np.array([[0, 0, 1]])
    t = Tensor(weight.copy())
    t.gather(indices).sum().backward()
    assert np.allclose(t.grad[0], [2.0, 2.0, 2.0])
    assert np.allclose(t.grad[1], [1.0, 1.0, 1.0])
