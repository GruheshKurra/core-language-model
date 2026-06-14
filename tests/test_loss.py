import numpy as np

from zyn.gradcheck import gradcheck
from zyn.loss import cross_entropy
from zyn.tensor import Tensor


def _ref_ce(z, t, ignore_index=None):
    flat = z.reshape(-1, z.shape[-1])
    tt = np.asarray(t).reshape(-1)
    m = flat.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(flat - m).sum(axis=-1, keepdims=True)))[:, 0]
    nll = lse - flat[np.arange(flat.shape[0]), tt]
    keep = np.ones_like(tt, dtype=bool) if ignore_index is None else (tt != ignore_index)
    return float((nll * keep).sum() / keep.sum())


def test_uniform_logits_equals_log_v():
    V = 10
    z = np.zeros((4, V))
    t = np.array([0, 1, 2, 3])
    loss = cross_entropy(Tensor(z), t).data
    assert np.isclose(float(loss), np.log(V))


def test_matches_reference():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2, 3, 7))
    t = rng.integers(0, 7, size=(2, 3))
    loss = cross_entropy(Tensor(z), t).data
    assert np.isclose(float(loss), _ref_ce(z, t))


def test_perfect_prediction_low_loss():
    z = np.full((3, 5), -50.0)
    t = np.array([1, 3, 0])
    z[np.arange(3), t] = 50.0
    loss = cross_entropy(Tensor(z), t).data
    assert float(loss) < 1e-10


def test_gradcheck():
    rng = np.random.default_rng(1)
    B, T, V = 2, 3, 6
    z = rng.normal(size=(B, T, V))
    t = rng.integers(0, V, size=(B, T))

    def f(zv):
        return _ref_ce(zv.reshape(B, T, V), t)

    tz = Tensor(z.copy())
    cross_entropy(tz, t).backward()
    assert gradcheck(f, z.reshape(-1), tz.grad.reshape(-1))


def test_ignore_index_masks_tokens():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(1, 4, 5))
    t = np.array([[1, 2, -1, 3]])
    loss = cross_entropy(Tensor(z), t, ignore_index=-1).data
    assert np.isclose(float(loss), _ref_ce(z, t, ignore_index=-1))


def test_ignore_index_zeroes_grad_on_masked_rows():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(1, 4, 5))
    t = np.array([[1, 2, -1, 3]])
    tz = Tensor(z.copy())
    cross_entropy(tz, t, ignore_index=-1).backward()
    assert np.allclose(tz.grad[0, 2], 0.0)
    assert not np.allclose(tz.grad[0, 0], 0.0)


def test_ignore_index_gradcheck():
    rng = np.random.default_rng(4)
    B, T, V = 1, 5, 6
    z = rng.normal(size=(B, T, V))
    t = np.array([[0, -1, 2, -1, 4]])

    def f(zv):
        return _ref_ce(zv.reshape(B, T, V), t, ignore_index=-1)

    tz = Tensor(z.copy())
    cross_entropy(tz, t, ignore_index=-1).backward()
    assert gradcheck(f, z.reshape(-1), tz.grad.reshape(-1))
