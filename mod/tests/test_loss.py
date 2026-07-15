import numpy as np

from mla import Tensor, gradcheck
from mla.loss import cross_entropy


def _logits(*shape):
    rng = np.random.default_rng(0)
    return Tensor(rng.standard_normal(shape))


def test_ce_gradcheck_flat():
    tgt = np.array([2, 0, 3, 1])
    ok, err = gradcheck(lambda l: cross_entropy(l, tgt), [_logits(4, 5)])
    assert ok, err


def test_ce_gradcheck_3d():
    tgt = np.array([[1, 4, 0], [2, 2, 3]])
    ok, err = gradcheck(lambda l: cross_entropy(l, tgt), [_logits(2, 3, 6)])
    assert ok, err


def test_ce_gradcheck_ignore():
    tgt = np.array([2, -1, 3, -1])
    ok, err = gradcheck(lambda l: cross_entropy(l, tgt), [_logits(4, 5)])
    assert ok, err


def test_ce_uniform_equals_logV():
    V = 7
    logits = Tensor(np.zeros((3, V)))
    tgt = np.array([0, 1, 2])
    loss = float(cross_entropy(logits, tgt).data)
    assert abs(loss - np.log(V)) < 1e-9


def test_ce_ignore_excludes_rows():
    logits = _logits(2, 5)
    full = float(cross_entropy(logits, np.array([2, 4])).data)
    row0 = float(cross_entropy(logits, np.array([2, -1])).data)
    row1 = float(cross_entropy(logits, np.array([-1, 4])).data)
    assert abs(full - 0.5 * (row0 + row1)) < 1e-9


def test_ce_perfect_prediction_low_loss():
    big = np.full((2, 4), -20.0)
    big[0, 1] = 20.0
    big[1, 3] = 20.0
    loss = float(cross_entropy(Tensor(big), np.array([1, 3])).data)
    assert loss < 1e-6
