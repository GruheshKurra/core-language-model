import numpy as np

from zyn.gradcheck import gradcheck


def test_sum_of_squares():
    x = np.random.randn(4, 3)
    f = lambda v: np.sum(v ** 2)
    assert gradcheck(f, x, analytic_grad=2 * x)


def test_matmul_scalar():
    W = np.random.randn(5, 3)
    x = np.random.randn(3)
    f = lambda v: np.sum(W @ v)
    assert gradcheck(f, x, analytic_grad=W.T @ np.ones(5))


def test_catches_wrong_grad():
    x = np.random.randn(4)
    f = lambda v: np.sum(v ** 2)
    assert not gradcheck(f, x, analytic_grad=3 * x, verbose=False)
