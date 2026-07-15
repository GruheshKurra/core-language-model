import numpy as np

from mla import Tensor, gradcheck, exp, log, rsqrt, silu, gelu, softmax


def _rand(*shape):
    rng = np.random.default_rng(1)
    return Tensor(rng.standard_normal(shape))


def _pos(*shape):
    rng = np.random.default_rng(2)
    return Tensor(np.abs(rng.standard_normal(shape)) + 0.5)


def test_exp():
    ok, err = gradcheck(lambda a: exp(a), [_rand(3, 4)])
    assert ok, err


def test_log():
    ok, err = gradcheck(lambda a: log(a), [_pos(3, 4)])
    assert ok, err


def test_rsqrt():
    ok, err = gradcheck(lambda a: rsqrt(a), [_pos(3, 4)])
    assert ok, err


def test_silu():
    ok, err = gradcheck(lambda a: silu(a), [_rand(3, 4)])
    assert ok, err


def test_gelu():
    ok, err = gradcheck(lambda a: gelu(a), [_rand(3, 4)])
    assert ok, err


def test_softmax():
    w = Tensor(np.random.default_rng(3).standard_normal((3, 5)))
    ok, err = gradcheck(lambda a: softmax(a, axis=-1).mul(w), [_rand(3, 5)])
    assert ok, err
