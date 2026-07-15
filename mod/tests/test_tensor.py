import numpy as np

from mla import Tensor, gradcheck


def _rand(*shape):
    rng = np.random.default_rng(0)
    return Tensor(rng.standard_normal(shape))


def test_add():
    ok, err = gradcheck(lambda a, b: a.add(b), [_rand(3, 4), _rand(3, 4)])
    assert ok, err


def test_add_broadcast():
    ok, err = gradcheck(lambda a, b: a.add(b), [_rand(3, 4), _rand(4)])
    assert ok, err


def test_mul():
    ok, err = gradcheck(lambda a, b: a.mul(b), [_rand(3, 4), _rand(3, 4)])
    assert ok, err


def test_mul_broadcast():
    ok, err = gradcheck(lambda a, b: a.mul(b), [_rand(2, 3, 4), _rand(4)])
    assert ok, err


def test_matmul():
    ok, err = gradcheck(lambda a, b: a.matmul(b), [_rand(3, 5), _rand(5, 2)])
    assert ok, err


def test_matmul_batched():
    ok, err = gradcheck(lambda a, b: a.matmul(b), [_rand(2, 3, 5), _rand(5, 2)])
    assert ok, err


def test_reshape():
    ok, err = gradcheck(lambda a: a.reshape(4, 3), [_rand(2, 6)])
    assert ok, err


def test_transpose():
    ok, err = gradcheck(lambda a: a.transpose((0, 2, 1)), [_rand(2, 3, 4)])
    assert ok, err


def test_sum_all():
    ok, err = gradcheck(lambda a: a.sum(), [_rand(3, 4)])
    assert ok, err


def test_sum_axis():
    ok, err = gradcheck(lambda a: a.sum(axis=1), [_rand(3, 4)])
    assert ok, err


def test_gather():
    idx = np.array([0, 2, 2, 4, 1])
    ok, err = gradcheck(lambda w: w.gather(idx), [_rand(5, 3)])
    assert ok, err


def test_chain_add_mul_sum():
    ok, err = gradcheck(lambda a, b: a.mul(b).add(a).sum(), [_rand(3, 4), _rand(3, 4)])
    assert ok, err
