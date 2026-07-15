import numpy as np

from mla.generate import sample_next, _softmax


def test_temperature_zero_is_greedy():
    logits = np.array([0.1, 3.0, 0.2, 2.9])
    assert sample_next(logits, temperature=0.0) == 1


def test_top_k_one_is_greedy():
    logits = np.array([1.0, 0.5, 5.0, 2.0, 0.0])
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert sample_next(logits, temperature=1.0, top_k=1, rng=rng) == 2


def test_top_p_restricts_to_nucleus():
    logits = np.log(np.array([0.70, 0.20, 0.05, 0.05]))
    rng = np.random.default_rng(1)
    picks = {sample_next(logits, temperature=1.0, top_p=0.85, rng=rng) for _ in range(200)}
    assert picks <= {0, 1}


def test_reproducible_with_seed():
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    a = [sample_next(logits, temperature=1.0, rng=np.random.default_rng(42)) for _ in range(5)]
    b = [sample_next(logits, temperature=1.0, rng=np.random.default_rng(42)) for _ in range(5)]
    assert a == b


def test_softmax_sums_to_one():
    z = np.array([2.0, -1.0, 0.5, 3.0])
    p = _softmax(z)
    assert abs(p.sum() - 1.0) < 1e-12
