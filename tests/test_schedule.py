import numpy as np

from zyn.schedule import cosine_lr


def test_warmup_is_linear():
    lr_max, W, S = 1.0, 10, 100
    vals = [cosine_lr(s, lr_max, W, S) for s in range(W)]
    assert np.isclose(vals[0], lr_max * 1 / W)
    assert np.isclose(vals[-1], lr_max)
    diffs = np.diff(vals)
    assert np.allclose(diffs, diffs[0])


def test_peak_at_warmup_end():
    assert np.isclose(cosine_lr(9, 1.0, 10, 100), 1.0)


def test_cosine_midpoint_is_half():
    lr_max, lr_min, W, S = 1.0, 0.0, 10, 110
    mid = W + (S - W) // 2
    assert np.isclose(cosine_lr(mid, lr_max, W, S, lr_min), 0.5, atol=1e-9)


def test_decays_to_lr_min_at_end():
    assert np.isclose(cosine_lr(100, 1.0, 10, 100, lr_min=0.1), 0.1)
    assert np.isclose(cosine_lr(500, 1.0, 10, 100, lr_min=0.1), 0.1)


def test_monotonic_decrease_after_warmup():
    vals = [cosine_lr(s, 1.0, 10, 100) for s in range(10, 100)]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


def test_no_warmup_starts_at_peak():
    assert np.isclose(cosine_lr(0, 1.0, 0, 100), 1.0)
