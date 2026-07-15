import math

from mla.schedule import lr_schedule


def test_warmup_linear():
    assert abs(lr_schedule(0, 1.0, 10, 110) - 0.1) < 1e-12
    assert abs(lr_schedule(4, 1.0, 10, 110) - 0.5) < 1e-12
    assert abs(lr_schedule(9, 1.0, 10, 110) - 1.0) < 1e-12


def test_peak_at_warmup_boundary():
    assert abs(lr_schedule(10, 1.0, 10, 110) - 1.0) < 1e-12


def test_cosine_midpoint():
    assert abs(lr_schedule(60, 1.0, 10, 110, min_lr=0.0) - 0.5) < 1e-12


def test_cosine_quarter():
    assert abs(lr_schedule(35, 1.0, 10, 110) - 0.5 * (1.0 + math.cos(math.pi * 0.25))) < 1e-12


def test_decays_to_min():
    assert abs(lr_schedule(110, 1.0, 10, 110) - 0.0) < 1e-12
    assert abs(lr_schedule(500, 1.0, 10, 110) - 0.0) < 1e-12


def test_min_lr_floor():
    assert abs(lr_schedule(110, 1.0, 10, 110, min_lr=0.1) - 0.1) < 1e-12
    assert abs(lr_schedule(60, 1.0, 10, 110, min_lr=0.2) - (0.2 + 0.8 * 0.5)) < 1e-12


def test_monotonic_non_increasing_after_warmup():
    xs = [lr_schedule(s, 1.0, 10, 110) for s in range(10, 111)]
    assert all(xs[i] >= xs[i + 1] - 1e-15 for i in range(len(xs) - 1))
