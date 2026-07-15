import numpy as np

from mla.backend import xp
from mla.tensor import Tensor
from mla.optim import AdamW, clip_grad_norm


def test_adamw_converges_quadratic():
    xp.random.seed(0)
    target = xp.asarray([3.0, -2.0, 0.5, 1.0])
    w = Tensor(xp.random.randn(4))
    opt = AdamW([w], lr=0.1, weight_decay=0.0)
    loss = None
    for _ in range(600):
        opt.zero_grad()
        diff = w.sub(target)
        loss = diff.mul(diff).sum()
        loss.backward()
        opt.step()
    assert float(loss.data) < 1e-4, float(loss.data)
    assert np.allclose(np.asarray(w.data), np.asarray(target), atol=1e-2)


def test_adamw_first_step_scale():
    w = Tensor(xp.asarray([10.0]))
    opt = AdamW([w], lr=0.1, weight_decay=0.0)
    opt.zero_grad()
    w.grad = xp.asarray([0.5])
    opt.step()
    assert abs(float(w.data[0]) - 9.9) < 1e-6


def test_weight_decay_pulls_to_zero():
    w = Tensor(xp.asarray([5.0]))
    opt = AdamW([w], lr=0.1, weight_decay=0.1)
    for _ in range(50):
        opt.zero_grad()
        w.grad = xp.zeros_like(w.data)
        opt.step()
    assert float(w.data[0]) < 5.0


def test_clip_grad_norm_scales():
    p = Tensor(xp.asarray([3.0, 4.0]))
    p.grad = xp.asarray([3.0, 4.0])
    total = clip_grad_norm([p], max_norm=1.0)
    assert abs(total - 5.0) < 1e-6
    new_norm = float((np.asarray(p.grad) ** 2).sum()) ** 0.5
    assert abs(new_norm - 1.0) < 1e-4


def test_clip_grad_norm_noop_under_threshold():
    p = Tensor(xp.asarray([0.3, 0.4]))
    p.grad = xp.asarray([0.3, 0.4])
    total = clip_grad_norm([p], max_norm=1.0)
    assert abs(total - 0.5) < 1e-6
    new_norm = float((np.asarray(p.grad) ** 2).sum()) ** 0.5
    assert abs(new_norm - 0.5) < 1e-6
