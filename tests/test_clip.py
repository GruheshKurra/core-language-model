import numpy as np

from zyn.optim import clip_grad_norm
from zyn.tensor import Tensor


def _global_norm(params):
    return np.sqrt(sum(float((p.grad * p.grad).sum()) for p in params))


def test_returns_total_norm_pre_clip():
    a = Tensor(np.zeros(3))
    b = Tensor(np.zeros(2))
    a.grad = np.array([3.0, 0.0, 0.0])
    b.grad = np.array([4.0, 0.0])
    norm = clip_grad_norm([a, b], max_norm=100.0)
    assert np.isclose(norm, 5.0)


def test_no_scaling_when_under_threshold():
    a = Tensor(np.zeros(2))
    a.grad = np.array([0.3, 0.4])
    g0 = a.grad.copy()
    clip_grad_norm([a], max_norm=10.0)
    assert np.allclose(a.grad, g0)


def test_scales_down_to_max_norm():
    a = Tensor(np.zeros(3))
    b = Tensor(np.zeros(2))
    a.grad = np.array([3.0, 0.0, 0.0])
    b.grad = np.array([4.0, 0.0])
    clip_grad_norm([a, b], max_norm=1.0, eps=0.0)
    assert np.isclose(_global_norm([a, b]), 1.0)


def test_direction_preserved():
    a = Tensor(np.zeros(4))
    a.grad = np.array([1.0, -2.0, 3.0, -4.0])
    g0 = a.grad.copy()
    clip_grad_norm([a], max_norm=1.0, eps=0.0)
    unit_before = g0 / np.linalg.norm(g0)
    unit_after = a.grad / np.linalg.norm(a.grad)
    assert np.allclose(unit_before, unit_after)


def test_clipping_in_training_loop():
    from zyn.gpt import GPT, GPTConfig
    from zyn.loss import cross_entropy
    from zyn.optim import AdamW

    rng = np.random.default_rng(0)
    model = GPT(GPTConfig(vocab_size=20, d_model=8, n_head=2, n_layer=2, max_seq=16))
    idx = rng.integers(0, 20, size=(2, 6))
    y = rng.integers(0, 20, size=(2, 6))
    opt = AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = cross_entropy(model(idx), y)
        loss.backward()
        norm = clip_grad_norm(model.parameters(), max_norm=1.0)
        assert norm >= 0.0
        assert _global_norm(model.parameters()) <= 1.0 + 1e-6
        opt.step()
        losses.append(float(loss.data))
    assert losses[-1] < losses[0]
