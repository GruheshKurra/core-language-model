import numpy as np

from zyn.optim import AdamW
from zyn.tensor import Tensor


def _reference_step(p, m, v, g, t, lr, b1, b2, eps, wd):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    if wd != 0.0:
        p = p - lr * wd * p
    p = p - lr * mh / (np.sqrt(vh) + eps)
    return p, m, v


def test_matches_reference_over_many_steps():
    rng = np.random.default_rng(0)
    p = Tensor(rng.normal(size=(3, 4)))
    opt = AdamW([p], lr=1e-2, weight_decay=0.01)

    rp = p.data.copy()
    rm = np.zeros_like(rp)
    rv = np.zeros_like(rp)

    for t in range(1, 21):
        g = rng.normal(size=(3, 4))
        p.grad = g.copy()
        opt.step()
        rp, rm, rv = _reference_step(rp, rm, rv, g, t, 1e-2, 0.9, 0.999, 1e-8, 0.01)
        assert np.allclose(p.data, rp, atol=1e-12)


def test_converges_on_quadratic():
    p = Tensor(np.array([5.0, -3.0, 2.0]))
    opt = AdamW([p], lr=0.1)
    for _ in range(2000):
        opt.zero_grad()
        p.grad = p.data.copy()
        opt.step()
    assert np.allclose(p.data, 0.0, atol=1e-3)


def test_first_step_magnitude_near_lr():
    p = Tensor(np.array([2.0, -5.0, 0.3]))
    lr = 0.01
    opt = AdamW([p], lr=lr)
    p0 = p.data.copy()
    p.grad = np.array([4.0, -1.0, 9.0])
    opt.step()
    assert np.allclose(np.abs(p.data - p0), lr, atol=1e-6)


def test_zero_grad():
    p = Tensor(np.ones((2, 2)))
    opt = AdamW([p])
    p.grad = np.full((2, 2), 7.0)
    opt.zero_grad()
    assert np.allclose(p.grad, 0.0)


def test_decoupled_weight_decay_shrinks_with_zero_grad():
    p = Tensor(np.array([10.0, -4.0]))
    lr, wd = 0.1, 0.5
    opt = AdamW([p], lr=lr, weight_decay=wd)
    p0 = p.data.copy()
    p.grad = np.zeros(2)
    opt.step()
    assert np.allclose(p.data, p0 * (1 - lr * wd))


def test_plain_adam_when_wd_zero():
    rng = np.random.default_rng(1)
    p = Tensor(rng.normal(size=4))
    opt = AdamW([p], lr=1e-3, weight_decay=0.0)
    p0 = p.data.copy()
    p.grad = np.zeros(4)
    opt.step()
    assert np.allclose(p.data, p0)


def test_trains_gpt_one_step_reduces_loss():
    from zyn.gpt import GPT, GPTConfig
    from zyn.loss import cross_entropy

    rng = np.random.default_rng(2)
    model = GPT(GPTConfig(vocab_size=20, d_model=8, n_head=2, n_layer=2, max_seq=16))
    idx = rng.integers(0, 20, size=(2, 6))
    y = rng.integers(0, 20, size=(2, 6))
    opt = AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = cross_entropy(model(idx), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    assert losses[-1] < losses[0]
