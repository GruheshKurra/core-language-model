import numpy as np

from zyn.checkpoint import load_checkpoint, save_checkpoint
from zyn.gpt import GPT, GPTConfig
from zyn.optim import AdamW
from zyn.train import train, train_step


def _setup(seed=0):
    np.random.seed(seed)
    model = GPT(GPTConfig(vocab_size=16, d_model=16, n_head=2, n_layer=2, max_seq=24))
    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    return model, opt


def test_roundtrip_params_and_logits(tmp_path):
    model, opt = _setup()
    x = np.random.randint(0, 16, size=(2, 6))
    y = np.random.randint(0, 16, size=(2, 6))
    for _ in range(5):
        train_step(model, opt, x, y)

    out_before = model(x).data
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, opt, step=5)

    model2, opt2, step2 = load_checkpoint(path)
    assert step2 == 5
    assert np.allclose(model2(x).data, out_before)


def test_optimizer_state_restored(tmp_path):
    model, opt = _setup()
    x = np.random.randint(0, 16, size=(1, 5))
    y = np.random.randint(0, 16, size=(1, 5))
    for _ in range(8):
        train_step(model, opt, x, y)

    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, opt, step=8)
    _, opt2, _ = load_checkpoint(path)

    assert opt2.t == opt.t
    for a, b in zip(opt.m, opt2.m):
        assert np.allclose(a, b)
    for a, b in zip(opt.v, opt2.v):
        assert np.allclose(a, b)


def test_resume_matches_uninterrupted(tmp_path):
    x = np.random.randint(0, 16, size=(1, 6))
    y = np.random.randint(0, 16, size=(1, 6))

    np.random.seed(123)
    m_full = GPT(GPTConfig(vocab_size=16, d_model=16, n_head=2, n_layer=2, max_seq=24))
    o_full = AdamW(m_full.parameters(), lr=5e-3)
    train(m_full, o_full, lambda: (x, y), steps=20, lr_max=5e-3, warmup_steps=4, max_steps=20)
    ref = m_full(x).data

    np.random.seed(123)
    m_a = GPT(GPTConfig(vocab_size=16, d_model=16, n_head=2, n_layer=2, max_seq=24))
    o_a = AdamW(m_a.parameters(), lr=5e-3)
    train(m_a, o_a, lambda: (x, y), steps=10, lr_max=5e-3, warmup_steps=4, max_steps=20)
    path = tmp_path / "mid.npz"
    save_checkpoint(path, m_a, o_a, step=10)

    m_b, o_b, step = load_checkpoint(path)
    train(m_b, o_b, lambda: (x, y), steps=10, lr_max=5e-3, warmup_steps=4,
          max_steps=20, start_step=step)
    resumed = m_b(x).data

    assert np.allclose(resumed, ref, atol=1e-10)
