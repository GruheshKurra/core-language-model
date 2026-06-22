import numpy as np

from zyn.gpt import GPT, GPTConfig
from zyn.optim import AdamW
from zyn.train import train, train_step


def _model(seed=0):
    np.random.seed(seed)
    return GPT(GPTConfig(vocab_size=16, d_model=16, n_head=2, n_layer=2, max_seq=24))


def test_train_step_returns_loss_and_norm():
    model = _model()
    opt = AdamW(model.parameters(), lr=1e-2)
    x = np.random.randint(0, 16, size=(2, 6))
    y = np.random.randint(0, 16, size=(2, 6))
    loss, gn = train_step(model, opt, x, y, max_norm=1.0)
    assert loss > 0
    assert gn >= 0


def test_lr_follows_schedule():
    model = _model()
    opt = AdamW(model.parameters(), lr=0.0)
    x = np.random.randint(0, 16, size=(1, 5))
    y = np.random.randint(0, 16, size=(1, 5))
    hist = train(model, opt, lambda: (x, y), steps=20, lr_max=1e-2,
                 warmup_steps=5, max_steps=20)
    assert np.isclose(hist[0]["lr"], 1e-2 * 1 / 5)
    assert np.isclose(hist[4]["lr"], 1e-2)
    assert hist[-1]["lr"] < hist[4]["lr"]


def test_sanity_overfit_one_paragraph():
    rng = np.random.default_rng(7)
    model = GPT(GPTConfig(vocab_size=24, d_model=32, n_head=4, n_layer=2, max_seq=32))
    seq = rng.integers(0, 24, size=(1, 17))
    x, y = seq[:, :-1], seq[:, 1:]
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    hist = train(model, opt, lambda: (x, y), steps=400, lr_max=3e-3,
                 warmup_steps=20, max_steps=400, max_norm=1.0)

    first = hist[0]["loss"]
    last = hist[-1]["loss"]
    assert last < 0.05, f"overfit failed: {first:.3f} -> {last:.3f}"

    logits = model(x).data
    preds = logits.argmax(axis=-1)
    assert np.array_equal(preds, y), "model did not memorize the sequence"
