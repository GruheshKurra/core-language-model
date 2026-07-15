import numpy as np

from mla.backend import xp
from mla.model import Config, Model
from mla.optim import AdamW
from mla.train import train, train_step


def _overfit_cfg():
    return Config(vocab_size=16, d_model=32, n_layers=2, n_heads=2,
                  n_kv_heads=1, head_dim=16, swiglu_hidden=32, seq_len=16)


def _seq():
    s = xp.asarray([[1, 5, 2, 9, 3, 7, 4, 8]])
    return s[:, :-1], s[:, 1:]


def test_train_step_reduces_loss():
    xp.random.seed(0)
    model = Model(_overfit_cfg())
    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    x, y = _seq()
    first = train_step(model, opt, x, y)
    for _ in range(20):
        last = train_step(model, opt, x, y)
    assert last < first


def test_overfit_one_sequence():
    xp.random.seed(0)
    model = Model(_overfit_cfg())
    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    x, y = _seq()
    hist = train(model, opt, [(x, y) for _ in range(800)],
                 peak_lr=1e-2, warmup_steps=40, total_steps=800)
    assert hist[-1] < 0.05, hist[-1]


def test_lr_follows_schedule_during_train():
    xp.random.seed(0)
    model = Model(_overfit_cfg())
    opt = AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
    x, y = _seq()
    train(model, opt, [(x, y) for _ in range(5)],
          peak_lr=1.0, warmup_steps=10, total_steps=100)
    assert abs(opt.lr - 1.0 * 5 / 10) < 1e-12
