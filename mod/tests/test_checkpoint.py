import numpy as np

from mla.backend import xp
from mla.model import Config, Model
from mla.optim import AdamW
from mla.train import train
from mla.checkpoint import save_checkpoint, load_checkpoint


def _cfg():
    return Config(vocab_size=16, d_model=32, n_layers=2, n_heads=2,
                  n_kv_heads=1, head_dim=16, swiglu_hidden=32, seq_len=16)


def _seq():
    s = xp.asarray([[1, 5, 2, 9, 3, 7, 4, 8]])
    return s[:, :-1], s[:, 1:]


def test_save_load_roundtrip_params(tmp_path):
    xp.random.seed(0)
    model = Model(_cfg())
    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    x, y = _seq()
    train(model, opt, [(x, y) for _ in range(6)],
          peak_lr=1e-2, warmup_steps=3, total_steps=30)
    ckpt = str(tmp_path / "ckpt.npz")
    save_checkpoint(ckpt, model, opt, step=6)

    model2, opt2, step2 = load_checkpoint(ckpt)
    assert step2 == 6
    assert opt2.t == opt.t
    for p, q in zip(model.parameters(), model2.parameters()):
        assert np.allclose(np.asarray(p.data), np.asarray(q.data))
    for a, b in zip(opt.m, opt2.m):
        assert np.allclose(np.asarray(a), np.asarray(b))
    for a, b in zip(opt.v, opt2.v):
        assert np.allclose(np.asarray(a), np.asarray(b))


def test_resume_equivalence(tmp_path):
    xp.random.seed(0)
    model = Model(_cfg())
    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    x, y = _seq()
    train(model, opt, [(x, y) for _ in range(10)],
          peak_lr=1e-2, warmup_steps=5, total_steps=40, start_step=0)
    ckpt = str(tmp_path / "ckpt.npz")
    save_checkpoint(ckpt, model, opt, step=10)

    hist_a = train(model, opt, [(x, y) for _ in range(6)],
                   peak_lr=1e-2, warmup_steps=5, total_steps=40, start_step=10)

    model2, opt2, step2 = load_checkpoint(ckpt)
    hist_b = train(model2, opt2, [(x, y) for _ in range(6)],
                   peak_lr=1e-2, warmup_steps=5, total_steps=40, start_step=10)

    assert np.allclose(hist_a, hist_b)
    for p, q in zip(model.parameters(), model2.parameters()):
        assert np.allclose(np.asarray(p.data), np.asarray(q.data))
