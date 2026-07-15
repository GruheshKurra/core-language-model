import numpy as np

from mla.model import Config, Model
from mla.kvcache import KVCache, forward_cached


def _tiny():
    return Config(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                  n_kv_heads=2, head_dim=8, swiglu_hidden=48, seq_len=16)


def test_cached_matches_full_forward():
    cfg = _tiny()
    model = Model(cfg)
    rng = np.random.default_rng(0)
    T = 10
    ids = rng.integers(0, cfg.vocab_size, size=T)

    full = model(np.array([ids], dtype=np.int64)).data[0]

    cache = KVCache(cfg.n_layers)
    step_logits = []
    for t in range(T):
        lg = forward_cached(model, np.array([[ids[t]]], dtype=np.int64), cache)
        step_logits.append(np.asarray(lg)[0, -1])
    inc = np.stack(step_logits)

    assert cache.length() == T
    assert np.allclose(full, inc, atol=1e-6, rtol=1e-6)


def test_cached_chunk_prefill_matches():
    cfg = _tiny()
    model = Model(cfg)
    rng = np.random.default_rng(1)
    T = 8
    ids = rng.integers(0, cfg.vocab_size, size=T)

    full = model(np.array([ids], dtype=np.int64)).data[0]

    cache = KVCache(cfg.n_layers)
    prefill = forward_cached(model, np.array([ids[:5]], dtype=np.int64), cache)
    assert np.allclose(np.asarray(prefill)[0], full[:5], atol=1e-6, rtol=1e-6)
    for t in range(5, T):
        lg = forward_cached(model, np.array([[ids[t]]], dtype=np.int64), cache)
        assert np.allclose(np.asarray(lg)[0, -1], full[t], atol=1e-6, rtol=1e-6)
