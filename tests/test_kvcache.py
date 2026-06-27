import numpy as np
import pytest

from zyn.generate import generate
from zyn.gpt import GPT, GPTConfig
from zyn.kvcache import CachedGPT, generate_cached


def _model(seed=0, max_seq=32):
    np.random.seed(seed)
    return GPT(GPTConfig(vocab_size=20, d_model=24, n_head=3, n_layer=3, max_seq=max_seq))


def test_prefill_logits_match_full_forward():
    model = _model()
    idx = np.random.randint(0, 20, size=(2, 7))
    cached = CachedGPT(model)
    last = cached.prefill(idx)
    full = model(idx).data[:, -1, :]
    assert np.allclose(last, full, atol=1e-8)


def test_decode_steps_match_full_recompute():
    model = _model()
    idx = np.random.randint(0, 20, size=(1, 5))
    cached = CachedGPT(model)
    cached.prefill(idx)
    seq = idx
    for _ in range(4):
        nxt = np.array([[int(model(seq).data[0, -1].argmax())]])
        seq = np.concatenate([seq, nxt], axis=1)
        step_logits = cached.decode_step(nxt[:, -1])
        full_logits = model(seq).data[:, -1, :]
        assert np.allclose(step_logits, full_logits, atol=1e-7)


def test_cached_greedy_equals_uncached_greedy():
    model = _model()
    idx = np.random.randint(0, 20, size=(2, 6))
    a = generate(model, idx, max_new_tokens=8, temperature=0.0)
    b = generate_cached(model, idx, max_new_tokens=8, temperature=0.0)
    assert np.array_equal(a, b)


def test_cached_output_shape_and_input_preserved():
    model = _model()
    idx = np.random.randint(0, 20, size=(1, 4))
    out = generate_cached(model, idx, max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 9)
    assert np.array_equal(out[:, :4], idx)


def test_eos_stops_cached_generation():
    model = _model()
    idx = np.random.randint(0, 20, size=(1, 4))
    eos = int(CachedGPT(model).prefill(idx).argmax(axis=-1)[0])
    out = generate_cached(model, idx, max_new_tokens=10, temperature=0.0, eos_id=eos)
    assert out.shape[1] == 5


def test_exceeding_max_seq_raises():
    model = _model(max_seq=8)
    cached = CachedGPT(model)
    with pytest.raises(ValueError):
        cached.feed(np.random.randint(0, 20, size=(1, 9)))
