import numpy as np

from zyn.generate import generate
from zyn.gpt import GPT, GPTConfig


def _model(seed=0, max_seq=24):
    np.random.seed(seed)
    return GPT(GPTConfig(vocab_size=16, d_model=16, n_head=2, n_layer=2, max_seq=max_seq))


def test_output_grows_by_new_tokens():
    model = _model()
    idx = np.random.randint(0, 16, size=(2, 4))
    out = generate(model, idx, max_new_tokens=5, temperature=0.0)
    assert out.shape == (2, 9)
    assert np.array_equal(out[:, :4], idx)


def test_greedy_is_deterministic():
    model = _model()
    idx = np.random.randint(0, 16, size=(1, 4))
    a = generate(model, idx, max_new_tokens=6, temperature=0.0)
    b = generate(model, idx, max_new_tokens=6, temperature=0.0)
    assert np.array_equal(a, b)


def test_top_k_one_equals_greedy():
    model = _model()
    idx = np.random.randint(0, 16, size=(1, 4))
    rng = np.random.default_rng(0)
    greedy = generate(model, idx, max_new_tokens=6, temperature=0.0)
    topk = generate(model, idx, max_new_tokens=6, temperature=1.0, top_k=1, rng=rng)
    assert np.array_equal(greedy, topk)


def test_1d_input_promoted_to_batch():
    model = _model()
    idx = np.array([1, 2, 3])
    out = generate(model, idx, max_new_tokens=3, temperature=0.0)
    assert out.shape == (1, 6)


def test_context_cropped_to_max_seq():
    model = _model(max_seq=8)
    idx = np.random.randint(0, 16, size=(1, 20))
    out = generate(model, idx, max_new_tokens=4, temperature=0.0)
    assert out.shape == (1, 24)


def test_eos_stops_generation():
    model = _model()
    idx = np.random.randint(0, 16, size=(1, 4))
    logits = model(idx).data[:, -1, :]
    eos = int(logits.argmax(axis=-1)[0])
    out = generate(model, idx, max_new_tokens=10, temperature=0.0, eos_id=eos)
    assert out.shape[1] == 5


def test_top_p_sampling_runs():
    model = _model()
    idx = np.random.randint(0, 16, size=(2, 4))
    rng = np.random.default_rng(1)
    out = generate(model, idx, max_new_tokens=4, temperature=0.8, top_p=0.9, rng=rng)
    assert out.shape == (2, 8)
    assert out.min() >= 0 and out.max() < 16
