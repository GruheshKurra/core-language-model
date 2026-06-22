import numpy as np

from zyn.eval import evaluate
from zyn.gpt import GPT, GPTConfig
from zyn.optim import AdamW
from zyn.train import train


def _model(seed=0, vocab=16):
    np.random.seed(seed)
    return GPT(GPTConfig(vocab_size=vocab, d_model=16, n_head=2, n_layer=2, max_seq=24))


def test_perplexity_is_exp_loss():
    model = _model()
    x = np.random.randint(0, 16, size=(2, 6))
    y = np.random.randint(0, 16, size=(2, 6))
    out = evaluate(model, [(x, y)])
    assert np.isclose(out["perplexity"], np.exp(out["loss"]))
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["tokens"] == 12


def test_random_model_perplexity_near_vocab():
    model = _model(vocab=16)
    x = np.random.randint(0, 16, size=(4, 8))
    y = np.random.randint(0, 16, size=(4, 8))
    out = evaluate(model, [(x, y)])
    assert 4.0 < out["perplexity"] < 64.0


def test_overfit_then_eval_perfect():
    rng = np.random.default_rng(7)
    model = GPT(GPTConfig(vocab_size=24, d_model=32, n_head=4, n_layer=2, max_seq=32))
    seq = rng.integers(0, 24, size=(1, 17))
    x, y = seq[:, :-1], seq[:, 1:]
    opt = AdamW(model.parameters(), lr=3e-3)
    train(model, opt, lambda: (x, y), steps=400, lr_max=3e-3,
          warmup_steps=20, max_steps=400)

    out = evaluate(model, [(x, y)])
    assert out["accuracy"] == 1.0
    assert out["perplexity"] < 1.1


def test_ignore_index_excludes_tokens():
    model = _model()
    x = np.random.randint(0, 16, size=(1, 5))
    y = np.array([[3, -100, 7, -100, 2]])
    out = evaluate(model, [(x, y)], ignore_index=-100)
    assert out["tokens"] == 3
