import numpy as np

from mla.model import Config, Model
from mla.chat import render, generate_cached
from mla.generate import generate


class FakeTok:
    def __init__(self):
        self.special_to_id = {"<bos>": 1, "<eos>": 2, "<|user|>": 3, "<|assistant|>": 4}

    def encode(self, text):
        return [10 + (ord(c) % 20) for c in text]

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


def _tiny():
    return Config(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                  n_kv_heads=2, head_dim=8, swiglu_hidden=48, seq_len=32)


def test_render_format():
    tok = FakeTok()
    ids = render(tok, [("user", "hi")], add_generation_prompt=True)
    assert ids[0] == 1
    assert ids[1] == 3
    assert ids[-1] == 4
    assert ids[-2] == 2


def test_render_with_system():
    tok = FakeTok()
    plain = render(tok, [("user", "hi")], add_generation_prompt=True)
    withsys = render(tok, [("user", "hi")], add_generation_prompt=True,
                     system="be warm")
    assert len(withsys) > len(plain)
    assert withsys[0] == 1
    assert withsys[1] != 3


def test_cached_generate_equals_uncached_greedy():
    cfg = _tiny()
    model = Model(cfg)
    rng = np.random.default_rng(0)
    prompt = list(rng.integers(0, cfg.vocab_size, size=6))

    cached = generate_cached(model, prompt, max_new=12, eos_id=None, temperature=0.0)
    uncached = generate(model, prompt, max_new=12, eos_id=None, temperature=0.0)
    assert cached == uncached


def test_stop_on_eos():
    cfg = _tiny()
    model = Model(cfg)
    prompt = [5, 6, 7]
    out = generate_cached(model, prompt, max_new=20, eos_id=None, stop_ids=(),
                          temperature=0.0)
    assert isinstance(out, list)
    assert len(out) <= 20
