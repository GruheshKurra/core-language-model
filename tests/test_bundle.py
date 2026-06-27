import json

import numpy as np

from zyn.bpe import BPETokenizer
from zyn.bundle import load_bundle, save_bundle
from zyn.gpt import GPT, GPTConfig


def _setup(seed=0):
    np.random.seed(seed)
    model = GPT(GPTConfig(vocab_size=262, d_model=16, n_head=2, n_layer=2, max_seq=32))
    tok = BPETokenizer()
    return model, tok


def test_bundle_roundtrip_logits(tmp_path):
    model, tok = _setup()
    idx = np.random.randint(0, 262, size=(2, 6))
    before = model(idx).data

    save_bundle(tmp_path / "b", model, tok)
    bundle = load_bundle(tmp_path / "b")

    assert np.allclose(bundle.model(idx).data, before)
    assert bundle.config.vocab_size == 262


def test_bundle_files_written(tmp_path):
    model, tok = _setup()
    save_bundle(tmp_path / "b", model, tok, meta={"note": "test"})
    info = json.loads((tmp_path / "b" / "bundle.json").read_text())
    assert info["version"] == 1
    assert info["meta"]["note"] == "test"
    assert (tmp_path / "b" / "tokenizer.json").exists()
    assert (tmp_path / "b" / "config.json").exists()


def test_bundle_cached_model_matches(tmp_path):
    model, tok = _setup()
    idx = np.random.randint(0, 262, size=(1, 5))
    save_bundle(tmp_path / "b", model, tok)
    bundle = load_bundle(tmp_path / "b")

    cached = bundle.cached_model()
    last = cached.prefill(idx)
    assert np.allclose(last, bundle.model(idx).data[:, -1, :], atol=1e-8)


def test_tokenizer_preserved(tmp_path):
    model, tok = _setup()
    save_bundle(tmp_path / "b", model, tok)
    bundle = load_bundle(tmp_path / "b")
    assert bundle.tokenizer.vocab_size == tok.vocab_size
    assert bundle.tokenizer.eos_id == tok.eos_id
