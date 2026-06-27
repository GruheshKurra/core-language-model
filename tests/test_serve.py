import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from zyn.bpe import BPETokenizer
from zyn.bundle import save_bundle
from zyn.gpt import GPT, GPTConfig


def _make_bundle(path):
    np.random.seed(0)
    model = GPT(GPTConfig(vocab_size=262, d_model=16, n_head=2, n_layer=2, max_seq=64))
    tok = BPETokenizer()
    save_bundle(path, model, tok)
    return path


def test_health_without_model(monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    from serve.app import create_app

    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False


def test_health_with_model(tmp_path, monkeypatch):
    d = _make_bundle(tmp_path / "bundle")
    monkeypatch.setenv("MODEL_DIR", str(d))
    from serve.app import create_app

    client = TestClient(create_app())
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["vocab_size"] == 262
    assert body["n_params"] > 0


def test_generate_endpoint(tmp_path, monkeypatch):
    d = _make_bundle(tmp_path / "bundle")
    monkeypatch.setenv("MODEL_DIR", str(d))
    from serve.app import create_app

    client = TestClient(create_app())
    r = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 3, "temperature": 0.0})
    assert r.status_code == 200
    body = r.json()
    assert len(body["tokens"]) <= 3
    assert isinstance(body["text"], str)


def test_chat_endpoint(tmp_path, monkeypatch):
    d = _make_bundle(tmp_path / "bundle")
    monkeypatch.setenv("MODEL_DIR", str(d))
    from serve.app import create_app

    client = TestClient(create_app())
    r = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "max_new_tokens": 3, "temperature": 0.0},
    )
    assert r.status_code == 200
    assert r.json()["message"]["role"] == "assistant"


def test_generate_returns_503_without_model(monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    from serve.app import create_app

    client = TestClient(create_app())
    r = client.post("/generate", json={"prompt": "hi"})
    assert r.status_code == 503
