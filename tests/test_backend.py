import numpy as np

from zyn import backend
from zyn.gpt import GPT, GPTConfig


def test_default_backend_is_numpy():
    assert backend.backend_name() == "numpy"
    assert backend.is_gpu() is False
    assert backend.xp is np


def test_to_numpy_passthrough():
    a = np.arange(6).reshape(2, 3)
    out = backend.to_numpy(a)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, a)


def test_model_uses_backend_arrays():
    np.random.seed(0)
    model = GPT(GPTConfig(vocab_size=16, d_model=16, n_head=2, n_layer=1, max_seq=8))
    idx = np.random.randint(0, 16, size=(1, 4))
    logits = model(idx).data
    assert logits.shape == (1, 4, 16)
