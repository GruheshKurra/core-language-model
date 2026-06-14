import numpy as np

from zyn.gpt import GPT, GPTConfig
from zyn.gradcheck import gradcheck
from zyn.tensor import Tensor


def _tiny():
    return GPT(GPTConfig(vocab_size=20, d_model=8, n_head=2, n_layer=2, max_seq=16))


def test_logits_shape():
    model = _tiny()
    idx = np.random.randint(0, 20, size=(3, 5))
    logits = model(idx)
    assert logits.shape == (3, 5, 20)


def test_weight_tying_no_separate_head():
    model = _tiny()
    params = model.parameters()
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids))
    assert id(model.tok_emb.weight) in ids
    n_blocks = 2
    expected = 1 + 1 + n_blocks * (2 + 4 + 2 + 4) + 2
    assert len(params) == expected


def test_tied_weight_gets_grad_from_both_paths():
    model = _tiny()
    idx = np.random.randint(0, 20, size=(1, 4))
    model(idx).sum().backward()
    assert np.any(model.tok_emb.weight.grad != 0.0)


def test_future_token_does_not_affect_past_logits():
    rng = np.random.default_rng(0)
    model = _tiny()
    idx = rng.integers(0, 20, size=(1, 5))
    base = model(idx.copy()).data
    mod = idx.copy()
    mod[0, 3] = (mod[0, 3] + 1) % 20
    out = model(mod).data
    assert np.allclose(base[0, :3], out[0, :3])
    assert not np.allclose(base[0, 3], out[0, 3])


def test_max_seq_guard():
    model = _tiny()
    try:
        model(np.zeros((1, 17), dtype=np.int64))
        assert False
    except ValueError:
        pass


def test_num_params_positive():
    model = _tiny()
    assert model.num_params() > 0


def test_param_gradcheck_final_ln():
    rng = np.random.default_rng(1)
    model = GPT(GPTConfig(vocab_size=12, d_model=4, n_head=2, n_layer=1, max_seq=8))
    idx = rng.integers(0, 12, size=(1, 3))
    g = rng.normal(size=(1, 3, 12))
    g0 = model.ln_f.gamma.data.copy()

    def f(gv):
        model.ln_f.gamma.data = gv.copy()
        out = model(idx.copy()).data
        model.ln_f.gamma.data = g0.copy()
        return float((out * g).sum())

    model.ln_f.gamma.data = g0.copy()
    model(idx.copy()).mul(Tensor(g)).sum().backward()
    assert gradcheck(f, g0.copy(), model.ln_f.gamma.grad, tol=1e-5)
