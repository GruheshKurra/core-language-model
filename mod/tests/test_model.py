import numpy as np

from mla.backend import xp
from mla.tensor import Tensor
from mla.gradcheck import gradcheck
from mla.model import (Config, TiedEmbedding, RoPE, RMSNorm, repeat_kv,
                       Attention, SwiGLU, Block, Model)


def _tiny_cfg():
    return Config(vocab_size=7, d_model=4, n_layers=1, n_heads=2,
                  n_kv_heads=1, head_dim=2, swiglu_hidden=8, seq_len=3)


def test_embed_shape():
    xp.random.seed(0)
    emb = TiedEmbedding(_tiny_cfg())
    ids = xp.asarray([[1, 3, 3], [0, 2, 5]])
    out = emb.embed(ids)
    assert out.shape == (2, 3, 4)


def test_project_shape():
    xp.random.seed(0)
    emb = TiedEmbedding(_tiny_cfg())
    h = Tensor(xp.random.randn(2, 3, 4))
    logits = emb.project(h)
    assert logits.shape == (2, 3, 7)


def test_tied_weight_gradcheck():
    xp.random.seed(0)
    ids = xp.asarray([[1, 3, 3], [0, 2, 5]])

    def f(w):
        emb = w.gather(ids)
        return emb.matmul(w.transpose())

    w = Tensor(xp.random.randn(7, 4))
    ok, rel = gradcheck(f, w)
    assert ok, f"tied-embedding gradcheck failed, max_rel={rel}"


def test_tied_row_asymmetry():
    xp.random.seed(0)
    ids = xp.asarray([[2]])

    def f(w):
        emb = w.gather(ids)
        return emb.matmul(w.transpose())

    w = Tensor(xp.random.randn(3, 2))
    f(w).backward()
    g = np.asarray(w.grad)
    assert np.abs(g[2]).sum() > np.abs(g[0]).sum()


def _rope_cfg():
    return Config(vocab_size=7, d_model=8, n_layers=1, n_heads=2,
                  n_kv_heads=1, head_dim=4, swiglu_hidden=8, seq_len=8)


def test_rope_shape():
    rope = RoPE(_rope_cfg())
    x = Tensor(xp.random.randn(2, 2, 3, 4))
    out = rope(x)
    assert out.shape == (2, 2, 3, 4)


def test_rope_pos0_identity():
    rope = RoPE(_rope_cfg())
    x = Tensor(xp.random.randn(1, 1, 1, 4))
    out = rope(x, offset=0)
    assert np.allclose(np.asarray(out.data), np.asarray(x.data))


def test_rope_gradcheck():
    xp.random.seed(0)
    rope = RoPE(_rope_cfg())

    def f(x):
        return rope(x, offset=2)

    x = Tensor(xp.random.randn(1, 2, 3, 4))
    ok, rel = gradcheck(f, x)
    assert ok, f"rope gradcheck failed, max_rel={rel}"


def test_rope_relative_position():
    xp.random.seed(0)
    rope = RoPE(_rope_cfg())
    q = xp.random.randn(1, 1, 1, 4)
    k = xp.random.randn(1, 1, 1, 4)

    def score(m, n):
        qm = rope(Tensor(q), offset=m)
        kn = rope(Tensor(k), offset=n)
        return float((qm.data * kn.data).sum())

    assert abs(score(1, 4) - score(3, 6)) < 1e-9
    assert abs(score(1, 4) - score(1, 5)) > 1e-6


def test_rmsnorm_shape():
    norm = RMSNorm(8)
    x = Tensor(xp.random.randn(2, 3, 8))
    assert norm(x).shape == (2, 3, 8)


def test_rmsnorm_unit_rms():
    norm = RMSNorm(8, eps=1e-12)
    x = Tensor(xp.random.randn(4, 8))
    y = np.asarray(norm(x).data)
    rms = np.sqrt((y ** 2).mean(axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-5)


def test_rmsnorm_gradcheck_x():
    xp.random.seed(0)
    norm = RMSNorm(8)

    def f(x):
        return norm(x)

    x = Tensor(xp.random.randn(2, 3, 8))
    ok, rel = gradcheck(f, x)
    assert ok, f"rmsnorm dx gradcheck failed, max_rel={rel}"


def test_rmsnorm_gradcheck_gamma():
    xp.random.seed(0)
    norm = RMSNorm(8)
    x = Tensor(xp.random.randn(2, 3, 8))

    def f(w):
        norm.weight = w
        return norm(x)

    g = Tensor(xp.random.randn(8))
    ok, rel = gradcheck(f, g)
    assert ok, f"rmsnorm dgamma gradcheck failed, max_rel={rel}"


def _attn_cfg():
    return Config(vocab_size=7, d_model=8, n_layers=1, n_heads=2,
                  n_kv_heads=1, head_dim=4, swiglu_hidden=8, seq_len=8)


def test_repeat_kv():
    x = Tensor(xp.arange(2 * 1 * 2 * 3).reshape(2, 1, 2, 3).astype(float))
    y = repeat_kv(x, 3)
    assert y.shape == (2, 3, 2, 3)
    yd = np.asarray(y.data)
    assert np.allclose(yd[:, 0], yd[:, 1])
    assert np.allclose(yd[:, 1], yd[:, 2])


def test_repeat_kv_gradcheck():
    xp.random.seed(0)

    def f(x):
        return repeat_kv(x, 3)

    x = Tensor(xp.random.randn(2, 1, 2, 3))
    ok, rel = gradcheck(f, x)
    assert ok, f"repeat_kv gradcheck failed, max_rel={rel}"


def test_attention_shape():
    xp.random.seed(0)
    cfg = _attn_cfg()
    attn = Attention(cfg)
    x = Tensor(xp.random.randn(2, 4, cfg.d_model))
    assert attn(x).shape == (2, 4, cfg.d_model)


def test_attention_causal():
    xp.random.seed(0)
    cfg = _attn_cfg()
    attn = Attention(cfg)
    x = xp.random.randn(1, 4, cfg.d_model)
    o1 = np.asarray(attn(Tensor(x)).data)
    x2 = x.copy()
    x2[0, 3] += 5.0
    o2 = np.asarray(attn(Tensor(x2)).data)
    assert np.allclose(o1[0, :3], o2[0, :3])
    assert not np.allclose(o1[0, 3], o2[0, 3])


def test_attention_gradcheck_x():
    xp.random.seed(0)
    cfg = _attn_cfg()
    attn = Attention(cfg)

    def f(x):
        return attn(x)

    x = Tensor(xp.random.randn(1, 3, cfg.d_model))
    ok, rel = gradcheck(f, x)
    assert ok, f"attention dx gradcheck failed, max_rel={rel}"


def test_attention_gradcheck_wq():
    xp.random.seed(0)
    cfg = _attn_cfg()
    attn = Attention(cfg)
    x = Tensor(xp.random.randn(1, 3, cfg.d_model))

    def f(w):
        attn.wq = w
        return attn(x)

    w = Tensor(xp.random.randn(cfg.d_model, cfg.n_heads * cfg.head_dim))
    ok, rel = gradcheck(f, w)
    assert ok, f"attention dwq gradcheck failed, max_rel={rel}"


def test_swiglu_shape():
    xp.random.seed(0)
    cfg = _attn_cfg()
    mlp = SwiGLU(cfg)
    x = Tensor(xp.random.randn(2, 3, cfg.d_model))
    assert mlp(x).shape == (2, 3, cfg.d_model)


def test_swiglu_gradcheck():
    xp.random.seed(0)
    cfg = _attn_cfg()
    mlp = SwiGLU(cfg)

    def f(x):
        return mlp(x)

    x = Tensor(xp.random.randn(2, 3, cfg.d_model))
    ok, rel = gradcheck(f, x)
    assert ok, f"swiglu gradcheck failed, max_rel={rel}"


def test_block_gradcheck():
    xp.random.seed(0)
    cfg = _attn_cfg()
    block = Block(cfg)

    def f(x):
        return block(x)

    x = Tensor(xp.random.randn(1, 3, cfg.d_model))
    ok, rel = gradcheck(f, x)
    assert ok, f"block gradcheck failed, max_rel={rel}"


def _tiny_model_cfg():
    return Config(vocab_size=7, d_model=8, n_layers=2, n_heads=2,
                  n_kv_heads=1, head_dim=4, swiglu_hidden=8, seq_len=8)


def test_model_forward_shape():
    xp.random.seed(0)
    cfg = _tiny_model_cfg()
    m = Model(cfg)
    ids = xp.asarray([[1, 3, 5, 0], [2, 2, 4, 6]])
    out = m(ids)
    assert out.shape == (2, 4, cfg.vocab_size)


def test_model_gradcheck():
    xp.random.seed(0)
    cfg = _tiny_model_cfg()
    m = Model(cfg)
    ids = xp.asarray([[1, 3, 5], [2, 4, 6]])

    def f(w):
        m.embed.weight = w
        return m(ids)

    w = Tensor(xp.random.randn(cfg.vocab_size, cfg.d_model) * 0.1)
    ok, rel = gradcheck(f, w)
    assert ok, f"full-model gradcheck failed, max_rel={rel}"


def test_param_count():
    m = Model(Config())
    assert m.n_params() == 3_869_184
