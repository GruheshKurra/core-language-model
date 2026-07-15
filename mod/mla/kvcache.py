import numpy as np

from .backend import xp
from .model import rope_tables, rotate_matrix


def _rmsnorm(x, w, eps=1e-5):
    ms = (x * x).mean(axis=-1, keepdims=True)
    return x * (1.0 / xp.sqrt(ms + eps)) * w


def _silu(z):
    return z / (1.0 + xp.exp(-z))


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = xp.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _rope(x, cos, sin, M, offset):
    T = x.shape[-2]
    c = cos[offset:offset + T]
    s = sin[offset:offset + T]
    return x * c + (x @ M) * s


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, nKV, T, dh = x.shape
    x = x.reshape(B, nKV, 1, T, dh)
    x = xp.broadcast_to(x, (B, nKV, n_rep, T, dh))
    return x.reshape(B, nKV * n_rep, T, dh)


class KVCache:
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def length(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def append(self, layer, k, v):
        if self.k[layer] is None:
            self.k[layer] = k
            self.v[layer] = v
        else:
            self.k[layer] = xp.concatenate([self.k[layer], k], axis=2)
            self.v[layer] = xp.concatenate([self.v[layer], v], axis=2)
        return self.k[layer], self.v[layer]


def forward_cached(model, ids, cache):
    cfg = model.cfg
    nH, nKV, dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    scale = 1.0 / (dh ** 0.5)
    cos, sin = rope_tables(cfg.seq_len, dh)
    M = rotate_matrix(dh)

    offset = cache.length()
    x = model.embed.weight.data[xp.asarray(ids, dtype=xp.int64)]
    B, T, _ = x.shape

    ii = xp.arange(T)
    q_abs = (offset + ii).reshape(T, 1)
    for li, blk in enumerate(model.blocks):
        a = blk.attn
        h = _rmsnorm(x, blk.attn_norm.weight.data)
        q = (h @ a.wq.data).reshape(B, T, nH, dh).transpose(0, 2, 1, 3)
        k = (h @ a.wk.data).reshape(B, T, nKV, dh).transpose(0, 2, 1, 3)
        v = (h @ a.wv.data).reshape(B, T, nKV, dh).transpose(0, 2, 1, 3)
        q = _rmsnorm(q, a.q_norm.weight.data)
        k = _rmsnorm(k, a.k_norm.weight.data)
        q = _rope(q, cos, sin, M, offset)
        k = _rope(k, cos, sin, M, offset)
        kf, vf = cache.append(li, k, v)
        kf = _repeat_kv(kf, nH // nKV)
        vf = _repeat_kv(vf, nH // nKV)
        scores = (q @ kf.transpose(0, 1, 3, 2)) * scale
        Tk = kf.shape[2]
        key_pos = xp.arange(Tk).reshape(1, Tk)
        mask = xp.where(key_pos > q_abs, -1e9, 0.0)
        scores = scores + mask
        attn = _softmax(scores)
        o = (attn @ vf).transpose(0, 2, 1, 3).reshape(B, T, nH * dh)
        x = x + o @ a.wo.data
        hn = _rmsnorm(x, blk.mlp_norm.weight.data)
        g = _silu(hn @ blk.mlp.wg.data)
        u = hn @ blk.mlp.wu.data
        x = x + (g * u) @ blk.mlp.wd.data

    x = _rmsnorm(x, model.final_norm.weight.data)
    return x @ model.embed.weight.data.T
