from dataclasses import dataclass

from .backend import xp
from .tensor import Tensor
from .functional import rsqrt, silu, softmax


@dataclass
class Config:
    vocab_size: int = 4096
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 32
    swiglu_hidden: int = 704
    seq_len: int = 256


def normal(shape, std):
    return Tensor(xp.random.randn(*shape) * std)


def rope_tables(seq_len, head_dim, base=10000.0):
    h = head_dim // 2
    i = xp.arange(h)
    theta = base ** (-2.0 * i / head_dim)
    m = xp.arange(seq_len)
    freqs = xp.outer(m, theta)
    emb = xp.concatenate([freqs, freqs], axis=-1)
    return xp.cos(emb), xp.sin(emb)


def rotate_matrix(head_dim):
    h = head_dim // 2
    M = xp.zeros((head_dim, head_dim))
    for j in range(head_dim):
        if j < h:
            M[j + h, j] = -1.0
        else:
            M[j - h, j] = 1.0
    return M


class RMSNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = Tensor(xp.ones(dim))

    def __call__(self, x):
        d = x.shape[-1]
        ms = x.mul(x).sum(axis=-1, keepdims=True).mul(1.0 / d)
        inv = rsqrt(ms.add(self.eps))
        return x.mul(inv).mul(self.weight)

    def parameters(self):
        return [self.weight]


class RoPE:
    def __init__(self, cfg, base=10000.0):
        self.cfg = cfg
        cos, sin = rope_tables(cfg.seq_len, cfg.head_dim, base)
        self.cos = cos
        self.sin = sin
        self.M = Tensor(rotate_matrix(cfg.head_dim))

    def __call__(self, x, offset=0):
        T = x.shape[-2]
        cos = Tensor(self.cos[offset:offset + T])
        sin = Tensor(self.sin[offset:offset + T])
        rot = x.matmul(self.M)
        return x.mul(cos).add(rot.mul(sin))


class TiedEmbedding:
    def __init__(self, cfg):
        self.cfg = cfg
        self.weight = normal((cfg.vocab_size, cfg.d_model), 0.02)

    def embed(self, ids):
        return self.weight.gather(ids)

    def project(self, h):
        return h.matmul(self.weight.transpose())

    def parameters(self):
        return [self.weight]


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    x = x.reshape(B, n_kv, 1, T, hd)
    ones = Tensor(xp.ones((1, 1, n_rep, 1, 1)))
    x = x.mul(ones)
    return x.reshape(B, n_kv * n_rep, T, hd)


def causal_mask(T_q, T_k, offset=0):
    m = xp.zeros((T_q, T_k))
    for i in range(T_q):
        m[i, offset + i + 1:] = -1e9
    return Tensor(m)


class Attention:
    def __init__(self, cfg):
        self.cfg = cfg
        dh = cfg.head_dim
        self.wq = normal((cfg.d_model, cfg.n_heads * dh), 0.02)
        self.wk = normal((cfg.d_model, cfg.n_kv_heads * dh), 0.02)
        self.wv = normal((cfg.d_model, cfg.n_kv_heads * dh), 0.02)
        self.wo = normal((cfg.n_heads * dh, cfg.d_model), 0.02)
        self.q_norm = RMSNorm(dh)
        self.k_norm = RMSNorm(dh)
        self.rope = RoPE(cfg)
        self.scale = 1.0 / (dh ** 0.5)

    def __call__(self, x, offset=0):
        cfg = self.cfg
        B, T, _ = x.shape
        nH, nKV, dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        q = x.matmul(self.wq).reshape(B, T, nH, dh).transpose((0, 2, 1, 3))
        k = x.matmul(self.wk).reshape(B, T, nKV, dh).transpose((0, 2, 1, 3))
        v = x.matmul(self.wv).reshape(B, T, nKV, dh).transpose((0, 2, 1, 3))
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q, offset)
        k = self.rope(k, offset)
        k = repeat_kv(k, nH // nKV)
        v = repeat_kv(v, nH // nKV)
        kt = k.transpose((0, 1, 3, 2))
        scores = q.matmul(kt).mul(self.scale)
        scores = scores.add(causal_mask(T, T, offset))
        attn = softmax(scores, axis=-1)
        out = attn.matmul(v).transpose((0, 2, 1, 3)).reshape(B, T, nH * dh)
        return out.matmul(self.wo)

    def parameters(self):
        return [self.wq, self.wk, self.wv, self.wo,
                self.q_norm.weight, self.k_norm.weight]


class SwiGLU:
    def __init__(self, cfg):
        self.cfg = cfg
        self.wg = normal((cfg.d_model, cfg.swiglu_hidden), 0.02)
        self.wu = normal((cfg.d_model, cfg.swiglu_hidden), 0.02)
        self.wd = normal((cfg.swiglu_hidden, cfg.d_model), 0.02)

    def __call__(self, x):
        g = silu(x.matmul(self.wg))
        u = x.matmul(self.wu)
        return g.mul(u).matmul(self.wd)

    def parameters(self):
        return [self.wg, self.wu, self.wd]


class Block:
    def __init__(self, cfg):
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def __call__(self, x, offset=0):
        x = x.add(self.attn(self.attn_norm(x), offset))
        x = x.add(self.mlp(self.mlp_norm(x)))
        return x

    def parameters(self):
        return (self.attn_norm.parameters() + self.attn.parameters()
                + self.mlp_norm.parameters() + self.mlp.parameters())


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.embed = TiedEmbedding(cfg)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model)

    def __call__(self, ids, offset=0):
        x = self.embed.embed(ids)
        for b in self.blocks:
            x = b(x, offset)
        x = self.final_norm(x)
        return self.embed.project(x)

    def parameters(self):
        ps = self.embed.parameters()
        for b in self.blocks:
            ps = ps + b.parameters()
        ps = ps + self.final_norm.parameters()
        return ps

    def n_params(self):
        return int(sum(p.data.size for p in self.parameters()))
