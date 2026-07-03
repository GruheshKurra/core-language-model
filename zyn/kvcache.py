from __future__ import annotations

from zyn.backend import xp as np

from zyn.functional import gelu as _gelu
from zyn.functional import layernorm as _layernorm
from zyn.functional import softmax as _softmax
from zyn.generate import _sample_next
from zyn.gpt import GPT


class CachedGPT:
    def __init__(self, model: GPT):
        cfg = model.config
        self.config = cfg
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.scale = 1.0 / np.sqrt(self.d_head)
        self.tok_emb = np.array(model.tok_emb.weight.data, copy=True)
        self.pos_emb = np.array(model.pos_emb.weight.data, copy=True)
        self.lnf = (np.array(model.ln_f.gamma.data, copy=True), np.array(model.ln_f.beta.data, copy=True), model.ln_f.eps)
        self.blocks = []
        for b in model.blocks:
            self.blocks.append(
                {
                    "ln1": (np.array(b.ln1.gamma.data, copy=True), np.array(b.ln1.beta.data, copy=True), b.ln1.eps),
                    "Wq": np.array(b.attn.W_q.data, copy=True),
                    "Wk": np.array(b.attn.W_k.data, copy=True),
                    "Wv": np.array(b.attn.W_v.data, copy=True),
                    "Wo": np.array(b.attn.W_o.data, copy=True),
                    "ln2": (np.array(b.ln2.gamma.data, copy=True), np.array(b.ln2.beta.data, copy=True), b.ln2.eps),
                    "W1": np.array(b.mlp.W1.data, copy=True),
                    "b1": np.array(b.mlp.b1.data, copy=True),
                    "W2": np.array(b.mlp.W2.data, copy=True),
                    "b2": np.array(b.mlp.b2.data, copy=True),
                }
            )
        self.reset()

    def reset(self) -> None:
        self.pos = 0
        self.cache = [{"k": None, "v": None} for _ in self.blocks]

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_head, self.d_head).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        B, H, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dh)

    def feed(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]
        B, T = idx.shape
        start = self.pos
        if start + T > self.config.max_seq:
            raise ValueError(f"sequence {start + T} exceeds max_seq {self.config.max_seq}")

        x = self.tok_emb[idx] + self.pos_emb[start : start + T][None, :, :]
        qpos = start + np.arange(T)
        for i, blk in enumerate(self.blocks):
            g1, b1n, eps1 = blk["ln1"]
            h = _layernorm(x, g1, b1n, eps1)
            q = self._split_heads(h @ blk["Wq"])
            k = self._split_heads(h @ blk["Wk"])
            v = self._split_heads(h @ blk["Wv"])
            ck = self.cache[i]["k"]
            cv = self.cache[i]["v"]
            if ck is not None:
                k = np.concatenate([ck, k], axis=2)
                v = np.concatenate([cv, v], axis=2)
            self.cache[i]["k"] = k
            self.cache[i]["v"] = v
            Lk = k.shape[2]
            scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            kpos = np.arange(Lk)
            forbid = kpos[None, :] > qpos[:, None]
            scores = np.where(forbid[None, None, :, :], -1e9, scores)
            attn = _softmax(scores)
            ctx = self._merge_heads(attn @ v)
            x = x + ctx @ blk["Wo"]
            g2, b2n, eps2 = blk["ln2"]
            h2 = _layernorm(x, g2, b2n, eps2)
            mlp = _gelu(h2 @ blk["W1"] + blk["b1"]) @ blk["W2"] + blk["b2"]
            x = x + mlp

        g, beta, eps = self.lnf
        x = _layernorm(x, g, beta, eps)
        logits = x @ self.tok_emb.T
        self.pos += T
        return logits

    def prefill(self, idx) -> np.ndarray:
        return self.feed(idx)[:, -1, :]

    def decode_step(self, token) -> np.ndarray:
        token = np.asarray(token, dtype=np.int64).reshape(-1, 1)
        return self.feed(token)[:, -1, :]


def generate_cached(
    model,
    idx: np.ndarray,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    cached = model if isinstance(model, CachedGPT) else CachedGPT(model)
    cached.reset()
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx[None, :]
    if rng is None:
        rng = np.random.default_rng()

    logits = cached.prefill(idx)
    out = idx
    finished = np.zeros(idx.shape[0], dtype=bool)
    for step in range(max_new_tokens):
        next_ids = _sample_next(logits, temperature, top_k, top_p, rng)
        if eos_id is not None:
            next_ids = np.where(finished, eos_id, next_ids)
            finished = finished | (next_ids == eos_id)
        out = np.concatenate([out, next_ids[:, None]], axis=1)
        if eos_id is not None and bool(finished.all()):
            break
        if step == max_new_tokens - 1:
            break
        logits = cached.decode_step(next_ids)
    return out
