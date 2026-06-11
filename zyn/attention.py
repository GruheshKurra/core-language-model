from __future__ import annotations

import numpy as np

from zyn.tensor import Tensor


class SelfAttention:

    def __init__(self, d_model: int, std: float = 0.02):
        self.d_model = d_model
        self.scale = 1.0 / np.sqrt(d_model)
        self.W_q = Tensor(np.random.randn(d_model, d_model).astype(np.float64) * std)
        self.W_k = Tensor(np.random.randn(d_model, d_model).astype(np.float64) * std)
        self.W_v = Tensor(np.random.randn(d_model, d_model).astype(np.float64) * std)
        self.W_o = Tensor(np.random.randn(d_model, d_model).astype(np.float64) * std)

    def __call__(self, x: Tensor) -> Tensor:
        if x.data.ndim != 3:
            raise ValueError(f"expected (B, T, d_model), got shape {x.shape}")
        T = x.shape[1]

        q = x.matmul(self.W_q)
        k = x.matmul(self.W_k)
        v = x.matmul(self.W_v)

        scores = q.matmul(k.transpose()).mul(self.scale)

        causal = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = scores.masked_fill(causal, -1e9)

        attn = scores.softmax(axis=-1)
        out = attn.matmul(v)
        return out.matmul(self.W_o)

    def parameters(self) -> list[Tensor]:
        return [self.W_q, self.W_k, self.W_v, self.W_o]


class MultiHeadAttention:

    def __init__(self, d_model: int, n_head: int, std: float = 0.02):
        if d_model % n_head != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_head {n_head}")
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = 1.0 / np.sqrt(self.d_head)
        self.W_q = Tensor(np.random.randn(d_model, d_model).astype(np.float64) * std)
        self.W_k = Tensor(np.random.randn(d_model, d_model).astype(np.float64) * std)
        self.W_v = Tensor(np.random.randn(d_model, d_model).astype(np.float64) * std)
        self.W_o = Tensor(np.random.randn(d_model, d_model).astype(np.float64) * std)

    def _split(self, t: Tensor, B: int, T: int) -> Tensor:
        return t.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)

    def __call__(self, x: Tensor) -> Tensor:
        if x.data.ndim != 3:
            raise ValueError(f"expected (B, T, d_model), got shape {x.shape}")
        B, T, _ = x.shape

        q = self._split(x.matmul(self.W_q), B, T)
        k = self._split(x.matmul(self.W_k), B, T)
        v = self._split(x.matmul(self.W_v), B, T)

        scores = q.matmul(k.transpose()).mul(self.scale)

        causal = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = scores.masked_fill(causal, -1e9)

        attn = scores.softmax(axis=-1)
        ctx = attn.matmul(v).transpose(1, 2).reshape(B, T, self.d_model)
        return ctx.matmul(self.W_o)

    def parameters(self) -> list[Tensor]:
        return [self.W_q, self.W_k, self.W_v, self.W_o]
