from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zyn.block import TransformerBlock
from zyn.embedding import Embedding
from zyn.layernorm import LayerNorm
from zyn.positional import PositionalEmbedding
from zyn.tensor import Tensor


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    d_ff: int | None = None
    max_seq: int = 256
    std: float = 0.02


class GPT:

    def __init__(self, config: GPTConfig):
        self.config = config
        self.tok_emb = Embedding(config.vocab_size, config.d_model, std=config.std)
        self.pos_emb = PositionalEmbedding(config.max_seq, config.d_model, std=config.std)
        self.blocks = [
            TransformerBlock(config.d_model, config.n_head, d_ff=config.d_ff, std=config.std)
            for _ in range(config.n_layer)
        ]
        self.ln_f = LayerNorm(config.d_model)

    def __call__(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim != 2:
            raise ValueError(f"expected (B, T) token ids, got shape {idx.shape}")
        T = idx.shape[1]
        if T > self.config.max_seq:
            raise ValueError(f"seq_len {T} exceeds max_seq {self.config.max_seq}")

        x = self.tok_emb(idx).add(self.pos_emb(T))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x.matmul(self.tok_emb.weight.transpose())

    def parameters(self) -> list[Tensor]:
        params = [*self.tok_emb.parameters(), *self.pos_emb.parameters()]
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.ln_f.parameters())
        return params

    def num_params(self) -> int:
        return sum(int(p.data.size) for p in self.parameters())
