from __future__ import annotations

import numpy as np

from zyn.tensor import Tensor


class PositionalEmbedding:

    def __init__(self, max_seq: int, d_model: int, std: float = 0.02):
        self.max_seq = max_seq
        self.d_model = d_model
        w = np.random.randn(max_seq, d_model).astype(np.float64) * std
        self.weight = Tensor(w)

    def __call__(self, seq_len: int) -> Tensor:
        if seq_len > self.max_seq:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq {self.max_seq}"
            )
        positions = np.arange(seq_len, dtype=np.int64)
        return self.weight.gather(positions, dim=0)

    def parameters(self) -> list[Tensor]:
        return [self.weight]
