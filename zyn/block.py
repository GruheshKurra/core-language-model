from __future__ import annotations

from zyn.attention import MultiHeadAttention
from zyn.layernorm import LayerNorm
from zyn.mlp import MLP
from zyn.tensor import Tensor


class TransformerBlock:

    def __init__(self, d_model: int, n_head: int, d_ff: int | None = None, std: float = 0.02):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, std=std)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff=d_ff, std=std)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.add(self.attn(self.ln1(x)))
        x = x.add(self.mlp(self.ln2(x)))
        return x

    def parameters(self) -> list[Tensor]:
        return [
            *self.ln1.parameters(),
            *self.attn.parameters(),
            *self.ln2.parameters(),
            *self.mlp.parameters(),
        ]
