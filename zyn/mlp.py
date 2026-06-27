from __future__ import annotations

from zyn.backend import xp as np
from zyn.backend import fdtype
from zyn.tensor import Tensor


class MLP:

    def __init__(self, d_model: int, d_ff: int | None = None, std: float = 0.02):
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.W1 = Tensor(np.random.randn(d_model, self.d_ff).astype(fdtype) * std)
        self.b1 = Tensor(np.zeros(self.d_ff, dtype=fdtype))
        self.W2 = Tensor(np.random.randn(self.d_ff, d_model).astype(fdtype) * std)
        self.b2 = Tensor(np.zeros(d_model, dtype=fdtype))

    def __call__(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {x.shape}")
        h = x.matmul(self.W1).add(self.b1).gelu()
        return h.matmul(self.W2).add(self.b2)

    def parameters(self) -> list[Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]
