from __future__ import annotations

from zyn.backend import xp as np
from zyn.backend import fdtype
from zyn.tensor import Tensor


class Embedding:
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        std: float = 0.02,
        padding_idx: int | None = None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        w = np.random.randn(vocab_size, d_model).astype(fdtype) * std
        if padding_idx is not None:
            w[padding_idx] = 0.0
        self.weight = Tensor(w)

    def zero_padding_grad(self) -> None:
        if self.padding_idx is not None:
            self.weight.grad[self.padding_idx] = 0.0

    def tie_to(self, head) -> None:
        head.weight = self.weight

    def __call__(self, indices: np.ndarray) -> Tensor:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 2:
            raise ValueError(f"expected (B, T) indices, got shape {indices.shape}")
        if indices.max(initial=0) >= self.vocab_size or indices.min(initial=0) < 0:
            raise ValueError(
                f"index out of range [0, {self.vocab_size}): "
                f"min={indices.min()} max={indices.max()}"
            )
        return self.weight.gather(indices, dim=0)

    def parameters(self) -> list[Tensor]:
        return [self.weight]
