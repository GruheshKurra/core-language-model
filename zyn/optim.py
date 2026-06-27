from __future__ import annotations

from zyn.backend import xp as np
from zyn.tensor import Tensor


def clip_grad_norm(params: list[Tensor], max_norm: float, eps: float = 1e-6) -> float:
    total_sq = 0.0
    for p in params:
        total_sq += float((p.grad * p.grad).sum())
    total_norm = float(np.sqrt(total_sq))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in params:
            p.grad = p.grad * scale
    return total_norm


class AdamW:

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            if self.weight_decay != 0.0:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
