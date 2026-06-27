from __future__ import annotations

from zyn.backend import xp as np
from zyn.tensor import Tensor


def cross_entropy(logits: Tensor, targets: np.ndarray, ignore_index: int | None = None) -> Tensor:
    z = logits.data
    if z.ndim < 2:
        raise ValueError(f"expected logits (..., V), got shape {z.shape}")
    V = z.shape[-1]
    flat = z.reshape(-1, V)
    t = np.asarray(targets, dtype=np.int64).reshape(-1)
    if t.shape[0] != flat.shape[0]:
        raise ValueError(f"targets {t.shape[0]} != logit rows {flat.shape[0]}")

    m = flat.max(axis=-1, keepdims=True)
    shifted = flat - m
    exp = np.exp(shifted)
    sum_exp = exp.sum(axis=-1, keepdims=True)
    log_sum_exp = m + np.log(sum_exp)
    probs = exp / sum_exp

    if ignore_index is not None:
        keep = t != ignore_index
    else:
        keep = np.ones_like(t, dtype=bool)
    safe_t = np.where(keep, t, 0)

    rows = np.arange(flat.shape[0])
    nll = log_sum_exp[:, 0] - flat[rows, safe_t]

    count = int(keep.sum())
    if count == 0:
        raise ValueError("all targets are ignore_index; no tokens to score")

    loss_val = float((nll * keep).sum() / count)
    out = Tensor(np.array(loss_val), (logits,))

    def _backward():
        grad = probs.copy()
        grad[rows, safe_t] -= 1.0
        grad[~keep] = 0.0
        grad /= count
        logits.grad += (grad * out.grad).reshape(z.shape)
    out._backward = _backward
    return out
