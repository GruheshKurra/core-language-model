from .backend import xp
from .tensor import Tensor


def cross_entropy(logits, targets, ignore_index=-1):
    data = logits.data
    V = data.shape[-1]
    flat = data.reshape(-1, V)
    tgt = xp.asarray(targets).reshape(-1).astype(xp.int64)
    mask = (tgt != ignore_index)
    tgt_safe = xp.where(mask, tgt, 0)
    n = max(1, int(mask.sum()))
    rows = xp.arange(flat.shape[0])

    z = flat - xp.max(flat, axis=-1, keepdims=True)
    e = xp.exp(z)
    p = e / xp.sum(e, axis=-1, keepdims=True)
    logp = xp.log(p[rows, tgt_safe])
    loss = -xp.sum(logp * mask) / n
    out = Tensor(xp.asarray(loss), (logits,), "cross_entropy")

    def _backward():
        g = p.copy()
        g[rows, tgt_safe] = g[rows, tgt_safe] - 1.0
        g = g * mask.reshape(-1, 1)
        g = g / n
        g = g * out.grad
        logits.grad = logits.grad + g.reshape(data.shape)

    out._backward = _backward
    return out
