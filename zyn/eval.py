from __future__ import annotations

from typing import Iterable

import numpy as np


def evaluate(model, batches: Iterable[tuple[np.ndarray, np.ndarray]], ignore_index: int | None = None) -> dict:
    total_nll = 0.0
    total_correct = 0
    total_tokens = 0

    for x, y in batches:
        logits = model(np.asarray(x, dtype=np.int64)).data
        V = logits.shape[-1]
        flat = logits.reshape(-1, V)
        t = np.asarray(y, dtype=np.int64).reshape(-1)

        if ignore_index is not None:
            keep = t != ignore_index
        else:
            keep = np.ones_like(t, dtype=bool)
        safe_t = np.where(keep, t, 0)

        m = flat.max(axis=-1, keepdims=True)
        log_sum_exp = m[:, 0] + np.log(np.exp(flat - m).sum(axis=-1))
        rows = np.arange(flat.shape[0])
        nll = log_sum_exp - flat[rows, safe_t]
        preds = flat.argmax(axis=-1)

        total_nll += float((nll * keep).sum())
        total_correct += int(((preds == t) & keep).sum())
        total_tokens += int(keep.sum())

    if total_tokens == 0:
        raise ValueError("no tokens to evaluate")

    loss = total_nll / total_tokens
    return {
        "loss": loss,
        "perplexity": float(np.exp(loss)),
        "accuracy": total_correct / total_tokens,
        "tokens": total_tokens,
    }
