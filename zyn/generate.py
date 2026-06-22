from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def _top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    k = min(k, logits.shape[-1])
    kth = np.sort(logits, axis=-1)[:, -k][:, None]
    return np.where(logits < kth, -np.inf, logits)


def _top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    order = np.argsort(logits, axis=-1)[:, ::-1]
    sorted_logits = np.take_along_axis(logits, order, axis=-1)
    cum = np.cumsum(_softmax(sorted_logits), axis=-1)
    remove = cum > p
    remove[:, 1:] = remove[:, :-1].copy()
    remove[:, 0] = False
    sorted_logits = np.where(remove, -np.inf, sorted_logits)
    out = np.empty_like(logits)
    rows = np.arange(logits.shape[0])[:, None]
    out[rows, order] = sorted_logits
    return out


def _sample_next(
    logits: np.ndarray,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if temperature == 0.0:
        return logits.argmax(axis=-1).astype(np.int64)
    logits = logits / temperature
    if top_k is not None:
        logits = _top_k_filter(logits, top_k)
    if top_p is not None:
        logits = _top_p_filter(logits, top_p)
    probs = _softmax(logits)
    out = np.empty(probs.shape[0], dtype=np.int64)
    for b in range(probs.shape[0]):
        out[b] = rng.choice(probs.shape[1], p=probs[b])
    return out


def generate(
    model,
    idx: np.ndarray,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx[None, :]
    if rng is None:
        rng = np.random.default_rng()
    max_seq = model.config.max_seq

    for _ in range(max_new_tokens):
        ctx = idx[:, -max_seq:]
        logits = model(ctx).data[:, -1, :]
        next_ids = _sample_next(logits, temperature, top_k, top_p, rng)
        idx = np.concatenate([idx, next_ids[:, None]], axis=1)
        if eos_id is not None and bool((next_ids == eos_id).all()):
            break

    return idx
