from __future__ import annotations

import gc
from typing import Callable

import numpy as np

from zyn.loss import cross_entropy
from zyn.optim import AdamW, clip_grad_norm
from zyn.schedule import cosine_lr


def train_step(
    model,
    optimizer: AdamW,
    x: np.ndarray,
    y: np.ndarray,
    max_norm: float = 1.0,
    ignore_index: int | None = None,
) -> tuple[float, float]:
    optimizer.zero_grad()
    logits = model(x)
    loss = cross_entropy(logits, y, ignore_index=ignore_index)
    loss.backward()
    model.tok_emb.zero_padding_grad()
    grad_norm = clip_grad_norm(model.parameters(), max_norm)
    optimizer.step()
    return float(loss.data), grad_norm


def train(
    model,
    optimizer: AdamW,
    get_batch: Callable[[], tuple[np.ndarray, np.ndarray]],
    steps: int,
    lr_max: float,
    warmup_steps: int,
    max_steps: int,
    lr_min: float = 0.0,
    max_norm: float = 1.0,
    ignore_index: int | None = None,
    start_step: int = 0,
    log_every: int = 0,
) -> list[dict]:
    history = []
    for i in range(steps):
        step = start_step + i
        optimizer.lr = cosine_lr(step, lr_max, warmup_steps, max_steps, lr_min)
        x, y = get_batch()
        loss, grad_norm = train_step(model, optimizer, x, y, max_norm, ignore_index)
        gc.collect()
        record = {"step": step, "loss": loss, "grad_norm": grad_norm, "lr": optimizer.lr}
        history.append(record)
        if log_every and (step % log_every == 0 or i == steps - 1):
            print(f"step {step:5d} | loss {loss:.4f} | gnorm {grad_norm:.3f} | lr {optimizer.lr:.2e}")
    return history
