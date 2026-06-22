from __future__ import annotations

import numpy as np


def cosine_lr(
    step: int,
    lr_max: float,
    warmup_steps: int,
    max_steps: int,
    lr_min: float = 0.0,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    if step >= max_steps:
        return lr_min
    span = max(max_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / span
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * progress))
