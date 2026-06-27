from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np


def load_tokens(path: str | Path, mmap: bool = True) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None)


class Batcher:
    def __init__(
        self,
        tokens: np.ndarray,
        batch_size: int,
        context_len: int,
        seed: int = 42,
    ):
        self.tokens = tokens
        self.B = batch_size
        self.T = context_len
        self.rng = np.random.default_rng(seed)
        if len(self.tokens) < self.T + 1:
            raise ValueError(f"need >= {self.T + 1} tokens, have {len(self.tokens)}")

    @property
    def max_start(self) -> int:
        return len(self.tokens) - self.T - 1

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        starts = self.rng.integers(0, self.max_start + 1, size=self.B)
        idx = starts[:, None] + np.arange(self.T)[None, :]
        x = np.asarray(self.tokens[idx], dtype=np.int64)
        y = np.asarray(self.tokens[idx + 1], dtype=np.int64)
        return x, y

    def epoch(self, shuffle: bool = True) -> "Iterator[tuple[np.ndarray, np.ndarray]]":
        n_blocks = len(self.tokens) // (self.T + 1)
        order = np.arange(n_blocks)
        if shuffle:
            self.rng.shuffle(order)
        for i in range(0, n_blocks - self.B + 1, self.B):
            starts = order[i : i + self.B] * (self.T + 1)
            idx = starts[:, None] + np.arange(self.T)[None, :]
            x = np.asarray(self.tokens[idx], dtype=np.int64)
            y = np.asarray(self.tokens[idx + 1], dtype=np.int64)
            yield x, y
