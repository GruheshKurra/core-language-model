from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zyn.bpe import BPETokenizer
from zyn.chat import render_messages

IGNORE_INDEX = -100


def record_to_messages(record: dict, system: str | None = None) -> list[dict]:
    instruction = record.get("instruction", "").strip()
    user_input = record.get("input", "").strip()
    output = record.get("output", "").strip()
    if user_input:
        user = f"{instruction}\n\n{user_input}"
    else:
        user = instruction
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": output})
    return messages


def build_instruct_stream(
    records: list[dict],
    tok: BPETokenizer,
    system: str | None = None,
    max_examples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tokens: list[int] = []
    mask: list[int] = []
    used = 0
    for record in records:
        if max_examples is not None and used >= max_examples:
            break
        if not record.get("output", "").strip():
            continue
        messages = record_to_messages(record, system=system)
        ids, m = render_messages(messages, tok, add_bos=True)
        tokens.extend(ids)
        mask.extend(m)
        used += 1
    dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
    return np.asarray(tokens, dtype=dtype), np.asarray(mask, dtype=np.uint8)


def load_instruct_records(jsonl_path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


class MaskedBatcher:
    def __init__(
        self,
        tokens: np.ndarray,
        mask: np.ndarray,
        batch_size: int,
        context_len: int,
        ignore_index: int = IGNORE_INDEX,
        seed: int = 42,
    ):
        self.tokens = np.asarray(tokens)
        self.mask = np.asarray(mask)
        if self.tokens.shape != self.mask.shape:
            raise ValueError("tokens and mask must have the same shape")
        self.B = batch_size
        self.T = context_len
        self.ignore_index = ignore_index
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
        keep = np.asarray(self.mask[idx + 1], dtype=bool)
        y = np.where(keep, y, self.ignore_index)
        return x, y
