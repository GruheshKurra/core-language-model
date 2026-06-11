from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SPECIALS = ["<pad>", "<bos>", "<eos>", "<tool_call>", "</tool_call>", "<tool_result>"]

PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


def _pretokens(text: str) -> list[str]:
    return PAT.findall(text)


def _word_to_ids(word: str, byte_base: int) -> list[int]:
    return [byte_base + b for b in word.encode("utf-8")]


def _merge_seq(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    a, b = pair
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, specials: list[str] | None = None):
        self.specials = list(specials) if specials else list(SPECIALS)
        self.byte_base = len(self.specials)
        self.special_to_id = {s: i for i, s in enumerate(self.specials)}
        self.merges: dict[tuple[int, int], int] = {}
        self.ranks: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self._build_base_vocab()

    def _build_base_vocab(self) -> None:
        self.vocab = {}
        for i, s in enumerate(self.specials):
            self.vocab[i] = s.encode("utf-8")
        for b in range(256):
            self.vocab[self.byte_base + b] = bytes([b])

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<eos>"]

    def train(
        self,
        text: str,
        vocab_size: int,
        min_freq: int = 2,
        verbose: bool = False,
    ) -> "BPETokenizer":
        floor = self.byte_base + 256
        if vocab_size < floor:
            raise ValueError(f"vocab_size must be >= {floor}")
        word_freq = Counter(_pretokens(text))
        word_ids = {w: _word_to_ids(w, self.byte_base) for w in word_freq}
        next_id = floor
        self.merges = {}
        while next_id < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for w, freq in word_freq.items():
                ids = word_ids[w]
                for a, b in zip(ids, ids[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            best, cnt = pair_counts.most_common(1)[0]
            if cnt < min_freq:
                break
            self.merges[best] = next_id
            self.vocab[next_id] = self.vocab[best[0]] + self.vocab[best[1]]
            for w in word_ids:
                word_ids[w] = _merge_seq(word_ids[w], best, next_id)
            if verbose:
                print(next_id, best, cnt)
            next_id += 1
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        return self

    def _encode_word(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            best_rank: int | None = None
            best_pair: tuple[int, int] | None = None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = pair
            if best_pair is None:
                break
            ids = _merge_seq(ids, best_pair, self.merges[best_pair])
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        out: list[int] = []
        if add_bos:
            out.append(self.bos_id)
        for w in _pretokens(text):
            out.extend(self._encode_word(_word_to_ids(w, self.byte_base)))
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, ids: list[int], skip_specials: bool = False) -> str:
        special_ids = set(range(len(self.specials)))
        parts: list[bytes] = []
        for i in ids:
            if skip_specials and i in special_ids:
                continue
            parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        data = {
            "specials": self.specials,
            "merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        t = cls(specials=data["specials"])
        for a, b, nid in data["merges"]:
            t.merges[(a, b)] = nid
            t.vocab[nid] = t.vocab[a] + t.vocab[b]
        t.ranks = {pair: i for i, pair in enumerate(t.merges)}
        return t
