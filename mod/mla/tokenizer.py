import json
import re
from collections import Counter
from pathlib import Path

SPECIALS = ["<pad>", "<bos>", "<eos>", "<|user|>", "<|assistant|>"]
N_SPECIAL = len(SPECIALS)
BYTE_OFFSET = N_SPECIAL
MERGE_OFFSET = N_SPECIAL + 256

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+""")
SPECIAL_PAT = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


class Tokenizer:
    def __init__(self):
        self.merges = []
        self.special_to_id = {s: i for i, s in enumerate(SPECIALS)}
        self.id_to_special = {i: s for i, s in enumerate(SPECIALS)}
        self.pair_rank = {}
        self.decode_pair = {}

    @property
    def vocab_size(self):
        return MERGE_OFFSET + len(self.merges)

    def _build_maps(self):
        self.pair_rank = {tuple(p): r for r, p in enumerate(self.merges)}
        self.decode_pair = {MERGE_OFFSET + r: tuple(p) for r, p in enumerate(self.merges)}

    @staticmethod
    def _merge_word(word, pair, new_id):
        a, b = pair
        out = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def _corpus_words(self, text):
        counts = Counter()
        for seg in SPECIAL_PAT.split(text):
            if not seg or seg in self.special_to_id:
                continue
            for chunk in PAT.findall(seg):
                word = tuple(BYTE_OFFSET + b for b in chunk.encode("utf-8"))
                counts[word] += 1
        return counts

    def train(self, text, vocab_size, verbose=True):
        words = dict(self._corpus_words(text))
        n_merges = vocab_size - MERGE_OFFSET
        self.merges = []
        for step in range(n_merges):
            pairs = Counter()
            for word, freq in words.items():
                for a, b in zip(word, word[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], p))
            new_id = MERGE_OFFSET + len(self.merges)
            self.merges.append([best[0], best[1]])
            words = {self._merge_word(w, best, new_id): c for w, c in words.items()}
            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{n_merges} pair={best} count={pairs[best]}")
        self._build_maps()

    def _encode_chunk(self, bts):
        word = [BYTE_OFFSET + b for b in bts]
        while len(word) >= 2:
            best = None
            best_rank = None
            for a, b in zip(word, word[1:]):
                r = self.pair_rank.get((a, b))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best = (a, b)
            if best is None:
                break
            word = list(self._merge_word(tuple(word), best, MERGE_OFFSET + best_rank))
        return word

    def encode(self, text):
        ids = []
        for seg in SPECIAL_PAT.split(text):
            if not seg:
                continue
            if seg in self.special_to_id:
                ids.append(self.special_to_id[seg])
                continue
            for chunk in PAT.findall(seg):
                ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def _expand(self, i):
        if i in self.decode_pair:
            a, b = self.decode_pair[i]
            return self._expand(a) + self._expand(b)
        return [i - BYTE_OFFSET]

    def decode(self, ids):
        out = []
        buf = []
        for i in ids:
            if i in self.id_to_special:
                if buf:
                    out.append(bytes(buf).decode("utf-8", errors="replace"))
                    buf = []
                out.append(self.id_to_special[i])
            else:
                buf.extend(self._expand(i))
        if buf:
            out.append(bytes(buf).decode("utf-8", errors="replace"))
        return "".join(out)

    def save(self, path):
        Path(path).write_text(
            json.dumps({"specials": SPECIALS, "merges": self.merges}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.merges = [list(m) for m in data["merges"]]
        tok._build_maps()
        return tok
