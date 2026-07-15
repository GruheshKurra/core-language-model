#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mla.tokenizer import Tokenizer

TOK = Path("data/tokenizer/tokenizer.json")
SPLITS = {
    "train": Path("data/splits/train.txt"),
    "val": Path("data/splits/val.txt"),
}
OUT_DIR = Path("data/tokenized")


def main():
    tok = Tokenizer.load(TOK)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in SPLITS.items():
        stream = []
        n_dialogues = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            stream.extend(tok.encode(line))
            n_dialogues += 1
        arr = np.array(stream, dtype=np.uint16)
        out = OUT_DIR / f"{name}.npy"
        np.save(out, arr)
        print(f"{name}: dialogues={n_dialogues} tokens={len(arr)} -> {out}")


if __name__ == "__main__":
    main()
