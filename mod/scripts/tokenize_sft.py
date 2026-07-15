#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mla.tokenizer import Tokenizer

TOK = Path("data/tokenizer/tokenizer.json")
SPLITS = {
    "train": Path("data/sft/train.txt"),
    "val": Path("data/sft/val.txt"),
}
OUT_DIR = Path("data/sft")


def build_mask(ids, bos, eos, user, assistant):
    mask = []
    role = None
    for t in ids:
        if t == bos:
            mask.append(0)
        elif t == user:
            role = "u"
            mask.append(0)
        elif t == assistant:
            role = "a"
            mask.append(0)
        else:
            mask.append(1 if role == "a" else 0)
    return mask


def main():
    tok = Tokenizer.load(TOK)
    bos = tok.special_to_id["<bos>"]
    eos = tok.special_to_id["<eos>"]
    user = tok.special_to_id["<|user|>"]
    assistant = tok.special_to_id["<|assistant|>"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in SPLITS.items():
        id_stream = []
        mask_stream = []
        n_dialogues = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            ids = tok.encode(line)
            mask = build_mask(ids, bos, eos, user, assistant)
            id_stream.extend(ids)
            mask_stream.extend(mask)
            n_dialogues += 1
        ids_arr = np.array(id_stream, dtype=np.uint16)
        mask_arr = np.array(mask_stream, dtype=np.uint8)
        np.save(OUT_DIR / f"{name}_ids.npy", ids_arr)
        np.save(OUT_DIR / f"{name}_mask.npy", mask_arr)
        frac = mask_arr.mean() if len(mask_arr) else 0.0
        print(f"{name}: dialogues={n_dialogues} tokens={len(ids_arr)} "
              f"assistant_frac={frac:.3f} -> {name}_ids.npy / {name}_mask.npy")


if __name__ == "__main__":
    main()
