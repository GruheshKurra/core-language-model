#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mla.tokenizer import Tokenizer

TRAIN = Path("data/splits/train.txt")
OUT = Path("data/tokenizer/tokenizer.json")
VOCAB_SIZE = 4096


def main():
    text = TRAIN.read_text(encoding="utf-8")
    tok = Tokenizer()
    print(f"training byte-BPE: target vocab={VOCAB_SIZE}")
    tok.train(text, VOCAB_SIZE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tok.save(OUT)
    print(f"vocab_size={tok.vocab_size} merges={len(tok.merges)}")
    print(f"wrote {OUT}")

    sample = "<bos><|user|>hey how are you?<eos><|assistant|>i'm good, you?<eos>"
    ids = tok.encode(sample)
    back = tok.decode(ids)
    print(f"sample_ids_len={len(ids)}")
    print(f"roundtrip_ok={back == sample}")
    print(f"decoded={back!r}")


if __name__ == "__main__":
    main()
