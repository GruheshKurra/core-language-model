#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("ZYN_DTYPE", "float32")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zyn.bpe import BPETokenizer
from zyn.checkpoint import load_checkpoint
from zyn.generate import generate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "pretrain-a6000.npz"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "processed" / "tokenizer.json"))
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    model, _, step = load_checkpoint(args.checkpoint)
    print(f"loaded step {step} | params {model.num_params():,} | vocab {tok.vocab_size}")
    print("enter a prompt (empty line or Ctrl-D to quit)\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            break
        ids = tok.encode(prompt, add_bos=True)
        out = generate(
            model,
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=tok.eos_id,
        )
        print(tok.decode(out[0].tolist(), skip_specials=True))
        print()


if __name__ == "__main__":
    main()
