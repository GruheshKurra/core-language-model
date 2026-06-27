#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zyn.bpe import BPETokenizer
from zyn.checkpoint import load_checkpoint, save_checkpoint
from zyn.instruct import IGNORE_INDEX, MaskedBatcher, build_instruct_stream, load_instruct_records
from zyn.optim import AdamW
from zyn.train import train

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "instruct.npz"))
    ap.add_argument("--tokenizer", default=str(PROCESSED / "tokenizer.json"))
    ap.add_argument("--source", default="python-codes-25k")
    ap.add_argument("--system", default="You are a helpful Python coding assistant.")
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    records = load_instruct_records(PROCESSED / args.source / "train.jsonl")
    max_examples = None if args.max_examples == 0 else args.max_examples
    tokens, mask = build_instruct_stream(records, tok, system=args.system, max_examples=max_examples)
    print(f"instruct stream: {len(tokens):,} tokens, {int(mask.sum()):,} trainable targets")

    model, _, _ = load_checkpoint(args.checkpoint)
    if model.config.vocab_size != tok.vocab_size:
        raise ValueError(
            f"vocab mismatch: model {model.config.vocab_size} vs tokenizer {tok.vocab_size}"
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batcher = MaskedBatcher(tokens, mask, args.batch_size, args.context, ignore_index=IGNORE_INDEX)

    train(
        model,
        optimizer,
        batcher.next_batch,
        steps=args.steps,
        lr_max=args.lr,
        warmup_steps=args.warmup,
        max_steps=args.steps,
        ignore_index=IGNORE_INDEX,
        log_every=args.log_every,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out, model, optimizer, step=args.steps)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
