#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zyn.checkpoint import load_checkpoint, save_checkpoint
from zyn.instruct import IGNORE_INDEX, MaskedBatcher
from zyn.optim import AdamW
from zyn.train import train

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "tools.npz"))
    ap.add_argument("--tokens", default=str(PROCESSED / "tools" / "tool-train.npy"))
    ap.add_argument("--mask", default=str(PROCESSED / "tools" / "tool-train-mask.npy"))
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    tokens = np.load(args.tokens)
    mask = np.load(args.mask)
    print(f"tool stream: {len(tokens):,} tokens, {int(mask.sum()):,} trainable targets")

    model, _, _ = load_checkpoint(args.checkpoint)
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
