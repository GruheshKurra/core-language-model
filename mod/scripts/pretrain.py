#!/usr/bin/env python3
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mla.backend import xp, NAME, DTYPE
from mla.model import Config, Model
from mla.optim import AdamW, clip_grad_norm
from mla.data import load_ids, get_batch
from mla.loss import cross_entropy
from mla.schedule import lr_schedule
from mla.eval import eval_loss
from mla.checkpoint import save_checkpoint

TRAIN = Path("data/tokenized/train.npy")
VAL = Path("data/tokenized/val.npy")
CKPT_DIR = Path("checkpoints")


def build_config(block_size, tiny):
    if tiny:
        return Config(vocab_size=4096, d_model=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, head_dim=16, swiglu_hidden=128, seq_len=block_size)
    return Config(seq_len=block_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--peak-lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-norm", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    if not TRAIN.exists():
        sys.exit(f"missing {TRAIN} — run scripts/tokenize_corpus.py first")

    xp.random.seed(args.seed)
    train_ids = load_ids(TRAIN)
    val_ids = load_ids(VAL)
    cfg = build_config(args.block_size, args.tiny)
    model = Model(cfg)
    opt = AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)

    CKPT_DIR.mkdir(exist_ok=True)
    print(f"backend={NAME} dtype={DTYPE} params={model.n_params():,} "
          f"train_tokens={len(train_ids):,} val_tokens={len(val_ids):,}")
    print(f"steps={args.steps} batch={args.batch_size} block={args.block_size} "
          f"peak_lr={args.peak_lr} warmup={args.warmup}")

    t0 = time.time()
    for step in range(args.steps):
        opt.lr = lr_schedule(step, args.peak_lr, args.warmup, args.steps, args.min_lr)
        x, y = get_batch(train_ids, args.block_size, args.batch_size, rng)
        opt.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        clip_grad_norm(opt.params, args.max_norm)
        opt.step()
        train_loss = float(loss.data)
        del loss
        gc.collect()

        if (step + 1) % args.log_every == 0:
            it_s = (step + 1) / (time.time() - t0)
            print(f"step {step + 1}/{args.steps} lr={opt.lr:.2e} "
                  f"loss={train_loss:.4f} {it_s:.2f} it/s")
        if (step + 1) % args.eval_every == 0:
            vl, ppl = eval_loss(model, val_ids, args.block_size,
                                args.batch_size, args.eval_batches, val_rng)
            print(f"  [eval] step {step + 1} val_loss={vl:.4f} ppl={ppl:.2f}")
        if (step + 1) % args.ckpt_every == 0:
            save_checkpoint(str(CKPT_DIR / "pretrain.npz"), model, opt, step + 1)
            print(f"  [ckpt] saved -> {CKPT_DIR / 'pretrain.npz'} @ step {step + 1}")

    save_checkpoint(str(CKPT_DIR / "pretrain_final.npz"), model, opt, args.steps)
    print(f"done. final checkpoint -> {CKPT_DIR / 'pretrain_final.npz'}")


if __name__ == "__main__":
    main()
