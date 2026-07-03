#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zyn.batching import Batcher, load_tokens
from zyn.bpe import BPETokenizer
from zyn.checkpoint import load_checkpoint, save_checkpoint
from zyn.eval import evaluate
from zyn.generate import generate
from zyn.gpt import GPT, GPTConfig
from zyn.optim import AdamW, clip_grad_norm
from zyn.loss import cross_entropy
from zyn.schedule import cosine_lr

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def _val_batches(batcher: Batcher, n: int) -> list:
    return [batcher.next_batch() for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    cfg = json.loads(_resolve(args.config).read_text(encoding="utf-8"))

    tokenizer_path = _resolve(cfg["tokenizer"])
    tok = BPETokenizer.load(tokenizer_path)
    vocab_size = int(cfg.get("vocab_size") or tok.vocab_size)

    mmap = bool(cfg.get("mmap", True))
    train_tokens = load_tokens(_resolve(cfg["train_tokens"]), mmap=mmap)
    val_tokens = load_tokens(_resolve(cfg["val_tokens"]), mmap=mmap)

    context = int(cfg["context"])
    batch_size = int(cfg["batch_size"])
    train_batcher = Batcher(train_tokens, batch_size, context, seed=cfg.get("seed", 42))
    val_batcher = Batcher(val_tokens, batch_size, context, seed=cfg.get("seed", 42) + 1)

    steps = int(cfg["steps"])
    lr_max = float(cfg["lr"])
    warmup = int(cfg["warmup"])
    weight_decay = float(cfg.get("weight_decay", 0.1))
    max_norm = float(cfg.get("max_norm", 1.0))
    eval_every = int(cfg.get("eval_every", 200))
    eval_batches = int(cfg.get("eval_batches", 20))
    log_every = int(cfg.get("log_every", 20))
    ckpt_path = _resolve(cfg.get("checkpoint", "checkpoints/pretrain.npz"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        model, optimizer, start_step = load_checkpoint(_resolve(args.resume))
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        model = GPT(
            GPTConfig(
                vocab_size=vocab_size,
                d_model=int(cfg["d_model"]),
                n_head=int(cfg["n_head"]),
                n_layer=int(cfg["n_layer"]),
                d_ff=cfg.get("d_ff"),
                max_seq=int(cfg["max_seq"]),
                std=float(cfg.get("std", 0.02)),
            )
        )
        optimizer = AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)
        start_step = 0

    print(f"model params: {model.num_params():,}")

    for i in range(steps):
        step = start_step + i
        optimizer.lr = cosine_lr(step, lr_max, warmup, start_step + steps)
        x, y = train_batcher.next_batch()
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        model.tok_emb.zero_padding_grad()
        grad_norm = clip_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        loss_val = float(loss.data)
        del logits, loss
        gc.collect()

        if log_every and step % log_every == 0:
            print(f"step {step:6d} | loss {loss_val:.4f} | gnorm {grad_norm:.3f} | lr {optimizer.lr:.2e}")

        if eval_every and step > 0 and step % eval_every == 0:
            metrics = evaluate(model, _val_batches(val_batcher, eval_batches))
            print(f"  eval step {step}: loss {metrics['loss']:.4f} ppl {metrics['perplexity']:.2f} acc {metrics['accuracy']:.3f}")
            save_checkpoint(ckpt_path, model, optimizer, step=step)

    save_checkpoint(ckpt_path, model, optimizer, step=start_step + steps)
    print(f"saved {ckpt_path.relative_to(ROOT)}")

    prompt = cfg.get("sample_prompt", "def ")
    ids = tok.encode(prompt, add_bos=True)
    out = generate(model, ids, max_new_tokens=64, temperature=0.8, top_k=40, eos_id=tok.eos_id)
    print("sample:", tok.decode(out[0].tolist(), skip_specials=True)[:300])


if __name__ == "__main__":
    main()
