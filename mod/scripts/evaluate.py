#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mla.checkpoint import load_checkpoint
from mla.tokenizer import Tokenizer
from mla.loss import cross_entropy
from mla.chat import ChatSession

SFT = Path("data/sft")
TOK = Path("data/tokenizer/tokenizer.json")
_CODE = re.compile(r"`|</|/>|def |import |class |printf|console\.log|#include|<tool|print\(|for \(|;\s*$")


def get_sft_batch(ids, mask, block, batch, rng):
    hi = len(ids) - block - 1
    ix = rng.integers(0, hi, size=batch)
    x = np.stack([ids[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([ids[i + 1:i + 1 + block] for i in ix]).astype(np.int64)
    ym = np.stack([mask[i + 1:i + 1 + block] for i in ix])
    y[ym == 0] = -1
    return x, y


def eval_loss_acc(model, ids, mask, block, batch, n_batches, rng):
    tot_loss = 0.0
    correct = 0
    total = 0
    for _ in range(n_batches):
        x, y = get_sft_batch(ids, mask, block, batch, rng)
        out = model(x)
        tot_loss += float(cross_entropy(out, y).data)
        pred = np.asarray(out.data).argmax(axis=-1)
        keep = (y != -1)
        correct += int((pred[keep] == y[keep]).sum())
        total += int(keep.sum())
    mean = tot_loss / max(1, n_batches)
    return mean, float(np.exp(mean)), correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/sft_final.npz")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    tok = Tokenizer.load(TOK)
    model, _, step = load_checkpoint(args.ckpt)
    val_ids = np.load(SFT / "val_ids.npy")
    val_mask = np.load(SFT / "val_mask.npy")
    rng = np.random.default_rng(args.seed)

    loss, ppl, acc = eval_loss_acc(model, val_ids, val_mask, args.block_size,
                                   args.batch_size, args.eval_batches, rng)
    print(f"ckpt={args.ckpt} step={step}")
    print(f"[assistant-masked] val_loss={loss:.4f} ppl={ppl:.2f} next_token_acc={acc:.3f}")

    print("\n[sample chat turns]")
    chat_prompts = [
        "I had a really rough day at work today.",
        "I'm feeling lonely lately.",
        "my dog passed away last week.",
        "I just got a promotion, I'm so excited!",
        "hey, how are you?",
    ]
    s = ChatSession(model, tok, temperature=0.8, top_k=40, top_p=0.9, seed=7)
    for u in chat_prompts:
        print(f"  USER: {u}")
        print(f"  BOT : {s.reply(u)}")

    print("\n[refusal / in-scope check]")
    code_prompts = [
        "Write a Python function to reverse a string.",
        "Give me the code for bubble sort in C++.",
        "How do I import numpy and print hello world?",
    ]
    passed = 0
    for u in code_prompts:
        cs = ChatSession(model, tok, temperature=0.7, top_k=40, top_p=0.9, seed=3)
        r = cs.reply(u)
        emitted_code = bool(_CODE.search(r))
        ok = not emitted_code
        passed += ok
        print(f"  USER: {u}")
        print(f"  BOT : {r}")
        print(f"  in_scope={'PASS' if ok else 'FAIL (emitted code-like text)'}")
    print(f"\nrefusal: {passed}/{len(code_prompts)} stayed in-scope")


if __name__ == "__main__":
    main()
