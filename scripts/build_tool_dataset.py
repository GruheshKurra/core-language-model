#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zyn.bpe import BPETokenizer
from zyn.tooldata import build_tool_stream, generate_conversations
from zyn.tools import Sandbox

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tools", default="calculator,list_dir")
    ap.add_argument("--tokenizer", default=str(PROCESSED / "tokenizer.json"))
    ap.add_argument("--out-dir", default=str(PROCESSED / "tools"))
    args = ap.parse_args()

    tools = tuple(t.strip() for t in args.tools.split(",") if t.strip())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sandbox = Sandbox(out_dir / "_sandbox", allow_write=True)

    conversations = generate_conversations(args.n, seed=args.seed, tools=tools, sandbox=sandbox)

    traces_path = out_dir / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as f:
        for messages in conversations:
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    tok_path = Path(args.tokenizer)
    if tok_path.exists():
        tok = BPETokenizer.load(tok_path)
        tokens, mask = build_tool_stream(conversations, tok)
        np.save(out_dir / "tool-train.npy", tokens)
        np.save(out_dir / "tool-train-mask.npy", mask)
        print(f"tokens={len(tokens):,} -> {(out_dir / 'tool-train.npy').relative_to(ROOT)}")
    else:
        print(f"tokenizer not found at {tok_path}; wrote raw traces only")

    print(f"conversations={len(conversations)} -> {traces_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
