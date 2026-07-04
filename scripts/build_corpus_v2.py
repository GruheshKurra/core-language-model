#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SEED = 42
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WS_RE = re.compile(r"\s+")


def norm_key(text: str) -> str:
    return hashlib.sha256(WS_RE.sub(" ", text).strip().encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub("", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def stream_code(n_docs: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    out: list[dict] = []
    for row in ds:
        text = clean_text(row.get("content", ""))
        if len(text) < 50:
            continue
        out.append({"source": "code", "text": text})
        if len(out) >= n_docs:
            break
    return out


def stream_web(n_docs: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    out: list[dict] = []
    for row in ds:
        text = clean_text(row.get("text", ""))
        if len(text) < 200:
            continue
        out.append({"source": "web", "text": text})
        if len(out) >= n_docs:
            break
    return out


def dedupe(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for rec in records:
        k = norm_key(rec["text"])
        if k in seen:
            continue
        seen.add(k)
        out.append(rec)
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-docs", type=int, default=120000)
    ap.add_argument("--web-docs", type=int, default=200000)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--out", default=str(PROCESSED / "mix"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.code_docs, args.web_docs = 200, 300

    print(f"streaming {args.code_docs} code docs...")
    code = stream_code(args.code_docs)
    print(f"streaming {args.web_docs} web docs...")
    web = stream_web(args.web_docs)

    records = dedupe(code + web)
    rng = random.Random(SEED)
    rng.shuffle(records)

    n_val = max(1, int(len(records) * args.val_ratio))
    val = records[:n_val]
    train = records[n_val:]

    out_dir = Path(args.out)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    chars = sum(len(r["text"]) for r in records)
    print(
        f"code={len(code)} web={len(web)} deduped={len(records)} "
        f"train={len(train)} val={len(val)} ~chars={chars:,} ~est_tokens={chars // 4:,}"
    )
    print(f"wrote {out_dir.relative_to(ROOT)}/train.jsonl + val.jsonl")


if __name__ == "__main__":
    main()
