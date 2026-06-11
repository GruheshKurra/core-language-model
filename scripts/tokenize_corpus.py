from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zyn.bpe import BPETokenizer, _pretokens, _word_to_ids

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def encode_stream(tok: BPETokenizer, jsonl_path: Path, max_bytes: int | None) -> list[int]:
    cache: dict[str, list[int]] = {}
    out: list[int] = []
    total = 0
    eos = tok.eos_id
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            for w in _pretokens(text):
                ids = cache.get(w)
                if ids is None:
                    ids = tok._encode_word(_word_to_ids(w, tok.byte_base))
                    cache[w] = ids
                out.extend(ids)
            out.append(eos)
            total += len(text)
            if max_bytes is not None and total >= max_bytes:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="codeparrot")
    ap.add_argument("--tokenizer", default=str(PROCESSED / "tokenizer.json"))
    ap.add_argument("--max-mb", type=float, default=20.0, help="cap per split; 0 = full")
    ap.add_argument("--out-dir", default=str(PROCESSED / "tokens"))
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = None if args.max_mb == 0 else int(args.max_mb * 1_000_000)

    for split in ("train", "val"):
        src = PROCESSED / args.source / f"{split}.jsonl"
        t0 = time.time()
        ids = encode_stream(tok, src, cap)
        arr = np.asarray(ids, dtype=dtype)
        out_path = out_dir / f"{args.source}-{split}.npy"
        np.save(out_path, arr)
        print(
            f"{split}: {len(arr):,} tokens dtype={arr.dtype} "
            f"-> {out_path.relative_to(ROOT)} ({time.time() - t0:.1f}s)"
        )


if __name__ == "__main__":
    main()
