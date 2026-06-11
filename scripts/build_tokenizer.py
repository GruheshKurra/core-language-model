from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zyn.bpe import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def read_sample(jsonl_path: Path, max_bytes: int) -> str:
    chunks: list[str] = []
    total = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            chunks.append(text)
            total += len(text)
            if total >= max_bytes:
                break
    return "\n".join(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="codeparrot")
    ap.add_argument("--sample-mb", type=float, default=2.0)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--out", default=str(PROCESSED / "tokenizer.json"))
    args = ap.parse_args()

    src = PROCESSED / args.source / "train.jsonl"
    sample = read_sample(src, int(args.sample_mb * 1_000_000))
    print(f"sample chars={len(sample):,} from {src.relative_to(ROOT)}")

    t0 = time.time()
    tok = BPETokenizer().train(sample, vocab_size=args.vocab_size, min_freq=args.min_freq)
    print(f"trained vocab_size={tok.vocab_size} merges={len(tok.merges)} in {time.time() - t0:.1f}s")

    tok.save(args.out)
    print(f"saved {Path(args.out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
