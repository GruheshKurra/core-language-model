#!/usr/bin/env python3
import random
from pathlib import Path

IN = Path("data/formatted/dialogues.txt")
TRAIN_OUT = Path("data/splits/train.txt")
VAL_OUT = Path("data/splits/val.txt")
SEED = 42
VAL_FRAC = 0.05


def main():
    lines = IN.read_text(encoding="utf-8").splitlines()
    total = len(lines)

    seen = set()
    unique = []
    for ln in lines:
        if ln in seen:
            continue
        seen.add(ln)
        unique.append(ln)
    n_dupes = total - len(unique)

    rng = random.Random(SEED)
    rng.shuffle(unique)

    n_val = int(len(unique) * VAL_FRAC)
    val = unique[:n_val]
    train = unique[n_val:]

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.write_text("\n".join(train) + "\n", encoding="utf-8")
    VAL_OUT.write_text("\n".join(val) + "\n", encoding="utf-8")

    print(f"total={total} dupes_dropped={n_dupes} unique={len(unique)}")
    print(f"train={len(train)} val={len(val)} seed={SEED}")


if __name__ == "__main__":
    main()
