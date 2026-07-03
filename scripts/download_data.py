#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="karthik-2905/core-language-model-data")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(out),
    )
    print(f"downloaded {args.repo} -> {out}")


if __name__ == "__main__":
    main()
