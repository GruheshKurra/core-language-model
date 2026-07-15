#!/usr/bin/env python3
import io
import json
import re
from pathlib import Path

import requests
import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/li2017dailydialog/daily_dialog/resolve/refs%2Fconvert%2Fparquet/default"
SPLITS = ("train", "validation", "test")
OUT = Path("data/raw/dailydialog_clean.jsonl")

_MULTI = re.compile(r"\s+")
_APOS = re.compile(r"\s+'\s*")
_PUNCT = re.compile(r"\s+([.,!?;:])")
_CODE = re.compile(r"`|</|/>|def |import |class |printf|console\.log|#include|<tool")


def clean_utt(u):
    u = u.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    u = _MULTI.sub(" ", u.strip())
    u = _APOS.sub("'", u)
    u = _PUNCT.sub(r"\1", u)
    return _MULTI.sub(" ", u).strip()


def has_code(turns):
    return bool(_CODE.search(" ".join(turns)))


def fetch_dialogs(split):
    url = f"{BASE}/{split}/0000.parquet"
    raw = requests.get(url, timeout=120).content
    table = pq.read_table(io.BytesIO(raw))
    return table.column("dialog").to_pylist()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = kept = dropped_short = dropped_code = 0
    with OUT.open("w", encoding="utf-8") as f:
        for split in SPLITS:
            for dialog in fetch_dialogs(split):
                seen += 1
                turns = [t for t in (clean_utt(u) for u in dialog) if t]
                if len(turns) < 2:
                    dropped_short += 1
                    continue
                if has_code(turns):
                    dropped_code += 1
                    continue
                f.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")
                kept += 1
    print(f"seen={seen} kept={kept} dropped_short={dropped_short} dropped_code={dropped_code}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
