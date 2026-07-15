#!/usr/bin/env python3
import io
import re
from pathlib import Path

import requests
import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm/resolve/refs%2Fconvert%2Fparquet/default"
SPLITS = {"train": "train", "val": "valid"}
OUT_DIR = Path("data/sft")

BOS = "<bos>"
EOS = "<eos>"
ROLES = ("<|user|>", "<|assistant|>")

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


def fetch_conversations(split):
    url = f"{BASE}/{split}/0000.parquet"
    raw = requests.get(url, timeout=120).content
    table = pq.read_table(io.BytesIO(raw))
    return table.column("conversations").to_pylist()


def format_dialogue(turns):
    parts = [BOS]
    for i, text in enumerate(turns):
        parts.append(ROLES[i % 2])
        parts.append(text)
        parts.append(EOS)
    return "".join(parts)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in SPLITS.items():
        seen = kept = dropped_short = dropped_code = 0
        out = OUT_DIR / f"{name}.txt"
        with out.open("w", encoding="utf-8") as f:
            for convo in fetch_conversations(split):
                seen += 1
                turns = [t for t in (clean_utt(c["content"]) for c in convo) if t]
                if len(turns) < 2:
                    dropped_short += 1
                    continue
                if has_code(turns):
                    dropped_code += 1
                    continue
                f.write(format_dialogue(turns) + "\n")
                kept += 1
        print(f"{name}: seen={seen} kept={kept} dropped_short={dropped_short} dropped_code={dropped_code} -> {out}")


if __name__ == "__main__":
    main()
