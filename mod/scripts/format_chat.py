#!/usr/bin/env python3
import json
from pathlib import Path

IN = Path("data/raw/dailydialog_clean.jsonl")
OUT = Path("data/formatted/dialogues.txt")

BOS = "<bos>"
EOS = "<eos>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
ROLES = (USER, ASSISTANT)


def format_dialogue(turns):
    parts = [BOS]
    for i, text in enumerate(turns):
        parts.append(ROLES[i % 2])
        parts.append(text)
        parts.append(EOS)
    return "".join(parts)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with IN.open(encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            turns = json.loads(line)["turns"]
            fout.write(format_dialogue(turns) + "\n")
            n += 1
    print(f"formatted={n}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
