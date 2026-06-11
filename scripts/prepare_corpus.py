#!/usr/bin/env python3

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
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


def doc_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_records(records: list[dict], train_ratio: float) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    if n_train == 0 and shuffled:
        n_train = 1
    if n_train == len(shuffled) and len(shuffled) > 1:
        n_train = len(shuffled) - 1
    return shuffled[:n_train], shuffled[n_train:]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def dedupe_records(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for rec in records:
        keys = (rec["id"], norm_key(rec["text"]))
        if keys[0] in seen or keys[1] in seen:
            continue
        seen.update(keys)
        out.append(rec)
    return out


def decompress_codeparrot_gz(gz_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(gz_path, "rt", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            dst.write(line + "\n")
            count += 1
    return count


def load_codeparrot_records(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            text = clean_text(row.get("content", ""))
            if len(text) < 50:
                continue
            records.append(
                {
                    "id": doc_id(text),
                    "source": "codeparrot",
                    "text": text,
                    "meta": {
                        "repo_name": row.get("repo_name"),
                        "path": row.get("path"),
                        "license": row.get("license"),
                        "shard": jsonl_path.name,
                        "row": i,
                    },
                }
            )
    return dedupe_records(records)


def load_stack_smol_records(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            text = clean_text(row.get("content", ""))
            if len(text) < 20:
                continue
            records.append(
                {
                    "id": doc_id(text),
                    "source": "the-stack-smol-xs",
                    "text": text,
                    "meta": {
                        "lang": row.get("lang"),
                        "ext": row.get("ext"),
                        "row": i,
                    },
                }
            )
    return dedupe_records(records)


def load_instruction_records(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = clean_text(row.get("instruction", ""))
            inp = clean_text(row.get("input", ""))
            output = clean_text(row.get("output", ""))
            text = clean_text(row.get("text", ""))
            if not instruction and not output:
                continue
            payload = f"{instruction}\n{inp}\n{output}"
            records.append(
                {
                    "id": doc_id(payload),
                    "source": "python-codes-25k",
                    "instruction": instruction,
                    "input": inp,
                    "output": output,
                    "text": text or payload,
                    "meta": {"row": i},
                }
            )
    return dedupe_records(records)


def process_dataset(
    name: str,
    records: list[dict],
    train_ratio: float,
) -> dict:
    train, val = split_records(records, train_ratio)
    out_dir = PROCESSED / name
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    return {
        "name": name,
        "total": len(records),
        "train": len(train),
        "val": len(val),
        "train_ratio": train_ratio,
        "out_dir": str(out_dir.relative_to(ROOT)),
    }


def main() -> None:
    gz_path = RAW / "codeparrot-clean" / "file-000000000054.json.gz"
    codeparrot_jsonl = RAW / "codeparrot-clean" / "codeparrot-054.jsonl"
    n_lines = decompress_codeparrot_gz(gz_path, codeparrot_jsonl)
    print(f"decompressed {gz_path.name} -> {codeparrot_jsonl.name} ({n_lines} lines)")

    manifest: list[dict] = []

    codeparrot = load_codeparrot_records(codeparrot_jsonl)
    manifest.append(process_dataset("codeparrot", codeparrot, train_ratio=0.99))

    pipeline_path = RAW / "pipeline" / "the-stack-smol-xs-python.jsonl"
    pipeline = load_stack_smol_records(pipeline_path)
    manifest.append(process_dataset("pipeline", pipeline, train_ratio=0.90))

    instruct_path = RAW / "python-codes-25k" / "python-codes-25k.jsonl"
    instruct = load_instruction_records(instruct_path)
    manifest.append(process_dataset("python-codes-25k", instruct, train_ratio=0.99))

    manifest_path = PROCESSED / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "note": (
                    "train/val only; use MBPP/HumanEval for held-out test eval later. "
                    "Dedup = whitespace-normalized SHA256 (catches reformatted copies, "
                    "not one-line-diff near-dupes). codeparrot = single shard 054."
                ),
                "datasets": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
