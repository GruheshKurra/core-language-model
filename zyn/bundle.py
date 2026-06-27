from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from zyn.bpe import BPETokenizer
from zyn.gpt import GPT, GPTConfig
from zyn.kvcache import CachedGPT

BUNDLE_VERSION = 1


@dataclass
class Bundle:
    model: GPT
    tokenizer: BPETokenizer
    config: GPTConfig

    def cached_model(self) -> CachedGPT:
        return CachedGPT(self.model)


def _config_dict(config: GPTConfig) -> dict:
    return {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "d_ff": config.d_ff,
        "max_seq": config.max_seq,
        "std": config.std,
    }


def save_bundle(
    path: str | Path,
    model: GPT,
    tokenizer: BPETokenizer,
    meta: dict | None = None,
) -> None:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    params = model.parameters()
    arrays = {f"p{i}": p.data for i, p in enumerate(params)}
    arrays["n_params"] = np.array(len(params))
    np.savez(out / "model.npz", **arrays)

    tokenizer.save(out / "tokenizer.json")

    config = _config_dict(model.config)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    info = {
        "format": "zyn-bundle",
        "version": BUNDLE_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "vocab_size": model.config.vocab_size,
        "n_params": int(model.num_params()),
        "arch": config,
    }
    if meta:
        info["meta"] = meta
    (out / "bundle.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_bundle(path: str | Path) -> Bundle:
    src = Path(path)
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg = GPTConfig(**config)
    model = GPT(cfg)

    data = np.load(src / "model.npz")
    params = model.parameters()
    n = int(data["n_params"])
    if n != len(params):
        raise ValueError(f"bundle has {n} params, model has {len(params)}")
    for i, p in enumerate(params):
        p.data = data[f"p{i}"].copy()

    tokenizer = BPETokenizer.load(src / "tokenizer.json")
    return Bundle(model=model, tokenizer=tokenizer, config=cfg)
