from __future__ import annotations

from pathlib import Path

import numpy as np

from zyn.backend import fdtype, to_numpy, xp
from zyn.gpt import GPT, GPTConfig
from zyn.optim import AdamW


def save_checkpoint(path: str | Path, model: GPT, optimizer: AdamW, step: int) -> None:
    arrays = {}
    params = model.parameters()
    for i, p in enumerate(params):
        arrays[f"p{i}"] = to_numpy(p.data)
    for i, m in enumerate(optimizer.m):
        arrays[f"m{i}"] = to_numpy(m)
    for i, v in enumerate(optimizer.v):
        arrays[f"v{i}"] = to_numpy(v)

    c = model.config
    meta = {
        "step": step,
        "n_params": len(params),
        "t": optimizer.t,
        "lr": optimizer.lr,
        "beta1": optimizer.beta1,
        "beta2": optimizer.beta2,
        "eps": optimizer.eps,
        "weight_decay": optimizer.weight_decay,
        "vocab_size": c.vocab_size,
        "d_model": c.d_model,
        "n_head": c.n_head,
        "n_layer": c.n_layer,
        "d_ff": -1 if c.d_ff is None else c.d_ff,
        "max_seq": c.max_seq,
        "std": c.std,
    }
    for k, val in meta.items():
        arrays[f"meta_{k}"] = np.array(val)
    np.savez(path, **arrays)


def load_checkpoint(path: str | Path) -> tuple[GPT, AdamW, int]:
    data = np.load(path)

    def g(k):
        return data[f"meta_{k}"]

    d_ff = int(g("d_ff"))
    cfg = GPTConfig(
        vocab_size=int(g("vocab_size")),
        d_model=int(g("d_model")),
        n_head=int(g("n_head")),
        n_layer=int(g("n_layer")),
        d_ff=None if d_ff < 0 else d_ff,
        max_seq=int(g("max_seq")),
        std=float(g("std")),
    )
    model = GPT(cfg)
    params = model.parameters()
    n = int(g("n_params"))
    if n != len(params):
        raise ValueError(f"checkpoint has {n} params, model has {len(params)}")
    for i, p in enumerate(params):
        p.data = xp.asarray(data[f"p{i}"], dtype=fdtype)

    optimizer = AdamW(
        params,
        lr=float(g("lr")),
        betas=(float(g("beta1")), float(g("beta2"))),
        eps=float(g("eps")),
        weight_decay=float(g("weight_decay")),
    )
    optimizer.t = int(g("t"))
    for i in range(n):
        optimizer.m[i] = xp.asarray(data[f"m{i}"], dtype=fdtype)
        optimizer.v[i] = xp.asarray(data[f"v{i}"], dtype=fdtype)

    return model, optimizer, int(g("step"))
