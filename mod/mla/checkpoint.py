import json
from dataclasses import asdict

import numpy as np

from .backend import to_numpy, to_device
from .model import Config, Model
from .optim import AdamW


def save_checkpoint(path, model, opt, step):
    arrays = {}
    params = model.parameters()
    for i, p in enumerate(params):
        arrays[f"p{i}"] = to_numpy(p.data)
    for i in range(len(opt.params)):
        arrays[f"m{i}"] = to_numpy(opt.m[i])
        arrays[f"v{i}"] = to_numpy(opt.v[i])
    meta = {
        "config": asdict(model.cfg),
        "step": step,
        "n_params": len(params),
        "opt": {
            "lr": opt.lr, "b1": opt.b1, "b2": opt.b2,
            "eps": opt.eps, "wd": opt.wd, "t": opt.t,
        },
    }
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)


def load_checkpoint(path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(data["meta"].item())
    cfg = Config(**meta["config"])
    model = Model(cfg)
    params = model.parameters()
    for i, p in enumerate(params):
        p.data = to_device(data[f"p{i}"])
    o = meta["opt"]
    opt = AdamW(params, lr=o["lr"], betas=(o["b1"], o["b2"]),
                eps=o["eps"], weight_decay=o["wd"])
    opt.t = o["t"]
    for i in range(len(params)):
        opt.m[i] = to_device(data[f"m{i}"])
        opt.v[i] = to_device(data[f"v{i}"])
    return model, opt, meta["step"]
