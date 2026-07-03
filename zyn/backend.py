from __future__ import annotations

import os

import numpy as _np


def _select_backend():
    name = os.environ.get("ZYN_BACKEND", "numpy").lower()
    if name in ("cuda", "cupy", "gpu"):
        try:
            import cupy as _cp

            return _cp
        except Exception:
            return _np
    return _np


xp = _select_backend()


def _select_dtype():
    name = os.environ.get("ZYN_DTYPE", "float64").lower()
    if name in ("float32", "f32", "32"):
        return xp.float32
    return xp.float64


fdtype = _select_dtype()


def to_numpy(array):
    if xp is _np:
        return _np.asarray(array)
    return xp.asnumpy(array)


def scatter_add(target, indices, source):
    if xp is _np:
        _np.add.at(target, indices, source)
    else:
        import cupyx

        cupyx.scatter_add(target, indices, source)


def is_gpu() -> bool:
    return xp is not _np


def backend_name() -> str:
    return "numpy" if xp is _np else "cupy"
