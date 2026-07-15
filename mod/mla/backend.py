import os

_BACKEND = os.environ.get("ZYN_BACKEND", "numpy").lower()

if _BACKEND in ("cuda", "gpu", "cupy"):
    import cupy as xp
    NAME = "cupy"
else:
    import numpy as xp
    NAME = "numpy"

_DEFAULT_DTYPE = "float32" if NAME == "cupy" else "float64"
DTYPE = xp.dtype(os.environ.get("ZYN_DTYPE", _DEFAULT_DTYPE))


def to_device(a):
    return xp.asarray(a, dtype=DTYPE)


def to_numpy(a):
    if NAME == "cupy":
        return xp.asnumpy(a)
    return a