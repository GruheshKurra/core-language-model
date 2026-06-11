from __future__ import annotations
from typing import Callable
import numpy as np


def numeric_grad(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = x.astype(np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]

        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig

        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def rel_error(a: np.ndarray, b: np.ndarray, delta: float = 1e-8) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + delta)


def gradcheck(f: Callable[[np.ndarray], float], x: np.ndarray, analytic_grad: np.ndarray,
              eps: float = 1e-5, tol: float = 1e-6, verbose: bool = True) -> bool:
    num = numeric_grad(f, x, eps)
    ana = np.asarray(analytic_grad, np.float64)
    if ana.shape != num.shape:
        raise ValueError(f"shape mismatch: analytic {ana.shape} vs numeric {num.shape}")

    err = rel_error(ana, num)
    worst = float(err.max())
    ok = worst < tol

    if verbose:
        bad = np.unravel_index(np.argmax(err), err.shape)
        print(f"[gradcheck] max rel error = {worst:.3e}  tol = {tol:.0e}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  worst at {bad}: analytic={ana[bad]:.6e} numeric={num[bad]:.6e}")
    return ok
