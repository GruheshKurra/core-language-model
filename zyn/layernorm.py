from __future__ import annotations

import numpy as np

from zyn.tensor import Tensor


class LayerNorm:

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = Tensor(np.ones(d_model, dtype=np.float64))
        self.beta = Tensor(np.zeros(d_model, dtype=np.float64))

    def __call__(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {x.shape}")

        data = x.data
        mu = data.mean(axis=-1, keepdims=True)
        xc = data - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        xhat = xc * inv_std
        out = Tensor(self.gamma.data * xhat + self.beta.data, (x, self.gamma, self.beta))

        def _backward():
            dout = out.grad
            reduce_axes = tuple(range(dout.ndim - 1))
            self.gamma.grad += (dout * xhat).sum(axis=reduce_axes)
            self.beta.grad += dout.sum(axis=reduce_axes)

            dxhat = dout * self.gamma.data
            mean_dxhat = dxhat.mean(axis=-1, keepdims=True)
            mean_dxhat_xhat = (dxhat * xhat).mean(axis=-1, keepdims=True)
            x.grad += inv_std * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat)
        out._backward = _backward
        return out

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]
