from .backend import xp, to_numpy
from .tensor import Tensor


def gradcheck(f, inputs, eps=1e-6, tol=1e-5):
    if isinstance(inputs, Tensor):
        inputs = [inputs]

    out = f(*inputs)
    out.backward()
    analytic = [to_numpy(t.grad).copy() for t in inputs]

    max_rel = 0.0
    for k, t in enumerate(inputs):
        flat = t.data.ravel()
        num = xp.zeros(flat.size, dtype=flat.dtype)
        for i in range(flat.size):
            orig = flat[i].copy()
            flat[i] = orig + eps
            plus = float(f(*inputs).data.sum())
            flat[i] = orig - eps
            minus = float(f(*inputs).data.sum())
            flat[i] = orig
            num[i] = (plus - minus) / (2.0 * eps)
        num = to_numpy(num).reshape(t.data.shape)
        a = analytic[k]
        denom = xp.maximum(1e-12, xp.abs(a) + xp.abs(num))
        rel = float((xp.abs(a - num) / denom).max())
        max_rel = max(max_rel, rel)

    return max_rel < tol, max_rel
