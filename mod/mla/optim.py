from .backend import xp


def clip_grad_norm(params, max_norm):
    total = 0.0
    for p in params:
        total = total + float((p.grad * p.grad).sum())
    total = total ** 0.5
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            p.grad = p.grad * scale
    return total


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [xp.zeros_like(p.data) for p in self.params]
        self.v = [xp.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
            mhat = self.m[i] / bc1
            vhat = self.v[i] / bc2
            p.data = p.data - self.lr * (mhat / (xp.sqrt(vhat) + self.eps) + self.wd * p.data)

    def zero_grad(self):
        for p in self.params:
            p.grad = xp.zeros_like(p.data)
