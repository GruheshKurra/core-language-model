import gc
import math

from .data import get_batch
from .loss import cross_entropy


def eval_loss(model, ids, block_size, batch_size, n_batches, rng, ignore_index=-1):
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(ids, block_size, batch_size, rng)
        loss = cross_entropy(model(x), y, ignore_index)
        total += float(loss.data)
        del loss
        gc.collect()
    mean = total / max(1, n_batches)
    return mean, math.exp(mean)
