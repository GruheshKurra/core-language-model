import numpy as np


def load_ids(path):
    return np.load(path)


def get_batch(ids, block_size, batch_size, rng):
    hi = len(ids) - block_size - 1
    ix = rng.integers(0, hi, size=batch_size)
    x = np.stack([ids[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([ids[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    return x, y


def iter_batches(ids, block_size, batch_size, rng, n_batches):
    for _ in range(n_batches):
        yield get_batch(ids, block_size, batch_size, rng)
