import numpy as np

from .backend import to_numpy


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sample_next(logits, temperature=1.0, top_k=0, top_p=0.0, rng=None):
    logits = np.asarray(to_numpy(logits), dtype=np.float64).reshape(-1)
    if temperature is None or temperature <= 0.0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        kth = np.partition(logits, -k)[-k]
        logits = np.where(logits < kth, -np.inf, logits)
    if top_p and 0.0 < top_p < 1.0:
        order = np.argsort(logits)[::-1]
        probs = _softmax(logits[order])
        cum = np.cumsum(probs)
        cutoff = int(np.searchsorted(cum, top_p)) + 1
        keep = order[:cutoff]
        filtered = np.full_like(logits, -np.inf)
        filtered[keep] = logits[keep]
        logits = filtered
    probs = _softmax(logits)
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(len(probs), p=probs))


def generate(model, prompt_ids, max_new=50, eos_id=None,
             temperature=1.0, top_k=0, top_p=0.0, rng=None):
    ids = list(prompt_ids)
    for _ in range(max_new):
        x = np.array([ids[-model.cfg.seq_len:]], dtype=np.int64)
        logits = model(x).data[0, -1]
        nxt = sample_next(logits, temperature, top_k, top_p, rng)
        ids.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
    return ids[len(prompt_ids):]
