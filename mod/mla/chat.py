import numpy as np

from .kvcache import KVCache, forward_cached
from .generate import sample_next


def render(tok, history, add_generation_prompt=True, system=None):
    bos = tok.special_to_id["<bos>"]
    eos = tok.special_to_id["<eos>"]
    roles = {"user": tok.special_to_id["<|user|>"],
             "assistant": tok.special_to_id["<|assistant|>"]}
    ids = [bos]
    if system:
        ids.extend(tok.encode(system))
    for role, text in history:
        ids.append(roles[role])
        ids.extend(tok.encode(text))
        ids.append(eos)
    if add_generation_prompt:
        ids.append(roles["assistant"])
    return ids


def generate_cached(model, prompt_ids, max_new=60, eos_id=None, stop_ids=(),
                    temperature=1.0, top_k=0, top_p=0.0, rng=None):
    cache = KVCache(model.cfg.n_layers)
    prompt = prompt_ids[-model.cfg.seq_len:]
    logits = forward_cached(model, np.array([prompt], dtype=np.int64), cache)
    last = np.asarray(logits)[0, -1]
    out = []
    for _ in range(max_new):
        nxt = sample_next(last, temperature, top_k, top_p, rng)
        if nxt == eos_id or nxt in stop_ids:
            break
        out.append(nxt)
        logits = forward_cached(model, np.array([[nxt]], dtype=np.int64), cache)
        last = np.asarray(logits)[0, -1]
    return out


class ChatSession:
    def __init__(self, model, tok, temperature=0.8, top_k=40, top_p=0.9,
                 max_new=60, seed=None, system=None):
        self.model = model
        self.tok = tok
        self.history = []
        self.system = system
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new = max_new
        self.rng = np.random.default_rng(seed)
        self.eos = tok.special_to_id["<eos>"]
        self.user = tok.special_to_id["<|user|>"]

    def reply(self, text):
        self.history.append(("user", text))
        prompt = render(self.tok, self.history, add_generation_prompt=True,
                        system=self.system)
        out = generate_cached(self.model, prompt, self.max_new, self.eos,
                              stop_ids=(self.user,), temperature=self.temperature,
                              top_k=self.top_k, top_p=self.top_p, rng=self.rng)
        answer = self.tok.decode(out).strip()
        self.history.append(("assistant", answer))
        return answer
