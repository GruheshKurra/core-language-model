import numpy as np

from zyn.bpe import BPETokenizer
from zyn.gpt import GPT, GPTConfig
from zyn.instruct import IGNORE_INDEX, MaskedBatcher, build_instruct_stream, record_to_messages
from zyn.optim import AdamW
from zyn.train import train


def _tok():
    return BPETokenizer()


def test_record_to_messages_combines_input():
    rec = {"instruction": "do x", "input": "data", "output": "result"}
    messages = record_to_messages(rec)
    assert messages[0]["role"] == "user"
    assert "do x" in messages[0]["content"]
    assert "data" in messages[0]["content"]
    assert messages[1]["content"] == "result"


def test_stream_shapes_match():
    tok = _tok()
    recs = [
        {"instruction": "a", "input": "", "output": "b"},
        {"instruction": "c", "input": "d", "output": "e"},
    ]
    tokens, mask = build_instruct_stream(recs, tok)
    assert tokens.shape == mask.shape
    assert mask.max() == 1
    assert mask.min() == 0


def test_masked_batcher_ignores_prompt_targets():
    tok = _tok()
    recs = [{"instruction": "question here", "input": "", "output": "answer text"}] * 8
    tokens, mask = build_instruct_stream(recs, tok)
    batcher = MaskedBatcher(tokens, mask, batch_size=2, context_len=16, seed=0)
    x, y = batcher.next_batch()
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
    assert (y == IGNORE_INDEX).any()


def test_overfit_instruction_response():
    tok = _tok()
    rec = {"instruction": "ping", "input": "", "output": "pong"}
    tokens, mask = build_instruct_stream([rec], tok)
    seq = tokens[None, :].astype(np.int64)
    mask_row = mask[None, :].astype(bool)
    x = seq[:, :-1]
    y_tokens = seq[:, 1:]
    y_keep = mask_row[:, 1:]
    y = np.where(y_keep, y_tokens, IGNORE_INDEX)

    vocab = tok.vocab_size
    np.random.seed(0)
    model = GPT(GPTConfig(vocab_size=vocab, d_model=64, n_head=4, n_layer=2, max_seq=seq.shape[1]))
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    hist = train(
        model,
        opt,
        lambda: (x, y),
        steps=300,
        lr_max=3e-3,
        warmup_steps=20,
        max_steps=300,
        ignore_index=IGNORE_INDEX,
    )
    assert hist[-1]["loss"] < 0.05

    logits = model(x).data
    preds = logits.argmax(axis=-1)
    keep = y != IGNORE_INDEX
    assert np.array_equal(preds[keep], y[keep])
