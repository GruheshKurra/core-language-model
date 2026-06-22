# Core Language Model

A small coding + general-purpose language model built **from scratch in pure NumPy** — no PyTorch, no JAX, no autograd library. Every piece (autograd engine, tokenizer, attention, layer norm) is hand-written and gradient-checked.

This is a from-first-principles build of a transformer language model, aimed at understanding every formula well enough to reimplement it without reference.

## Why pure NumPy?

The goal isn't a competitive model — it's a fully transparent one. Every tensor operation, every gradient, every attention score is computed by code you can read top to bottom. Once the stack is verified end-to-end on CPU, it's designed to swap to CuPy for GPU scaling without touching the math.

## Project layout

```
zyn/                Core library
├── tensor.py       Autograd Tensor: add, mul, matmul, relu, gelu, exp, log, sum, gather, broadcasting
├── gradcheck.py     Numerical gradient checker (verifies backward() against finite differences)
├── embedding.py     Token embedding layer
├── positional.py     Positional encoding
├── attention.py      Scaled dot-product self-attention with causal masking
├── multihead.py       Multi-head attention
├── layernorm.py        Layer normalization
├── mlp.py              Feed-forward block
├── block.py            Transformer block (attention + MLP + residual + layer norm)
├── gpt.py              Full GPT stack + GPTConfig (tied embeddings)
├── loss.py             Cross-entropy with ignore_index masking
├── optim.py            AdamW + gradient clipping
├── schedule.py         Cosine LR with warmup/decay
├── train.py            Training loop + single train step
├── checkpoint.py       Save / load / resume (weights + optimizer state)
├── eval.py             Val loss, perplexity, next-token accuracy
├── generate.py         Sampling: greedy / temperature / top-k / top-p
├── batching.py        Next-token (x, y) batch construction
└── bpe.py             Byte-pair encoding tokenizer (train, encode, decode, special tokens)

scripts/
├── prepare_corpus.py  Clean, dedupe, and split raw corpus data
├── build_tokenizer.py  Train the BPE tokenizer on a corpus sample
└── tokenize_corpus.py   Tokenize and serialize the full corpus

tests/                Unit tests for every module above (pytest)
serve/                Inference API (planned)
checkpoints/          Saved model weights (not tracked)
```

## Status

Built incrementally with a strict checklist. **Complete and tested:** gradient-checked autograd, BPE tokenizer with special tokens, batching, embeddings, positional encoding, self-attention, multi-head attention, layer norm, MLP, transformer block with residuals, the full GPT stack, cross-entropy loss, AdamW with gradient clipping, cosine LR warmup/decay, training loop, checkpoint save/load/resume, and the evaluation suite (val loss, perplexity, next-token accuracy, sampling generation).

A one-paragraph overfit test drives loss below 0.05 with exact token-level memorization, proving the autograd + training stack learns end-to-end.

Planned: instruction + tool-calling fine-tuning (loss masking is already wired via `ignore_index`), KV-cache for inference, and a `/generate` serving API.

## Data strategy

The corpus targets **code + general text**, not toy story datasets:

1. **Pipeline bring-up** — small Python JSONL sample for clean/split/tokenize/overfit testing
2. **Pretraining** — Python-heavy code corpus, with a smaller general-text slice
3. **Instruction fine-tuning** — code instruction/response pairs
4. **Tool-calling fine-tuning** — synthetic `<tool_call>` / `</tool_call>` / `<tool_result>` traces

BPE vocabulary: 8k–16k tokens, with reserved special tokens (`<pad>`, `<bos>`, `<eos>`, `<tool_call>`, `</tool_call>`, `<tool_result>`).

## Running tests

```bash
python -m pytest -q
```

102 tests covering the autograd engine, gradient checks, tokenizer, batching, every model layer, the full GPT stack, loss, optimizer, LR schedule, gradient clipping, training loop, checkpointing, evaluation, and sampling.

## Scale

Targeting ~1–10M parameters on CPU for the initial build. GPU scaling (CuPy backend) and a larger corpus come later, once the full stack is verified correct.
