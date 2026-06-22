# Core Language Model — Build Checklist

NumPy only. Build + test each yourself. Don't move on until current item works.

## Foundation

- [x] Folder layout (`zyn/`, `tests/`, `data/`, `checkpoints/`, `serve/`)
- [x] Numerical gradient checker
- [x] Autograd Tensor (add, mul, matmul, relu, exp, log, sum)
- [x] Reverse-mode backward (topological order)
- [x] Broadcast-aware gradients

## Data

- [x] Collect + clean corpus, train/val split
- [x] BPE tokenizer (train, encode, decode)
- [x] Special tokens (`<bos> <eos> <pad> <tool_call> </tool_call> <tool_result>`)
- [x] Save/load tokenizer
- [x] Batching (next-token x/y pairs)

## Model

- [x] Token embeddings
- [x] Positional encoding
- [x] Self-attention (Q/K/V, scaling, causal mask)
- [x] Multi-head
- [x] LayerNorm
- [x] MLP block
- [x] Residual connections
- [x] Stack blocks into full GPT + config

## Pretraining

- [x] Cross-entropy loss
- [x] Adam/AdamW optimizer
- [x] Gradient clipping
- [x] LR warmup + decay
- [x] Sanity overfit one paragraph
- [x] Training loop
- [x] Checkpoint save/load/resume

## Evaluation

- [x] Val loss + perplexity
- [x] Next-token accuracy
- [x] Sample generations per checkpoint

## Fine-tuning

- [ ] Instruction dataset + loss masking
- [ ] Instruction fine-tune from checkpoint
- [ ] Tool-calling dataset (synthetic)
- [ ] Tool-calling fine-tune

## Inference

- [ ] Sampling (greedy, temperature, top-k, top-p)
- [ ] KV-cache
- [ ] Tool-calling runtime (call → execute → inject result → resume)

## Hosting

- [ ] Package model + tokenizer + config bundle
- [ ] Load + inference-only mode
- [ ] Serving API (`/generate`, `/chat`, `/health`)
- [ ] Dockerfile + env config
- [ ] Deploy + smoke test live URL

## Scaling (later)

- [ ] Swap NumPy → CuPy (GPU)
- [ ] Bigger config + larger corpus
- [ ] Expand tool dataset
- [ ] Batched/concurrent serving