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
- [ ] MLP block
- [ ] Residual connections
- [ ] Stack blocks into full GPT + config

## Pretraining

- [ ] Cross-entropy loss
- [ ] Adam/AdamW optimizer
- [ ] Gradient clipping
- [ ] LR warmup + decay
- [ ] Sanity overfit one paragraph
- [ ] Training loop
- [ ] Checkpoint save/load/resume

## Evaluation

- [ ] Val loss + perplexity
- [ ] Next-token accuracy
- [ ] Sample generations per checkpoint

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