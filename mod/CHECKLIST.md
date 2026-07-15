# Model A — From-Scratch Mini Chat Companion — Build Checklist

Pure NumPy (CuPy swap allowed for GPU). Tiny. **General chat only — no code, no tools.** Build + test each yourself; don't move on until current item works. Every autograd op passes gradcheck; every layer passes a shape/grad test; training passes a one-paragraph overfit before scaling.

## Scope

| Property | Target |
|---|---|
| Capability | short English small-talk / companion replies |
| Explicitly NOT | code, math tools, retrieval, function-calling |
| Params | ~1–8M (CPU-trainable) |
| Arch | modern-tiny: RoPE + RMSNorm + SwiGLU + GQA, weight-tied head |
| Framework | pure NumPy (`np`), optional `ZYN_BACKEND=cuda` CuPy swap |
| Tokenizer | small byte-BPE, vocab ~4k, chat special tokens |

## Foundation

- [x] Folder layout (`mla/`, `tests/`, `data/`, `checkpoints/`, `serve/`)
- [x] Backend switch (NumPy / CuPy, dtype float64 for gradcheck / float32 GPU)
- [x] Numerical gradient checker (finite diff + rel-error)
- [x] Autograd Tensor (add, mul, matmul, reshape, transpose, sum, gather)
- [x] Activations as ops (silu/gelu, exp, log, softmax, rsqrt)
- [x] Reverse-mode backward (topological order) + broadcast-aware grads

## Data (chat only)

- [x] Collect + clean small-talk / dialogue corpus (chit-chat, no code)
- [x] Chat formatting (`<|user|>` / `<|assistant|>` turns, `<bos>`/`<eos>`)
- [x] Train/val split (by conversation, seed=42) + dedup
- [x] Byte-BPE tokenizer (train ~4k, encode, decode, save/load)
- [x] Chat special tokens reserved before byte base
- [x] Batching (packed turns → next-token x/y windows)

## Model (modern-tiny decoder)

- [x] Token embeddings (tied to output head)
- [x] RoPE (rotary positional embeddings)
- [x] RMSNorm (Pre-LN placement)
- [x] Self-attention (Q/K/V, scaled, causal mask)
- [x] Grouped-query attention (fewer KV heads)
- [x] QK-Norm before RoPE (stability)
- [x] SwiGLU MLP (gated, 2/3 hidden-dim rule)
- [x] Residual connections
- [x] Stack blocks → full model + config

## Pretraining

- [x] Cross-entropy loss (ignore-index for padding)
- [x] AdamW optimizer + grad clipping
- [x] LR warmup + cosine decay
- [x] Sanity: overfit one dialogue (loss < 0.05)
- [x] Training loop + checkpoint save/load/resume

## Chat fine-tune

- [x] Instruction/chat dataset + loss masking (train assistant turns only)
- [x] Fine-tune from pretrain checkpoint
- [x] Persona/system prompt conditioning (companion tone)

## Evaluation

- [x] Val loss + perplexity
- [x] Next-token accuracy
- [x] Sample chat turns per checkpoint (manual read)
- [x] Refusal check: coding prompt → stays in-scope (no code attempt)

## Inference

- [x] Sampling (greedy, temperature, top-k, top-p)
- [x] KV-cache (logits identical to full forward)
- [x] Chat runtime (render turns → generate → stop on `<eos>`)

## Hosting

- [ ] Package model + tokenizer + config bundle
- [ ] Inference-only load mode
- [ ] Serving API (`/chat`, `/health`)
- [ ] Dockerfile + env config
- [ ] Deploy + smoke test live URL

## Scaling (later)

- [ ] NumPy → CuPy for GPU train/infer
- [ ] Bigger config + larger chat corpus
- [ ] Batched/concurrent serving
