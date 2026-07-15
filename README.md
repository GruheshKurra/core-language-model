# AL-1 — Core Language Model Lab

Two small language models, built side by side with different philosophies.

| Sub-project | What | Framework |
|---|---|---|
| [`mod/`](mod) | **Model A** — tiny **from-scratch** general-chat companion. No coding, no tools. | pure NumPy (CuPy GPU swap) |
| `model-b-opensource/` | Take a small OSS base, apply 2026 techniques, improve + ship. *(coming)* | PyTorch / HuggingFace |

## `mod/` — Model A (from scratch)

A **3.87M-parameter** decoder-only LM written entirely in pure NumPy — custom reverse-mode autograd, byte-BPE tokenizer, RoPE + RMSNorm + Grouped-Query Attention + QK-Norm + SwiGLU, weight-tied head, AdamW training, KV-cache, sampling, and a multi-turn chat runtime. Pretrained on DailyDialog, chat-fine-tuned on EmpatheticDialogues with assistant-only loss masking. 78 tests passing (autograd gradchecks, KV-cache logit-equivalence, sampling, chat runtime).

- **Mirror (org):** https://github.com/Zynthetix/model-a-from-scratch
- **Weights (Hugging Face):** https://huggingface.co/karthik-2905/model-a-scratch

See [`mod/README.md`](mod/README.md) for architecture, results, and reproduce-from-scratch steps.

## Reset note

The pre-2026 from-scratch coding-GPT build was wiped in July 2026. This repository is the fresh two-model lab described above.
