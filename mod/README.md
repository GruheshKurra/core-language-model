# Model A — From-Scratch Mini Chat Companion

A **3.87M-parameter** decoder-only language model built **entirely from scratch in pure NumPy** — no PyTorch, no JAX, no autograd library. Custom reverse-mode autograd, byte-BPE tokenizer, training loop, KV-cache, and sampler are all hand-written. Optional CuPy backend swap (`ZYN_BACKEND=cuda`) for GPU training.

Scope is deliberately narrow: **short English small-talk / companion replies only** — no code generation, tools, retrieval, or function-calling, by design.

> This repository contains **source code only**. Trained weights and datasets are not committed. Download the checkpoints from the [Hugging Face mirror](https://huggingface.co/karthik-2905/model-a-scratch), or regenerate everything with the scripts below.

## Architecture (modern-tiny decoder)

| Component | Choice |
|---|---|
| Positions | RoPE (rotary) |
| Norm | RMSNorm, Pre-LN |
| Attention | Grouped-Query Attention (8 query heads, 2 KV heads) + QK-Norm |
| MLP | SwiGLU (2/3 hidden-dim rule) |
| Head | Weight-tied to token embedding |
| Tokenizer | Byte-level BPE, vocab 4096, chat special tokens |
| Layers / d_model / head_dim | 4 / 256 / 32 |
| Context length | 256 |
| Params | 3,869,184 |

## Results

- **Pretrain** ([DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog)) — val perplexity ≈ **25**
- **Chat SFT** ([EmpatheticDialogues](https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm), assistant-only loss masking) — masked val ppl ≈ **33**, next-token accuracy **0.325**
- **Refusal / in-scope** — 3/3 coding prompts answered as chit-chat, no code emitted

## Layout

```
mla/        core library — tensor.py (autograd), model.py, tokenizer.py,
            kvcache.py, generate.py, chat.py, optim.py, loss.py, ...
scripts/    pretrain.py, finetune_sft.py, build_sft_corpus.py,
            tokenize_sft.py, evaluate.py, ...
tests/      gradcheck, KV-cache equivalence, sampling, chat-runtime tests
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q          # 78 tests: autograd gradchecks, KV-cache equivalence, ...
```

## Reproduce from scratch

```bash
# 1. Build + tokenize the pretraining corpus
python scripts/build_corpus.py
python scripts/format_chat.py
python scripts/split_data.py
python scripts/train_tokenizer.py
python scripts/tokenize_corpus.py

# 2. Pretrain (CPU, or ZYN_BACKEND=cuda on GPU)
python scripts/pretrain.py --steps 2000

# 3. Build the SFT corpus and fine-tune (assistant-only loss masking)
python scripts/build_sft_corpus.py
python scripts/tokenize_sft.py
python scripts/finetune_sft.py

# 4. Evaluate
python scripts/evaluate.py
```

## Use a trained checkpoint

```python
from mla.checkpoint import load_checkpoint
from mla.tokenizer import Tokenizer
from mla.chat import ChatSession

tok = Tokenizer.load("data/tokenizer/tokenizer.json")
model, _, _ = load_checkpoint("checkpoints/sft_final.npz")

chat = ChatSession(model, tok, temperature=0.8, top_k=40, top_p=0.9)
print(chat.reply("I had a rough day today."))
```

Inference features: greedy / temperature / top-k / top-p sampling, KV-cache (numerically identical to full forward, verified in tests), multi-turn chat runtime, optional `system=` persona conditioning.

## Limitations

It is a **3.87M toy** — expect valence errors, incoherence, and weak multi-turn memory. A from-scratch learning artifact, not a production assistant. Chat-only by design. Trained on non-commercial data (DailyDialog CC-BY-NC-SA, EmpatheticDialogues CC-BY-NC) → **non-commercial use only**.

## License

Code: for research and non-commercial use. Training data licenses (CC-BY-NC) apply to any derived weights.
