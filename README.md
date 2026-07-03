# Core Language Model

A small coding + general-purpose language model built **from scratch in pure NumPy** — no PyTorch, no JAX, no autograd library. Every piece (autograd engine, tokenizer, attention, layer norm) is hand-written and gradient-checked.

This is a from-first-principles build of a transformer language model, aimed at understanding every formula well enough to reimplement it without reference.

## Why pure NumPy?

The goal isn't a competitive model — it's a fully transparent one. Every tensor operation, every gradient, every attention score is computed by code you can read top to bottom. Once the stack is verified end-to-end on CPU, it's designed to swap to CuPy for GPU scaling without touching the math.

## Project layout

```
zyn/                Core library
├── backend.py      Array backend selector (NumPy default, CuPy via ZYN_BACKEND=cuda)
├── tensor.py       Autograd Tensor: add, mul, matmul, relu, gelu, exp, log, sum, gather, broadcasting
├── gradcheck.py     Numerical gradient checker (verifies backward() against finite differences)
├── embedding.py     Token embedding layer
├── positional.py     Positional encoding
├── attention.py      Scaled dot-product self-attention + multi-head, causal masking
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
├── bpe.py             Byte-pair encoding tokenizer (train, encode, decode, special tokens)
├── chat.py            Chat/instruction templating -> (tokens, loss_mask)
├── instruct.py        Instruction stream packing + MaskedBatcher (loss masking)
├── tools.py           Tool registry: file tools path-jailed to a root; shell/python gated, not OS-isolated
├── tooldata.py        Synthetic tool-call trace generation
├── kvcache.py         KV-cached inference (CachedGPT) + generate_cached
├── agent.py           Tool-calling runtime (call -> execute -> inject -> resume)
└── bundle.py          Package / load inference bundle (weights + tokenizer + config)

scripts/
├── prepare_corpus.py   Clean, dedupe, and split raw corpus data
├── build_tokenizer.py   Train the BPE tokenizer on a corpus sample
├── tokenize_corpus.py    Tokenize and serialize the full corpus
├── pretrain.py           Config-driven pretraining (resume, eval, checkpoint)
├── finetune_instruct.py   Instruction fine-tune from a checkpoint
├── finetune_tools.py       Tool-calling fine-tune from a checkpoint
├── build_tool_dataset.py    Generate the synthetic tool-call dataset
└── smoke_test.py             Hit /health + /generate on a deployed server

serve/                FastAPI inference API
├── app.py            /generate, /chat, /health (create_app + uvicorn entrypoint)
├── schemas.py         Pydantic request/response models
├── config.py          Env-driven settings (MODEL_DIR, sampling, sandbox gating)
└── batcher.py         Async micro-batching queue for concurrent serving

configs/              Pretraining presets (small-cpu, medium-gpu)
tests/                Unit tests for every module above (pytest)
checkpoints/          Saved model weights (not tracked)
```

## Status

Built incrementally with a strict checklist. **Complete and tested:** gradient-checked autograd, BPE tokenizer with special tokens, batching, embeddings, positional encoding, self-attention, multi-head attention, layer norm, MLP, transformer block with residuals, the full GPT stack, cross-entropy loss, AdamW with gradient clipping, cosine LR warmup/decay, training loop, checkpoint save/load/resume, and the evaluation suite (val loss, perplexity, next-token accuracy, sampling generation).

A one-paragraph overfit test drives loss below 0.05 with exact token-level memorization, proving the autograd + training stack learns end-to-end.

**Also complete and tested:** instruction fine-tuning with loss masking (`chat.py` + `instruct.py`), a Claude-Code/Codex-style tool suite (`tools.py`; file tools path-jailed to a sandbox root, `bash`/`python_eval` gated behind flags and run on the host without OS isolation), synthetic tool-call datasets (`tooldata.py`), a KV-cached inference path proven token-identical to full recompute (`kvcache.py`), the tool-calling runtime (`agent.py`), inference bundles (`bundle.py`), a FastAPI serving API (`serve/`), a `Dockerfile`, and a NumPy/CuPy backend switch (`backend.py`). Heavy pretraining/fine-tuning runs are config-driven scripts intended for GPU hosts (e.g. RunPod); the Mac path verifies correctness via tiny synthetic-overfit and unit tests.

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

153 tests covering the autograd engine, gradient checks, tokenizer, batching, every model layer, the full GPT stack, loss + masking, optimizer, LR schedule, gradient clipping, training loop, checkpointing, evaluation, sampling, chat templating, instruction masking, the sandboxed tool suite, synthetic tool datasets, KV-cache equivalence, the tool-calling agent, inference bundles, the FastAPI endpoints, the micro-batcher, and the backend switch. The serving tests are skipped automatically if `fastapi`/`httpx` are not installed.

## Serving

```bash
pip install -r requirements-serve.txt
MODEL_DIR=path/to/bundle uvicorn serve.app:app --host 0.0.0.0 --port 8000
python scripts/smoke_test.py --url http://localhost:8000
```

Or with Docker:

```bash
docker build -t zyn-lm .
docker run -p 8000:8000 -e MODEL_DIR=/app/bundle -v $(pwd)/bundle:/app/bundle zyn-lm
```

Endpoints: `GET /health`, `POST /generate`, `POST /chat` (set `"tools": true` to enable the tool loop. File tools are path-jailed to `SANDBOX_DIR`; `write_file`/`edit_file` need `ALLOW_WRITE`, and `bash`/`python_eval` need `ALLOW_SHELL` — the latter run on the host with full filesystem/network access and no OS isolation, so only enable them in a disposable environment).

## Training and fine-tuning (GPU hosts)

```bash
python scripts/pretrain.py --config configs/small-cpu.json
python scripts/finetune_instruct.py --checkpoint checkpoints/pretrain-small.npz
python scripts/build_tool_dataset.py --n 2000
python scripts/finetune_tools.py --checkpoint checkpoints/instruct.npz
```

## RunPod A6000 quickstart

Everything (code + tokenized corpus + tokenizer) is mirrored on the Hugging Face hub, so one clone is self-contained. On a fresh RunPod PyTorch pod (1× RTX A6000, CUDA 12):

```bash
git lfs install
git clone https://huggingface.co/datasets/karthik-2905/core-language-model-data
cd core-language-model-data
bash scripts/setup_pod.sh          # installs deps + cupy (data already present)
ZYN_BACKEND=cuda ZYN_DTYPE=float32 CUPY_TF32=1 \
  python scripts/pretrain.py --config configs/a6000.json
```

The `.npy` token files are Git-LFS pointers — `git lfs install` before cloning pulls them automatically. The mirror also lives on GitHub at `GruheshKurra/core-language-model` (code only); `scripts/setup_pod.sh` fetches the data from HF if it's missing, so either clone works.

`configs/a6000.json` uses `batch_size=128`, `seq=256`, `12000` steps (~2.2 epochs over the 180M-token codeparrot shard). `CUPY_TF32=1` enables Ampere tensor-core matmuls; `mmap=false` loads the token stream fully into RAM (the pod has plenty). Checkpoints land in `checkpoints/pretrain-a6000.npz` every 500 steps — resume with `--resume checkpoints/pretrain-a6000.npz`.

Run ~50 steps first, check `nvidia-smi` VRAM headroom and step time, then raise `batch_size` if there's room before committing to the full run.

## Scale

Targeting ~1–10M parameters on CPU for the initial build. GPU scaling uses the `zyn/backend.py` switch (`ZYN_BACKEND=cuda` with CuPy installed) and the `configs/medium-gpu.json` preset; run the heavy jobs on a GPU host (e.g. RunPod) once the stack is verified correct on CPU.

The compute dtype is set by `ZYN_DTYPE` (default `float64`). Gradient checks and overfit tests rely on `float64`; on GPU set `ZYN_DTYPE=float32` for usable throughput, since float64 runs at roughly 1/32 the rate of float32 on consumer NVIDIA cards.
