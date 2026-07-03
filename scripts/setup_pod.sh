#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt
pip install "cupy-cuda12x" "huggingface_hub[hf_transfer]"

if [ ! -f data/processed/tokens/codeparrot-train.npy ]; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  python scripts/download_data.py --repo "${DATA_REPO:-karthik-2905/core-language-model-data}"
else
  echo "data already present, skipping download"
fi

echo "ready. start pretraining with:"
echo "  ZYN_BACKEND=cuda ZYN_DTYPE=float32 CUPY_TF32=1 python scripts/pretrain.py --config configs/a6000.json"
