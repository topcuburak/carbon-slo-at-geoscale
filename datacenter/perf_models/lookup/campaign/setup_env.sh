#!/usr/bin/env bash
# =============================================================================
#  drain campaign environment setup
# =============================================================================
#
# Creates a clean `llm_bench` conda env with pinned vLLM + PyTorch for
# CUDA 12.1 wheels (compatible with NVIDIA driver 590+). Downloads the
# LLaMA-3.1-8B-Instruct weights via HF token.
#
# Usage (run on the 4×A100 box):
#
#   export HF_TOKEN=hf_xxx...           # must be exported before running
#   bash datacenter/perf_models/lookup/campaign/setup_env.sh
#
# After it finishes:
#
#   conda activate llm_bench
#   python datacenter/perf_models/lookup/campaign/run_pilot.py
#
# =============================================================================

set -euo pipefail

ENV_NAME="${ENV_NAME:-llm_bench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VLLM_VERSION="${VLLM_VERSION:-0.6.3.post1}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
CUDA_CHANNEL="${CUDA_CHANNEL:-cu121}"
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running:"
    echo "  export HF_TOKEN=hf_xxx..."
    exit 1
fi

echo "============================================================"
echo "  Setting up drain campaign environment"
echo "  env:    $ENV_NAME"
echo "  python: $PYTHON_VERSION"
echo "  vllm:   $VLLM_VERSION"
echo "  torch:  $TORCH_VERSION+$CUDA_CHANNEL"
echo "  model:  $MODEL_ID"
echo "============================================================"

# Source conda
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "ERROR: could not find conda.sh. Adjust this script to your conda path."
    exit 1
fi

# Recreate env if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing $ENV_NAME env..."
    conda env remove -n "$ENV_NAME" -y
fi

conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME"

# Install PyTorch from the CUDA 12.1 index (works with CUDA 13.1 driver)
pip install --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision" \
    "torchaudio"

# vLLM and friends. vllm 0.6.3 pulls its own compatible transformers/xformers.
pip install "vllm==${VLLM_VERSION}"

# Pin transformers to a version vLLM 0.6.3 is known to work with.
# Later transformers versions (>=4.46) introduced TokenizersBackend changes
# that broke `all_special_tokens_extended` access in vLLM 0.6.3.
pip install "transformers==4.45.2"

# Extras for data loading / analysis
pip install pandas numpy matplotlib huggingface-hub

# Huggingface login via token
python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"

# Pre-download the model so the pilot doesn't spend time on it
python - <<PY
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id=os.environ.get("MODEL_ID_OVERRIDE", "$MODEL_ID"),
    token=os.environ["HF_TOKEN"],
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"],
)
print(f"Model downloaded to: {path}")
PY

# Quick smoke test that vllm imports cleanly and sees 4 GPUs
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

import vllm
print(f"vLLM: {vllm.__version__}")
PY

echo ""
echo "============================================================"
echo "  Setup complete. To activate:"
echo "     conda activate $ENV_NAME"
echo "  Then run the pilot:"
echo "     python datacenter/perf_models/lookup/campaign/run_pilot.py"
echo "============================================================"
