#!/bin/bash
set -euo pipefail

# Rebuild the project venv for the CUDA module stack used by the GPU Slurm jobs.
# Run from the repository root on the cluster login/compute environment.

CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0-gcc-13.2.0}"
CUDNN_MODULE="${CUDNN_MODULE:-cudnn/8.7.0.84-11.8-gcc-13.2.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi

if type module >/dev/null 2>&1; then
  module load "$CUDA_MODULE"
  module load "$CUDNN_MODULE"
  module list
else
  echo "Environment modules are not available; continuing without module loads."
fi

uv venv --python "$PYTHON_VERSION"

# Install PyTorch and DGL from CUDA 11.8 wheel indexes before the editable package.
uv pip install --reinstall \
  torch==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu118

uv pip install --reinstall \
  dgl==1.1.3 \
  -f https://data.dgl.ai/wheels/cu118/repo.html

uv pip install -e .

.venv/bin/python stgnn/cuda_check.py
