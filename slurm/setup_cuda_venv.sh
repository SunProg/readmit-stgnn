#!/bin/bash
set -euo pipefail

# Rebuild the project venv for the CUDA module stack used by the GPU Slurm jobs.
# Run from the repository root on the cluster login/compute environment.

CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0-gcc-13.2.0}"
CUDNN_MODULE="${CUDNN_MODULE:-cudnn/8.7.0.84-11.8-gcc-13.2.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.2.1}"
DGL_VERSION="${DGL_VERSION:-1.1.3}"
CUDA_WHEEL_TAG="${CUDA_WHEEL_TAG:-cu118}"
EXPECTED_TORCH_CUDA_VERSION="${EXPECTED_TORCH_CUDA_VERSION:-11.8}"
NUMPY_CONSTRAINT="${NUMPY_CONSTRAINT:-numpy<2}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/$CUDA_WHEEL_TAG}"
DGL_WHEEL_URL="${DGL_WHEEL_URL:-https://data.dgl.ai/wheels/$CUDA_WHEEL_TAG/repo.html}"

if [[ ! -f stgnn/train.py || ! -f pyproject.toml ]]; then
  echo "Run this script from the readmit-stgnn repository root." >&2
  exit 1
fi

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

echo "Recreating .venv with Python $PYTHON_VERSION for CUDA wheel tag $CUDA_WHEEL_TAG"
rm -rf .venv
uv venv .venv --python "$PYTHON_VERSION"

# Install PyTorch and DGL from CUDA 11.8 wheel indexes before the editable package.
uv pip install --reinstall \
  "torch==$TORCH_VERSION" \
  --index-url "$PYTORCH_INDEX_URL"

uv pip install --reinstall \
  "dgl==$DGL_VERSION+$CUDA_WHEEL_TAG" \
  -f "$DGL_WHEEL_URL"

uv pip install -e . "$NUMPY_CONSTRAINT"

.venv/bin/python - <<PY
import sys

import dgl
import numpy
import torch

errors = []
if int(numpy.__version__.split(".", 1)[0]) >= 2:
    errors.append("expected NumPy <2, got {}".format(numpy.__version__))
if "+$CUDA_WHEEL_TAG" not in torch.__version__:
    errors.append("expected torch.__version__ to include '+$CUDA_WHEEL_TAG'")
if torch.version.cuda != "$EXPECTED_TORCH_CUDA_VERSION":
    errors.append(
        "expected torch.version.cuda to be $EXPECTED_TORCH_CUDA_VERSION, got {}".format(
            torch.version.cuda
        )
    )
print("python_executable={}".format(sys.executable))
print("numpy.__version__={}".format(numpy.__version__))
print("torch.__version__={}".format(torch.__version__))
print("torch.version.cuda={}".format(torch.version.cuda))
print("dgl.__version__={}".format(dgl.__version__))

if errors:
    raise SystemExit("CUDA wheel verification failed: " + "; ".join(errors))
PY

.venv/bin/python stgnn/cuda_check.py
