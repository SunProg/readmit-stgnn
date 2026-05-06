#!/bin/bash -l
#SBATCH --job-name=rnn-ehr-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -euo pipefail

# Usage:
#   sbatch slurm/train_ehr_rnn.sbatch
#   sbatch --export=ALL,PREPROC_DIR=$HOME/repo/readmit-stgnn/stgnn/data/mimic_processed,SAVE_DIR=$HOME/repo/readmit-stgnn/stgnn/data/ckp/ehr_rnn slurm/train_ehr_rnn.sbatch
#
# Optional:
#   PYTHON_BIN=/path/to/python
#   LOG_DIR=/path/to/logs
#   DEMO_FILE=/path/to/mimic_admission_demo.csv
#   EHR_FEATURE_FILE=/path/to/ehr_preprocessed_seq_by_day.pkl
#   EDGE_EHR_FILE=/path/to/ehr_preprocessed_seq_by_day.pkl
#   TEST_HADM_IDS_FILE=/path/to/test_hadm_ids.csv
#   EHR_ENCODER_NAME=embedder  # only when EHR_FEATURE_FILE has categorical cols
#   HPARAM_SEARCH=False
#   COMPILE_MODEL=True
#   CUDA_MODULE=cuda/11.8.0-gcc-13.2.0
#   CUDNN_MODULE=cudnn/8.7.0.84-11.8-gcc-13.2.0

CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0-gcc-13.2.0}"
CUDNN_MODULE="${CUDNN_MODULE:-cudnn/8.7.0.84-11.8-gcc-13.2.0}"
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # Some Slurm shells do not initialize environment modules by default.
  source /etc/profile.d/modules.sh
fi
if type module >/dev/null 2>&1; then
  module load "$CUDA_MODULE"
  module load "$CUDNN_MODULE"
else
  echo "Environment modules are not available; skipping CUDA/CUDNN module loads."
fi

if [[ -z "${REPO_DIR:-}" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; do
    if [[ -n "$candidate" && -f "$candidate/stgnn/train.py" ]]; then
      REPO_DIR="$candidate"
      break
    fi
  done
fi

if [[ -z "${REPO_DIR:-}" || ! -f "$REPO_DIR/stgnn/train.py" ]]; then
  echo "Could not locate repository root. Submit from the repo or export REPO_DIR=/path/to/readmit-stgnn." >&2
  exit 1
fi

PREPROC_DIR="${PREPROC_DIR:-$REPO_DIR/stgnn/data/mimic_processed}"
SAVE_DIR="${SAVE_DIR:-$REPO_DIR/stgnn/data/ckp/ehr_rnn}"
DEMO_FILE="${DEMO_FILE:-$PREPROC_DIR/mimic_admission_demo.csv}"
CAT_EHR_FEATURE_FILE="$PREPROC_DIR/ehr_preprocessed_seq_by_day_cat_embedding.pkl"
DEFAULT_EHR_FEATURE_FILE="$PREPROC_DIR/ehr_preprocessed_seq_by_day.pkl"
ONE_HOT_EDGE_EHR_FILE="$PREPROC_DIR/ehr_preprocessed_seq_by_day_one_hot.pkl"
if [[ -z "${EHR_FEATURE_FILE:-}" ]]; then
  if [[ -f "$CAT_EHR_FEATURE_FILE" ]]; then
    EHR_FEATURE_FILE="$CAT_EHR_FEATURE_FILE"
    EHR_FEATURE_SELECTION="auto-selected categorical embedding features"
  else
    EHR_FEATURE_FILE="$DEFAULT_EHR_FEATURE_FILE"
    EHR_FEATURE_SELECTION="auto-selected standard preprocessed features"
  fi
else
  EHR_FEATURE_SELECTION="explicit EHR_FEATURE_FILE"
fi
if [[ -z "${EDGE_EHR_FILE:-}" ]]; then
  if [[ -f "$ONE_HOT_EDGE_EHR_FILE" ]]; then
    EDGE_EHR_FILE="$ONE_HOT_EDGE_EHR_FILE"
    EDGE_EHR_SELECTION="auto-selected one-hot edge features"
  else
    EDGE_EHR_FILE="$DEFAULT_EHR_FEATURE_FILE"
    EDGE_EHR_SELECTION="auto-selected standard preprocessed edge features"
  fi
else
  EDGE_EHR_SELECTION="explicit EDGE_EHR_FILE"
fi
TEST_HADM_IDS_FILE="${TEST_HADM_IDS_FILE:-}"
if [[ -z "${EHR_ENCODER_NAME+x}" ]]; then
  if [[ "$EHR_FEATURE_FILE" == "$CAT_EHR_FEATURE_FILE" ]]; then
    EHR_ENCODER_NAME="embedder"
  else
    EHR_ENCODER_NAME=""
  fi
fi
HPARAM_SEARCH="${HPARAM_SEARCH:-True}"
COMPILE_MODEL="${COMPILE_MODEL:-False}"
if [[ -z "${PYTHON_BIN:-}" && -x "$REPO_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
LOG_DIR="${LOG_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/logs}"
DGL_DOWNLOAD_DIR="${DGL_DOWNLOAD_DIR:-$SAVE_DIR/dgl}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export DGL_DOWNLOAD_DIR

mkdir -p "$LOG_DIR" "$SAVE_DIR" "$DGL_DOWNLOAD_DIR"
cd "$REPO_DIR"

if type module >/dev/null 2>&1; then
  module list
fi

echo "EHR feature selection: $EHR_FEATURE_SELECTION -> $EHR_FEATURE_FILE"
echo "Edge EHR feature selection: $EDGE_EHR_SELECTION -> $EDGE_EHR_FILE"
if [[ ! -f "$DEMO_FILE" ]]; then
  echo "DEMO_FILE does not exist: $DEMO_FILE" >&2
  exit 1
fi
if [[ ! -f "$EHR_FEATURE_FILE" ]]; then
  echo "EHR_FEATURE_FILE does not exist: $EHR_FEATURE_FILE" >&2
  echo "Expected one of: $CAT_EHR_FEATURE_FILE or $DEFAULT_EHR_FEATURE_FILE" >&2
  exit 1
fi
if [[ ! -f "$EDGE_EHR_FILE" ]]; then
  echo "EDGE_EHR_FILE does not exist: $EDGE_EHR_FILE" >&2
  echo "Expected one of: $ONE_HOT_EDGE_EHR_FILE or $DEFAULT_EHR_FEATURE_FILE" >&2
  exit 1
fi

echo "Running EHR-based RNN training..."
echo "REPO_DIR=$REPO_DIR"
echo "SAVE_DIR=$SAVE_DIR"
echo "DEMO_FILE=$DEMO_FILE"
echo "EHR_FEATURE_FILE=$EHR_FEATURE_FILE"
echo "EDGE_EHR_FILE=$EDGE_EHR_FILE"
echo "TEST_HADM_IDS_FILE=$TEST_HADM_IDS_FILE"
echo "EHR_ENCODER_NAME=$EHR_ENCODER_NAME"
echo "HPARAM_SEARCH=$HPARAM_SEARCH"
echo "COMPILE_MODEL=$COMPILE_MODEL"
echo "CUDA_MODULE=$CUDA_MODULE"
echo "CUDNN_MODULE=$CUDNN_MODULE"
echo "DGLBACKEND=$DGLBACKEND"
echo "DGL_DOWNLOAD_DIR=$DGL_DOWNLOAD_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi
"$PYTHON_BIN" stgnn/cuda_check.py

train_args=(
  stgnn/train.py
  --save_dir "$SAVE_DIR"
  --demo_file "$DEMO_FILE"
  --edge_modality demo
  --feature_type non-imaging
  --ehr_feature_file "$EHR_FEATURE_FILE"
  --edge_ehr_file "$EDGE_EHR_FILE"
  --ehr_types demo icd lab med
  --edge_top_perc 0.01
  --sim_measure euclidean
  --use_gauss_kernel True
  --max_seq_len_ehr 9
  --hidden_dim 256
  --num_gcn_layers 1
  --num_rnn_layers 1
  --add_bias True
  --g_conv graphsage
  --aggregator_type mean
  --num_classes 1
  --dropout 0.2
  --activation_fn elu
  --metric_name auroc
  --lr 3e-3
  --l2_wd 5e-4
  --patience 10
  --pos_weight 4
  --num_epochs 100
  --final_pool last
  --model_name rnn
  --require_cuda True
  --t_model gru
  --cat_emb_dim 3
  --hparam_search "$HPARAM_SEARCH"
  --compile_model "$COMPILE_MODEL"
)

if [[ -n "$TEST_HADM_IDS_FILE" ]]; then
  train_args+=(--test_hadm_ids_file "$TEST_HADM_IDS_FILE")
fi

if [[ -n "$EHR_ENCODER_NAME" ]]; then
  train_args+=(--ehr_encoder_name "$EHR_ENCODER_NAME")
fi

"$PYTHON_BIN" "${train_args[@]}"

echo "Done."
