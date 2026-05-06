#!/bin/bash
#SBATCH --job-name=lstm-ehr-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -euo pipefail

# Usage:
#   sbatch slurm/train_ehr_lstm.sbatch
#   sbatch --export=ALL,PREPROC_DIR=$HOME/repo/readmit-stgnn/stgnn/data/mimic_processed,SAVE_DIR=$HOME/repo/readmit-stgnn/stgnn/data/ckp/ehr_lstm slurm/train_ehr_lstm.sbatch
#
# Optional:
#   PYTHON_BIN=/path/to/python
#   LOG_DIR=/path/to/logs
#   DEMO_FILE=/path/to/mimic_admission_demo.csv
#   EHR_FEATURE_FILE=/path/to/ehr_preprocessed_seq_by_day_cat_embedding.pkl
#   EDGE_EHR_FILE=/path/to/ehr_preprocessed_seq_by_day_one_hot.pkl
#   TEST_HADM_IDS_FILE=/path/to/test_hadm_ids.csv
#   EHR_ENCODER_NAME=embedder
#   HPARAM_SEARCH=False
#   COMPILE_MODEL=True

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
SAVE_DIR="${SAVE_DIR:-$REPO_DIR/stgnn/data/ckp/ehr_lstm}"
DEMO_FILE="${DEMO_FILE:-$PREPROC_DIR/mimic_admission_demo.csv}"
EHR_FEATURE_FILE="${EHR_FEATURE_FILE:-$PREPROC_DIR/ehr_preprocessed_seq_by_day_cat_embedding.pkl}"
EDGE_EHR_FILE="${EDGE_EHR_FILE:-$PREPROC_DIR/ehr_preprocessed_seq_by_day_one_hot.pkl}"
TEST_HADM_IDS_FILE="${TEST_HADM_IDS_FILE:-}"
EHR_ENCODER_NAME="${EHR_ENCODER_NAME:-embedder}"
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

echo "Running EHR-based LSTM training..."
echo "REPO_DIR=$REPO_DIR"
echo "SAVE_DIR=$SAVE_DIR"
echo "DEMO_FILE=$DEMO_FILE"
echo "EHR_FEATURE_FILE=$EHR_FEATURE_FILE"
echo "EDGE_EHR_FILE=$EDGE_EHR_FILE"
echo "TEST_HADM_IDS_FILE=$TEST_HADM_IDS_FILE"
echo "EHR_ENCODER_NAME=$EHR_ENCODER_NAME"
echo "HPARAM_SEARCH=$HPARAM_SEARCH"
echo "COMPILE_MODEL=$COMPILE_MODEL"
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
  --model_name lstm
  --require_cuda True
  --t_model gru
  --ehr_encoder_name "$EHR_ENCODER_NAME"
  --cat_emb_dim 3
  --hparam_search "$HPARAM_SEARCH"
  --compile_model "$COMPILE_MODEL"
)

if [[ -n "$TEST_HADM_IDS_FILE" ]]; then
  train_args+=(--test_hadm_ids_file "$TEST_HADM_IDS_FILE")
fi

"$PYTHON_BIN" "${train_args[@]}"

echo "Done."
