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
#   EHR_FEATURE_FILE=/path/to/ehr_preprocessed_seq_by_day.pkl
#   EDGE_EHR_FILE=/path/to/ehr_preprocessed_seq_by_day.pkl
#   TEST_HADM_IDS_FILE=/path/to/test_hadm_ids.csv
#   COMPILE_MODEL=True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PREPROC_DIR="${PREPROC_DIR:-$REPO_DIR/stgnn/data/mimic_processed}"
SAVE_DIR="${SAVE_DIR:-$REPO_DIR/stgnn/data/ckp/ehr_lstm}"
DEMO_FILE="${DEMO_FILE:-$PREPROC_DIR/mimic_admission_demo.csv}"
EHR_FEATURE_FILE="${EHR_FEATURE_FILE:-$PREPROC_DIR/ehr_preprocessed_seq_by_day.pkl}"
EDGE_EHR_FILE="${EDGE_EHR_FILE:-$PREPROC_DIR/ehr_preprocessed_seq_by_day.pkl}"
TEST_HADM_IDS_FILE="${TEST_HADM_IDS_FILE:-}"
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
echo "COMPILE_MODEL=$COMPILE_MODEL"
echo "DGLBACKEND=$DGLBACKEND"
echo "DGL_DOWNLOAD_DIR=$DGL_DOWNLOAD_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

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
  --train_batch_size 128
  --hidden_dim 256
  --num_rnn_layers 3
  --dropout 0.2
  --metric_name auroc
  --lr 0.001
  --l2_wd 5e-4
  --patience 10
  --pos_weight 1
  --num_epochs 100
  --model_name lstm
  --compile_model "$COMPILE_MODEL"
)

if [[ -n "$TEST_HADM_IDS_FILE" ]]; then
  train_args+=(--test_hadm_ids_file "$TEST_HADM_IDS_FILE")
fi

"$PYTHON_BIN" "${train_args[@]}"

echo "Done."
