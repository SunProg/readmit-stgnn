#!/bin/bash
#SBATCH --job-name=stgnn-ehr-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -euo pipefail

# Usage:
#   sbatch slurm/train_ehr_stgnn.sbatch
#   sbatch --export=ALL,PREPROC_DIR=$HOME/repo/readmit-stgnn/stgnn/data/mimic_processed,SAVE_DIR=$HOME/repo/readmit-stgnn/stgnn/data/ckp/ehr_stgnn slurm/train_ehr_stgnn.sbatch
#
# Optional:
#   PYTHON_BIN=/path/to/python
#   LOG_DIR=/path/to/logs
#   DEMO_FILE=/path/to/mimic_admission_demo.csv
#   EHR_FEATURE_FILE=/path/to/ehr_preprocessed_seq_by_day.pkl
#   EDGE_EHR_FILE=/path/to/ehr_preprocessed_seq_by_day.pkl
#   TEST_HADM_IDS_FILE=/path/to/note_only_test_hadm_subject_ids.csv
#   EHR_ENCODER_NAME=embedder  # only when EHR_FEATURE_FILE has categorical cols

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PREPROC_DIR="${PREPROC_DIR:-$REPO_DIR/stgnn/data/mimic_processed}"
SAVE_DIR="${SAVE_DIR:-$REPO_DIR/stgnn/data/ckp/ehr_stgnn}"
DEMO_FILE="${DEMO_FILE:-$PREPROC_DIR/mimic_admission_demo.csv}"
EHR_FEATURE_FILE="${EHR_FEATURE_FILE:-$PREPROC_DIR/ehr_preprocessed_seq_by_day.pkl}"
EDGE_EHR_FILE="${EDGE_EHR_FILE:-$PREPROC_DIR/ehr_preprocessed_seq_by_day.pkl}"
TEST_HADM_IDS_FILE="${TEST_HADM_IDS_FILE:-}"
EHR_ENCODER_NAME="${EHR_ENCODER_NAME:-}"
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

echo "Running EHR-based STGNN training..."
echo "REPO_DIR=$REPO_DIR"
echo "SAVE_DIR=$SAVE_DIR"
echo "DEMO_FILE=$DEMO_FILE"
echo "EHR_FEATURE_FILE=$EHR_FEATURE_FILE"
echo "EDGE_EHR_FILE=$EDGE_EHR_FILE"
echo "TEST_HADM_IDS_FILE=$TEST_HADM_IDS_FILE"
echo "EHR_ENCODER_NAME=$EHR_ENCODER_NAME"
echo "DGLBACKEND=$DGLBACKEND"
echo "DGL_DOWNLOAD_DIR=$DGL_DOWNLOAD_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

train_args=(
  stgnn/train.py
  --save_dir "$SAVE_DIR" \
  --demo_file "$DEMO_FILE" \
  --edge_modality demo \
  --feature_type non-imaging \
  --ehr_feature_file "$EHR_FEATURE_FILE" \
  --edge_ehr_file "$EDGE_EHR_FILE" \
  --ehr_types demo icd lab med \
  --edge_top_perc 0.01 \
  --sim_measure euclidean \
  --use_gauss_kernel True \
  --max_seq_len_ehr 9 \
  --hidden_dim 256 \
  --num_gcn_layers 1 \
  --num_rnn_layers 1 \
  --add_bias True \
  --g_conv graphsage \
  --aggregator_type mean \
  --num_classes 1 \
  --dropout 0.2 \
  --activation_fn elu \
  --metric_name auroc \
  --lr 3e-3 \
  --l2_wd 5e-4 \
  --patience 10 \
  --pos_weight 4 \
  --num_epochs 100 \
  --final_pool last \
  --model_name stgnn \
  --t_model gru \
  --cat_emb_dim 3
)

if [[ -n "$TEST_HADM_IDS_FILE" ]]; then
  train_args+=(--test_hadm_ids_file "$TEST_HADM_IDS_FILE")
fi

if [[ -n "$EHR_ENCODER_NAME" ]]; then
  train_args+=(--ehr_encoder_name "$EHR_ENCODER_NAME")
fi

"$PYTHON_BIN" "${train_args[@]}"

echo "Done."
