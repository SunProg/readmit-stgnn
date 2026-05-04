#!/bin/bash
#SBATCH --job-name=stgnn-cohort
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G

set -euo pipefail

# Usage:
#   sbatch slurm/get_mimic_cohort.sbatch
#   sbatch --export=ALL,RAW_DATA_DIR=$HOME/database/mimic-iv/mimiciv/2.2,SAVE_DIR=$HOME/repo/readmit-stgnn/stgnn/data/mimic_processed,SPLIT_JSON=$HOME/repo/ehr-text-multi-modal/results/mimiciv/splits/ethos_train_val_patients__mimic_train_timelines_p241015__vf-0p04.json,TEST_PARQUET=$HOME/repo/ehr-text-multi-modal/results/mimiciv/ethos_zero_shot/test_eval/hospital_readmission/output_with_note.predictions.parquet slurm/get_mimic_cohort.sbatch
#
# Optional:
#   PYTHON_BIN=/path/to/python

RAW_DATA_DIR="${RAW_DATA_DIR:-$HOME/database/mimic-iv/mimiciv/2.2}"
SAVE_DIR="${SAVE_DIR:-$HOME/repo/readmit-stgnn/stgnn/data/mimic_processed}"
SPLIT_JSON="${SPLIT_JSON:-$HOME/repo/ehr-text-multi-modal/results/mimiciv/splits/ethos_train_val_patients__mimic_train_timelines_p241015__vf-0p04.json}"
TEST_PARQUET="${TEST_PARQUET:-$HOME/repo/ehr-text-multi-modal/results/mimiciv/ethos_zero_shot/test_eval/hospital_readmission/output_with_note.predictions.parquet}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$(dirname "$0")/../logs" "$SAVE_DIR"

echo "Running cohort extraction..."
echo "RAW_DATA_DIR=$RAW_DATA_DIR"
echo "SAVE_DIR=$SAVE_DIR"
echo "SPLIT_JSON=$SPLIT_JSON"
echo "TEST_PARQUET=$TEST_PARQUET"

"$PYTHON_BIN" stgnn/get_mimic_cohort.py \
  --raw_data_dir "$RAW_DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --split_json "$SPLIT_JSON" \
  --test_parquet "$TEST_PARQUET"

echo "Done."
