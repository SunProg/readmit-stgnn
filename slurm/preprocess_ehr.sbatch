#!/bin/bash
#SBATCH --job-name=stgnn-ehr-preproc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G

set -euo pipefail

# Usage:
#   sbatch slurm/preprocess_ehr.sbatch
#   sbatch --export=ALL,COHORT_DIR=$HOME/repo/readmit-stgnn/stgnn/data/mimic_processed,SAVE_DIR=$HOME/repo/readmit-stgnn/stgnn/data/mimic_processed slurm/preprocess_ehr.sbatch
#
# Optional:
#   PYTHON_BIN=/path/to/python
#   LOG_DIR=/path/to/logs

COHORT_DIR="${COHORT_DIR:-$HOME/repo/readmit-stgnn/stgnn/data/mimic_processed}"
SAVE_DIR="${SAVE_DIR:-$HOME/repo/readmit-stgnn/stgnn/data/mimic_processed}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/logs}"

mkdir -p "$LOG_DIR" "$SAVE_DIR"

echo "Running EHR preprocessing..."
echo "COHORT_DIR=$COHORT_DIR"
echo "SAVE_DIR=$SAVE_DIR"

"$PYTHON_BIN" stgnn/preprocess_ehr.py \
  --cohort_dir "$COHORT_DIR" \
  --save_dir "$SAVE_DIR"

echo "Done."
