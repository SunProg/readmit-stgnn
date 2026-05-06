# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python package for MIMIC-IV readmission modeling. Core code lives in `stgnn/`: data loading and preprocessing are in `stgnn/data/`, model definitions are in `stgnn/model/`, and executable pipeline scripts include `stgnn/get_mimic_cohort.py`, `stgnn/preprocess_ehr.py`, `stgnn/train.py`, and `stgnn/gnn_explainer.py`. MIMIC-derived sample or generated artifacts are under `stgnn/data/`; keep large private datasets and checkpoints out of commits. Slurm entry points are in `slurm/`. `run_train.sh` is a local example training command.

## Build, Test, and Development Commands

- `pip install -e .`: install the package from `pyproject.toml` for local development.
- `pip install -r requirements.txt`: install the pinned dependency set when reproducing older runs.
- `python stgnn/get_mimic_cohort.py --raw_data_dir <mimic> --cxr_data_dir <cxr> --save_dir <out>`: build cohort CSVs.
- `python stgnn/preprocess_ehr.py --cohort_dir <cohort> --save_dir <out>`: generate EHR sequence features.
- `bash run_train.sh`: run the included non-imaging training example.
- `sbatch slurm/get_mimic_cohort.sbatch` and `sbatch slurm/preprocess_ehr.sbatch`: run preprocessing on Slurm; pass paths via `--export` as shown in `README.md`.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation and descriptive snake_case names for functions, variables, files, and CLI flags. Keep model classes in PascalCase. Preserve the repository's artifact filename contracts: downstream code expects names such as `mimic_admission_demo.csv`, `mimic_hosp_icd_subgroups.csv`, and `ehr_preprocessed_seq_by_day.pkl`. Node identifiers follow the `subject_id_hadm_id` convention. Prefer small, local changes over broad refactors.

## Testing Guidelines

There is currently no `tests/` directory or configured pytest/lint setup. When changing behavior, add focused tests under `tests/` using `test_*.py` names, or document a reproducible validation command. For training changes, run a one-epoch smoke test with representative preprocessed files and verify that train/val/test masks and output directories are created correctly.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, for example `Fix cohort data file path resolution` and `Reduce preprocess parallelism to avoid stalls`. Follow that style: one logical change per commit, with details in the body when needed. Pull requests should describe the pipeline stage affected, list validation commands or Slurm jobs run, mention data assumptions, and note any changes to expected artifact filenames or CLI arguments.

## Security & Configuration Tips

Do not commit protected MIMIC source data, credentials, generated checkpoints, or large intermediate files. Prefer path arguments and Slurm `--export` variables over hard-coded absolute paths. Check CUDA/DGL compatibility before changing dependency versions.
