# Copilot Instructions for `readmit-stgnn`

## Build, test, and lint commands

This repository is Python-based and packaged with `setuptools` (`pyproject.toml`).

```bash
# install dependencies from project metadata
pip install -e .

# install pinned dependencies used by this repo
pip install -r requirements.txt
```

There are currently **no configured lint or automated test commands** in this repository (no `tests/`, `pytest` config, or lint config files).

For focused validation of training code paths, use a short-run training invocation:

```bash
python stgnn/train.py \
  --save_dir <save-dir> \
  --demo_file <cohort-dir>/mimic_admission_demo.csv \
  --edge_modality demo \
  --feature_type non-imaging \
  --ehr_feature_file <ehr-dir>/ehr_preprocessed_seq_by_day.pkl \
  --edge_ehr_file <ehr-dir>/ehr_preprocessed_seq_by_day.pkl \
  --ehr_types demo icd lab med \
  --model_name rnn \
  --num_epochs 1
```

## High-level architecture

The codebase is organized around a 3-stage pipeline:

1. `stgnn/get_mimic_cohort.py` builds cohort-level CSV artifacts from MIMIC-IV (`mimic_admission_demo.csv`, plus filtered ICD/lab/med files).
2. `stgnn/preprocess_ehr.py` expands the cohort into day-level rows, joins ICD/lab/med features, and writes sequence features (`ehr_preprocessed_seq_by_day.pkl`).
3. `stgnn/train.py` loads `ReadmissionDataset` (`stgnn/data/dataset.py`), builds model(s) from `stgnn/model/*`, trains with BCE-with-logits, and evaluates on val/test masks.

Model code is split by role:
- `stgnn/model/model.py`: graph-temporal components (`GraphRNN`, `GConvLayers`, custom GraphSAGE-backed GRU cell).
- `stgnn/model/simple_rnn.py` and `simple_lstm.py`: non-graph temporal baselines.
- `stgnn/model/fusion.py`: multimodal fusion model.
- `stgnn/gnn_explainer.py`: post-hoc node-level explainability path.

## Key conventions specific to this repo

- **Artifact filename contract is strict across stages.**  
  Downstream scripts expect exact names from upstream preprocessing (`mimic_admission_demo.csv`, `mimic_hosp_icd_subgroups.csv`, `mimic_hosp_lab_filtered.csv`, `mimic_hosp_med_filtered.csv`, `ehr_preprocessed_seq_by_day.pkl`).

- **Node identity convention is `subject_id_hadm_id`.**  
  This key is created in preprocessing and reused throughout dataset/model assembly.

- **Train/val/test are data columns, not runtime split logic.**  
  Splits are read from the `splits` column and converted into graph node masks (`train_mask`, `val_mask`, `test_mask`).

- **Boolean CLI args are string-parsed.**  
  `args.py` uses a custom `str2bool` parser that treats only `"true"`/`"True"` as true; anything else becomes false.

- **Run outputs are auto-namespaced under `save_dir`.**  
  `utils.get_save_dir()` always creates `train/train-XX` or `test/test-XX` subdirectories under the provided base path.

- **Current training path uses non-graph mode in `train.py`.**  
  `ReadmissionDataset(..., is_graph=False)` is hardcoded in `train.py`, so the default training loop operates on node feature sequences (dummy graph object still carries masks/labels/features).
