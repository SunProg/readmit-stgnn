"""Collect test metrics for readmit-stgnn models from the CREATE cluster.

Downloads EHR-LSTM results via rsync, then scans ehr_lstm / ehr_rnn / ehr_stgnn
checkpoint dirs. For any completed run missing test results or AUPRC, it
re-runs test evaluation via stgnn/train.py. Results are saved to a CSV file.

Usage:
    python scripts/collect_test_metrics.py               # rsync + collect
    python scripts/collect_test_metrics.py --skip-rsync  # local cache only
    python scripts/collect_test_metrics.py --dry-run     # preview rsync, no changes
    python scripts/collect_test_metrics.py --no-run-test # skip auto test evaluation
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import pandas as pd

CLUSTER_HOST = "create"
REMOTE_USER = "k23139234"
REMOTE_REPO = f"/users/{REMOTE_USER}/repo/readmit-stgnn"

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CKP = REPO_ROOT / "stgnn" / "data" / "ckp"
OUTPUT_CSV = REPO_ROOT / "results" / "test_metrics_summary.csv"

MODEL_CKP_DIRS = {
    "ehr_lstm": LOCAL_CKP / "ehr_lstm",
    "ehr_rnn": LOCAL_CKP / "ehr_rnn",
    "ehr_stgnn": LOCAL_CKP / "ehr_stgnn",
}

# Only ehr_lstm is downloaded from CREATE (the job that was run there)
REMOTE_SYNC_MODELS = ["ehr_lstm"]

METRIC_COLS = [
    "auroc", "auprc", "F1", "precision", "recall",
    "specificity", "acc", "auroc_ci_lower", "auroc_ci_upper",
    "auprc_ci_lower", "auprc_ci_upper", "loss",
]


def rsync_from_cluster(model_name: str, dry_run: bool = False) -> None:
    remote = f"{CLUSTER_HOST}:{REMOTE_REPO}/stgnn/data/ckp/{model_name}/"
    local = MODEL_CKP_DIRS[model_name]
    local.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-av", "--progress"]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [remote, str(local) + "/"]
    print(f"rsyncing {remote} → {local}")
    subprocess.run(cmd, check=True)


def _resolve_path(p: str | Path) -> Path:
    """Resolve a path that may be absolute, relative to REPO_ROOT, or an
    absolute path from a remote cluster (converted to local by finding the
    repo root segment in the path)."""
    path = Path(p)
    if not path.is_absolute():
        return REPO_ROOT / path
    if path.exists():
        return path
    # Remote absolute path: find the repo directory name segment and remap
    repo_name = REPO_ROOT.name
    parts = path.parts
    for i, part in enumerate(parts):
        if part == repo_name:
            relative = Path(*parts[i + 1:])
            return REPO_ROOT / relative
    return path


def get_effective_dir(run_dir: Path) -> tuple[Path, Path | None]:
    """Return (effective_dir, best_ckpt) for a training run dir.

    For hparam search runs (have best_hparams.json), effective_dir is the best
    trial subdir. For regular runs, effective_dir is run_dir itself. Returns
    (effective_dir, None) when the run has no completed checkpoint.
    """
    best_hparams_file = run_dir / "best_hparams.json"
    if best_hparams_file.exists():
        with open(best_hparams_file) as f:
            best_hparams = json.load(f)
        trial_dir = _resolve_path(best_hparams["save_dir"])
        ckpt = trial_dir / "best.pth.tar"
        if ckpt.exists():
            return trial_dir, ckpt
        return trial_dir, None

    ckpt = run_dir / "best.pth.tar"
    if ckpt.exists():
        return run_dir, ckpt
    return run_dir, None


def find_test_results(effective_dir: Path) -> dict | None:
    """Return the test metrics dict from test_predictions.pkl, or None."""
    # Directly in dir (STGNN-style, run with --resume_run_dir)
    pkl = effective_dir / "test_predictions.pkl"
    if pkl.exists():
        return _load_pkl_results(pkl)

    # In test/test-XX subdir (LSTM-style)
    test_parent = effective_dir / "test"
    if test_parent.exists():
        test_runs = sorted(
            test_parent.glob("test-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        for t in reversed(test_runs):
            pkl = t / "test_predictions.pkl"
            if pkl.exists():
                return _load_pkl_results(pkl)

    return None


def _load_pkl_results(pkl_path: Path) -> dict | None:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data.get("results")


def run_test_evaluation(best_ckpt: Path, effective_dir: Path) -> dict | None:
    """Invoke stgnn/train.py with --do_train False to compute test metrics."""
    args_file = effective_dir / "args.json"
    if not args_file.exists():
        print(f"    No args.json in {effective_dir}, skipping.")
        return None

    with open(args_file) as f:
        train_args = json.load(f)

    cmd = ["uv", "run", "python", "stgnn/train.py"]

    # Pass through all training args, skipping keys we'll override
    skip_keys = {
        "save_dir", "load_model_path", "do_train",
        "require_cuda", "hparam_search", "resume_run_dir",
        "cuda", "maximize_metric",
    }
    for key, val in train_args.items():
        if key in skip_keys:
            continue
        if isinstance(val, bool):
            cmd += [f"--{key}", str(val)]
        elif isinstance(val, list):
            if val:
                cmd += [f"--{key}"] + [str(v) for v in val]
        elif val is not None:
            cmd += [f"--{key}", str(val)]

    cmd += [
        "--do_train", "False",
        "--hparam_search", "False",
        "--load_model_path", str(best_ckpt),
        "--save_dir", str(effective_dir),
        "--require_cuda", "False",
        "--thresh_search", "True",
    ]

    print(f"    Running: python stgnn/train.py --do_train False --load_model_path {best_ckpt.name} ...")
    # Snapshot existing pkls so we can detect the newly written one after subprocess
    existing_pkls = {p: p.stat().st_mtime for p in effective_dir.rglob("test_predictions.pkl") if p.exists()}
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"    Test evaluation failed (exit {result.returncode}).")
        return None

    # Prefer the pkl that was created/updated by the subprocess
    new_pkls = [
        p for p in effective_dir.rglob("test_predictions.pkl")
        if p.stat().st_mtime > existing_pkls.get(p, 0)
    ]
    if new_pkls:
        newest = max(new_pkls, key=lambda p: p.stat().st_mtime)
        return _load_pkl_results(newest)
    return find_test_results(effective_dir)


def _to_python(obj):
    """Recursively convert numpy scalars to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    try:
        return obj.item()  # numpy scalar → Python scalar
    except AttributeError:
        return obj


def _write_results_summary(effective_dir: Path, test_results: dict) -> None:
    """Write/update results_summary.json with val + test metrics."""
    summary_path = effective_dir / "results_summary.json"
    summary: dict = {}
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except json.JSONDecodeError:
            print(f"    Warning: {summary_path} is malformed — overwriting.")
            summary = {}

    summary["test"] = _to_python(test_results)

    val_pkl = effective_dir / "val_predictions.pkl"
    if val_pkl.exists() and "val" not in summary:
        with open(val_pkl, "rb") as f:
            val_data = pickle.load(f)
        if "results" in val_data:
            summary["val"] = _to_python(val_data["results"])

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4, sort_keys=True)


def collect_metrics(model_name: str, ckp_dir: Path, run_test: bool = True) -> list[dict]:
    rows: list[dict] = []
    train_dir = ckp_dir / "train"
    if not train_dir.exists():
        print(f"  No train/ dir at {ckp_dir}")
        return rows

    run_dirs = sorted(
        train_dir.glob("train-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )

    for run_dir in run_dirs:
        effective_dir, best_ckpt = get_effective_dir(run_dir)

        if best_ckpt is None:
            print(f"  {run_dir.name}: no best.pth.tar — skipping")
            continue

        print(f"  {run_dir.name}: checking test results...")
        results = find_test_results(effective_dir)

        if results is None:
            if run_test:
                print(f"  {run_dir.name}: no test results — running test evaluation")
                results = run_test_evaluation(best_ckpt, effective_dir)
            else:
                print(f"  {run_dir.name}: no test results (--no-run-test set, skipping)")
        elif "auprc" not in results or "auprc_ci_lower" not in results or "auroc_ci_lower" not in results:
            if run_test:
                if "auprc" not in results:
                    reason = "AUPRC missing"
                elif "auprc_ci_lower" not in results:
                    reason = "AUPRC CI missing"
                else:
                    reason = "AUROC CI missing"
                print(f"  {run_dir.name}: {reason} — re-running test evaluation")
                results = run_test_evaluation(best_ckpt, effective_dir)
            else:
                print(f"  {run_dir.name}: CI missing (--no-run-test set, skipping)")
        else:
            print(
                f"  {run_dir.name}: auroc={results.get('auroc', float('nan')):.4f}"
                f"  auprc={results.get('auprc', float('nan')):.4f}"
            )

        if results is not None:
            row: dict = {"model": model_name, "run": run_dir.name}
            for col in METRIC_COLS:
                row[col] = results.get(col)
            rows.append(row)
            _write_results_summary(effective_dir, results)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-rsync", action="store_true",
        help="Skip rsync, use local cache only.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview rsync without downloading.",
    )
    parser.add_argument(
        "--no-run-test", action="store_true",
        help="Do not auto-run test evaluation for missing results.",
    )
    args = parser.parse_args()

    if not args.skip_rsync:
        for model_name in REMOTE_SYNC_MODELS:
            print(f"=== Downloading {model_name} from CREATE ===")
            rsync_from_cluster(model_name, dry_run=args.dry_run)
        if args.dry_run:
            return

    run_test = not args.no_run_test
    all_rows: list[dict] = []

    for model_name, ckp_dir in MODEL_CKP_DIRS.items():
        print(f"\n=== {model_name} ===")
        rows = collect_metrics(model_name, ckp_dir, run_test=run_test)
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo completed runs found.")
        return

    df = pd.DataFrame(all_rows, columns=["model", "run"] + METRIC_COLS)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n=== Test Metrics Summary ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
