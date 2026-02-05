import argparse
import csv
from pathlib import Path
import sys

import numpy as np

# Allow running as a script without installation
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meancut_core.core import meancut
from meancut_core.metrics import clust_eval
from scipy.optimize import linear_sum_assignment


def _parse_params(param_str):
    if not param_str:
        return {}

    params = {}
    for item in param_str.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid param '{item}', expected key=value")
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key in {"k1", "k2", "noise"}:
            params[key] = int(val)
        elif key in {"ratio", "bandwidth"}:
            params[key] = float(val)
        elif key in {"embed", "normalize"}:
            params[key] = val.lower() in {"true", "1", "yes", "y"}
        elif key == "kernel":
            params[key] = val
        else:
            raise ValueError(f"Unknown parameter '{key}'")

    return params


def _load_baselines(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _load_labels(path):
    return np.loadtxt(path, delimiter=",").astype(int).ravel()


def _align_labels(ref_labels, pred_labels):
    ref_labels = np.asarray(ref_labels, dtype=int).ravel()
    pred_labels = np.asarray(pred_labels, dtype=int).ravel()

    ref_vals = np.unique(ref_labels)
    pred_vals = np.unique(pred_labels)

    counts = np.zeros((len(pred_vals), len(ref_vals)), dtype=int)
    for i, pv in enumerate(pred_vals):
        mask = pred_labels == pv
        for j, rv in enumerate(ref_vals):
            counts[i, j] = np.sum(ref_labels[mask] == rv)

    row_ind, col_ind = linear_sum_assignment(-counts)
    mapping = {pred_vals[r]: ref_vals[c] for r, c in zip(row_ind, col_ind)}

    aligned = np.full_like(pred_labels, fill_value=-1)
    for pv, rv in mapping.items():
        aligned[pred_labels == pv] = rv

    return aligned, mapping


def main():
    repo_root = ROOT.parent
    default_baseline = repo_root / "baselines" / "matlab_metrics.csv"
    default_out = repo_root / "baselines" / "metrics_compare.csv"

    parser = argparse.ArgumentParser(description="Compare Python MeanCut with MATLAB baselines")
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(default_baseline),
        help="Path to matlab_metrics.csv",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="",
        help="Comma-separated parameters (e.g., k1=20,k2=20,ratio=0.2)",
    )
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    parser.add_argument(
        "--out",
        type=str,
        default=str(default_out),
        help="Path to output merged metrics CSV",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {baseline_path}. Run MeanCut_Matlab/run_baseline.m first."
        )

    rows = _load_baselines(baseline_path)
    params = _parse_params(args.params)

    headers = ["ACC", "NMI", "ARI", "Fscore", "JI", "RI"]
    pass_all = True

    print(f"Baseline: {baseline_path}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for row in rows:
        dataset = row["dataset"]
        dataset_path = repo_root / "Synthetic Datasets" / dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        baseline_vals = np.array([float(row[h]) for h in headers], dtype=float)

        data = np.loadtxt(dataset_path)
        X = data[:, :-1]
        ref = data[:, -1].astype(int)

        cluster = meancut(X, **params)
        py_vals = np.array(clust_eval(ref, cluster), dtype=float)

        diff = np.abs(py_vals - baseline_vals)
        ok = np.allclose(py_vals, baseline_vals, rtol=args.rtol, atol=args.atol)
        pass_all = pass_all and ok

        status = "PASS" if ok else "FAIL"
        print(f"{dataset}: {status}")
        for i, h in enumerate(headers):
            print(f"  {h}: baseline={baseline_vals[i]:.6f} python={py_vals[i]:.6f} diff={diff[i]:.6f}")

        label_path = repo_root / "baselines" / "labels" / dataset.replace(".txt", "_labels.csv")
        if label_path.exists():
            matlab_labels = _load_labels(label_path)
            if matlab_labels.shape[0] != cluster.shape[0]:
                print("  Labels: SKIP (length mismatch)")
                pass_all = False
            else:
                aligned_py, _ = _align_labels(matlab_labels, cluster)
                mismatches = int(np.sum(aligned_py != matlab_labels))
                ratio = mismatches / matlab_labels.shape[0]
                label_status = "PASS" if mismatches == 0 else "FAIL"
                print(f"  Labels: {label_status} mismatches={mismatches} ratio={ratio:.6f}")
                pass_all = pass_all and (mismatches == 0)
        else:
            print("  Labels: SKIP (baseline labels not found)")

        out_row = {"dataset": dataset}
        for i, h in enumerate(headers):
            out_row[f"mat_{h}"] = baseline_vals[i]
            out_row[f"py_{h}"] = py_vals[i]
            out_row[f"diff_{h}"] = diff[i]
        out_rows.append(out_row)

    fieldnames = ["dataset"]
    for h in headers:
        fieldnames.extend([f"mat_{h}", f"py_{h}", f"diff_{h}"])

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Merged metrics written to: {out_path}")

    if not pass_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
