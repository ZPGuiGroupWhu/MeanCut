import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

# Allow running as a script without installation
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meancut_core.core import meancut
from meancut_core.metrics import clust_eval
from meancut_core.plotting import plotcluster2


def _parse_params(param_str):
    # Parse comma-separated key=value pairs into a parameter dict.
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


def _resolve_dataset(arg, repo_root):
    # Accept either a short dataset name (e.g., DS1) or a full path.
    dataset_path = Path(arg)
    if dataset_path.exists():
        return dataset_path

    name = arg.strip()
    if not name:
        raise ValueError("Dataset name cannot be empty")

    if not name.lower().endswith(".txt"):
        name = f"{name}.txt"

    resolved = repo_root / "Synthetic Datasets" / name
    if resolved.exists():
        return resolved

    raise FileNotFoundError(f"Dataset not found: {arg}")


def main():
    repo_root = ROOT.parent
    default_dataset = "DS1"

    parser = argparse.ArgumentParser(description="MeanCut main workflow")
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Dataset name (e.g., DS1) or file path",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="",
        help="Comma-separated parameters (e.g., k1=20,k2=20,ratio=0.2)",
    )
    parser.add_argument("--plot", action="store_true", help="Show clustering plot")
    args = parser.parse_args()

    # Resolve dataset path and load data (last column is ground-truth label).
    dataset_path = _resolve_dataset(args.dataset, repo_root)

    data = np.loadtxt(dataset_path)
    X = data[:, :-1]
    ref = data[:, -1].astype(int)

    # Run MeanCut with optional overrides and compute evaluation metrics.
    params = _parse_params(args.params)
    cluster = meancut(X, **params)

    ACC, NMI, ARI, Fscore, JI, RI = clust_eval(ref, cluster)

    print("MeanCut results")
    print(f"Dataset: {dataset_path}")
    print(f"ACC   : {ACC:.6f}")
    print(f"NMI   : {NMI:.6f}")
    print(f"ARI   : {ARI:.6f}")
    print(f"Fscore: {Fscore:.6f}")
    print(f"JI    : {JI:.6f}")
    print(f"RI    : {RI:.6f}")

    if args.plot:
        # Plot first two dimensions with cluster labels.
        plotcluster2(X, cluster)
        plt.show()


if __name__ == "__main__":
    main()
