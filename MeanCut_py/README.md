# MeanCut Python

This folder contains:

- `main.py`: run MeanCut and print metrics (optional plot).
- `meancut_core/`: core implementation modules.
- `requirements.txt`: dependencies.

## Setup

```bash
conda activate py
pip install -r MeanCut_Python/requirements.txt
```

## Run

```bash
python MeanCut_Python/main.py --dataset DS1
python MeanCut_Python/main.py --dataset DS1 --plot
python MeanCut_Python/main.py --dataset "Synthetic Datasets/DS5.txt"
```

## Parameters

Pass parameters with `--params` as a comma-separated list, for example:

```bash
python MeanCut_Python/main.py --params "k1=20,k2=20,ratio=0.2,kernel=Laplacian,bandwidth=2"
```

Available parameters and defaults:

- `k1=20`: KNN size for boundary detection (DGF).
- `k2=20`: KNN size for graph/MST construction.
- `ratio=0.0`: Boundary point ratio in `[0, 1]` (0 disables boundary detection).
- `embed=True`: Apply PCA for large/high-dimensional data.
- `normalize=True`: Min-max normalization to `[0, 1]`.
- `noise=0`: Minimum cluster size threshold (smaller clusters are treated as noise).
- `kernel=Laplacian`: Edge weighting kernel (`Laplacian` or `Gaussian`).
- `bandwidth=2`: Kernel bandwidth parameter.
