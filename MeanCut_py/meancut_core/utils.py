import numpy as np
from scipy.spatial import cKDTree


def minmax_scale(X, min_val=0.0, max_val=1.0):
    """Scale each feature to [min_val, max_val] (MATLAB mapminmax style)."""
    X = np.asarray(X, dtype=float)
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    scale = xmax - xmin
    scale[scale == 0] = 1.0
    # Use multiplication by reciprocal to match MATLAB mapminmax rounding
    scale_inv = 1.0 / scale
    X_std = (X - xmin) * scale_inv
    return X_std * (max_val - min_val) + min_val


def init_pca(X, no_dims, contri):
    """Preprocess data with PCA for large scale / high-dimensional cases."""
    X = np.asarray(X, dtype=float)

    # Make sure data is zero mean
    mean = X.mean(axis=0)
    X = X - mean

    # Covariance matrix
    C = np.cov(X, rowvar=False, bias=False)
    C[np.isnan(C)] = 0.0
    C[np.isinf(C)] = 0.0

    # Eigendecomposition (symmetric)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Find the best dimension
    if np.sum(eigvals) == 0:
        best_dim = no_dims
    else:
        idx = (np.cumsum(eigvals) / np.sum(eigvals)) < contri
        if np.max(idx) == 0:
            best_dim = no_dims
        else:
            best_dim = max(no_dims, min(int(np.where(idx)[0][-1]) + 1, 2000))

    return X @ eigvecs[:, :best_dim]


def brute_knn(data, query, k):
    """Deterministic KNN using (distance, index) ordering for MATLAB-style ties."""
    data = np.asarray(data, dtype=float)
    query = np.asarray(query, dtype=float)

    if k < 1:
        raise ValueError("k must be >= 1")

    n_data = data.shape[0]
    idx = np.empty((query.shape[0], k), dtype=int)
    dist = np.empty((query.shape[0], k), dtype=float)
    data_idx = np.arange(n_data, dtype=int)

    # Use squared distances for ordering to preserve tiny tie-breaks
    # that can be lost after sqrt (matches MATLAB knnsearch behavior).
    for i, q in enumerate(query):
        diff = data - q
        d2 = np.einsum("ij,ij->i", diff, diff)
        order = np.lexsort((data_idx, d2))
        sel = order[:k]
        idx[i] = sel
        dist[i] = np.sqrt(d2[sel])

    return idx, dist


def knn_search(data, query, k, brute_threshold=10000):
    """MATLAB knnsearch-like helper using Euclidean distance."""
    data = np.asarray(data, dtype=float)
    query = np.asarray(query, dtype=float)

    if k < 1:
        raise ValueError("k must be >= 1")

    if data.shape[0] <= brute_threshold:
        return brute_knn(data, query, k)

    tree = cKDTree(data)
    dist, idx = tree.query(query, k=k)

    dist = np.asarray(dist)
    idx = np.asarray(idx)
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    # Stable tie-breaking on (distance, index) for returned neighbors
    for i in range(dist.shape[0]):
        order = np.lexsort((idx[i], dist[i]))
        dist[i] = dist[i, order]
        idx[i] = idx[i, order]

    return idx, dist


class UnionFind:
    """Disjoint-set data structure for MST and connected components."""

    def __init__(self, n):
        self.parent = np.arange(n, dtype=int)
        self.size = np.ones(n, dtype=int)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def component_labels(self):
        """Return component labels in order of first appearance (MATLAB conncomp style)."""
        root_to_label = {}
        labels = np.empty_like(self.parent)
        next_label = 0
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in root_to_label:
                root_to_label[root] = next_label
                next_label += 1
            labels[i] = root_to_label[root]
        return labels
