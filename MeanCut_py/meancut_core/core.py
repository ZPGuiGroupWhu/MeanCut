import numpy as np

from .utils import minmax_scale, init_pca, knn_search, UnionFind


def meancut(
    X,
    k1=20,
    k2=20,
    ratio=0.0,
    embed=True,
    normalize=True,
    noise=0,
    kernel="Laplacian",
    bandwidth=2,
):
    """
    This is a novel spectral clustering algorithm based on path similarity and degree descent criterion.
    This function returns cluster labels for the N by D matrix X. Each row in X represents an observation.

    Parameters are:
    - k1: number of nearest neighbors for DGF (boundary detection).
    - k2: number of nearest neighbors for MST construction.
    - ratio: percentile of boundary points to the total number of points in [0, 1].
    - embed: if True, PCA is applied for large-scale and high-dimensional data.
    - normalize: if True, scale features with min-max normalization.
    - noise: cluster size threshold for noisy clusters.
    - kernel: kernel function ("Laplacian" or "Gaussian").
    - bandwidth: bandwidth parameter for the kernel function.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (N x D)")

    # Remove duplicate observations
    X, orig_id = np.unique(X, axis=0, return_inverse=True)

    # Normalize the data
    if normalize:
        X = minmax_scale(X, 0.0, 1.0)

    n, dim = X.shape

    if n <= 1:
        cluster = np.ones(n, dtype=int)
        return cluster[orig_id]

    # Perform PCA for large-scale and high-dimensional data
    if embed and n >= 5000 and dim >= 50:
        X = init_pca(X, 50, 0.8)

    # Calculate DGF to identify boundary and internal points
    if ratio > 0:
        k1_eff = min(k1, max(n - 1, 1))
        dgf_knn, dgf_dis = knn_search(X, X, k1_eff + 1)
        dgf_knn = dgf_knn[:, 1:]
        dgf_dis = dgf_dis[:, 1:]

        density = np.mean(dgf_dis, axis=1)
        grad = np.sum((density[dgf_knn] - density[:, None]) / dgf_dis, axis=1)
        grad_sort = np.sort(grad)
        grad_index = int(np.ceil(n * ratio))
        grad_index = min(max(grad_index, 0), len(grad_sort) - 1)
        gradT = grad_sort[grad_index]

        edg = np.where(grad < gradT)[0]
        int_idx = np.setdiff1d(np.arange(n), edg, assume_unique=False)
    else:
        edg = np.array([], dtype=int)
        int_idx = np.arange(n, dtype=int)

    Y = X[int_idx]

    # Construct a KNN graph
    k2_eff = min(k2, max(len(int_idx) - 1, 0))
    edges = _build_knn_graph(Y, k2_eff)

    # Generate MST
    mst_edges = _fast_mst(Y, edges, kernel, bandwidth)

    # Optimize MeanCut with degree descent
    if Y.shape[0] < 3:
        int_clust = np.ones(Y.shape[0], dtype=int)
    elif Y.shape[0] < 50000:
        pW = _path_w1(mst_edges, Y.shape[0])
        int_clust = _degree_descent1(pW, noise)
    else:
        degs = _path_w(mst_edges, Y.shape[0])
        int_clust = _degree_descent(mst_edges, degs, noise)

    # Assign boundary points to nearest internal points
    clust = np.zeros(n, dtype=int)
    clust[int_idx] = int_clust
    if edg.size > 0:
        near_int, _ = knn_search(X[int_idx], X[edg], 1)
        near_int = near_int.reshape(-1)
        clust[edg] = int_clust[near_int]

    cluster = clust[orig_id]
    return cluster


def _build_knn_graph(Y, k2):
    """Build an undirected KNN graph as a list of weighted edges."""
    n = Y.shape[0]
    if n <= 1 or k2 <= 0:
        return []

    get_knn, knn_dis = knn_search(Y, Y, k2 + 1)
    get_knn = get_knn[:, 1:]
    knn_dis = knn_dis[:, 1:]

    row_ids = np.repeat(np.arange(n), k2)
    col_ids = get_knn.reshape(-1)
    weights = knn_dis.reshape(-1)

    u = np.minimum(row_ids, col_ids)
    v = np.maximum(row_ids, col_ids)
    pairs = np.stack([u, v], axis=1)

    # Keep unique undirected edges
    _, unique_idx = np.unique(pairs, axis=0, return_index=True)
    pairs = pairs[unique_idx]
    weights = weights[unique_idx]

    # Sort edges to match MATLAB unique(...,'rows') ordering more closely
    order = np.lexsort((weights, pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    weights = weights[order]

    edges = [(int(pairs[i, 0]), int(pairs[i, 1]), float(weights[i])) for i in range(len(weights))]
    return edges


def _fast_mst(X, edges, kernel, delta):
    """Fast approximate maximum spanning tree algorithm (MATLAB-style)."""
    n = X.shape[0]
    edge_list = list(edges)

    # Identify connected components
    uf = UnionFind(n)
    for u, v, _ in edge_list:
        uf.union(u, v)
    comp_labels = uf.component_labels()
    clus_num = int(comp_labels.max()) + 1 if n > 0 else 0

    # Connect components by nearest neighbors (MATLAB-style chaining)
    if clus_num >= 2:
        all_id = np.arange(n)
        for i in range(clus_num - 1):
            clus_id = np.where(comp_labels == i)[0]
            rest_id = np.setdiff1d(all_id, clus_id, assume_unique=False)
            if clus_id.size == 0 or rest_id.size == 0:
                all_id = rest_id
                continue

            pt_id, near_dis = knn_search(X[rest_id], X[clus_id], 1)
            near_dis = near_dis.reshape(-1)
            pt_id = pt_id.reshape(-1)
            min_dis = float(np.min(near_dis))
            near_id = np.where(near_dis == min_dis)[0]

            for nid in near_id:
                u = int(clus_id[nid])
                v = int(rest_id[pt_id[nid]])
                edge_list.append((u, v, min_dis))

            all_id = rest_id

    # Kruskal MST
    edge_list_sorted = sorted(edge_list, key=lambda e: (e[2], e[0], e[1]))
    uf_mst = UnionFind(n)
    mst_edges = []
    for u, v, w in edge_list_sorted:
        if uf_mst.union(u, v):
            mst_edges.append((u, v, w))

    # Apply kernel weights
    kernel_l = str(kernel).lower()
    if kernel_l == "laplacian":
        mst_edges = [(u, v, float(np.exp(-w / delta))) for u, v, w in mst_edges]
    elif kernel_l == "gaussian":
        mst_edges = [(u, v, float(np.exp(-(w ** 2) / (2 * (delta ** 2))))) for u, v, w in mst_edges]
    else:
        raise ValueError("Kernel must be 'Laplacian' or 'Gaussian'")

    return mst_edges


def _path_w(mst_edges, n):
    """Calculate the path-based degree sequence (for large-scale data)."""
    edges_sorted = sorted(mst_edges, key=lambda e: e[2], reverse=True)
    visit = np.full(n, -1, dtype=int)
    degs = np.ones(n, dtype=float)

    for i, j, w in edges_sorted:
        i = int(i)
        j = int(j)
        if visit[i] == -1 and visit[j] == -1:
            degs[i] += w
            degs[j] += w
            visit[i] = j
            visit[j] = j
        elif visit[i] == -1 and visit[j] != -1:
            set_id_j = np.where(visit == visit[j])[0]
            degs[i] += len(set_id_j) * w
            degs[set_id_j] += w
            visit[i] = visit[j]
        elif visit[i] != -1 and visit[j] == -1:
            set_id_i = np.where(visit == visit[i])[0]
            degs[j] += len(set_id_i) * w
            degs[set_id_i] += w
            visit[j] = visit[i]
        else:
            set_id_i = np.where(visit == visit[i])[0]
            set_id_j = np.where(visit == visit[j])[0]
            degs[set_id_i] += len(set_id_j) * w
            degs[set_id_j] += len(set_id_i) * w
            visit[set_id_j] = visit[i]

    return degs


def _path_w1(mst_edges, n):
    """Calculate the path-based similarity matrix (for small/medium data)."""
    edges_sorted = sorted(mst_edges, key=lambda e: e[2], reverse=True)
    visit = np.full(n, -1, dtype=int)
    pW = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(pW, 1.0)

    for i, j, w in edges_sorted:
        i = int(i)
        j = int(j)
        if visit[i] == -1 and visit[j] == -1:
            pW[i, j] = w
            pW[j, i] = w
            visit[i] = j
            visit[j] = j
        elif visit[i] == -1 and visit[j] != -1:
            set_id_j = np.where(visit == visit[j])[0]
            pW[i, set_id_j] = w
            pW[set_id_j, i] = w
            visit[i] = visit[j]
        elif visit[i] != -1 and visit[j] == -1:
            set_id_i = np.where(visit == visit[i])[0]
            pW[set_id_i, j] = w
            pW[j, set_id_i] = w
            visit[j] = visit[i]
        else:
            set_id_i = np.where(visit == visit[i])[0]
            set_id_j = np.where(visit == visit[j])[0]
            pW[np.ix_(set_id_i, set_id_j)] = w
            pW[np.ix_(set_id_j, set_id_i)] = w
            visit[set_id_j] = visit[i]

    return pW


def _degree_descent1(W, noiseT):
    """Greedy optimization with degree descent criterion algorithm (matrix version)."""
    W = np.asarray(W, dtype=np.float32)
    n = W.shape[0]
    cluster = np.zeros(n, dtype=int)
    mark = 1

    degs = np.sum(W, axis=1)
    sort_id = np.argsort(-degs, kind="mergesort")
    deg_sort = degs[sort_id]

    W_sort = W[sort_id][:, sort_id]

    while np.any(cluster == 0):
        id_idx = np.flatnonzero(cluster == 0)[0]
        num = 1
        sum_w = 1.0
        sum_deg = np.float32(deg_sort[id_idx])
        meancut = (1.0 / (n - 1)) * (1.0 - sum_w / sum_deg)
        storage = [id_idx]

        for i in range(id_idx + 1, n):
            if cluster[i] == 0:
                num += 1
                sum_deg = np.float32(sum_deg + deg_sort[i])
                sum_w_add = 2.0 * float(np.sum(W_sort[i, storage])) + 1.0
                sum_w += sum_w_add
                cut = (1.0 / (n - num)) * (1.0 - sum_w / sum_deg)
                if cut <= meancut:
                    meancut = cut
                    storage.append(i)
                else:
                    num -= 1
                    sum_deg = np.float32(sum_deg - deg_sort[i])
                    sum_w -= sum_w_add

        cluster[storage] = mark
        mark += 1

    # Reorder back to the original indexing
    cluster = cluster[np.argsort(sort_id)]
    cluster = _apply_noise_threshold(cluster, noiseT)
    return cluster


def _degree_descent(mst_edges, degs, noiseT):
    """Greedy optimization with degree descent criterion algorithm (tree version)."""
    n = len(degs)
    cluster = np.zeros(n, dtype=int)
    mark = 1

    degs = np.asarray(degs, dtype=float)
    sort_id = np.argsort(-degs, kind="mergesort")
    deg_sort = degs[sort_id]

    # Build adjacency list from tree edges
    adj = [[] for _ in range(n)]
    for u, v, w in mst_edges:
        u = int(u)
        v = int(v)
        adj[u].append((v, w))
        adj[v].append((u, w))

    while np.any(cluster == 0):
        id_idx = np.flatnonzero(cluster == 0)
        if id_idx.size == 0:
            break

        num = 1
        sum_w = 1.0
        sum_deg = float(deg_sort[id_idx[0]])
        meancut = (1.0 / (n - 1)) * (1.0 - sum_w / sum_deg)

        root = int(sort_id[id_idx[0]])
        storage = [root]

        # Compute min edge weight on the path from root to all nodes
        min_w = _path_min_weights(adj, root)
        pw_val = min_w[sort_id[id_idx[1:]]]

        for i in range(1, len(id_idx)):
            num += 1
            sum_deg += float(deg_sort[id_idx[i]])
            sum_w_add = 2.0 * float(pw_val[i - 1]) * len(storage) + 1.0
            sum_w += sum_w_add
            cut = (1.0 / (n - num)) * (1.0 - sum_w / sum_deg)
            if cut <= meancut:
                meancut = cut
                storage.append(int(sort_id[id_idx[i]]))
            else:
                num -= 1
                sum_deg -= float(deg_sort[id_idx[i]])
                sum_w -= sum_w_add

        mask = np.isin(sort_id, storage)
        cluster[mask] = mark
        mark += 1

    cluster = cluster[np.argsort(sort_id)]
    cluster = _apply_noise_threshold(cluster, noiseT)
    return cluster


def _path_min_weights(adj, root):
    """Minimum edge weight along the path from root to all nodes in a tree."""
    n = len(adj)
    min_w = np.full(n, np.inf, dtype=float)
    min_w[root] = np.inf

    stack = [(root, -1, np.inf)]
    while stack:
        node, parent, cur_min = stack.pop()
        for nbr, w in adj[node]:
            if nbr == parent:
                continue
            new_min = min(cur_min, w)
            min_w[nbr] = new_min
            stack.append((nbr, node, new_min))

    return min_w


def _apply_noise_threshold(cluster, noiseT):
    """Remove clusters smaller than noiseT and relabel the rest sequentially."""
    if noiseT <= 0:
        # Still relabel sequentially to match MATLAB behavior
        noiseT = 1

    mark = 1
    out = cluster.copy()
    max_cluster = int(out.max()) if out.size > 0 else 0
    for i in range(1, max_cluster + 1):
        num = int(np.sum(out == i))
        if num < noiseT:
            out[out == i] = 0
        else:
            out[out == i] = mark
            mark += 1

    return out
