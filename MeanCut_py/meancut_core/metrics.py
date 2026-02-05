import numpy as np
from scipy.optimize import linear_sum_assignment


def clust_eval(ref, res):
    """Evaluate clustering results with multiple metrics (MATLAB ClustEval style)."""
    ref = np.asarray(ref, dtype=int).ravel()
    res = np.asarray(res, dtype=int).ravel()
    if ref.shape[0] != res.shape[0]:
        raise ValueError("ref and res must have the same length")

    n = ref.shape[0]
    if n <= 1:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    # Pair counting
    same_res = res[:, None] == res[None, :]
    same_ref = ref[:, None] == ref[None, :]
    triu = np.triu_indices(n, k=1)

    SS = int(np.sum((same_res & same_ref)[triu]))
    DD = int(np.sum((~same_res & ~same_ref)[triu]))
    SD = int(np.sum((~same_res & same_ref)[triu]))
    DS = int(np.sum((same_res & ~same_ref)[triu]))

    denom_ari = (2 * (SS * DD - DS * SD)) + (DS + SD) * (SS + SD + DS + DD)
    ARI = 0.0 if denom_ari == 0 else (2 * (SS * DD - DS * SD)) / denom_ari

    precision = 0.0 if (SS + DS) == 0 else SS / (SS + DS)
    recall = 0.0 if (SS + SD) == 0 else SS / (SS + SD)
    RI = (SS + DD) / (SS + SD + DS + DD)
    Fscore = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    JI = 0.0 if (SS + SD + DS) == 0 else SS / (SS + SD + DS)

    # Accuracy with Hungarian assignment
    p = np.unique(ref)
    c = np.unique(res)
    P_size = len(p)
    C_size = len(c)

    Pid = (p[:, None] == ref[None, :]).astype(float)
    Cid = (c[:, None] == res[None, :]).astype(float)
    CP = Cid @ Pid.T

    row_ind, col_ind = linear_sum_assignment(-CP)
    acc = CP[row_ind, col_ind].sum() / n

    NMI = get_nmi(ref, res)
    return acc, NMI, ARI, Fscore, JI, RI


def get_nmi(A, B):
    """Compute normalized mutual information (MATLAB GetNMI style)."""
    A = np.asarray(A, dtype=int).ravel()
    B = np.asarray(B, dtype=int).ravel()

    N = A.size
    Ca = int(np.max(A)) if N > 0 else 0
    Cb = int(np.max(B)) if N > 0 else 0

    if Ca == 0 or Cb == 0:
        return 0.0

    N1 = []
    D1 = np.zeros(Ca, dtype=float)
    D2 = np.zeros(Cb, dtype=float)

    for i in range(1, Ca + 1):
        N_idot = np.sum(A == i)
        if N_idot > 0:
            D1[i - 1] = N_idot * np.log(N_idot / N)
        for j in range(1, Cb + 1):
            N_dotj = np.sum(B == j)
            N_ij = np.sum((A == i) & (B == j))
            if N_ij == 0:
                N1.append(0.0)
            else:
                N1.append(N_ij * np.log((N_ij * N) / (N_idot * N_dotj)))
            if N_dotj > 0:
                D2[j - 1] = N_dotj * np.log(N_dotj / N)

    denom = np.sum(D1) + np.sum(D2)
    if denom == 0:
        return 0.0

    return (-2.0 * np.sum(N1)) / denom
