import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

def nmf_coclustering(M, k, nmf_kwargs=None, random_state=0):
    assert np.all(M >= 0), "NMF requires nonnegative entries."
    nmf_kwargs = nmf_kwargs or {}
    nmf = NMF(n_components=k, init='nndsvda', random_state=random_state, **nmf_kwargs)
    W = nmf.fit_transform(M.astype(float))
    H = nmf.components_
    row_groups = np.argmax(W, axis=1)
    col_groups = np.argmax(H, axis=0)
    return row_groups, col_groups, W, H

def block_density_matrix(M, row_groups, col_groups, k, mode="mean"):
    D = np.zeros((k, k), dtype=float)
    for p in range(k):
        r_idx = np.where(row_groups == p)[0]
        for q in range(k):
            c_idx = np.where(col_groups == q)[0]
            if r_idx.size == 0 or c_idx.size == 0:
                D[p, q] = 0.0
                continue
            block = M[np.ix_(r_idx, c_idx)]
            if mode == "mean":
                D[p, q] = block.mean()
            elif mode == "sum":
                D[p, q] = block.sum()
            elif mode == "nnz_frac":
                D[p, q] = np.count_nonzero(block) / block.size
            else:
                raise ValueError("Unknown density mode")
    return D

def diagonalize_blocks_via_assignment(D):
    # maximize diagonal sum -> minimize -D
    row_idx, col_match = linear_sum_assignment(-D)
    row_order_clusters = list(range(D.shape[0]))           # keep rows in natural order
    col_order_clusters = list(col_match)                   # reorder columns to align dense blocks to diagonal
    return row_order_clusters, col_order_clusters

def build_perm_from_groups(groups, cluster_order):
    perm = []
    for c in cluster_order:
        perm.extend(np.where(groups == c)[0].tolist())
    return np.array(perm, dtype=int)

def groupnteach_permutation(M, k, density_mode="mean", nmf_kwargs=None, random_state=0):
    row_groups, col_groups, W, H = nmf_coclustering(M, k, nmf_kwargs, random_state)
    D = block_density_matrix(M, row_groups, col_groups, k, mode=density_mode)
    row_order_clusters, col_order_clusters = diagonalize_blocks_via_assignment(D)
    row_perm = build_perm_from_groups(row_groups, row_order_clusters)
    col_perm = build_perm_from_groups(col_groups, col_order_clusters)
    M_perm = M[np.ix_(row_perm, col_perm)]
    row_inv = np.argsort(row_perm)
    col_inv = np.argsort(col_perm)
    return M_perm, row_perm, col_perm, row_inv, col_inv, (row_groups, col_groups), D

def apply_permutation(M, row_perm, col_perm):
    return M[np.ix_(row_perm, col_perm)]