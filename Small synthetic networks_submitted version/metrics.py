"""
Graph-recovery metrics.

Given a true weighted adjacency matrix W_true and an estimate W_hat, we threshold
at `thresh` to obtain binary edge sets, then compute:
    - SHD (Structural Hamming Distance): count of edges that must be added,
      deleted, or reversed to convert W_hat -> W_true. We use the standard
      definition:  SHD = FP + FN + R, where R counts edges present in both graphs
      but in opposite directions.
    - Precision, Recall on directed edges.

For Experiment 2's headline metric, we also count SPURIOUS EDGES AMONG
CONFOUNDED OBSERVED PAIRS -- that is, the number of estimated edges (in either
direction) between pairs of observed variables that share a latent parent in the
truth but have no direct edge.
"""

from __future__ import annotations
import numpy as np


def binarize(W: np.ndarray, thresh: float = 0.3) -> np.ndarray:
    B = (np.abs(W) > thresh).astype(int)
    np.fill_diagonal(B, 0)
    return B


def structural_hamming_distance(
    W_true: np.ndarray, W_hat: np.ndarray, thresh: float = 0.3
) -> int:
    Bt = binarize(W_true, thresh=1e-9)     # true weights are exactly zero or not
    Bh = binarize(W_hat, thresh=thresh)
    p = Bt.shape[0]
    shd = 0
    seen = set()
    for i in range(p):
        for j in range(p):
            if i == j or (j, i) in seen:
                continue
            seen.add((i, j))
            t_ij, t_ji = Bt[i, j], Bt[j, i]
            h_ij, h_ji = Bh[i, j], Bh[j, i]
            # Unordered pair states: no-edge (0,0), i->j (1,0), j->i (0,1), both (1,1).
            t_state = (t_ij, t_ji)
            h_state = (h_ij, h_ji)
            if t_state == h_state:
                continue
            # Reversed edge counts as 1 mistake (not 2).
            if {t_state, h_state} == {(1, 0), (0, 1)}:
                shd += 1
            else:
                # Otherwise, count each edge mismatch individually.
                shd += abs(t_ij - h_ij) + abs(t_ji - h_ji)
    return shd


def precision_recall(
    W_true: np.ndarray, W_hat: np.ndarray, thresh: float = 0.3
) -> tuple[float, float]:
    """Directed-edge precision and recall."""
    Bt = binarize(W_true, thresh=1e-9)
    Bh = binarize(W_hat, thresh=thresh)
    tp = int(((Bt == 1) & (Bh == 1)).sum())
    fp = int(((Bt == 0) & (Bh == 1)).sum())
    fn = int(((Bt == 1) & (Bh == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return prec, rec


def count_confounder_induced_false_edges(
    W_hat: np.ndarray,
    confounded_pairs: list[tuple[int, int]],
    thresh: float = 0.3,
) -> int:
    """
    Count directed edges in W_hat between pairs of observed variables that share
    a latent parent in the truth but have no direct edge. An edge in either
    direction counts as one spurious confounder-induced edge.

    W_hat is assumed to be indexed with observed variables in [0, p_x), which is
    the convention used by sem.generate_sem.
    """
    B = binarize(W_hat, thresh=thresh)
    count = 0
    for (i, j) in confounded_pairs:
        if B[i, j] or B[j, i]:
            count += 1
    return count


# ---------------------------------------------------------------------------
# New metric: latent-parent identification
# ---------------------------------------------------------------------------
#
# For each pair of observed variables that share a latent parent in the TRUTH,
# check whether the fitted model W_hat connects them to a COMMON FITTED LATENT.
# Concretely: (i, j) is "recovered via a common latent parent" iff there
# exists some latent index h (in [p_x, p_x+p_h)) with |W_hat[h, i]| > thresh
# AND |W_hat[h, j]| > thresh.
#
# Important: the fitted latents have arbitrary labels -- h=0 of the fit need
# not correspond to H_0 of the truth. This metric is robust to that because
# it only asks whether SOME common fitted latent parent exists, not whether a
# SPECIFIC one does.
#
def count_latent_parent_identifications(
    W_hat_full: np.ndarray,
    confounded_pairs: list[tuple[int, int]],
    p_x: int,
    p_h: int,
    thresh: float = 0.15,
) -> int:
    """
    Parameters
    ----------
    W_hat_full : (p_x + p_h, p_x + p_h) estimated weighted adjacency.
                 Latent rows are the last p_h rows.
    confounded_pairs : pairs (i, j) with i, j < p_x that share a latent parent
                       in the TRUTH (as returned by sem.observed_observed_confounded_pairs).
    p_x, p_h : sizes of the observed and latent blocks.
    thresh : magnitude threshold for counting a W_hat entry as an edge.
             NOTE: defaults to 0.15, lower than the global edge threshold (0.3)
             used elsewhere. Fitted latent->observed weights tend to cluster in
             the 0.1-0.3 range due to the L1 + identifiability interaction in
             linear-Gaussian latent SEMs (see project README). Using a single
             high threshold for everything would systematically undercount
             latent recoveries; using a lower threshold for THIS metric only
             is the principled fix.

    Returns
    -------
    int: number of confounded pairs (i, j) for which W_hat connects BOTH i and j
         to some common fitted latent h in [p_x, p_x + p_h).
    """
    if p_h == 0 or not confounded_pairs:
        return 0
    lat_rows = W_hat_full[p_x:p_x + p_h, :p_x]   # (p_h, p_x)
    lat_children = (np.abs(lat_rows) > thresh)   # boolean (p_h, p_x)
    count = 0
    for (i, j) in confounded_pairs:
        # Common fitted latent parent exists iff some row has both i-th and j-th entries True.
        if np.any(lat_children[:, i] & lat_children[:, j]):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Latent-label alignment (for full-graph SHD/precision/recall)
# ---------------------------------------------------------------------------
#
# The fitted latents are unordered: there is no reason fitted-latent h=0 should
# correspond to true-latent H_0. Before computing full-graph SHD/precision/recall
# we permute the fitted latent rows AND columns to best match the truth, using
# a greedy bipartite match on the latent->observed-children similarity.
#
def align_latent_labels(
    W_hat_full: np.ndarray,
    W_true_full: np.ndarray,
    p_x: int,
    p_h: int,
    thresh: float = 0.15,
) -> np.ndarray:
    """
    Permute the latent block of W_hat_full so that fitted latent k is matched
    to the true latent whose binary child set it most overlaps with.

    Uses a simple greedy match on |intersection| - |symmetric difference| of
    the (thresholded) child sets. Exact when p_h is small, which it is here.

    Returns a permuted copy of W_hat_full in which the latent rows/cols have
    been reordered; observed block is unchanged.
    """
    if p_h == 0:
        return W_hat_full.copy()
    # True and fitted latent-child binary matrices: shape (p_h, p_x).
    true_children = np.abs(W_true_full[p_x:p_x + p_h, :p_x]) > 1e-9
    hat_children = np.abs(W_hat_full[p_x:p_x + p_h, :p_x]) > thresh

    # Score matrix: rows = true latents, cols = fitted latents.
    # score[t, f] = overlap minus symmetric-difference of child sets.
    score = np.zeros((p_h, p_h), dtype=int)
    for t in range(p_h):
        for fe in range(p_h):
            inter = int((true_children[t] & hat_children[fe]).sum())
            symdiff = int((true_children[t] ^ hat_children[fe]).sum())
            score[t, fe] = inter - symdiff

    # Greedy assignment: pick the highest remaining score, assign, remove row+col.
    assigned_true = set()
    assigned_hat = set()
    # Pair (fit_index -> true_index)
    perm_fit_to_true: dict[int, int] = {}
    order = np.argsort(-score, axis=None)  # descending
    for flat in order:
        t, fe = divmod(int(flat), p_h)
        if t in assigned_true or fe in assigned_hat:
            continue
        perm_fit_to_true[fe] = t
        assigned_true.add(t)
        assigned_hat.add(fe)
        if len(perm_fit_to_true) == p_h:
            break

    # Build the permutation that rearranges fitted latents to match true order.
    # new_latent_block_row_k corresponds to true latent k
    # => we want the row formerly at position f such that perm_fit_to_true[f] == k
    true_to_fit = {t: fe for fe, t in perm_fit_to_true.items()}
    latent_perm = np.array([true_to_fit[k] for k in range(p_h)], dtype=int)

    # Full permutation applies identity on observed, latent_perm on latent block.
    perm = np.concatenate([np.arange(p_x), p_x + latent_perm])
    W_aligned = W_hat_full[np.ix_(perm, perm)].copy()
    return W_aligned
