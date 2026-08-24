"""
Run Experiment 2 one (n, seed) at a time for the v6 9-method matrix.
Appends rows to results/exp2_rows.csv. Resume-safe.

Usage:
    python run_exp2_chunk.py <n> <seed_start> <seed_end>
Example:
    python run_exp2_chunk.py 200 0 10     # 10 seeds at n=200
"""
from __future__ import annotations
import csv, os, sys, time
import numpy as np

from sem import generate_sem, observed_observed_confounded_pairs
from diff_struct_em import fit_diff_struct_em, fit_diff_struct_em_stability
from baselines import (
    fit_observed_only_dagma, fit_observed_only_notears,
    fit_oracle_fully_observed_dagma,
)
from metrics import (
    structural_hamming_distance, precision_recall,
    count_confounder_induced_false_edges,
    count_latent_parent_identifications, align_latent_labels,
)

RES_DIR = "results"
os.makedirs(RES_DIR, exist_ok=True)
CSV_PATH = os.path.join(RES_DIR, "exp2_rows.csv")

P_X, P_H = 8, 2
THRESH = 0.3
LATENT_THRESH = 0.15
N_OUTER = 15
M_STEP_ITERS = 300
LAM = 0.02
RHO = 0.5
CONFOUNDING_REGIME = "strong"
# Stability selection.
STAB_N_INITS = 16
STAB_EDGE_THRESH = 0.15
STAB_FREQ_THRESH = 0.75

# 9 methods per (n, seed) -- see METHODS registry in experiments.py.
N_METHODS_PER_CELL = 12

FIELDS = [
    "method", "n", "seed", "n_confounded_pairs",
    "shd_obs", "precision_obs", "recall_obs",
    "spurious_confounder_edges",
    "shd_full", "precision_full", "recall_full",
    "latent_parent_recoveries",
]


def _already_done() -> set[tuple[int, int]]:
    if not os.path.exists(CSV_PATH):
        return set()
    seen: dict[tuple[int, int], int] = {}
    with open(CSV_PATH, "r") as f:
        for r in csv.DictReader(f):
            key = (int(r["n"]), int(r["seed"]))
            seen[key] = seen.get(key, 0) + 1
    return {k for k, c in seen.items() if c >= N_METHODS_PER_CELL}


def _append_rows(rows: list[dict]) -> None:
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def _row(name: str, n: int, seed: int, n_confounded: int,
         W_true_full: np.ndarray, W_true_obs: np.ndarray, pairs: list,
         W_obs: np.ndarray, W_full_aligned: np.ndarray | None,
         binary_input: bool) -> dict:
    obs_thresh = 0.5 if binary_input else THRESH
    lat_thresh = 0.5 if binary_input else LATENT_THRESH
    full_thresh = 0.5 if binary_input else THRESH

    d = {
        "method": name, "n": n, "seed": seed, "n_confounded_pairs": n_confounded,
        "shd_obs": structural_hamming_distance(W_true_obs, W_obs, thresh=obs_thresh),
        "precision_obs": precision_recall(W_true_obs, W_obs, thresh=obs_thresh)[0],
        "recall_obs": precision_recall(W_true_obs, W_obs, thresh=obs_thresh)[1],
        "spurious_confounder_edges": count_confounder_induced_false_edges(W_obs, pairs, thresh=obs_thresh),
        "shd_full": "", "precision_full": "", "recall_full": "", "latent_parent_recoveries": "",
    }
    if W_full_aligned is not None:
        d["shd_full"] = structural_hamming_distance(W_true_full, W_full_aligned, thresh=full_thresh)
        p_full, r_full = precision_recall(W_true_full, W_full_aligned, thresh=full_thresh)
        d["precision_full"] = p_full
        d["recall_full"] = r_full
        d["latent_parent_recoveries"] = count_latent_parent_identifications(
            W_full_aligned, pairs, P_X, P_H, thresh=lat_thresh)
    return d


def run_one(n: int, seed: int) -> None:
    data = generate_sem(
        p_x=P_X, p_h=P_H, n=n, seed=seed,
        confounding_regime=CONFOUNDING_REGIME,
    )
    pairs = observed_observed_confounded_pairs(data)
    n_confounded = len(pairs)
    W_true_full = data.W_true
    W_true_obs = W_true_full[:P_X, :P_X]

    t0 = time.time()
    rows: list[dict] = []

    # Observed-only baselines (random init).
    W_notears_obs = fit_observed_only_notears(data.X, lam=LAM, init_seed=seed)
    W_dagma_obs   = fit_observed_only_dagma(data.X, lam=LAM, init_seed=seed)
    rows.append(_row("notears_obs", n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_notears_obs, None, binary_input=False))
    rows.append(_row("dagma_obs",   n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_dagma_obs, None, binary_input=False))

    # Observed-only baselines (OLS warm-start, NEW in v7).
    W_notears_obs_warm = fit_observed_only_notears(
        data.X, lam=LAM, init_seed=seed, init_mode="ols_warmstart")
    W_dagma_obs_warm   = fit_observed_only_dagma(
        data.X, lam=LAM, init_seed=seed, init_mode="ols_warmstart")
    rows.append(_row("notears_obs_warmstart", n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_notears_obs_warm, None, binary_input=False))
    rows.append(_row("dagma_obs_warmstart", n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_dagma_obs_warm, None, binary_input=False))

    # NOTEARS + latents (single fit).
    res_nt = fit_diff_struct_em(
        data.X, P_X, P_H, n_outer=N_OUTER, m_step_iters=M_STEP_ITERS,
        lam=LAM, latent_l1_scale=0.0, rho=RHO,
        acyclicity="notears", init_seed=seed, init_mode="random",
        e_step_mode="full", enforce_improving=True,
    )
    W_nt = res_nt.W
    W_nt_aligned = align_latent_labels(W_nt, W_true_full, P_X, P_H, thresh=LATENT_THRESH)
    rows.append(_row("notears_latent_single", n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_nt[:P_X, :P_X], W_nt_aligned, binary_input=False))

    # DAGMA + latents (single fit).
    res_dg = fit_diff_struct_em(
        data.X, P_X, P_H, n_outer=N_OUTER, m_step_iters=M_STEP_ITERS,
        lam=LAM, latent_l1_scale=0.0, rho=RHO,
        acyclicity="dagma", init_seed=seed, init_mode="random",
        e_step_mode="full", enforce_improving=True,
    )
    W_dg = res_dg.W
    W_dg_aligned = align_latent_labels(W_dg, W_true_full, P_X, P_H, thresh=LATENT_THRESH)
    rows.append(_row("dagma_latent_single", n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_dg[:P_X, :P_X], W_dg_aligned, binary_input=False))

    # Stability-selected versions.
    W_nt_stable, _ = fit_diff_struct_em_stability(
        data.X, P_X, P_H, n_inits=STAB_N_INITS, base_init_seed=1000 * seed,
        edge_thresh=STAB_EDGE_THRESH, freq_thresh=STAB_FREQ_THRESH,
        n_outer=N_OUTER, m_step_iters=M_STEP_ITERS,
        lam=LAM, rho=RHO, acyclicity="notears", init_mode="random",
        enforce_improving=True,
    )
    W_nt_stable_aligned = align_latent_labels(W_nt_stable, W_true_full, P_X, P_H, thresh=0.5)
    rows.append(_row("notears_latent_stable", n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_nt_stable[:P_X, :P_X], W_nt_stable_aligned, binary_input=True))

    W_dg_stable, _ = fit_diff_struct_em_stability(
        data.X, P_X, P_H, n_inits=STAB_N_INITS, base_init_seed=1000 * seed + 500,
        edge_thresh=STAB_EDGE_THRESH, freq_thresh=STAB_FREQ_THRESH,
        n_outer=N_OUTER, m_step_iters=M_STEP_ITERS,
        lam=LAM, rho=RHO, acyclicity="dagma", init_mode="random",
        enforce_improving=True,
    )
    W_dg_stable_aligned = align_latent_labels(W_dg_stable, W_true_full, P_X, P_H, thresh=0.5)
    rows.append(_row("dagma_latent_stable", n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_dg_stable[:P_X, :P_X], W_dg_stable_aligned, binary_input=True))

    # DAGMA + latents + factor warmstart (principled init).
    res_dg_warm = fit_diff_struct_em(
        data.X, P_X, P_H, n_outer=N_OUTER, m_step_iters=M_STEP_ITERS,
        lam=LAM, latent_l1_scale=0.0, rho=RHO,
        acyclicity="dagma", init_seed=seed, init_mode="factor_warmstart",
        e_step_mode="full", enforce_improving=True,
    )
    W_warm = res_dg_warm.W
    W_warm_aligned = align_latent_labels(W_warm, W_true_full, P_X, P_H, thresh=LATENT_THRESH)
    rows.append(_row("dagma_latent_warmstart", n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_warm[:P_X, :P_X], W_warm_aligned, binary_input=False))

    # NOTEARS + latents + factor warmstart (NEW in v7).
    res_nt_warm = fit_diff_struct_em(
        data.X, P_X, P_H, n_outer=N_OUTER, m_step_iters=M_STEP_ITERS,
        lam=LAM, latent_l1_scale=0.0, rho=RHO,
        acyclicity="notears", init_seed=seed, init_mode="factor_warmstart",
        e_step_mode="full", enforce_improving=True,
    )
    W_nt_warm = res_nt_warm.W
    W_nt_warm_aligned = align_latent_labels(W_nt_warm, W_true_full, P_X, P_H, thresh=LATENT_THRESH)
    rows.append(_row("notears_latent_warmstart", n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_nt_warm[:P_X, :P_X], W_nt_warm_aligned, binary_input=False))

    # L1 ablation: latent_l1_scale=1.0 (symmetric regularization).
    res_dg_sym = fit_diff_struct_em(
        data.X, P_X, P_H, n_outer=N_OUTER, m_step_iters=M_STEP_ITERS,
        lam=LAM, latent_l1_scale=1.0, rho=RHO,
        acyclicity="dagma", init_seed=seed, init_mode="random",
        e_step_mode="full", enforce_improving=True,
    )
    W_sym = res_dg_sym.W
    W_sym_aligned = align_latent_labels(W_sym, W_true_full, P_X, P_H, thresh=LATENT_THRESH)
    rows.append(_row("dagma_latent_symL1", n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_sym[:P_X, :P_X], W_sym_aligned, binary_input=False))

    # Oracle.
    W_oracle = fit_oracle_fully_observed_dagma(data.X, data.H, P_X, P_H, lam=LAM, init_seed=seed)
    rows.append(_row("oracle_dagma", n, seed, n_confounded, W_true_full, W_true_obs, pairs,
                     W_oracle[:P_X, :P_X], W_oracle, binary_input=False))

    elapsed = time.time() - t0
    _append_rows(rows)

    def _sp(name):
        return next(r["spurious_confounder_edges"] for r in rows if r["method"] == name)
    def _lr(name):
        v = next(r["latent_parent_recoveries"] for r in rows if r["method"] == name)
        return str(v) if v != "" else "-"
    print(
        f"[n={n} seed={seed}] {elapsed:.0f}s  confounded={n_confounded}  "
        f"spurious DG obs/sgl/stbl/warm/symL1 = "
        f"{_sp('dagma_obs')}/{_sp('dagma_latent_single')}/"
        f"{_sp('dagma_latent_stable')}/{_sp('dagma_latent_warmstart')}/"
        f"{_sp('dagma_latent_symL1')}  "
        f"NT obs/sgl/stbl/warm = "
        f"{_sp('notears_obs')}/{_sp('notears_latent_single')}/"
        f"{_sp('notears_latent_stable')}/{_sp('notears_latent_warmstart')}  "
        f"orc={_sp('oracle_dagma')}  "
        f"warm-recov DG/NT/orc = "
        f"{_lr('dagma_latent_warmstart')}/{_lr('notears_latent_warmstart')}/"
        f"{_lr('oracle_dagma')}",
        flush=True,
    )


def main():
    n = int(sys.argv[1])
    s0 = int(sys.argv[2])
    s1 = int(sys.argv[3])
    done = _already_done()
    for seed in range(s0, s1):
        if (n, seed) in done:
            print(f"skip (n={n}, seed={seed}) already done")
            continue
        run_one(n, seed)


if __name__ == "__main__":
    main()
