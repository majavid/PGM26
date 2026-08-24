"""
Experiment paralleling Friedman §5.3 (UAI 1998): learning structure when
the true network has hidden variables and the learner does not know how
many.

Setup:
    - True networks: mediator-9x2 and confounder-8x3 (Friedman's
      Figure 1a and 1b adapted to linear-Gaussian; see friedman_sem.py).
    - Sample sizes n_train in {500, 1000, 2000, 4000}, mirroring Friedman.
    - For each (network, n_train, seed): sweep p_h_fit in {0, 1, 2, 3, 4}
      and fit our method under three configurations:
          (i)   random-init single fit
          (ii)  factor-warmstart single fit
          (iii) observed-only DAGMA (only at p_h_fit = 0)
    - Headline metric: test-set log-loss difference vs ground-truth (i.e.
      empirical KL up to the test entropy), paralleling Friedman's Table 2.
    - Test set: n_test = 5000 samples drawn independently from the same SEM.
    - Model selection: pick best p_h_fit per (n_train, seed) by held-out
      log-loss, paralleling Friedman's "select the network with highest score".

Outputs:
    results/friedman_rows.csv  (one row per method-config x p_h_fit x cell)
    figures/friedman_*.pdf
"""
from __future__ import annotations
import csv, os, time
import numpy as np
import matplotlib.pyplot as plt

from friedman_sem import (
    generate_mediator_9x2, generate_confounder_8x3, confounded_observed_pairs,
)
from test_loglik import kl_loss_vs_truth
from diff_struct_em import fit_diff_struct_em
from baselines import fit_observed_only_dagma
from metrics import count_latent_parent_identifications, align_latent_labels


FIG_DIR = "figures"
RES_DIR = "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
CSV_PATH = os.path.join(RES_DIR, "friedman_rows.csv")

NETWORKS = {
    "mediator-9x2":   generate_mediator_9x2,
    "confounder-8x3": generate_confounder_8x3,
}
N_TRAIN_VALUES = (500, 1000, 2000, 4000)
P_H_FIT_VALUES = (0, 1, 2, 3, 4)
N_SEEDS = 10
N_TEST = 5000

# Inner-fit hyperparameters (consistent with v7 production settings).
N_OUTER = 15
M_STEP_ITERS = 300
LAM = 0.02
RHO = 0.5

FIELDS = [
    "network", "n_train", "seed", "p_h_fit", "method",
    "kl_test", "latent_recov", "n_confounded", "fit_time_sec",
]


def _append_rows(rows: list[dict]) -> None:
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def _already_done(network: str, n_train: int, seed: int) -> bool:
    """Resume safety: skip a (network, n_train, seed) cell if all rows are present."""
    if not os.path.exists(CSV_PATH):
        return False
    expected_rows_per_cell = (
        len(P_H_FIT_VALUES) * 2   # random + warmstart for each p_h_fit
        + 1                       # obs-only DAGMA at p_h_fit = 0
    )
    count = 0
    with open(CSV_PATH, "r") as f:
        for r in csv.DictReader(f):
            if (r["network"] == network
                    and int(r["n_train"]) == n_train
                    and int(r["seed"]) == seed):
                count += 1
    return count >= expected_rows_per_cell


def _fit_one_cell(network: str, n_train: int, seed: int) -> list[dict]:
    """Fit all method x p_h_fit combinations for one cell, return rows."""
    gen = NETWORKS[network]
    data = gen(n_train=n_train, n_test=N_TEST, seed=seed)
    p_x = data.p_x
    p_h_true = data.p_h
    pairs = confounded_observed_pairs(data)
    n_confounded = len(pairs)
    rows: list[dict] = []

    # --- Observed-only DAGMA (only meaningful at p_h_fit = 0) ---
    t0 = time.time()
    W_obs = fit_observed_only_dagma(data.X_train, lam=LAM, init_seed=seed)
    elapsed = time.time() - t0
    kl = kl_loss_vs_truth(
        data.X_test,
        W_fit=W_obs, sigma2_fit=1.0,
        W_true=data.W_true, sigma2_true=data.sigma2,
        p_x=p_x,
    )
    rows.append(dict(
        network=network, n_train=n_train, seed=seed, p_h_fit=0,
        method="obs_only_dagma", kl_test=kl,
        latent_recov=float("nan"), n_confounded=n_confounded,
        fit_time_sec=elapsed,
    ))

    # --- Sweep p_h_fit for both random-init and factor-warmstart ---
    for p_h_fit in P_H_FIT_VALUES:
        for init_mode, label in [("random", "latent_random"),
                                  ("factor_warmstart", "latent_warmstart")]:
            t0 = time.time()
            try:
                res = fit_diff_struct_em(
                    data.X_train, p_x, p_h_fit,
                    n_outer=N_OUTER, m_step_iters=M_STEP_ITERS,
                    lam=LAM, latent_l1_scale=0.0, rho=RHO,
                    acyclicity="dagma", init_seed=seed, init_mode=init_mode,
                    e_step_mode="full", enforce_improving=True,
                )
                W_fit = res.W
                sigma2_fit = float(res.sigma2)
                kl = kl_loss_vs_truth(
                    data.X_test,
                    W_fit=W_fit, sigma2_fit=sigma2_fit,
                    W_true=data.W_true, sigma2_true=data.sigma2,
                    p_x=p_x,
                )
                # Latent-parent recovery: align fitted latent block to truth,
                # then count how many ground-truth confounded pairs are recovered.
                # Only meaningful if p_h_fit >= p_h_true (otherwise alignment is
                # ill-defined). For p_h_fit < p_h_true we report NaN.
                if p_h_fit > 0 and p_h_fit >= p_h_true and n_confounded > 0:
                    # Pad fit to match true graph dimensions for alignment if needed.
                    # align_latent_labels expects W_fit and W_true to have the same shape.
                    p_full_true = p_x + p_h_true
                    p_full_fit = p_x + p_h_fit
                    if p_full_fit > p_full_true:
                        # Pad W_true with zero rows/cols to match.
                        W_true_padded = np.zeros((p_full_fit, p_full_fit))
                        W_true_padded[:p_full_true, :p_full_true] = data.W_true
                        W_aligned = align_latent_labels(
                            W_fit, W_true_padded, p_x, p_h_fit, thresh=0.15)
                    else:
                        W_aligned = align_latent_labels(
                            W_fit, data.W_true, p_x, p_h_true, thresh=0.15)
                    n_recov = count_latent_parent_identifications(
                        W_aligned, pairs, p_x, max(p_h_fit, p_h_true), thresh=0.15)
                else:
                    n_recov = float("nan")
            except Exception as e:
                kl = float("nan")
                n_recov = float("nan")
                print(f"  WARN: {label} p_h_fit={p_h_fit} failed: {e}")
            elapsed = time.time() - t0
            rows.append(dict(
                network=network, n_train=n_train, seed=seed, p_h_fit=p_h_fit,
                method=label, kl_test=kl,
                latent_recov=n_recov, n_confounded=n_confounded,
                fit_time_sec=elapsed,
            ))

    return rows


def run_all(networks=None, n_train_values=None, seeds=None) -> None:
    """Main loop. Resume-safe via per-cell skip."""
    networks = networks or list(NETWORKS.keys())
    n_train_values = n_train_values or N_TRAIN_VALUES
    seeds = seeds or range(N_SEEDS)

    for network in networks:
        for n_train in n_train_values:
            for seed in seeds:
                if _already_done(network, n_train, seed):
                    print(f"skip {network} n={n_train} seed={seed} (done)")
                    continue
                t0 = time.time()
                rows = _fit_one_cell(network, n_train, seed)
                _append_rows(rows)
                # Quick inline summary: best random vs best warmstart at the
                # true p_h, and obs-only baseline.
                def _kl(method, p_h_fit):
                    for r in rows:
                        if r["method"] == method and r["p_h_fit"] == p_h_fit:
                            return r["kl_test"]
                    return None
                p_h_true = 2 if network == "mediator-9x2" else 3
                obs_kl = _kl("obs_only_dagma", 0)
                rand_kl = _kl("latent_random", p_h_true)
                warm_kl = _kl("latent_warmstart", p_h_true)
                print(f"[{network} n={n_train} seed={seed}] "
                      f"({time.time()-t0:.0f}s) "
                      f"obs={obs_kl:.3f}  rand@p_h={p_h_true}={rand_kl:.3f}  "
                      f"warm@p_h={p_h_true}={warm_kl:.3f}",
                      flush=True)


# -----------------------------------------------------------------------------
# Aggregation and plotting
# -----------------------------------------------------------------------------

def _load_rows() -> list[dict]:
    if not os.path.exists(CSV_PATH):
        return []
    out = []
    with open(CSV_PATH, "r") as f:
        for r in csv.DictReader(f):
            out.append({
                "network": r["network"],
                "n_train": int(r["n_train"]),
                "seed": int(r["seed"]),
                "p_h_fit": int(r["p_h_fit"]),
                "method": r["method"],
                "kl_test": float(r["kl_test"]),
                "latent_recov": (float(r["latent_recov"])
                                  if r.get("latent_recov", "") not in ("", "nan")
                                  else float("nan")),
                "n_confounded": int(r.get("n_confounded", 0) or 0),
                "fit_time_sec": float(r["fit_time_sec"]),
            })
    return out


def _aggregate(rows: list[dict]) -> list[dict]:
    """Mean +/- SE per (network, n_train, p_h_fit, method) cell."""
    keys: dict = {}
    for r in rows:
        if np.isnan(r["kl_test"]):
            continue
        k = (r["network"], r["n_train"], r["p_h_fit"], r["method"])
        keys.setdefault(k, []).append(r["kl_test"])
    summary = []
    for k, values in keys.items():
        v = np.array(values)
        summary.append(dict(
            network=k[0], n_train=k[1], p_h_fit=k[2], method=k[3],
            kl_mean=float(v.mean()),
            kl_se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
            n_seeds=len(v),
        ))
    return summary


def plot_kl_vs_phfit(summary: list[dict], out_dir: str = FIG_DIR) -> None:
    """One figure per network: KL vs p_h_fit, lines for each method,
    panels for each n_train."""
    METHOD_LABELS = {
        "obs_only_dagma":  "DAGMA obs-only",
        "latent_random":   "Latent EM, random init",
        "latent_warmstart": "Latent EM, factor warm-start",
    }
    METHOD_COLORS = {
        "obs_only_dagma":  "#d62728",
        "latent_random":   "#aec7e8",
        "latent_warmstart": "#17becf",
    }

    for network in NETWORKS:
        p_h_true = 2 if network == "mediator-9x2" else 3
        fig, axes = plt.subplots(1, len(N_TRAIN_VALUES),
                                 figsize=(3.5 * len(N_TRAIN_VALUES), 3.6),
                                 sharey=True)
        for ax, n_train in zip(axes, N_TRAIN_VALUES):
            for method, label in METHOD_LABELS.items():
                xs, ys, errs = [], [], []
                for p_h_fit in P_H_FIT_VALUES:
                    if method == "obs_only_dagma" and p_h_fit != 0:
                        continue
                    matches = [s for s in summary
                               if s["network"] == network
                               and s["n_train"] == n_train
                               and s["p_h_fit"] == p_h_fit
                               and s["method"] == method]
                    if not matches:
                        continue
                    xs.append(p_h_fit)
                    ys.append(matches[0]["kl_mean"])
                    errs.append(matches[0]["kl_se"])
                if not xs:
                    continue
                if method == "obs_only_dagma":
                    # Single point at p_h_fit = 0; draw as a marker only.
                    ax.errorbar(xs, ys, yerr=errs, fmt="s", color=METHOD_COLORS[method],
                                label=label, markersize=8, capsize=3)
                else:
                    ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=METHOD_COLORS[method],
                                label=label, markersize=6, capsize=3, lw=1.5)
            # Mark the true p_h.
            ax.axvline(p_h_true, color="gray", ls="--", lw=0.8, alpha=0.6)
            ax.text(p_h_true, ax.get_ylim()[1] * 0.92,
                    f"true p_h={p_h_true}", ha="center", fontsize=8, color="gray")
            ax.set_xlabel("$p_h^{\\mathrm{fit}}$")
            ax.set_title(f"$n_{{\\mathrm{{train}}}}={n_train}$")
            ax.grid(alpha=0.3)
            ax.set_xticks(list(P_H_FIT_VALUES))
        axes[0].set_ylabel("Test KL vs ground truth\n(lower is better)")
        axes[-1].legend(loc="best", fontsize=8, frameon=True)
        fig.suptitle(f"Friedman §5.3 replication: {network}", y=1.02)
        fig.tight_layout()
        out = os.path.join(out_dir, f"friedman_{network.replace('-', '_')}.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


def model_selection_table(summary: list[dict]) -> str:
    """For each (network, n_train, method): pick the best p_h_fit by mean KL,
    report (best p_h_fit, KL at that p_h, KL at true p_h, KL at p_h=0).
    Paralleling Friedman's Table 2 'pick highest score' analysis."""
    lines = ["", "=== Model-selection table ===",
             "best p_h_fit chosen by held-out mean KL across seeds.",
             ""]
    for network in NETWORKS:
        p_h_true = 2 if network == "mediator-9x2" else 3
        lines.append(f"--- {network} (true p_h = {p_h_true}) ---")
        lines.append(f"{'n_train':>8} {'method':>22}  "
                     f"{'best p_h':>9}  {'KL@best':>8}  "
                     f"{'KL@true':>8}  {'KL@0':>8}")
        for n_train in N_TRAIN_VALUES:
            for method in ("obs_only_dagma", "latent_random", "latent_warmstart"):
                if method == "obs_only_dagma":
                    matches = [s for s in summary
                               if s["network"] == network
                               and s["n_train"] == n_train
                               and s["method"] == method]
                    if matches:
                        kl0 = matches[0]["kl_mean"]
                        lines.append(f"{n_train:>8} {method:>22}  "
                                     f"{'n/a':>9}  {'n/a':>8}  "
                                     f"{'n/a':>8}  {kl0:>+8.3f}")
                    continue
                cell = [s for s in summary
                        if s["network"] == network
                        and s["n_train"] == n_train
                        and s["method"] == method]
                if not cell:
                    continue
                best = min(cell, key=lambda s: s["kl_mean"])
                kl_true = next((s["kl_mean"] for s in cell
                                if s["p_h_fit"] == p_h_true), float("nan"))
                kl0 = next((s["kl_mean"] for s in cell
                            if s["p_h_fit"] == 0), float("nan"))
                lines.append(f"{n_train:>8} {method:>22}  "
                             f"{best['p_h_fit']:>9}  "
                             f"{best['kl_mean']:>+8.3f}  "
                             f"{kl_true:>+8.3f}  {kl0:>+8.3f}")
            lines.append("")
    return "\n".join(lines)


def main():
    run_all()
    rows = _load_rows()
    summary = _aggregate(rows)
    plot_kl_vs_phfit(summary)
    print(model_selection_table(summary))


if __name__ == "__main__":
    main()
