"""
Three experiments, per the revised design:

1. Monotonicity of observed-data log-likelihood (Theorem 2 validation).
2. Ablation: posterior mean only vs full posterior second moments (within Exp 1).
3. Practical benefit under latent confounding: proposed vs observed-only vs oracle.

Outputs:
    figures/*.pdf       (vector, paper-ready)
    results/*.json/csv  (raw + aggregated numbers)

Run locally:
    python experiments.py           # full run  (~25 min on a laptop CPU)
    python experiments.py --fast    # smoke run (~1 min)
"""

from __future__ import annotations
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("pdf")  # vector backend; all saved figures are PDF
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sem import generate_sem, observed_observed_confounded_pairs
from diff_struct_em import fit_diff_struct_em, fit_diff_struct_em_stability
from baselines import (
    fit_observed_only_dagma, fit_observed_only_notears,
    fit_oracle_fully_observed_dagma, fit_oracle_fully_observed_notears,
)
from metrics import (
    structural_hamming_distance,
    precision_recall,
    count_confounder_induced_false_edges,
    count_latent_parent_identifications,
    align_latent_labels,
)


FIG_DIR = "figures"
RES_DIR = "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Method registry (Exp 2)
# -----------------------------------------------------------------------------
# Twelve methods spanning five axes:
#   acyclicity    : NOTEARS or DAGMA
#   observability : observed-only (existing method) or +latents (our wrapper)
#   stability     : single fit or stability selection
#   init          : random init or warm-start (factor for latent, OLS for obs-only)
#   latent L1     : latent_l1_scale = 0.0 (default) or 1.0 (symmetric ablation)
#
# Plus `oracle_dagma` as the upper bound. No wasted cells -- every pairing
# answers one scientific question:
#   obs vs any latent method                  -> does latent wrapping help?
#   _single vs _stable                        -> does stability selection help?
#   _single vs _warmstart                     -> does factor warmstart help (latent)?
#   _obs vs _obs_warmstart                    -> does OLS warmstart help (obs-only)?
#   dagma_latent_single vs dagma_latent_symL1 -> does unregularized latent matter?
#   dagma_*_warmstart vs notears_*_warmstart  -> is warmstart benefit acyclicity-agnostic?
#
# Naming convention:
#   *_obs              = observed-only, random init        (existing methods, baseline)
#   *_obs_warmstart    = observed-only, OLS warm-start     (NEW in v7: tests if init alone
#                                                           closes the gap to latent methods)
#   *_latent_single    = +latents, random init, single fit
#   *_latent_stable    = +latents, random init, stability selection
#   *_latent_warmstart = +latents, factor warm-start
#   *_latent_symL1     = +latents, latent_l1_scale=1.0    (DAGMA only, ablation)
METHODS = [
    "notears_obs",
    "notears_obs_warmstart",      # NEW in v7
    "dagma_obs",
    "dagma_obs_warmstart",        # NEW in v7
    "notears_latent_single",
    "notears_latent_stable",
    "notears_latent_warmstart",
    "dagma_latent_single",
    "dagma_latent_stable",
    "dagma_latent_warmstart",
    "dagma_latent_symL1",
    "oracle_dagma",
]
PRETTY = {
    "notears_obs":              "NOTEARS (obs only)",
    "notears_obs_warmstart":    "NOTEARS (obs only, OLS warm-start)",
    "dagma_obs":                "DAGMA (obs only)",
    "dagma_obs_warmstart":      "DAGMA (obs only, OLS warm-start)",
    "notears_latent_single":    "NOTEARS + latents (single fit)",
    "notears_latent_stable":    "NOTEARS + latents (stability sel.)",
    "notears_latent_warmstart": "NOTEARS + latents (factor warm-start)",
    "dagma_latent_single":      "DAGMA + latents (single fit)",
    "dagma_latent_stable":      "DAGMA + latents (stability sel.)",
    "dagma_latent_warmstart":   "DAGMA + latents (factor warm-start)",
    "dagma_latent_symL1":       "DAGMA + latents (sym. L1, ablation)",
    "oracle_dagma":             "Oracle (fully observed)",
}
COLORS = {
    "notears_obs":              "#ff7f0e",   # orange
    "notears_obs_warmstart":    "#ffbb78",   # light orange (warmstart variant)
    "dagma_obs":                "#d62728",   # red
    "dagma_obs_warmstart":      "#ff9896",   # light red (warmstart variant)
    "notears_latent_single":    "#c5b0d5",   # light purple
    "notears_latent_stable":    "#9467bd",   # dark purple
    "notears_latent_warmstart": "#e377c2",   # pink (NOTEARS+latent warmstart)
    "dagma_latent_single":      "#aec7e8",   # light blue
    "dagma_latent_stable":      "#1f77b4",   # dark blue
    "dagma_latent_warmstart":   "#17becf",   # teal (DAGMA+latent warmstart)
    "dagma_latent_symL1":       "#8c564b",   # brown (ablation)
    "oracle_dagma":             "#2ca02c",   # green
}

# Subsets used by specific plots.
OBSERVED_ONLY_METHODS = (
    "notears_obs", "notears_obs_warmstart",
    "dagma_obs", "dagma_obs_warmstart",
)
LATENT_METHODS = (
    "notears_latent_single", "notears_latent_stable", "notears_latent_warmstart",
    "dagma_latent_single", "dagma_latent_stable", "dagma_latent_warmstart",
    "dagma_latent_symL1",
    "oracle_dagma",
)
# Headline: representative from each axis. Adds obs+warmstart so the reader
# can see whether a better-initialized observed-only model alone closes the
# latent-modeling gap (it shouldn't, by the paper's argument).
HEADLINE_METHODS = (
    "notears_obs", "notears_obs_warmstart", "notears_latent_warmstart",
    "dagma_obs", "dagma_obs_warmstart", "dagma_latent_warmstart",
    "oracle_dagma",
)
# 2x2 (NOTEARS/DAGMA) x (obs/latent) using single-fit:
GRID_2X2_METHODS = (
    "notears_obs", "notears_latent_single",
    "dagma_obs", "dagma_latent_single",
)
# Stability-selection comparison:
STABILITY_COMPARISON_METHODS = (
    "dagma_latent_single", "dagma_latent_stable",
    "notears_latent_single", "notears_latent_stable",
)
# Ablation on modeling choices (L1 asymmetry, warmstart) -- DAGMA only.
MODELING_ABLATION_METHODS = (
    "dagma_latent_single",       # default (unregularized latents, random init)
    "dagma_latent_symL1",        # symmetric L1 (strong regularization)
    "dagma_latent_warmstart",    # factor warmstart
)
# Warmstart-across-acyclicities comparison.
WARMSTART_ACROSS_ACYCLICITIES = (
    "notears_obs", "notears_latent_single", "notears_latent_warmstart",
    "dagma_obs",   "dagma_latent_single",   "dagma_latent_warmstart",
)
# NEW in v7: warmstart effect comparison (obs-only vs latent), within each acyclicity.
# Tests whether warmstart's benefit comes from latent modeling or just from a
# better starting point in general.
WARMSTART_OBS_VS_LATENT = (
    "notears_obs", "notears_obs_warmstart", "notears_latent_warmstart",
    "dagma_obs",   "dagma_obs_warmstart",   "dagma_latent_warmstart",
)


# =============================================================================
# Experiment 1 + Ablation: monotonicity
# =============================================================================

def run_experiment_1(
    n_seeds: int = 10,
    n: int = 1000,
    p_x: int = 8,
    p_h: int = 2,
    n_outer: int = 15,
    m_step_iters: int = 300,
) -> dict:
    """
    Exp 1 runs a 2x2 ablation crossing:
        E-step variant           : {full posterior (1st + 2nd moments), mean-only imputation}
        improving-M-step safeguard: {enforced, not enforced}

    Reporting all four cells lets the reader separate two effects that were
    previously confounded:
      - Does dropping posterior second moments break monotonicity (the paper's
        "1st AND 2nd moments" emphasis)?
      - Does removing the improving-M-step safeguard (i.e. not checking that
        each M-step actually improves Q) break monotonicity?

    Theorem 2 requires BOTH conditions -- exact q_t = p(H|X; W_t, theta_t), and
    F(q_t, W_{t+1}, theta_{t+1}) >= F(q_t, W_t, theta_t). The 2x2 decomposes
    those assumptions empirically.
    """
    # Configurations: (e_step_mode, enforce_improving, human-readable label).
    configs = [
        ("full",      True,  "full_enforce"),
        ("full",      False, "full_noenforce"),
        ("mean_only", True,  "mean_enforce"),
        ("mean_only", False, "mean_noenforce"),
    ]
    traces: dict[str, list] = {label: [] for _, _, label in configs}
    neg_step_counts: dict[str, list] = {label: [] for _, _, label in configs}

    for seed in range(n_seeds):
        data = generate_sem(p_x=p_x, p_h=p_h, n=n, seed=seed)
        for e_mode, enforce, label in configs:
            res = fit_diff_struct_em(
                data.X, p_x, p_h,
                n_outer=n_outer, m_step_iters=m_step_iters,
                lam=0.02, rho=0.5,
                init_seed=seed,
                e_step_mode=e_mode,
                enforce_improving=enforce,
            )
            traces[label].append(list(map(float, res.loglik_trace)))
            diffs = np.diff(res.loglik_trace)
            neg_step_counts[label].append(int((diffs < -1e-3).sum()))

    # Summary: per-configuration counts.
    summary = {"configs": [], "n_seeds": n_seeds, "n_total_steps_per_seed": n_outer}
    for _, _, label in configs:
        summary["configs"].append({
            "label": label,
            "total_negative_steps": int(sum(neg_step_counts[label])),
            "seeds_with_any_decrease": int(sum(1 for c in neg_step_counts[label] if c > 0)),
        })
    # Backwards-compatible top-level keys: "full" == "full_enforce", the
    # theorem-respecting configuration; "mean_only" == "mean_noenforce", the
    # fully-unguarded ablation.
    summary["full_total_negative_steps"] = summary["configs"][0]["total_negative_steps"]
    summary["full_seeds_with_any_decrease"] = summary["configs"][0]["seeds_with_any_decrease"]
    summary["mean_only_total_negative_steps"] = summary["configs"][3]["total_negative_steps"]
    summary["mean_only_seeds_with_any_decrease"] = summary["configs"][3]["seeds_with_any_decrease"]

    with open(os.path.join(RES_DIR, "exp1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RES_DIR, "exp1_traces.json"), "w") as f:
        json.dump(traces, f)

    plot_exp1_trajectories(traces, os.path.join(FIG_DIR, "exp1_monotonicity.pdf"))
    plot_exp1_aggregate_2x2(summary, os.path.join(FIG_DIR, "exp1_aggregate.pdf"))

    return {"traces": traces, "neg_step_counts": neg_step_counts, "summary": summary}


def plot_exp1_trajectories(traces: dict, out_path: str) -> None:
    """
    Trajectory figure with a 2x2 grid of subplots -- one per ablation cell --
    and a two-category legend (monotone vs non-monotone).

    Expects `traces` keyed by four configuration labels produced by
    run_experiment_1: "full_enforce", "full_noenforce", "mean_enforce",
    "mean_noenforce". Each trajectory is min-max normalized to [0, 1] so the
    SHAPE is visible regardless of absolute height.
    """
    MONO = "#1f77b4"
    BAD = "#d62728"
    # Grid layout: rows = E-step variant, cols = improving safeguard.
    panels = [
        ("full_enforce",   "Full E-step  +  enforce improving\n(theorem respects)"),
        ("full_noenforce", "Full E-step  +  NO enforcement\n(theorem assumption (b) violated)"),
        ("mean_enforce",   "Mean-only E-step  +  enforce improving\n(theorem assumption (a) violated)"),
        ("mean_noenforce", "Mean-only E-step  +  NO enforcement\n(both assumptions violated)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True, sharex=True)
    for ax, (label, title) in zip(axes.flat, panels):
        mode_traces = [np.asarray(t) for t in traces.get(label, [])]
        n_seeds = len(mode_traces)
        n_bad = 0
        for tr in mode_traces:
            lo, hi = float(tr.min()), float(tr.max())
            y = (tr - lo) / (hi - lo) if hi > lo else np.zeros_like(tr)
            is_bad = bool((np.diff(tr) < -1e-3).any())
            color = BAD if is_bad else MONO
            ax.plot(y, color=color,
                    lw=1.6 if is_bad else 1.2,
                    alpha=0.95 if is_bad else 0.55)
            n_bad += int(is_bad)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        handles = [
            Line2D([0], [0], color=MONO, lw=1.6, alpha=0.85,
                   label=f"monotone  ({n_seeds - n_bad}/{n_seeds})"),
            Line2D([0], [0], color=BAD, lw=1.6,
                   label=f"non-monotone  ({n_bad}/{n_seeds})"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)

    for ax in axes[:, 0]:
        ax.set_ylabel("Log-likelihood\n(min-max normalized)")
    for ax in axes[-1, :]:
        ax.set_xlabel("EM outer iteration")
    fig.suptitle("Experiment 1: observed-data log-likelihood trajectories (2 x 2 ablation)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_exp1_aggregate_2x2(summary: dict, out_path: str) -> None:
    """
    Aggregate bars across the 2x2 ablation, showing both:
      (a) % of non-monotone EM steps, and
      (b) % of seeds with any likelihood decrease,
    per configuration. This decomposes the previously-confounded "full vs
    mean-only" comparison into the two independent axes: E-step variant and
    improving-M-step safeguard.
    """
    n_seeds = summary["n_seeds"]
    n_steps_per_seed = summary["n_total_steps_per_seed"]
    total_steps = n_seeds * n_steps_per_seed
    cfgs = summary["configs"]

    labels = {
        "full_enforce":   "Full +\nenforce",
        "full_noenforce": "Full +\nno-enforce",
        "mean_enforce":   "Mean-only +\nenforce",
        "mean_noenforce": "Mean-only +\nno-enforce",
    }
    # Color by whether the configuration matches the theorem's assumptions.
    # Only (full, enforce) is the full GEM procedure; the others violate at least one assumption.
    bar_colors = {
        "full_enforce":   "#1f77b4",
        "full_noenforce": "#ff7f0e",
        "mean_enforce":   "#ff7f0e",
        "mean_noenforce": "#d62728",
    }

    order = ["full_enforce", "full_noenforce", "mean_enforce", "mean_noenforce"]
    by_label = {c["label"]: c for c in cfgs}

    xs = np.arange(len(order))
    xtick_labels = [labels[k] for k in order]
    colors = [bar_colors[k] for k in order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: non-monotone steps.
    ax = axes[0]
    vals_steps = [100 * by_label[k]["total_negative_steps"] / total_steps for k in order]
    counts_steps = [by_label[k]["total_negative_steps"] for k in order]
    bars = ax.bar(xs, vals_steps, color=colors, alpha=0.88)
    for bar, bad in zip(bars, counts_steps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{bad}/{total_steps}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_ylabel("Non-monotone EM steps (% of all steps)")
    ax.set_title("Step-level monotonicity failure rate")
    ax.set_ylim(0, max(max(vals_steps), 1) * 1.25 + 5)
    ax.grid(alpha=0.3, axis="y")

    # Panel B: seeds with any decrease.
    ax = axes[1]
    vals_seeds = [100 * by_label[k]["seeds_with_any_decrease"] / n_seeds for k in order]
    counts_seeds = [by_label[k]["seeds_with_any_decrease"] for k in order]
    bars = ax.bar(xs, vals_seeds, color=colors, alpha=0.88)
    for bar, bad in zip(bars, counts_seeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{bad}/{n_seeds}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_ylabel("Seeds with ≥ 1 likelihood decrease (%)")
    ax.set_title("Seed-level failure rate")
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Experiment 1: Theorem 2 assumptions ablation "
        f"(n_seeds={n_seeds}, {n_steps_per_seed} EM iters each)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Backwards-compatible alias for external callers.
def plot_exp1_aggregate(summary: dict, out_path: str) -> None:
    """Deprecated: prefer plot_exp1_aggregate_2x2. Dispatches to it if the
    summary dict has the new 'configs' key, otherwise falls back to the old
    2-bar layout for backwards compatibility with v4 summaries."""
    if "configs" in summary:
        return plot_exp1_aggregate_2x2(summary, out_path)
    # --- legacy 2-bar fallback (v4 and earlier) ---
    n_seeds = summary["n_seeds"]
    n_steps_per_seed = summary["n_total_steps_per_seed"]
    total_steps = n_seeds * n_steps_per_seed
    full_bad_steps = summary["full_total_negative_steps"]
    mean_bad_steps = summary["mean_only_total_negative_steps"]
    full_bad_seeds = summary["full_seeds_with_any_decrease"]
    mean_bad_seeds = summary["mean_only_seeds_with_any_decrease"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["Full\n(1st + 2nd moments)", "Ablation\n(mean only)"]
    bar_colors = ["#1f77b4", "#d62728"]
    ax = axes[0]
    vals = [100 * full_bad_steps / total_steps, 100 * mean_bad_steps / total_steps]
    bars = ax.bar(labels, vals, color=bar_colors, alpha=0.88)
    for bar, bad in zip(bars, [full_bad_steps, mean_bad_steps]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{bad} / {total_steps}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Non-monotone EM steps (% of all steps)")
    ax.set_title("Theorem 2 validation: step-level monotonicity")
    ax.set_ylim(0, max(vals) * 1.25 + 5)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    vals = [100 * full_bad_seeds / n_seeds, 100 * mean_bad_seeds / n_seeds]
    bars = ax.bar(labels, vals, color=bar_colors, alpha=0.88)
    for bar, bad in zip(bars, [full_bad_seeds, mean_bad_seeds]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{bad} / {n_seeds}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Seeds with ≥ 1 likelihood decrease (%)")
    ax.set_title("Seed-level failure rate")
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Experiment 1 summary (n_seeds={n_seeds}, {n_steps_per_seed} EM iters each)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Experiment 2: structural recovery under latent confounding
# =============================================================================

def run_experiment_2(
    n_seeds: int = 10,
    n_values: tuple[int, ...] = (200, 1000, 5000),
    p_x: int = 8,
    p_h: int = 2,
    thresh: float = 0.3,
    latent_thresh: float = 0.15,
    n_outer: int = 15,
    m_step_iters: int = 300,
    confounding_regime: str = "strong",
    # Stability selection hyperparameters (used by *_stable methods).
    stability_n_inits: int = 16,
    stability_edge_thresh: float = 0.15,
    stability_freq_thresh: float = 0.75,
) -> dict:
    """
    Fit a 9-method experimental matrix (see METHODS registry at top of file)
    on synthetic linear-Gaussian SEMs with explicit latent confounders.

    confounding_regime:
      "strong"   (default): latents parent 3-5 obs nodes with weights in [1,2];
                            observed-observed density 0.1 with weights in [0.3,1].
                            Designed so confounding structure is the dominant
                            source of observed covariance.
      "original": v3-v5 regime; latent and observed-observed weights both in [0.5, 1.5].

    Metrics recorded per method:
      - Observed-subgraph SHD, precision, recall
      - Spurious confounder-induced edges
      - Latent-parent identifications (latent-modeling methods only)
      - Full-graph SHD, precision, recall (latent-modeling methods only)
    """
    rows = []
    for n in n_values:
        for seed in range(n_seeds):
            data = generate_sem(
                p_x=p_x, p_h=p_h, n=n, seed=seed,
                confounding_regime=confounding_regime,
            )
            pairs = observed_observed_confounded_pairs(data)
            n_confounded_pairs = len(pairs)
            W_true_full = data.W_true
            W_true_obs = W_true_full[:p_x, :p_x]

            def _finalize_row(name: str, W_obs_subgraph: np.ndarray,
                              W_full_aligned: np.ndarray | None,
                              binary_input: bool) -> dict:
                obs_thresh = 0.5 if binary_input else thresh
                lat_thresh = 0.5 if binary_input else latent_thresh
                full_thresh = 0.5 if binary_input else thresh
                shd_obs = structural_hamming_distance(W_true_obs, W_obs_subgraph, thresh=obs_thresh)
                prec_obs, rec_obs = precision_recall(W_true_obs, W_obs_subgraph, thresh=obs_thresh)
                spur = count_confounder_induced_false_edges(W_obs_subgraph, pairs, thresh=obs_thresh)
                row = {
                    "method": name, "n": n, "seed": seed,
                    "n_confounded_pairs": n_confounded_pairs,
                    "shd_obs": shd_obs,
                    "precision_obs": prec_obs,
                    "recall_obs": rec_obs,
                    "spurious_confounder_edges": spur,
                    "shd_full": float("nan"),
                    "precision_full": float("nan"),
                    "recall_full": float("nan"),
                    "latent_parent_recoveries": float("nan"),
                }
                if W_full_aligned is not None:
                    row["shd_full"] = structural_hamming_distance(
                        W_true_full, W_full_aligned, thresh=full_thresh)
                    p_full, r_full = precision_recall(
                        W_true_full, W_full_aligned, thresh=full_thresh)
                    row["precision_full"] = p_full
                    row["recall_full"] = r_full
                    row["latent_parent_recoveries"] = count_latent_parent_identifications(
                        W_full_aligned, pairs, p_x, p_h, thresh=lat_thresh)
                return row

            t0 = time.time()

            # --- Observed-only baselines (random init) ---
            W_notears_obs = fit_observed_only_notears(data.X, lam=0.02, init_seed=seed)
            W_dagma_obs   = fit_observed_only_dagma  (data.X, lam=0.02, init_seed=seed)
            rows.append(_finalize_row("notears_obs", W_notears_obs, None, binary_input=False))
            rows.append(_finalize_row("dagma_obs",   W_dagma_obs,   None, binary_input=False))

            # --- Observed-only baselines with OLS warm-start (NEW in v7) ---
            # The obs-only analogue of the latent factor warm-start: initialize
            # the observed adjacency from per-variable OLS regressions (the
            # closest unconstrained Gaussian fit), then let the acyclicity
            # penalty sparsify it into a DAG. Tests whether better initialization
            # alone (without latent modeling) closes the confounding-handling gap.
            W_notears_obs_warm = fit_observed_only_notears(
                data.X, lam=0.02, init_seed=seed, init_mode="ols_warmstart")
            W_dagma_obs_warm   = fit_observed_only_dagma(
                data.X, lam=0.02, init_seed=seed, init_mode="ols_warmstart")
            rows.append(_finalize_row("notears_obs_warmstart", W_notears_obs_warm, None, binary_input=False))
            rows.append(_finalize_row("dagma_obs_warmstart",   W_dagma_obs_warm,   None, binary_input=False))

            # --- Latent-wrapped methods (single fit) ---
            # NOTEARS + latents (single fit, random init, latent_l1_scale=0).
            res_nt_single = fit_diff_struct_em(
                data.X, p_x, p_h, n_outer=n_outer, m_step_iters=m_step_iters,
                lam=0.02, latent_l1_scale=0.0, rho=0.5,
                acyclicity="notears", init_seed=seed, init_mode="random",
                e_step_mode="full", enforce_improving=True,
            )
            W_nt_single = res_nt_single.W
            W_nt_single_aligned = align_latent_labels(W_nt_single, W_true_full, p_x, p_h, thresh=latent_thresh)
            rows.append(_finalize_row("notears_latent_single",
                                      W_nt_single[:p_x, :p_x], W_nt_single_aligned,
                                      binary_input=False))

            # DAGMA + latents (single fit, random init, latent_l1_scale=0).
            res_dg_single = fit_diff_struct_em(
                data.X, p_x, p_h, n_outer=n_outer, m_step_iters=m_step_iters,
                lam=0.02, latent_l1_scale=0.0, rho=0.5,
                acyclicity="dagma", init_seed=seed, init_mode="random",
                e_step_mode="full", enforce_improving=True,
            )
            W_dg_single = res_dg_single.W
            W_dg_single_aligned = align_latent_labels(W_dg_single, W_true_full, p_x, p_h, thresh=latent_thresh)
            rows.append(_finalize_row("dagma_latent_single",
                                      W_dg_single[:p_x, :p_x], W_dg_single_aligned,
                                      binary_input=False))

            # --- Latent-wrapped methods (stability selection) ---
            W_nt_stable, _ = fit_diff_struct_em_stability(
                data.X, p_x, p_h,
                n_inits=stability_n_inits, base_init_seed=1000 * seed,
                edge_thresh=stability_edge_thresh, freq_thresh=stability_freq_thresh,
                n_outer=n_outer, m_step_iters=m_step_iters,
                lam=0.02, rho=0.5, acyclicity="notears",
                init_mode="random", enforce_improving=True,
            )
            W_nt_stable_aligned = align_latent_labels(W_nt_stable, W_true_full, p_x, p_h, thresh=0.5)
            rows.append(_finalize_row("notears_latent_stable",
                                      W_nt_stable[:p_x, :p_x], W_nt_stable_aligned,
                                      binary_input=True))

            W_dg_stable, _ = fit_diff_struct_em_stability(
                data.X, p_x, p_h,
                n_inits=stability_n_inits, base_init_seed=1000 * seed + 500,
                edge_thresh=stability_edge_thresh, freq_thresh=stability_freq_thresh,
                n_outer=n_outer, m_step_iters=m_step_iters,
                lam=0.02, rho=0.5, acyclicity="dagma",
                init_mode="random", enforce_improving=True,
            )
            W_dg_stable_aligned = align_latent_labels(W_dg_stable, W_true_full, p_x, p_h, thresh=0.5)
            rows.append(_finalize_row("dagma_latent_stable",
                                      W_dg_stable[:p_x, :p_x], W_dg_stable_aligned,
                                      binary_input=True))

            # --- Latent-wrapped, factor WARMSTART (principled init) ---
            # DAGMA + warmstart
            res_dg_warm = fit_diff_struct_em(
                data.X, p_x, p_h, n_outer=n_outer, m_step_iters=m_step_iters,
                lam=0.02, latent_l1_scale=0.0, rho=0.5,
                acyclicity="dagma", init_seed=seed, init_mode="factor_warmstart",
                e_step_mode="full", enforce_improving=True,
            )
            W_dg_warm = res_dg_warm.W
            W_dg_warm_aligned = align_latent_labels(W_dg_warm, W_true_full, p_x, p_h, thresh=latent_thresh)
            rows.append(_finalize_row("dagma_latent_warmstart",
                                      W_dg_warm[:p_x, :p_x], W_dg_warm_aligned,
                                      binary_input=False))

            # NOTEARS + warmstart (NEW in v7).
            # Tests whether warmstart's benefit transfers across acyclicity choices.
            res_nt_warm = fit_diff_struct_em(
                data.X, p_x, p_h, n_outer=n_outer, m_step_iters=m_step_iters,
                lam=0.02, latent_l1_scale=0.0, rho=0.5,
                acyclicity="notears", init_seed=seed, init_mode="factor_warmstart",
                e_step_mode="full", enforce_improving=True,
            )
            W_nt_warm = res_nt_warm.W
            W_nt_warm_aligned = align_latent_labels(W_nt_warm, W_true_full, p_x, p_h, thresh=latent_thresh)
            rows.append(_finalize_row("notears_latent_warmstart",
                                      W_nt_warm[:p_x, :p_x], W_nt_warm_aligned,
                                      binary_input=False))

            # --- L1 asymmetry ablation: latent_l1_scale=1.0 (symmetric L1) ---
            res_dg_symL1 = fit_diff_struct_em(
                data.X, p_x, p_h, n_outer=n_outer, m_step_iters=m_step_iters,
                lam=0.02, latent_l1_scale=1.0, rho=0.5,
                acyclicity="dagma", init_seed=seed, init_mode="random",
                e_step_mode="full", enforce_improving=True,
            )
            W_dg_symL1 = res_dg_symL1.W
            W_dg_symL1_aligned = align_latent_labels(W_dg_symL1, W_true_full, p_x, p_h, thresh=latent_thresh)
            rows.append(_finalize_row("dagma_latent_symL1",
                                      W_dg_symL1[:p_x, :p_x], W_dg_symL1_aligned,
                                      binary_input=False))

            # --- Oracle ---
            W_oracle_full = fit_oracle_fully_observed_dagma(
                data.X, data.H, p_x, p_h, lam=0.02, init_seed=seed)
            rows.append(_finalize_row("oracle_dagma",
                                      W_oracle_full[:p_x, :p_x], W_oracle_full,
                                      binary_input=False))

            elapsed = time.time() - t0
            # Short inline report on the key axis comparisons.
            def _sp(name):
                return next(r["spurious_confounder_edges"] for r in rows[-12:] if r["method"] == name)
            def _lr(name):
                v = next(r["latent_parent_recoveries"] for r in rows[-12:] if r["method"] == name)
                return f"{v:.0f}" if not np.isnan(v) else "-"
            print(
                f"[exp2 n={n} seed={seed}] ({elapsed:.0f}s) confounded={n_confounded_pairs}  "
                f"spurious DG obs/sgl/stbl/warm/symL1 = "
                f"{_sp('dagma_obs')}/{_sp('dagma_latent_single')}/"
                f"{_sp('dagma_latent_stable')}/{_sp('dagma_latent_warmstart')}/"
                f"{_sp('dagma_latent_symL1')}  "
                f"NT obs/sgl/stbl/warm = "
                f"{_sp('notears_obs')}/{_sp('notears_latent_single')}/"
                f"{_sp('notears_latent_stable')}/{_sp('notears_latent_warmstart')}  "
                f"orc={_sp('oracle_dagma')}  "
                f"latent-recov DG warm/orc = "
                f"{_lr('dagma_latent_warmstart')}/{_lr('oracle_dagma')}  "
                f"NT warm = {_lr('notears_latent_warmstart')}",
                flush=True,
            )

    csv_path = os.path.join(RES_DIR, "exp2_rows.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = _aggregate_exp2(rows)
    with open(os.path.join(RES_DIR, "exp2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------

    # Headline: one representative per method family on the spurious-edges metric.
    plot_exp2_bars(summary, "spurious",
                   ylabel="Spurious confounder-induced edges\n(lower is better)",
                   title="Experiment 2 (headline): spurious edges between observed pairs sharing a latent parent",
                   out_path=os.path.join(FIG_DIR, "exp2_spurious_edges.pdf"),
                   methods_subset=HEADLINE_METHODS)
    plot_exp2_bars(summary, "shd",
                   ylabel="SHD on observed subgraph (lower is better)",
                   title="Experiment 2: SHD (observed subgraph)",
                   out_path=os.path.join(FIG_DIR, "exp2_shd.pdf"),
                   methods_subset=HEADLINE_METHODS)
    plot_exp2_bars(summary, "precision",
                   ylabel="Precision on observed subgraph (higher is better)",
                   title="Experiment 2: edge precision (observed subgraph)",
                   out_path=os.path.join(FIG_DIR, "exp2_precision.pdf"),
                   methods_subset=HEADLINE_METHODS, ylim=(0.0, 1.05))
    plot_exp2_bars(summary, "recall",
                   ylabel="Recall on observed subgraph (higher is better)",
                   title="Experiment 2: edge recall (observed subgraph)",
                   out_path=os.path.join(FIG_DIR, "exp2_recall.pdf"),
                   methods_subset=HEADLINE_METHODS, ylim=(0.0, 1.05))

    # Latent-modeling methods only: full-graph metrics.
    plot_exp2_bars(summary, "latent_recov",
                   ylabel="Confounded pairs connected via a common\nfitted latent parent (higher is better)",
                   title="Experiment 2: latent-parent identification",
                   out_path=os.path.join(FIG_DIR, "exp2_latent_recovery.pdf"),
                   methods_subset=LATENT_METHODS)
    plot_exp2_bars(summary, "shd_full",
                   ylabel="SHD on full (X, H) graph (lower is better)",
                   title="Experiment 2: SHD on full graph",
                   out_path=os.path.join(FIG_DIR, "exp2_shd_full.pdf"),
                   methods_subset=LATENT_METHODS)
    plot_exp2_bars(summary, "precision_full",
                   ylabel="Precision on full (X, H) graph (higher is better)",
                   title="Experiment 2: edge precision on full graph",
                   out_path=os.path.join(FIG_DIR, "exp2_precision_full.pdf"),
                   methods_subset=LATENT_METHODS, ylim=(0.0, 1.05))
    plot_exp2_bars(summary, "recall_full",
                   ylabel="Recall on full (X, H) graph (higher is better)",
                   title="Experiment 2: edge recall on full graph",
                   out_path=os.path.join(FIG_DIR, "exp2_recall_full.pdf"),
                   methods_subset=LATENT_METHODS, ylim=(0.0, 1.05))

    # 2x2 (acyclicity x observability) using single-fit. Tests orthogonality.
    plot_exp2_bars(summary, "spurious",
                   ylabel="Spurious confounder-induced edges\n(lower is better)",
                   title="Experiment 2: latent wrapper benefit within each acyclicity family (single fit)",
                   out_path=os.path.join(FIG_DIR, "exp2_grid2x2_spurious.pdf"),
                   methods_subset=GRID_2X2_METHODS)

    # Stability selection comparison: single vs stable, for both acyclicities.
    plot_exp2_bars(summary, "spurious",
                   ylabel="Spurious confounder-induced edges\n(lower is better)",
                   title="Experiment 2: does stability selection help? (single fit vs consensus)",
                   out_path=os.path.join(FIG_DIR, "exp2_stability_comparison.pdf"),
                   methods_subset=STABILITY_COMPARISON_METHODS)
    plot_exp2_bars(summary, "latent_recov",
                   ylabel="Latent-parent recoveries (higher is better)",
                   title="Experiment 2: does stability selection help? (latent-parent recovery)",
                   out_path=os.path.join(FIG_DIR, "exp2_stability_comparison_latent.pdf"),
                   methods_subset=STABILITY_COMPARISON_METHODS)

    # Modeling-choice ablation: L1 asymmetry and factor warmstart.
    plot_exp2_bars(summary, "spurious",
                   ylabel="Spurious confounder-induced edges\n(lower is better)",
                   title="Experiment 2: modeling-choice ablations (DAGMA+latents, single fit)",
                   out_path=os.path.join(FIG_DIR, "exp2_modeling_ablation_spurious.pdf"),
                   methods_subset=MODELING_ABLATION_METHODS)
    plot_exp2_bars(summary, "latent_recov",
                   ylabel="Latent-parent recoveries (higher is better)",
                   title="Experiment 2: modeling-choice ablations (latent-parent recovery)",
                   out_path=os.path.join(FIG_DIR, "exp2_modeling_ablation_latent.pdf"),
                   methods_subset=MODELING_ABLATION_METHODS)

    # Warmstart across acyclicities (NEW in v7): tests whether warmstart's
    # benefit is acyclicity-agnostic (the same orthogonality argument as the
    # rest of the paper, applied to initialization). Within each acyclicity
    # family we expect the ordering: obs-only > +latents (random) > +latents (warmstart).
    plot_exp2_bars(summary, "spurious",
                   ylabel="Spurious confounder-induced edges\n(lower is better)",
                   title="Experiment 2: warmstart benefit within each acyclicity family",
                   out_path=os.path.join(FIG_DIR, "exp2_warmstart_across_acyclicities_spurious.pdf"),
                   methods_subset=WARMSTART_ACROSS_ACYCLICITIES)
    plot_exp2_bars(summary, "latent_recov",
                   ylabel="Latent-parent recoveries (higher is better)",
                   title="Experiment 2: warmstart benefit on latent-parent recovery (NOTEARS vs DAGMA)",
                   out_path=os.path.join(FIG_DIR, "exp2_warmstart_across_acyclicities_latent.pdf"),
                   methods_subset=WARMSTART_ACROSS_ACYCLICITIES)

    # NEW in v7: OLS warm-start (obs-only) vs factor warm-start (+latents).
    # Tests whether warm-start's benefit comes from latent modeling or just
    # from a better starting point in general. If obs-only OLS warm-start
    # already matches latent warm-start, latent modeling is doing nothing
    # beyond initialization. If latent warm-start strictly dominates, the
    # latent wrapper is contributing real explanatory capacity.
    plot_exp2_bars(summary, "spurious",
                   ylabel="Spurious confounder-induced edges\n(lower is better)",
                   title="Experiment 2: OLS warm-start (obs-only) vs factor warm-start (+latents)",
                   out_path=os.path.join(FIG_DIR, "exp2_warmstart_obs_vs_latent_spurious.pdf"),
                   methods_subset=WARMSTART_OBS_VS_LATENT)
    plot_exp2_bars(summary, "shd",
                   ylabel="SHD on observed subgraph (lower is better)",
                   title="Experiment 2: OLS vs factor warm-start (SHD)",
                   out_path=os.path.join(FIG_DIR, "exp2_warmstart_obs_vs_latent_shd.pdf"),
                   methods_subset=WARMSTART_OBS_VS_LATENT)

    # Per-seed strip plots.
    plot_exp2_strip(rows, value_key="spurious_confounder_edges",
                    ylabel="Spurious confounder-induced edges (per seed)",
                    title="Experiment 2: per-seed spread of spurious edges",
                    out_path=os.path.join(FIG_DIR, "exp2_strip_spurious.pdf"),
                    methods_subset=HEADLINE_METHODS)
    plot_exp2_strip(rows, value_key="latent_parent_recoveries",
                    ylabel="Latent-parent recoveries (per seed)",
                    title="Experiment 2: per-seed latent-parent identifications",
                    out_path=os.path.join(FIG_DIR, "exp2_strip_latent_recovery.pdf"),
                    methods_subset=LATENT_METHODS)

    return {"rows": rows, "summary": summary}


def _aggregate_exp2(rows: list[dict]) -> list[dict]:
    """Aggregate per-(method, n) means, stds, and standard errors for all metrics."""
    agg: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        agg[(r["method"], r["n"])].append(r)

    # Metrics to summarize. Full-graph/latent-recovery metrics will be NaN for
    # observed-only DAGMA; np.nanmean handles that gracefully.
    METRICS = [
        "spurious_confounder_edges",
        "shd_obs", "precision_obs", "recall_obs",
        "shd_full", "precision_full", "recall_full",
        "latent_parent_recoveries",
    ]
    # Short aliases used in the JSON keys.
    ALIAS = {
        "spurious_confounder_edges": "spurious",
        "shd_obs": "shd", "precision_obs": "precision", "recall_obs": "recall",
        "shd_full": "shd_full",
        "precision_full": "precision_full",
        "recall_full": "recall_full",
        "latent_parent_recoveries": "latent_recov",
    }

    summary = []
    for (method, n), rs in sorted(agg.items()):
        row = {"method": method, "n": n, "n_seeds": len(rs)}
        for m in METRICS:
            arr = np.array([r[m] for r in rs], dtype=float)
            valid = arr[~np.isnan(arr)]
            k = len(valid)
            alias = ALIAS[m]
            row[f"{alias}_mean"] = float(np.nanmean(arr)) if k > 0 else float("nan")
            row[f"{alias}_std"] = float(np.nanstd(arr, ddof=0)) if k > 0 else float("nan")
            row[f"{alias}_se"] = float(np.nanstd(arr, ddof=0) / np.sqrt(k)) if k > 0 else float("nan")
        summary.append(row)
    return summary


def _lookup(summary: list[dict], method: str, n: int, key: str) -> float:
    for row in summary:
        if row["method"] == method and row["n"] == n:
            return row[key]
    raise KeyError(f"no row for method={method}, n={n}, key={key}")


def plot_exp2_bars(
    summary: list[dict],
    metric: str,               # e.g. "spurious", "shd", "precision", "recall", "latent_recov"
    ylabel: str,
    title: str,
    out_path: str,
    ylim: tuple[float, float] | None = None,
    methods_subset: tuple[str, ...] | None = None,
) -> None:
    """
    Grouped bar chart with standard-error bars. One bar per method (optionally
    restricted to `methods_subset`) for each sample size, colored consistently
    across all Exp 2 figures.

    Skips a method for a given (method, n) if the value is NaN (e.g. observed-only
    has no full-graph metric).
    """
    mean_key = f"{metric}_mean"
    se_key = f"{metric}_se"
    n_values = sorted({r["n"] for r in summary})
    methods = list(methods_subset) if methods_subset is not None else list(METHODS)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(n_values))
    # Center offsets around 0.
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width

    for k, m in enumerate(methods):
        means = [_lookup(summary, m, n, mean_key) for n in n_values]
        ses = [_lookup(summary, m, n, se_key) for n in n_values]
        # Replace NaN with 0 for the plot bar; we do not plot a bar where data is missing.
        means_plot = [0.0 if np.isnan(v) else v for v in means]
        ses_plot = [0.0 if np.isnan(v) else v for v in ses]
        ax.bar(x + offsets[k], means_plot, width, yerr=ses_plot, capsize=3,
               label=PRETTY[m], color=COLORS[m], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in n_values])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_exp2_strip(
    rows: list[dict],
    value_key: str,
    ylabel: str,
    title: str,
    out_path: str,
    methods_subset: tuple[str, ...] | None = None,
) -> None:
    """Per-seed strip plot of any Exp 2 metric.

    Skips NaN values (e.g. latent_parent_recoveries for observed-only DAGMA).
    """
    n_values = sorted({r["n"] for r in rows})
    methods = list(methods_subset) if methods_subset is not None else list(METHODS)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(n_values), figsize=(4.5 * len(n_values), 4.2), sharey=True)
    if len(n_values) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        for k, m in enumerate(methods):
            vals = [r[value_key] for r in rows
                    if r["method"] == m and r["n"] == n and not np.isnan(r[value_key])]
            if not vals:
                continue
            xpos = np.full(len(vals), k) + rng.normal(0, 0.06, len(vals))
            ax.scatter(xpos, vals, color=COLORS[m], alpha=0.75, s=40,
                       edgecolor="k", linewidth=0.3)
            ax.plot([k - 0.25, k + 0.25], [np.mean(vals)] * 2, color="k", lw=2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([PRETTY[m] for m in methods], rotation=20, ha="right", fontsize=9)
        ax.set_title(f"n = {n}")
        ax.grid(alpha=0.3, axis="y")

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================

def main(fast: bool = False) -> None:
    if fast:
        print("=== FAST MODE: fewer seeds, fewer iters, fewer stability inits ===")
        e1 = run_experiment_1(n_seeds=3, n_outer=8, m_step_iters=150)
        e2 = run_experiment_2(n_seeds=3, n_values=(200, 1000),
                              n_outer=8, m_step_iters=150,
                              stability_n_inits=4, stability_freq_thresh=0.75)
    else:
        e1 = run_experiment_1(n_seeds=10, n_outer=15, m_step_iters=300)
        e2 = run_experiment_2(n_seeds=10, n_values=(200, 1000, 5000),
                              n_outer=15, m_step_iters=300,
                              stability_n_inits=16, stability_freq_thresh=0.75)

    print("\n=== Experiment 1 summary ===")
    print(json.dumps(e1["summary"], indent=2))
    print("\n=== Experiment 2 summary (mean ± SE over seeds) ===")
    for row in e2["summary"]:
        method_str = PRETTY[row["method"]]
        lr_str = (f"  latent-recov={row['latent_recov_mean']:.2f}±{row['latent_recov_se']:.2f}"
                  if not np.isnan(row["latent_recov_mean"]) else "  latent-recov=    n/a   ")
        shd_full_str = (f"  SHDfull={row['shd_full_mean']:.2f}±{row['shd_full_se']:.2f}"
                        if not np.isnan(row["shd_full_mean"]) else "  SHDfull=  n/a ")
        print(f"  {method_str:38s}  n={row['n']:>5d}  "
              f"spurious={row['spurious_mean']:.2f}±{row['spurious_se']:.2f}  "
              f"SHDobs={row['shd_mean']:.2f}±{row['shd_se']:.2f}  "
              f"P={row['precision_mean']:.2f}±{row['precision_se']:.2f}  "
              f"R={row['recall_mean']:.2f}±{row['recall_se']:.2f}"
              f"{lr_str}{shd_full_str}")
    print(f"\nFigures saved to: {FIG_DIR}/   (all PDF)")
    print(f"Numbers saved to: {RES_DIR}/")


if __name__ == "__main__":
    import sys
    fast = "--fast" in sys.argv
    main(fast=fast)
