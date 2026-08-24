# Differentiable Structural EM — Experiments

Self-contained experimental code. adding **warm-start
variants for observed-only baselines**, so warm-start initialization can be
compared head-to-head between obs-only and latent methods, for both NOTEARS
and DAGMA.



**Two new methods:** `notears_obs_warmstart` and `dagma_obs_warmstart`.
These fit observed-only NOTEARS / DAGMA but initialize $W$ from a ridge-OLS
regression of each variable on all others, instead of from random noise.
This is the **obs-only analogue** of v0's latent factor warm-start: both
initializations use the empirical second-moment structure of the data
($X^\top X / n$), but applied to different parts of the model (observed
adjacency for obs-only, latent rows for latent-wrapped).

**The new scientific question:** does warm-start's benefit come from latent
modeling, or from initialization in general? If `obs_warmstart` already
matches the latent warm-start methods, then latent modeling is doing nothing
beyond initialization (a strong falsifier for the paper). If latent
warm-start strictly dominates `obs_warmstart`, then latent modeling is
contributing real explanatory capacity.

The experimental matrix is now 12 methods spanning five axes:

| axis | values |
|---|---|
| acyclicity    | NOTEARS, DAGMA |
| observability | observed-only, +latents |
| stability     | single fit, stability selection |
| init          | random, OLS warm-start (obs-only), factor warm-start (latent) |
| latent L1     | 0.0 (default), 1.0 (symmetric, ablation) |

## Method index

| method | acyclicity | observability | init | notes |
|---|---|---|---|---|
| `notears_obs`              | NOTEARS | obs-only | random | baseline |
| `notears_obs_warmstart`    | NOTEARS | obs-only | OLS warm-start | NEW in v1 |
| `dagma_obs`                | DAGMA   | obs-only | random | baseline |
| `dagma_obs_warmstart`      | DAGMA   | obs-only | OLS warm-start | NEW in v1 |
| `notears_latent_single`    | NOTEARS | +latents | random | single fit |
| `notears_latent_stable`    | NOTEARS | +latents | random | stability sel. |
| `notears_latent_warmstart` | NOTEARS | +latents | factor warm-start | |
| `dagma_latent_single`      | DAGMA   | +latents | random | single fit |
| `dagma_latent_stable`      | DAGMA   | +latents | random | stability sel. |
| `dagma_latent_warmstart`   | DAGMA   | +latents | factor warm-start | v0 headline |
| `dagma_latent_symL1`       | DAGMA   | +latents | random | L1 ablation (symmetric) |
| `oracle_dagma`             | DAGMA   | oracle   | random | upper bound |

## Two warm-starts, one principle

Both warm-start strategies use the same idea — initialize from the empirical
second-moment structure of the data — applied to different parts of the model:

- **Latent factor warm-start** (`fit_diff_struct_em(init_mode="factor_warmstart")`):
  initializes the **latent rows** of $W$ from the top-$p_h$ eigenvectors of
  $X^\top X / n$ scaled by $\sqrt{\lambda_k}$. Classical PCA / factor
  analysis init.
- **OLS warm-start** (`fit_observed_only_*(init_mode="ols_warmstart")`):
  initializes the **observed adjacency** $W$ from per-variable ridge-OLS
  regressions. The unconstrained Gaussian-MLE solution that the M-step would
  converge to without the acyclicity penalty. The optimizer then sparsifies
  it into a DAG, rather than discovering it from random noise.

Both are principled latent-variable / DAG initialization strategies adapted
from the classical literature; neither is an ad-hoc trick.

## Files

```
sem.py              SEM generator with confounding_regime arg.
diff_struct_em.py   Proposed method. acyclicity ∈ {notears, dagma}, init_mode
                    ∈ {random, factor_warmstart}.
baselines.py        Observed-only + oracle baselines. Now also init_mode ∈
                    {random, ols_warmstart} for obs-only methods.
metrics.py          Graph-recovery metrics.
experiments.py      Runs both experiments and produces all figures (PDF).
run_exp2_chunk.py   Checkpointed one-seed-at-a-time Exp 2 runner (12 methods).
```

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib torch

# Full run (10 seeds × 3 n; ~4-5 hours on a laptop CPU)
python experiments.py

# Smoke (3 seeds × 2 n; ~20 min)
python experiments.py --fast

# Chunked (resume-safe; recommended for long sessions)
python run_exp2_chunk.py 200  0 10
python run_exp2_chunk.py 1000 0 10
python run_exp2_chunk.py 5000 0 10
```

## Figures produced (all PDF)

### Experiment 1 (2×2 ablation)
`exp1_monotonicity.pdf`, `exp1_aggregate.pdf`

### Experiment 2 — main comparisons
- `exp2_spurious_edges.pdf`, `exp2_shd.pdf`, `exp2_precision.pdf`,
  `exp2_recall.pdf` — across HEADLINE_METHODS (7 methods including both
  obs+OLS and latent+factor warmstarts)
- `exp2_latent_recovery.pdf`, `exp2_shd_full.pdf`, etc. — latent-modeling
  methods only

### Experiment 2 — focused comparisons
- `exp2_grid2x2_spurious.pdf` — NOTEARS/DAGMA × obs/latent (single fit)
- `exp2_warmstart_across_acyclicities_spurious.pdf` and `_latent.pdf` —
  warmstart benefit within each acyclicity family
- `exp2_warmstart_obs_vs_latent_spurious.pdf` and `_shd.pdf` — **NEW in v1**:
  OLS obs-warmstart vs factor latent-warmstart, head-to-head
- `exp2_stability_comparison.pdf`, `_latent.pdf` — single fit vs stability
- `exp2_modeling_ablation_spurious.pdf`, `_latent.pdf` — DAGMA L1 / warmstart

### Per-seed strip plots
`exp2_strip_spurious.pdf`, `exp2_strip_latent_recovery.pdf`

## What the results should tell you

Three predictions worth testing:

**1. OLS warm-start should help obs-only by very little** (or not at all).
Obs-only's ceiling on confounding metrics is structural — without latent
capacity, the model can't represent a hidden common cause regardless of
where it starts. A small smoke-test confirmed this on one seed (SHD=6 with
or without OLS warm-start). At 10 seeds we expect the same.

**2. Factor warm-start (latent) should still dominate.** The v0 result was
warmstart ≈ oracle on latent recovery; v1 should show this gap is
specifically about latent modeling, not initialization quality.

**3. Both warm-starts should be acyclicity-agnostic.** NOTEARS and DAGMA
versions of OLS warm-start should perform similarly to each other. Same for
factor warm-start. This supports the orthogonality-of-acyclicity claim.

If those predictions hold, the paper's strongest framing is:
*"Latent-variable modeling and principled initialization are both necessary
for recovery of confounding structure: initialization without latent capacity
cannot represent confounders (cf. obs-warmstart results), and latent capacity
without principled initialization gets stuck in local optima
(cf. latent-single results). Combined, the framework recovers confounding
structure at near-oracle fidelity."*

If `obs_warmstart` matches `latent_warmstart`, that's a much weaker
contribution and you should report it honestly — but I expect it won't.



