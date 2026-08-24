"""
Baseline methods for Experiment 2.

Observed-only baselines:
  - fit_observed_only_dagma(X)   : linear-Gaussian DAG over X using DAGMA log-det acyclicity.
  - fit_observed_only_notears(X) : linear-Gaussian DAG over X using NOTEARS matrix-exp acyclicity.
Oracle (fully-observed) baselines:
  - fit_oracle_fully_observed_dagma(X, H)
  - fit_oracle_fully_observed_notears(X, H)

Both acyclicity penalties share the same L2 regression loss and the same rho
schedule, so the comparison is apples-to-apples -- any difference between the
two is attributable to the penalty, not to incidental optimizer choices.

These observed-only / oracle baselines are the standard fully-observed
continuous-optimization DAG learners. Our proposed method wraps them with an
explicit-latent SEM and an exact Gaussian E-step; see diff_struct_em.py.
"""

from __future__ import annotations
import numpy as np
import torch
from diff_struct_em import _acyclicity_penalty


def _ols_warmstart_init(Y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """
    OLS warm-start for an observed-only linear DAG.

    For each variable j, regress Y[:, j] on Y[:, all-other] using ridge-stabilized
    OLS, and place the resulting coefficients in W[:, j]. The diagonal stays zero.
    Result: W is the (unconstrained, non-DAG) solution that minimizes
            (1/2n) ||Y - Y W||_F^2 + (ridge/2) ||W||_F^2.
    The acyclicity-constrained optimization then starts from this point and
    sparsifies it into a DAG, rather than starting from random noise.

    The ridge term is just for numerical stability; with ridge=1e-3 the resulting
    W is essentially the standard OLS estimate.

    This is the obs-only analogue of the latent factor warm-start in
    diff_struct_em._init_W_factor_warmstart: both initialize from the empirical
    second-moment structure of the data, rather than from random init.
    """
    n, p = Y.shape
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    YtY = Y_c.T @ Y_c / n  # (p, p) empirical covariance
    W = np.zeros((p, p))
    for j in range(p):
        # Regress Y[:, j] on all OTHER variables: W[k, j] for k != j.
        idx = np.array([k for k in range(p) if k != j])
        # Closed-form ridge OLS on the (p-1, p-1) submatrix.
        A = YtY[np.ix_(idx, idx)] + ridge * np.eye(p - 1)
        b = YtY[idx, j]
        w_j = np.linalg.solve(A, b)
        W[idx, j] = w_j
    np.fill_diagonal(W, 0.0)
    return W


def _fit_linear_dag(
    Y: np.ndarray,
    *,
    acyclicity: str = "dagma",     # "dagma" or "notears"
    lam: float = 0.02,
    rho_schedule: tuple[float, ...] = (1.0, 10.0, 100.0),
    s: float = 1.0,
    n_iter_per_rho: int = 800,
    lr: float = 0.01,
    init_seed: int = 0,
    init_mode: str = "random",     # "random" or "ols_warmstart"
) -> np.ndarray:
    """
    Fit a linear-Gaussian DAG to data Y using continuous acyclicity-constrained
    optimization.

    Objective (per rho step):
        (1/(2n)) ||Y - Y W||_F^2  +  lam ||W||_1  +  rho * h(W)

    with h chosen per `acyclicity`:
        "dagma"   -> h(W) = -log det(s*I - W*W) + p log s
        "notears" -> h(W) = tr(exp(W*W)) - p

    The rho schedule progressively tightens acyclicity. This is the standard
    augmented-Lagrangian-style strategy used by both NOTEARS and DAGMA.

    init_mode:
      "random"        : v6 default. W initialized from N(0, 0.1).
      "ols_warmstart" : W initialized from ridge-OLS regression of each
                        variable on all others. This is the obs-only analogue
                        of the latent factor warm-start; both use empirical
                        second-moment structure of the data.
    """
    assert init_mode in ("random", "ols_warmstart")
    n, p = Y.shape
    dtype = torch.float64
    device = torch.device("cpu")

    rng = np.random.default_rng(init_seed)
    if init_mode == "random":
        W_init = rng.normal(0, 0.1, size=(p, p))
    else:  # ols_warmstart
        W_init = _ols_warmstart_init(Y)
    np.fill_diagonal(W_init, 0.0)
    W = torch.tensor(W_init, dtype=dtype, device=device, requires_grad=True)
    Y_t = torch.tensor(Y, dtype=dtype, device=device)
    diag_mask = 1.0 - torch.eye(p, dtype=dtype, device=device)

    for rho in rho_schedule:
        opt = torch.optim.Adam([W], lr=lr)
        for _ in range(n_iter_per_rho):
            opt.zero_grad()
            W_masked = W * diag_mask
            resid = Y_t - Y_t @ W_masked
            loss_data = 0.5 * (resid ** 2).sum() / n
            loss_l1 = lam * W_masked.abs().sum()
            loss_h = rho * _acyclicity_penalty(W_masked, s, acyclicity)
            loss = loss_data + loss_l1 + loss_h
            loss.backward()
            with torch.no_grad():
                W.grad.mul_(diag_mask)
            opt.step()

    with torch.no_grad():
        return (W * diag_mask).cpu().numpy()


# ----- Observed-only baselines -----

def fit_observed_only_dagma(
    X: np.ndarray, *, lam: float = 0.02, init_seed: int = 0,
    init_mode: str = "random",
) -> np.ndarray:
    """Fit a DAG over X only (ignoring latents) with DAGMA log-det acyclicity.

    init_mode: "random" (v6 default) or "ols_warmstart" (v7 addition; the
               obs-only analogue of the latent factor warmstart).
    """
    X_c = X - X.mean(axis=0, keepdims=True)
    return _fit_linear_dag(X_c, acyclicity="dagma", lam=lam,
                           init_seed=init_seed, init_mode=init_mode)


def fit_observed_only_notears(
    X: np.ndarray, *, lam: float = 0.02, init_seed: int = 0,
    init_mode: str = "random",
) -> np.ndarray:
    """Fit a DAG over X only (ignoring latents) with NOTEARS matrix-exp acyclicity.

    init_mode: "random" (v6 default) or "ols_warmstart" (v7 addition).
    """
    X_c = X - X.mean(axis=0, keepdims=True)
    return _fit_linear_dag(X_c, acyclicity="notears", lam=lam,
                           init_seed=init_seed, init_mode=init_mode)


# ----- Oracle (fully-observed) baselines -----

def fit_oracle_fully_observed_dagma(
    X: np.ndarray, H: np.ndarray, p_x: int, p_h: int,
    *, lam: float = 0.02, init_seed: int = 0,
) -> np.ndarray:
    """Fit a DAG over Z=(X, H) with H revealed, using DAGMA acyclicity."""
    Z = np.hstack([X, H])
    Z_c = Z - Z.mean(axis=0, keepdims=True)
    return _fit_linear_dag(Z_c, acyclicity="dagma", lam=lam, init_seed=init_seed)


def fit_oracle_fully_observed_notears(
    X: np.ndarray, H: np.ndarray, p_x: int, p_h: int,
    *, lam: float = 0.02, init_seed: int = 0,
) -> np.ndarray:
    """Fit a DAG over Z=(X, H) with H revealed, using NOTEARS acyclicity."""
    Z = np.hstack([X, H])
    Z_c = Z - Z.mean(axis=0, keepdims=True)
    return _fit_linear_dag(Z_c, acyclicity="notears", lam=lam, init_seed=init_seed)


# ----- Legacy alias (backward compatibility) -----

fit_oracle_fully_observed = fit_oracle_fully_observed_dagma
