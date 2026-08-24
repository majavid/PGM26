"""
Test-set log-likelihood for linear-Gaussian SEMs with explicit latent variables.

Given a fitted SEM (W, sigma2) over Z = (X, H) of dimension p = p_x + p_h,
the marginal distribution over the observed variables X is:
    X ~ N(0, Sigma_X(W, sigma2))
where Sigma_X is the upper-left p_x x p_x block of
    Sigma_Z = (I - W)^{-T} (sigma2 * I) (I - W)^{-1}.

We evaluate average log-density on a held-out test set and report it as a
KL-style "log-loss difference" against the ground truth, paralleling
Friedman's reporting convention (his Table 2 cells are
    [log p_true(x) - log p_fitted(x)] averaged over the test set,
which is the test-set KL divergence up to the test-set entropy.)
"""
from __future__ import annotations
import numpy as np


def _marginal_obs_covariance(
    W: np.ndarray, sigma2: float, p_x: int,
) -> np.ndarray:
    """
    Compute Sigma_X = covariance of the observed marginal under the SEM
    Z = W^T Z + eps, eps ~ N(0, sigma2 I).

    Sigma_Z = (I - W)^{-T} (sigma2 I) (I - W)^{-1}    (in row-vector convention)
    Sigma_X = Sigma_Z[:p_x, :p_x].

    With our convention W[i, j] = weight on edge i -> j, and treating row
    vectors so that Z_row = eps_row @ (I - W)^{-1}, the covariance of Z_row
    is (I - W)^{-T} (sigma2 I) (I - W)^{-1} -- which is what we compute.
    """
    p = W.shape[0]
    A_inv = np.linalg.inv(np.eye(p) - W)
    Sigma_Z = A_inv.T @ (sigma2 * np.eye(p)) @ A_inv
    return Sigma_Z[:p_x, :p_x]


def gaussian_loglik_per_sample(
    X: np.ndarray, Sigma: np.ndarray, jitter: float = 1e-8,
) -> np.ndarray:
    """
    Log-density of each row of X under N(0, Sigma).
    Returns shape-(n,) vector of log densities.
    Adds a small ridge to Sigma for numerical stability.
    """
    n, p = X.shape
    Sigma_reg = Sigma + jitter * np.eye(p)
    sign, logdet = np.linalg.slogdet(Sigma_reg)
    if sign <= 0:
        # Sigma not PD -- fall back to pseudo-inverse with eigenvalue floor.
        eigvals, eigvecs = np.linalg.eigh(Sigma_reg)
        eigvals = np.maximum(eigvals, jitter)
        logdet = float(np.log(eigvals).sum())
        Sigma_inv = (eigvecs * (1.0 / eigvals)) @ eigvecs.T
    else:
        Sigma_inv = np.linalg.inv(Sigma_reg)

    # Quadratic form (x_i - 0)^T Sigma_inv (x_i - 0) for each row, computed
    # vectorized as sum of (X @ Sigma_inv) * X along axis 1.
    quads = np.einsum("ij,jk,ik->i", X, Sigma_inv, X)
    const = -0.5 * (p * np.log(2.0 * np.pi) + logdet)
    return const - 0.5 * quads


def avg_test_loglik(
    X_test: np.ndarray, W: np.ndarray, sigma2: float, p_x: int,
) -> float:
    """
    Average log-density per test row under the SEM (W, sigma2) marginal over X.
    """
    X_test = X_test - X_test.mean(axis=0, keepdims=True)
    Sigma_X = _marginal_obs_covariance(W, sigma2, p_x)
    return float(gaussian_loglik_per_sample(X_test, Sigma_X).mean())


def kl_loss_vs_truth(
    X_test: np.ndarray,
    W_fit: np.ndarray, sigma2_fit: float,
    W_true: np.ndarray, sigma2_true: float,
    p_x: int,
) -> float:
    """
    Test-set log-loss difference vs the ground truth, paralleling Friedman's
    Table 2: for each test row x_test, compute log p_true(x_test) - log p_fit(x_test),
    then average. This is the empirical KL(p_true || p_fit) up to the
    (constant) test-set entropy.

    Lower is better. Zero (or slightly negative due to finite test set)
    means the fit matches the truth.
    """
    return avg_test_loglik(X_test, W_true, sigma2_true, p_x) \
         - avg_test_loglik(X_test, W_fit, sigma2_fit, p_x)
