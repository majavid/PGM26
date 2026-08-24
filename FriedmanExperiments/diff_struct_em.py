"""
Differentiable Structural EM for explicit-latent linear-Gaussian DAGs.

Model:  Z = W^T Z + eps,  eps ~ N(0, sigma^2 I),  Z = (X, H)
        W has zero diagonal; W acyclic => (I - W) triangular up to reordering.
        We assume zero intercepts (data will be centered).

E-step (exact, Gaussian):
    Given current (W, sigma^2), compute Sigma_Z = sigma^2 (I - W^T)^{-1} (I - W)^{-1}.
    For each sample x^{(i)}:
        m^{(i)} = Sigma_HX Sigma_XX^{-1} x^{(i)}
        V       = Sigma_HH - Sigma_HX Sigma_XX^{-1} Sigma_XH      (same for all i)
    Then the expected second moment is
        S = (1/n) sum_i E_q[ z^{(i)} z^{(i)T} ]
    with X block = (1/n) X^T X, cross = (1/n) X^T M, HH block = V + (1/n) M^T M,
    where M stacks m^{(i)}.

M-step (improving, differentiable):
    Minimize -Q(W, sigma^2) + lambda ||W||_1 + rho * h(W)
    where the expected complete-data negative log-likelihood (up to constants) is
        -Q / n = (p/2) log(sigma^2) + (1 / (2 sigma^2)) tr( (I-W)^T (I-W) S )
    using |det(I-W)| = 1 for acyclic W with zero diagonal.
    Acyclicity: DAGMA log-det  h(W) = -log det(s I - W*W) + p log s.

Observed-data log-likelihood (for monitoring):
    log p(X; W, sigma^2) = -(n/2) [ p_x log(2 pi) + log det Sigma_XX
                                    + tr( Sigma_XX^{-1} (1/n) X^T X ) ]
    (this is the zero-mean Gaussian log-likelihood on the X marginal of the full SEM).

NOTE ON L1 REGULARIZATION (disclosed openly in README):
  The L1 penalty is applied row-wise, with observed-row weight `lam` and
  latent-row weight `lam * latent_l1_scale`. Default `latent_l1_scale=0.0`
  means latent rows are unregularized, while observed rows are regularized at
  `lam=0.02`. This asymmetry is a deliberate modeling choice motivated by the
  identifiability issue in observational linear-Gaussian latent SEMs (paper
  Section 3): without it, L1 drives fitted latent->observed weights to zero
  and the model collapses to an observed-only explanation. This choice does
  help the proposed method use latents more effectively than an equal-L1
  treatment would, and readers should judge whether it is appropriate for
  their own setting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch


# ---------- E-step (numpy; closed-form Gaussian conditioning) -----------------

def _joint_covariance(W: np.ndarray, sigma2: float) -> np.ndarray:
    """Sigma_Z = sigma^2 (I - W^T)^{-1} (I - W)^{-1}."""
    p = W.shape[0]
    I = np.eye(p)
    A = I - W                           # (p, p)
    # (I - W) is invertible when W is acyclic with zero diagonal.
    A_inv = np.linalg.inv(A)
    return sigma2 * A_inv.T @ A_inv


def e_step_gaussian(
    X: np.ndarray,
    W: np.ndarray,
    sigma2: float,
    p_x: int,
    p_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        S        : (p, p) expected second-moment matrix, averaged over samples
        M        : (n, p_h) posterior means for each sample
        V        : (p_h, p_h) posterior covariance (shared across samples)
    """
    n = X.shape[0]
    p = p_x + p_h
    Sigma = _joint_covariance(W, sigma2)
    Sxx = Sigma[:p_x, :p_x]
    Shx = Sigma[p_x:, :p_x]
    Shh = Sigma[p_x:, p_x:]

    # Solve Sxx M^T = Shx^T for stability: M = X Sxx^{-1} Shx^T = X (Sxx^{-1} Sxx_h_cols)
    # We want: m_i = Shx Sxx^{-1} x_i,  i.e.  M = X @ Sxx^{-1} @ Shx^T  (n x p_h).
    Sxx_inv_ShxT = np.linalg.solve(Sxx, Shx.T)  # (p_x, p_h)
    M = X @ Sxx_inv_ShxT                        # (n, p_h)
    V = Shh - Shx @ Sxx_inv_ShxT                # (p_h, p_h)

    # Assemble expected second moment. Rows/cols: [X-block | H-block].
    # E[z z^T] per-sample has:
    #   XX block:  x_i x_i^T            -> sum = X^T X
    #   XH block:  x_i m_i^T            -> sum = X^T M
    #   HH block:  V + m_i m_i^T        -> sum = n V + M^T M
    S = np.zeros((p, p))
    S[:p_x, :p_x] = (X.T @ X) / n
    S[:p_x, p_x:] = (X.T @ M) / n
    S[p_x:, :p_x] = S[:p_x, p_x:].T
    S[p_x:, p_x:] = V + (M.T @ M) / n
    return S, M, V


def observed_log_likelihood(
    X: np.ndarray, W: np.ndarray, sigma2: float, p_x: int, p_h: int
) -> float:
    """Zero-mean Gaussian log-likelihood of the X marginal under the current SEM."""
    n = X.shape[0]
    Sigma = _joint_covariance(W, sigma2)
    Sxx = Sigma[:p_x, :p_x]
    # Add tiny jitter for numerical safety when W is nearly singular near DAG boundary.
    jitter = 1e-10 * np.trace(Sxx) / p_x
    Sxx_j = Sxx + jitter * np.eye(p_x)
    sign, logdet = np.linalg.slogdet(Sxx_j)
    if sign <= 0:
        return float("-inf")
    emp = (X.T @ X) / n
    Sxx_inv = np.linalg.inv(Sxx_j)
    quad = np.trace(Sxx_inv @ emp)
    # log p(X) = -(n/2) [ p_x log(2 pi) + log det Sxx + tr(Sxx^{-1} emp) ]
    return -0.5 * n * (p_x * np.log(2.0 * np.pi) + logdet + quad)


# ---------- M-step (PyTorch; DAGMA-style log-det acyclicity) ------------------

def _dagma_h(W: torch.Tensor, s: float) -> torch.Tensor:
    """h(W) = -log det(s I - W*W) + p log s.   0 iff W is a DAG (Bello et al.)."""
    p = W.shape[0]
    M = s * torch.eye(p, dtype=W.dtype, device=W.device) - W * W
    # slogdet: handles sign; we require M positive definite for h to be well-defined.
    sign, logabsdet = torch.linalg.slogdet(M)
    # If sign is non-positive, return a large penalty so the optimizer is pushed back.
    # (In practice we schedule s and rho to keep us in the feasible region.)
    safe = torch.where(sign > 0, -logabsdet + p * np.log(s), torch.full_like(logabsdet, 1e6))
    return safe


def _notears_h(W: torch.Tensor) -> torch.Tensor:
    """
    NOTEARS matrix-exponential acyclicity characterization (Zheng et al., 2018):

        h(W) = tr(exp(W * W)) - p,

    where * is elementwise product. h(W) >= 0 with equality iff W is a DAG.
    This is the acyclicity penalty originally used by NOTEARS; DAGMA's
    log-determinant alternative (Bello et al., 2022) is typically better
    conditioned at larger p, but at p ~ 10-20 the two are numerically similar
    and either can plug into the Structural EM M-step.
    """
    p = W.shape[0]
    return torch.trace(torch.matrix_exp(W * W)) - p


def _acyclicity_penalty(W: torch.Tensor, s: float, kind: str) -> torch.Tensor:
    """Dispatch to the requested acyclicity penalty."""
    if kind == "dagma":
        return _dagma_h(W, s)
    if kind == "notears":
        return _notears_h(W)
    raise ValueError(f"unknown acyclicity kind: {kind!r} (expected 'dagma' or 'notears')")


def _expected_nll(W: torch.Tensor, log_sigma2: torch.Tensor, S: torch.Tensor, p: int) -> torch.Tensor:
    """
    Expected complete-data negative log-likelihood per sample (constants in 2*pi dropped).

    Complete-data log p(z | W, sigma^2)
        = -(p/2) log(2 pi sigma^2) + log|det(I - W)| - (1/(2 sigma^2)) ||(I - W^T) z||^2
    (the Jacobian of eps -> z = (I - W^T)^{-1} eps has |det| = 1/|det(I - W)|).
    E[||(I - W^T) z||^2] = tr((I - W)(I - W)^T S).

    So -Q/n = (p/2) log sigma^2 - log|det(I - W)| + (1/(2 sigma^2)) tr((I-W)(I-W)^T S).
    Keeping the log-det term is important during optimization because W can move
    away from the DAG region where log|det(I-W)| = 0.
    """
    sigma2 = torch.exp(log_sigma2)
    I = torch.eye(p, dtype=W.dtype, device=W.device)
    A = I - W
    quad = torch.trace(A @ A.T @ S)
    sign, logabsdet = torch.linalg.slogdet(A)
    # If sign <= 0, (I - W) has non-positive determinant -> reflection or singular.
    # Return a large penalty so the optimizer is pushed back toward valid region.
    logdet = torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, -1e6))
    return 0.5 * p * log_sigma2 - logdet + 0.5 * quad / sigma2


def m_step(
    W_init: np.ndarray,
    log_sigma2_init: float,
    S: np.ndarray,
    *,
    lam: float = 0.02,
    l1_row_mult: np.ndarray | None = None,  # (p,) multiplier applied per row of the L1 penalty
    rho: float = 1.0,
    s: float = 1.0,
    acyclicity: str = "dagma",              # "dagma" (log-det) or "notears" (matrix-exp)
    n_iter: int = 400,
    lr: float = 0.01,
    enforce_improving: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, float, float]:
    """
    Improving differentiable M-step. Returns (W_new, sigma2_new, final_objective).
    We enforce zero diagonal on W by masking gradients there.

    If enforce_improving=True, we accept the optimizer's output only if it improves
    the UNPENALIZED expected complete-data objective Q (equivalently, decreases
    -Q/n = (p/2) log sigma^2 + (1/(2 sigma^2)) tr((I-W)(I-W)^T S)). This makes the
    outer loop a valid generalized-EM: Theorem 2 needs F(q_t, W_{t+1}, theta_{t+1})
    >= F(q_t, W_t, theta_t), which, since H(q_t) does not depend on (W, theta), is
    equivalent to Q(W_{t+1}, theta_{t+1}; q_t) >= Q(W_t, theta_t; q_t).
    Penalty terms (L1, acyclicity) are optimization heuristics, not part of F.
    """
    p = W_init.shape[0]
    device = torch.device("cpu")
    dtype = torch.float64  # need double precision for slogdet stability

    W = torch.tensor(W_init, dtype=dtype, device=device, requires_grad=True)
    log_sigma2 = torch.tensor(log_sigma2_init, dtype=dtype, device=device, requires_grad=True)
    S_t = torch.tensor(S, dtype=dtype, device=device)
    diag_mask = 1.0 - torch.eye(p, dtype=dtype, device=device)

    # Baseline Q-value (used only for the accept/reject check).
    with torch.no_grad():
        nll0 = _expected_nll(W * diag_mask, log_sigma2, S_t, p).item()

    # Row-wise L1 weight. Default: uniform (ones).
    if l1_row_mult is None:
        l1_weight = torch.ones(p, dtype=dtype, device=device)
    else:
        l1_weight = torch.tensor(l1_row_mult, dtype=dtype, device=device)
    l1_weight_mat = l1_weight.unsqueeze(1).expand(p, p)  # row i has weight l1_weight[i]

    opt = torch.optim.Adam([W, log_sigma2], lr=lr)

    for it in range(n_iter):
        opt.zero_grad()
        W_masked = W * diag_mask
        nll = _expected_nll(W_masked, log_sigma2, S_t, p)
        l1 = lam * (l1_weight_mat * W_masked.abs()).sum()
        h = _acyclicity_penalty(W_masked, s, acyclicity)
        loss = nll + l1 + rho * h
        loss.backward()
        # Zero out diagonal gradient to keep diagonal at 0.
        with torch.no_grad():
            W.grad.mul_(diag_mask)
        opt.step()
        if verbose and it % 100 == 0:
            print(f"    m-step it={it}  loss={loss.item():.4f}  h={h.item():.2e}")

    with torch.no_grad():
        W_cand = (W * diag_mask).cpu().numpy()
        sigma2_cand = float(torch.exp(log_sigma2).item())
        # Unpenalized Q-value at the candidate.
        nll1 = _expected_nll(W * diag_mask, log_sigma2, S_t, p).item()
        obj_final = float(loss.item())

    if enforce_improving and nll1 > nll0 + 1e-8:
        # M-step did not improve Q -> reject the update (return the input).
        # This makes the outer loop a valid generalized-EM.
        return W_init.copy(), float(np.exp(log_sigma2_init)), obj_final

    return W_cand, sigma2_cand, obj_final


# ---------- Outer EM loop -----------------------------------------------------

@dataclass
class EMResult:
    W: np.ndarray
    sigma2: float
    loglik_trace: list[float] = field(default_factory=list)  # observed-data log-lik per outer iter
    W_trace: list[np.ndarray] = field(default_factory=list)


def _init_W(p: int, seed: int, scale: float = 0.1, p_x: int | None = None,
            latent_init_scale: float = 0.5) -> np.ndarray:
    """
    Initialize W with small random entries.

    If p_x is provided (so the function knows which rows are latents), latent
    rows (the last p - p_x rows) get a LARGER initial scale. This is important:
    with all-small init, latent->observed weights start near zero and the
    gradient signal from the data is too weak to push them away from zero,
    causing the model to collapse to an "explain everything via observed-
    observed edges" solution. A larger latent init is a standard remedy in
    latent-variable model fitting.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, scale, size=(p, p))
    if p_x is not None and p_x < p:
        # Reinitialize latent rows at a larger scale.
        n_latent = p - p_x
        W[p_x:, :] = rng.normal(0, latent_init_scale, size=(n_latent, p))
    np.fill_diagonal(W, 0.0)
    return W


def _init_W_factor_warmstart(
    X: np.ndarray,
    p_x: int,
    p_h: int,
    *,
    seed: int,
    obs_row_scale: float = 0.1,
    latent_row_scale: float = 1.0,
) -> np.ndarray:
    """
    Factor-analysis-style warm start for the latent rows.

    Standard latent-Gaussian-model initialization idea: the leading principal
    directions of the empirical covariance of X are the directions along which
    a latent factor would explain the most variance in X, so they are a
    principled initialization for the latent->observed rows of W.

    Concretely:
      - Center X.
      - Compute the top-p_h eigenvectors u_1, ..., u_{p_h} of X^T X / n.
      - Rescale each by sqrt(lambda_k) * latent_row_scale so that the implied
        latent factor has variance ~ lambda_k under the SEM.
      - Use those as the latent->observed rows of W (rows [p_x, p_x + p_h)).
      - Observed->observed rows use the usual small random init.

    This is NOT the same as treating H as an observed variable (which would
    be the oracle); it uses only X, the same information the optimizer has,
    and just chooses a starting point that is not in the all-zero latent
    basin.

    `seed` is still used to randomize the observed rows and to sign-flip
    the eigenvectors (eigenvectors are only identified up to sign).
    """
    rng = np.random.default_rng(seed)
    p = p_x + p_h

    # Observed rows: small random init, zero diagonal (handled later).
    W = rng.normal(0, obs_row_scale, size=(p, p))

    # Eigendecompose the empirical covariance of X.
    X_c = X - X.mean(axis=0, keepdims=True)
    Sigma_x = (X_c.T @ X_c) / X_c.shape[0]                       # (p_x, p_x)
    eigvals, eigvecs = np.linalg.eigh(Sigma_x)                   # ascending
    # Take top-p_h.
    top_vals = eigvals[-p_h:][::-1]                              # descending
    top_vecs = eigvecs[:, -p_h:][:, ::-1]                        # (p_x, p_h)

    # For each latent k, set latent_k -> X_i edge weight to the k-th eigenvector's
    # i-th component, scaled by sqrt(lambda_k) so that the *contribution* of each
    # latent to the observed covariance matches the scale of that eigenmode.
    # Random sign flip per latent so the optimizer is not forced into one chirality.
    for k in range(p_h):
        signs = rng.choice([-1.0, 1.0])
        weight_row = signs * top_vecs[:, k] * np.sqrt(max(top_vals[k], 1e-12))
        weight_row = weight_row * latent_row_scale
        # Place in the latent row W[p_x + k, :p_x]; latent->latent is zero,
        # latent-row -> latent-col also zero (latents have no parents here).
        W[p_x + k, :p_x] = weight_row
        W[p_x + k, p_x:] = 0.0

    np.fill_diagonal(W, 0.0)
    return W


def fit_diff_struct_em(
    X: np.ndarray,
    p_x: int,
    p_h: int,
    *,
    n_outer: int = 15,
    m_step_iters: int = 400,
    m_step_lr: float = 0.01,
    lam: float = 0.02,
    latent_l1_scale: float = 0.0,   # multiplier on L1 for latent rows; 0 = latents unregularized
    rho: float = 1.0,
    s: float = 1.0,
    acyclicity: str = "dagma",      # "dagma" or "notears"
    init_seed: int = 0,
    init_mode: str = "random",      # "random" or "factor_warmstart"
    e_step_mode: str = "full",   # "full" or "mean_only" (ablation)
    enforce_improving: bool = True,
    verbose: bool = False,
) -> EMResult:
    """
    Proposed method: exact Gaussian E-step + improving differentiable M-step.

    e_step_mode:
      "full"      -> use full posterior second moments (the paper's prescription)
      "mean_only" -> replace V in the HH block by zero (posterior-mean imputation)
                     This is the ablation that disables posterior second moments.

    latent_l1_scale:
      Rows of W corresponding to latent variables (the last p_h rows) have their
      L1 penalty scaled by this factor. Without this, L1 drives fitted latent
      rows toward zero because the observed marginal can often be explained by
      observed-observed edges alone -- an identifiability artifact of the
      linear-Gaussian family. Setting latent_l1_scale < 1 preserves sparsity in
      observed-observed edges while letting the model use the latents.

    init_mode:
      "random"            : the v5 default. Random normal init with latent rows
                            at a larger scale (latent_init_scale=0.5).
      "factor_warmstart"  : initialize latent rows from the top-p_h principal
                            components of X^T X / n, scaled by sqrt(eigenvalue).
                            This is the classical factor-analysis warm start
                            adapted to the SEM setting -- principled in that
                            it places the latent starting points in the
                            directions along which X has the most structure,
                            rather than in a random direction where the
                            gradient signal might be weak.
    """
    assert e_step_mode in ("full", "mean_only")
    assert init_mode in ("random", "factor_warmstart")
    p = p_x + p_h
    X = X - X.mean(axis=0, keepdims=True)  # center to match zero-intercept model

    # Per-row L1 multiplier: 1.0 for observed rows, latent_l1_scale for latent rows.
    l1_row_mult = np.ones(p)
    l1_row_mult[p_x:] = latent_l1_scale

    if init_mode == "factor_warmstart" and p_h > 0:
        W = _init_W_factor_warmstart(X, p_x, p_h, seed=init_seed)
    else:
        W = _init_W(p, seed=init_seed, p_x=p_x)
    sigma2 = float(X.var(axis=0).mean())   # reasonable scalar init
    loglik_trace: list[float] = []
    W_trace: list[np.ndarray] = []

    for t in range(n_outer):
        # ---- E-step (exact Gaussian) ----
        S, M, V = e_step_gaussian(X, W, sigma2, p_x, p_h)
        if e_step_mode == "mean_only":
            # Drop posterior covariance -> use imputed z^T z instead of E[z z^T].
            S_ablate = S.copy()
            S_ablate[p_x:, p_x:] -= V   # subtract the V contribution from HH block
            S_used = S_ablate
        else:
            S_used = S

        # ---- Observed-data log-likelihood at current (W, sigma^2) BEFORE M-step.
        ll = observed_log_likelihood(X, W, sigma2, p_x, p_h)
        loglik_trace.append(ll)
        W_trace.append(W.copy())
        if verbose:
            print(f"[EM iter {t}] ll = {ll:.3f}  sigma2={sigma2:.3f}")

        # ---- M-step ----
        W, sigma2, _ = m_step(
            W_init=W, log_sigma2_init=float(np.log(max(sigma2, 1e-6))),
            S=S_used, lam=lam, l1_row_mult=l1_row_mult,
            rho=rho, s=s, acyclicity=acyclicity,
            n_iter=m_step_iters, lr=m_step_lr,
            enforce_improving=enforce_improving, verbose=False,
        )

    # Final log-lik at the last (W, sigma2).
    ll_final = observed_log_likelihood(X, W, sigma2, p_x, p_h)
    loglik_trace.append(ll_final)
    W_trace.append(W.copy())

    return EMResult(W=W, sigma2=sigma2, loglik_trace=loglik_trace, W_trace=W_trace)


# =============================================================================
# Stability selection
# =============================================================================
#
# Multiple random initializations of fit_diff_struct_em are run; each fit's W is
# binarized at `edge_thresh`; the consensus mask retains edges that appear in at
# least `freq_thresh` fraction of fits.
#
# Two key implementation choices:
#
# 1. Latent permutation is NOT canonical across fits -- fit A's latent 0 may
#    correspond to truth's latent 0, fit B's latent 0 to truth's latent 1. So
#    consensus over the LATENT BLOCK only makes sense AFTER aligning each fit's
#    latents to a common reference. We use the FIRST fit as the reference and
#    align fits 2..n_inits to it via the same greedy bipartite match used
#    elsewhere (see metrics.align_latent_labels). The OBSERVED block is
#    consensus-aggregated directly without permutation.
#
# 2. The output is a binary CONSENSUS MATRIX (entries in {0, 1}). Downstream
#    metric code keys off |W| > thresh, so we return a float matrix with values
#    0.0 or 1.0; metric thresholds below 1.0 will treat the 1s as edges.

def _binarize_with_diag_zero(W: np.ndarray, thresh: float) -> np.ndarray:
    B = (np.abs(W) > thresh).astype(np.float64)
    np.fill_diagonal(B, 0.0)
    return B


def _align_to_reference(
    B_other: np.ndarray, B_ref: np.ndarray, p_x: int, p_h: int
) -> np.ndarray:
    """
    Permute the latent rows/cols of B_other to best match the latent block of
    B_ref, using greedy bipartite match on observed-children Jaccard score.

    Both inputs are BINARY (0/1) matrices of shape (p_x + p_h, p_x + p_h).
    Observed block is unchanged.
    """
    if p_h == 0:
        return B_other.copy()
    ref_children = B_ref[p_x:p_x + p_h, :p_x] > 0      # (p_h, p_x)
    other_children = B_other[p_x:p_x + p_h, :p_x] > 0  # (p_h, p_x)

    score = np.zeros((p_h, p_h), dtype=int)
    for r in range(p_h):
        for o in range(p_h):
            inter = int((ref_children[r] & other_children[o]).sum())
            symdiff = int((ref_children[r] ^ other_children[o]).sum())
            score[r, o] = inter - symdiff

    assigned_ref, assigned_other = set(), set()
    other_to_ref: dict[int, int] = {}
    for flat in np.argsort(-score, axis=None):
        r, o = divmod(int(flat), p_h)
        if r in assigned_ref or o in assigned_other:
            continue
        other_to_ref[o] = r
        assigned_ref.add(r); assigned_other.add(o)
        if len(other_to_ref) == p_h:
            break

    ref_to_other = {r: o for o, r in other_to_ref.items()}
    latent_perm = np.array([ref_to_other[k] for k in range(p_h)], dtype=int)
    perm = np.concatenate([np.arange(p_x), p_x + latent_perm])
    return B_other[np.ix_(perm, perm)].copy()


def fit_diff_struct_em_stability(
    X: np.ndarray,
    p_x: int,
    p_h: int,
    *,
    n_inits: int = 16,
    base_init_seed: int = 0,
    edge_thresh: float = 0.15,
    freq_thresh: float = 0.75,
    # Inner-fit hyperparameters (passed through to fit_diff_struct_em):
    n_outer: int = 15,
    m_step_iters: int = 300,
    m_step_lr: float = 0.01,
    lam: float = 0.02,
    latent_l1_scale: float = 0.0,
    rho: float = 1.0,
    s: float = 1.0,
    acyclicity: str = "dagma",
    init_mode: str = "random",
    enforce_improving: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, list]:
    """
    Run fit_diff_struct_em n_inits times with different random initial Ws.
    Binarize each fit's W at edge_thresh, align latent blocks across fits
    using the first fit as reference, and return a consensus binary matrix
    where entry (i, j) is 1 iff the edge appears in >= freq_thresh fraction
    of the n_inits fits.

    Parameters
    ----------
    n_inits : number of random initializations to run. Default 16; with fewer
              inits a strict freq_thresh can under-select real edges.
    base_init_seed : seeds used are base_init_seed + 0, ..., base_init_seed + n_inits - 1.
    edge_thresh : magnitude threshold for binarizing each individual fit's W.
                  Defaults to 0.15 to match the latent-parent metric: fitted
                  latent->observed weights cluster in the 0.15-0.5 range due
                  to the L1 + identifiability interaction. The strict
                  freq_thresh below filters unreliable edges that show up in
                  some fits but not consistently.
    freq_thresh : minimum fraction of fits in which an edge must appear.
                  Default 0.75. Stricter thresholds (0.8 as in Meinshausen-
                  Buhlmann) may undercount at small n_inits; looser ones
                  (0.5, the v4 default) let too many unstable edges through.

    Returns
    -------
    W_consensus : (p_x + p_h, p_x + p_h) float matrix with values in {0.0, 1.0}.
                  Latent labels match the first fit's labels; further alignment
                  to the truth (e.g. via metrics.align_latent_labels) is the
                  caller's responsibility.
    fit_results : list of EMResult objects, one per initialization, in case
                  the caller wants to inspect them (e.g. to plot trajectories).
    """
    p = p_x + p_h
    fit_results = []
    binary_masks: list[np.ndarray] = []
    reference_binary: np.ndarray | None = None

    for k in range(n_inits):
        res = fit_diff_struct_em(
            X, p_x, p_h,
            n_outer=n_outer, m_step_iters=m_step_iters, m_step_lr=m_step_lr,
            lam=lam, latent_l1_scale=latent_l1_scale, rho=rho, s=s,
            acyclicity=acyclicity,
            init_seed=base_init_seed + k,
            init_mode=init_mode,
            e_step_mode="full", enforce_improving=enforce_improving,
            verbose=False,
        )
        fit_results.append(res)
        B_k = _binarize_with_diag_zero(res.W, thresh=edge_thresh)

        if reference_binary is None:
            reference_binary = B_k
            binary_masks.append(B_k)
        else:
            B_k_aligned = _align_to_reference(B_k, reference_binary, p_x, p_h)
            binary_masks.append(B_k_aligned)

        if verbose:
            print(f"  stability-selection fit {k+1}/{n_inits}: "
                  f"{int(B_k.sum())} edges in this fit")

    stack = np.stack(binary_masks, axis=0)        # (n_inits, p, p)
    freq = stack.mean(axis=0)                      # (p, p) edge frequency
    W_consensus = (freq >= freq_thresh).astype(np.float64)
    np.fill_diagonal(W_consensus, 0.0)
    return W_consensus, fit_results
