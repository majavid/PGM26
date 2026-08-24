"""
Synthetic linear-Gaussian SEM with explicit latent confounders.

Model:  Z = W^T Z + eps,  eps ~ N(0, sigma^2 I),  Z = (X, H)
Each latent H_j is constrained to be a parent of 2-4 observed variables
(this is the confounding structure the paper is designed to address).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SEMData:
    W_true: np.ndarray          # (p_x + p_h) x (p_x + p_h), column i = parents of node i
    X: np.ndarray               # n x p_x   (observed)
    H: np.ndarray               # n x p_h   (hidden at training time)
    p_x: int
    p_h: int
    latent_children: list[list[int]]  # for each latent, indices of its observed children
    noise_var: float


def _random_dag_order(p: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random permutation defining a topological order."""
    return rng.permutation(p)


def generate_sem(
    p_x: int = 10,
    p_h: int = 2,
    n: int = 1000,
    confounding_regime: str | None = None,
    # Low-level knobs (used if confounding_regime is None or to override the preset):
    edge_density: float | None = None,
    children_per_latent: tuple[int, int] | None = None,
    weight_range: tuple[float, float] | None = None,
    latent_weight_range: tuple[float, float] | None = None,
    noise_var: float = 1.0,
    seed: int = 0,
) -> SEMData:
    """
    Generate a linear-Gaussian SEM where:
    - Latents H are root-level variables (no parents among Z).
    - Each latent is a parent of `children_per_latent` observed variables.
    - Observed-observed edges are drawn at density `edge_density`, consistent
      with a random topological order over X.
    - Observed-observed weights ~ Uniform([-b, -a] U [a, b]) with (a, b) = weight_range.
    - Latent->observed weights ~ Uniform([-b, -a] U [a, b]) with (a, b) = latent_weight_range.
      Letting latent edges have a DIFFERENT weight band than observed edges is the
      knob that makes confounding the dominant explanation of the observed covariance.
    - Equal-variance Gaussian noise (Peters & Buhlmann identifiable subclass).

    Confounding regime presets:
      "original" (default when regime=None):
          edge_density=0.20, children_per_latent=(2, 4),
          weight_range=(0.5, 1.5), latent_weight_range=(0.5, 1.5)
          This is what v3-v5 used. Observed and latent edges have the same weight
          band, so confounding and direct observed-observed explanations are of
          comparable magnitude -- the "should the model use a latent?" signal is
          weak.
      "strong":
          edge_density=0.10, children_per_latent=(3, 5),
          weight_range=(0.3, 1.0), latent_weight_range=(1.0, 2.0)
          Sparser observed DAG + more children per latent + stronger latent
          weights. The covariance induced by a single latent dominates the
          covariance induced by the sparse observed chain, so parsimony should
          reward models that place a latent. Still identifiable (equal error
          variances), still acyclic, still linear-Gaussian.

    Convention: W[i, j] is the weight of edge i -> j.
    So column j holds the parents of node j, and  Z = W^T Z + eps.
    """
    # Resolve regime to concrete knobs.
    if confounding_regime in (None, "original"):
        preset = dict(edge_density=0.20, children_per_latent=(2, 4),
                      weight_range=(0.5, 1.5), latent_weight_range=(0.5, 1.5))
    elif confounding_regime == "strong":
        preset = dict(edge_density=0.10, children_per_latent=(3, 5),
                      weight_range=(0.3, 1.0), latent_weight_range=(1.0, 2.0))
    else:
        raise ValueError(f"unknown confounding_regime: {confounding_regime!r}")

    # Explicit kwargs override the preset.
    if edge_density is None: edge_density = preset["edge_density"]
    if children_per_latent is None: children_per_latent = preset["children_per_latent"]
    if weight_range is None: weight_range = preset["weight_range"]
    if latent_weight_range is None: latent_weight_range = preset["latent_weight_range"]

    rng = np.random.default_rng(seed)
    p = p_x + p_h
    a_obs, b_obs = weight_range
    a_lat, b_lat = latent_weight_range

    # Node indexing: first p_x are observed (X), last p_h are latent (H).
    x_idx = np.arange(p_x)
    h_idx = np.arange(p_x, p)

    W = np.zeros((p, p))

    # 1. Latent -> observed edges (the confounding). Uses latent_weight_range.
    latent_children: list[list[int]] = []
    lo, hi = children_per_latent
    for h in h_idx:
        k = rng.integers(lo, hi + 1)
        children = rng.choice(x_idx, size=k, replace=False)
        latent_children.append(children.tolist())
        for c in children:
            w = rng.uniform(a_lat, b_lat) * rng.choice([-1.0, 1.0])
            W[h, c] = w

    # 2. Observed -> observed edges, sparse, consistent with a random topological order.
    #    Uses weight_range.
    order = _random_dag_order(p_x, rng)
    pos = np.empty(p_x, dtype=int)
    pos[order] = np.arange(p_x)
    for i in x_idx:
        for j in x_idx:
            if pos[i] < pos[j]:
                if rng.uniform() < edge_density:
                    w = rng.uniform(a_obs, b_obs) * rng.choice([-1.0, 1.0])
                    W[i, j] = w

    # 3. Sanity: (I - W) invertible because W is upper-triangular in the joint order
    #    [latents first, then X in topological order].
    I_minus_W = np.eye(p) - W
    # Sample:  Z = (I - W^T)^{-1} eps
    sigma = np.sqrt(noise_var)
    eps = rng.normal(0.0, sigma, size=(n, p))
    # Solve (I - W^T) Z^T = eps^T  =>  Z^T = (I - W^T)^{-1} eps^T
    Z = np.linalg.solve(I_minus_W.T, eps.T).T  # n x p

    X = Z[:, :p_x]
    H = Z[:, p_x:]

    return SEMData(
        W_true=W,
        X=X,
        H=H,
        p_x=p_x,
        p_h=p_h,
        latent_children=latent_children,
        noise_var=noise_var,
    )


def observed_observed_confounded_pairs(data: SEMData) -> list[tuple[int, int]]:
    """
    Pairs of observed variables (i, j), i != j, that
      (a) share a latent parent in the true graph, and
      (b) have NO direct edge between them in either direction.

    These are the pairs where observed-only methods are expected to
    hallucinate a spurious direct edge. The set is unordered (i < j).
    """
    pairs: set[tuple[int, int]] = set()
    W = data.W_true
    p_x = data.p_x
    for children in data.latent_children:
        for a_ in children:
            for b_ in children:
                if a_ >= b_:
                    continue
                if W[a_, b_] == 0.0 and W[b_, a_] == 0.0:
                    pairs.add((int(a_), int(b_)))
    return sorted(pairs)


if __name__ == "__main__":
    d = generate_sem(seed=0)
    print("W_true shape:", d.W_true.shape)
    print("n edges in true graph:", int((d.W_true != 0).sum()))
    print("latent children:", d.latent_children)
    print("confounded pairs:", observed_observed_confounded_pairs(d))
    print("X mean, std:", d.X.mean(), d.X.std())
