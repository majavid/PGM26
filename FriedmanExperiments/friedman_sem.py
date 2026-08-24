"""
Linear-Gaussian SEM generators for the two synthetic networks Friedman uses
in section 5.3 of "The Bayesian Structural EM Algorithm" (UAI 1998):

  - Network (a) "3x1+1x3+3":  mediator topology
        3 top observed (X1, X2, X3) -> 2 hidden mediators (H1, H2)
                                    -> 6 bottom observed (Y1, Y2, Y3 from H1;
                                                          Z1, Z2, Z3 from H2)
        Total: p_x = 9 observed, p_h = 2 hidden mediators.
        Hidden variables are NOT roots; they receive parents from {X1, X2, X3}.

  - Network (b) "3x8":  joint-confounder topology
        3 hidden roots (H1, H2, H3), each parenting 4 consecutive observed.
        Adjacent hiddens share 2 children:
            H1 -> X1, X2, X3, X4
            H2 -> X3, X4, X5, X6
            H3 -> X5, X6, X7, X8
        Total: p_x = 8 observed, p_h = 3 hidden roots.

CAVEAT (must be reported in the paper): Friedman's original networks are
binary multinomial Bayesian networks. We replicate the topologies but
parameterize them as linear-Gaussian SEMs because our method does not apply
to discrete multinomials. The structural recovery question (Can the
algorithm recover hidden-variable structure when p_h is unknown?) remains
the same; the distributional family does not.

Variable ordering in W: observed variables come first, then hiddens.
W[i, j] is the weight on edge i -> j (so column j holds the parents of j).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FriedmanSEMData:
    """Container for a sampled SEM."""
    name: str                    # "mediator-9x2" or "confounder-8x3"
    X_train: np.ndarray          # (n_train, p_x)
    X_test: np.ndarray           # (n_test, p_x)
    H_train: np.ndarray          # (n_train, p_h) -- the unobserved truth
    H_test: np.ndarray           # (n_test, p_h)
    W_true: np.ndarray           # (p_x + p_h, p_x + p_h) full ground-truth adjacency
    sigma2: float                # noise variance used for every node
    p_x: int
    p_h: int
    edge_list: list              # [(i, j, weight), ...] for inspection


def _sample_signed_uniform(rng: np.random.Generator, n: int,
                           lo: float, hi: float) -> np.ndarray:
    """Sample n values from Uniform([-hi, -lo] U [lo, hi])."""
    mag = rng.uniform(lo, hi, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    return mag * sign


def _topological_order_mediator(p_x: int = 9, p_h: int = 2) -> list[int]:
    """
    For mediator-9x2:
      Index 0..2 are top observed X1..X3 (sources / roots in this DAG).
      Index 3..5 are bottom observed Y1..Y3.
      Index 6..8 are bottom observed Z1..Z3.
      Index 9, 10 are hidden mediators H1, H2.
    Topological order: top-X first, then mediators, then bottom-Y/Z.
    """
    return [0, 1, 2, 9, 10, 3, 4, 5, 6, 7, 8]


def generate_mediator_9x2(
    n_train: int = 1000,
    n_test: int = 5000,
    weight_lo: float = 0.7,
    weight_hi: float = 1.3,
    noise_var: float = 1.0,
    seed: int = 0,
) -> FriedmanSEMData:
    """
    Network (a): 3 top observed -> 2 hidden mediators -> 6 bottom observed.

    Variable ordering in returned arrays:
        X_train[:, 0:3] = X1, X2, X3   (top observed)
        X_train[:, 3:6] = Y1, Y2, Y3   (bottom-left observed, children of H1)
        X_train[:, 6:9] = Z1, Z2, Z3   (bottom-right observed, children of H2)
        H_train[:, 0]   = H1
        H_train[:, 1]   = H2

    Total p_x = 9, p_h = 2. Indices in W follow the same ordering with
    hiddens at positions 9, 10.
    """
    rng = np.random.default_rng(seed)
    p_x, p_h = 9, 2
    p = p_x + p_h
    W = np.zeros((p, p))
    edges: list = []

    # Top observed -> hidden mediators (all 3 to both).
    # Indices: X1=0, X2=1, X3=2; H1=9, H2=10.
    weights_top_to_h = _sample_signed_uniform(rng, 6, weight_lo, weight_hi)
    pairs_top_to_h = [(0, 9), (1, 9), (2, 9), (0, 10), (1, 10), (2, 10)]
    for (i, j), w in zip(pairs_top_to_h, weights_top_to_h):
        W[i, j] = w
        edges.append((i, j, w))

    # Hidden mediators -> bottom observed.
    # H1 -> Y1, Y2, Y3   (indices 9 -> 3, 4, 5)
    # H2 -> Z1, Z2, Z3   (indices 10 -> 6, 7, 8)
    weights_h_to_bot = _sample_signed_uniform(rng, 6, weight_lo, weight_hi)
    pairs_h_to_bot = [(9, 3), (9, 4), (9, 5), (10, 6), (10, 7), (10, 8)]
    for (i, j), w in zip(pairs_h_to_bot, weights_h_to_bot):
        W[i, j] = w
        edges.append((i, j, w))

    # Sample data: Z = W^T Z + eps with eps ~ N(0, sigma2 I).
    # Equivalent: Z = (I - W^T)^-1 eps.
    Z_train = _sample_sem(W, n_train, noise_var, rng)
    Z_test = _sample_sem(W, n_test, noise_var, rng)

    # Split into observed (X) and hidden (H).
    X_train = Z_train[:, :p_x]
    X_test = Z_test[:, :p_x]
    H_train = Z_train[:, p_x:]
    H_test = Z_test[:, p_x:]

    return FriedmanSEMData(
        name="mediator-9x2",
        X_train=X_train, X_test=X_test,
        H_train=H_train, H_test=H_test,
        W_true=W, sigma2=noise_var,
        p_x=p_x, p_h=p_h, edge_list=edges,
    )


def generate_confounder_8x3(
    n_train: int = 1000,
    n_test: int = 5000,
    weight_lo: float = 0.7,
    weight_hi: float = 1.3,
    noise_var: float = 1.0,
    seed: int = 0,
) -> FriedmanSEMData:
    """
    Network (b): 3 hidden roots, each parenting 4 consecutive observed
    variables, with adjacent hiddens sharing 2 children.
        H1 -> X1, X2, X3, X4
        H2 -> X3, X4, X5, X6
        H3 -> X5, X6, X7, X8

    Variable ordering in returned arrays:
        X_train[:, 0..7] = X1..X8
        H_train[:, 0..2] = H1, H2, H3
    """
    rng = np.random.default_rng(seed)
    p_x, p_h = 8, 3
    p = p_x + p_h
    W = np.zeros((p, p))
    edges: list = []

    # H1 = index 8, H2 = 9, H3 = 10.  X_k = index k-1 (so X1=0, X8=7).
    children_per_h = [
        (8,  [0, 1, 2, 3]),   # H1 -> X1, X2, X3, X4
        (9,  [2, 3, 4, 5]),   # H2 -> X3, X4, X5, X6
        (10, [4, 5, 6, 7]),   # H3 -> X5, X6, X7, X8
    ]
    for (h_idx, children) in children_per_h:
        weights = _sample_signed_uniform(rng, len(children), weight_lo, weight_hi)
        for c, w in zip(children, weights):
            W[h_idx, c] = w
            edges.append((h_idx, c, w))

    Z_train = _sample_sem(W, n_train, noise_var, rng)
    Z_test = _sample_sem(W, n_test, noise_var, rng)

    X_train = Z_train[:, :p_x]
    X_test = Z_test[:, :p_x]
    H_train = Z_train[:, p_x:]
    H_test = Z_test[:, p_x:]

    return FriedmanSEMData(
        name="confounder-8x3",
        X_train=X_train, X_test=X_test,
        H_train=H_train, H_test=H_test,
        W_true=W, sigma2=noise_var,
        p_x=p_x, p_h=p_h, edge_list=edges,
    )


def _sample_sem(W: np.ndarray, n: int, noise_var: float,
                rng: np.random.Generator) -> np.ndarray:
    """
    Sample n rows from the linear-Gaussian SEM Z = W^T Z + eps,
    eps ~ N(0, noise_var * I).
    Closed form: Z = eps @ (I - W)^{-1} treating eps as (n, p) row vectors.

    (Note: with the convention W[i,j] = weight on edge i -> j, the SEM
    equation is Z = W^T Z + eps where Z, eps are column vectors;
    transposing gives Z_row = eps_row @ (I - W^T)^{-T} = eps_row @ (I - W)^{-1}.
    Wait, let me redo this carefully.

      Z (column) = W^T Z + eps
      (I - W^T) Z = eps
      Z = (I - W^T)^{-1} eps

    For n rows arranged as (n, p):
      Z_rows^T = (I - W^T)^{-1} eps_rows^T
      Z_rows = eps_rows @ ((I - W^T)^{-1})^T = eps_rows @ (I - W)^{-1}.
    )
    """
    p = W.shape[0]
    eps = rng.normal(0.0, np.sqrt(noise_var), size=(n, p))
    A_inv = np.linalg.inv(np.eye(p) - W)
    return eps @ A_inv


def get_friedman_sem(name: str, **kwargs) -> FriedmanSEMData:
    """Convenience dispatcher."""
    if name in ("mediator", "mediator-9x2", "a", "3x1+1x3+3"):
        return generate_mediator_9x2(**kwargs)
    if name in ("confounder", "confounder-8x3", "b", "3x8"):
        return generate_confounder_8x3(**kwargs)
    raise ValueError(f"unknown Friedman network: {name!r}")


def confounded_observed_pairs(data: FriedmanSEMData) -> list[tuple[int, int]]:
    """
    Return list of (i, j) pairs of observed indices (i < j) such that some
    hidden variable is an ancestor of both X_i and X_j and there is no
    direct edge X_i -> X_j or X_j -> X_i in the truth.

    For confounder-8x3 (latents are roots): a hidden parent of both X_i
    and X_j is enough.
    For mediator-9x2 (latents are children of top X): we count pairs of
    bottom-X variables that share a hidden ANCESTOR -- which is any pair
    of (Y_a, Y_b) within the Y group, any pair (Z_a, Z_b) within the Z group,
    and any cross-group (Y_a, Z_b) pair (since both H1 and H2 ultimately
    descend from the same top-X variables but the bottom variables only
    share the more proximal hidden mediator if grouped). To stay close to
    Friedman's intent of "the hidden mediates the observed correlation",
    we count Y-Y, Z-Z, AND Y-Z pairs as confounded -- they are all driven
    by the latent bottleneck.

    Returns pairs as (smaller_index, larger_index) tuples.
    """
    p_x = data.p_x
    W = data.W_true
    has_direct_edge = (np.abs(W[:p_x, :p_x]) > 0).astype(bool)

    # Compute hidden-ancestors of each observed variable.
    # For confounder-8x3, hidden -> obs directly, so ancestors = direct parents.
    # For mediator-9x2, the bottom-Xs have hidden as direct parent and the top-Xs
    # are ancestors of the hidden. For "shares a hidden ancestor", we just need
    # a hidden h such that h is an ancestor of both X_i and X_j.
    p = data.p_x + data.p_h
    # Hidden variables occupy indices [p_x, p_x + p_h).
    hidden_indices = list(range(p_x, p))

    # For each hidden h, compute its set of observed descendants.
    descendants_of_hidden: dict[int, set[int]] = {h: set() for h in hidden_indices}
    for h in hidden_indices:
        # BFS from h in the directed graph defined by W (W[i,j] != 0 means i -> j).
        visited = {h}
        stack = [h]
        while stack:
            node = stack.pop()
            children = np.where(np.abs(W[node, :]) > 0)[0].tolist()
            for c in children:
                if c not in visited:
                    visited.add(c)
                    stack.append(c)
        descendants_of_hidden[h] = {n for n in visited if n < p_x}

    # A pair (i, j) is "confounded" if some hidden has both i and j in its descendants.
    pairs: list[tuple[int, int]] = []
    for i in range(p_x):
        for j in range(i + 1, p_x):
            if has_direct_edge[i, j] or has_direct_edge[j, i]:
                continue
            for h in hidden_indices:
                if i in descendants_of_hidden[h] and j in descendants_of_hidden[h]:
                    pairs.append((i, j))
                    break
    return pairs
