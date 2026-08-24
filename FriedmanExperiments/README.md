# Friedman §5.3 hidden-variable experiment

Replication of Friedman (UAI 1998) §5.3 "Hidden Variables" experimental
setup, adapted to linear-Gaussian SEMs.

## Topologies (see Friedman Figure 1)

- **mediator-9x2** ("3x1+1x3+3"): 3 top observed -> 2 hidden mediators
  -> 6 bottom observed (3+3 split). Hidden variables are *not* roots.
  9 observed, 2 hidden.
- **confounder-8x3** ("3x8"): 3 hidden roots, each parenting 4 consecutive
  observed variables; adjacent hiddens share 2 children.
  8 observed, 3 hidden.

## Caveat (must be reported in the paper)

Friedman's original networks are **binary multinomial**. We replicate the
topologies but parameterize them as **linear-Gaussian** SEMs because our
method does not apply to discrete multinomials. The structural recovery
question is the same; the distributional family is different.

## Files

```
friedman_sem.py              Generators for the two networks; latent-pair labels.
test_loglik.py               Marginal-X log-likelihood on a held-out test set.
diff_struct_em.py, baselines.py, metrics.py
                             Imported from v7 unchanged.
experiments_friedman.py      Driver. Sweeps p_h_fit ∈ {0,1,2,3,4} for each
                             (network, n_train, seed), reports test KL and
                             latent-parent recoveries, plots and tables.
```

## Run

```bash
pip install numpy matplotlib torch
python experiments_friedman.py
```

Each (network, n_train, seed) cell fits 11 models. With 2 networks ×
4 n_train × 10 seeds = 80 cells, expect ~1.5–2 hours on a laptop CPU.

## Two metrics, two purposes

- **Test KL** (Friedman's metric): cleanly separates obs-only from any
  latent model, but does NOT discriminate among different p_h_fit values
  in the linear-Gaussian setting -- any latent model with enough rank
  matches the second-order moments equally well.
- **Latent-parent recoveries**: counts how many ground-truth confounded
  pairs are recovered through a common fitted latent parent. THIS is
  what discriminates random-init from warmstart, and at-true-p_h from
  over-specified p_h.

The headline finding to look for: warmstart matches obs-only on test KL
when latents are present, but **substantially outperforms random init on
latent-parent recovery, including on the harder mediator topology** where
the truth violates the warmstart's "latents as roots" implicit prior.
