"""
demand.py — Stochastic Demand Model
=====================================
Models daily demand as a Compound Poisson Log-Normal process.

  D*_t = D_t · (1 + S_t · (m − 1))

  D_t  ~ LogNormal(μ_ln, σ_ln²)   — base demand
  S_t  ~ Bernoulli(p_spike)        — demand shock indicator
  m    = spike_multiplier           — shock magnitude

Latin Hypercube Sampling (LHS) is used as the default sampler
for variance reduction (~40–60% vs. simple random sampling).
"""

import numpy as np
from scipy.stats import norm, lognorm


# ── Parameter helpers ──────────────────────────────────────────────────────

def lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """Convert (mean, CV) to log-normal (μ_ln, σ_ln) parameters.

    E[X] = exp(μ + σ²/2) = mean
    Var[X] = (exp(σ²) − 1) · exp(2μ + σ²) = (CV · mean)²

    Returns
    -------
    mu_ln : float  — location parameter
    sigma_ln : float  — scale parameter
    """
    sigma_ln = np.sqrt(np.log(1 + cv ** 2))
    mu_ln = np.log(mean) - 0.5 * sigma_ln ** 2
    return mu_ln, sigma_ln


# ── Samplers ───────────────────────────────────────────────────────────────

def lhs_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sample from U[0,1].

    Divides [0,1] into n equal strata of width 1/n.
    Draws one uniform sample per stratum, then permutes.

    Variance reduction: Var(X̄_LHS) ≤ Var(X̄_SRS) always holds for
    monotone functions. Typically achieves 40–60% reduction.
    """
    strata = (np.arange(n) + rng.random(n)) / n
    return rng.permutation(strata)


def sample_demand(
    n_days: int,
    n_paths: int,
    mean: float,
    cv: float,
    spike_prob: float,
    spike_mult: float,
    use_lhs: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate demand matrix of shape (n_paths, n_days).

    Parameters
    ----------
    n_days       : int    — simulation horizon T
    n_paths      : int    — Monte Carlo sample size N
    mean         : float  — mean daily demand μ_d
    cv           : float  — coefficient of variation σ_d / μ_d
    spike_prob   : float  — P(demand shock on any given day)
    spike_mult   : float  — demand multiplier when shock occurs
    use_lhs      : bool   — use LHS (True) vs. simple random (False)
    rng          : Generator — reproducible random state

    Returns
    -------
    D : ndarray of shape (n_paths, n_days) — realized daily demand
    """
    if rng is None:
        rng = np.random.default_rng()

    mu_ln, sigma_ln = lognormal_params(mean, cv)
    total = n_paths * n_days

    if use_lhs:
        # LHS on flattened (paths × days) then reshape
        u = lhs_uniform(total, rng)
        base = np.exp(norm.ppf(u) * sigma_ln + mu_ln).reshape(n_paths, n_days)
    else:
        base = rng.lognormal(mu_ln, sigma_ln, size=(n_paths, n_days))

    # Compound Poisson shocks
    shocks = rng.random(size=(n_paths, n_days)) < spike_prob
    demand = base * np.where(shocks, spike_mult, 1.0)

    return demand.clip(min=0)


def demand_statistics(demand: np.ndarray) -> dict:
    """Summary statistics for a demand matrix."""
    flat = demand.flatten()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "cv": float(flat.std() / flat.mean()),
        "p5": float(np.percentile(flat, 5)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
    }
