"""
supplier.py — Supplier Failure Model
======================================
Models multi-supplier reliability using:

  1. Per-cycle Bernoulli failure trials  (approximates Weibull TTF)
  2. Gaussian Copula for correlated failures between suppliers

Weibull CDF (theoretical basis):
  F(t) = 1 − exp(−(t/λ)^k)

Gaussian Copula (joint failure correlation):
  Σ = [[1, ρ, ...], [ρ, 1, ...], ...]   — correlation matrix
  L = cholesky(Σ)                         — Cholesky decomposition
  Z = L @ N(0,1)^n                        — correlated normals
  U_i = Φ(Z_i)                           — uniform marginals
  fail_i = U_i < p_fail                  — threshold mapping
"""

import numpy as np
from scipy.stats import norm


# ── Copula ─────────────────────────────────────────────────────────────────

def gaussian_copula_uniforms(
    n_samples: int,
    n_suppliers: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample correlated U[0,1] via Gaussian copula.

    All pairs share the same pairwise correlation ρ (exchangeable copula).
    Uses Cholesky decomposition for numerical stability.

    Parameters
    ----------
    n_samples    : int   — number of reorder events to simulate
    n_suppliers  : int   — number of Tier-1 suppliers
    rho          : float — pairwise correlation in [0, 1)
    rng          : Generator

    Returns
    -------
    U : ndarray of shape (n_samples, n_suppliers) in (0, 1)
    """
    if n_suppliers == 1:
        return rng.random((n_samples, 1))

    # Build correlation matrix (exchangeable)
    corr = np.full((n_suppliers, n_suppliers), rho)
    np.fill_diagonal(corr, 1.0)

    # Cholesky decomposition
    L = np.linalg.cholesky(corr)

    # Independent standard normals → correlated via L
    Z_indep = rng.standard_normal((n_samples, n_suppliers))
    Z_corr = Z_indep @ L.T

    # Map to U[0,1] via standard normal CDF
    U = norm.cdf(Z_corr)
    return U


# ── Lead time sampler (Gamma distribution) ─────────────────────────────────

def sample_lead_times(
    n: int,
    mean: float,
    shape_k: float,
    rng: np.random.Generator,
    failure_multiplier_range: tuple[float, float] = (1.5, 3.5),
    failed: np.ndarray | None = None,
) -> np.ndarray:
    """Sample lead times from Gamma distribution.

    L ~ Gamma(k, θ),  θ = mean / k
    E[L] = k·θ = mean,  Var[L] = k·θ² = mean²/k

    The Gamma distribution is always positive and right-skewed —
    realistic for logistics delays.

    If `failed` is provided (boolean array), failed-supplier orders
    incur an additional random delay multiplier.
    """
    theta = mean / shape_k
    L = rng.gamma(shape=shape_k, scale=theta, size=n)

    if failed is not None and failed.any():
        mult_lo, mult_hi = failure_multiplier_range
        mults = rng.uniform(mult_lo, mult_hi, size=n)
        L = np.where(failed, L * mults, L)

    return np.maximum(1, np.round(L)).astype(int)


# ── Main supplier simulation ────────────────────────────────────────────────

def simulate_supplier_events(
    n_reorder_events: int,
    n_suppliers: int,
    fail_prob: float,
    rho: float,
    lead_mean: float,
    lead_k: float,
    rng: np.random.Generator,
) -> dict:
    """Simulate a batch of reorder events with correlated supplier failures.

    For each reorder event:
      - Each supplier independently (but correlated) may fail
      - If ≥1 supplier available: normal lead time
      - If all fail: emergency procurement (longer delay, partial fill)

    Returns
    -------
    dict with keys:
      'any_available'  : bool array (n_reorder_events,)
      'all_failed'     : bool array (n_reorder_events,)
      'lead_times'     : int array  (n_reorder_events,)
      'fill_rate'      : float array (n_reorder_events,) — fraction of EOQ fulfilled
    """
    # Correlated failure draws via Gaussian copula
    U = gaussian_copula_uniforms(n_reorder_events, n_suppliers, rho, rng)
    failures = U < fail_prob                 # shape (n_reorder_events, n_suppliers)

    n_failed = failures.sum(axis=1)          # failed suppliers per event
    any_available = n_failed < n_suppliers
    all_failed = n_failed == n_suppliers

    # Lead times
    lead_times = sample_lead_times(
        n_reorder_events, lead_mean, lead_k, rng,
        failed=any_available & (n_failed > 0),  # partial failure → longer
    )

    # Emergency supplier on total failure (longer lead, partial fill)
    emergency_lead = sample_lead_times(
        n_reorder_events, lead_mean, lead_k, rng,
        failure_multiplier_range=(2.5, 5.0),
        failed=all_failed,
    )
    final_leads = np.where(all_failed, emergency_lead, lead_times)

    # Fill rate: all good → 100%, partial failure → 80–100%, all fail → 40–60%
    fill_rate = np.where(
        all_failed,
        rng.uniform(0.4, 0.6, n_reorder_events),
        np.where(
            any_available & (n_failed > 0),
            rng.uniform(0.8, 1.0, n_reorder_events),
            1.0,
        ),
    )

    return {
        "any_available": any_available,
        "all_failed": all_failed,
        "lead_times": final_leads,
        "fill_rate": fill_rate,
        "n_failed_per_event": n_failed,
    }


def supplier_reliability_stats(events: dict) -> dict:
    """Compute summary reliability statistics from simulated events."""
    return {
        "total_failure_prob": float(events["all_failed"].mean()),
        "partial_failure_prob": float(
            (events["n_failed_per_event"] > 0).mean() - events["all_failed"].mean()
        ),
        "mean_lead_time": float(events["lead_times"].mean()),
        "p95_lead_time": float(np.percentile(events["lead_times"], 95)),
        "mean_fill_rate": float(events["fill_rate"].mean()),
    }
