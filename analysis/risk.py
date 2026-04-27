"""
risk.py — Risk Metrics & Analysis
====================================
Computes financial and operational risk metrics from simulation results.

Metrics implemented
-------------------
  VaR_α       Value at Risk at confidence level α
  CVaR_α      Conditional VaR (Expected Shortfall)
  Service SL  Fill-rate based service level (Type II)
  Bullwhip    Variance amplification ratio upstream
  Sensitivity Tornado chart via one-at-a-time parameter variation
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable


# ── Value at Risk & Expected Shortfall ────────────────────────────────────

def var(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Value at Risk at confidence level α.

    VaR_α = inf{l : P(L > l) ≤ 1 − α} = quantile(losses, α)
    """
    return float(np.percentile(losses, alpha * 100))


def cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) at confidence level α.

    CVaR_α = E[L | L ≥ VaR_α]

    CVaR is coherent (satisfies sub-additivity, monotonicity,
    translation invariance, positive homogeneity). VaR is not.
    """
    threshold = var(losses, alpha)
    tail = losses[losses >= threshold]
    return float(tail.mean()) if len(tail) > 0 else threshold


def risk_metrics(losses: np.ndarray) -> dict:
    """Full suite of risk metrics for a loss distribution."""
    losses = np.asarray(losses)
    return {
        "mean": float(losses.mean()),
        "std": float(losses.std()),
        "skewness": float(_skewness(losses)),
        "kurtosis": float(_kurtosis(losses)),
        "var_90": var(losses, 0.90),
        "var_95": var(losses, 0.95),
        "var_99": var(losses, 0.99),
        "cvar_90": cvar(losses, 0.90),
        "cvar_95": cvar(losses, 0.95),
        "cvar_99": cvar(losses, 0.99),
        "worst_1pct_avg": float(np.percentile(losses, 99)),
    }


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    mu, sigma = x.mean(), x.std()
    if sigma == 0:
        return 0.0
    return float(((x - mu) ** 3).mean() / sigma ** 3)


def _kurtosis(x: np.ndarray) -> float:
    n = len(x)
    mu, sigma = x.mean(), x.std()
    if sigma == 0:
        return 0.0
    return float(((x - mu) ** 4).mean() / sigma ** 4 - 3)


# ── Sensitivity / Tornado Analysis ────────────────────────────────────────

@dataclass
class SensitivityResult:
    parameter: str
    baseline: float
    low_value: float
    high_value: float
    low_output: float
    high_output: float

    @property
    def swing(self) -> float:
        return abs(self.high_output - self.low_output)

    @property
    def direction(self) -> str:
        return "positive" if self.high_output > self.low_output else "negative"


def sensitivity_analysis(
    base_params,
    sim_fn: Callable,
    output_fn: Callable = None,
    delta: float = 0.25,
    n_paths_sensitivity: int = 2000,
) -> list[SensitivityResult]:
    """One-at-a-time (OAT) sensitivity analysis for tornado charts.

    For each parameter p:
      1. Run sim with p × (1 − δ)  → low output
      2. Run sim with p × (1 + δ)  → high output
      3. Record swing = |high − low|

    Results are sorted by swing (largest first) for the tornado chart.

    Parameters
    ----------
    base_params : SimParams  — baseline configuration
    sim_fn      : callable   — run(SimParams) → SimResult
    output_fn   : callable   — SimResult → float  (default: service_level)
    delta       : float      — fractional perturbation (±25% default)
    """
    import copy

    if output_fn is None:
        output_fn = lambda r: r.service_level * 100

    param_names = [
        ("mean_demand",    "Mean demand"),
        ("demand_cv",      "Demand CV"),
        ("spike_prob",     "Spike probability"),
        ("fail_prob",      "Supplier fail prob"),
        ("lead_mean",      "Mean lead time"),
        ("n_suppliers",    "Num. suppliers"),
        ("supplier_corr",  "Supplier correlation"),
        ("rop",            "Reorder point"),
        ("eoq",            "Order quantity"),
    ]

    results = []
    for attr, label in param_names:
        base_val = getattr(base_params, attr)

        # Low scenario
        p_lo = copy.deepcopy(base_params)
        p_lo.n_paths = n_paths_sensitivity
        setattr(p_lo, attr, base_val * (1 - delta))
        r_lo = sim_fn(p_lo)
        out_lo = output_fn(r_lo)

        # High scenario
        p_hi = copy.deepcopy(base_params)
        p_hi.n_paths = n_paths_sensitivity
        setattr(p_hi, attr, base_val * (1 + delta))
        r_hi = sim_fn(p_hi)
        out_hi = output_fn(r_hi)

        results.append(SensitivityResult(
            parameter=label,
            baseline=base_val,
            low_value=base_val * (1 - delta),
            high_value=base_val * (1 + delta),
            low_output=out_lo,
            high_output=out_hi,
        ))

    return sorted(results, key=lambda r: r.swing, reverse=True)


# ── Scenario Analysis ─────────────────────────────────────────────────────

SCENARIOS = {
    "baseline": {},
    "demand_spike": {"spike_prob": 0.15, "spike_mult": 3.0},
    "single_supplier_fail": {"fail_prob": 0.25, "n_suppliers": 1},
    "all_suppliers_fail": {"fail_prob": 0.5, "supplier_corr": 0.9},
    "lead_time_delay": {"lead_mean_multiplier": 2.0},
    "combined_shock": {"spike_prob": 0.12, "fail_prob": 0.2, "lead_mean_multiplier": 1.5},
}


def run_scenarios(base_params, sim_fn: Callable) -> dict[str, dict]:
    """Run predefined stress scenarios and return summary stats for each."""
    import copy
    results = {}
    for name, overrides in SCENARIOS.items():
        p = copy.deepcopy(base_params)
        p.n_paths = min(base_params.n_paths, 3000)

        # Apply overrides
        mult = overrides.pop("lead_mean_multiplier", None)
        for k, v in overrides.items():
            setattr(p, k, v)
        if mult is not None:
            p.lead_mean = base_params.lead_mean * mult

        r = sim_fn(p)
        results[name] = r.summary()

    return results


# ── Convergence diagnostics ───────────────────────────────────────────────

def convergence_study(
    base_params,
    sim_fn: Callable,
    sizes: list[int] | None = None,
) -> list[dict]:
    """Measure estimate stability across increasing N.

    Returns list of dicts with n, estimate, CI width, and runtime.
    """
    import copy
    import time

    if sizes is None:
        sizes = [100, 250, 500, 1000, 2500, 5000, 10000]

    rows = []
    for n in sizes:
        p = copy.deepcopy(base_params)
        p.n_paths = n

        t0 = time.perf_counter()
        r = sim_fn(p)
        elapsed = time.perf_counter() - t0

        sl = r.service_level
        lo, hi = r.confidence_interval()
        ci_width = (hi - lo) * 100

        rows.append({
            "n": n,
            "service_level": round(sl * 100, 3),
            "ci_lo": round(lo * 100, 3),
            "ci_hi": round(hi * 100, 3),
            "ci_width": round(ci_width, 3),
            "relative_error_pct": round(ci_width / 2 / (sl * 100) * 100, 3),
            "runtime_ms": round(elapsed * 1000, 1),
        })

    return rows
