"""
inventory.py — Monte Carlo Inventory Simulator
=================================================
Implements a (s, Q) continuous-review inventory policy:

  - Review inventory every period (day)
  - If I_t < s (ROP):  place order of Q units
  - Order arrives after stochastic lead time L ~ Gamma(k, θ)
  - Demand D*_t ~ Compound Poisson Log-Normal
  - Suppliers may fail with probability p (correlated via Copula)

Key equations
-------------
  I_t   = I_{t-1} + R_{t-1} − D*_t       (stock balance)
  SS    = Z_α · σ_D · √μ_L               (safety stock)
  ROP   = μ_D · μ_L + SS                 (reorder point)
  EOQ   = √(2·A·D / h)                   (economic order qty)

where A = ordering cost, h = holding cost per unit per period.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .demand import sample_demand
from .supplier import simulate_supplier_events


# ── Parameter dataclass ────────────────────────────────────────────────────

@dataclass
class SimParams:
    # Demand
    mean_demand: float = 200.0
    demand_cv: float = 0.20
    spike_prob: float = 0.05
    spike_mult: float = 2.5

    # Supplier
    fail_prob: float = 0.08
    lead_mean: float = 10.0
    lead_k: float = 4.0
    n_suppliers: int = 2
    supplier_corr: float = 0.30

    # Inventory policy (s, Q)
    rop: float = 800.0          # reorder point s
    eoq: float = 1500.0         # order quantity Q
    init_inv: float = 2000.0    # I_0

    # Cost parameters
    holding_cost: float = 1.0    # $/unit/day
    shortage_cost: float = 10.0  # $/unit short
    ordering_cost: float = 200.0 # $/order
    unit_cost: float = 0.5       # $/unit ordered

    # Simulation
    sim_days: int = 90
    n_paths: int = 5000
    use_lhs: bool = True
    seed: Optional[int] = None

    def safety_stock(self, service_level: float = 0.95) -> float:
        """Textbook safety stock formula: SS = Z_α · σ_D · √μ_L"""
        z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}.get(service_level, 1.645)
        sigma_d = self.mean_demand * self.demand_cv
        return z * sigma_d * np.sqrt(self.lead_mean)

    def recommended_rop(self, service_level: float = 0.95) -> float:
        """ROP = μ_D · μ_L + SS"""
        return self.mean_demand * self.lead_mean + self.safety_stock(service_level)

    def eoq_formula(self) -> float:
        """EOQ = √(2 · A · D_annual / h)"""
        d_annual = self.mean_demand * 365
        return np.sqrt(2 * self.ordering_cost * d_annual / self.holding_cost)


# ── Simulation result ──────────────────────────────────────────────────────

@dataclass
class SimResult:
    params: SimParams

    # Per-path summary statistics (shape: n_paths)
    final_inventory: np.ndarray = field(default_factory=lambda: np.array([]))
    total_shortfall: np.ndarray = field(default_factory=lambda: np.array([]))
    total_demand: np.ndarray = field(default_factory=lambda: np.array([]))
    stockout_days: np.ndarray = field(default_factory=lambda: np.array([]))
    total_holding_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    total_ordering_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    total_shortage_cost: np.ndarray = field(default_factory=lambda: np.array([]))

    # Time-series (shape: sim_days)
    daily_stockout_prob: np.ndarray = field(default_factory=lambda: np.array([]))
    daily_mean_inventory: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def total_cost(self) -> np.ndarray:
        return self.total_holding_cost + self.total_ordering_cost + self.total_shortage_cost

    @property
    def service_level(self) -> float:
        """Fill rate: fraction of demand units fulfilled."""
        return 1.0 - self.total_shortfall.sum() / self.total_demand.sum()

    @property
    def stockout_probability(self) -> float:
        """Fraction of paths that experienced ≥1 stockout."""
        return float((self.stockout_days > 0).mean())

    @property
    def var95(self) -> float:
        return float(np.percentile(self.total_cost, 95))

    @property
    def cvar95(self) -> float:
        thresh = self.var95
        tail = self.total_cost[self.total_cost >= thresh]
        return float(tail.mean()) if len(tail) > 0 else thresh

    @property
    def bullwhip_ratio(self) -> float:
        """Var(orders) / Var(demand) — bullwhip effect measure."""
        # Approximate from per-path totals
        return float(self.total_ordering_cost.var() / max(self.total_demand.var(), 1e-9))

    def confidence_interval(self, stat: str = "service_level", alpha: float = 0.05) -> tuple:
        """Bootstrap 95% CI for a scalar statistic."""
        if stat == "service_level":
            # CLT-based CI for proportion
            p = self.service_level
            n = len(self.final_inventory)
            se = np.sqrt(p * (1 - p) / n)
            z = 1.96
            return (p - z * se, p + z * se)
        raise NotImplementedError(f"CI not implemented for {stat}")

    def summary(self) -> dict:
        lo, hi = self.confidence_interval()
        return {
            "service_level": round(self.service_level * 100, 2),
            "service_level_ci95": (round(lo * 100, 2), round(hi * 100, 2)),
            "stockout_probability": round(self.stockout_probability * 100, 2),
            "mean_stockout_days": round(float(self.stockout_days.mean()), 2),
            "median_final_inventory": round(float(np.median(self.final_inventory)), 0),
            "p5_final_inventory": round(float(np.percentile(self.final_inventory, 5)), 0),
            "mean_total_cost": round(float(self.total_cost.mean()), 2),
            "var_95": round(self.var95, 2),
            "cvar_95": round(self.cvar95, 2),
            "bullwhip_ratio": round(self.bullwhip_ratio, 4),
            "n_paths": len(self.final_inventory),
        }


# ── Core simulation engine ─────────────────────────────────────────────────

def run(params: SimParams) -> SimResult:
    """Run Monte Carlo inventory simulation.

    Vectorizes across paths where possible; falls back to a
    path-by-path loop for the order pipeline (pending orders
    with variable lead times are inherently sequential per path).

    Complexity: O(N · T) time, O(N · T) space for demand matrix.
    """
    rng = np.random.default_rng(params.seed)
    N, T = params.n_paths, params.sim_days

    # ── Pre-generate demand matrix (vectorized, LHS) ──────────────────────
    D = sample_demand(
        n_days=T,
        n_paths=N,
        mean=params.mean_demand,
        cv=params.demand_cv,
        spike_prob=params.spike_prob,
        spike_mult=params.spike_mult,
        use_lhs=params.use_lhs,
        rng=rng,
    )  # shape: (N, T)

    # ── Output arrays ─────────────────────────────────────────────────────
    final_inv = np.zeros(N)
    total_shortfall = np.zeros(N)
    total_demand_sum = np.zeros(N)
    stockout_days = np.zeros(N, dtype=int)
    hold_cost = np.zeros(N)
    order_cost = np.zeros(N)
    short_cost = np.zeros(N)
    daily_so_count = np.zeros(T)
    daily_inv_sum = np.zeros(T)

    # ── Path-by-path simulation loop ─────────────────────────────────────
    # (Order pipeline requires per-path state — can't be vectorized easily)
    for i in range(N):
        inv = float(params.init_inv)
        pending: list[tuple[int, float]] = []  # (arrival_day, qty)
        on_order = False  # simple: only one outstanding order at a time

        for d in range(T):
            # Receive arriving orders
            arrived = [qty for arr, qty in pending if arr <= d]
            pending = [(arr, qty) for arr, qty in pending if arr > d]
            inv += sum(arrived)
            if arrived:
                on_order = False

            # Satisfy demand
            dem = float(D[i, d])
            total_demand_sum[i] += dem
            if inv >= dem:
                inv -= dem
            else:
                short = dem - inv
                total_shortfall[i] += short
                short_cost[i] += short * params.shortage_cost
                stockout_days[i] += 1
                inv = 0.0

            hold_cost[i] += inv * params.holding_cost
            daily_so_count[d] += 1 if inv == 0 else 0
            daily_inv_sum[d] += inv

            # Reorder trigger: (s, Q) policy
            if inv < params.rop and not on_order:
                # Simulate this single reorder event
                evt = simulate_supplier_events(
                    n_reorder_events=1,
                    n_suppliers=params.n_suppliers,
                    fail_prob=params.fail_prob,
                    rho=params.supplier_corr,
                    lead_mean=params.lead_mean,
                    lead_k=params.lead_k,
                    rng=rng,
                )
                lt = int(evt["lead_times"][0])
                fill = float(evt["fill_rate"][0])
                qty = params.eoq * fill
                pending.append((d + lt, qty))
                order_cost[i] += params.ordering_cost + qty * params.unit_cost
                on_order = True

        final_inv[i] = inv

    # ── Aggregate time-series ─────────────────────────────────────────────
    daily_so_prob = daily_so_count / N
    daily_mean_inv = daily_inv_sum / N

    return SimResult(
        params=params,
        final_inventory=final_inv,
        total_shortfall=total_shortfall,
        total_demand=total_demand_sum,
        stockout_days=stockout_days,
        total_holding_cost=hold_cost,
        total_ordering_cost=order_cost,
        total_shortage_cost=short_cost,
        daily_stockout_prob=daily_so_prob,
        daily_mean_inventory=daily_mean_inv,
    )
