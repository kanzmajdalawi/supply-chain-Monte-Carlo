"""
tests/test_inventory.py — Integration tests for the MC simulation engine
"""

import numpy as np
import pytest
from sim.inventory import SimParams, run
from sim.demand import sample_demand, lognormal_params
from sim.supplier import gaussian_copula_uniforms, simulate_supplier_events
from analysis.risk import var, cvar, risk_metrics


# ── Demand tests ───────────────────────────────────────────────────────────

class TestDemand:
    def test_lognormal_mean(self):
        """Sample mean should be close to configured mean."""
        rng = np.random.default_rng(0)
        D = sample_demand(1, 50_000, mean=200, cv=0.2,
                          spike_prob=0, spike_mult=1, rng=rng)
        assert abs(D.mean() - 200) / 200 < 0.02  # within 2%

    def test_lognormal_cv(self):
        """Sample CV should match configured CV (ignoring spike=0)."""
        rng = np.random.default_rng(0)
        D = sample_demand(1, 100_000, mean=200, cv=0.25,
                          spike_prob=0, spike_mult=1, rng=rng)
        cv_empirical = D.std() / D.mean()
        assert abs(cv_empirical - 0.25) < 0.01

    def test_demand_always_positive(self):
        rng = np.random.default_rng(42)
        D = sample_demand(90, 1000, mean=150, cv=0.4,
                          spike_prob=0.1, spike_mult=3, rng=rng)
        assert (D >= 0).all()

    def test_spikes_increase_mean(self):
        rng = np.random.default_rng(7)
        D_no_spike = sample_demand(90, 5000, 200, 0.2, 0, 3, rng=rng)
        rng = np.random.default_rng(7)
        D_spike = sample_demand(90, 5000, 200, 0.2, 0.1, 3, rng=rng)
        assert D_spike.mean() > D_no_spike.mean()

    def test_lhs_vs_srs_lower_variance(self):
        """LHS should give lower variance of sample mean than SRS."""
        means_lhs, means_srs = [], []
        for seed in range(50):
            rng = np.random.default_rng(seed)
            D_lhs = sample_demand(1, 500, 200, 0.2, 0, 1, use_lhs=True, rng=rng)
            rng = np.random.default_rng(seed)
            D_srs = sample_demand(1, 500, 200, 0.2, 0, 1, use_lhs=False, rng=rng)
            means_lhs.append(D_lhs.mean())
            means_srs.append(D_srs.mean())
        # LHS variance of mean should be ≤ SRS (with high probability)
        var_lhs = np.var(means_lhs)
        var_srs = np.var(means_srs)
        assert var_lhs <= var_srs * 1.2  # allow 20% slack for finite sample noise


# ── Supplier tests ─────────────────────────────────────────────────────────

class TestSupplier:
    def test_failure_probability_marginal(self):
        """Each supplier's marginal failure prob should match p_fail."""
        rng = np.random.default_rng(0)
        U = gaussian_copula_uniforms(100_000, 3, rho=0.3, rng=rng)
        p_fail = 0.1
        empirical = (U < p_fail).mean(axis=0)
        np.testing.assert_allclose(empirical, p_fail, atol=0.005)

    def test_correlation_direction(self):
        """Higher ρ should increase joint failure probability."""
        rng = np.random.default_rng(1)
        p_fail = 0.1
        U_lo = gaussian_copula_uniforms(50_000, 2, rho=0.0, rng=rng)
        rng = np.random.default_rng(1)
        U_hi = gaussian_copula_uniforms(50_000, 2, rho=0.8, rng=rng)
        joint_lo = ((U_lo < p_fail).all(axis=1)).mean()
        joint_hi = ((U_hi < p_fail).all(axis=1)).mean()
        assert joint_hi > joint_lo

    def test_lead_time_positive(self):
        rng = np.random.default_rng(2)
        evt = simulate_supplier_events(1000, 2, 0.1, 0.3, 10, 4, rng)
        assert (evt["lead_times"] >= 1).all()

    def test_fill_rate_bounds(self):
        rng = np.random.default_rng(3)
        evt = simulate_supplier_events(5000, 2, 0.3, 0.5, 10, 4, rng)
        assert (evt["fill_rate"] >= 0).all()
        assert (evt["fill_rate"] <= 1).all()


# ── Risk metric tests ─────────────────────────────────────────────────────

class TestRiskMetrics:
    def test_var_quantile(self):
        losses = np.arange(1, 101, dtype=float)
        assert var(losses, 0.95) == pytest.approx(95, abs=1)

    def test_cvar_exceeds_var(self):
        rng = np.random.default_rng(0)
        losses = rng.exponential(scale=1000, size=10_000)
        assert cvar(losses, 0.95) > var(losses, 0.95)

    def test_cvar_coherent_monotone(self):
        """CVaR should increase as alpha increases."""
        rng = np.random.default_rng(0)
        losses = rng.lognormal(7, 0.5, 10_000)
        assert cvar(losses, 0.90) <= cvar(losses, 0.95) <= cvar(losses, 0.99)


# ── Integration tests ─────────────────────────────────────────────────────

class TestSimulation:
    @pytest.fixture
    def base_params(self):
        return SimParams(
            mean_demand=200, demand_cv=0.2, spike_prob=0.05, spike_mult=2.5,
            fail_prob=0.08, lead_mean=10, lead_k=4, n_suppliers=2, supplier_corr=0.3,
            rop=800, eoq=1500, init_inv=2000, sim_days=60, n_paths=500, seed=42,
        )

    def test_service_level_bounds(self, base_params):
        r = run(base_params)
        assert 0.0 <= r.service_level <= 1.0

    def test_service_level_improves_with_more_suppliers(self, base_params):
        base_params.n_paths = 500
        r1 = run(base_params)

        base_params.n_suppliers = 4
        r4 = run(base_params)
        # More suppliers → higher (or equal) service level
        assert r4.service_level >= r1.service_level - 0.05  # allow noise

    def test_service_level_degrades_with_higher_fail_prob(self, base_params):
        base_params.fail_prob = 0.01
        r_reliable = run(base_params)

        base_params.fail_prob = 0.40
        r_unreliable = run(base_params)
        assert r_reliable.service_level >= r_unreliable.service_level - 0.05

    def test_cvar_exceeds_var_in_sim(self, base_params):
        r = run(base_params)
        assert r.cvar95 >= r.var95

    def test_seed_reproducibility(self, base_params):
        r1 = run(base_params)
        r2 = run(base_params)
        np.testing.assert_array_equal(r1.final_inventory, r2.final_inventory)

    def test_output_shapes(self, base_params):
        r = run(base_params)
        assert len(r.final_inventory) == base_params.n_paths
        assert len(r.daily_stockout_prob) == base_params.sim_days
        assert len(r.daily_mean_inventory) == base_params.sim_days
