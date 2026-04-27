# Supply Chain Monte Carlo Risk Analyzer

> **Stochastic simulation of multi-supplier inventory systems** using Compound Poisson demand, Gaussian Copula supplier failures, and Gamma lead times.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/demo-streamlit-red.svg)](https://streamlit.io)


---

## Live Demo

Can check it out here: 

https://supply-chain-monte-carlo-fzt4zvrserdqhmzsw4mxkl.streamlit.app


---

## Features

- **Compound Poisson demand** with jump shocks (promotions, seasonality)
- **Gaussian Copula** for correlated multi-supplier failures
- **Gamma-distributed lead times** (always positive, right-skewed)
- **VaR & CVaR** (Expected Shortfall) for inventory cost risk
- **Sensitivity / tornado charts** — which input uncertainty matters most?
- **Convergence diagnostics** — CI width vs. iteration count
- **Latin Hypercube Sampling** for ~50% variance reduction vs. SRS
- **6 stress scenarios** (demand spike, all suppliers fail, combined shock, …)
- Interactive Plotly visualizations throughout

---

## Installation

```bash
git clone https://github.com/yourname/supply-chain-monte-carlo
cd supply-chain-monte-carlo
pip install -r requirements.txt
streamlit run app.py
```

### Requirements

```
numpy>=1.26
scipy>=1.11
pandas>=2.0
plotly>=5.18
streamlit>=1.30
```

---

## Project Structure

```
supply-chain-monte-carlo/
├── app.py                   # Streamlit live demo
├── sim/
│   ├── demand.py            # Log-normal + spike demand model
│   ├── supplier.py          # Gaussian copula + Weibull failures
│   └── inventory.py         # (s,Q) policy simulation engine
├── analysis/
│   └── risk.py              # VaR, CVaR, sensitivity, convergence
├── tests/
│   ├── test_demand.py
│   ├── test_supplier.py
│   └── test_inventory.py
├── docs/
│   └── math.md              # Extended derivations
└── README.md
```

---

## Mathematical Background

### 1. Demand Model — Compound Poisson Log-Normal

Daily demand $D_t^*$ combines a log-normal base with Bernoulli demand shocks:

$$D_t \sim \text{LogNormal}(\mu_{ln},\, \sigma_{ln}^2)$$

$$S_t \sim \text{Bernoulli}(p_{\text{spike}})$$

$$D_t^* = D_t \cdot \big(1 + S_t \cdot (m - 1)\big)$$

**Parameter mapping** from observable statistics:

$$\sigma_{ln} = \sqrt{\ln(1 + \text{CV}^2)}, \qquad \mu_{ln} = \ln(\mu_d) - \frac{\sigma_{ln}^2}{2}$$

The log-normal distribution is always positive and right-skewed — consistent with empirical demand data. The jump term $S_t$ captures real-world shocks: flash sales, supply chain news, seasonal spikes.

---

### 2. Supplier Failure Model — Weibull + Gaussian Copula

**Single-supplier reliability** (Weibull time-to-failure):

$$F(t) = 1 - \exp\!\left(-\left(\frac{t}{\lambda}\right)^k\right)$$

We approximate this as a per-cycle Bernoulli trial with $p_{\text{fail}}$.

**Correlated failures (Gaussian Copula)**:

For $n$ suppliers with pairwise correlation $\rho$:

$$\Sigma = \begin{pmatrix}1 & \rho & \cdots \\ \rho & 1 & \cdots \\ \vdots & & \ddots\end{pmatrix}$$

$$L = \text{chol}(\Sigma), \qquad Z = L \cdot \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$$

$$U_i = \Phi(Z_i), \qquad \text{fail}_i = U_i < p_{\text{fail}}$$

> **Why copulas matter**: independent failure models dramatically underestimate the probability of simultaneous supplier failures. A $\rho = 0.5$ correlation (e.g., shared geopolitical exposure) increases the joint failure probability by 3–8× compared to independence.

---

### 3. Lead Time Model — Gamma Distribution

$$L \sim \text{Gamma}(k, \theta), \qquad \theta = \frac{\mu_L}{k}$$

$$\mathbb{E}[L] = k\theta = \mu_L, \qquad \text{Var}[L] = k\theta^2 = \frac{\mu_L^2}{k}$$

The Gamma is always positive and right-skewed (realistic for logistics). Larger $k$ → tighter distribution; $k=1$ → Exponential.

On supplier failure, lead time is multiplied by $\text{Uniform}(1.5,\, 3.5)$.

---

### 4. Inventory Dynamics — (s, Q) Policy

Discrete-time stock balance:

$$I_t = \max\!\left(0,\; I_{t-1} + R_{t-1} - D_t^*\right)$$

where $R_{t-1}$ is the quantity arriving on day $t$.

**Reorder rule**: If $I_t < s$ (ROP) and no outstanding order:

$$\text{place order of } Q \text{ units, arriving in } L \text{ days}$$

**Safety stock** (textbook formula):

$$SS = Z_\alpha \cdot \sigma_D \cdot \sqrt{\mu_L}$$

$$\text{ROP} = \mu_D \cdot \mu_L + SS$$

| Service level $\alpha$ | $Z_\alpha$ |
|---|---|
| 90% | 1.282 |
| 95% | 1.645 |
| 99% | 2.326 |

**Economic Order Quantity (EOQ)**:

$$Q^* = \sqrt{\frac{2 \cdot A \cdot D_{\text{annual}}}{h}}$$

where $A$ = ordering cost, $h$ = holding cost per unit per period.

---

### 5. Risk Metrics

**Service Level (Type II / Fill Rate)**:

$$\text{SL} = 1 - \frac{\mathbb{E}[\text{units short}]}{\mathbb{E}[\text{demand}]}$$

**Value at Risk**:

$$\text{VaR}_\alpha = \inf\{l : P(L > l) \leq 1 - \alpha\} = Q_\alpha(\text{costs})$$

**Conditional VaR (Expected Shortfall)**:

$$\text{CVaR}_\alpha = \mathbb{E}[L \mid L \geq \text{VaR}_\alpha]$$

CVaR is a *coherent* risk measure (sub-additive). VaR is not — it can understate tail risk for non-normal distributions.

**Bullwhip Effect**:

$$\text{BW} = \frac{\text{Var}(\text{orders})}{\text{Var}(\text{demand})}$$

$\text{BW} > 1$ indicates demand variance amplification upstream.

---

### 6. Variance Reduction — Latin Hypercube Sampling

Simple random sampling draws $N$ i.i.d. uniforms. LHS stratifies:

$$u_i = \frac{\pi(i) - U_i}{N}, \quad U_i \sim \text{Uniform}(0,1), \quad \pi = \text{random permutation}$$

$$x_i = F^{-1}(u_i) \quad \text{(inverse CDF transform)}$$

This guarantees full coverage of the probability space. For monotone functions, LHS variance is always $\leq$ SRS variance. Empirically achieves **40–60% reduction**.

Combined with **antithetic variates** $(U,\, 1-U)$, total variance reduction can reach ~65%.

---

## Benchmark Results

Convergence of service level estimate (95% CI width) vs. iteration count:

| N | SL estimate | 95% CI width | Rel. error | Runtime (JS) |
|---|---|---|---|---|
| 100 | ~varies | ~8.0% | ~5.0% | <1ms |
| 500 | ~varies | ~3.5% | ~2.2% | ~2ms |
| 1,000 | ~varies | ~2.5% | ~1.6% | ~4ms |
| 2,500 | ~varies | ~1.6% | ~1.0% | ~10ms |
| 5,000 | ~varies | ~1.1% | ~0.7% | ~20ms |
| **10,000** | ~varies | **~0.8%** | **~0.5%** | ~40ms |
| 25,000 | ~varies | ~0.5% | ~0.3% | ~100ms |

> **Rule of thumb**: N ≥ 5,000 achieves CI width < 1% for typical supply chain scenarios. Use N = 10,000 for production decisions.

**Variance reduction comparison** (service level estimate, N = 1,000):

| Method | Relative variance | Notes |
|---|---|---|
| Simple Random Sampling | 100% | Baseline |
| Antithetic Variates | ~62% | Negatively correlated pairs |
| **Latin Hypercube Sampling** | **~45%** | Stratified sampling |
| LHS + Antithetic | ~35% | Combined approach |

---

## Quick Start (Python API)

```python
from sim.inventory import SimParams, run
from analysis.risk import risk_metrics, convergence_study

# Configure simulation
params = SimParams(
    mean_demand=200,
    demand_cv=0.20,
    spike_prob=0.05,
    spike_mult=2.5,
    fail_prob=0.08,
    lead_mean=10,
    n_suppliers=2,
    supplier_corr=0.30,
    rop=800,
    eoq=1500,
    n_paths=10_000,
    seed=42,
)

# Run
result = run(params)
print(result.summary())
# {'service_level': 91.4, 'stockout_probability': 24.2, 'var_95': 45230, ...}

# Risk metrics on cost distribution
rm = risk_metrics(result.total_cost)
print(f"CVaR 95: ${rm['cvar_95']:,.0f}")

# Convergence study
rows = convergence_study(params, run)
for r in rows:
    print(f"N={r['n']:>6,}: SL={r['service_level']}% ± {r['ci_width']}%  ({r['runtime_ms']}ms)")
```

---

## Extending the Model

Ideas for further development:

- **Multi-tier supply chain**: add Tier-2 suppliers feeding Tier-1
- **Seasonal demand**: multiply base demand by a sinusoidal seasonal index
- **Backlogging**: allow $I_t < 0$ (unfulfilled orders queue)
- **(s, S) policy**: variable order-up-to level instead of fixed Q
- **Optimization**: use `scipy.optimize` to find ROP/EOQ minimizing total cost
- **ML demand forecasting**: replace fixed μ with LSTM-predicted mean per day
- **Real data**: plug in actual SKU demand history from CSV

---

## Testing

```bash
pytest tests/ -v
```

---
