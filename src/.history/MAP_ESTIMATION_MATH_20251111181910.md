# MAP Estimation for Poisson Arrival Rate: Mathematical Details

## Poisson Process Background

In a Poisson process with rate λ:
- Inter-arrival times τ ~ Exponential(λ)
- Number of arrivals in time t ~ Poisson(λt)

## MLE Estimation

Given n observed inter-arrival times: τ₁, τ₂, ..., τₙ

**Likelihood:**
$$L(\lambda | \tau) = \prod_{i=1}^{n} \lambda e^{-\lambda \tau_i} = \lambda^n e^{-\lambda \sum_{i=1}^{n} \tau_i}$$

**Log-likelihood:**
$$\log L(\lambda | \tau) = n \log(\lambda) - \lambda \sum_{i=1}^{n} \tau_i$$

**MLE estimate:**
$$\hat{\lambda}_{MLE} = \frac{n}{\sum_{i=1}^{n} \tau_i}$$

This is the **sample rate**: number of events divided by total time.

## MAP Estimation

### Prior Distribution
We use a **Gamma prior** (conjugate prior for Poisson rate):
$$\lambda \sim \text{Gamma}(\alpha, \beta)$$

Where:
- α = shape parameter (pseudo-count of prior arrivals)
- β = rate parameter (pseudo-count of prior time)
- Prior mean: E[λ] = α/β
- Prior variance: Var[λ] = α/β²

**PDF:**
$$p(\lambda) = \frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha-1} e^{-\beta\lambda}$$

### Posterior Distribution

Given observed data, the posterior is also Gamma:
$$\lambda | \tau \sim \text{Gamma}(\alpha + n, \beta + \sum_{i=1}^{n} \tau_i)$$

**Key insight**: Gamma prior + Exponential likelihood → Gamma posterior (conjugacy)

### MAP Estimate

The MAP estimate is the **mode** of the posterior:
$$\hat{\lambda}_{MAP} = \frac{\alpha + n - 1}{\beta + \sum_{i=1}^{n} \tau_i}$$

For α > 1, this is well-defined. For α ≤ 1, use posterior mean instead:
$$\hat{\lambda}_{MAP} = \frac{\alpha + n}{\beta + \sum_{i=1}^{n} \tau_i}$$

## Comparison: MLE vs MAP

### As a Formula
$$\hat{\lambda}_{MAP} = \frac{(\alpha - 1) + n}{(\beta) + \sum \tau_i} = \frac{n_{prior} + n_{data}}{T_{prior} + T_{data}}$$

Where:
- n_prior = α - 1 (effective prior observations)
- T_prior = β (effective prior time)
- n_data = n (actual observations)
- T_data = Σ τᵢ (actual time)

**Interpretation**: MAP is like MLE with extra "pseudo-observations" from the prior!

### Special Cases

1. **No prior (α → 0, β → 0)**:
   - MAP → MLE
   - Pure data-driven

2. **Infinite data (n → ∞)**:
   - MAP → MLE
   - Prior overwhelmed by data

3. **Strong prior (α large, β large)**:
   - MAP stays close to prior mean α/β
   - Requires lots of data to shift estimate

4. **Weak prior (α small, β small)**:
   - MAP quickly approaches MLE
   - Prior only matters early on

## Prior Selection Guidelines

### Setting Prior Parameters

Given desired prior mean μ₀ and "equivalent sample size" n₀:

$$\alpha = n_0 + 1$$
$$\beta = \frac{n_0}{\mu_0}$$

**Example**: Want λ ≈ 0.05 with equivalent of 5 prior observations:
- α = 6
- β = 5 / 0.05 = 100
- Prior mean: 6/100 = 0.06 ≈ 0.05 ✓

### Weak Prior (Recommended Start)
```python
prior_shape = 2.0          # α
prior_rate = 2.0 / 0.05    # β = 40.0
# Equivalent to ~1 prior observation
```

### Medium Prior
```python
prior_shape = 5.0          # α
prior_rate = 5.0 / 0.05    # β = 100.0
# Equivalent to ~4 prior observations
```

### Strong Prior
```python
prior_shape = 10.0         # α
prior_rate = 10.0 / 0.05   # β = 200.0
# Equivalent to ~9 prior observations
```

## Convergence Analysis

### After n observations:

**MLE estimate:**
$$\hat{\lambda}_{MLE} = \frac{n}{\sum \tau_i}$$

**MAP estimate:**
$$\hat{\lambda}_{MAP} = \frac{\alpha + n - 1}{\beta + \sum \tau_i}$$

**Difference:**
$$|\hat{\lambda}_{MAP} - \hat{\lambda}_{MLE}| = \left|\frac{\alpha - 1}{\beta + \sum \tau_i} - \frac{n}{\sum \tau_i}\right|$$

As n → ∞:
- Σ τᵢ → n/λ_true (by Law of Large Numbers)
- β becomes negligible compared to Σ τᵢ
- (α - 1) becomes negligible compared to n
- Therefore: MAP → MLE

### Rate of Convergence

For weak prior (α = 2, β = 40):
- After 10 observations: ~10% difference
- After 50 observations: ~2% difference
- After 100 observations: ~1% difference

## Implementation in Code

```python
def _update_map_estimate(self, n, sum_tau, alpha, beta):
    """
    Update MAP estimate given data and prior.
    
    Args:
        n: Number of observed inter-arrivals
        sum_tau: Sum of inter-arrival times
        alpha: Prior shape parameter
        beta: Prior rate parameter
    
    Returns:
        MAP estimate of λ
    """
    posterior_shape = alpha + n
    posterior_rate = beta + sum_tau
    
    if posterior_shape > 1:
        # Use mode of Gamma distribution
        lambda_map = (posterior_shape - 1) / posterior_rate
    else:
        # Use mean when mode undefined
        lambda_map = posterior_shape / posterior_rate
    
    return lambda_map
```

## Uncertainty Quantification

One advantage of MAP: we have the full posterior!

**Posterior mean:**
$$E[\lambda | \tau] = \frac{\alpha + n}{\beta + \sum \tau_i}$$

**Posterior variance:**
$$\text{Var}[\lambda | \tau] = \frac{\alpha + n}{(\beta + \sum \tau_i)^2}$$

**95% Credible Interval:**
Use quantiles of Gamma(α + n, β + Σ τᵢ) distribution

This can be used for:
- Confidence-weighted decisions
- Exploration-exploitation trade-offs
- Risk-aware planning

## References

1. Gelman, A., et al. (2013). *Bayesian Data Analysis*. 3rd edition.
2. Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Chapter 2.
3. Murphy, K. (2012). *Machine Learning: A Probabilistic Perspective*. Chapter 3.

---

**Note**: Current implementation uses MAP estimate (point estimate). Future work could leverage full posterior for uncertainty-aware decisions.
