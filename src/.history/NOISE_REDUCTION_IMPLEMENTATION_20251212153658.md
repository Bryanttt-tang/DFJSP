# Noise Reduction Implementation Summary

## Problem Identified

Training plots showed that **Rule-Based RL** and **Reactive RL** have very noisy reward curves, while **Perfect RL** is smooth - despite Rule-Based RL having only 10 actions.

### Root Cause: Environment Stochasticity, NOT Action Space Size

**Perfect RL (Smooth):**
- All jobs known at t=0 (deterministic environment)
- Same initial state → same trajectory every episode
- Reward variance only from policy learning

**Rule-Based & Reactive RL (Noisy):**
- Jobs arrive via **Poisson process** (λ=0.05-0.1)
- Random arrivals → different episode trajectories
- High reward variance from **BOTH policy AND environment**
- Each episode: Different jobs arrive at different times → completely different makespans

**Mathematical explanation:**
```
Total variance = Policy variance + Environment variance

Perfect RL:    σ² = σ²_policy + 0          ≈ small
Rule-Based RL: σ² = σ²_policy + σ²_Poisson  ≈ LARGE (Poisson has exponential variance!)
```

---

## Solutions Implemented ✅

### 1. **Doubled Rollout Buffer Size** (n_steps: 2048 → 4096)

**Why:** Collect more samples per update to average out Poisson variance

**Impact:**
- More episodes per rollout → variance averaged across more random arrivals
- Better value function estimates
- Slightly slower training (more steps before update)

### 2. **Doubled Batch Size** (batch_size: 512/1024 → 1024)

**Why:** Larger batches produce more stable gradient estimates

**Impact:**
- Gradient estimates less affected by individual noisy episodes
- Smoother policy updates
- Requires more memory

### 3. **Increased GAE Lambda** (gae_lambda: 0.95 → 0.98)

**Why:** Higher lambda reduces variance in advantage estimates (slight bias tradeoff)

**Impact:**
- Advantage estimates less sensitive to value function errors
- Smoother learning signal
- Small increase in bias (acceptable for episodic tasks)

### 4. **Added Explanatory Comments**

Clear documentation of why these changes matter for stochastic environments.

---

## Changes Applied

### Rule-Based RL (Lines 3500-3530)

**Before:**
```python
n_steps=2048,       # Larger rollout buffer
batch_size=512,     # Larger batches for stable gradients
gae_lambda=0.95,    # Standard GAE
ent_coef=0.01,      # Higher entropy bonus
```

**After:**
```python
n_steps=4096,       # ⭐ DOUBLED: Average out Poisson variance
batch_size=1024,    # ⭐ DOUBLED: More stable gradients despite noise
gae_lambda=0.98,    # ⭐ HIGHER: Reduce variance (slight bias tradeoff)
ent_coef=0.03,      # Higher entropy bonus for exploration
```

**Added documentation:**
```python
# NOISE REDUCTION for Poisson arrivals (high environment variance):
# - DOUBLED n_steps (2048→4096): More samples per update to average variance
# - DOUBLED batch_size (512→1024): Larger batches for stable gradients
# - HIGHER gae_lambda (0.95→0.98): Reduce variance in advantage estimates
```

---

### Reactive RL (Lines 3415-3445)

**Before:**
```python
n_steps=2048,       # Large rollout for better value estimates
batch_size=1024,    # Increased from 512
gae_lambda=0.95,    # GAE for advantage estimation
```

**After:**
```python
n_steps=4096,       # ⭐ DOUBLED: Average out Poisson variance
batch_size=1024,    # Large batches for stable gradients
gae_lambda=0.98,    # ⭐ HIGHER: Reduce variance in advantage estimates
```

**Added documentation:**
```python
# OPTIMIZED hyperparameters with NOISE REDUCTION for Poisson stochastic environment
# Challenge: High variance from random Poisson arrivals affects gradient quality
# Solution: Larger batches + longer rollouts to average out environment variance
```

---

### Proactive RL (Lines 3650-3680)

**Before:**
```python
n_steps=2048,       # LARGER rollouts for lower variance
batch_size=512,     # LARGER batches for stable gradients
gae_lambda=0.95,    # Lower GAE for less bias
```

**After:**
```python
n_steps=4096,       # ⭐ DOUBLED: Average out Poisson variance
batch_size=1024,    # ⭐ DOUBLED: More stable gradients despite noise
gae_lambda=0.98,    # ⭐ HIGHER: Reduce variance in advantage estimates
```

**Added documentation:**
```python
# OPTIMIZED hyperparameters with NOISE REDUCTION for Proactive RL
# Challenge: Large action space (jobs × machines) + stochastic Poisson arrivals + prediction
# Solution: Conservative updates, better exploration, LARGER batches to handle variance
```

---

## Expected Results

### Training Metrics Improvement

**Before (Noisy):**
```
Mean Episode Reward: -380 to -320 (high variance, oscillating)
KL Divergence: Spiky (0.002 → 0.010)
Policy Loss: Oscillating
```

**After (Smoother):**
```
Mean Episode Reward: Smoother curve with clear trend
KL Divergence: < 0.01, more stable
Policy Loss: Less oscillation, clearer convergence
```

### Performance Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Training stability** | Noisy, hard to see progress | Smoother, clearer trends | +++ |
| **Convergence speed** | May converge slower | May converge faster | + |
| **Memory usage** | Lower | Higher (2x batch) | - |
| **Time per update** | Faster | Slightly slower (2x samples) | - |
| **Final performance** | Same (eventually) | Same or better | = or + |

**Net benefit:** More stable training, easier to tune, clearer when converged.

---

## Why This Works: Statistical Perspective

### Central Limit Theorem Application

With larger batches, the gradient estimate converges to true gradient:

```
Gradient estimate ~ N(μ, σ²/n)

Where:
- μ = true gradient
- σ² = variance per sample (high due to Poisson)
- n = batch size

Doubling n: σ²/2n → Standard error reduced by √2 ≈ 1.41x
```

### GAE Variance Reduction

Higher λ (0.98 vs 0.95) means:
- More weight on actual returns (less on value function estimates)
- Value function errors have less impact
- Advantage estimates more stable

**Tradeoff:** Slightly higher bias, but worth it for noisy environments.

---

## Alternative Solutions (Not Implemented)

These could be added if noise persists:

### 1. Reward Normalization
```python
from stable_baselines3.common.vec_env import VecNormalize

vec_env = VecNormalize(
    vec_env,
    norm_obs=False,
    norm_reward=True,    # Normalize rewards
    clip_reward=10.0     # Clip extremes
)
```

### 2. Fixed Arrival Patterns (for testing)
```python
# Use deterministic arrivals instead of Poisson
arrival_times = [10, 20, 30, 40, 50]  # Fixed
# Instead of: Poisson(λ=0.1)
```

### 3. Multi-Seed Averaging
```python
# Train multiple seeds, average results
for seed in [0, 42, 123, 456, 789]:
    model = train(..., seed=seed)
```

### 4. Longer Training
```python
# With high variance, need more samples
total_timesteps = 3_000_000  # 3M instead of 1M
```

---

## Validation

To verify the changes work:

1. **Check training plots:**
   - Reward curve should be smoother
   - KL divergence should stay < 0.01
   - Less oscillation in policy loss

2. **Monitor convergence:**
   - Training should show clearer improvement trend
   - Final performance should match or exceed previous

3. **Compare metrics:**
   - Entropy should decrease gradually (learning)
   - Explained variance should increase (better value function)

---

## Summary

**Problem:** Poisson arrivals create high environment variance → noisy training

**Solution:** Average out variance with larger batches/rollouts + reduce advantage variance

**Trade-offs:** Slightly slower training, more memory, but much more stable

**Expected outcome:** Smoother training curves, clearer convergence, same or better final performance

The key insight: **Action space size doesn't determine training noise - environment stochasticity does!** Rule-Based RL (10 actions) and Reactive RL (large action space) are both noisy because they face the **same Poisson variance problem**.
