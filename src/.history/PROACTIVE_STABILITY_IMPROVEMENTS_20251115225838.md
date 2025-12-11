# Proactive RL Training Stability Improvements

## Problem Analysis

The original training metrics showed severe instability:
- **Mean episode reward**: Very noisy (-120 to -180, high variance)
- **Entropy loss**: Increasing from -2.0 to -1.0 (agent not converging)
- **Episode length**: Highly variable (55-95 steps, unstable)
- **Policy loss**: Oscillating (not converging smoothly)

## Root Causes Identified

### 1. **Observation Space Too Large & Complex (128 dimensions)**

**OLD observation space:**
```python
- Job ready indicators: 12 dims
- Machine free times: 6 dims
- Processing times: 12×6 = 72 dims
- Job progress: 12 dims
- Predicted arrivals: 12 dims
- Actual arrivals: 12 dims  # REDUNDANT!
- Arrival rate + progress: 2 dims
Total: 128 dimensions
```

**Problems:**
- Both predicted AND actual arrival times (confusing/redundant)
- Processing times for ALL job-machine pairs (72 features, mostly zeros for unarrived jobs)
- Arrival rate as global feature (not very informative)

### 2. **Weak Hyperparameters**

**OLD settings:**
```python
n_steps=1024           # Too small for proactive decisions
batch_size=256         # Small batches → noisy gradients
n_epochs=2             # Too few updates per rollout
ent_coef=0.001         # Too low → no exploration of wait timing
learning_rate=3e-4     # Fixed → no adaptation
```

### 3. **No Observation Normalization**

Raw observations with different scales (times, rates, progress) → poor gradient flow

### 4. **Action Space Complexity**

Wait actions (10, inf) might not provide enough granularity for learning optimal timing

## Solutions Implemented

### ✅ 1. Simplified Observation Space (116 dimensions)

**NEW observation space:**
```python
- Job ready time: 12 dims
- Job progress: 12 dims
- Machine free time: 6 dims
- Processing times: 72 dims
- Predicted arrivals: 12 dims (REMOVED actual arrivals + rate)
- Global progress: 2 dims
Total: 116 dimensions (-12 dims = 9.4% reduction)
```

**Benefits:**
- Removed redundant actual arrival times (agent learns from predictions only)
- Removed noisy arrival rate feature
- Cleaner signal for policy learning

### ✅ 2. Improved Hyperparameters

**NEW settings:**
```python
n_steps=2048                        # 2× larger rollouts → better value estimates
batch_size=512                      # 2× larger batches → stable gradients
n_epochs=5                          # 2.5× more updates (proactive is complex!)
gamma=1.0                           # Undiscounted (makespan)
gae_lambda=0.95                     # GAE for advantage estimation
ent_coef=0.01                       # 10× higher → exploration of wait timing
learning_rate=linear_schedule(3e-4) # Decaying LR: 3e-4 → 3e-5 for stability
```

**Benefits:**
- Larger rollouts capture longer-term effects of wait actions
- Larger batches reduce gradient noise
- More epochs per rollout improve sample efficiency
- Higher entropy encourages exploring different wait timings
- Decaying learning rate prevents late-stage oscillations

### ✅ 3. Observation Normalization

**Added VecNormalize wrapper:**
```python
vec_env = VecNormalize(
    vec_env,
    norm_obs=True,        # Normalize obs to mean=0, std=1
    norm_reward=False,    # Don't normalize rewards (makespan is meaningful)
    clip_obs=10.0,        # Clip to [-10, 10]
    gamma=1.0
)
```

**Benefits:**
- Standardized observation scales → better gradient flow
- Improved neural network convergence
- More stable policy updates

### ✅ 4. Action Space Kept Simple

**Kept 2 wait actions:**
```python
wait_durations = [10.0, float('inf')]  # Short wait OR wait for next event
```

**Rationale:**
- Simpler action space → faster initial learning
- Two options cover key cases: (1) wait for machine to free, (2) wait for arrival
- Agent focuses on WHEN to wait, not HOW LONG

## Expected Improvements

### Training Stability
- ✅ **Smoother reward curves** (less variance)
- ✅ **Decreasing entropy** (converging policy)
- ✅ **Stable episode lengths** (consistent behavior)
- ✅ **Decreasing policy loss** (better convergence)

### Learning Speed
- ✅ **Faster initial learning** (cleaner observations)
- ✅ **Better sample efficiency** (more epochs, larger batches)
- ✅ **More stable late-stage training** (decaying LR)

### Final Performance
- ✅ **Better makespan** (learned optimal wait timing)
- ✅ **More deterministic policy** (low entropy at convergence)
- ✅ **Complete schedules** (no premature termination)

## Comparison: Before vs After

| Metric | BEFORE | AFTER (Expected) |
|--------|--------|------------------|
| Observation dims | 128 | 116 (-9.4%) |
| n_steps | 1024 | 2048 (+100%) |
| batch_size | 256 | 512 (+100%) |
| n_epochs | 2 | 5 (+150%) |
| ent_coef | 0.001 | 0.01 (+900%) |
| Learning rate | Fixed 3e-4 | Decaying 3e-4→3e-5 |
| Obs normalization | ❌ None | ✅ VecNormalize |
| Episode reward variance | HIGH | LOW |
| Entropy (late training) | -1.0 (high) | -2.0 to -3.0 (low) |
| Episode length variance | HIGH | LOW |
| Policy convergence | Poor | Good |

## Training Time Impact

**Estimated training time increase:**
- Larger rollouts (2048 vs 1024): +100% per rollout
- More epochs (5 vs 2): +150% per update
- Larger batches (512 vs 256): ~0% (same # updates)
- VecNormalize: ~+5% overhead

**Total: ~2.5× slower per timestep, BUT:**
- Better sample efficiency → need fewer timesteps
- More stable → less wasted exploration
- Expected: Similar or BETTER wall-clock time to convergence

## Next Steps

1. **Train with new hyperparameters** - Monitor metrics for stability
2. **Compare to reactive baseline** - Ensure proactive benefits maintained
3. **Tune if needed:**
   - If still unstable: increase batch_size to 1024
   - If too slow: reduce n_epochs to 4
   - If entropy too high: reduce ent_coef to 0.005

## Code Changes Summary

### Modified Files:
- `proactive_sche.py`:
  - Reduced observation space from 128 to 116 dims
  - Added VecNormalize wrapper
  - Improved hyperparameters (n_steps, batch_size, n_epochs, ent_coef)
  - Added learning rate schedule
  - Simplified `_get_observation()` method

### Key Functions Changed:
1. `__init__()` - Updated obs_size calculation
2. `_get_observation()` - Removed redundant features
3. `train_proactive_agent()` - New hyperparameters + VecNormalize

## Validation

To verify improvements, check:
1. ✅ Episode reward curve smoothness (less spikes)
2. ✅ Entropy decreasing over time (converging policy)
3. ✅ Episode length stabilizing (consistent behavior)
4. ✅ Policy loss decreasing (better gradient updates)
5. ✅ Final makespan ≤ Reactive RL (proactive should win!)
