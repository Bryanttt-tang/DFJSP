# Proactive Scheduling Implementation

## Overview
The `ProactiveDynamicFJSPEnv` environment implements **proactive scheduling** where the agent learns to predict job arrival times using Maximum Likelihood Estimation (MLE) and can schedule jobs before they actually arrive.

## Key Features

### 1. **Arrival Prediction System (`ArrivalPredictor` class)**

#### Maximum Likelihood Estimation (MLE)
- **Principle**: For a Poisson process, the MLE of the arrival rate λ is: `λ = 1 / mean(inter-arrival times)`
- **Learning**: The predictor improves as it observes more arrival events

#### Three Levels of Learning:
1. **Cross-Episode Learning**: Updates after each episode completes
   - Accumulates all observed inter-arrival times
   - Refines the global rate estimate
   
2. **Within-Episode Learning**: Updates during the current episode
   - As jobs arrive, observes actual arrival times
   - Immediately updates predictions for remaining jobs
   
3. **Initial Guess**: Starts with a default rate (0.05) and improves over time

### 2. **Proactive Action Masking**

The agent can schedule jobs in two scenarios:

```python
# Traditional: Job has arrived
if job_id in self.arrived_jobs:
    can_schedule = True

# PROACTIVE: Job predicted to arrive within prediction_window
elif job_id in self.predicted_arrival_times:
    predicted_time = self.predicted_arrival_times[job_id]
    if predicted_time <= self.event_time + self.prediction_window:
        can_schedule = True  # Allow proactive scheduling
```

**Prediction Window** (`prediction_window=10.0`): Controls how far ahead the agent can schedule. Larger window = more proactive but higher risk of misprediction.

### 3. **Enhanced Observation Space**

The agent observes:
- Standard features: job progress, machine availability, processing times
- **NEW - Predicted arrival times**: Normalized time until predicted arrival
- **NEW - Prediction confidence**: Based on number of observations (0-1 scale)
- **NEW - Estimated arrival rate**: Current MLE estimate of λ

### 4. **Reward Shaping for Proactive Decisions**

```python
# Standard reward: minimize makespan increase
reward = -(self.current_makespan - previous_makespan)

# Misprediction penalty: if scheduled before actual arrival
if job_id not in self.arrived_jobs:
    actual_arrival = self.job_arrival_times[job_id]
    if actual_arrival > start_time:
        misprediction_penalty = -10.0
        reward += misprediction_penalty
```

This encourages:
- ✅ Accurate predictions
- ✅ Conservative scheduling when uncertain
- ✅ Aggressive scheduling when confident

## Mathematical Foundation

### Poisson Process MLE

Given observed inter-arrival times: `τ₁, τ₂, ..., τₙ`

**MLE for rate parameter:**
```
λ̂ = n / Σᵢ τᵢ = 1 / mean(τ)
```

**Prediction of next arrival:**
```
E[next arrival | last = t] = t + 1/λ̂
```

### Online Learning Algorithm

```python
def update_with_new_observation(inter_arrival_time):
    observations.append(inter_arrival_time)
    λ̂ = 1 / mean(observations)
    
    # Predict next k arrivals
    for i in range(k):
        next_arrival[i] = last_arrival + (i+1) * (1/λ̂)
```

## Usage Example

```python
from proactive_sche import ProactiveDynamicFJSPEnv, mask_fn

# Create proactive environment
env = ProactiveDynamicFJSPEnv(
    jobs_data=ENHANCED_JOBS_DATA,
    machine_list=MACHINE_LIST,
    initial_jobs=[0, 1, 2],
    arrival_rate=0.08,  # True rate (hidden from agent)
    prediction_window=10.0,  # Allow scheduling 10 time units ahead
    max_time_horizon=100,
    reward_mode="makespan_increment"
)

# Wrap with action masker
env = ActionMasker(env, mask_fn)

# Train with MaskablePPO
model = MaskablePPO("MlpPolicy", env, ...)
model.learn(total_timesteps=100000)
```

## Key Differences from Reactive Scheduling

| Feature | Reactive (`PoissonDynamicFJSPEnv`) | Proactive (`ProactiveDynamicFJSPEnv`) |
|---------|-----------------------------------|--------------------------------------|
| **Information** | Only knows about arrived jobs | Predicts future arrivals |
| **Scheduling** | Can only schedule arrived jobs | Can schedule predicted jobs |
| **Learning** | Learns from arrivals in episode | Learns across episodes (MLE) |
| **Action Mask** | Binary (arrived/not arrived) | Continuous (predicted arrival time) |
| **Observation** | Current state only | Includes predictions & confidence |
| **Risk** | Zero (no mispredictions) | Penalty for wrong predictions |

## Training Strategy

### Phase 1: Exploration (Episodes 1-100)
- Predictor has poor estimates
- Agent learns to be conservative
- Mostly waits for actual arrivals

### Phase 2: Refinement (Episodes 100-1000)
- Predictor improves with data
- Agent starts proactive scheduling
- Balances risk vs. reward

### Phase 3: Exploitation (Episodes 1000+)
- Accurate predictions
- Aggressive proactive scheduling
- Near-optimal performance

## Expected Performance

**Hypothesis**: `Proactive RL ≤ Dynamic RL ≤ Static RL`

- **Best Case**: Perfect predictions → schedules optimally ahead of time
- **Worst Case**: Poor predictions → penalties offset gains, similar to Dynamic RL
- **Realistic**: 5-15% improvement over pure reactive Dynamic RL after sufficient training

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prediction_window` | 10.0 | Time horizon for proactive scheduling |
| `initial_rate_guess` | 0.05 | Starting estimate of λ |
| `misprediction_penalty` | -10.0 | Reward penalty for wrong predictions |

## Future Enhancements

1. **Bayesian Learning**: Use posterior distributions instead of point estimates
2. **Job-Specific Predictions**: Different rates for different job types
3. **Confidence Intervals**: Schedule only within 95% confidence bounds
4. **Adaptive Window**: Adjust `prediction_window` based on confidence
5. **Multi-Step Prediction**: Predict multiple future arrivals simultaneously

## Implementation Notes

- The true `arrival_rate` is hidden from the agent (only used to generate arrivals)
- Agent must learn purely from observations
- MLE is unbiased and consistent for large samples
- Early episodes have high uncertainty → conservative behavior expected
- Can visualize learning by tracking `estimated_rate` over episodes

