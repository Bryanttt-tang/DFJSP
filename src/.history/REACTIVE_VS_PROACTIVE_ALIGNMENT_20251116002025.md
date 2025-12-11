# Reactive vs Proactive RL Alignment

## Overview
This document describes the changes made to align Reactive and Proactive RL environments and training for fair comparison. The goal is to show that **the ONLY difference is the arrival predictor in observations**, not in action space, hyperparameters, or reward shaping.

## Key Insight
**The `use_predictor_for_wait` parameter was REDUNDANT** because:
1. Both wait execution functions (`_execute_wait_action_flexible` and `_execute_wait_action_with_predictor_guidance`) were IDENTICAL
2. The predictor already guides the agent through observations (predicted arrival times)
3. No reward shaping is needed - predictor learns from corrections, not penalties

## Changes Made

### 1. Removed Redundant `use_predictor_for_wait` Parameter

**Before:**
```python
def __init__(self, ..., use_predictor_for_wait=True, max_wait_time=10.0, ...):
    self.use_predictor_for_wait = use_predictor_for_wait
    self.max_wait_time = max_wait_time
    
    if self.use_predictor_for_wait:
        reward, done = self._execute_wait_action_with_predictor_guidance(wait_duration)
    else:
        reward, done = self._execute_wait_action_flexible(wait_duration)
```

**After:**
```python
def __init__(self, ..., predictor_mode='mle', ...):
    # Removed use_predictor_for_wait and max_wait_time
    
    reward, done = self._execute_wait_action(wait_duration)
```

**Rationale:**
- The predictor guides agent through observations, not reward shaping
- Simpler, clearer code
- Emphasizes that proactive RL = reactive RL + predictor observations

### 2. Merged Two Identical Wait Action Functions

**Before:**
- `_execute_wait_action_flexible()` - 60 lines
- `_execute_wait_action_with_predictor_guidance()` - 60 lines (IDENTICAL!)

**After:**
- `_execute_wait_action()` - Single unified function

**Key Points:**
```python
def _execute_wait_action(self, wait_duration):
    """
    Execute wait action with SIMPLE makespan_increment reward.
    
    The predictor guides the agent through OBSERVATIONS (predicted arrivals),
    NOT through reward shaping. This keeps rewards consistent between
    reactive and proactive RL.
    """
    # ... identical logic for both reactive and proactive ...
    reward = -(self.current_makespan - previous_makespan)  # IDENTICAL
    return reward, done
```

### 3. Aligned Action Spaces

**Reactive RL Before:**
- 1 wait action: Wait to next event
- Action space: `Discrete(num_jobs * num_machines + 1)`

**Reactive RL After (IDENTICAL to Proactive):**
- 2 wait actions: [10 units, next event]
- Action space: `Discrete(num_jobs * num_machines + 2)`
- `wait_durations = [10.0, float('inf')]`

**Code Changes:**
```python
# Reactive RL - Now IDENTICAL to proactive RL
self.wait_durations = [10.0, float('inf')]  # 10 units or next event
num_wait_actions = len(self.wait_durations)
self.action_space = spaces.Discrete(num_scheduling_actions + num_wait_actions)
self.wait_action_start = num_scheduling_actions
```

### 4. Aligned Hyperparameters

**Reactive RL Before:**
```python
learning_rate=3e-4,          # Fixed
n_steps=1024,
batch_size=256,
n_epochs=2,                  # Low
ent_coef=0.001,              # Low exploration
```

**Reactive RL After (IDENTICAL to Proactive):**
```python
learning_rate=linear_schedule(3e-4),  # Decaying (3e-4 → 3e-5)
n_steps=2048,                         # +100% (better value estimates)
batch_size=512,                       # +100% (more stable gradients)
n_epochs=5,                           # +150% (more updates per rollout)
ent_coef=0.01,                        # +900% (higher exploration)
```

**Rationale:**
- Both methods now have IDENTICAL model capacity
- Both use same learning rate schedule for stability
- Both have same exploration coefficient
- Fair comparison: differences come from predictor, not hyperparameters

### 5. Updated Action Masking

**Reactive RL Action Masking - Now IDENTICAL to Proactive:**
```python
def action_masks(self):
    # ... schedule actions for arrived jobs ...
    
    # Wait actions: Enable all wait options (IDENTICAL to proactive RL)
    has_unarrived_jobs = len(self.arrived_jobs) < len(self.job_ids)
    has_schedulable_work = np.any(mask[:self.wait_action_start])
    
    if has_unarrived_jobs or has_schedulable_work:
        for wait_idx in range(len(self.wait_durations)):
            action_idx = self.wait_action_start + wait_idx
            mask[action_idx] = True
    
    return mask
```

### 6. Updated Step Functions

**Both environments now handle multiple wait actions identically:**

```python
# Reactive RL step() - Now IDENTICAL wait logic to proactive
if job_idx is None:  # Wait action
    wait_idx = action - self.wait_action_start
    wait_duration = self.wait_durations[wait_idx]
    
    if wait_duration == float('inf'):
        new_event_time, new_arrivals = self._advance_to_next_arrival()
    else:
        target_time = self.event_time + wait_duration
        next_event_time = self._get_next_event_time()
        target_time = min(target_time, next_event_time)
        if target_time > self.event_time:
            new_arrivals = self._update_event_time_and_arrivals(target_time)
    
    self.current_makespan = max(self.current_makespan, self.event_time)
    reward = -(self.current_makespan - previous_makespan)  # IDENTICAL
```

## Summary of Differences: Reactive vs Proactive RL

| Feature | Reactive RL | Proactive RL | Difference? |
|---------|-------------|--------------|-------------|
| **Action Space** | job×machine + 2 waits | job×machine + 2 waits | ❌ IDENTICAL |
| **Wait Durations** | [10, ∞] | [10, ∞] | ❌ IDENTICAL |
| **Reward** | -Δmakespan | -Δmakespan | ❌ IDENTICAL |
| **Hyperparameters** | n_steps=2048, batch=512, epochs=5, ent=0.01, LR decay | n_steps=2048, batch=512, epochs=5, ent=0.01, LR decay | ❌ IDENTICAL |
| **Observations** | Job ready time, progress, machine free time, proc times, arrivals | **SAME** + **predicted arrivals** | ✅ ONLY DIFFERENCE |
| **Predictor** | None | ArrivalPredictor (MLE/MAP) | ✅ ONLY DIFFERENCE |

## Expected Experimental Results

With this alignment, we can now test the hypothesis:

**Hypothesis:** Proactive RL ≥ Reactive RL ≥ Static RL

**Rationale:**
1. **Proactive = Reactive + Predictor observations**
   - If predictor is accurate → proactive can plan better → lower makespan
   - If predictor is poor → proactive = reactive (ignores predictions) → same makespan
   - Best case: Proactive > Reactive (good predictions help)
   - Worst case: Proactive = Reactive (bad predictions ignored)
   - **Never: Proactive < Reactive** (additional info can't hurt with proper training)

2. **Fair Comparison:**
   - Same model capacity (network architecture)
   - Same hyperparameters (learning rate, batch size, epochs, entropy)
   - Same action space (2 wait options)
   - Same reward function (negative makespan increment)
   - **Only difference:** Proactive has predicted arrival times in observations

3. **Key Metrics:**
   - **Makespan:** Primary performance metric
   - **Training stability:** Both should converge smoothly with aligned hyperparameters
   - **Predictor learning:** Track estimated arrival rate convergence to true rate
   - **Wait behavior:** Compare wait action frequency and timing

## Validation Checklist

Before running experiments:
- ✅ Reactive RL has 2 wait actions [10, ∞]
- ✅ Proactive RL has 2 wait actions [10, ∞]
- ✅ Both use `linear_schedule` learning rate decay
- ✅ Both use n_steps=2048, batch_size=512, n_epochs=5, ent_coef=0.01
- ✅ Both use IDENTICAL reward: -Δmakespan
- ✅ Proactive has `use_predictor_for_wait` parameter removed
- ✅ Proactive uses single `_execute_wait_action()` function
- ✅ Observation space difference: Proactive adds predicted arrival times

## Training Commands

```python
# Train Reactive RL (baseline)
reactive_model = train_dynamic_agent(
    ENHANCED_JOBS_DATA, MACHINE_LIST,
    initial_jobs=5,
    arrival_rate=0.08,
    total_timesteps=500000,
    learning_rate=3e-4
)

# Train Proactive RL (with predictor)
proactive_model = train_proactive_agent(
    ENHANCED_JOBS_DATA, MACHINE_LIST,
    initial_jobs=5,
    arrival_rate=0.08,
    total_timesteps=500000,
    learning_rate=3e-4,
    predictor_mode='mle'  # or 'map'
)
```

## Files Modified

1. **ProactiveDynamicFJSPEnv class (lines ~1200-1700):**
   - Removed `use_predictor_for_wait` parameter
   - Merged wait action functions
   - Updated docstring

2. **PoissonDynamicFJSPEnv class (lines ~530-1180):**
   - Added multiple wait actions (2 options)
   - Updated action masking
   - Updated step function
   - Aligned with proactive wait logic

3. **train_dynamic_agent() function (lines ~2841-2906):**
   - Updated hyperparameters to match proactive
   - Added learning rate schedule
   - Increased n_steps, batch_size, n_epochs, ent_coef

4. **train_proactive_agent() function (lines ~2909-3100):**
   - Updated docstring (removed misleading "more complex" justification)
   - Clarified that hyperparameters are IDENTICAL to reactive

## Conclusion

The code now clearly demonstrates that:
1. **Proactive RL = Reactive RL + Arrival Predictor Observations**
2. No reward shaping, no different hyperparameters, no action space differences
3. Fair, controlled comparison to test if arrival prediction helps
4. Simpler, cleaner code that's easier to understand and maintain

The predictor guides the agent through **observations**, not through **rewards** or **action constraints**. This is the correct and principled approach to proactive scheduling with RL.
