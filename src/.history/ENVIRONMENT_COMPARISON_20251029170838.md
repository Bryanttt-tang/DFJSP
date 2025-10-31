# Deep Comparison of Three FJSP Scheduling Environments

## Overview

This document provides a comprehensive comparison of three reinforcement learning environments for Flexible Job Shop Scheduling Problem (FJSP) with dynamic job arrivals:

1. **PoissonDynamicFJSPEnv** (Reactive RL)
2. **ProactiveDynamicFJSPEnv** (Proactive Scheduling) 
3. **PerfectKnowledgeFJSPEnv** (Perfect Knowledge RL)

---

## 1. Core Concept & Information Level

### PoissonDynamicFJSPEnv (Reactive RL)
- **Information Level**: **NO FUTURE KNOWLEDGE**
- **Arrival Knowledge**: Only knows arrival distribution (Poisson rate λ)
- **Behavior**: **Purely reactive** - can only schedule jobs that have **already arrived**
- **Prediction**: **None** - reacts to arrivals as they happen
- **Use Case**: Baseline for comparing reactive vs proactive strategies

### ProactiveDynamicFJSPEnv (Proactive Scheduling)
- **Information Level**: **LEARNED PREDICTIONS**
- **Arrival Knowledge**: Learns to predict arrivals from historical data across episodes
- **Behavior**: **Proactive** - can schedule jobs predicted to arrive within `prediction_window`
- **Prediction**: **Yes** - uses `ArrivalPredictor` with MLE to learn arrival patterns
- **Use Case**: Tests if learning arrival patterns improves scheduling

### PerfectKnowledgeFJSPEnv (Perfect Knowledge RL)
- **Information Level**: **COMPLETE ORACLE**
- **Arrival Knowledge**: Knows **exact** arrival times of all jobs upfront
- **Behavior**: **Fully informed** - can plan entire schedule optimally from start
- **Prediction**: **N/A** - has ground truth, no prediction needed
- **Use Case**: Upper bound / optimal benchmark for RL performance

---

## 2. Observation Space Comparison

### 2.1 PoissonDynamicFJSPEnv (Reactive RL)

**Observation Components** (when `cheat=False`):
```python
obs_size = (
    num_jobs +                         # Ready job indicators (1 if arrived & has next op)
    num_jobs +                         # Job progress (completed_ops / total_ops)
    len(machines) +                    # Machine next_free times (normalized)
    num_jobs * len(machines) +         # Processing times for READY operations only
    num_jobs +                         # Normalized arrival times (1.0 if not arrived)
    2                                  # Arrival progress & makespan progress
)
```

**Key Features**:
- ✅ Shows which jobs have **already arrived**
- ✅ Processing times only for **arrived jobs**
- ❌ **NO future arrival information** (only shows 1.0 for unarrived jobs)
- ❌ **NO prediction capabilities**
- ❌ **NO prediction window**

**Arrival Information**:
```python
# For arrived jobs: normalized arrival time
normalized_arrival_time = min(1.0, arrival_time / max_time_horizon)

# For unarrived jobs: constant 1.0 (no information)
obs.append(1.0)  # Cannot distinguish between different future arrivals!
```

---

### 2.2 ProactiveDynamicFJSPEnv (Proactive Scheduling)

**Observation Components**:
```python
obs_size = (
    len(job_ids) +                      # Ready job indicators (arrived OR predicted within window)
    len(machines) +                     # Machine idle status
    len(job_ids) * len(machines) +      # Processing times for next ops
    len(job_ids) +                      # Job progress
    len(job_ids) +                      # ⭐ NEW: Predicted arrival times (normalized)
    len(job_ids) +                      # ⭐ NEW: Arrival status (arrived/not arrived)
    3                                   # ⭐ NEW: Arrival progress, makespan progress, estimated rate
)
```

**Key Features**:
- ✅ Shows **predicted arrival times** for unarrived jobs
- ✅ Processing times for **both arrived AND predicted jobs** within window
- ✅ **Estimated arrival rate** from MLE predictor
- ✅ Distinguishes between arrived (1.0), predicted (0.5), and far future (0.0) jobs
- ✅ **Prediction confidence** (commented out but available)

**Arrival Prediction Information**:
```python
# 1. Ready job indicators with prediction status
if job_id in arrived_jobs:
    ready_jobs.append(1.0)          # Arrived: certainty
elif job_id in predicted_arrival_times:
    if pred_time <= event_time + prediction_window:
        ready_jobs.append(0.5)      # Predicted within window: partial certainty
    else:
        ready_jobs.append(0.0)      # Predicted too far
else:
    ready_jobs.append(0.0)          # Unknown

# 2. Predicted arrival times (UNIQUE TO PROACTIVE)
time_until = max(0, pred_time - event_time)
normalized = min(1.0, time_until / (prediction_window * 2))

# 3. Estimated arrival rate from MLE (UNIQUE TO PROACTIVE)
estimated_rate = arrival_predictor.get_stats()['estimated_rate']
normalized_rate = min(1.0, estimated_rate / 0.2)
```

**Prediction Window Advantage**:
- Can schedule jobs up to `prediction_window` time units in advance
- Example: `prediction_window=10.0` means can schedule jobs predicted to arrive in next 10 time units
- Allows **proactive resource allocation** and **load balancing**

---

### 2.3 PerfectKnowledgeFJSPEnv (Perfect Knowledge)

**Observation Components**:
```python
obs_size = (
    num_jobs +                         # Ready job indicators (all jobs always visible)
    len(machines) +                    # Machine idle status
    num_jobs * len(machines) +         # Processing times for ALL ready operations
    num_jobs +                         # Job progress
    num_jobs                           # ⭐ PERFECT: Exact future arrival times
)
```

**Key Features**:
- ✅ **Exact arrival times** for ALL jobs (not predictions)
- ✅ All jobs visible from start (can plan entire schedule)
- ✅ Processing times for all jobs regardless of arrival status
- ❌ Not realistic - serves as **theoretical upper bound**

**Perfect Arrival Information**:
```python
# Exact arrival times (ORACLE KNOWLEDGE)
for job_id in job_ids:
    arrival_time = job_arrival_times[job_id]
    # Direct normalized arrival time (NO uncertainty)
    obs.append(min(1.0, arrival_time / max_time_horizon))
```

---

## 3. Action Masking Comparison

### 3.1 PoissonDynamicFJSPEnv (Reactive RL)

**Action Masking Logic**:
```python
def action_masks(self):
    mask = np.zeros(action_space.n, dtype=bool)
    
    for job_id in job_ids:
        # ❌ ONLY if job has ARRIVED
        if job_id in arrived_jobs:
            next_op_idx = next_operation[job_id]
            if next_op_idx < len(jobs[job_id]):
                operation = jobs[job_id][next_op_idx]
                for machine in operation['proc_times'].keys():
                    # Allow scheduling on compatible machines
                    mask[action_idx] = True
    
    # Always allow WAIT action
    mask[WAIT_ACTION] = True
    
    return mask
```

**Characteristics**:
- ⛔ **Cannot schedule jobs that haven't arrived** (strict constraint)
- ✅ Can schedule arrived jobs on compatible machines
- ✅ Can WAIT to advance time to next event
- **Restrictive**: May be forced to wait even when machines idle

---

### 3.2 ProactiveDynamicFJSPEnv (Proactive Scheduling)

**Action Masking Logic**:
```python
def action_masks(self):
    mask = np.zeros(action_space.n, dtype=bool)
    
    for job_id in job_ids:
        op_idx = job_progress[job_id]
        if op_idx >= len(jobs[job_id]):
            continue
            
        operation = jobs[job_id][op_idx]
        available_machines = list(operation['proc_times'].keys())
        
        # ✅ TRADITIONAL: Job has arrived
        if job_id in arrived_jobs:
            for machine in available_machines:
                mask[action_idx] = True
        
        # ⭐ PROACTIVE: Job predicted to arrive within prediction_window
        elif job_id in predicted_arrival_times:
            predicted_time = predicted_arrival_times[job_id]
            # Allow scheduling if predicted to arrive within window
            if predicted_time <= event_time + prediction_window:
                for machine in available_machines:
                    mask[action_idx] = True
    
    # Always allow WAIT
    mask[wait_action_idx] = True
    
    return mask
```

**Characteristics**:
- ✅ Can schedule **arrived jobs** (like Reactive RL)
- ⭐ **UNIQUE**: Can schedule **predicted jobs within prediction window**
- ✅ More flexible action space than Reactive RL
- ⚠️ Risk: Mispredictions may cause suboptimal schedules (but predictor learns from errors)

**Prediction Window Impact**:
```python
# Example with prediction_window = 10.0
# Current event_time = 5.0
# Job predicted to arrive at t=12.0

if 12.0 <= 5.0 + 10.0:  # 12.0 <= 15.0
    # ✅ Can schedule this job proactively
    mask[action_idx] = True
else:
    # ❌ Too far in future, cannot schedule yet
    mask[action_idx] = False
```

---

### 3.3 PerfectKnowledgeFJSPEnv (Perfect Knowledge)

**Action Masking Logic**:
```python
def action_masks(self):
    mask = np.full(action_space.n, False, dtype=bool)
    
    if operations_scheduled >= total_operations:
        return mask  # All done
    
    # ⭐ ALL JOBS AVAILABLE (perfect knowledge)
    for job_idx, job_id in enumerate(job_ids):
        next_op_idx = next_operation[job_id]
        if next_op_idx >= len(jobs[job_id]):
            continue
            
        # Simply check machine compatibility (no arrival constraint!)
        for machine_idx, machine in enumerate(machines):
            if machine in jobs[job_id][next_op_idx]['proc_times']:
                action = job_idx * len(machines) + machine_idx
                mask[action] = True
    
    return mask
```

**Characteristics**:
- ✅ **No arrival constraints** - all jobs schedulable from start
- ✅ Only checks precedence (operation order within job) and machine compatibility
- ❌ **No WAIT action** (not needed since all jobs visible)
- 🎯 **Optimal action space** - maximum flexibility for learning

---

## 4. Future Job Arrival Prediction

### 4.1 PoissonDynamicFJSPEnv (Reactive RL)

**Prediction Capability**: ❌ **NONE**

**Approach**:
- No prediction mechanism
- Only knows jobs have arrived via `arrived_jobs` set
- Updates arrivals based on Poisson process during episode

```python
def _update_event_time_and_arrivals(self, new_event_time):
    """Update arrivals when event_time advances."""
    num_new_arrivals = 0
    
    for job_id, arrival_time in self.job_arrival_times.items():
        if job_id not in self.arrived_jobs and arrival_time <= new_event_time:
            self.arrived_jobs.add(job_id)  # Job has now arrived
            num_new_arrivals += 1
    
    self.event_time = new_event_time
    return num_new_arrivals
```

**Limitations**:
- Cannot anticipate future arrivals
- May leave machines idle waiting for unknown arrivals
- Suboptimal resource utilization

---

### 4.2 ProactiveDynamicFJSPEnv (Proactive Scheduling)

**Prediction Capability**: ✅ **YES - Using ArrivalPredictor with MLE**

**Prediction Components**:

#### 4.2.1 ArrivalPredictor Class
```python
class ArrivalPredictor:
    """
    Predicts job arrival times using Maximum Likelihood Estimation (MLE).
    
    Key Features:
    1. Cross-Episode Learning: Uses ALL historical inter-arrival times
    2. Within-Episode Learning: Updates predictions as jobs arrive
    3. Adaptive Learning: Combines historical data with current observations
    4. Misprediction Correction: Adjusts estimates based on errors
    """
    
    def __init__(self, initial_rate_guess=0.05):
        self.global_inter_arrivals = []        # Cross-episode history (last 100)
        self.current_episode_inter_arrivals = []
        self.current_estimated_rate = initial_rate_guess
        self.prediction_errors = []
```

#### 4.2.2 Prediction Workflow

**Step 1: Episode Initialization**
```python
def reset_episode(self):
    """Reset for new episode while keeping global history."""
    self.current_episode_inter_arrivals = []
    self.last_observed_arrival = None
```

**Step 2: Observe Arrivals Within Episode**
```python
def observe_arrival(self, arrival_time):
    """Update predictor when a job arrives."""
    if self.last_observed_arrival is not None:
        inter_arrival = arrival_time - self.last_observed_arrival
        self.current_episode_inter_arrivals.append(inter_arrival)
        
        # Update MLE estimate with new observation
        self._update_mle_estimate()
    
    self.last_observed_arrival = arrival_time
```

**Step 3: Predict Future Arrivals**
```python
def predict_next_arrivals(self, current_time, num_jobs_to_predict, last_known_arrival=None):
    """
    Predict arrival times for future jobs using current MLE estimate.
    
    Returns:
        List of predicted arrival times
    """
    predictions = []
    
    # Use last known arrival or current time as anchor
    anchor_time = last_known_arrival if last_known_arrival is not None else current_time
    
    # Generate predictions using estimated rate
    for i in range(num_jobs_to_predict):
        # Expected inter-arrival time = 1 / rate
        expected_inter_arrival = 1.0 / self.current_estimated_rate
        predicted_arrival = anchor_time + expected_inter_arrival * (i + 1)
        predictions.append(predicted_arrival)
    
    return predictions
```

**Step 4: Correct Mispredictions**
```python
def correct_prediction(self, job_id, predicted_time, actual_time):
    """
    Adjust predictor based on prediction error.
    This enables LEARNING from mistakes!
    """
    error = abs(actual_time - predicted_time)
    self.prediction_errors.append(error)
    
    # Could adjust rate based on error (future enhancement)
```

**Step 5: Cross-Episode Learning**
```python
def finalize_episode(self, all_arrival_times):
    """
    Update global knowledge with this episode's arrival data.
    Enables TRANSFER LEARNING across episodes!
    """
    # Extract inter-arrival times
    sorted_arrivals = sorted([t for t in all_arrival_times.values() if t < float('inf')])
    
    for i in range(1, len(sorted_arrivals)):
        inter_arrival = sorted_arrivals[i] - sorted_arrivals[i-1]
        self.global_inter_arrivals.append(inter_arrival)
    
    # Keep only recent history (last 100 episodes worth)
    if len(self.global_inter_arrivals) > 1000:
        self.global_inter_arrivals = self.global_inter_arrivals[-1000:]
    
    # Update global MLE estimate
    self._update_global_mle()
```

#### 4.2.3 Prediction Usage in Environment

**Update Predictions Each Step**:
```python
def _update_predictions(self):
    """Update predictions for unarrived jobs."""
    unarrived_jobs = [j for j in job_ids if j not in arrived_jobs]
    
    if len(unarrived_jobs) == 0:
        return
    
    # Find last known arrival for anchoring
    last_known_arrival = None
    if len(arrived_jobs) > 0:
        last_known_arrival = max([job_arrival_times[j] for j in arrived_jobs])
    
    # Predict using MLE predictor
    predictions = arrival_predictor.predict_next_arrivals(
        current_time=event_time,
        num_jobs_to_predict=len(unarrived_jobs),
        last_known_arrival=last_known_arrival
    )
    
    # Map predictions to jobs
    for job_id, predicted_time in zip(sorted(unarrived_jobs), predictions):
        predicted_arrival_times[job_id] = predicted_time
```

**Proactive Scheduling with Predictions**:
```python
def step(self, action):
    # When scheduling a predicted (not yet arrived) job:
    if job_id not in arrived_jobs:
        # Use ACTUAL arrival time (from ground truth)
        actual_arrival = job_arrival_times[job_id]
        
        # Check for misprediction
        if job_id in predicted_arrival_times:
            predicted_arrival = predicted_arrival_times[job_id]
            
            # Correct predictor (enables learning!)
            arrival_predictor.correct_prediction(
                job_id, predicted_arrival, actual_arrival
            )
        
        # Calculate start time using actual arrival
        start_time = max(machine_ready, actual_arrival, event_time)
```

**Benefits**:
- ✅ **Cross-episode learning**: Gets better over many episodes
- ✅ **Adaptive**: Updates predictions within episode as jobs arrive
- ✅ **Self-correcting**: Learns from mispredictions
- ✅ **Confidence awareness**: Can assess prediction quality

---

### 4.3 PerfectKnowledgeFJSPEnv (Perfect Knowledge)

**Prediction Capability**: ❌ **NOT NEEDED (Has Ground Truth)**

**Approach**:
- Receives exact arrival times in constructor: `arrival_times` dict
- No prediction required since all information known upfront

```python
def __init__(self, jobs_data, machine_list, arrival_times, ...):
    self.job_arrival_times = arrival_times.copy()  # Perfect knowledge
    
    # All jobs visible from start
    self.arrived_jobs = set(self.job_ids)  # Can plan for all jobs
```

**Scheduling with Perfect Knowledge**:
```python
def step(self, action):
    job_id = job_ids[job_idx]
    
    # Use exact arrival time for scheduling
    job_ready_time = job_arrival_times[job_id]  # No prediction needed!
    
    # Calculate optimal start time
    start_time = max(machine_available_time, job_ready_time)
```

**Advantages**:
- 🎯 **Optimal planning**: Can create globally optimal schedule
- 🎯 **No uncertainty**: No prediction errors
- ⚠️ **Unrealistic**: Not achievable in real-world scenarios

---

## 5. Prediction Window Analysis

### Comparison Table

| Feature | Reactive RL | Proactive RL | Perfect Knowledge |
|---------|------------|--------------|-------------------|
| **Has Prediction Window** | ❌ No | ✅ **Yes** (`prediction_window` param) | ❌ N/A (all visible) |
| **Can Schedule Unarrived Jobs** | ❌ No | ✅ **Yes** (within window) | ✅ Yes (all jobs) |
| **Prediction Method** | None | MLE with cross-episode learning | Ground truth |
| **Learning Component** | Standard RL | RL + Arrival Predictor | Standard RL |
| **Uncertainty** | High (no foresight) | Medium (learned predictions) | None (oracle) |

### Prediction Window Impact (Proactive RL)

**Example Scenario**:
```python
prediction_window = 10.0  # Can schedule jobs predicted in next 10 time units
event_time = 5.0
predicted_arrival_times = {
    3: 8.0,   # Predicted to arrive at t=8
    4: 12.0,  # Predicted to arrive at t=12
    5: 20.0   # Predicted to arrive at t=20
}

# Check which jobs can be scheduled proactively:
# Job 3: 8.0 <= 5.0 + 10.0 (15.0) ✅ Can schedule
# Job 4: 12.0 <= 5.0 + 10.0 (15.0) ✅ Can schedule
# Job 5: 20.0 <= 5.0 + 10.0 (15.0) ❌ Cannot schedule (too far)
```

**Prediction Window Trade-offs**:

**Small Window** (e.g., 5.0):
- ✅ More conservative (lower misprediction risk)
- ❌ Less proactive (limited planning horizon)
- ❌ Closer to Reactive RL behavior

**Large Window** (e.g., 20.0):
- ✅ More proactive (can plan far ahead)
- ✅ Better resource utilization
- ⚠️ Higher misprediction risk (longer horizon = more uncertainty)
- ⚠️ If predictions wrong, may waste resources

**Optimal Window** (e.g., 10.0-15.0):
- ⚖️ Balances proactivity with prediction accuracy
- ⚖️ Adapts as predictor improves over episodes

---

## 6. Key Differences Summary

### Information Hierarchy

```
Perfect Knowledge RL
    ↑ (knows everything)
    |
Proactive RL (prediction_window + MLE predictor)
    ↑ (learns patterns)
    |
Reactive RL (no foresight)
    ↑ (only current state)
```

### Observable Information

| Information Type | Reactive RL | Proactive RL | Perfect RL |
|-----------------|-------------|--------------|------------|
| Current arrivals | ✅ Yes | ✅ Yes | ✅ Yes |
| Future arrival predictions | ❌ No | ✅ **Yes (MLE)** | ✅ Yes (exact) |
| Prediction confidence | ❌ N/A | ✅ **Yes** | ✅ N/A (certain) |
| Historical arrival data | ❌ No | ✅ **Yes (cross-episode)** | ✅ Yes (given) |
| Arrival rate estimate | ❌ No | ✅ **Yes (learned)** | ✅ Yes (given) |

### Action Space Flexibility

| Capability | Reactive RL | Proactive RL | Perfect RL |
|-----------|-------------|--------------|------------|
| Schedule arrived jobs | ✅ Yes | ✅ Yes | ✅ Yes |
| Schedule predicted jobs | ❌ No | ✅ **Yes (within window)** | ✅ Yes (all) |
| WAIT action | ✅ Yes | ✅ Yes | ❌ No (not needed) |
| Plan ahead | ❌ No | ✅ **Limited (window)** | ✅ Full |

### Learning Components

| Component | Reactive RL | Proactive RL | Perfect RL |
|-----------|-------------|--------------|------------|
| Policy Network | ✅ MaskablePPO | ✅ MaskablePPO | ✅ MaskablePPO |
| Value Network | ✅ Yes | ✅ Yes | ✅ Yes |
| **Arrival Predictor** | ❌ None | ✅ **ArrivalPredictor (MLE)** | ❌ None |
| Cross-episode learning | ❌ No | ✅ **Yes (predictor)** | ❌ No |
| Prediction error feedback | ❌ N/A | ✅ **Yes (correction)** | ❌ N/A |

---

## 7. Expected Performance Hierarchy

Based on information available:

```
Perfect Knowledge RL (best possible)
    ≥
Proactive RL (learned predictions)
    ≥
Reactive RL (no foresight)
    ≥
Static RL (assumes all at t=0)
```

**Why This Hierarchy?**

1. **Perfect RL**: Has complete information → optimal planning possible
2. **Proactive RL**: Has learned predictions → better resource allocation than reactive
3. **Reactive RL**: Can react to arrivals → better than ignoring dynamics
4. **Static RL**: Ignores arrivals → worst for dynamic scenarios

**Proactive RL Advantages Over Reactive RL**:
- ✅ Can prepare resources for predicted arrivals
- ✅ Reduces idle time through proactive scheduling
- ✅ Better load balancing across machines
- ✅ Improves over episodes as predictor learns
- ⚠️ Risk: Mispredictions may cause suboptimal schedules (but learns from errors)

---

## 8. Code Implementation Highlights

### 8.1 Unique to Proactive RL

**Arrival Predictor Initialization**:
```python
# In __init__
self.arrival_predictor = ArrivalPredictor(initial_rate_guess=0.05)
self.prediction_window = prediction_window  # UNIQUE parameter
```

**Prediction Updates**:
```python
# Called every step
def _update_predictions(self):
    unarrived_jobs = [j for j in job_ids if j not in arrived_jobs]
    predictions = arrival_predictor.predict_next_arrivals(...)
    for job_id, pred_time in zip(sorted(unarrived_jobs), predictions):
        predicted_arrival_times[job_id] = pred_time
```

**Misprediction Handling**:
```python
# In step() when scheduling predicted job
if job_id not in arrived_jobs:
    actual_arrival = job_arrival_times[job_id]
    if job_id in predicted_arrival_times:
        arrival_predictor.correct_prediction(
            job_id, predicted_arrival, actual_arrival
        )
```

**Cross-Episode Learning**:
```python
# Called at episode end
def finalize_episode(self):
    arrival_predictor.finalize_episode(job_arrival_times)
```

### 8.2 Unique to Perfect Knowledge RL

**All Jobs Visible**:
```python
# In reset()
self.arrived_jobs = set(self.job_ids)  # All jobs available from start
```

**No WAIT Action**:
```python
# Action space without WAIT
self.action_space = spaces.Discrete(num_jobs * len(machines))
```

---

## 9. Conclusion

### Summary of Key Distinctions

**PoissonDynamicFJSPEnv (Reactive RL)**:
- Purely reactive to arrivals
- No prediction capability
- Observation shows only arrived jobs
- Action mask restricts to arrived jobs only

**ProactiveDynamicFJSPEnv (Proactive Scheduling)** ⭐:
- **UNIQUE**: Has `ArrivalPredictor` with MLE
- **UNIQUE**: Has `prediction_window` parameter
- **UNIQUE**: Observation includes predicted arrivals & estimated rate
- **UNIQUE**: Action mask allows scheduling predicted jobs within window
- **UNIQUE**: Cross-episode learning for arrival patterns
- **UNIQUE**: Self-correcting through prediction error feedback

**PerfectKnowledgeFJSPEnv (Perfect Knowledge)**:
- Oracle knowledge of all arrivals
- No prediction needed (has ground truth)
- All jobs schedulable from start
- Serves as theoretical upper bound

### Research Implications

The **ProactiveDynamicFJSPEnv** is the most sophisticated environment because:

1. **Bridges reactive and perfect knowledge**: Learns to predict like Perfect RL but maintains realism
2. **Addresses real-world uncertainty**: Predictions have errors but improve over time
3. **Multi-level learning**: Combines RL policy learning with statistical arrival prediction
4. **Adaptive behavior**: Adjusts predictions based on observations and corrections
5. **Scalable**: Cross-episode learning means performance improves with experience

This makes Proactive RL the most **practically relevant** while still being **significantly better than purely reactive approaches**.
