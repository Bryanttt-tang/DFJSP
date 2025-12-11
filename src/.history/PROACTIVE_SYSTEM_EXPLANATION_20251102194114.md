# Comprehensive Explanation: Proactive Dynamic FJSP System

## Fixed Bug 🐛

**Error:** `AttributeError: 'float' object has no attribute 'time'`

**Root Cause:** At line 85, global scope code had:
```python
for i, (time, job_id, job_type) in enumerate(ARRIVAL_SEQUENCE[:10]):
```

This assigned `time` as a float, shadowing the `time` module imported at line 8. When `main()` later tried to call `time.time()`, it failed.

**Fix:** Changed variable name from `time` to `arr_time`:
```python
for i, (arr_time, job_id, job_type) in enumerate(ARRIVAL_SEQUENCE[:10]):
```

---

## 1. Job Data Creation and Normalization

### 1.1 Realistic Dataset Generation

The system uses `generate_realistic_fjsp_dataset()` from `utils.py` to create jobs with:

```python
ENHANCED_JOBS_DATA, MACHINE_LIST, MACHINE_METADATA = generate_realistic_fjsp_dataset(
    num_initial_jobs=5,        # Jobs available at t=0
    num_future_jobs=15,        # Jobs arriving dynamically
    total_num_machines=6,      # Number of machines
    job_type_distribution={'short': 0.5, 'moderate': 0.3, 'long': 0.2},
    machine_speed_variance=0.5,  # Speed differences between machines
    seed=GLOBAL_SEED
)
```

**Output Format (Metadata Structure):**
```python
{
    0: {
        'type': 'short',
        'operations': [
            {'proc_times': {'M0': 5.2, 'M1': 7.8, 'M2': 6.5}},  # Operation 0
            {'proc_times': {'M0': 4.1, 'M3': 6.2}}               # Operation 1
        ]
    },
    1: {
        'type': 'moderate',
        'operations': [...]
    },
    ...
}
```

**Job Types:**
- **SHORT**: 1-2 operations, processing times 5-15 units
- **MODERATE**: 2-4 operations, processing times 15-30 units
- **LONG**: 3-5 operations, processing times 30-50 units

**Machine Heterogeneity:**
- **FAST machines**: 0.6-0.8× speed multiplier (process faster)
- **MEDIUM machines**: 0.9-1.1× speed multiplier (normal)
- **SLOW machines**: 1.2-1.5× speed multiplier (process slower)

Processing times are adjusted by machine speed factors:
```python
# Base processing time: 10 units
# On FAST machine (0.7×): 7 units
# On SLOW machine (1.3×): 13 units
# Difference: 85% longer on slow vs fast!
```

### 1.2 Arrival Sequence Generation

```python
DETERMINISTIC_ARRIVAL_TIMES, ARRIVAL_SEQUENCE = generate_realistic_arrival_sequence(
    jobs_data=ENHANCED_JOBS_DATA,
    num_initial_jobs=5,
    arrival_rate=0.08,          # Mean: 1 job per 12.5 time units
    pattern_strength=0.5,       # 50% pattern, 50% random
    seed=GLOBAL_SEED
)
```

**Soft Probabilistic Patterns:**
- After 4+ SHORT jobs → 50% chance of LONG job next
- After LONG job → 70% chance of SHORT job next
- Otherwise → Sample from job type distribution

**Output:**
```python
DETERMINISTIC_ARRIVAL_TIMES = {0: 0.0, 1: 0.0, 2: 0.0, 3: 8.2, 4: 15.7, ...}
ARRIVAL_SEQUENCE = [
    (0.0, 0, 'short'),
    (0.0, 1, 'moderate'),
    (0.0, 2, 'short'),
    (8.2, 3, 'short'),
    (15.7, 4, 'long'),   # Pattern: 4 SHORT → LONG
    (22.1, 5, 'short'),  # Pattern: After LONG → SHORT
    ...
]
```

### 1.3 Data Normalization

**Problem:** Environments need simple format, but we want to keep metadata for analysis.

**Solution:** `normalize_jobs_data()` function:

```python
def normalize_jobs_data(jobs_data):
    """
    Convert from:  {job_id: {'type': 'short', 'operations': [...]}}
    Convert to:    {job_id: [...]} (just operations)
    """
    normalized = collections.OrderedDict()
    
    first_job = list(jobs_data.values())[0]
    if isinstance(first_job, dict) and 'operations' in first_job:
        # Metadata format - extract operations
        for job_id, job_info in jobs_data.items():
            normalized[job_id] = job_info['operations']
    else:
        # Already simple format
        normalized = jobs_data
    
    return normalized
```

**Usage:**
```python
JOBS_WITH_METADATA = ENHANCED_JOBS_DATA  # Keep for analysis
ENHANCED_JOBS_DATA = normalize_jobs_data(ENHANCED_JOBS_DATA)  # For environments
```

**Result:**
```python
# Before normalization:
{0: {'type': 'short', 'operations': [{'proc_times': {...}}, ...]}}

# After normalization:
{0: [{'proc_times': {...}}, ...]}
```

This ensures compatibility with all environments while preserving job type information for analysis.

---

## 2. Arrival Time and Job Type Prediction

### 2.1 ArrivalPredictor Class Overview

The `ArrivalPredictor` uses **Maximum Likelihood Estimation (MLE)** for Poisson processes with **cross-episode learning**.

**Key Innovation:** Accumulates knowledge from ALL past episodes (last 100), not just current episode.

### 2.2 Initialization

```python
class ArrivalPredictor:
    def __init__(self, initial_rate_guess=0.05):
        # Cross-episode learning storage
        self.global_inter_arrivals = []  # ALL inter-arrival times from past episodes
        
        # Current episode tracking
        self.current_episode_arrivals = []  # Arrival times this episode
        self.current_episode_inter_arrivals = []  # Inter-arrivals this episode
        
        # MLE estimates
        self.global_estimated_rate = initial_rate_guess  # From ALL history
        self.current_estimated_rate = initial_rate_guess  # Current + history
        
        # Correction tracking
        self.prediction_errors = []  # Track mispredictions
```

### 2.3 Within-Episode Learning

**Step 1: Observe Arrival**
```python
def observe_arrival(self, arrival_time):
    """Called each time a job actually arrives"""
    self.current_episode_arrivals.append(arrival_time)
    self.current_episode_arrivals.sort()
    
    # Calculate inter-arrival time
    if len(self.current_episode_arrivals) >= 2:
        last_arrival = self.current_episode_arrivals[-2]
        inter_arrival = arrival_time - last_arrival
        
        if inter_arrival > 0:
            self.current_episode_inter_arrivals.append(inter_arrival)
            self._update_mle_estimate()  # IMMEDIATE update!
```

**Step 2: Update MLE Estimate (Combines Historical + Current)**
```python
def _update_mle_estimate(self):
    """Uses BOTH global history AND current episode data"""
    # Combine all data
    all_data = self.global_inter_arrivals + self.current_episode_inter_arrivals
    
    if len(all_data) > 0:
        # Weight recent observations more
        if len(self.current_episode_inter_arrivals) >= 3:
            weighted_data = (self.global_inter_arrivals + 
                           self.current_episode_inter_arrivals * 2)
            mean_inter_arrival = np.mean(weighted_data)
        else:
            mean_inter_arrival = np.mean(all_data)
        
        if mean_inter_arrival > 0:
            # MLE for Poisson: λ̂ = 1 / E[τ]
            self.current_estimated_rate = 1.0 / mean_inter_arrival
```

**Mathematics:**

For Poisson process with rate λ, inter-arrival times follow Exponential(λ):
- E[τ] = 1/λ  (expected inter-arrival time)
- MLE: λ̂ = 1 / (sample mean of inter-arrivals)

**Example:**
```
Episode 1-50 observed inter-arrivals: [10, 12, 11, 13, 9, 12, ...]
Mean inter-arrival ≈ 11.2 time units
Estimated rate: λ̂ = 1/11.2 ≈ 0.089 jobs/time unit

Episode 51, current observations: [9.5, 10.3]
Combined mean: (11.2 × 50 + 9.5 × 2 + 10.3 × 2) / 54 ≈ 11.0
Updated rate: λ̂ = 1/11.0 ≈ 0.091 jobs/time unit
```

### 2.4 Predicting Future Arrivals

```python
def predict_next_arrivals(self, current_time, num_jobs_to_predict, 
                         last_known_arrival=None):
    """Predict when next N jobs will arrive"""
    # Use current estimate (includes ALL historical data!)
    if self.current_estimated_rate <= 0:
        mean_inter_arrival = 1.0 / self.initial_rate
    else:
        mean_inter_arrival = 1.0 / self.current_estimated_rate
    
    # Anchor to last known arrival
    if last_known_arrival is not None and last_known_arrival >= current_time:
        anchor_time = last_known_arrival
    elif len(self.current_episode_arrivals) > 0:
        anchor_time = self.current_episode_arrivals[-1]
    else:
        anchor_time = current_time
    
    # Predict at regular intervals
    predictions = []
    for i in range(1, num_jobs_to_predict + 1):
        predicted_time = anchor_time + i * mean_inter_arrival
        predictions.append(predicted_time)
    
    return predictions
```

**Example:**
```
Current time: t=25
Last known arrival: t=22
Estimated mean inter-arrival: 12.5 time units
Number to predict: 3

Predictions:
  Job 1: 22 + 1×12.5 = 34.5
  Job 2: 22 + 2×12.5 = 47.0
  Job 3: 22 + 3×12.5 = 59.5
```

**Job Type Prediction:**

The predictor does NOT predict job types explicitly. Instead:
1. Job types are embedded in the dataset generation (soft patterns)
2. Agent learns to recognize patterns through observations
3. Observation space includes job type information (indirectly via processing times)

Patterns agent can learn:
```
Pattern 1: After seeing 4 consecutive jobs with short processing times 
          (5-15 units) → likely LONG job next (30-50 units)

Pattern 2: After seeing job with long processing times 
          → likely SHORT job next
```

### 2.5 Prediction Confidence

```python
def get_confidence(self):
    """Confidence based on total observations across ALL episodes"""
    total_observations = len(self.global_inter_arrivals)
    
    if total_observations == 0:
        return 0.0
    
    # Confidence grows with sqrt of observations
    # 50% at ~25 obs, 90% at ~100 obs
    confidence = 1.0 - np.exp(-np.sqrt(total_observations) / 5.0)
    return np.clip(confidence, 0.0, 1.0)
```

**Confidence Curve:**
```
Episodes  |  Observations  |  Confidence
----------|----------------|-------------
1-5       |  5-15          |  0.2-0.4
10-20     |  20-50         |  0.5-0.7
50+       |  100+          |  0.8-0.95
```

### 2.6 Cross-Episode Learning

```python
def finalize_episode(self, all_arrival_times):
    """Accumulate knowledge from this episode to global history"""
    # Extract all inter-arrival times
    arrival_list = sorted([t for t in all_arrival_times.values() if t > 0])
    
    episode_inter_arrivals = []
    for i in range(1, len(arrival_list)):
        inter_arrival = arrival_list[i] - arrival_list[i-1]
        if inter_arrival > 0:
            episode_inter_arrivals.append(inter_arrival)
    
    # ADD TO GLOBAL HISTORY - key to cross-episode learning!
    self.global_inter_arrivals.extend(episode_inter_arrivals)
    
    # Update global MLE with ALL data
    self._update_global_mle()
    
    # Reset current episode
    self.current_episode_inter_arrivals = []
```

**Accumulation Example:**
```
Episode 1:  [11.2, 13.1, 9.8] → global = [11.2, 13.1, 9.8]
Episode 2:  [12.5, 10.9] → global = [11.2, 13.1, 9.8, 12.5, 10.9]
Episode 3:  [11.8, 12.2, 10.5] → global = [11.2, ..., 11.8, 12.2, 10.5]
...
Episode 100: global has ~200-300 inter-arrival observations

Benefit: Episode 100 predictions use ALL past data, not just current episode!
```

---

## 3. Prediction Correction Mechanism

### 3.1 When Corrections Happen

In original design (now removed), when agent scheduled a job **before** it arrived (proactive scheduling):

```python
# In step() function (OLD VERSION - removed in enhanced version)
if job_id not in self.arrived_jobs:
    # Scheduled proactively
    actual_arrival = self.job_arrival_times[job_id]
    
    if job_id in self.predicted_arrival_times:
        predicted_arrival = self.predicted_arrival_times[job_id]
        
        # Correct the predictor
        self.arrival_predictor.correct_prediction(
            job_id, predicted_arrival, actual_arrival
        )
```

### 3.2 Correction Method

```python
def correct_prediction(self, job_id, predicted_time, actual_time):
    """Learn from prediction errors"""
    prediction_error = actual_time - predicted_time
    self.prediction_errors.append(prediction_error)
    
    # If consistently over/under-estimating, adjust rate
    if len(self.prediction_errors) >= 5:
        mean_error = np.mean(self.prediction_errors[-20:])
        
        # Correction logic:
        # mean_error > 0: Predict too early → decrease rate
        # mean_error < 0: Predict too late → increase rate
        if abs(mean_error) > 0.5:
            correction_factor = 1.0 - (mean_error / (1.0/self.current_estimated_rate)) * 0.1
            correction_factor = np.clip(correction_factor, 0.5, 2.0)
            self.current_estimated_rate *= correction_factor
```

**Example:**
```
Scenario: Predictor consistently predicts arrivals 3 units too early

Predictions vs Reality:
  Predicted t=10, Actual t=13 → error = +3
  Predicted t=22, Actual t=25 → error = +3
  Predicted t=35, Actual t=38 → error = +3
  ...

Mean error ≈ +3 (positive = too early)

Current rate: λ = 0.08 → mean inter-arrival = 12.5
Correction: factor = 1.0 - (3 / 12.5) * 0.1 = 0.976

New rate: λ = 0.08 * 0.976 = 0.078
New mean inter-arrival: 1/0.078 ≈ 12.8 (slightly longer, less optimistic)
```

### 3.3 Enhanced Version (Current Implementation)

**Key Change:** Prediction window removed! Agent can ONLY schedule arrived jobs.

Predictions now used ONLY to guide wait decisions (not scheduling), so explicit correction is less critical. Instead:

1. **Observation updates:** Agent sees actual arrivals and learns
2. **Natural correction:** Predictor updates estimates as real arrivals observed
3. **Reward shaping:** Good predictions → better wait rewards → agent learns to trust predictor

---

## 4. Observation Space Design for Wait Learning

### 4.1 Why Observation Space Matters for Wait Learning

**Critical Insight:** Wait actions have DELAYED consequences, so agent needs rich temporal information to learn when waiting is beneficial.

**Bad observation:** Just current machine states
- Agent can't predict future value of waiting
- Learns: "Waiting = negative reward, always schedule now"

**Good observation:** Current state + future predictions + confidence
- Agent can evaluate: "Is waiting likely to yield valuable job?"
- Learns: "Wait if predictor confident AND predicts valuable job soon"

### 4.2 ProactiveDynamicFJSPEnv Observation Space

```python
# ENHANCED observation space (line 1266-1282)
obs_size = (
    len(self.job_ids) +                      # 1. Ready job indicators
    len(self.machines) +                     # 2. Machine idle status
    len(self.job_ids) * len(self.machines) + # 3. Processing times
    len(self.job_ids) +                      # 4. Job progress
    len(self.job_ids) +                      # 5. Predicted arrival times ⭐
    len(self.job_ids) +                      # 6. Prediction confidence ⭐
    3                                        # 7. Estimated arrival rate ⭐
)
```

**Breakdown:**

**Component 1: Ready Job Indicators (n_jobs values)**
```python
for job_id in self.job_ids:
    if (job_id in self.arrived_jobs and 
        self.job_progress[job_id] < len(self.jobs[job_id])):
        obs.append(1.0)  # Job ready to schedule
    else:
        obs.append(0.0)  # Not ready
```

**Component 2: Machine Idle Status (n_machines values)**
```python
for machine in self.machines:
    if self.machine_end_times[machine] <= self.event_time:
        obs.append(1.0)  # Machine idle
    else:
        obs.append(0.0)  # Machine busy
```

**Component 3: Processing Times (n_jobs × n_machines values)**
```python
for job_id in self.job_ids:
    if (job_id in self.arrived_jobs and 
        self.job_progress[job_id] < len(self.jobs[job_id])):
        next_op = self.jobs[job_id][self.job_progress[job_id]]
        
        for machine in self.machines:
            if machine in next_op['proc_times']:
                proc_time = next_op['proc_times'][machine]
                normalized = min(1.0, proc_time / self.max_proc_time)
                obs.append(normalized)
            else:
                obs.append(0.0)
    else:
        for machine in self.machines:
            obs.append(0.0)
```

**Component 4: Job Progress (n_jobs values)**
```python
for job_id in self.job_ids:
    progress = self.job_progress[job_id] / len(self.jobs[job_id])
    obs.append(progress)
```

**⭐ Component 5: Predicted Arrival Times (n_jobs values) - CRITICAL FOR WAIT LEARNING**
```python
for job_id in self.job_ids:
    if job_id in self.arrived_jobs or job_id in self.completed_jobs:
        obs.append(0.0)  # Already arrived
    elif job_id in self.predicted_arrival_times:
        pred_time = self.predicted_arrival_times[job_id]
        time_until_arrival = pred_time - self.event_time
        
        # Normalize: 0 = arriving now, 1 = arriving far future
        normalized = min(1.0, time_until_arrival / self.max_time_horizon)
        obs.append(max(0.0, normalized))
    else:
        obs.append(1.0)  # No prediction yet
```

**Why this helps wait learning:**
- Value close to 0 → predicted arrival soon → **WAIT might be good**
- Value close to 1 → predicted arrival far away → **SCHEDULE now**

**Example:**
```
Current time: t=20
Max horizon: 200

Job 5 predicted at t=25:
  time_until = 25-20 = 5
  normalized = 5/200 = 0.025 ⚡ Very small → arriving soon!
  
Job 8 predicted at t=120:
  time_until = 120-20 = 100
  normalized = 100/200 = 0.5 → arriving later

Agent learns: "Observation ~0.025 → wait 5 units likely yields arrival"
```

**⭐ Component 6: Prediction Confidence (n_jobs values) - CRITICAL FOR TRUST**
```python
confidence = self.arrival_predictor.get_confidence()

for job_id in self.job_ids:
    if job_id in self.arrived_jobs or job_id in self.completed_jobs:
        obs.append(1.0)  # Already arrived (100% certain)
    elif job_id in self.predicted_arrival_times:
        obs.append(confidence)  # Use predictor's confidence
    else:
        obs.append(0.0)  # No prediction
```

**Why this helps wait learning:**
- High confidence (0.8-0.9) → **Trust prediction, wait if arrival soon**
- Low confidence (0.1-0.3) → **Don't trust, schedule now**

**Learning progression:**
```
Episode 1-50:   Confidence ~0.3 → agent learns to mostly ignore predictions
Episode 50-200: Confidence ~0.6 → agent starts using predictions for wait decisions
Episode 200+:   Confidence ~0.8 → agent relies on predictions strategically
```

**⭐ Component 7: Estimated Arrival Rate (3 values) - TEMPORAL CONTEXT**
```python
# Current rate estimate
if self.arrival_predictor.current_estimated_rate > 0:
    normalized_rate = min(1.0, self.arrival_predictor.current_estimated_rate / (2.0 * self.arrival_rate))
    obs.append(normalized_rate)
else:
    obs.append(0.5)

# Mean inter-arrival time
mean_inter = (1.0 / self.arrival_predictor.current_estimated_rate 
              if self.arrival_predictor.current_estimated_rate > 0 
              else self.max_time_horizon)
normalized_inter = min(1.0, mean_inter / self.max_time_horizon)
obs.append(normalized_inter)

# Number of observations (experience)
num_obs = len(self.arrival_predictor.global_inter_arrivals)
normalized_obs = min(1.0, num_obs / 100.0)
obs.append(normalized_obs)
```

**Why this helps wait learning:**

1. **Estimated rate** tells agent "how fast are jobs arriving?"
   - High rate (0.8) → frequent arrivals → **waiting cheaper**
   - Low rate (0.2) → infrequent arrivals → **waiting expensive**

2. **Mean inter-arrival** tells agent "typical time between arrivals"
   - Small value (0.1) → arrivals every ~20 time units → **short waits OK**
   - Large value (0.8) → arrivals every ~160 time units → **long waits BAD**

3. **Number of observations** tells agent "how much has predictor learned?"
   - High value (0.9) → 90+ observations → **predictor reliable**
   - Low value (0.1) → 10 observations → **predictor uncertain**

### 4.3 Observation Space Comparison

**PoissonDynamicFJSPEnv (Reactive RL) - 83 dimensions:**
```python
obs_size = (
    self.num_jobs +                         # Ready job indicators
    self.num_jobs +                         # Job progress
    len(self.machines) +                    # Machine states
    self.num_jobs * len(self.machines) +    # Processing times
    self.num_jobs +                         # Time since last arrival
    2                                       # Arrival progress, makespan progress
)
```
- **No prediction information**
- Agent learns: "Wait is always bad (negative reward)"
- Result: Aggressive scheduling, rarely waits

**ProactiveDynamicFJSPEnv - 103 dimensions (+20):**
```python
obs_size = (
    # ... same as Reactive RL ...
    self.num_jobs +     # Predicted arrival times ⭐
    self.num_jobs +     # Prediction confidence ⭐
    3                   # Rate estimate, inter-arrival, observations ⭐
)
```
- **Rich prediction information**
- Agent learns: "Wait is good IF prediction confident AND arrival soon"
- Result: Strategic waiting, balances immediate vs future value

### 4.4 How Agent Learns to Wait Using Observations

**Training Progression:**

**Early Episodes (1-100):**
```
Observations:
  predicted_arrival[J5] = 0.95 (far away, low confidence)
  confidence[J5] = 0.2 (low)
  
Agent action: Wait 5 units
Result: No arrival (wasted time)
Reward: -0.5 (base wait penalty)

Learning: "When confidence low, don't wait long"
```

**Mid Episodes (100-500):**
```
Observations:
  predicted_arrival[J7] = 0.05 (soon!, medium confidence)
  confidence[J7] = 0.6 (medium)
  mean_inter_arrival = 0.15 (arrivals every 30 units)
  
Agent action: Wait 5 units
Result: J7 arrives at t+4.8!
Reward: -0.35 (wait penalty) + 0.35 (alignment bonus) = 0.0

Learning: "When confidence medium+ AND prediction soon, wait pays off"
```

**Late Episodes (500+):**
```
Observations:
  predicted_arrival[J10] = 0.03 (very soon!, high confidence)
  confidence[J10] = 0.85 (high)
  current_rate = 0.15 (frequent arrivals)
  machine_idle = [1, 1, 0] (2 idle)
  schedulable_jobs = [J8] (SHORT job available)
  
Agent decision tree (learned):
  IF predicted_arrival < 0.1 AND confidence > 0.7:
      IF job is predicted LONG (high proc_times):
          → Wait to assign LONG job to idle FAST machine
      ELSE:
          → Schedule current SHORT job, wait is not worth it
  ELSE:
      → Schedule now
      
Result: Sophisticated strategic waiting policy
```

---

## Summary: Complete System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION (Once)                                        │
├─────────────────────────────────────────────────────────────────┤
│ • Generate realistic job dataset (types, machine speeds)        │
│ • Create arrival sequence (Poisson + soft patterns)             │
│ • Normalize job data for environment compatibility              │
│ • Initialize ArrivalPredictor (initial_rate_guess=0.05)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. EPISODE START (Each Episode)                                 │
├─────────────────────────────────────────────────────────────────┤
│ • Reset predictor (keep global history!)                        │
│ • Generate arrival times using TRUE rate (hidden from agent)    │
│ • Initialize jobs at t=0                                        │
│ • Make initial predictions (low confidence, ~0.2)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. STEP LOOP (Each Timestep)                                    │
├─────────────────────────────────────────────────────────────────┤
│ A. GET OBSERVATION:                                              │
│    • Current state (machines, jobs, progress)                   │
│    • Predicted arrival times (from predictor)                   │
│    • Prediction confidence (from history)                       │
│    • Rate estimates (from MLE)                                  │
│                                                                  │
│ B. AGENT CHOOSES ACTION:                                        │
│    • Scheduling: Only ARRIVED jobs allowed                      │
│    • Waiting: 6 durations [1,2,3,5,10,∞]                       │
│    • Uses observation to decide                                 │
│                                                                  │
│ C. EXECUTE ACTION:                                              │
│    If scheduling:                                               │
│      → Assign operation to machine                              │
│      → Update makespan                                          │
│      → Reward = -makespan_increment                             │
│                                                                  │
│    If waiting:                                                  │
│      → Advance time by duration                                 │
│      → Check for new arrivals                                   │
│      → Calculate predictor-guided reward:                       │
│         * Base penalty (-duration × 0.1)                        │
│         * Alignment bonus (if prediction correct)               │
│         * Opportunity cost (idle machines penalty)              │
│         * Patience bonus (arrivals during wait)                 │
│      → Observe new arrivals → update predictor MLE              │
│                                                                  │
│ D. UPDATE PREDICTOR:                                            │
│    • Add new observations to current episode data               │
│    • Recalculate MLE using global history + current data        │
│    • Update confidence based on total observations              │
│    • Generate new predictions for unarrived jobs                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. EPISODE END                                                   │
├─────────────────────────────────────────────────────────────────┤
│ • Finalize predictor: Add episode data to global history        │
│ • Update global MLE estimate with ALL data                      │
│ • Next episode starts with improved predictions!                │
└─────────────────────────────────────────────────────────────────┘
```

**Key Innovations:**

1. ✅ **Cross-episode learning**: Predictor gets smarter over 100s of episodes
2. ✅ **Confidence weighting**: Weak early predictions, strong later predictions
3. ✅ **Rich observations**: Agent sees predictions + confidence + rate estimates
4. ✅ **Strategic waiting**: Agent learns WHEN to wait based on predictions
5. ✅ **No cheating**: Only schedules ARRIVED jobs (reactive scheduling)
6. ✅ **Predictor guidance**: Helps agent learn faster, but doesn't control decisions

This design creates a sophisticated learning system where the agent develops strategic waiting policies guided by increasingly accurate arrival predictions!
