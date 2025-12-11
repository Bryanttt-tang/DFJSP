# Simplified FJSP Setting: Machine Heterogeneity Focus

## Overview

This document explains the **simplified experimental setting** where we focus on **machine heterogeneity** as the primary source of strategic decision-making, while treating jobs as homogeneous.

## Key Design Decisions

### 1. **Jobs are Homogeneous**
- **No job type classification** (no SHORT/MODERATE/LONG categories)
- All jobs have similar structure (2-4 operations)
- However, **high variance in processing times** across operations (5-50 time units)
- This variance creates waiting opportunities without explicit job categories

### 2. **Machine Heterogeneity is the Strategic Element**
Machines are categorized by speed factors:

| Category | Speed Factor | Processing Time | Example |
|----------|--------------|-----------------|---------|
| **FAST** | 0.6 - 0.8 | 20-40% faster | 40 * 0.7 = **28 units** |
| **MEDIUM** | 0.9 - 1.1 | Baseline | 40 * 1.0 = **40 units** |
| **SLOW** | 1.2 - 1.5 | 20-50% slower | 40 * 1.5 = **60 units** |

**Key Insight**: The gap between fast and slow machines (up to 2.5x) creates **meaningful waiting decisions**!

### 3. **Simple Poisson Arrivals (No Patterns)**
- Jobs arrive according to a **Poisson process** with rate λ
- **Without loss of generality**, assume specific job arrival sequence (J0→J1→J2→...)
- Since jobs are homogeneous, the specific sequence doesn't matter
- Only the **timing** of arrivals (controlled by λ) affects scheduling decisions

### 4. **Wait Action Becomes Critical**

The agent must learn to answer:
> "Should I schedule this job NOW on an available slow machine, or WAIT for a fast machine?"

#### Example Decision Scenario:
```
Current state:
- Job J5 arrived with operation requiring 40 time units (base processing time)
- SLOW machine (M2, speed=1.5) is FREE NOW
- FAST machine (M0, speed=0.7) will be free in 5 time units

Option A: Schedule NOW on slow machine
- Processing time: 40 * 1.5 = 60 units
- Completion: current_time + 60

Option B: WAIT 5 units for fast machine
- Wait cost: 5 units
- Processing time: 40 * 0.7 = 28 units
- Completion: current_time + 5 + 28 = current_time + 33
- SAVINGS: 60 - 33 = 27 time units! ✅
```

The agent learns:
- **Small operations** (proc_time ≈ 5-10): Schedule immediately (gap is small)
- **Large operations** (proc_time ≈ 40-50): Wait for fast machine (gap is huge!)
- **Medium operations**: Depends on machine availability and next job arrival

## Why This Setting is Sufficient

### 1. **No Job Classification Needed**
- Processing time variance **naturally creates heterogeneity**
- Some operations are 10x longer than others (5 vs 50)
- Agent learns to treat long operations differently without explicit labels

### 2. **Machine Speed is Universal**
- Fast machines process **ALL jobs** faster (not job-type specific)
- Strategic waiting benefits **apply universally**
- Simpler to analyze and understand

### 3. **Poisson Arrivals are Realistic**
- In practice, job arrivals are often modeled as Poisson processes
- Without loss of generality assumption is standard in queuing theory
- Focuses on **arrival timing uncertainty**, not arrival sequence patterns

## Training and Testing

### Training (ProactiveDynamicFJSPEnv)
- **Arrival Generation**: Simple Poisson sampling
- **Episode Diversity**: Different arrival times each episode
- **Learning Target**: Predict arrival rate λ via MLE
- **Strategic Learning**: When to wait vs schedule immediately

```python
# Training: Jobs arrive via Poisson process
arrival_times = {}
current_time = 0.0
for job_id in dynamic_job_ids:
    inter_arrival = np.random.exponential(1.0 / arrival_rate)
    current_time += inter_arrival
    arrival_times[job_id] = current_time
```

### Testing (Fixed Scenarios)
- **Same Poisson Process**: Generate fixed test scenarios with different seeds
- **Consistent Evaluation**: All methods tested on identical arrival sequences
- **Generalization Test**: Test seeds ≠ training seed

```python
# Test: Same Poisson generation, but fixed seeds
test_scenarios = generate_test_scenarios(
    jobs_data, 
    initial_jobs=[0, 1, 2], 
    arrival_rate=0.08,
    num_scenarios=10
)
```

## Strategic Depth Analysis

Even without job classification, the problem has rich strategic depth:

### 1. **Machine Selection Trade-offs**
| Factor | Fast Machine | Slow Machine |
|--------|--------------|--------------|
| Processing time | Low (0.6-0.8x) | High (1.2-1.5x) |
| Availability | Often busy | More often free |
| Wait cost | May need to wait | Available now |
| Long-term benefit | Frees up capacity | Blocks capacity longer |

### 2. **Arrival Prediction Impact**
- **Proactive RL** learns to predict when next job will arrive
- If next job arrival is **imminent**, might wait instead of scheduling
- If next job arrival is **far**, schedule now to avoid idleness

### 3. **Processing Time Awareness**
Agent learns operation value:
- **High processing time** (40-50): **High value** of fast machine access
- **Low processing time** (5-10): **Low value** of fast machine access
- Emergent strategy: "Save fast machines for long operations"

## Comparison to Previous Setting

| Aspect | Previous (Complex) | New (Simplified) |
|--------|-------------------|------------------|
| **Job Types** | SHORT/MODERATE/LONG | Homogeneous (variance only) |
| **Arrivals** | Pattern-based (job type clusters) | Simple Poisson |
| **Strategic Element** | Job type + machine speed | Machine speed only |
| **Wait Decision** | "Wait for fast machine for LONG jobs" | "Wait for fast machine for HIGH proc_time ops" |
| **Complexity** | High (two dimensions) | Lower (one dimension) |
| **Interpretability** | Harder to analyze | Easier to understand |

## Expected Behavior

### Reactive RL (Baseline)
- Only knows arrival rate λ
- Cannot predict specific arrival times
- Schedules greedily based on current state
- Performance: **Baseline**

### Proactive RL (Our Approach)
- Learns arrival rate λ via MLE
- Predicts next job arrival time
- Uses WAIT action strategically
- Performance: **Better than Reactive** (if wait learning succeeds)

### Perfect Knowledge RL (Upper Bound)
- Knows exact arrival times
- Optimal scheduling with perfect foresight
- Performance: **Near MILP optimal** (0.1-2% gap)

### Static RL (Lower Bound)
- Assumes all jobs available at t=0
- No awareness of dynamic arrivals
- Performance: **Worst** (ignores arrival information)

## Implementation Changes

### Dataset Generation
```python
# NEW: Simplified dataset with machine heterogeneity only
jobs_data, machine_list, machine_metadata = generate_simplified_fjsp_dataset(
    num_initial_jobs=5,
    num_future_jobs=10,
    total_num_machines=4,
    machine_speed_variance=0.6,  # High heterogeneity
    proc_time_variance_range=(5, 50),  # Wide range
    seed=GLOBAL_SEED
)
```

### Training
```python
# ProactiveDynamicFJSPEnv now uses simple Poisson arrivals
env = ProactiveDynamicFJSPEnv(
    jobs_data, machine_list,
    initial_jobs=5,
    arrival_rate=0.08,  # Hidden from agent
    # No jobs_with_metadata parameter
    # No pattern_strength parameter
)
```

### Testing
```python
# Test scenarios use same Poisson generation
test_scenarios = generate_test_scenarios(
    jobs_data,
    initial_jobs=[0, 1, 2],
    arrival_rate=0.08,
    num_scenarios=10
)
```

## Research Questions

With this simplified setting, we can cleanly investigate:

1. **Does learning arrival predictions help?**
   - Compare Proactive RL vs Reactive RL
   - Isolate benefit of prediction from other factors

2. **How effective is the WAIT action?**
   - Analyze wait action frequency and duration
   - Measure makespan improvement from strategic waiting

3. **Can we learn machine-job matching?**
   - Does agent learn "long operations → fast machines"?
   - Implicit learning without explicit job classification

4. **How close can we get to optimal?**
   - Compare Proactive RL vs Perfect Knowledge RL
   - Quantify cost of arrival uncertainty

## Conclusion

This simplified setting:
- ✅ **Focuses on machine heterogeneity** as primary strategic element
- ✅ **Eliminates unnecessary complexity** (no job type patterns)
- ✅ **Keeps strategic depth** (processing time variance + machine speeds)
- ✅ **Makes wait action critical** (up to 2.5x difference in processing time)
- ✅ **Easier to analyze and understand**
- ✅ **Without loss of generality** (Poisson arrivals are standard)

The key insight: **Machine speed differences + processing time variance are sufficient to create rich strategic decisions**, without needing explicit job type classification!
