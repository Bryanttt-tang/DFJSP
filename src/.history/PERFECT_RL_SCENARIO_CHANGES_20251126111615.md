# Perfect RL Per-Scenario Training & Regret Analysis

## Summary of Changes

This update modifies the training and evaluation pipeline so that:
1. **Perfect RL trains separately for each test scenario** with scenario-specific arrival times
2. **Models are saved with scenario indices** (e.g., `perfect_knowledge_rl_model_scenario_0.zip`)
3. **Evaluation loads the correct model per scenario** from the saved files
4. **Comprehensive regret analysis** calculates performance gap vs MILP Optimal for all methods

---

## Changes to `proactive_sche.py`

### 1. Updated `train_perfect_knowledge_agent()` function signature

**Line 3116**: Added `scenario_idx=0` parameter

```python
def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, 
                                  reward_mode="makespan_increment", learning_rate=3e-4,
                                  num_initializations=3, milp_optimal=None, scenario_idx=0):
```

**Purpose**: Accepts scenario index to save model with unique filename per scenario.

---

### 2. Updated model saving path

**Line 3273**: Changed from single model to scenario-specific model

```python
# OLD:
model_path = "perfect_knowledge_rl_model.zip"

# NEW:
model_path = f"perfect_knowledge_rl_model_scenario_{scenario_idx}.zip"
```

**Result**: Each scenario gets its own Perfect RL model file.

---

### 3. Updated training call in main loop

**Line 6090**: Pass scenario index when training Perfect RL

```python
perfect_model = train_perfect_knowledge_agent(
    ENHANCED_JOBS_DATA, MACHINE_LIST,
    arrival_times=scenario_arrivals,
    total_timesteps=perfect_timesteps,
    reward_mode="makespan_increment", 
    learning_rate=learning_rate,
    num_initializations=3,
    milp_optimal=milp_makespan if milp_is_valid else None,
    scenario_idx=i  # ⭐ NEW: Pass scenario index
)
```

**Result**: Each test scenario trains its own Perfect RL model with exact arrival times.

---

## Changes to `eval.py`

### 1. Updated `load_trained_models()` function

**Line 54**: Added `num_scenarios` parameter and scenario-specific loading

```python
def load_trained_models(num_scenarios=1):
    """
    Load all pre-trained RL models from disk.
    For Perfect Knowledge RL, loads scenario-specific models.
    
    Args:
        num_scenarios: Number of test scenarios (for Perfect RL models)
    
    Returns:
        dict: Dictionary containing all loaded models
    """
```

**Key changes**:
- Loads single models for Reactive RL, Proactive RL, Rule-Based RL, Static RL
- Loads **list of Perfect RL models** (one per scenario):
  ```python
  models['perfect_knowledge_rl'] = []
  for scenario_idx in range(num_scenarios):
      model_path = f'perfect_knowledge_rl_model_scenario_{scenario_idx}.zip'
      # Load and append to list
  ```

**Result**: Returns `models['perfect_knowledge_rl']` as a list where `models['perfect_knowledge_rl'][i]` is the model for scenario `i`.

---

### 2. Updated scenario evaluation loop

**Line 205**: Use scenario-specific Perfect RL model

```python
# OLD:
if models.get('perfect_knowledge_rl'):
    perfect_makespan, perfect_schedule = evaluate_perfect_knowledge_on_scenario(
        models['perfect_knowledge_rl'], ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals
    )

# NEW:
perfect_rl_models = models.get('perfect_knowledge_rl', [])
if i < len(perfect_rl_models) and perfect_rl_models[i] is not None:
    print(f"  Evaluating Perfect Knowledge RL (scenario-specific model {i})...")
    perfect_makespan, perfect_schedule = evaluate_perfect_knowledge_on_scenario(
        perfect_rl_models[i], ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals
    )
```

**Result**: Each scenario is evaluated with its corresponding trained Perfect RL model.

---

### 3. Added comprehensive regret analysis

**Lines 357-415**: New regret calculation in `print_results_analysis()`

```python
# Calculate regret against MILP Optimal (scenario-by-scenario)
print("\n" + "="*80)
print("REGRET ANALYSIS (vs MILP Optimal)")
print("="*80)

num_scenarios = len(all_results['MILP Optimal'])
regret_results = {}

# For each method, calculate regret on each scenario
for method in ['Perfect Knowledge RL', 'Proactive RL', 'Reactive RL', 
               'Rule-Based RL', 'Static RL (dynamic)', 'Static RL (static)', 'Best Heuristic']:
    
    scenario_regrets = []
    for scenario_idx in range(num_scenarios):
        milp_makespan = all_results['MILP Optimal'][scenario_idx]
        method_makespan = all_results[method][scenario_idx]
        
        if (milp_makespan is not None and milp_makespan != float('inf') and
            method_makespan is not None and method_makespan != float('inf')):
            regret = ((method_makespan - milp_makespan) / milp_makespan) * 100
            scenario_regrets.append(regret)
    
    if scenario_regrets:
        avg_regret = np.mean(scenario_regrets)
        std_regret = np.std(scenario_regrets)
        min_regret = np.min(scenario_regrets)
        max_regret = np.max(scenario_regrets)
        regret_results[method] = {
            'mean': avg_regret,
            'std': std_regret,
            'min': min_regret,
            'max': max_regret,
            'count': len(scenario_regrets)
        }
        print(f"{method:25s}: {avg_regret:+6.2f}% ± {std_regret:5.2f}% " +
              f"(range: [{min_regret:+.2f}%, {max_regret:+.2f}%], n={len(scenario_regrets)})")
```

**Output format**:
```
REGRET ANALYSIS (vs MILP Optimal)
================================================================================
Perfect Knowledge RL     :  +2.15% ±  1.23% (range: [+0.50%, +4.20%], n=10)
Proactive RL             : +15.40% ±  3.50% (range: [+10.20%, +22.10%], n=10)
Reactive RL              : +18.30% ±  4.20% (range: [+12.50%, +26.40%], n=10)
...
```

---

### 4. Updated function return signature

**Line 417**: Returns regret results

```python
# OLD:
return avg_results, std_results

# NEW:
return avg_results, std_results, regret_results
```

---

### 5. Updated main function flow

**Lines 810-825**: Load models after generating scenarios

```python
# OLD ORDER:
# Step 1: Load models
# Step 2: Generate scenarios

# NEW ORDER:
# Step 1: Generate test scenarios (to know num_scenarios)
test_scenarios = generate_test_scenarios(...)

# Step 2: Load models (pass num_scenarios for Perfect RL)
models = load_trained_models(num_scenarios=num_scenarios)
```

**Reason**: Need to know `num_scenarios` before loading scenario-specific Perfect RL models.

---

## Workflow Summary

### Training Phase (`proactive_sche.py`)

```
For each test scenario i:
  1. Compute MILP Optimal solution
  2. Train Perfect RL with exact arrival times for scenario i
  3. Save model as: perfect_knowledge_rl_model_scenario_{i}.zip
  4. Evaluate Perfect RL on scenario i
  5. Evaluate other RL methods (Reactive, Proactive, Rule-Based, Static)
  6. Evaluate heuristics
```

### Evaluation Phase (`eval.py`)

```
1. Generate N test scenarios
2. Load N Perfect RL models (one per scenario)
3. For each scenario i:
   - Load Perfect RL model i
   - Evaluate all methods on scenario i
   - Store results
4. Calculate regret analysis:
   - For each method and each scenario:
     - regret = (method_makespan - milp_makespan) / milp_makespan * 100%
   - Report mean, std, min, max regret across scenarios
```

---

## Expected Output

### Training Output

```
Training Perfect Knowledge RL for scenario 1...
  Scenario arrival times: {0: 0.0, 1: 0.0, 2: 0.0, 3: 5.2, 4: 12.3, ...}
  Target: MILP Optimal = 97.00
  --- Initialization 1/3 (seed=13345) ---
  ...
  Init 1 makespan: 98.50 (gap: +1.55%) ⭐ NEW BEST
  ...
  ✅ Best initialization: 1 with makespan 98.50
  📊 Final gap to MILP: +1.55%
  💾 Perfect Knowledge RL model saved to: perfect_knowledge_rl_model_scenario_0.zip
```

### Evaluation Output

```
LOADING PRE-TRAINED RL MODELS
================================================================================
✅ Loaded reactive_rl: reactive_rl_model.zip
✅ Loaded proactive_rl: proactive_rl_model.zip
✅ Loaded rule_based_rl: rule_based_rl_model.zip
✅ Loaded static_rl: static_rl_model.zip

📁 Loading Perfect Knowledge RL models for 10 scenarios...
✅ Loaded Perfect RL for scenario 0: perfect_knowledge_rl_model_scenario_0.zip
✅ Loaded Perfect RL for scenario 1: perfect_knowledge_rl_model_scenario_1.zip
...

REGRET ANALYSIS (vs MILP Optimal)
================================================================================
Perfect Knowledge RL     :  +2.15% ±  1.23% (range: [+0.50%, +4.20%], n=10)
Proactive RL             : +15.40% ±  3.50% (range: [+10.20%, +22.10%], n=10)
Reactive RL              : +18.30% ±  4.20% (range: [+12.50%, +26.40%], n=10)
Rule-Based RL            : +22.50% ±  5.10% (range: [+15.30%, +30.20%], n=10)
Static RL (dynamic)      : +35.20% ±  7.80% (range: [+25.10%, +45.30%], n=10)
Best Heuristic           : +25.10% ±  6.20% (range: [+18.40%, +35.60%], n=10)

SUMMARY
================================================================================
Total test scenarios: 10
Valid MILP solutions: 10/10

Best method by average regret:
  Perfect Knowledge RL: +2.15% ± 1.23%
```

---

## Benefits

1. ✅ **Fair Comparison**: Each scenario has its optimal RL baseline (trained on exact arrivals)
2. ✅ **Rigorous Evaluation**: Scenario-specific models eliminate cross-scenario contamination
3. ✅ **Comprehensive Metrics**: Regret analysis shows not just averages but also ranges and consistency
4. ✅ **Research-Ready**: Publication-quality regret statistics (mean ± std, min/max ranges)
5. ✅ **Debugging**: Easy to identify which scenarios are hardest for each method

---

## Files Modified

- ✅ `proactive_sche.py`: Updated training to save per-scenario models
- ✅ `eval.py`: Updated loading and evaluation to use per-scenario models
- ✅ Added comprehensive regret analysis

## Files Generated (Training)

- `perfect_knowledge_rl_model_scenario_0.zip`
- `perfect_knowledge_rl_model_scenario_1.zip`
- `perfect_knowledge_rl_model_scenario_2.zip`
- ... (one per test scenario)

## Usage

```bash
# 1. Train all models (including per-scenario Perfect RL)
python proactive_sche.py

# 2. Evaluate and calculate regret
python eval.py
```
