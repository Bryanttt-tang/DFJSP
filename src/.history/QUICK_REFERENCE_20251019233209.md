# Quick Reference Guide: FJSP Dataset Generation

## Basic Usage

```python
from utils import generate_fjsp_dataset, generate_arrival_times

# Generate dataset
jobs_data, machine_list = generate_fjsp_dataset(
    num_initial_jobs=3,     # Jobs at t=0
    num_future_jobs=4,      # Dynamic arrivals
    total_num_machines=3,   # Number of machines
    seed=42                 # For reproducibility
)

# Generate arrivals
arrival_times = generate_arrival_times(
    num_initial_jobs=3,
    num_future_jobs=4,
    arrival_mode='deterministic',  # or 'poisson'
    arrival_rate=0.08,
    seed=42
)
```

## Generation Rules

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| Operations per job | Uniform | [1, 5] |
| Machines per operation | Uniform | [1, M] |
| Processing time | Uniform | [1, 10] |

## Arrival Modes

### Deterministic
- Initial jobs: t=0
- Future jobs: t=4, 8, 12, 16, ...

### Poisson
- Initial jobs: t=0
- Future jobs: Exponential(λ) inter-arrivals

## Common Configurations

```python
# Small (debugging)
generate_fjsp_dataset(2, 2, 2, seed=42)

# Default
generate_fjsp_dataset(3, 4, 3, seed=42)

# Medium
generate_fjsp_dataset(5, 5, 4, seed=42)

# Large
generate_fjsp_dataset(10, 10, 5, seed=42)
```

## Using in Main Files

Replace this line in `backup_no_wait.py`, `proactive_sche.py`, or `reactive_scheduling.py`:

```python
# Uncomment to use generated data:
ENHANCED_JOBS_DATA, MACHINE_LIST = generate_fjsp_dataset(
    num_initial_jobs=3, 
    num_future_jobs=4, 
    total_num_machines=3, 
    seed=GLOBAL_SEED
)
```

## Files

- `utils.py` - Core generation functions
- `test_utils.py` - Basic examples
- `advanced_examples.py` - Research scenarios
- `DATASET_GENERATION_README.md` - Full documentation
- `DATASET_GENERATION_SUMMARY.md` - Implementation summary

## Key Functions

| Function | Purpose |
|----------|---------|
| `generate_fjsp_dataset()` | Create job/operation structure |
| `generate_arrival_times()` | Create arrival schedule |
| `print_dataset_info()` | Display dataset details |

## Tips

1. **Reproducibility**: Use same seed → same dataset
2. **Variety**: Use different seeds → different instances
3. **Testing**: Use `seed >= 1000` for test sets
4. **Training**: Use `seed < 1000` for training sets
5. **Comparison**: Fix seed when comparing algorithms
