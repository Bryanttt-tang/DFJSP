# Flexible JSP Dataset Generation Guide

This guide explains how to generate custom Flexible Job Shop Problem (FJSP) datasets instead of using the predefined `ENHANCED_JOBS_DATA`.

## Quick Start

### Option 1: Use the Convenience Function (Recommended)

```python
from backup_no_wait import create_custom_problem_instance

# Generate a problem with 5 initial jobs + 2 dynamic jobs, 3 machines
jobs_data, machine_list, arrival_times = create_custom_problem_instance(
    num_initial_jobs=5,
    num_dynamic_jobs=2,
    num_machines=3,
    seed=42  # For reproducibility
)
```

### Option 2: Fine-Grained Control

```python
from backup_no_wait import generate_fjsp_dataset, generate_arrival_times

# Generate job structure
jobs_data, machine_list = generate_fjsp_dataset(
    num_jobs=7,
    num_machines=3,
    max_ops_per_job=5,
    min_ops_per_job=1,
    min_proc_time=1,
    max_proc_time=10,
    seed=42
)

# Generate arrival times
arrival_times = generate_arrival_times(
    num_initial_jobs=5,
    num_dynamic_jobs=2,
    arrival_rate=0.08,
    seed=42
)
```

## Function Reference

### `generate_fjsp_dataset()`

Generates the job structure with operations and processing times.

**Parameters:**
- `num_jobs` (int): Total number of jobs to generate
- `num_machines` (int): Total number of machines available
- `max_ops_per_job` (int, default=5): Maximum operations per job
- `min_ops_per_job` (int, default=1): Minimum operations per job
- `min_proc_time` (int, default=1): Minimum processing time
- `max_proc_time` (int, default=10): Maximum processing time
- `seed` (int, optional): Random seed for reproducibility

**Returns:**
- `jobs_data`: OrderedDict with structure:
  ```python
  {
      0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5}}],
      1: [{'proc_times': {'M1': 7, 'M2': 5}}, ...],
      ...
  }
  ```
- `machine_list`: List of machine names, e.g., `['M0', 'M1', 'M2']`

**Generation Rules:**
- Number of operations per job: Uniform random in `[min_ops_per_job, max_ops_per_job]`
- Number of available machines per operation: Uniform random in `[1, num_machines]`
- Processing time on each available machine: Uniform random in `[min_proc_time, max_proc_time]`

### `generate_arrival_times()`

Generates arrival times using a Poisson process for dynamic jobs.

**Parameters:**
- `num_initial_jobs` (int): Number of jobs arriving at t=0
- `num_dynamic_jobs` (int): Number of jobs arriving dynamically
- `arrival_rate` (float, default=0.08): Poisson arrival rate (λ)
- `max_time_horizon` (int, default=200): Maximum time horizon
- `seed` (int, optional): Random seed

**Returns:**
- `arrival_times`: Dictionary mapping job_id to arrival time
  ```python
  {0: 0, 1: 0, 2: 0, 3: 8, 4: 15, 5: 22, 6: 30}
  ```

**Arrival Process:**
- Initial jobs (0 to num_initial_jobs-1): Arrive at t=0
- Dynamic jobs: Inter-arrival times follow Exponential(1/arrival_rate)

### `create_custom_problem_instance()`

Convenience function that combines both generation steps.

**Parameters:**
- `num_initial_jobs` (int, default=5): Jobs arriving at t=0
- `num_dynamic_jobs` (int, default=2): Jobs arriving dynamically
- `num_machines` (int, default=3): Total number of machines
- `max_ops_per_job` (int, default=5): Maximum operations per job
- `arrival_rate` (float, default=0.08): Poisson arrival rate
- `seed` (int, optional): Random seed

**Returns:** `(jobs_data, machine_list, arrival_times)`

## Usage Examples

### Example 1: Small Test Instance

```python
# 5 jobs, 3 machines, simple operations
jobs, machines, arrivals = create_custom_problem_instance(
    num_initial_jobs=3,
    num_dynamic_jobs=2,
    num_machines=3,
    max_ops_per_job=3,
    seed=42
)
```

### Example 2: Large Benchmark Instance

```python
# 20 jobs, 10 machines, complex operations
jobs, machines, arrivals = create_custom_problem_instance(
    num_initial_jobs=10,
    num_dynamic_jobs=10,
    num_machines=10,
    max_ops_per_job=8,
    arrival_rate=0.05,  # Slower arrivals
    seed=12345
)
```

### Example 3: Static Scheduling (All Jobs at t=0)

```python
# Generate jobs
jobs, machines = generate_fjsp_dataset(
    num_jobs=10,
    num_machines=5,
    seed=42
)

# All jobs arrive at t=0 (static scheduling)
arrival_times = {job_id: 0 for job_id in range(len(jobs))}
```

### Example 4: Highly Dynamic Scenario

```python
# Fast arrivals, many dynamic jobs
jobs, machines, arrivals = create_custom_problem_instance(
    num_initial_jobs=2,     # Only 2 initial jobs
    num_dynamic_jobs=15,    # 15 dynamic jobs
    num_machines=5,
    arrival_rate=0.15,      # Fast arrivals
    seed=999
)
```

## Integration with Training

Replace the predefined dataset in your training code:

```python
# OLD: Using predefined dataset
# jobs_data = ENHANCED_JOBS_DATA
# machine_list = MACHINE_LIST
# arrival_times = DETERMINISTIC_ARRIVAL_TIMES

# NEW: Generate custom dataset
jobs_data, machine_list, arrival_times = create_custom_problem_instance(
    num_initial_jobs=5,
    num_dynamic_jobs=2,
    num_machines=3,
    seed=42
)

# Use with environment
env = PoissonDynamicFJSPEnv(
    jobs_data=jobs_data,
    machine_list=machine_list,
    initial_jobs=list(range(5)),  # First 5 jobs are initial
    arrival_rate=0.08
)
```

## Parameter Guidelines

### Problem Size
- **Small**: 5-10 jobs, 3-5 machines (quick testing)
- **Medium**: 10-20 jobs, 5-10 machines (standard benchmarks)
- **Large**: 20+ jobs, 10+ machines (scalability testing)

### Operations Complexity
- **Simple**: 1-3 operations per job
- **Moderate**: 3-5 operations per job
- **Complex**: 5-8 operations per job

### Machine Flexibility
- **Low**: num_machines = 2-3 (high bottleneck)
- **Medium**: num_machines = 5-7 (balanced)
- **High**: num_machines = 10+ (many options)

### Arrival Rate
- **Static**: arrival_rate = N/A (all jobs at t=0)
- **Slow dynamic**: arrival_rate = 0.02-0.05
- **Moderate dynamic**: arrival_rate = 0.05-0.1
- **Fast dynamic**: arrival_rate = 0.1-0.2

### Processing Times
- **Short**: 1-5 time units (quick operations)
- **Medium**: 1-10 time units (standard)
- **Long**: 5-20 time units (slow operations)

## Reproducibility

Always set a seed for reproducible experiments:

```python
# Same seed = same dataset
jobs1, _, _ = create_custom_problem_instance(seed=42)
jobs2, _, _ = create_custom_problem_instance(seed=42)
# jobs1 == jobs2 ✓

# Different seeds = different datasets
jobs3, _, _ = create_custom_problem_instance(seed=123)
# jobs1 != jobs3 ✓
```

## Running the Examples

```bash
# Run the example script
python example_generate_dataset.py

# This will show:
# - Example 1: Basic generation with separate functions
# - Example 2: Quick setup with convenience function
# - Example 3: Large instance generation
# - Example 4: Custom parameter control
```

## Validation

After generating a dataset, verify:

1. **Job structure**: Each job has ≥1 operation
2. **Machine compatibility**: Each operation has ≥1 compatible machine
3. **Arrival times**: Initial jobs at t=0, dynamic jobs > 0
4. **Processing times**: All times > 0

```python
# Quick validation
assert len(jobs_data) > 0, "No jobs generated"
assert all(len(ops) > 0 for ops in jobs_data.values()), "Empty job found"
assert all(arrival_times[i] == 0 for i in range(num_initial_jobs)), "Initial jobs should arrive at t=0"
print("✅ Dataset validation passed")
```

## Tips

1. **Start small**: Test with 5-7 jobs first
2. **Use seeds**: Always use seeds for experiments
3. **Check statistics**: Verify average operations, processing times match expectations
4. **Save datasets**: Pickle important instances for benchmarking
5. **Document parameters**: Record all parameters used for generated datasets

## Troubleshooting

**Problem**: Generated jobs too simple/complex
- **Solution**: Adjust `max_ops_per_job` and `min_ops_per_job`

**Problem**: Arrivals too clustered/spread out
- **Solution**: Adjust `arrival_rate` (higher = faster arrivals)

**Problem**: Not enough machine flexibility
- **Solution**: Increase `num_machines`

**Problem**: Processing times too uniform
- **Solution**: Increase range: `min_proc_time=1, max_proc_time=20`

## Next Steps

1. Run `example_generate_dataset.py` to see generation in action
2. Generate datasets matching your research requirements
3. Integrate with `PoissonDynamicFJSPEnv` or `PerfectKnowledgeFJSPEnv`
4. Train agents on custom datasets
5. Compare performance across different problem characteristics
