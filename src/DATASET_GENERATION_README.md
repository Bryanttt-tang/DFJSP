# FJSP Dataset Generation Utilities

## Overview

The `utils.py` module provides flexible utilities for generating Flexible Job Shop Problem (FJSP) datasets with customizable parameters. This replaces hardcoded datasets and allows for easy experimentation with different problem sizes and configurations.

## Features

- **Dynamic Dataset Generation**: Generate FJSP instances with configurable job counts, machine counts, and operation parameters
- **Flexible Arrival Patterns**: Support for both deterministic and Poisson-distributed job arrivals
- **Reproducibility**: Seed-based random generation for consistent results
- **Information Display**: Built-in functions to visualize dataset properties

## Functions

### 1. `generate_fjsp_dataset(num_initial_jobs, num_future_jobs, total_num_machines, seed=None)`

Generate a complete FJSP dataset with random job configurations.

**Parameters:**
- `num_initial_jobs` (int): Number of jobs available at time t=0
- `num_future_jobs` (int): Number of jobs that arrive dynamically during execution
- `total_num_machines` (int): Total number of machines available (M)
- `seed` (int, optional): Random seed for reproducibility

**Returns:**
- `jobs_data` (OrderedDict): Dictionary mapping job_id → list of operations
- `machine_list` (list): List of machine names ['M0', 'M1', ..., 'M{M-1}']

**Generation Rules:**
- Number of operations per job: Uniform[1, 5]
- Number of available machines per operation: Uniform[1, M]
- Processing time per operation-machine pair: Uniform[1, 10]

**Example:**
```python
from utils import generate_fjsp_dataset

# Generate a dataset with 5 initial jobs, 5 future jobs, and 3 machines
jobs_data, machine_list = generate_fjsp_dataset(
    num_initial_jobs=5,
    num_future_jobs=5,
    total_num_machines=3,
    seed=42
)
```

### 2. `generate_arrival_times(num_initial_jobs, num_future_jobs, arrival_mode='deterministic', arrival_rate=0.08, seed=None)`

Generate job arrival times for dynamic scheduling scenarios.

**Parameters:**
- `num_initial_jobs` (int): Number of jobs available at time 0
- `num_future_jobs` (int): Number of jobs arriving dynamically
- `arrival_mode` (str): 'deterministic' or 'poisson'
  - `'deterministic'`: Evenly spaced arrivals (spacing = 4 time units)
  - `'poisson'`: Poisson-distributed arrivals using exponential inter-arrival times
- `arrival_rate` (float): Rate parameter λ for Poisson arrivals (events per time unit)
- `seed` (int, optional): Random seed for reproducibility

**Returns:**
- `arrival_times` (dict): Dictionary mapping job_id → arrival time

**Example:**
```python
from utils import generate_arrival_times

# Deterministic arrivals
arrival_times_det = generate_arrival_times(
    num_initial_jobs=3,
    num_future_jobs=4,
    arrival_mode='deterministic',
    seed=42
)

# Poisson arrivals
arrival_times_poisson = generate_arrival_times(
    num_initial_jobs=3,
    num_future_jobs=4,
    arrival_mode='poisson',
    arrival_rate=0.08,
    seed=42
)
```

### 3. `print_dataset_info(jobs_data, machine_list, arrival_times=None)`

Display detailed information about a generated dataset.

**Parameters:**
- `jobs_data` (OrderedDict): Job data structure
- `machine_list` (list): List of machine names
- `arrival_times` (dict, optional): Job arrival times

**Example:**
```python
from utils import generate_fjsp_dataset, generate_arrival_times, print_dataset_info

jobs_data, machine_list = generate_fjsp_dataset(3, 4, 3, seed=42)
arrival_times = generate_arrival_times(3, 4, 'deterministic', seed=42)

print_dataset_info(jobs_data, machine_list, arrival_times)
```

## Usage in Existing Code

### Option 1: Replace Hardcoded Dataset (Recommended for Experiments)

```python
from utils import generate_fjsp_dataset, generate_arrival_times

# Replace ENHANCED_JOBS_DATA
ENHANCED_JOBS_DATA, MACHINE_LIST = generate_fjsp_dataset(
    num_initial_jobs=3,
    num_future_jobs=4,
    total_num_machines=3,
    seed=42  # Use GLOBAL_SEED for consistency
)

# Generate corresponding arrival times
DETERMINISTIC_ARRIVAL_TIMES = generate_arrival_times(
    num_initial_jobs=3,
    num_future_jobs=4,
    arrival_mode='deterministic',
    seed=42
)
```

### Option 2: Keep Predefined Dataset (Default)

The files (`backup_no_wait.py`, `proactive_sche.py`, `reactive_scheduling.py`) have been updated to import the utilities but keep the original hardcoded dataset by default. To switch to generated data, simply uncomment the generation code:

```python
# Uncomment these lines to use generated data:
# ENHANCED_JOBS_DATA, MACHINE_LIST = generate_fjsp_dataset(
#     num_initial_jobs=3, 
#     num_future_jobs=4, 
#     total_num_machines=3, 
#     seed=GLOBAL_SEED
# )
```

## Testing

Run the test script to see examples of different dataset configurations:

```bash
python test_utils.py
```

This will generate and display:
1. Small dataset (3 initial, 4 future, 3 machines)
2. Larger dataset (5 initial, 5 future, 4 machines)
3. Dataset with Poisson arrivals
4. Minimal dataset for debugging
5. Large-scale dataset (10 initial, 10 future, 5 machines)

## Dataset Structure

The generated `jobs_data` follows this structure:

```python
jobs_data = {
    0: [  # Job 0
        {'proc_times': {'M0': 5, 'M1': 3}},      # Operation 0
        {'proc_times': {'M1': 7, 'M2': 4}},      # Operation 1
        {'proc_times': {'M0': 2}},                # Operation 2
    ],
    1: [  # Job 1
        {'proc_times': {'M0': 6, 'M1': 8, 'M2': 4}},  # Operation 0
        {'proc_times': {'M2': 5}},                     # Operation 1
    ],
    # ... more jobs
}
```

Each operation specifies which machines can process it and the processing time on each machine.

## Dataset Properties

### Job Properties
- **Number of operations per job**: Uniform[1, 5]
  - Ensures variety in job complexity
  - Minimum 1 operation (simple jobs)
  - Maximum 5 operations (complex jobs)

### Operation Properties
- **Machine availability**: Uniform[1, M]
  - At least 1 machine can process each operation
  - Maximum M machines available (full flexibility)
  - Creates varying degrees of flexibility

- **Processing times**: Uniform[1, 10]
  - Integer processing times from 1 to 10 time units
  - No zero-time operations
  - Reasonable range for scheduling problems

### Arrival Patterns

#### Deterministic Arrivals
- Initial jobs (0 to `num_initial_jobs-1`) arrive at t=0
- Future jobs arrive at regular intervals (every 4 time units)
- Example for 3 initial, 4 future:
  - Jobs 0,1,2: t=0
  - Job 3: t=4
  - Job 4: t=8
  - Job 5: t=12
  - Job 6: t=16

#### Poisson Arrivals
- Initial jobs arrive at t=0
- Future jobs follow Poisson process with rate λ
- Inter-arrival times ~ Exponential(λ)
- More realistic for dynamic environments
- Introduces stochasticity in arrival patterns

## Examples

### Example 1: Generate Custom Dataset for Training

```python
from utils import generate_fjsp_dataset, generate_arrival_times

# Create a larger problem instance
jobs_data, machine_list = generate_fjsp_dataset(
    num_initial_jobs=10,
    num_future_jobs=10,
    total_num_machines=5,
    seed=12345
)

arrival_times = generate_arrival_times(
    num_initial_jobs=10,
    num_future_jobs=10,
    arrival_mode='poisson',
    arrival_rate=0.1,
    seed=12345
)

# Use with environment
env = PoissonDynamicFJSPEnv(
    jobs_data=jobs_data,
    machine_list=machine_list,
    initial_jobs=list(range(10)),
    arrival_rate=0.1
)
```

### Example 2: Generate Multiple Test Scenarios

```python
from utils import generate_fjsp_dataset, generate_arrival_times

test_scenarios = []
for i in range(10):
    jobs_data, machine_list = generate_fjsp_dataset(
        num_initial_jobs=5,
        num_future_jobs=5,
        total_num_machines=3,
        seed=1000 + i  # Different seed for each scenario
    )
    
    arrival_times = generate_arrival_times(
        num_initial_jobs=5,
        num_future_jobs=5,
        arrival_mode='poisson',
        arrival_rate=0.08,
        seed=1000 + i
    )
    
    test_scenarios.append({
        'scenario_id': i,
        'jobs_data': jobs_data,
        'machine_list': machine_list,
        'arrival_times': arrival_times,
        'seed': 1000 + i
    })
```

### Example 3: Systematic Parameter Study

```python
from utils import generate_fjsp_dataset

# Study impact of problem size
for num_machines in [2, 3, 4, 5]:
    for num_jobs in [5, 10, 15, 20]:
        jobs_data, machine_list = generate_fjsp_dataset(
            num_initial_jobs=num_jobs // 2,
            num_future_jobs=num_jobs // 2,
            total_num_machines=num_machines,
            seed=42
        )
        
        # Train or evaluate model with this configuration
        # ...
```

## Migration Guide

To migrate from hardcoded datasets to generated datasets:

1. **Import utilities** (already done in updated files):
   ```python
   from utils import generate_fjsp_dataset, generate_arrival_times, print_dataset_info
   ```

2. **Replace dataset definition**:
   ```python
   # Old way:
   # ENHANCED_JOBS_DATA = collections.OrderedDict({...})
   # MACHINE_LIST = ['M0', 'M1', 'M2']
   
   # New way:
   ENHANCED_JOBS_DATA, MACHINE_LIST = generate_fjsp_dataset(
       num_initial_jobs=3,
       num_future_jobs=4,
       total_num_machines=3,
       seed=GLOBAL_SEED
   )
   ```

3. **Generate arrival times** (if needed):
   ```python
   DETERMINISTIC_ARRIVAL_TIMES = generate_arrival_times(
       num_initial_jobs=3,
       num_future_jobs=4,
       arrival_mode='deterministic',
       seed=GLOBAL_SEED
   )
   ```

4. **Test the changes**:
   - Verify that your code runs without errors
   - Check that generated datasets have expected properties
   - Compare results with original hardcoded dataset

## Notes

- All random generation uses numpy and Python's random module
- Setting a seed ensures reproducible datasets across runs
- The same seed will produce the same dataset every time
- Different seeds produce different problem instances
- Use `print_dataset_info()` to verify generated datasets

## Benefits

1. **Flexibility**: Easy to generate datasets of any size
2. **Reproducibility**: Seed-based generation ensures consistent results
3. **Experimentation**: Quickly test different problem configurations
4. **Scalability**: Generate small problems for debugging or large ones for benchmarking
5. **Standardization**: Consistent generation rules across experiments
6. **Maintainability**: Single source of truth for dataset generation logic

## Future Enhancements

Potential additions to the utility functions:
- Custom operation count distributions
- Custom processing time distributions
- Precedence constraints between operations
- Machine groups or families
- Setup times between operations
- Due dates for jobs
- Job priorities or weights
- Export/import dataset formats (JSON, CSV)
