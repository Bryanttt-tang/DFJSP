"""
Utility functions for Flexible Job Shop Scheduling Problem (FJSP)
"""
import collections
import random
import numpy as np


def generate_fjsp_dataset(num_initial_jobs, num_future_jobs, total_num_machines, seed=None):
    """
    Generate a Flexible Job Shop Problem dataset with random job configurations.
    
    Parameters:
    -----------
    num_initial_jobs : int
        Number of jobs available at time 0
    num_future_jobs : int
        Number of jobs that arrive dynamically
    total_num_machines : int
        Total number of machines available (M)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    jobs_data : collections.OrderedDict
        Dictionary mapping job_id to list of operations
        Each operation has 'proc_times' dict mapping machine to processing time
    machine_list : list
        List of machine names ['M0', 'M1', ..., 'M{total_num_machines-1}']
        
    Generation Rules:
    -----------------
    - Number of operations per job: Uniform[1, 5]
    - Number of available machines per operation: Uniform[1, M]
    - Processing time per operation-machine pair: Uniform[1, 50]
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    total_jobs = num_initial_jobs + num_future_jobs
    machine_list = [f'M{i}' for i in range(total_num_machines)]
    
    jobs_data = collections.OrderedDict()
    
    for job_id in range(total_jobs):
        # Number of operations for this job: Uniform[1, 5]
        num_operations = random.randint(1, 5)
        
        operations = []
        for op_idx in range(num_operations):
            # Number of available machines for this operation: Uniform[1, M]
            num_available_machines = random.randint(1, total_num_machines)
            
            # Randomly select which machines are available
            available_machines = random.sample(machine_list, num_available_machines)
            
            # Generate processing times for each available machine: Uniform[1, 10]
            proc_times = {}
            for machine in available_machines:
                proc_time = random.randint(1, 50)
                proc_times[machine] = proc_time
            
            operations.append({'proc_times': proc_times})
        
        jobs_data[job_id] = operations
    
    return jobs_data, machine_list


def generate_realistic_fjsp_dataset(num_initial_jobs, num_future_jobs, total_num_machines, 
                                    job_type_distribution=None, machine_speed_variance=0.5,
                                    seed=None):
    """
    Generate a REALISTIC Flexible Job Shop Problem dataset with:
    1. Job categorization (short/moderate/long jobs)
    2. Machine heterogeneity (fast/medium/slow machines)
    3. Structured processing time relationships
    
    This creates strategic depth: waiting for fast machines vs scheduling on slow machines.
    
    Parameters:
    -----------
    num_initial_jobs : int
        Number of jobs available at time 0
    num_future_jobs : int
        Number of jobs that arrive dynamically
    total_num_machines : int
        Total number of machines (will be categorized as fast/medium/slow)
    job_type_distribution : dict, optional
        Distribution of job types: {'short': 0.5, 'moderate': 0.3, 'long': 0.2}
        If None, uses default distribution
    machine_speed_variance : float
        How much machines differ in speed (0=identical, 1=very different)
        Controls the gap between fast and slow machines
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    jobs_data : collections.OrderedDict
        Dictionary mapping job_id to list of operations
        Each job has metadata: {'type': 'short'|'moderate'|'long', 'operations': [...]}
    machine_list : list
        List of machine names
    machine_metadata : dict
        Machine information: {machine_name: {'speed_factor': float, 'category': str}}
        
    Job Categories:
    ---------------
    - SHORT jobs: 1-2 operations, base processing time 5-15 per operation
    - MODERATE jobs: 2-4 operations, base processing time 15-30 per operation
    - LONG jobs: 3-5 operations, base processing time 30-50 per operation
    
    Machine Categories:
    -------------------
    - FAST machines: speed_factor = 0.6-0.8 (40% faster than baseline)
    - MEDIUM machines: speed_factor = 0.9-1.1 (baseline speed)
    - SLOW machines: speed_factor = 1.2-1.5 (50% slower than baseline)
    
    Key Design Principle:
    ---------------------
    Processing time = base_time * machine_speed_factor
    
    This creates meaningful wait decisions:
    - Long job on slow machine: 50 * 1.5 = 75 time units
    - Long job on fast machine: 50 * 0.7 = 35 time units
    - Difference: 40 time units! (Worth waiting for fast machine!)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Default job type distribution
    if job_type_distribution is None:
        job_type_distribution = {
            'short': 0.5,      # 50% short jobs
            'moderate': 0.3,   # 30% moderate jobs
            'long': 0.2        # 20% long jobs
        }
    
    total_jobs = num_initial_jobs + num_future_jobs
    
    # Step 1: Create machine categorization (fast/medium/slow)
    machine_list = [f'M{i}' for i in range(total_num_machines)]
    machine_metadata = _create_machine_categories(machine_list, machine_speed_variance)
    
    # Step 2: Determine job types for all jobs
    job_types = _assign_job_types(total_jobs, job_type_distribution)
    
    # Step 3: Generate jobs with realistic processing times
    jobs_data = collections.OrderedDict()
    
    for job_id in range(total_jobs):
        job_type = job_types[job_id]
        
        # Determine number of operations based on job type
        if job_type == 'short':
            num_operations = random.randint(1, 2)
            base_proc_time_range = (5, 15)
        elif job_type == 'moderate':
            num_operations = random.randint(2, 4)
            base_proc_time_range = (15, 30)
        else:  # long
            num_operations = random.randint(3, 5)
            base_proc_time_range = (30, 50)
        
        operations = []
        for op_idx in range(num_operations):
            # Sample base processing time for this operation
            base_proc_time = random.randint(*base_proc_time_range)
            
            # Determine which machines can process this operation
            # Not all machines can process all operations (adds realism)
            num_compatible_machines = random.randint(
                max(2, total_num_machines // 2),  # At least half machines, minimum 2
                total_num_machines
            )
            compatible_machines = random.sample(machine_list, num_compatible_machines)
            
            # Calculate actual processing times based on machine speed factors
            proc_times = {}
            for machine in compatible_machines:
                speed_factor = machine_metadata[machine]['speed_factor']
                # Actual time = base_time * speed_factor
                # Add small random noise (±10%) for realism
                noise = random.uniform(0.9, 1.1)
                actual_proc_time = int(base_proc_time * speed_factor * noise)
                actual_proc_time = max(1, actual_proc_time)  # Ensure at least 1
                proc_times[machine] = actual_proc_time
            
            operations.append({'proc_times': proc_times})
        
        # Store job with metadata
        jobs_data[job_id] = {
            'type': job_type,
            'operations': operations
        }
    
    return jobs_data, machine_list, machine_metadata


def _create_machine_categories(machine_list, speed_variance):
    """
    Categorize machines into fast/medium/slow with different speed factors.
    
    Speed variance controls how different the machines are:
    - 0.0: All machines identical (no heterogeneity)
    - 0.5: Moderate differences (realistic)
    - 1.0: Large differences (high heterogeneity)
    """
    num_machines = len(machine_list)
    machine_metadata = {}
    
    # Distribute machines across categories
    num_fast = max(1, num_machines // 4)      # 25% fast
    num_slow = max(1, num_machines // 4)      # 25% slow
    num_medium = num_machines - num_fast - num_slow  # 50% medium
    
    # Shuffle to randomize which machines are fast/slow
    shuffled_machines = machine_list.copy()
    random.shuffle(shuffled_machines)
    
    machine_idx = 0
    
    # Assign FAST machines
    for i in range(num_fast):
        machine = shuffled_machines[machine_idx]
        # Fast machines: 0.6 to 0.8 (20-40% faster)
        speed_factor = random.uniform(
            1.0 - 0.4 * speed_variance,  # At variance=0.5: 0.8
            1.0 - 0.2 * speed_variance   # At variance=0.5: 0.9
        )
        machine_metadata[machine] = {
            'speed_factor': speed_factor,
            'category': 'fast'
        }
        machine_idx += 1
    
    # Assign MEDIUM machines
    for i in range(num_medium):
        machine = shuffled_machines[machine_idx]
        # Medium machines: 0.9 to 1.1 (baseline)
        speed_factor = random.uniform(
            1.0 - 0.1 * speed_variance,
            1.0 + 0.1 * speed_variance
        )
        machine_metadata[machine] = {
            'speed_factor': speed_factor,
            'category': 'medium'
        }
        machine_idx += 1
    
    # Assign SLOW machines
    for i in range(num_slow):
        machine = shuffled_machines[machine_idx]
        # Slow machines: 1.2 to 1.5 (20-50% slower)
        speed_factor = random.uniform(
            1.0 + 0.2 * speed_variance,  # At variance=0.5: 1.1
            1.0 + 0.5 * speed_variance   # At variance=0.5: 1.25
        )
        machine_metadata[machine] = {
            'speed_factor': speed_factor,
            'category': 'slow'
        }
        machine_idx += 1
    
    return machine_metadata


def _assign_job_types(total_jobs, distribution):
    """
    Assign job types to all jobs based on distribution.
    
    Returns list of job types in order: ['short', 'short', 'long', 'moderate', ...]
    """
    job_types = []
    
    # Calculate counts
    num_short = int(total_jobs * distribution['short'])
    num_moderate = int(total_jobs * distribution['moderate'])
    num_long = total_jobs - num_short - num_moderate  # Remainder
    
    # Create list with all types
    job_types.extend(['short'] * num_short)
    job_types.extend(['moderate'] * num_moderate)
    job_types.extend(['long'] * num_long)
    
    # Shuffle to randomize order (but we'll use patterns during arrival generation)
    random.shuffle(job_types)
    
    return job_types


def generate_arrival_times(num_initial_jobs, num_future_jobs, arrival_mode='deterministic', 
                           arrival_rate=0.08, seed=None):
    """
    Generate job arrival times for dynamic scheduling.
    
    Parameters:
    -----------
    num_initial_jobs : int
        Number of jobs available at time 0
    num_future_jobs : int
        Number of jobs arriving dynamically
    arrival_mode : str
        'deterministic' - evenly spaced arrivals
        'poisson' - Poisson distributed arrivals
        'custom' - provide custom times
    arrival_rate : float
        Rate parameter for Poisson arrivals (used only if arrival_mode='poisson')
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    arrival_times : dict
        Dictionary mapping job_id to arrival time
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    total_jobs = num_initial_jobs + num_future_jobs
    arrival_times = {}
    
    # Initial jobs arrive at time 0
    for job_id in range(num_initial_jobs):
        arrival_times[job_id] = 0.0
    
    if arrival_mode == 'deterministic':
        # Evenly spaced arrivals
        if num_future_jobs > 0:
            spacing = 4  # Default spacing of 4 time units
            for i, job_id in enumerate(range(num_initial_jobs, total_jobs)):
                arrival_times[job_id] = (i + 1) * spacing
                
    elif arrival_mode == 'poisson':
        # Poisson distributed arrivals
        current_time = 0.0
        for job_id in range(num_initial_jobs, total_jobs):
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival_time
            arrival_times[job_id] = round(current_time)
    
    return arrival_times


def print_dataset_info(jobs_data, machine_list, arrival_times=None):
    """
    Print detailed information about the generated dataset.
    
    Parameters:
    -----------
    jobs_data : collections.OrderedDict
        Job data structure
    machine_list : list
        List of machine names
    arrival_times : dict, optional
        Job arrival times
    """
    print("\n" + "="*80)
    print("FLEXIBLE JOB SHOP PROBLEM DATASET")
    print("="*80)
    
    print(f"\nMachines: {len(machine_list)}")
    print(f"Machine List: {machine_list}")
    
    print(f"\nJobs: {len(jobs_data)}")
    total_operations = sum(len(ops) for ops in jobs_data.values())
    print(f"Total Operations: {total_operations}")
    
    if arrival_times:
        initial_jobs = sum(1 for t in arrival_times.values() if t == 0)
        future_jobs = len(arrival_times) - initial_jobs
        print(f"Initial Jobs (t=0): {initial_jobs}")
        print(f"Future Jobs: {future_jobs}")
    
    print("\n" + "-"*80)
    print("Job Details:")
    print("-"*80)
    
    for job_id, operations in jobs_data.items():
        arrival_str = f" (arrives at t={arrival_times[job_id]})" if arrival_times else ""
        print(f"\nJob {job_id}{arrival_str}:")
        print(f"  Operations: {len(operations)}")
        
        for op_idx, operation in enumerate(operations):
            proc_times = operation['proc_times']
            machines = sorted(proc_times.keys())
            print(f"    Op {op_idx}: {len(machines)} machines available")
            for machine in machines:
                print(f"      {machine}: {proc_times[machine]} time units")
    
    print("\n" + "="*80)

def print_dataset_table(jobs_data, machine_list):
    """Print generated dataset in a readable table format."""
    print("\n" + "="*100)
    print("GENERATED FJSP DATASET STRUCTURE")
    print("="*100)
    print(f"Total Jobs: {len(jobs_data)} | Machines: {machine_list}")
    print("="*100)
    
    # Calculate max operations for formatting
    max_ops = max(len(ops) for ops in jobs_data.values())
    
    # Print header
    header = f"{'Job':^6} | {'#Ops':^5} | "
    for i in range(max_ops):
        header += f"Operation {i} {' '*20}| "
    print(header)
    print("-"*100)
    
    # Print each job
    for job_id, operations in jobs_data.items():
        job_type = "INIT" if job_id < 3 else "FUTR"
        row = f"J{job_id} ({job_type}) | {len(operations):^5} | "
        
        for op_idx, operation in enumerate(operations):
            proc_times = operation['proc_times']
            # Format: M0:4, M1:6
            op_str = ", ".join([f"{m}:{t}" for m, t in sorted(proc_times.items())])
            row += f"{op_str:^28} | "
        
        # Fill empty columns
        for _ in range(max_ops - len(operations)):
            row += f"{'-':^28} | "
        
        print(row)
    
    print("="*100)
    
    # Print statistics
    total_ops = sum(len(ops) for ops in jobs_data.values())
    avg_ops = total_ops / len(jobs_data)
    
    all_machines_per_op = []
    all_proc_times = []
    for job_ops in jobs_data.values():
        for op in job_ops:
            all_machines_per_op.append(len(op['proc_times']))
            all_proc_times.extend(op['proc_times'].values())
    
    print(f"\nDataset Statistics:")
    print(f"  Total Operations: {total_ops}")
    print(f"  Avg Operations/Job: {avg_ops:.2f}")
    print(f"  Avg Machines/Operation: {np.mean(all_machines_per_op):.2f}")
    print(f"  Avg Processing Time: {np.mean(all_proc_times):.2f}")
    print(f"  Processing Time Range: [{min(all_proc_times)}, {max(all_proc_times)}]")
    print("="*100 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Example 1: Small dataset (3 initial, 4 future, 3 machines)")
    jobs_data, machine_list = generate_fjsp_dataset(
        num_initial_jobs=3,
        num_future_jobs=4,
        total_num_machines=3,
        seed=42
    )
    arrival_times = generate_arrival_times(
        num_initial_jobs=3,
        num_future_jobs=4,
        arrival_mode='deterministic',
        seed=42
    )
    print_dataset_info(jobs_data, machine_list, arrival_times)
    
    print("\n" + "="*80)
    print("\nExample 2: Larger dataset with Poisson arrivals")
    jobs_data2, machine_list2 = generate_fjsp_dataset(
        num_initial_jobs=5,
        num_future_jobs=5,
        total_num_machines=4,
        seed=123
    )
    arrival_times2 = generate_arrival_times(
        num_initial_jobs=5,
        num_future_jobs=5,
        arrival_mode='poisson',
        arrival_rate=0.08,
        seed=123
    )
    print_dataset_info(jobs_data2, machine_list2, arrival_times2)
