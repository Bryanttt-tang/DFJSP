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
    - Processing time per operation-machine pair: Uniform[1, 10]
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
                proc_time = random.randint(1, 10)
                proc_times[machine] = proc_time
            
            operations.append({'proc_times': proc_times})
        
        jobs_data[job_id] = operations
    
    return jobs_data, machine_list


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
