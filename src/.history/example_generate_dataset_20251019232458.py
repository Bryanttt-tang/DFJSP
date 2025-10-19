"""
Example script demonstrating how to generate custom FJSP datasets.

This shows how to use the new dataset generation functions to create
flexible job shop problems with custom parameters instead of using
predefined datasets.
"""

import collections
import numpy as np
import random

# Import the generation functions from backup_no_wait
from backup_no_wait import (
    generate_fjsp_dataset, 
    generate_arrival_times,
    create_custom_problem_instance
)


def print_dataset_summary(jobs_data, machine_list, arrival_times):
    """Print a summary of the generated dataset."""
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total Jobs: {len(jobs_data)}")
    print(f"Machines: {machine_list}")
    print(f"Total Operations: {sum(len(ops) for ops in jobs_data.values())}")
    print(f"\nArrival Times: {arrival_times}")
    
    print(f"\n{'Job Details:':-^70}")
    for job_id, operations in jobs_data.items():
        arrival = arrival_times.get(job_id, 0)
        print(f"\nJob {job_id} (arrives at t={arrival}):")
        for op_idx, op_data in enumerate(operations):
            proc_times = op_data['proc_times']
            machines_str = ', '.join([f"{m}: {t}" for m, t in proc_times.items()])
            print(f"  Operation {op_idx+1}: {{{machines_str}}}")
    print(f"{'='*70}\n")


def example_1_basic_generation():
    """Example 1: Basic dataset generation with separate functions."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Generation (Separate Functions)")
    print("="*70)
    
    # Set seed for reproducibility
    seed = 42
    
    # Generate jobs
    jobs_data, machine_list = generate_fjsp_dataset(
        num_jobs=7,           # Total of 7 jobs
        num_machines=3,       # 3 machines (M0, M1, M2)
        max_ops_per_job=4,    # Each job has 1-4 operations
        min_ops_per_job=2,    # At least 2 operations per job
        min_proc_time=1,      # Processing times: 1-10
        max_proc_time=10,
        seed=seed
    )
    
    # Generate arrival times (5 initial, 2 dynamic)
    arrival_times = generate_arrival_times(
        num_initial_jobs=5,
        num_dynamic_jobs=2,
        arrival_rate=0.08,
        seed=seed
    )
    
    print_dataset_summary(jobs_data, machine_list, arrival_times)
    
    return jobs_data, machine_list, arrival_times


def example_2_convenience_function():
    """Example 2: Using the convenience function for quick setup."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Quick Setup (Convenience Function)")
    print("="*70)
    
    # Create a complete problem instance with one function call
    jobs_data, machine_list, arrival_times = create_custom_problem_instance(
        num_initial_jobs=3,    # 3 jobs arrive at t=0
        num_dynamic_jobs=4,    # 4 jobs arrive dynamically
        num_machines=4,        # 4 machines
        max_ops_per_job=5,     # Up to 5 operations per job
        arrival_rate=0.1,      # Higher arrival rate
        seed=123
    )
    
    print_dataset_summary(jobs_data, machine_list, arrival_times)
    
    return jobs_data, machine_list, arrival_times


def example_3_large_instance():
    """Example 3: Generate a larger problem instance."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Large Instance (10 jobs, 5 machines)")
    print("="*70)
    
    jobs_data, machine_list, arrival_times = create_custom_problem_instance(
        num_initial_jobs=6,
        num_dynamic_jobs=4,
        num_machines=5,
        max_ops_per_job=6,
        arrival_rate=0.05,     # Lower arrival rate = jobs spread out more
        seed=999
    )
    
    # Just print summary statistics for large instance
    print(f"\nTotal Jobs: {len(jobs_data)}")
    print(f"Machines: {machine_list}")
    print(f"Total Operations: {sum(len(ops) for ops in jobs_data.values())}")
    print(f"Arrival Times: {arrival_times}")
    
    # Print first 3 jobs as example
    print(f"\nFirst 3 jobs (example):")
    for job_id in range(min(3, len(jobs_data))):
        operations = jobs_data[job_id]
        arrival = arrival_times.get(job_id, 0)
        print(f"\nJob {job_id} (arrives at t={arrival}): {len(operations)} operations")
        for op_idx, op_data in enumerate(operations[:2]):  # Show first 2 ops
            proc_times = op_data['proc_times']
            machines_str = ', '.join([f"{m}: {t}" for m, t in proc_times.items()])
            print(f"  Op {op_idx+1}: {{{machines_str}}}")
        if len(operations) > 2:
            print(f"  ... ({len(operations)-2} more operations)")


def example_4_custom_parameters():
    """Example 4: Fine-grained control with custom parameters."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Parameters (Specific Requirements)")
    print("="*70)
    
    # Scenario: Small machines (2), many jobs (8), short operations (1-5 time units)
    jobs_data, machine_list = generate_fjsp_dataset(
        num_jobs=8,
        num_machines=2,        # Only 2 machines (bottleneck scenario)
        max_ops_per_job=3,     # Max 3 operations per job
        min_ops_per_job=2,     # Min 2 operations per job
        min_proc_time=1,       # Short processing times
        max_proc_time=5,
        seed=555
    )
    
    # All jobs arrive at t=0 (static scheduling)
    arrival_times = {job_id: 0 for job_id in range(len(jobs_data))}
    
    print_dataset_summary(jobs_data, machine_list, arrival_times)


if __name__ == "__main__":
    print("\n" + "🔧"*35)
    print("FLEXIBLE JSP DATASET GENERATION EXAMPLES")
    print("🔧"*35)
    
    # Run all examples
    example_1_basic_generation()
    example_2_convenience_function()
    example_3_large_instance()
    example_4_custom_parameters()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Use generate_fjsp_dataset() for custom job structures")
    print("2. Use generate_arrival_times() for dynamic scheduling scenarios")
    print("3. Use create_custom_problem_instance() for quick, complete setup")
    print("4. Adjust parameters to match your research requirements:")
    print("   - num_jobs: Scale problem size")
    print("   - num_machines: Control resource constraints")
    print("   - max_ops_per_job: Job complexity")
    print("   - arrival_rate: Dynamic vs static scheduling")
    print("   - seed: Reproducibility")
    print("="*70 + "\n")
