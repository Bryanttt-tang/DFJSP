"""
Advanced Examples: Using Dataset Generation Utilities for Research

This file demonstrates various research scenarios using the flexible dataset generation utilities.
"""

from utils import generate_fjsp_dataset, generate_arrival_times, print_dataset_info
import numpy as np


def example_1_parameter_study():
    """
    Example 1: Systematic parameter study
    Study how problem size affects algorithm performance
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Parameter Study - Impact of Problem Size")
    print("="*80)
    
    results = []
    
    # Test different problem sizes
    for num_machines in [2, 3, 4, 5]:
        for total_jobs in [5, 10, 15]:
            num_initial = total_jobs // 2
            num_future = total_jobs - num_initial
            
            jobs_data, machine_list = generate_fjsp_dataset(
                num_initial_jobs=num_initial,
                num_future_jobs=num_future,
                total_num_machines=num_machines,
                seed=42  # Fixed seed for fair comparison
            )
            
            # Calculate dataset complexity metrics
            total_ops = sum(len(ops) for ops in jobs_data.values())
            avg_ops_per_job = total_ops / len(jobs_data)
            
            results.append({
                'machines': num_machines,
                'jobs': total_jobs,
                'total_operations': total_ops,
                'avg_ops_per_job': avg_ops_per_job
            })
            
            print(f"\nConfig: {num_machines} machines, {total_jobs} jobs")
            print(f"  Total operations: {total_ops}")
            print(f"  Avg ops/job: {avg_ops_per_job:.2f}")
    
    return results


def example_2_arrival_comparison():
    """
    Example 2: Compare different arrival patterns
    Test deterministic vs. Poisson arrivals
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Arrival Pattern Comparison")
    print("="*80)
    
    # Same job structure, different arrival patterns
    jobs_data, machine_list = generate_fjsp_dataset(
        num_initial_jobs=3,
        num_future_jobs=4,
        total_num_machines=3,
        seed=123
    )
    
    # Deterministic arrivals
    arrivals_det = generate_arrival_times(
        num_initial_jobs=3,
        num_future_jobs=4,
        arrival_mode='deterministic',
        seed=123
    )
    
    # Poisson arrivals
    arrivals_poisson = generate_arrival_times(
        num_initial_jobs=3,
        num_future_jobs=4,
        arrival_mode='poisson',
        arrival_rate=0.08,
        seed=123
    )
    
    print("\nDeterministic Arrivals:")
    for job_id, arr_time in sorted(arrivals_det.items()):
        print(f"  Job {job_id}: t={arr_time}")
    
    print("\nPoisson Arrivals:")
    for job_id, arr_time in sorted(arrivals_poisson.items()):
        print(f"  Job {job_id}: t={arr_time:.2f}")
    
    return arrivals_det, arrivals_poisson


def example_3_multiple_scenarios():
    """
    Example 3: Generate multiple test scenarios
    Create diverse test cases for robust evaluation
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Generate Multiple Test Scenarios")
    print("="*80)
    
    num_scenarios = 5
    scenarios = []
    
    for i in range(num_scenarios):
        jobs_data, machine_list = generate_fjsp_dataset(
            num_initial_jobs=3,
            num_future_jobs=4,
            total_num_machines=3,
            seed=1000 + i  # Different seed for each scenario
        )
        
        arrival_times = generate_arrival_times(
            num_initial_jobs=3,
            num_future_jobs=4,
            arrival_mode='poisson',
            arrival_rate=0.08,
            seed=1000 + i
        )
        
        scenarios.append({
            'scenario_id': i,
            'jobs_data': jobs_data,
            'machine_list': machine_list,
            'arrival_times': arrival_times,
            'seed': 1000 + i
        })
        
        print(f"\nScenario {i+1}:")
        total_ops = sum(len(ops) for ops in jobs_data.values())
        print(f"  Total operations: {total_ops}")
        print(f"  Arrival times: {sorted(arrival_times.values())}")
    
    return scenarios


def example_4_complexity_levels():
    """
    Example 4: Generate datasets with different complexity levels
    Easy, Medium, Hard problem instances
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Complexity Levels")
    print("="*80)
    
    configs = {
        'Easy': {'jobs': (2, 2), 'machines': 2},
        'Medium': {'jobs': (5, 5), 'machines': 3},
        'Hard': {'jobs': (10, 10), 'machines': 5},
        'Very Hard': {'jobs': (15, 15), 'machines': 8}
    }
    
    datasets = {}
    
    for level, config in configs.items():
        num_initial, num_future = config['jobs']
        num_machines = config['machines']
        
        jobs_data, machine_list = generate_fjsp_dataset(
            num_initial_jobs=num_initial,
            num_future_jobs=num_future,
            total_num_machines=num_machines,
            seed=42
        )
        
        total_ops = sum(len(ops) for ops in jobs_data.values())
        total_jobs = len(jobs_data)
        
        datasets[level] = {
            'jobs_data': jobs_data,
            'machine_list': machine_list,
            'stats': {
                'total_jobs': total_jobs,
                'total_operations': total_ops,
                'machines': num_machines,
                'avg_ops_per_job': total_ops / total_jobs
            }
        }
        
        print(f"\n{level} Problem:")
        print(f"  Jobs: {total_jobs}")
        print(f"  Machines: {num_machines}")
        print(f"  Total operations: {total_ops}")
        print(f"  Avg ops/job: {total_ops/total_jobs:.2f}")
    
    return datasets


def example_5_training_vs_testing():
    """
    Example 5: Generate separate training and testing datasets
    Use different seeds to ensure no overlap
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Training vs Testing Datasets")
    print("="*80)
    
    # Training dataset (seed < 1000)
    print("\nGenerating Training Datasets (10 instances)...")
    training_sets = []
    for i in range(10):
        jobs_data, machine_list = generate_fjsp_dataset(
            num_initial_jobs=5,
            num_future_jobs=5,
            total_num_machines=3,
            seed=i
        )
        training_sets.append({
            'id': i,
            'jobs_data': jobs_data,
            'machine_list': machine_list
        })
    
    print(f"Created {len(training_sets)} training instances")
    
    # Testing dataset (seed >= 1000)
    print("\nGenerating Testing Datasets (5 instances)...")
    testing_sets = []
    for i in range(1000, 1005):
        jobs_data, machine_list = generate_fjsp_dataset(
            num_initial_jobs=5,
            num_future_jobs=5,
            total_num_machines=3,
            seed=i
        )
        testing_sets.append({
            'id': i,
            'jobs_data': jobs_data,
            'machine_list': machine_list
        })
    
    print(f"Created {len(testing_sets)} testing instances")
    
    return training_sets, testing_sets


def example_6_arrival_rate_sensitivity():
    """
    Example 6: Study sensitivity to arrival rate
    Generate scenarios with different arrival intensities
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Arrival Rate Sensitivity Analysis")
    print("="*80)
    
    # Same job structure
    jobs_data, machine_list = generate_fjsp_dataset(
        num_initial_jobs=5,
        num_future_jobs=5,
        total_num_machines=3,
        seed=42
    )
    
    # Different arrival rates
    arrival_rates = [0.05, 0.08, 0.1, 0.15, 0.2]
    
    arrival_scenarios = {}
    
    for rate in arrival_rates:
        arrivals = generate_arrival_times(
            num_initial_jobs=5,
            num_future_jobs=5,
            arrival_mode='poisson',
            arrival_rate=rate,
            seed=42
        )
        
        # Calculate statistics
        future_arrivals = [t for job_id, t in arrivals.items() if t > 0]
        if future_arrivals:
            mean_arrival = np.mean(future_arrivals)
            max_arrival = np.max(future_arrivals)
        else:
            mean_arrival = max_arrival = 0
        
        arrival_scenarios[rate] = {
            'arrivals': arrivals,
            'mean_arrival_time': mean_arrival,
            'max_arrival_time': max_arrival
        }
        
        print(f"\nArrival Rate λ={rate}:")
        print(f"  Mean arrival time: {mean_arrival:.2f}")
        print(f"  Max arrival time: {max_arrival:.2f}")
        print(f"  Arrival times: {sorted([t for t in arrivals.values() if t > 0])}")
    
    return arrival_scenarios


def example_7_custom_experiment():
    """
    Example 7: Custom experiment setup
    Combine different aspects for specific research question
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Custom Experiment - Machine Utilization Study")
    print("="*80)
    
    # Research Question: How does number of machines affect scheduling efficiency?
    # Fixed: job count, varied: machine count
    
    fixed_jobs = (5, 5)  # 5 initial, 5 future
    machine_counts = [2, 3, 4, 5, 6]
    
    experiment_data = []
    
    for num_machines in machine_counts:
        jobs_data, machine_list = generate_fjsp_dataset(
            num_initial_jobs=fixed_jobs[0],
            num_future_jobs=fixed_jobs[1],
            total_num_machines=num_machines,
            seed=42  # Same seed for fair comparison
        )
        
        # Calculate flexibility metric
        total_machine_options = 0
        total_operations = 0
        
        for job_ops in jobs_data.values():
            for op in job_ops:
                total_operations += 1
                total_machine_options += len(op['proc_times'])
        
        avg_flexibility = total_machine_options / total_operations if total_operations > 0 else 0
        
        experiment_data.append({
            'num_machines': num_machines,
            'total_operations': total_operations,
            'avg_machine_options_per_op': avg_flexibility,
            'flexibility_ratio': avg_flexibility / num_machines
        })
        
        print(f"\n{num_machines} Machines:")
        print(f"  Total operations: {total_operations}")
        print(f"  Avg machines/operation: {avg_flexibility:.2f}")
        print(f"  Flexibility ratio: {avg_flexibility/num_machines:.2f}")
    
    return experiment_data


def main():
    """
    Run all examples
    """
    print("="*80)
    print("ADVANCED DATASET GENERATION EXAMPLES FOR RESEARCH")
    print("="*80)
    
    # Run all examples
    results_1 = example_1_parameter_study()
    results_2 = example_2_arrival_comparison()
    results_3 = example_3_multiple_scenarios()
    results_4 = example_4_complexity_levels()
    results_5 = example_5_training_vs_testing()
    results_6 = example_6_arrival_rate_sensitivity()
    results_7 = example_7_custom_experiment()
    
    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80)
    print("\nThese examples demonstrate how to use the dataset generation utilities")
    print("for various research scenarios. Adapt them to your specific needs.")


if __name__ == "__main__":
    main()
