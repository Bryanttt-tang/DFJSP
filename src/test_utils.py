"""
Test script to demonstrate the usage of dataset generation utilities.
This shows how to generate flexible JSP datasets with different configurations.
"""

from utils import generate_fjsp_dataset, generate_arrival_times, print_dataset_info

def main():
    print("="*80)
    print("FJSP DATASET GENERATION - DEMONSTRATION")
    print("="*80)
    
    # Example 1: Small dataset matching the original configuration
    print("\n" + "="*80)
    print("EXAMPLE 1: Small Dataset (3 initial, 4 future jobs, 3 machines)")
    print("="*80)
    
    jobs_data_1, machine_list_1 = generate_fjsp_dataset(
        num_initial_jobs=3,
        num_future_jobs=4,
        total_num_machines=3,
        seed=42
    )
    
    arrival_times_1 = generate_arrival_times(
        num_initial_jobs=3,
        num_future_jobs=4,
        arrival_mode='deterministic',
        seed=42
    )
    
    print_dataset_info(jobs_data_1, machine_list_1, arrival_times_1)
    
    # Example 2: Larger dataset with more machines
    print("\n" + "="*80)
    print("EXAMPLE 2: Larger Dataset (5 initial, 5 future jobs, 4 machines)")
    print("="*80)
    
    jobs_data_2, machine_list_2 = generate_fjsp_dataset(
        num_initial_jobs=5,
        num_future_jobs=5,
        total_num_machines=4,
        seed=123
    )
    
    arrival_times_2 = generate_arrival_times(
        num_initial_jobs=5,
        num_future_jobs=5,
        arrival_mode='deterministic',
        seed=123
    )
    
    print_dataset_info(jobs_data_2, machine_list_2, arrival_times_2)
    
    # Example 3: Poisson arrivals
    print("\n" + "="*80)
    print("EXAMPLE 3: Dataset with Poisson Arrivals (4 initial, 6 future jobs, 3 machines)")
    print("="*80)
    
    jobs_data_3, machine_list_3 = generate_fjsp_dataset(
        num_initial_jobs=4,
        num_future_jobs=6,
        total_num_machines=3,
        seed=456
    )
    
    arrival_times_3 = generate_arrival_times(
        num_initial_jobs=4,
        num_future_jobs=6,
        arrival_mode='poisson',
        arrival_rate=0.08,
        seed=456
    )
    
    print_dataset_info(jobs_data_3, machine_list_3, arrival_times_3)
    
    # Example 4: Very small dataset for debugging
    print("\n" + "="*80)
    print("EXAMPLE 4: Minimal Dataset (2 initial, 2 future jobs, 2 machines)")
    print("="*80)
    
    jobs_data_4, machine_list_4 = generate_fjsp_dataset(
        num_initial_jobs=2,
        num_future_jobs=2,
        total_num_machines=2,
        seed=789
    )
    
    arrival_times_4 = generate_arrival_times(
        num_initial_jobs=2,
        num_future_jobs=2,
        arrival_mode='deterministic',
        seed=789
    )
    
    print_dataset_info(jobs_data_4, machine_list_4, arrival_times_4)
    
    # Example 5: Large-scale dataset
    print("\n" + "="*80)
    print("EXAMPLE 5: Large-scale Dataset (10 initial, 10 future jobs, 5 machines)")
    print("="*80)
    
    jobs_data_5, machine_list_5 = generate_fjsp_dataset(
        num_initial_jobs=10,
        num_future_jobs=10,
        total_num_machines=5,
        seed=999
    )
    
    arrival_times_5 = generate_arrival_times(
        num_initial_jobs=10,
        num_future_jobs=10,
        arrival_mode='poisson',
        arrival_rate=0.1,
        seed=999
    )
    
    print_dataset_info(jobs_data_5, machine_list_5, arrival_times_5)
    
    print("\n" + "="*80)
    print("HOW TO USE IN YOUR CODE:")
    print("="*80)
    print("""
# Import the utilities
from utils import generate_fjsp_dataset, generate_arrival_times

# Generate a dataset
jobs_data, machine_list = generate_fjsp_dataset(
    num_initial_jobs=5,      # Jobs available at time 0
    num_future_jobs=5,       # Jobs arriving dynamically
    total_num_machines=3,    # Number of machines
    seed=42                  # For reproducibility
)

# Generate arrival times
arrival_times = generate_arrival_times(
    num_initial_jobs=5,
    num_future_jobs=5,
    arrival_mode='deterministic',  # or 'poisson'
    arrival_rate=0.08,             # Only used for Poisson
    seed=42
)

# Use in your environment
env = PoissonDynamicFJSPEnv(
    jobs_data=jobs_data,
    machine_list=machine_list,
    initial_jobs=list(range(5)),  # First 5 jobs
    arrival_rate=0.08
)
    """)

if __name__ == "__main__":
    main()
