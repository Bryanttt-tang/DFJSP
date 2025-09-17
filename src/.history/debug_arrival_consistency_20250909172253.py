"""
Debug script to check arrival time consistency between methods
"""
import numpy as np
import random
import collections

# Job data
ENHANCED_JOBS_DATA = collections.OrderedDict({
    0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}, {'proc_times': {'M0': 2}}],
    1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M1': 3, 'M2': 4}}],
    2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M0': 3, 'M1': 4}}, {'proc_times': {'M2': 7}}],
    3: [{'proc_times': {'M1': 8}}, {'proc_times': {'M2': 2}}, {'proc_times': {'M0': 5, 'M1': 6}}],
    4: [{'proc_times': {'M0': 6, 'M1': 9}}, {'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M2': 6}}],
    5: [{'proc_times': {'M1': 5, 'M2': 8}}, {'proc_times': {'M0': 6}}, {'proc_times': {'M1': 4, 'M2': 3}}],
    6: [{'proc_times': {'M0': 7, 'M2': 4}}, {'proc_times': {'M0': 5, 'M1': 6}}, {'proc_times': {'M1': 3}}, {'proc_times': {'M0': 2, 'M2': 5}}]
})

MACHINE_LIST = ['M0', 'M1', 'M2']

def generate_test_scenarios(jobs_data, initial_jobs=[0, 1, 2], arrival_rate=0.08, num_scenarios=5):
    """Generate test scenarios with fixed Poisson arrival times."""
    print(f"\n--- Generating {num_scenarios} Test Scenarios ---")
    print(f"Initial jobs: {initial_jobs}, Arrival rate: {arrival_rate}")
    
    scenarios = []
    for i in range(num_scenarios):
        np.random.seed(100 + i)  # Fixed seeds for reproducibility
        arrival_times = {}
        
        # Initial jobs arrive at t=0
        for job_id in initial_jobs:
            arrival_times[job_id] = 0.0
        
        # Generate Poisson arrivals for remaining jobs
        remaining_jobs = [j for j in jobs_data.keys() if j not in initial_jobs]
        current_time = 0.0
        
        for job_id in remaining_jobs:
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival_time
            if current_time <= 200:  # Max simulation time
                arrival_times[job_id] = current_time
            else:
                arrival_times[job_id] = float('inf')  # Won't arrive
        
        scenarios.append({
            'scenario_id': i,
            'arrival_times': arrival_times,
            'seed': 100 + i
        })
        print(f"Scenario {i+1}: {arrival_times}")
    
    return scenarios

if __name__ == "__main__":
    print("Testing arrival time generation consistency")
    print("=" * 60)
    
    # Generate test scenarios
    scenarios = generate_test_scenarios(ENHANCED_JOBS_DATA, initial_jobs=[0, 1, 2], arrival_rate=0.05, num_scenarios=3)
    
    # Check first scenario
    first_scenario = scenarios[0]
    arrival_times = first_scenario['arrival_times']
    
    print(f"\nFirst scenario arrival times: {arrival_times}")
    
    # Filter jobs that actually arrive within simulation time
    valid_jobs = {job_id: arr_time for job_id, arr_time in arrival_times.items() if arr_time < 200}
    print(f"Jobs that arrive within simulation time: {valid_jobs}")
    
    # Check which jobs should be processed by each method
    initial_jobs = [job_id for job_id, arr_time in arrival_times.items() if arr_time == 0]
    dynamic_jobs = [job_id for job_id, arr_time in arrival_times.items() if 0 < arr_time < 200]
    late_jobs = [job_id for job_id, arr_time in arrival_times.items() if arr_time >= 200]
    
    print(f"\nJob categorization:")
    print(f"Initial jobs (t=0): {initial_jobs}")
    print(f"Dynamic jobs (Poisson): {dynamic_jobs}")
    print(f"Jobs that won't arrive: {late_jobs}")
    
    print(f"\nAll three methods should schedule jobs: {sorted(initial_jobs + dynamic_jobs)}")
    print(f"None should schedule jobs: {sorted(late_jobs)}")
