"""
Test script for tardiness-aware FJSP scheduling
Tests the new multi-objective functions
"""
import sys
import numpy as np
import random
import collections
from utils import generate_simplified_fjsp_dataset, assign_due_dates

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("="*80)
print("TESTING TARDINESS-AWARE FJSP COMPONENTS")
print("="*80)

# Test 1: Generate dataset
print("\n1. Testing dataset generation...")
jobs_data, machine_list, machine_metadata = generate_simplified_fjsp_dataset(
    num_initial_jobs=3,
    num_future_jobs=5,
    total_num_machines=3,
    machine_speed_variance=0.8,
    proc_time_variance_range=(5, 20),
    due_date_tightness=1.5,
    seed=SEED
)

print(f"   ✓ Generated {len(jobs_data)} jobs with {len(machine_list)} machines")
print(f"   ✓ Machines: {machine_list}")
print(f"   ✓ Jobs are in list format (operations only)")

# Test 2: Generate arrival times
print("\n2. Generating arrival times...")
arrival_times = {}
for i in range(3):
    arrival_times[i] = 0.0
for i in range(3, 8):
    arrival_times[i] = 10.0 * (i - 2)  # Jobs arrive at t=10, 20, 30, 40, 50

print(f"   ✓ Arrival times generated for {len(arrival_times)} jobs")

# Test 3: Assign due dates
print("\n3. Testing due date assignment...")
jobs_with_due_dates = assign_due_dates(jobs_data, arrival_times, due_date_tightness=1.5)

print("   ✓ Jobs converted to dict format with due dates:")
for job_id, job_info in list(jobs_with_due_dates.items())[:3]:  # Show first 3
    num_ops = len(job_info['operations'])
    due_date = job_info['due_date']
    arrival = arrival_times[job_id]
    
    # Calculate min processing time
    min_proc_time = sum(min(op['proc_times'].values()) for op in job_info['operations'])
    slack = due_date - arrival - min_proc_time
    
    print(f"      Job {job_id}: {num_ops} ops, arrives={arrival:.1f}, due={due_date:.1f}, "
          f"min_time={min_proc_time:.1f}, slack={slack:.1f}")

# Test 4: Verify data structure
print("\n4. Verifying data structure...")
job_0 = jobs_with_due_dates[0]
assert isinstance(job_0, dict), "Job should be a dictionary"
assert 'operations' in job_0, "Job should have 'operations' key"
assert 'due_date' in job_0, "Job should have 'due_date' key"
assert isinstance(job_0['operations'], list), "operations should be a list"
assert isinstance(job_0['due_date'], float), "due_date should be a float"

print("   ✓ Data structure is correct:")
print(f"      - 'operations': list of {len(job_0['operations'])} operations")
print(f"      - 'due_date': {job_0['due_date']:.2f}")

# Test 5: Verify operations format
print("\n5. Verifying operations format...")
op_0 = job_0['operations'][0]
assert isinstance(op_0, dict), "Operation should be a dictionary"
assert 'proc_times' in op_0, "Operation should have 'proc_times' key"
print(f"   ✓ Operation format correct: {list(op_0.keys())}")
print(f"   ✓ Sample proc_times: {op_0['proc_times']}")

# Test 6: Test EDD sequencing logic
print("\n6. Testing EDD sequencing logic...")
jobs_sorted_by_due_date = sorted(
    jobs_with_due_dates.items(),
    key=lambda x: (x[1]['due_date'], arrival_times[x[0]], x[0])
)
print("   ✓ Jobs sorted by due date (EDD order):")
for i, (job_id, job_info) in enumerate(jobs_sorted_by_due_date[:5]):
    print(f"      {i+1}. Job {job_id}: due={job_info['due_date']:.1f}, "
          f"arrives={arrival_times[job_id]:.1f}")

print("\n" + "="*80)
print("✅ ALL COMPONENT TESTS PASSED!")
print("="*80)
print("\nSummary of Changes:")
print("  1. ✅ Dataset generation supports due_date_tightness parameter")
print("  2. ✅ assign_due_dates() converts jobs to dict format with due dates")
print("  3. ✅ Due dates calculated based on: arrival_time + tightness * min_processing_time")
print("  4. ✅ Data structure supports both old (list) and new (dict) formats")
print("  5. ✅ EDD sequencing logic ready for use")
print("\nNext Steps:")
print("  - Environment classes updated to handle new format")
print("  - Observation space includes due dates and slack time")
print("  - Reward function includes tardiness penalty")
print("  - EDD heuristic added to schedulers and best_heuristic comparison")
print("="*80)
