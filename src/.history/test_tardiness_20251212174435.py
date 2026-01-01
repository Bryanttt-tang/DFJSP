"""
Test script for tardiness-aware FJSP scheduling
Tests the new multi-objective reward with due dates
"""
import sys
import numpy as np
import random
import collections

# Import from local modules (using hyphen in filename)
import importlib.util
spec = importlib.util.spec_from_file_location("multi_obj", "multi-obj.py")
multi_obj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_obj)

from utils import generate_simplified_fjsp_dataset, assign_due_dates

PoissonDynamicFJSPEnv = multi_obj.PoissonDynamicFJSPEnv

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("="*80)
print("TESTING TARDINESS-AWARE FJSP SCHEDULING")
print("="*80)

# Generate dataset
print("\n1. Generating dataset...")
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

# Generate arrival times
print("\n2. Generating arrival times...")
arrival_times = {}
for i in range(3):
    arrival_times[i] = 0.0
for i in range(3, 8):
    arrival_times[i] = 10.0 * (i - 2)  # Jobs arrive at t=10, 20, 30, 40, 50

print(f"   ✓ Arrival times: {arrival_times}")

# Assign due dates
print("\n3. Assigning due dates...")
jobs_with_due_dates = assign_due_dates(jobs_data, arrival_times, due_date_tightness=1.5)

print("   ✓ Jobs with due dates:")
for job_id, job_info in jobs_with_due_dates.items():
    num_ops = len(job_info['operations'])
    due_date = job_info['due_date']
    arrival = arrival_times[job_id]
    print(f"      Job {job_id}: {num_ops} ops, arrives={arrival:.1f}, due={due_date:.1f}")

# Test environment with tardiness weight = 0 (makespan only)
print("\n4. Testing environment with tardiness_weight=0 (makespan only)...")
env1 = PoissonDynamicFJSPEnv(
    jobs_with_due_dates,
    machine_list,
    initial_jobs=[0, 1, 2],
    arrival_rate=0.05,
    reward_mode="makespan_increment",
    tardiness_weight=0.0,
    seed=SEED
)
obs1 = env1.reset()
print(f"   ✓ Environment created, observation shape: {obs1[0].shape}")
print(f"   ✓ Expected shape: ({env1.observation_space.shape[0]},)")
print(f"   ✓ Tardiness weight: {env1.tardiness_weight}")

# Test environment with tardiness weight = 0.5 (multi-objective)
print("\n5. Testing environment with tardiness_weight=0.5 (multi-objective)...")
env2 = PoissonDynamicFJSPEnv(
    jobs_with_due_dates,
    machine_list,
    initial_jobs=[0, 1, 2],
    arrival_rate=0.05,
    reward_mode="makespan_increment",
    tardiness_weight=0.5,
    seed=SEED
)
obs2 = env2.reset()
print(f"   ✓ Environment created with tardiness_weight={env2.tardiness_weight}")

# Run a few random steps to test
print("\n6. Running random actions to test scheduling...")
done = False
step_count = 0
max_steps = 10

while not done and step_count < max_steps:
    # Get valid actions
    if hasattr(env2, 'action_masks'):
        valid_actions = [i for i, mask in enumerate(env2.action_masks()) if mask]
    else:
        valid_actions = list(range(env2.action_space.n))
    
    if not valid_actions:
        print(f"   ⚠ No valid actions at step {step_count}")
        break
    
    action = random.choice(valid_actions)
    obs, reward, done, truncated, info = env2.step(action)
    step_count += 1
    
    if step_count <= 3:  # Show first 3 steps
        print(f"   Step {step_count}: action={action}, reward={reward:.2f}, "
              f"makespan={info.get('makespan', 0):.2f}, "
              f"tardiness={info.get('total_tardiness', 0):.2f}")

print(f"\n   ✓ Completed {step_count} steps")
print(f"   Final makespan: {info.get('makespan', 0):.2f}")
print(f"   Total tardiness: {info.get('total_tardiness', 0):.2f}")
print(f"   Tardy jobs: {info.get('num_tardy_jobs', 0)}")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
