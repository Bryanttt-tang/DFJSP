"""
Test to verify proactive environment evaluation fix.
Tests that custom arrival times persist after reset.
"""

import numpy as np
from proactive_sche import ProactiveDynamicFJSPEnv
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    return env.action_masks()

# Simple test jobs
test_jobs = {
    0: [{'proc_times': {'M1': 10, 'M2': 15}}],
    1: [{'proc_times': {'M1': 12, 'M2': 10}}],
    2: [{'proc_times': {'M1': 8, 'M2': 12}}],
}

machines = ['M1', 'M2']

# Custom arrival times for evaluation
custom_arrivals = {
    0: 0.0,   # Initial job
    1: 5.0,   # Arrives at t=5
    2: 10.0,  # Arrives at t=10
}

print("=== Testing Proactive Evaluation Arrival Time Override ===\n")

# Create environment
env = ProactiveDynamicFJSPEnv(
    test_jobs,
    machines,
    initial_jobs=[0],  # Only J0 initially
    arrival_rate=0.1,
    reward_mode="makespan_increment",
    seed=42
)

env = ActionMasker(env, mask_fn)

# Reset (generates random arrivals)
obs, _ = env.reset()

print("BEFORE override:")
print(f"  Arrival times: {env.env.job_arrival_times}")
print(f"  Arrived jobs: {env.env.arrived_jobs}")

# Override with custom arrivals (simulating evaluation)
env.env.job_arrival_times = custom_arrivals.copy()
env.env.arrived_jobs = {job_id for job_id, arr_time in custom_arrivals.items() if arr_time <= 0}

print("\nAFTER override:")
print(f"  Arrival times: {env.env.job_arrival_times}")
print(f"  Arrived jobs: {env.env.arrived_jobs}")

# Verify arrivals are correct
if env.env.job_arrival_times == custom_arrivals:
    print("\n✅ Arrival times correctly set!")
else:
    print("\n❌ Arrival times mismatch!")
    print(f"Expected: {custom_arrivals}")
    print(f"Got: {env.env.job_arrival_times}")

# Test that jobs arrive at correct times
print("\n=== Testing Dynamic Arrivals ===")
step_count = 0
max_steps = 50

while step_count < max_steps:
    # Get action masks
    action_masks = env.action_masks()
    
    if not any(action_masks):
        print(f"\n❌ No valid actions at step {step_count}!")
        print(f"   Arrived: {env.env.arrived_jobs}, Completed: {env.env.completed_jobs}")
        break
    
    # Pick first valid scheduling action (not wait)
    scheduling_actions = action_masks[:env.env.wait_action_start]
    if any(scheduling_actions):
        action = np.where(scheduling_actions)[0][0]
        job_id = action // len(machines)
        machine_idx = action % len(machines)
        print(f"Step {step_count}: Schedule J{job_id} on {machines[machine_idx]}")
    else:
        # Only wait actions available
        wait_actions = action_masks[env.env.wait_action_start:]
        action = env.env.wait_action_start + np.where(wait_actions)[0][0]
        print(f"Step {step_count}: Wait (event_time={env.env.event_time:.2f})")
    
    obs, reward, done, truncated, info = env.step(action)
    step_count += 1
    
    # Report arrivals
    print(f"  → Arrived: {sorted(env.env.arrived_jobs)}, Completed: {sorted(env.env.completed_jobs)}, Event time: {env.env.event_time:.2f}")
    
    if done:
        print(f"\n✅ Episode completed at step {step_count}")
        break

# Check if all operations scheduled
total_ops_scheduled = sum(len(sched) for sched in env.env.machine_schedules.values())
total_expected = sum(len(ops) for ops in test_jobs.values())

print(f"\n=== FINAL RESULT ===")
print(f"Operations scheduled: {total_ops_scheduled}/{total_expected}")
print(f"Makespan: {env.env.current_makespan:.2f}")

if total_ops_scheduled == total_expected:
    print("✅ Complete schedule achieved!")
else:
    print(f"❌ Incomplete: missing {total_expected - total_ops_scheduled} operations")

print("\n" + "="*50)
