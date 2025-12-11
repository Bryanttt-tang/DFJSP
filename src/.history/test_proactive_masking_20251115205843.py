"""
Test script to verify proactive environment action masking fix.
Simulates the scenario where all jobs arrived but some blocked on precedence.
"""

import numpy as np
from proactive_sche import ProactiveDynamicFJSPEnv
from sb3_contrib.common.wrappers import ActionMasker

# Simple test jobs
test_jobs = {
    0: [
        {'proc_times': {'M1': 10, 'M2': 15}},
        {'proc_times': {'M1': 8, 'M2': 12}},
    ],
    1: [
        {'proc_times': {'M1': 12, 'M2': 10}},
        {'proc_times': {'M1': 9, 'M2': 11}},
    ],
}

machines = ['M1', 'M2']

# Create environment with all jobs arriving at t=0
env = ProactiveDynamicFJSPEnv(
    test_jobs,
    machines,
    initial_jobs=[0, 1],  # Both arrive at t=0
    arrival_rate=0.1,
    reward_mode="makespan_increment",
    seed=42
)

# Test scenario
print("=== Testing Action Masking Fix ===")
print(f"Jobs: {list(test_jobs.keys())}")
print(f"Total operations: {sum(len(ops) for ops in test_jobs.values())}")

obs, info = env.reset()

# Simulate scheduling first operation of each job on same machine
print("\nStep 1: Schedule J0-O1 on M1")
action = 0 * len(machines) + 0  # job 0, machine M1
obs, reward, done, truncated, info = env.step(action)
print(f"  Reward: {reward:.2f}, Done: {done}")
print(f"  Completed jobs: {env.completed_jobs}")
print(f"  Job progress: {env.job_progress}")

print("\nStep 2: Schedule J1-O1 on M1")
action = 1 * len(machines) + 0  # job 1, machine M1
obs, reward, done, truncated, info = env.step(action)
print(f"  Reward: {reward:.2f}, Done: {done}")
print(f"  Completed jobs: {env.completed_jobs}")
print(f"  Job progress: {env.job_progress}")

# Now both J0-O2 and J1-O2 are ready but may be waiting for precedence or machines
print("\n=== CRITICAL TEST: Check action masks ===")
print("At this point:")
print(f"  - All jobs arrived: {len(env.arrived_jobs) == len(env.job_ids)}")
print(f"  - All jobs completed: {len(env.completed_jobs) == len(env.job_ids)}")
print(f"  - Arrived jobs: {env.arrived_jobs}")
print(f"  - Completed jobs: {env.completed_jobs}")

mask = env.action_masks()
print(f"\nAction mask summary:")
print(f"  Total actions: {len(mask)}")
print(f"  Valid actions: {np.sum(mask)}")
print(f"  Valid scheduling actions: {np.sum(mask[:env.wait_action_start])}")
print(f"  Valid wait actions: {np.sum(mask[env.wait_action_start:])}")

if np.sum(mask) == 0:
    print("\n❌ BUG STILL EXISTS: No valid actions!")
    print("   This would cause incomplete schedules!")
else:
    print("\n✅ FIX SUCCESSFUL: Valid actions available!")
    
    if np.sum(mask[env.wait_action_start:]) > 0:
        print("   Wait actions enabled - agent can wait for machines to free")
    
    # Try to complete the schedule
    step_count = 2
    max_steps = 20
    
    while not done and step_count < max_steps:
        mask = env.action_masks()
        if not np.any(mask):
            print(f"\n❌ No valid actions at step {step_count}!")
            break
        
        # Pick first valid action
        valid_actions = np.where(mask)[0]
        action = valid_actions[0]
        
        if action >= env.wait_action_start:
            wait_idx = action - env.wait_action_start
            wait_dur = env.wait_durations[wait_idx]
            print(f"\nStep {step_count + 1}: Wait {wait_dur} units")
        else:
            job_id = action // len(machines)
            machine_idx = action % len(machines)
            print(f"\nStep {step_count + 1}: Schedule J{job_id}-O{env.job_progress[job_id]+1} on {machines[machine_idx]}")
        
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Reward: {reward:.2f}, Done: {done}")
        print(f"  Completed: {env.completed_jobs}/{set(env.job_ids)}")
        
        step_count += 1
        
        if done:
            print(f"\n✅ Episode completed successfully!")
            break
    
    total_ops_scheduled = sum(len(sched) for sched in env.machine_schedules.values())
    total_expected = sum(len(ops) for ops in test_jobs.values())
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Operations scheduled: {total_ops_scheduled}/{total_expected}")
    
    if total_ops_scheduled == total_expected:
        print("✅ Complete schedule achieved!")
    else:
        print(f"❌ Incomplete schedule: missing {total_expected - total_ops_scheduled} operations")

print("\n" + "="*50)
