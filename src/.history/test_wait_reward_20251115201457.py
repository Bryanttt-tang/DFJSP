"""
Test script to verify wait action reward calculation is correct.
"""

import numpy as np
import random
from proactive_sche import ProactiveDynamicFJSPEnv
from utils import generate_simplified_fjsp_dataset
from sb3_contrib.common.wrappers import ActionMasker

# Set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("="*80)
print("Testing Wait Action Reward Calculation")
print("="*80)

# Generate simple dataset
jobs_data, machine_list, machine_metadata = generate_simplified_fjsp_dataset(
    num_initial_jobs=2,
    num_future_jobs=2,
    total_num_machines=2,
    seed=SEED
)

print(f"\nDataset: {len(jobs_data)} jobs, {len(machine_list)} machines")

def mask_fn(env):
    return env.action_masks()

# Create environment
env = ProactiveDynamicFJSPEnv(
    jobs_data, machine_list,
    initial_jobs=2,
    arrival_rate=0.1,
    predictor_mode='mle',
    seed=SEED
)
env = ActionMasker(env, mask_fn)
obs, _ = env.reset()

print(f"\nInitial state:")
print(f"  Event time: {env.env.event_time:.2f}")
print(f"  Makespan: {env.env.current_makespan:.2f}")
print(f"  Machine end times: {env.env.machine_end_times}")
print(f"  Arrived jobs: {env.env.arrived_jobs}")

# Test Case 1: Schedule an operation (machines become busy)
print(f"\n{'='*80}")
print("TEST CASE 1: Schedule operation while machines idle")
print("="*80)

action_mask = env.action_masks()
scheduling_actions = [i for i in range(env.env.wait_action_start) if action_mask[i]]
if scheduling_actions:
    action = scheduling_actions[0]
    print(f"Action: Schedule job {action // len(machine_list)} on machine {action % len(machine_list)}")
    
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"\nAfter scheduling:")
    print(f"  Event time: {env.env.event_time:.2f}")
    print(f"  Makespan: {env.env.current_makespan:.2f}")
    print(f"  Machine end times: {env.env.machine_end_times}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Expected: reward = -makespan_increment = -{env.env.current_makespan:.4f}")

# Test Case 2: Wait while machines are BUSY
print(f"\n{'='*80}")
print("TEST CASE 2: Wait while machines are BUSY")
print("="*80)

prev_makespan = env.env.current_makespan
prev_event_time = env.env.event_time
max_machine_time = max(env.env.machine_end_times.values())

action_mask = env.action_masks()
wait_actions = [i for i in range(env.env.wait_action_start, env.action_space.n) if action_mask[i]]
if wait_actions:
    action = wait_actions[0]  # First wait action (duration=1)
    wait_idx = action - env.env.wait_action_start
    wait_duration = env.env.wait_durations[wait_idx]
    
    print(f"Action: Wait for {wait_duration} time units")
    print(f"\nBefore wait:")
    print(f"  Event time: {prev_event_time:.2f}")
    print(f"  Makespan: {prev_makespan:.2f}")
    print(f"  Max machine time: {max_machine_time:.2f}")
    print(f"  Machines are: {'BUSY' if max_machine_time > prev_event_time else 'IDLE'}")
    
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"\nAfter wait:")
    print(f"  Event time: {env.env.event_time:.2f}")
    print(f"  Makespan: {env.env.current_makespan:.2f}")
    print(f"  Max machine time: {max(env.env.machine_end_times.values()):.2f}")
    print(f"  Reward: {reward:.4f}")
    
    # Analyze reward
    if max_machine_time > prev_event_time:
        print(f"\n✓ CORRECT: Machines were BUSY during wait")
        print(f"  Makespan should stay at max_machine_time = {max_machine_time:.2f}")
        print(f"  Reward should be near 0 (no makespan increase)")
        print(f"  Actual reward: {reward:.4f}")
        if abs(reward) < 0.1:
            print(f"  ✅ PASS: Wait while busy has minimal penalty")
        else:
            print(f"  ❌ FAIL: Wait while busy should have ~0 reward!")
    else:
        print(f"\n⚠️  Machines were IDLE during wait")
        print(f"  Makespan increases from {prev_makespan:.2f} to {env.env.event_time:.2f}")
        print(f"  Reward should be -{(env.env.event_time - prev_makespan):.4f}")

# Test Case 3: Wait while machines are IDLE (bad!)
print(f"\n{'='*80}")
print("TEST CASE 3: Wait while machines are IDLE (should be heavily penalized)")
print("="*80)

# Force machines to be idle
env.env.machine_end_times = {m: env.env.event_time - 1 for m in env.env.machines}
prev_makespan = env.env.current_makespan
prev_event_time = env.env.event_time

action_mask = env.action_masks()
wait_actions = [i for i in range(env.env.wait_action_start, env.action_space.n) if action_mask[i]]
if wait_actions:
    action = wait_actions[1]  # Second wait action (duration=2)
    wait_idx = action - env.env.wait_action_start
    wait_duration = env.env.wait_durations[wait_idx]
    
    print(f"Action: Wait for {wait_duration} time units")
    print(f"\nBefore wait:")
    print(f"  Event time: {prev_event_time:.2f}")
    print(f"  Makespan: {prev_makespan:.2f}")
    print(f"  Machine end times: {env.env.machine_end_times}")
    print(f"  All machines IDLE: {all(t <= prev_event_time for t in env.env.machine_end_times.values())}")
    
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"\nAfter wait:")
    print(f"  Event time: {env.env.event_time:.2f}")
    print(f"  Makespan: {env.env.current_makespan:.2f}")
    print(f"  Reward: {reward:.4f}")
    
    expected_penalty = -(env.env.event_time - prev_event_time)
    print(f"\n✓ Machines were IDLE - wasting time!")
    print(f"  Expected penalty: ~{expected_penalty:.4f}")
    print(f"  Actual reward: {reward:.4f}")
    
    if reward < -1.0:
        print(f"  ✅ PASS: Idle wait is heavily penalized")
    else:
        print(f"  ⚠️  WARNING: Idle wait penalty might be too small")

print("\n" + "="*80)
print("Summary:")
print("="*80)
print("Key insight:")
print("  - Wait while machines BUSY → reward ≈ 0 (good strategic wait)")
print("  - Wait while machines IDLE → reward << 0 (bad, wasting time)")
print("  - This teaches agent to wait strategically for fast machines!")
print("="*80)
