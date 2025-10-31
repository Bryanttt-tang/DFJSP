"""
Verification script to check Perfect Knowledge RL observation space and normalization.
Run this BEFORE training to ensure everything is correct.
"""

import numpy as np
import sys
sys.path.append('/Users/tanu/Desktop/PhD/Scheduling/src')

from proactive_sche import (
    PerfectKnowledgeFJSPEnv, 
    ENHANCED_JOBS_DATA, 
    MACHINE_LIST,
    ActionMasker,
    mask_fn
)

print("="*80)
print("PERFECT KNOWLEDGE RL - OBSERVATION SPACE VERIFICATION")
print("="*80)

# Create test arrival times
arrival_times = {i: i * 4 for i in range(len(ENHANCED_JOBS_DATA))}
print(f"\n📋 Test Configuration:")
print(f"   Jobs: {len(ENHANCED_JOBS_DATA)}")
print(f"   Machines: {len(MACHINE_LIST)}")
print(f"   Arrival times: {arrival_times}")

# Create environment
print(f"\n🏗️  Creating environment...")
env = PerfectKnowledgeFJSPEnv(
    ENHANCED_JOBS_DATA, 
    MACHINE_LIST, 
    arrival_times,
    reward_mode="makespan_increment"
)
env = ActionMasker(env, mask_fn)

# Reset and get initial observation
print(f"\n🔄 Resetting environment...")
obs, info = env.reset()

print(f"\n✅ Observation Space Check:")
print(f"   Expected shape: ({env.observation_space.shape[0]},)")
print(f"   Actual shape: {obs.shape}")
print(f"   Match: {'✅ YES' if obs.shape[0] == env.observation_space.shape[0] else '❌ NO'}")

# Calculate expected size
num_jobs = len(ENHANCED_JOBS_DATA)
num_machines = len(MACHINE_LIST)
expected_size = (
    num_jobs +                          # Job ready times
    num_jobs +                          # Job progress
    num_machines +                      # Machine free times
    num_jobs * num_machines +          # Processing times (CRITICAL!)
    num_jobs +                          # Arrival times
    1                                   # Current makespan
)
print(f"\n📐 Expected Size Breakdown:")
print(f"   Job ready times:      {num_jobs}")
print(f"   Job progress:         {num_jobs}")
print(f"   Machine free times:   {num_machines}")
print(f"   Processing times:     {num_jobs * num_machines} ⭐ CRITICAL COMPONENT")
print(f"   Arrival times:        {num_jobs}")
print(f"   Current makespan:     1")
print(f"   ------------------------")
print(f"   Total:                {expected_size}")
print(f"   Actual:               {obs.shape[0]}")
print(f"   Match: {'✅ YES' if obs.shape[0] == expected_size else '❌ NO'}")

# Check observation values
print(f"\n📊 Observation Value Ranges:")
print(f"   Min value: {obs.min():.6f} (should be 0.0)")
print(f"   Max value: {obs.max():.6f} (should be ≤ 1.0)")
print(f"   Contains NaN: {np.any(np.isnan(obs))}")
print(f"   Contains Inf: {np.any(np.isinf(obs))}")

# Extract and analyze observation components
idx = 0

# Job ready times
job_ready_times = obs[idx:idx+num_jobs]
idx += num_jobs
print(f"\n1️⃣  Job Ready Times:")
print(f"   Range: [{job_ready_times.min():.4f}, {job_ready_times.max():.4f}]")
print(f"   Sample: {job_ready_times[:5]}")

# Job progress
job_progress = obs[idx:idx+num_jobs]
idx += num_jobs
print(f"\n2️⃣  Job Progress:")
print(f"   Range: [{job_progress.min():.4f}, {job_progress.max():.4f}]")
print(f"   Sample: {job_progress[:5]}")
print(f"   All zeros (initial): {'✅ YES' if np.allclose(job_progress, 0.0) else '❌ NO'}")

# Machine free times
machine_free = obs[idx:idx+num_machines]
idx += num_machines
print(f"\n3️⃣  Machine Free Times:")
print(f"   Range: [{machine_free.min():.4f}, {machine_free.max():.4f}]")
print(f"   Values: {machine_free}")
print(f"   All zeros (initial): {'✅ YES' if np.allclose(machine_free, 0.0) else '❌ NO'}")

# Processing times (CRITICAL!)
proc_times = obs[idx:idx+num_jobs*num_machines]
idx += num_jobs * num_machines
print(f"\n4️⃣  Processing Times ⭐ CRITICAL:")
print(f"   Range: [{proc_times.min():.4f}, {proc_times.max():.4f}]")
print(f"   Non-zero count: {np.count_nonzero(proc_times)}")
print(f"   Sample (first 10): {proc_times[:10]}")
if proc_times.max() > 0:
    print(f"   ✅ Processing times PRESENT in observation")
else:
    print(f"   ❌ WARNING: All processing times are ZERO!")

# Arrival times
arrival_obs = obs[idx:idx+num_jobs]
idx += num_jobs
print(f"\n5️⃣  Arrival Times:")
print(f"   Range: [{arrival_obs.min():.4f}, {arrival_obs.max():.4f}]")
print(f"   Sample: {arrival_obs[:5]}")

# Current makespan
makespan = obs[idx]
print(f"\n6️⃣  Current Makespan:")
print(f"   Value: {makespan:.4f}")
print(f"   Zero (initial): {'✅ YES' if makespan == 0.0 else '❌ NO'}")

# Normalization consistency check
print(f"\n🔍 NORMALIZATION CONSISTENCY CHECK:")
print(f"   max_time_horizon: {env.max_time_horizon}")
print(f"   max_proc_time: {env.max_proc_time}")

# Check if processing times use same normalization as other times
# Get a sample processing time
for job_id in env.job_ids:
    if env.next_operation[job_id] < len(env.jobs[job_id]):
        op = env.jobs[job_id][env.next_operation[job_id]]
        for machine, proc_time in op['proc_times'].items():
            # Calculate what normalization was used
            job_idx = env.job_ids.index(job_id)
            machine_idx = env.machines.index(machine)
            obs_idx = num_jobs + num_jobs + num_machines + job_idx * num_machines + machine_idx
            
            observed_value = obs[obs_idx]
            
            # Try both normalizations
            norm_by_horizon = proc_time / env.max_time_horizon
            norm_by_max_proc = proc_time / env.max_proc_time
            
            print(f"\n   Example: Job {job_id}, Machine {machine}, Proc Time {proc_time}")
            print(f"   Observed value: {observed_value:.6f}")
            print(f"   If normalized by max_time_horizon ({env.max_time_horizon}): {norm_by_horizon:.6f}")
            print(f"   If normalized by max_proc_time ({env.max_proc_time}): {norm_by_max_proc:.6f}")
            
            if abs(observed_value - norm_by_horizon) < 1e-6:
                print(f"   ✅ Using max_time_horizon normalization (CORRECT!)")
            elif abs(observed_value - norm_by_max_proc) < 1e-6:
                print(f"   ❌ Using max_proc_time normalization (WRONG!)")
            else:
                print(f"   ⚠️  Unknown normalization!")
            
            break  # Just check one example
        break

# Test a few steps
print(f"\n🎮 Testing Environment Steps:")
for step in range(3):
    action_masks = env.action_masks()
    valid_actions = [i for i, mask in enumerate(action_masks) if mask]
    
    if not valid_actions:
        print(f"   Step {step}: No valid actions available")
        break
    
    action = np.random.choice(valid_actions)
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"   Step {step}: action={action}, reward={reward:.2f}, makespan={info['makespan']:.2f}")
    print(f"             obs_range=[{obs.min():.4f}, {obs.max():.4f}], valid={not (np.any(np.isnan(obs)) or np.any(np.isinf(obs)))}")
    
    if done:
        print(f"   Episode finished!")
        break

print(f"\n{'='*80}")
print(f"VERIFICATION COMPLETE")
print(f"{'='*80}")

# Final checklist
print(f"\n✅ FINAL CHECKLIST:")
checks = []
checks.append(("Observation shape correct", obs.shape[0] == expected_size))
checks.append(("Processing times included", num_jobs * num_machines in [expected_size]))
checks.append(("Values in [0,1] range", obs.min() >= 0 and obs.max() <= 1))
checks.append(("No NaN values", not np.any(np.isnan(obs))))
checks.append(("No Inf values", not np.any(np.isinf(obs))))
checks.append(("Processing times non-zero", proc_times.max() > 0))

for check_name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"   {status} {check_name}")

all_passed = all(passed for _, passed in checks)
if all_passed:
    print(f"\n🎉 ALL CHECKS PASSED! Ready to train.")
else:
    print(f"\n⚠️  SOME CHECKS FAILED! Fix issues before training.")

print(f"\n{'='*80}\n")
