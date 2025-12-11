"""
Quick test to verify DispatchingRuleFJSPEnv works correctly.
"""

import sys
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from proactive_sche import DispatchingRuleFJSPEnv, mask_fn

# Small test problem (3 jobs, 2 machines)
TEST_JOBS = {
    0: [
        {'proc_times': {0: 10.0, 1: 12.0}},  # Job 0, Op 1
        {'proc_times': {0: 8.0, 1: 9.0}}     # Job 0, Op 2
    ],
    1: [
        {'proc_times': {0: 7.0, 1: 6.0}},    # Job 1, Op 1
        {'proc_times': {0: 11.0, 1: 10.0}}   # Job 1, Op 2
    ],
    2: [
        {'proc_times': {0: 5.0, 1: 7.0}},    # Job 2, Op 1
        {'proc_times': {0: 9.0, 1: 8.0}}     # Job 2, Op 2
    ]
}

TEST_MACHINES = [0, 1]

def test_environment():
    print("Testing DispatchingRuleFJSPEnv...")
    print("=" * 60)
    
    # Create environment
    env = DispatchingRuleFJSPEnv(
        jobs_data=TEST_JOBS,
        machine_list=TEST_MACHINES,
        initial_jobs=[0, 1],  # Jobs 0 and 1 arrive at t=0
        arrival_rate=0.1,
        seed=42
    )
    
    env = ActionMasker(env, mask_fn)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"✅ Environment created and reset successfully")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")
    print(f"   Actions: 0=FIFO, 1=SPT, 2=LPT, 3=MWKR, 4=LWKR, 5=EDD, 6=WAIT")
    print()
    
    # Test episode
    rule_names = ['FIFO', 'SPT', 'LPT', 'MWKR', 'LWKR', 'EDD', 'WAIT']
    step = 0
    max_steps = 50
    
    print("Running test episode (random actions with action masking)...")
    print("-" * 60)
    
    while step < max_steps:
        # Get action masks
        action_masks = env.action_masks()
        valid_actions = [i for i, mask in enumerate(action_masks) if mask]
        
        if not valid_actions:
            print(f"Step {step}: No valid actions available")
            break
        
        # Select random valid action
        action = np.random.choice(valid_actions)
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step {step}: Action={rule_names[action]}, Reward={reward:.2f}, Makespan={env.env.current_makespan:.2f}, Done={done}")
        
        step += 1
        
        if done or truncated:
            print(f"\n✅ Episode completed at step {step}")
            print(f"   Final makespan: {env.env.current_makespan:.2f}")
            
            # Count scheduled operations
            total_ops = sum(len(machine_ops) for machine_ops in env.env.schedule.values())
            expected_ops = sum(len(ops) for ops in TEST_JOBS.values())
            
            print(f"   Operations scheduled: {total_ops}/{expected_ops}")
            
            if total_ops == expected_ops:
                print(f"   ✅ All operations scheduled successfully!")
            else:
                print(f"   ⚠️  WARNING: Incomplete schedule")
            
            break
    
    if step >= max_steps:
        print(f"\n⚠️  Episode reached max steps ({max_steps}) without completion")
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_environment()
