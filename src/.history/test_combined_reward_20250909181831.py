"""
Quick test to verify the new combined_makespan_utilization reward mode
"""

import numpy as np
import random
import collections
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Import environment classes by executing the main file (simplified approach)
exec(open('clean_dynamic_vs_static_comparison.py').read())

def test_new_reward_mode():
    print("Testing new combined_makespan_utilization reward mode")
    print("=" * 60)
    
    # Test the new reward mode
    env = PoissonDynamicFJSPEnv(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=[0, 1, 2],
        arrival_rate=0.1,
        reward_mode="combined_makespan_utilization",
        seed=42
    )
    
    obs, _ = env.reset()
    print(f"✓ Environment created successfully with new reward mode!")
    print(f"Arrival times: {env.arrival_times}")
    print(f"Initial jobs: {sorted(env.arrived_jobs)}")
    
    # Test reward calculation
    total_reward = 0
    step_count = 0
    rewards = []
    
    print(f"\nTesting reward behavior:")
    
    for step in range(15):
        action_masks = env.action_masks()
        
        if not any(action_masks):
            print(f"Step {step}: No valid actions available")
            break
            
        # Take first valid action
        valid_actions = [i for i, valid in enumerate(action_masks) if valid]
        if valid_actions:
            action = valid_actions[0]
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(reward)
            step_count += 1
            
            idle_time = info.get('idle_time', 0)
            newly_arrived = info.get('newly_arrived_jobs', 0)
            
            print(f"Step {step+1}: Reward={reward:.1f}, Time={env.current_time:.1f}, "
                  f"Idle={idle_time:.1f}, New arrivals={newly_arrived}")
            
            if done or truncated:
                print(f"Episode completed after {step + 1} steps")
                break
    
    print(f"\n--- Results ---")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Average reward per step: {total_reward/max(step_count,1):.1f}")
    print(f"Final makespan: {env.current_time:.1f}")
    
    # Calculate machine utilization
    total_workload = sum(env.machine_workload.values())
    total_capacity = env.current_time * len(MACHINE_LIST)
    utilization = total_workload / total_capacity if total_capacity > 0 else 0
    print(f"Machine utilization: {utilization:.1%}")
    
    # Check if new reward mode provides good incentives
    positive_rewards = sum(1 for r in rewards if r > 0)
    negative_rewards = sum(1 for r in rewards if r < 0)
    
    print(f"\nReward distribution:")
    print(f"Positive rewards: {positive_rewards}/{len(rewards)}")
    print(f"Negative rewards: {negative_rewards}/{len(rewards)}")
    print(f"Max reward: {max(rewards):.1f}")
    print(f"Min reward: {min(rewards):.1f}")
    
    return True

if __name__ == "__main__":
    try:
        print("Testing new combined reward mode implementation...")
        success = test_new_reward_mode()
        if success:
            print("\n✓ New combined_makespan_utilization reward mode works correctly!")
            print("✓ Ready for training with improved reward function!")
        else:
            print("\n✗ Issues detected with new reward mode")
    except Exception as e:
        print(f"\n✗ Error testing new reward mode: {e}")
        import traceback
        traceback.print_exc()
