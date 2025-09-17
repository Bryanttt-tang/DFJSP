#!/usr/bin/env python3
"""
Quick test to verify if Dynamic RL is actually learning to use arrival information.
"""

import numpy as np
import torch
from dynamic_poisson_fjsp import *

def test_observation_sensitivity():
    """Test if the observation space properly captures arrival dynamics."""
    print("=== TESTING OBSERVATION SENSITIVITY ===")
    
    # Create two identical environments with small arrival rates to avoid division by zero
    env1 = PoissonDynamicFJSPEnv(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=[0, 1],  # Start with 2 jobs
        arrival_rate=0.001,  # Very small rate to minimize random arrivals
        reward_mode="makespan_increment"
    )
    
    env2 = PoissonDynamicFJSPEnv(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=[0, 1],  # Start with 2 jobs
        arrival_rate=0.001,  # Very small rate to minimize random arrivals
        reward_mode="makespan_increment"
    )
    
    # Reset both environments first
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    
    # Manually override arrival times for remaining jobs AFTER reset
    env1.arrival_times.update({2: 5.0, 3: 10.0, 4: 15.0, 5: 20.0, 6: 25.0})
    env2.arrival_times.update({2: 2.0, 3: 4.0, 4: 6.0, 5: 8.0, 6: 10.0})
    
    # Set up arrival events manually
    env1.next_arrival_events = [(5.0, 2), (10.0, 3), (15.0, 4), (20.0, 5), (25.0, 6)]
    env2.next_arrival_events = [(2.0, 2), (4.0, 3), (6.0, 4), (8.0, 5), (10.0, 6)]
    
    # Get observations after setting different arrival patterns
    obs1 = env1._get_observation()
    obs2 = env2._get_observation()
    
    print(f"Environment 1 - Next arrivals: {env1.next_arrival_events[:3]}")
    print(f"Environment 2 - Next arrivals: {env2.next_arrival_events[:3]}")
    print(f"Observation difference norm: {np.linalg.norm(obs1 - obs2):.4f}")
    
    # Check specific arrival-related features
    obs_size = len(obs1)
    arrival_features_1 = obs1[-20:]  # Last 20 features likely include arrival info
    arrival_features_2 = obs2[-20:]
    
    print(f"Arrival features difference: {np.linalg.norm(arrival_features_1 - arrival_features_2):.4f}")
    
    if np.linalg.norm(arrival_features_1 - arrival_features_2) < 0.01:
        print("⚠️  WARNING: Observations are very similar despite different arrival patterns!")
        return False
    else:
        print("✓ Observations properly capture arrival differences")
        return True

def test_action_masking_dynamics():
    """Test if action masking adapts to arrival timing."""
    print("\n=== TESTING DYNAMIC ACTION MASKING ===")
    
    env = PoissonDynamicFJSPEnv(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=[0, 1],
        arrival_rate=0.0,
        reward_mode="makespan_increment"
    )
    
    # Set up imminent arrival
    env.arrival_times = {0: 0, 1: 0, 2: 2.0, 3: 10.0, 4: 20.0, 5: 30.0, 6: 40.0}
    env.next_arrival_events = [(2.0, 2), (10.0, 3), (20.0, 4), (30.0, 5), (40.0, 6)]
    
    obs, _ = env.reset()
    env.current_time = 1.5  # Very close to next arrival at t=2.0
    
    # Get action masks when arrival is imminent
    mask_imminent = env.action_masks()
    valid_actions_imminent = np.sum(mask_imminent)
    
    # Move time forward so no arrivals are imminent
    env.current_time = 5.0
    env._update_arrivals(5.0)  # Job 2 should have arrived
    
    mask_distant = env.action_masks()
    valid_actions_distant = np.sum(mask_distant)
    
    print(f"Valid actions when arrival imminent: {valid_actions_imminent}")
    print(f"Valid actions when arrivals distant: {valid_actions_distant}")
    
    if valid_actions_imminent != valid_actions_distant:
        print("✓ Action masking adapts to arrival timing")
        return True
    else:
        print("⚠️  Action masking doesn't adapt to arrival timing")
        return False

def test_reward_sensitivity():
    """Test if reward function properly incentivizes dynamic behavior."""
    print("\n=== TESTING REWARD SENSITIVITY ===")
    
    env = PoissonDynamicFJSPEnv(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=[0, 1],
        arrival_rate=0.0,
        reward_mode="makespan_increment"
    )
    
    env.arrival_times = {0: 0, 1: 0, 2: 3.0, 3: 6.0, 4: 9.0, 5: 12.0, 6: 15.0}
    env.next_arrival_events = [(3.0, 2), (6.0, 3), (9.0, 4), (12.0, 5), (15.0, 6)]
    
    obs, _ = env.reset()
    
    # Simulate a scenario where keeping machines available is beneficial
    env.current_time = 2.5  # Just before arrival at t=3.0
    previous_time = env.current_time
    
    # Test reward when making a good dynamic decision (short operation before arrival)
    reward_good = env._calculate_reward(
        proc_time=2.0,  # Short operation
        idle_time=0.5,  # Small idle time
        done=False,
        num_new_arrivals=0,
        previous_time=previous_time,
        current_time=previous_time + 2.0
    )
    
    # Test reward when making a poor dynamic decision (long operation blocking arrival)
    reward_bad = env._calculate_reward(
        proc_time=8.0,  # Long operation
        idle_time=0.5,  # Same idle time
        done=False,
        num_new_arrivals=0,
        previous_time=previous_time,
        current_time=previous_time + 8.0
    )
    
    print(f"Reward for good dynamic decision: {reward_good:.2f}")
    print(f"Reward for poor dynamic decision: {reward_bad:.2f}")
    
    if reward_good > reward_bad:
        print("✓ Reward function incentivizes dynamic behavior")
        return True
    else:
        print("⚠️  Reward function doesn't properly incentivize dynamic behavior")
        return False

def main():
    """Run all tests."""
    print("TESTING DYNAMIC RL LEARNING CAPABILITY")
    print("=" * 60)
    
    test1 = test_observation_sensitivity()
    test2 = test_action_masking_dynamics() 
    test3 = test_reward_sensitivity()
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"Observation sensitivity: {'✓' if test1 else '✗'}")
    print(f"Action masking dynamics: {'✓' if test2 else '✗'}")
    print(f"Reward sensitivity: {'✓' if test3 else '✗'}")
    
    if all([test1, test2, test3]):
        print("\n✓ All tests passed - RL should be able to learn dynamic behavior")
        print("💡 Issue might be in training hyperparameters or insufficient training time")
    else:
        print("\n✗ Some tests failed - RL environment needs fixes for dynamic learning")
        print("💡 Focus on fixing the failed components")

if __name__ == "__main__":
    main()
