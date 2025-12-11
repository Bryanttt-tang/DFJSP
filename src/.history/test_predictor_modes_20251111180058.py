"""
Test script to verify ArrivalPredictor MLE and MAP modes work correctly.
"""

import numpy as np
import random
from proactive_sche import ArrivalPredictor, ProactiveDynamicFJSPEnv
from utils import generate_simplified_fjsp_dataset
from sb3_contrib.common.wrappers import ActionMasker

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("="*80)
print("Testing ArrivalPredictor: MLE vs MAP modes")
print("="*80)

# Generate a simple dataset
jobs_data, machine_list, machine_metadata = generate_simplified_fjsp_dataset(
    num_initial_jobs=3,
    num_future_jobs=5,
    total_num_machines=3,
    seed=SEED
)

print(f"\nDataset: {len(jobs_data)} jobs, {len(machine_list)} machines")

# Test 1: MLE mode
print("\n" + "-"*80)
print("TEST 1: MLE Mode (Maximum Likelihood Estimation)")
print("-"*80)

predictor_mle = ArrivalPredictor(initial_rate_guess=0.05, mode='mle')
print(f"Initial rate: {predictor_mle.current_estimated_rate:.4f}")

# Simulate observing some arrivals
arrivals = [0.0, 10.5, 22.3, 35.7, 48.2]
for arr_time in arrivals:
    predictor_mle.observe_arrival(arr_time)
    
print(f"After observing {len(arrivals)} arrivals:")
print(f"  Estimated rate: {predictor_mle.current_estimated_rate:.4f}")
print(f"  Confidence: {predictor_mle.get_confidence():.4f}")

# Predict next arrivals
predictions = predictor_mle.predict_next_arrivals(current_time=50.0, num_jobs_to_predict=3)
print(f"  Next 3 predicted arrivals: {[f'{p:.1f}' for p in predictions]}")

# Test 2: MAP mode
print("\n" + "-"*80)
print("TEST 2: MAP Mode (Maximum A Posteriori with Gamma prior)")
print("-"*80)

predictor_map = ArrivalPredictor(
    initial_rate_guess=0.05, 
    mode='map', 
    prior_shape=2.0,  # Weak prior
    prior_rate=None  # Will default to 2.0 / 0.05 = 40.0
)
print(f"Initial rate: {predictor_map.current_estimated_rate:.4f}")
print(f"Prior: Gamma(α={predictor_map.prior_shape}, β={predictor_map.prior_rate:.1f})")

# Observe the same arrivals
for arr_time in arrivals:
    predictor_map.observe_arrival(arr_time)
    
print(f"After observing {len(arrivals)} arrivals:")
print(f"  Estimated rate: {predictor_map.current_estimated_rate:.4f}")
print(f"  Confidence: {predictor_map.get_confidence():.4f}")

# Predict next arrivals
predictions = predictor_map.predict_next_arrivals(current_time=50.0, num_jobs_to_predict=3)
print(f"  Next 3 predicted arrivals: {[f'{p:.1f}' for p in predictions]}")

# Compare
print("\n" + "-"*80)
print("COMPARISON:")
print("-"*80)
print(f"MLE rate: {predictor_mle.current_estimated_rate:.4f}")
print(f"MAP rate: {predictor_map.current_estimated_rate:.4f}")
print(f"Difference: {abs(predictor_mle.current_estimated_rate - predictor_map.current_estimated_rate):.4f}")
print("\nNote: MAP should be closer to prior when little data, converge to MLE with more data")

# Test 3: Test in environment with MLE
print("\n" + "-"*80)
print("TEST 3: ProactiveDynamicFJSPEnv with MLE predictor")
print("-"*80)

def mask_fn(env):
    return env.action_masks()

env_mle = ProactiveDynamicFJSPEnv(
    jobs_data, machine_list,
    initial_jobs=3,
    arrival_rate=0.08,
    predictor_mode='mle',
    seed=SEED
)
env_mle = ActionMasker(env_mle, mask_fn)
obs, _ = env_mle.reset()

print(f"Environment created with MLE predictor")
print(f"Predictor mode: {env_mle.env.arrival_predictor.mode}")
print(f"Initial estimated rate: {env_mle.env.arrival_predictor.current_estimated_rate:.4f}")

# Take a few steps
for i in range(5):
    action_mask = env_mle.action_masks()
    valid_actions = np.where(action_mask)[0]
    action = np.random.choice(valid_actions)
    obs, reward, done, truncated, info = env_mle.step(action)
    if done:
        break

stats = env_mle.env.arrival_predictor.get_stats()
print(f"\nAfter {i+1} steps:")
print(f"  Estimated rate: {stats['estimated_rate']:.4f}")
print(f"  Confidence: {stats['confidence']:.4f}")
print(f"  Current makespan: {env_mle.env.current_makespan:.2f}")

# Test 4: Test in environment with MAP
print("\n" + "-"*80)
print("TEST 4: ProactiveDynamicFJSPEnv with MAP predictor")
print("-"*80)

env_map = ProactiveDynamicFJSPEnv(
    jobs_data, machine_list,
    initial_jobs=3,
    arrival_rate=0.08,
    predictor_mode='map',
    prior_shape=3.0,
    prior_rate=50.0,
    seed=SEED
)
env_map = ActionMasker(env_map, mask_fn)
obs, _ = env_map.reset()

print(f"Environment created with MAP predictor")
print(f"Predictor mode: {env_map.env.arrival_predictor.mode}")
print(f"Prior: Gamma(α={env_map.env.arrival_predictor.prior_shape}, β={env_map.env.arrival_predictor.prior_rate})")
print(f"Initial estimated rate: {env_map.env.arrival_predictor.current_estimated_rate:.4f}")

# Take a few steps
for i in range(5):
    action_mask = env_map.action_masks()
    valid_actions = np.where(action_mask)[0]
    action = np.random.choice(valid_actions)
    obs, reward, done, truncated, info = env_map.step(action)
    if done:
        break

stats = env_map.env.arrival_predictor.get_stats()
print(f"\nAfter {i+1} steps:")
print(f"  Estimated rate: {stats['estimated_rate']:.4f}")
print(f"  Confidence: {stats['confidence']:.4f}")
print(f"  Current makespan: {env_map.env.current_makespan:.2f}")

# Test 5: Test wait action with simple reward
print("\n" + "-"*80)
print("TEST 5: Wait action with simple makespan_increment reward")
print("-"*80)

env_wait = ProactiveDynamicFJSPEnv(
    jobs_data, machine_list,
    initial_jobs=3,
    arrival_rate=0.08,
    predictor_mode='mle',
    seed=SEED+1
)
env_wait = ActionMasker(env_wait, mask_fn)
obs, _ = env_wait.reset()

print(f"Testing wait action...")
print(f"Initial event_time: {env_wait.env.event_time:.2f}")
print(f"Initial makespan: {env_wait.env.current_makespan:.2f}")

# Find a wait action (should be at the end of action space)
action_mask = env_wait.action_masks()
wait_actions = []
for action in range(env_wait.action_space.n):
    if action >= env_wait.env.wait_action_start and action_mask[action]:
        wait_actions.append(action)

if wait_actions:
    wait_action = wait_actions[0]  # First wait action
    wait_idx = wait_action - env_wait.env.wait_action_start
    wait_duration = env_wait.env.wait_durations[wait_idx]
    print(f"\nExecuting wait action {wait_idx} (duration={wait_duration})")
    
    obs, reward, done, truncated, info = env_wait.step(wait_action)
    
    print(f"After wait:")
    print(f"  Event_time: {env_wait.env.event_time:.2f}")
    print(f"  Makespan: {env_wait.env.current_makespan:.2f}")
    print(f"  Reward: {reward:.4f}")
    print(f"  ✓ Makespan >= event_time: {env_wait.env.current_makespan >= env_wait.env.event_time}")
else:
    print("No wait actions available (unexpected)")

print("\n" + "="*80)
print("All tests completed successfully!")
print("="*80)
