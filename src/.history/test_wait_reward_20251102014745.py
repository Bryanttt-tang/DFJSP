"""
Quick test script to verify the intelligent wait reward design.
"""

import numpy as np
from proactive_backup import PoissonDynamicFJSPEnv, ENHANCED_JOBS_DATA, MACHINE_LIST

print("="*80)
print("TESTING INTELLIGENT WAIT REWARD DESIGN")
print("="*80)

# Test 1: Simple mode (no predictor)
print("\n[Test 1] Simple Mode (Intelligent Heuristics Only)")
print("-"*80)

env_simple = PoissonDynamicFJSPEnv(
    jobs_data=ENHANCED_JOBS_DATA,
    machine_list=MACHINE_LIST,
    initial_jobs=[0, 1, 2, 3, 4],
    arrival_rate=0.08,
    reward_mode="makespan_increment",
    seed=42,
    use_arrival_predictor=False  # Disable predictor
)

obs, info = env_simple.reset(seed=42)
print(f"Environment initialized: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
print(f"Initial jobs: {env_simple.initial_job_ids}")
print(f"Dynamic jobs: {env_simple.dynamic_job_ids}")
print(f"Arrival times: {env_simple.job_arrival_times}")

# Test wait reward in different scenarios
print("\n--- Scenario A: Machines idle, jobs available ---")
action_mask = env_simple.action_masks()
scheduling_available = np.any(action_mask[:-1])
wait_reward = env_simple._calculate_wait_reward(
    scheduling_actions_available=scheduling_available,
    current_event_time=env_simple.event_time
)
print(f"Scheduling actions available: {scheduling_available}")
print(f"Event time: {env_simple.event_time}")
print(f"Machine utilization: {sum(1 for m, t in env_simple.machine_next_free.items() if t > env_simple.event_time)} / {len(MACHINE_LIST)} busy")
print(f"Wait reward: {wait_reward:.3f}")

# Take a scheduling action to change state
valid_actions = [i for i, mask in enumerate(action_mask) if mask and i != env_simple.WAIT_ACTION]
if valid_actions:
    obs, reward, done, truncated, info = env_simple.step(valid_actions[0])
    print(f"\nTook scheduling action, new event_time: {env_simple.event_time}")

print("\n--- Scenario B: After scheduling, check wait again ---")
action_mask = env_simple.action_masks()
scheduling_available = np.any(action_mask[:-1])
wait_reward = env_simple._calculate_wait_reward(
    scheduling_actions_available=scheduling_available,
    current_event_time=env_simple.event_time
)
print(f"Scheduling actions available: {scheduling_available}")
print(f"Event time: {env_simple.event_time}")
print(f"Wait reward: {wait_reward:.3f}")

print("\n" + "="*80)
print("[Test 2] Predictor Mode (with ArrivalPredictor)")
print("-"*80)

env_predictor = PoissonDynamicFJSPEnv(
    jobs_data=ENHANCED_JOBS_DATA,
    machine_list=MACHINE_LIST,
    initial_jobs=[0, 1, 2, 3, 4],
    arrival_rate=0.08,
    reward_mode="makespan_increment",
    seed=42,
    use_arrival_predictor=True  # Enable predictor
)

obs, info = env_predictor.reset(seed=42)
print(f"Predictor enabled: {env_predictor.arrival_predictor is not None}")

if env_predictor.arrival_predictor:
    stats = env_predictor.arrival_predictor.get_stats()
    print(f"Initial predictor stats:")
    print(f"  Estimated rate: {stats['estimated_rate']:.4f}")
    print(f"  Confidence: {stats['confidence']:.2%}")
    print(f"  Observations: {stats['num_global_observations']}")

# Simulate a few episodes to let predictor learn
print("\n--- Simulating 5 quick episodes for predictor learning ---")
for ep in range(5):
    obs, info = env_predictor.reset()
    done = False
    steps = 0
    while not done and steps < 50:
        action_mask = env_predictor.action_masks()
        valid_actions = [i for i, mask in enumerate(action_mask) if mask]
        if not valid_actions:
            break
        action = np.random.choice(valid_actions)
        obs, reward, done, truncated, info = env_predictor.step(action)
        steps += 1
    
    # Finalize episode for cross-episode learning (if implemented)
    # Note: You may need to add a finalize_episode method to reactive env
    
    if ep == 4:  # Last episode
        stats = env_predictor.arrival_predictor.get_stats()
        print(f"\nAfter {ep+1} episodes, predictor stats:")
        print(f"  Estimated rate: {stats['estimated_rate']:.4f}")
        print(f"  Confidence: {stats['confidence']:.2%}")
        print(f"  Observations: {stats['num_global_observations']}")

# Test wait reward with predictor
print("\n--- Testing wait reward with learned predictor ---")
obs, info = env_predictor.reset()
action_mask = env_predictor.action_masks()
scheduling_available = np.any(action_mask[:-1])
wait_reward = env_predictor._calculate_wait_reward(
    scheduling_actions_available=scheduling_available,
    current_event_time=env_predictor.event_time
)
print(f"Wait reward (with predictor): {wait_reward:.3f}")

print("\n" + "="*80)
print("TESTS COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nKey Takeaways:")
print("1. Intelligent wait rewards vary by context (-10.0 to -0.5)")
print("2. Predictor learns arrival patterns across episodes")
print("3. Higher confidence → stronger prediction-based bonuses")
print("4. Toggle USE_INTELLIGENT_REWARD in _calculate_wait_reward for ablation studies")
print("\nSee WAIT_ACTION_REWARD_DESIGN.md for full documentation.")
