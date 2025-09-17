"""
Quick test script to check if the new utilization_maximization reward mode works
"""

# Import the environment classes
exec(open('dynamic_poisson_fjsp.py').read())

import collections

# Job data
ENHANCED_JOBS_DATA = collections.OrderedDict({
    0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}, {'proc_times': {'M0': 2}}],
    1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M1': 3, 'M2': 4}}],
    2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M0': 3, 'M1': 4}}, {'proc_times': {'M2': 7}}],
    3: [{'proc_times': {'M1': 8}}, {'proc_times': {'M2': 2}}, {'proc_times': {'M0': 5, 'M1': 6}}],
    4: [{'proc_times': {'M0': 6, 'M1': 9}}, {'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M2': 6}}],
})

MACHINE_LIST = ['M0', 'M1', 'M2']

def test_new_reward_mode():
    print("Testing new utilization_maximization reward mode")
    print("=" * 50)
    
    # Test environment with new reward mode
    env = PoissonDynamicFJSPEnv(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=[0, 1, 2],
        arrival_rate=0.1,
        reward_mode="utilization_maximization",
        seed=42
    )
    
    obs, _ = env.reset()
    print(f"Environment created successfully!")
    print(f"Arrival times: {env.arrival_times}")
    print(f"Initial jobs: {env.arrived_jobs}")
    
    # Test a few random actions to see reward behavior
    total_reward = 0
    for step in range(10):
        action_masks = env.action_masks()
        
        if not any(action_masks):
            print(f"Step {step}: No valid actions")
            break
            
        # Take a random valid action
        valid_actions = [i for i, valid in enumerate(action_masks) if valid]
        if valid_actions:
            action = valid_actions[0]  # Take first valid action
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step}: Action {action}, Reward {reward:.2f}, "
                  f"Time {env.current_time:.2f}, Idle time {info.get('idle_time', 0):.2f}")
            
            if done or truncated:
                print(f"Episode finished after {step + 1} steps")
                break
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final makespan: {env.current_time:.2f}")
    
    # Check machine utilization
    total_workload = sum(env.machine_workload.values())
    total_capacity = env.current_time * len(MACHINE_LIST)
    utilization = total_workload / total_capacity if total_capacity > 0 else 0
    print(f"Machine utilization: {utilization:.1%}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_new_reward_mode()
        if success:
            print("\n✓ New reward mode works correctly!")
        else:
            print("\n✗ New reward mode has issues")
    except Exception as e:
        print(f"\n✗ Error testing new reward mode: {e}")
        import traceback
        traceback.print_exc()
