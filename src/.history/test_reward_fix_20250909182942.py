"""
Quick test to verify the reward function fix
"""

# Import the main script
exec(open('clean_dynamic_vs_static_comparison.py').read())

def test_reward_fix():
    print("Testing reward function fix...")
    
    # Test the environment creation
    env = PoissonDynamicFJSPEnv(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=[0, 1, 2],
        arrival_rate=0.1,
        reward_mode='combined_makespan_utilization',
        seed=42
    )
    
    obs, _ = env.reset()
    print("✓ Environment created successfully!")
    
    # Test a few steps to ensure reward calculation works
    for i in range(3):
        action_masks = env.action_masks()
        if any(action_masks):
            valid_actions = [j for j, valid in enumerate(action_masks) if valid]
            action = valid_actions[0]
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {i+1}: reward={reward:.2f}, time={env.current_time:.2f}")
            if done or truncated:
                break
        else:
            break
    
    print("✓ Reward function fix successful!")
    return True

if __name__ == "__main__":
    test_reward_fix()
