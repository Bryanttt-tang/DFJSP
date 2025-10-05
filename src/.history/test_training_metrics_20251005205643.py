#!/usr/bin/env python3

# Simple test script to verify training metrics collection is working

import sys
import os
sys.path.append('/Users/tanu/Desktop/PhD/Scheduling/src')

from clean_dynamic_vs_static_comparison import (
    ENHANCED_JOBS_DATA, MACHINE_LIST, TRAINING_METRICS, 
    train_dynamic_agent, plot_training_metrics, reset_training_metrics
)

def test_training_metrics():
    """Test that training metrics are being collected properly."""
    print("🧪 Testing training metrics collection...")
    
    # Reset metrics
    reset_training_metrics()
    
    # Check initial state
    print(f"Initial TRAINING_METRICS state:")
    for key, value in TRAINING_METRICS.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")
    
    # Run a short training with verbose callback
    print(f"\n🏃 Running short Dynamic RL training (30k steps) with verbose debugging...")
    try:
        # Create a simple test environment first to verify it works
        test_env = PoissonDynamicFJSPEnv(
            ENHANCED_JOBS_DATA, 
            MACHINE_LIST, 
            initial_jobs=[0, 1, 2], 
            arrival_rate=0.1,
            reward_mode="makespan_increment"
        )
        print(f"✅ Test environment created successfully")
        
        # Test a few episodes manually to verify episode completion
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from sb3_contrib.common.wrappers import ActionMasker
        
        def make_test_env():
            env = PoissonDynamicFJSPEnv(
                ENHANCED_JOBS_DATA, 
                MACHINE_LIST, 
                initial_jobs=[0, 1, 2], 
                arrival_rate=0.1,
                reward_mode="makespan_increment"
            )
            env = ActionMasker(env, mask_fn)
            env = Monitor(env)
            return env
        
        vec_env = DummyVecEnv([make_test_env])
        
        # Test manual episode completion
        print("🔍 Testing manual episode completion...")
        obs = vec_env.reset()
        done = False
        steps = 0
        episode_reward = 0
        
        while not done and steps < 200:
            # Random valid action
            action_masks = vec_env.env_method('action_masks')[0]
            if any(action_masks):
                valid_actions = [i for i, mask in enumerate(action_masks) if mask]
                action = [np.random.choice(valid_actions)]
            else:
                action = [0]
            
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            steps += 1
            
            if done[0]:
                print(f"✅ Manual episode completed in {steps} steps, reward: {episode_reward:.4f}")
                break
        
        if not done[0]:
            print(f"⚠️  Manual episode did not complete in {steps} steps")
        
        # Now run actual training with enhanced callback
        model = train_dynamic_agent(
            ENHANCED_JOBS_DATA, 
            MACHINE_LIST, 
            initial_jobs=[0, 1, 2], 
            arrival_rate=0.1, 
            total_timesteps=30000,  # Short training for testing
            reward_mode="makespan_increment"
        )
        print("✅ Training completed successfully!")
        
        # Check final state with detailed analysis
        print(f"\nFinal TRAINING_METRICS state:")
        total_metrics_captured = 0
        for key, value in TRAINING_METRICS.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} items")
                total_metrics_captured += len(value)
                if len(value) > 0:
                    if key == 'episode_rewards':
                        print(f"    Episode rewards sample: {value[:5]}...")
                        print(f"    Episode rewards range: {min(value):.4f} to {max(value):.4f}")
                    elif key == 'episode_lengths':
                        print(f"    Episode lengths sample: {value[:5]}...")
                    elif key in ['policy_loss', 'value_loss']:
                        print(f"    {key} sample: {value[-3:]}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nMETRICS ANALYSIS:")
        print(f"  Total metrics captured: {total_metrics_captured}")
        print(f"  Episode rewards captured: {len(TRAINING_METRICS['episode_rewards'])}")
        print(f"  Training steps recorded: {len(TRAINING_METRICS['timesteps'])}")
        
        if len(TRAINING_METRICS['episode_rewards']) == 0:
            print(f"  ❌ NO EPISODE REWARDS CAPTURED!")
            print(f"  Debugging suggestions:")
            print(f"    1. Check if Monitor wrapper is properly applied")
            print(f"    2. Verify VecMonitor is capturing episode statistics")  
            print(f"    3. Ensure episodes are actually completing during training")
            print(f"    4. Check callback verbose level and timing")
        else:
            print(f"  ✅ Episode rewards successfully captured!")
        
        # Try plotting
        print(f"\n📊 Testing plot_training_metrics()...")
        plot_training_metrics()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_metrics()