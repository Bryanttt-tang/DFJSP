#!/usr/bin/env python3
"""
Quick test to debug the EnhancedTrainingCallback
"""
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from clean_dynamic_vs_static_comparison import *

def test_callback_functionality():
    """Test if the callback properly logs episode rewards and training metrics."""
    print("=== TESTING ENHANCED TRAINING CALLBACK ===")
    
    # Reset training metrics
    reset_training_metrics()
    
    print("1. Testing callback initialization...")
    
    # Create a small dynamic environment for quick testing
    def make_test_env():
        env = PoissonDynamicFJSPEnv(
            ENHANCED_JOBS_DATA, MACHINE_LIST, 
            initial_jobs=[0, 1, 2],
            arrival_rate=0.5,
            reward_mode="makespan_increment",
            seed=42,
            max_time_horizon=50  # Short for quick testing
        )
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_test_env])
    vec_env = VecMonitor(vec_env)
    
    print("2. Creating model...")
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,  # Smaller for quick testing
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    
    print("3. Creating callback...")
    callback = EnhancedTrainingCallback("Test", verbose=2)  # High verbosity
    
    print("4. Starting short training run...")
    try:
        model.learn(total_timesteps=1000, callback=callback)  # Very short test
        print("✅ Training completed successfully!")
        
        print("\n5. Checking captured metrics...")
        print(f"Episode rewards: {len(TRAINING_METRICS['episode_rewards'])}")
        print(f"Policy loss: {len(TRAINING_METRICS['policy_loss'])}")
        print(f"Value loss: {len(TRAINING_METRICS['value_loss'])}")
        print(f"Total loss: {len(TRAINING_METRICS['total_loss'])}")
        
        if TRAINING_METRICS['episode_rewards']:
            print(f"First 3 episode rewards: {TRAINING_METRICS['episode_rewards'][:3]}")
            print(f"Episode rewards match other metrics: {len(TRAINING_METRICS['episode_rewards']) == len(TRAINING_METRICS['policy_loss'])}")
        else:
            print("❌ No episode rewards captured!")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_callback_functionality()