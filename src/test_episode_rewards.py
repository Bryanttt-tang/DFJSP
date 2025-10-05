#!/usr/bin/env python3

"""
Quick test to verify episode reward capture with enhanced callback
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean_dynamic_vs_static_comparison import (
    train_dynamic_agent, reset_training_metrics, TRAINING_METRICS,
    plot_training_metrics
)

def test_episode_reward_capture():
    """Test if episode rewards are captured correctly."""
    print("🔬 Testing episode reward capture...")
    
    # Reset metrics
    reset_training_metrics()
    print(f"Initial TRAINING_METRICS keys: {list(TRAINING_METRICS.keys())}")
    print(f"Initial episode_rewards: {TRAINING_METRICS['episode_rewards']}")
    
    # Train for short duration with verbose output
    try:
        print("\n🚀 Starting dynamic agent training (short test)...")
        dynamic_makespan, dynamic_agent = train_dynamic_agent(
            total_timesteps=1000,  # Very short training
            verbose=1,  # Enable debug output
            save_model=False
        )
        
        print(f"\n📊 Post-training TRAINING_METRICS:")
        print(f"  Episode rewards count: {len(TRAINING_METRICS['episode_rewards'])}")
        print(f"  Episode rewards: {TRAINING_METRICS['episode_rewards'][:5]}...")  # First 5
        print(f"  Value loss count: {len(TRAINING_METRICS['value_loss'])}")
        print(f"  Timesteps count: {len(TRAINING_METRICS['timesteps'])}")
        
        # Test plotting
        if len(TRAINING_METRICS['episode_rewards']) > 0:
            print("\n✅ Episode rewards captured! Testing plot...")
            plot_training_metrics()
            print("✅ Plot generated successfully!")
        else:
            print("\n❌ No episode rewards captured!")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_episode_reward_capture()