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
    
    # Run a short training
    print(f"\n🏃 Running short Dynamic RL training (10k steps)...")
    try:
        model = train_dynamic_agent(
            ENHANCED_JOBS_DATA, 
            MACHINE_LIST, 
            initial_jobs=[0, 1, 2], 
            arrival_rate=0.1, 
            total_timesteps=10000,  # Short training for testing
            reward_mode="makespan_increment"
        )
        print("✅ Training completed successfully!")
        
        # Check final state
        print(f"\nFinal TRAINING_METRICS state:")
        for key, value in TRAINING_METRICS.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} items")
                if len(value) > 0 and key == 'episode_rewards':
                    print(f"    Sample: {value[:3]}...")
            else:
                print(f"  {key}: {value}")
        
        # Try plotting
        print(f"\n📊 Testing plot_training_metrics()...")
        plot_training_metrics()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_metrics()