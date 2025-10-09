#!/usr/bin/env python3

"""
Simple test to debug the training callback and metrics logging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean_dynamic_vs_static_comparison import (
    ENHANCED_JOBS_DATA, MACHINE_LIST, TRAINING_METRICS,
    train_perfect_knowledge_agent, plot_training_metrics, reset_training_metrics
)

def test_callback_debug():
    """Test the training callback with minimal steps to debug the issue."""
    print("🧪 DEBUGGING TRAINING CALLBACK")
    print("=" * 50)
    
    # Use a small test scenario
    arrival_times = [0, 1, 3, 5, 7]  # Simple arrival pattern
    
    print(f"Training with {len(ENHANCED_JOBS_DATA)} jobs and arrival times: {arrival_times}")
    print("Using minimal timesteps to quickly debug the callback...")
    
    # Reset metrics
    reset_training_metrics()
    
    # Check initial state
    print(f"Initial TRAINING_METRICS episode_rewards: {len(TRAINING_METRICS['episode_rewards'])} items")
    
    # Train with very few timesteps
    try:
        model = train_perfect_knowledge_agent(
            ENHANCED_JOBS_DATA, 
            MACHINE_LIST, 
            arrival_times=arrival_times,
            total_timesteps=1000,  # Very small for quick test
            reward_mode="makespan_increment", 
            learning_rate=3e-4
        )
        
        print(f"\n📊 POST-TRAINING METRICS CHECK:")
        print(f"Episode rewards captured: {len(TRAINING_METRICS['episode_rewards'])}")
        print(f"Episode counts captured: {len(TRAINING_METRICS['episode_count'])}")
        print(f"Value losses captured: {len(TRAINING_METRICS['value_loss'])}")
        print(f"Policy losses captured: {len(TRAINING_METRICS['policy_loss'])}")
        
        if len(TRAINING_METRICS['episode_rewards']) > 0:
            print(f"First few episode rewards: {TRAINING_METRICS['episode_rewards'][:5]}")
            print(f"Last few episode rewards: {TRAINING_METRICS['episode_rewards'][-5:]}")
        else:
            print("❌ NO EPISODE REWARDS CAPTURED!")
            
        print(f"\n📊 Testing plot_training_metrics()...")
        plot_training_metrics()
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_callback_debug()