#!/usr/bin/env python3
"""
Fixed Dynamic vs Static RL Comparison
Based on the working possion_job_backup.py structure
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import gymnasium as gym
import torch
from tqdm import tqdm
import time
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Global tracking for arrival time distribution analysis
TRAINING_ARRIVAL_TIMES = []  # Track all arrival times during training
TRAINING_EPISODE_COUNT = 0   # Track episode count

# --- Expanded Job Data for Better Generalization ---
# Exact dataset from test3_backup.py that achieved makespan=43 with dynamic RL
ENHANCED_JOBS_DATA = collections.OrderedDict({
    0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}, {'proc_times': {'M0': 2}}],
    1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M1': 3, 'M2': 4}}],
    2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M0': 3, 'M1': 4}}, {'proc_times': {'M2': 7}}],
    3: [{'proc_times': {'M1': 8}}, {'proc_times': {'M2': 2}}, {'proc_times': {'M0': 5, 'M1': 6}}],
    4: [{'proc_times': {'M0': 6, 'M1': 9}}, {'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M2': 6}}],
    5: [{'proc_times': {'M1': 5, 'M2': 8}}, {'proc_times': {'M0': 6}}, {'proc_times': {'M1': 4, 'M2': 3}}],
    6: [{'proc_times': {'M0': 7, 'M2': 4}}, {'proc_times': {'M0': 5, 'M1': 6}}, {'proc_times': {'M1': 3}}, {'proc_times': {'M0': 2, 'M2': 5}}],
})

# Deterministic arrival times - simplified integer values for better learning
DETERMINISTIC_ARRIVAL_TIMES = {0: 0, 1: 0, 2: 0, 3: 8, 4: 12, 5: 16, 6: 20}

MACHINE_LIST = ['M0', 'M1', 'M2']

def mask_fn(env):
    """Mask function for ActionMasker wrapper"""
    return env.action_masks()

def main():
    """Simple test using the optimized heuristics from the main file"""
    print("=" * 80)
    print("TESTING OPTIMIZED HEURISTICS FROM MAIN FILE")
    print("=" * 80)
    
    try:
        # Import the working functions from the main file
        from clean_dynamic_vs_static_comparison import (
            simple_spt_heuristic, basic_greedy_scheduler,
            optimized_spt_scheduler, earliest_completion_scheduler
        )
        
        print("✅ Successfully imported heuristic functions!")
        
        # Test with simple arrival times
        arrival_times = {0: 0, 1: 0, 2: 0, 3: 8, 4: 12, 5: 16, 6: 20}
        
        print(f"\nTesting heuristics with arrival times: {arrival_times}")
        print("-" * 60)
        
        results = []
        
        # Test Simple SPT
        try:
            makespan, schedule = simple_spt_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_times)
            results.append(("Simple SPT", makespan))
            print(f"✅ Simple SPT: {makespan:.2f}")
        except Exception as e:
            print(f"❌ Simple SPT failed: {e}")
        
        # Test Basic Greedy
        try:
            makespan, schedule = basic_greedy_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_times)
            results.append(("Basic Greedy", makespan))
            print(f"✅ Basic Greedy: {makespan:.2f}")
        except Exception as e:
            print(f"❌ Basic Greedy failed: {e}")
        
        # Test Optimized SPT
        try:
            makespan, schedule = optimized_spt_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_times)
            results.append(("Optimized SPT", makespan))
            print(f"✅ Optimized SPT: {makespan:.2f}")
        except Exception as e:
            print(f"❌ Optimized SPT failed: {e}")
        
        # Test Earliest Completion
        try:
            makespan, schedule = earliest_completion_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_times)
            results.append(("Earliest Completion", makespan))
            print(f"✅ Earliest Completion: {makespan:.2f}")
        except Exception as e:
            print(f"❌ Earliest Completion failed: {e}")
        
        # Show results
        print("\n" + "=" * 60)
        print("HEURISTIC PERFORMANCE SUMMARY")
        print("=" * 60)
        
        if results:
            results.sort(key=lambda x: x[1])
            print(f"Expected competitive range: ~35-45")
            print("-" * 40)
            
            for i, (name, makespan) in enumerate(results, 1):
                status = "🎯 EXCELLENT" if makespan < 40 else "✅ GOOD" if makespan < 50 else "⚠️ OK" if makespan < 60 else "❌ POOR"
                print(f"{i}. {name:20s}: {makespan:6.2f} {status}")
            
            best_name, best_makespan = results[0]
            print(f"\n🏆 BEST HEURISTIC: {best_name} (makespan: {best_makespan:.2f})")
            
            if best_makespan < 45:
                print("✅ SUCCESS: Heuristics are performing competitively!")
                print("✅ The main file heuristics are working correctly.")
                print("\n💡 You can now run the full comparison by fixing the main() function call.")
                print("   The heuristics themselves are working - the issue is likely in the")
                print("   main() function's training or evaluation sections.")
            else:
                print("🟡 MODERATE: Heuristics working but could be better optimized")
        else:
            print("❌ No heuristics worked - there are import/function issues")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("The main file has syntax or import issues that prevent loading.")
        print("Consider running the working backup file instead:")
        print("  python possion_job_backup.py")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()