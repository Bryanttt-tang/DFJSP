#!/usr/bin/env python3
"""
Minimal test to identify the specific issues in clean_dynamic_vs_static_comparison.py
"""

import sys
import traceback

try:
    print("Testing imports...")
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
    print("✓ All imports successful")
    
    print("Testing basic data structures...")
    # Test the data structures from the main file
    ENHANCED_JOBS_DATA = collections.OrderedDict({
        0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}, {'proc_times': {'M0': 2}}],
        1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M1': 3, 'M2': 4}}],
        2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M0': 3, 'M1': 4}}, {'proc_times': {'M2': 7}}],
    })
    MACHINE_LIST = ['M0', 'M1', 'M2']
    print("✓ Data structures created successfully")
    
    print("Testing environment creation...")
    # Test basic environment creation
    from clean_dynamic_vs_static_comparison import StaticFJSPEnv
    env = StaticFJSPEnv(ENHANCED_JOBS_DATA, MACHINE_LIST)
    print("✓ StaticFJSPEnv created successfully")
    
    print("Testing environment reset...")
    obs, info = env.reset()
    print(f"✓ Environment reset successful, obs shape: {obs.shape}")
    
    print("Testing optimized heuristics...")
    from clean_dynamic_vs_static_comparison import simple_spt_heuristic, basic_greedy_scheduler
    arrival_times = {0: 0, 1: 0, 2: 0}
    
    makespan1, schedule1 = simple_spt_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_times)
    print(f"✓ simple_spt_heuristic: {makespan1:.2f}")
    
    makespan2, schedule2 = basic_greedy_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_times) 
    print(f"✓ basic_greedy_scheduler: {makespan2:.2f}")
    
    print("🎉 All basic components working!")
    print("The issue may be in the main() function or complex interactions.")
    
except Exception as e:
    print(f"❌ Error found: {e}")
    print("Full traceback:")
    traceback.print_exc()
    
    # Test if the issue is with the main file import
    try:
        print("\nTesting direct import of main file...")
        import clean_dynamic_vs_static_comparison as main_module
        print("✓ Main file imported successfully")
    except Exception as e2:
        print(f"❌ Main file import failed: {e2}")
        traceback.print_exc()