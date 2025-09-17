#!/usr/bin/env python3
"""
Quick test version of Poisson Dynamic FJSP with reduced training time for debugging
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gymnasium import spaces
import random
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
import collections
import math
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Use smaller job set for quick testing
QUICK_JOBS_DATA = collections.OrderedDict({
    # Initial jobs (available at start)
    0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}],
    1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}],
    2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M2': 7}}],
    
    # Dynamic jobs (arrive according to Poisson process) 
    3: [{'proc_times': {'M1': 8}}, {'proc_times': {'M0': 5}}],
    4: [{'proc_times': {'M0': 6}}, {'proc_times': {'M2': 4}}]
})

MACHINE_LIST = ['M0', 'M1', 'M2']

def quick_test():
    """Quick test to verify our fixes work."""
    print("="*60)
    print("QUICK POISSON FJSP TEST")
    print("="*60)
    print(f"Jobs: {len(QUICK_JOBS_DATA)}, Machines: {len(MACHINE_LIST)}")
    print("Initial jobs: 0-2, Dynamic jobs: 3-4")
    
    # Test 1: SPT Heuristic
    print("\n1. Testing SPT Heuristic...")
    from dynamic_poisson_fjsp import heuristic_spt_poisson
    
    spt_makespan, spt_schedule, spt_arrivals = heuristic_spt_poisson(
        QUICK_JOBS_DATA, MACHINE_LIST,
        initial_jobs=[0, 1, 2],
        arrival_rate=0.3,
        seed=42
    )
    print(f"SPT Result: {spt_makespan:.2f}")
    print(f"SPT Arrivals: {spt_arrivals}")
    
    # Test 2: Quick RL Training
    print("\n2. Quick RL Training (5000 timesteps)...")
    from dynamic_poisson_fjsp import train_poisson_agent, evaluate_poisson_agent
    
    rl_model = train_poisson_agent(
        QUICK_JOBS_DATA, MACHINE_LIST,
        initial_jobs=[0, 1, 2],
        arrival_rate=0.3,
        total_timesteps=5000  # Very quick training
    )
    
    # Test 3: RL Evaluation
    print("\n3. RL Evaluation...")
    rl_makespan, rl_schedule = evaluate_poisson_agent(
        rl_model, QUICK_JOBS_DATA, MACHINE_LIST,
        initial_jobs=[0, 1, 2],
        arrival_rate=0.3,
        num_episodes=3,
        debug=True
    )
    print(f"RL Result: {rl_makespan:.2f}")
    
    # Comparison
    print("\n4. COMPARISON:")
    print(f"SPT Makespan: {spt_makespan:.2f}")
    print(f"RL Makespan:  {rl_makespan:.2f}")
    
    if rl_makespan < spt_makespan:
        improvement = ((spt_makespan - rl_makespan) / spt_makespan) * 100
        print(f"✓ RL improves by {improvement:.1f}%")
    else:
        gap = ((rl_makespan - spt_makespan) / spt_makespan) * 100
        print(f"✗ RL is {gap:.1f}% worse than SPT")
        print("Possible issues:")
        print("- Training timesteps too low")
        print("- Reward function needs tuning")
        print("- Action space too large")
        print("- Observation space inadequate")
    
    return spt_makespan, rl_makespan

if __name__ == "__main__":
    quick_test()
