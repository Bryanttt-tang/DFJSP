"""
Verify observation space dimensions for Proactive RL environment.
"""

from proactive_sche import ProactiveDynamicFJSPEnv
import numpy as np

# Test with typical problem size
test_jobs = {
    i: [
        {'proc_times': {'M1': 10+i, 'M2': 15+i, 'M3': 12+i}},
        {'proc_times': {'M1': 8+i, 'M2': 11+i, 'M3': 9+i}},
    ]
    for i in range(12)
}

machines = ['M1', 'M2', 'M3']

print("=== Proactive RL Observation Space Verification ===\n")

env = ProactiveDynamicFJSPEnv(
    test_jobs,
    machines,
    initial_jobs=[0, 1, 2, 3, 4],
    arrival_rate=0.08,
    seed=42
)

print(f"Number of jobs: {len(test_jobs)}")
print(f"Number of machines: {len(machines)}")
print(f"Total operations: {sum(len(ops) for ops in test_jobs.values())}")

print(f"\nObservation space shape: {env.observation_space.shape}")
print(f"Observation space dims: {env.observation_space.shape[0]}")

# Get an actual observation
obs, _ = env.reset()

print(f"\nActual observation shape: {obs.shape}")
print(f"Actual observation dims: {len(obs)}")

# Verify calculation
num_jobs = len(test_jobs)
num_machines = len(machines)

expected_dims = (
    num_jobs +                    # Job ready time
    num_jobs +                    # Job progress
    num_machines +                # Machine free time
    num_jobs * num_machines +     # Processing times
    num_jobs +                    # Predicted arrivals
    2                             # Global progress
)

print(f"\nExpected dims breakdown:")
print(f"  Job ready time:        {num_jobs}")
print(f"  Job progress:          {num_jobs}")
print(f"  Machine free time:     {num_machines}")
print(f"  Processing times:      {num_jobs * num_machines}")
print(f"  Predicted arrivals:    {num_jobs}")
print(f"  Global progress:       2")
print(f"  TOTAL:                 {expected_dims}")

if len(obs) == expected_dims:
    print("\n✅ Observation space dimensions correct!")
else:
    print(f"\n❌ Mismatch! Expected {expected_dims}, got {len(obs)}")

# Test observation values
print(f"\nObservation value ranges:")
print(f"  Min: {obs.min():.4f}")
print(f"  Max: {obs.max():.4f}")
print(f"  Mean: {obs.mean():.4f}")
print(f"  Std: {obs.std():.4f}")

# Check for NaN/Inf
if np.any(np.isnan(obs)):
    print("\n⚠️  WARNING: NaN values detected!")
if np.any(np.isinf(obs)):
    print("\n⚠️  WARNING: Inf values detected!")

print("\n" + "="*50)
