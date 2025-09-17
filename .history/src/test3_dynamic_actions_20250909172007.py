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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import collections
# Skip PULP import if it causes issues
try:
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
except ImportError:
    print("Warning: PULP not available, MILP solver will not work")
import argparse
import sys
import math
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- 1. Gantt Chart Plotter ---
def plot_gantt(schedule, machines, title="Schedule", save_path=None):
    """Plot Gantt chart for the schedule"""
    if not schedule or all(len(ops) == 0 for ops in schedule.values()):
        print("No schedule to plot - schedule is empty")
        return

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(14, len(machines) * 0.8 + 1))

    for idx, m in enumerate(machines):
        machine_ops = schedule.get(m, [])
        machine_ops.sort(key=lambda x: x[1])

        for op_data in machine_ops:
            if len(op_data) == 3:
                job_id_str, start, end = op_data
                try:
                    # Extract job number from "J0-O1"
                    j = int(job_id_str.split('-')[0][1:])
                except (ValueError, IndexError):
                    j = hash(job_id_str) % len(colors)
                
                ax.broken_barh(
                    [(start, end - start)],
                    (idx * 10, 8),
                    facecolors=colors[j % len(colors)],
                    edgecolor='black',
                    alpha=0.8
                )
                label = job_id_str
                ax.text(start + (end - start) / 2, idx * 10 + 4,
                       label, color='white', fontsize=10,
                       ha='center', va='center', weight='bold')

    ax.set_yticks([i * 10 + 4 for i in range(len(machines))])
    ax.set_yticklabels(machines)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gantt chart saved to {save_path}")
    
    plt.show()

# --- 2. Improved Dynamic RL Environment with Dynamic Action Space ---
class DynamicFJSPEnvV2(gym.Env):
    """
    Improved Dynamic FJSP Environment with dynamic action space.
    Actions are (job_id, machine_id) pairs that are valid at the current step.
    This is much more efficient than the fixed action space approach.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, job_arrival_times=None, reward_mode="makespan_increment"):
        super().__init__()
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        self.reward_mode = reward_mode
        
        if job_arrival_times is None:
            self.job_arrival_times = {job_id: 0 for job_id in self.job_ids}
        else:
            self.job_arrival_times = job_arrival_times

        # Calculate maximum possible actions (upper bound)
        # Each job can have at most one operation ready at any time
        # Each operation can be processed on multiple machines
        max_possible_actions = self.num_jobs * len(self.machines)
        
        # Use a more reasonable action space size
        self.action_space = spaces.Discrete(max_possible_actions)
        
        # Observation space (more compact)
        obs_size = (
            len(self.machines) +  # Machine availability
            self.num_jobs +       # Job progress (next operation index)
            self.num_jobs +       # Job arrival status
            1                     # Current makespan
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed, options=options)
            random.seed(seed)
            np.random.seed(seed)
        
        # State tracking
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation_index = {job_id: 0 for job_id in self.job_ids}  # Which operation is next for each job
        
        self.current_time = 0.0
        self.operations_scheduled = 0
        self.episode_step = 0
        self.max_episode_steps = self.total_operations * 3  # Safety limit
        
        # Handle job arrivals at time 0
        self.arrived_jobs = {
            job_id for job_id, arrival_time in self.job_arrival_times.items()
            if arrival_time <= self.current_time
        }
        
        # Current valid actions (list of (job_id, machine_name) tuples)
        self.valid_actions = []
        self._update_valid_actions()
        
        return self._get_observation(), {}

    def _update_valid_actions(self):
        """Update the list of valid actions based on current state"""
        self.valid_actions = []
        
        for job_id in self.job_ids:
            # Skip if job hasn't arrived yet
            if job_id not in self.arrived_jobs:
                continue
                
            # Get the next operation index for this job
            next_op_idx = self.next_operation_index[job_id]
            
            # Skip if job is complete
            if next_op_idx >= len(self.jobs[job_id]):
                continue
                
            # Get the operation data
            operation = self.jobs[job_id][next_op_idx]
            
            # Add all valid (job, machine) pairs for this operation
            for machine_name in operation['proc_times'].keys():
                if machine_name in self.machines:  # Ensure machine exists
                    self.valid_actions.append((job_id, machine_name))
        
        # If no valid actions, we need to advance time to next job arrival
        if not self.valid_actions:
            self._advance_time_to_next_arrival()

    def _advance_time_to_next_arrival(self):
        """Advance time to the next job arrival if no operations are ready"""
        future_arrivals = [
            arrival for arrival in self.job_arrival_times.values() 
            if arrival > self.current_time
        ]
        
        if future_arrivals:
            next_arrival_time = min(future_arrivals)
            self.current_time = next_arrival_time
            
            # Update arrived jobs
            self.arrived_jobs.update({
                job_id for job_id, arrival_time in self.job_arrival_times.items()
                if arrival_time <= self.current_time
            })
            
            # Update valid actions
            self._update_valid_actions()

    def action_masks(self):
        """Return action mask for MaskablePPO"""
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # Mark valid actions
        for i in range(len(self.valid_actions)):
            if i < self.action_space.n:
                mask[i] = True
        
        # If no valid actions, enable the first action to prevent crashes
        if not np.any(mask) and self.action_space.n > 0:
            mask[0] = True
            
        return mask

    def step(self, action):
        self.episode_step += 1
        
        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        # Validate action
        if action >= len(self.valid_actions):
            # Invalid action - give negative reward but continue
            return self._get_observation(), -100.0, False, False, {"error": "Invalid action index"}
        
        # Get the job and machine from the action
        job_id, machine_name = self.valid_actions[action]
        next_op_idx = self.next_operation_index[job_id]
        
        # Double-check validity (should always be true)
        if next_op_idx >= len(self.jobs[job_id]):
            return self._get_observation(), -100.0, False, False, {"error": "Job already complete"}
        
        operation = self.jobs[job_id][next_op_idx]
        if machine_name not in operation['proc_times']:
            return self._get_observation(), -100.0, False, False, {"error": "Machine cannot process operation"}
        
        # Schedule the operation
        proc_time = operation['proc_times'][machine_name]
        
        # Calculate start time considering machine availability and precedence
        machine_available_time = self.machine_next_free.get(machine_name, 0.0)
        
        # Precedence constraint: operation cannot start until previous operation is done
        if next_op_idx > 0:
            prev_op_end_time = self.operation_end_times[job_id][next_op_idx - 1]
        else:
            prev_op_end_time = self.job_arrival_times.get(job_id, 0.0)
        
        start_time = max(self.current_time, machine_available_time, prev_op_end_time)
        end_time = start_time + proc_time
        
        # Update state
        previous_makespan = max(self.machine_next_free.values()) if self.machine_next_free else 0.0
        self.machine_next_free[machine_name] = end_time
        self.operation_end_times[job_id][next_op_idx] = end_time
        self.next_operation_index[job_id] += 1
        self.operations_scheduled += 1
        
        # Update current time and check for new arrivals
        self.current_time = max(self.current_time, end_time)
        newly_arrived = {
            j_id for j_id, arrival in self.job_arrival_times.items()
            if previous_makespan < arrival <= self.current_time
        }
        self.arrived_jobs.update(newly_arrived)

        # Record in schedule
        self.schedule[machine_name].append((f"J{job_id}-O{next_op_idx+1}", start_time, end_time))

        # Update valid actions for next step
        self._update_valid_actions()

        # Check termination
        terminated = self.operations_scheduled >= self.total_operations
        
        # Calculate reward
        current_makespan = max(self.machine_next_free.values()) if self.machine_next_free else 0.0
        idle_time = max(0, start_time - machine_available_time)
        reward = self._calculate_reward(proc_time, idle_time, terminated, previous_makespan, current_makespan)
        
        info = {
            "makespan": current_makespan,
            "valid_actions_count": len(self.valid_actions),
            "operations_scheduled": self.operations_scheduled
        }
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan=None, current_makespan=None):
        if self.reward_mode == "makespan_increment":
            # Reward for minimizing makespan increment
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment
                
                # Add completion bonus
                if done:
                    reward += 100.0
                    
                return reward
            else:
                return -proc_time
                
        elif self.reward_mode == "processing_time":
            # Simple negative processing time reward
            reward = -proc_time
            if done:
                reward += 100.0
            return reward
            
        else:  # "improved" mode (default)
            reward = 10.0  # Base reward for completing an operation
            reward -= proc_time * 0.1  # Small penalty for processing time
            reward -= idle_time * 0.5   # Penalty for idle time
            
            if done:
                reward += 200.0
                # Bonus for shorter makespan
                if current_makespan and current_makespan > 0:
                    reward += max(0, 300.0 / current_makespan)
            
            return reward

    def _get_observation(self):
        """Generate observation vector"""
        obs = []
        
        # Normalization factor
        max_time = max(max(self.machine_next_free.values(), default=0), self.current_time, 1.0)
        
        # Machine availability (normalized)
        for machine in self.machines:
            availability = self.machine_next_free.get(machine, 0.0) / max_time
            obs.append(min(1.0, max(0.0, availability)))
        
        # Job progress (next operation index normalized by total operations in job)
        for job_id in self.job_ids:
            total_ops = len(self.jobs[job_id])
            progress = self.next_operation_index[job_id] / max(total_ops, 1)
            obs.append(min(1.0, max(0.0, progress)))
            
        # Job arrival status
        for job_id in self.job_ids:
            arrived = 1.0 if job_id in self.arrived_jobs else 0.0
            obs.append(arrived)
            
        # Current makespan (normalized)
        makespan_norm = self.current_time / max_time
        obs.append(min(1.0, max(0.0, makespan_norm)))
        
        # Ensure proper shape and data type
        obs_array = np.array(obs, dtype=np.float32)
        
        # Handle NaN or infinite values
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array

    def get_current_valid_actions_info(self):
        """Debug function to see current valid actions"""
        info = []
        for i, (job_id, machine_name) in enumerate(self.valid_actions):
            next_op = self.next_operation_index[job_id]
            proc_time = self.jobs[job_id][next_op]['proc_times'][machine_name]
            info.append(f"Action {i}: Job {job_id}, Op {next_op+1}, Machine {machine_name}, Time {proc_time}")
        return info

# --- 3. Training and Evaluation Functions ---
def mask_fn(env):
    """Mask function for ActionMasker wrapper"""
    return env.action_masks()

def train_agent(jobs_data, machine_list, train_arrivals, log_name, total_timesteps=50000, reward_mode="makespan_increment"):
    print(f"\n--- Training Agent: {log_name} with {reward_mode} reward ---")
    
    def make_env():
        env = DynamicFJSPEnvV2(jobs_data, machine_list, train_arrivals, reward_mode=reward_mode)
        env = Monitor(env)
        return env

    # Use DummyVecEnv with ActionMasker for MaskablePPO
    vec_env = DummyVecEnv([make_env])
    vec_env = ActionMasker(vec_env, mask_fn)
    
    # MaskablePPO configuration
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    return model

def evaluate_agent(model, jobs_data, machine_list, eval_arrivals, scenario_name, reward_mode="makespan_increment"):
    print(f"\n--- Evaluating Agent on Scenario: {scenario_name} ---")
    
    test_env = DynamicFJSPEnvV2(jobs_data, machine_list, eval_arrivals, reward_mode=reward_mode)
    
    best_makespan = float('inf')
    best_schedule = None
    
    # Run multiple episodes to get best result
    num_episodes = 5
    print(f"Testing for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = test_env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 1000:  # Safety limit
            # Get action mask and choose action
            action_mask = test_env.action_masks()
            
            if not np.any(action_mask):
                print(f"Episode {episode}: No valid actions available, terminating")
                break
            
            # Use model to predict action
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            
            # Take step
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                makespan = info.get("makespan", float('inf'))
                print(f"Episode {episode}: Makespan = {makespan:.2f}, Reward = {total_reward:.2f}")
                
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = test_env.schedule.copy()
                break
    
    print(f"Evaluation complete. Best Makespan: {best_makespan}")
    
    if best_schedule:
        plot_gantt(best_schedule, machine_list, f"{scenario_name} - RL Agent (Makespan: {best_makespan:.2f})")
    
    return best_makespan, best_schedule

# --- 4. Main Test Function ---
def test_dynamic_actions_environment():
    """Test the new dynamic actions environment"""
    
    # Load test instance
    try:
        # Read and execute the test instance file
        with open("test_instance.txt", "r") as f:
            content = f.read()
        
        # Create a namespace to execute the code
        namespace = {}
        exec(content, namespace)
        
        jobs_data = namespace['jobs_data']
        machine_list = namespace['machine_list']
        job_arrival_times = namespace['job_arrival_times']
        
        print("Loaded test instance successfully!")
        print(f"Jobs: {len(jobs_data)}")
        print(f"Machines: {machine_list}")
        print(f"Arrival times: {job_arrival_times}")
        
    except Exception as e:
        print(f"Error loading test instance: {e}")
        print("Using default test data instead...")
        
        # Use a simple test case instead
        jobs_data = collections.OrderedDict({
            0: [
                {'proc_times': {'M0': 7.85, 'M2': 8.07, 'M1': 2.15}},
                {'proc_times': {'M2': 6.79, 'M0': 8.4, 'M1': 4.99}},
            ],
            1: [
                {'proc_times': {'M2': 4.19, 'M0': 9.74, 'M1': 9.04}},
                {'proc_times': {'M1': 5.2, 'M0': 1.39, 'M2': 2.39}},
                {'proc_times': {'M1': 3.93, 'M0': 4.33, 'M2': 5.23}},
                {'proc_times': {'M2': 5.28, 'M0': 3.04, 'M1': 7.03}},
            ],
            2: [
                {'proc_times': {'M1': 7.14, 'M0': 2.26, 'M2': 2.8}},
                {'proc_times': {'M0': 7.35, 'M1': 8.03, 'M2': 5.13}},
                {'proc_times': {'M2': 2.03, 'M1': 7.02}},
            ],
            3: [
                {'proc_times': {'M1': 6.03, 'M2': 3.74}},
                {'proc_times': {'M1': 2.93, 'M0': 4.68}},
            ],
        })
        machine_list = ['M0', 'M1', 'M2']
        job_arrival_times = {0: 0.0, 1: 0.0, 2: 13.34, 3: 27.25}
    
    # Test the environment
    print("\n--- Testing Dynamic Actions Environment ---")
    print("Creating environment...")
    env = DynamicFJSPEnvV2(jobs_data, machine_list, job_arrival_times, reward_mode="makespan_increment")
    
    print("Resetting environment...")
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Observation space shape: {env.observation_space.shape}")
    
    step_count = 0
    total_reward = 0
    max_steps = 20  # Limit for testing
    
    print(f"\nStarting environment test (max {max_steps} steps)...")
    
    while step_count < max_steps:  # Test first 20 steps
        print(f"\n--- Step {step_count} ---")
        
        # Get valid actions
        try:
            valid_actions_info = env.get_current_valid_actions_info()
            action_mask = env.action_masks()
            
            print(f"Valid actions ({len(env.valid_actions)}):")
            for i, info in enumerate(valid_actions_info[:3]):  # Show first 3
                print(f"  {info}")
            if len(valid_actions_info) > 3:
                print(f"  ... and {len(valid_actions_info) - 3} more")
        except Exception as e:
            print(f"Error getting valid actions: {e}")
            break
        
        # Choose first valid action
        valid_indices = np.where(action_mask)[0]
        if len(valid_indices) == 0:
            print("No valid actions available!")
            break
        
        action = valid_indices[0]
        print(f"Chosen action: {action}")
        
        # Take step
        try:
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Reward: {reward:.2f}, Done: {done}")
            print(f"Operations scheduled: {info.get('operations_scheduled', 'N/A')}")
            
            step_count += 1
            
            if done:
                print(f"\nEpisode completed in {step_count} steps!")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Final makespan: {info.get('makespan', 'N/A')}")
                
                # Plot result (skip plotting for now to avoid hanging)
                print("Environment test completed successfully!")
                # plot_gantt(env.schedule, machine_list, f"Dynamic Actions Test (Makespan: {info.get('makespan', 'N/A'):.2f})")
                break
        except Exception as e:
            print(f"Error taking step: {e}")
            break
    
    if step_count >= max_steps:
        print(f"\nReached maximum test steps ({max_steps})")
        print(f"Total reward so far: {total_reward:.2f}")
        
    print("Test completed!")

if __name__ == "__main__":
    test_dynamic_actions_environment()
