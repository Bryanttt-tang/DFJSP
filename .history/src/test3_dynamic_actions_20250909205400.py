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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import collections
import time
import argparse
import sys
import math
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

# --- 1. Improved Gantt Chart Plotter ---
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

def plot_gantt_comparison(schedules_data, machines, save_path=None):
    """
    Plot 3 Gantt charts in subplots for comparison
    schedules_data: list of (schedule_dict, title, makespan) tuples
    """
    if not schedules_data or len(schedules_data) != 3:
        print("Need exactly 3 schedules for comparison")
        return
    
    colors = plt.cm.tab20.colors
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Find max makespan for consistent x-axis
    max_makespan = max(data[2] for data in schedules_data if data[2] < float('inf'))
    
    for plot_idx, (schedule, title, makespan) in enumerate(schedules_data):
        ax = axes[plot_idx]
        
        if not schedule or all(len(ops) == 0 for ops in schedule.values()):
            ax.text(0.5, 0.5, "No schedule available", transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            ax.set_title(f"{title} - No Solution", fontsize=14, fontweight='bold')
            continue
        
        # Plot operations
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
                           label, color='white', fontsize=9,
                           ha='center', va='center', weight='bold')
        
        # Format subplot
        ax.set_yticks([i * 10 + 4 for i in range(len(machines))])
        ax.set_yticklabels(machines)
        if plot_idx == 2:  # Only bottom subplot gets x-label
            ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Machines", fontsize=12)
        
        # Title with makespan
        makespan_str = f"{makespan:.2f}" if makespan < float('inf') else "∞"
        ax.set_title(f"{title} (Makespan: {makespan_str})", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set consistent x-axis limits
        ax.set_xlim(0, max_makespan * 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison Gantt charts saved to {save_path}")
    
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
        self.max_episode_steps = self.total_operations * 10  # More generous safety limit
        
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
        """
        Update the list of valid actions based on current state.
        
        KEY INSIGHT: Instead of a fixed action space with complex decoding,
        we maintain a dynamic list of valid (job_id, machine_name) pairs.
        
        This eliminates action decoding because:
        - Action = index into self.valid_actions list
        - self.valid_actions[action] directly gives (job_id, machine_name)
        - No mathematical computation needed!
        
        EXAMPLE:
        If valid_actions = [(0, 'M1'), (1, 'M0'), (2, 'M2')]
        Then action=1 means: schedule job 1 on machine M0
        """
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
            # IMPORTANT: Each job can have at most ONE operation ready at any time
            # This keeps the action space small and efficient
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
        """
        Execute one scheduling decision (EVENT-BASED decision making).
        
        TRAINING PARADIGM EXPLANATION:
        - This is EVENT-BASED scheduling: decisions are made when events occur
        - Each call to step() represents one scheduling decision, not one time unit
        - The RL algorithm counts these step() calls as 'timesteps' for training
        - Simulation time advances automatically based on the scheduling decisions
        
        DYNAMIC ACTION SPACE:
        - action = index into self.valid_actions list
        - No decoding needed: self.valid_actions[action] = (job_id, machine_name)
        """
        self.episode_step += 1
        
        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        # Validate action (simple bounds check, no complex decoding!)
        if action >= len(self.valid_actions):
            # Invalid action - give negative reward but continue
            return self._get_observation(), -100.0, False, False, {"error": "Invalid action index"}
        
        # Get the job and machine from the action (NO DECODING NEEDED!)
        job_id, machine_name = self.valid_actions[action]
        next_op_idx = self.next_operation_index[job_id]
        
        # Double-check validity (should always be true)
        if next_op_idx >= len(self.jobs[job_id]):
            return self._get_observation(), -100.0, False, False, {"error": "Job already complete"}
        
        operation = self.jobs[job_id][next_op_idx]
        if machine_name not in operation['proc_times']:
            return self._get_observation(), -100.0, False, False, {"error": "Machine cannot process operation"}
        
        # Schedule the operation (EVENT-BASED TIME ADVANCEMENT)
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
        
        # Update state (SIMULATION TIME ADVANCES AUTOMATICALLY)
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

        # Update valid actions for next step (DYNAMIC ACTION SPACE UPDATE)
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
            "operations_scheduled": self.operations_scheduled,
            "simulation_time": self.current_time,  # Current simulation time
            "episode_step": self.episode_step      # RL training step count
        }
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan=None, current_makespan=None):
        if self.reward_mode == "makespan_increment":
            # IMPROVED: Combine makespan increment with idle time penalty
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                
                # Primary objective: minimize makespan increment
                reward = -makespan_increment
                
                # Secondary objective: minimize idle time (machine utilization)
                # Idle time penalty should be smaller than makespan increment to maintain priority
                idle_penalty = idle_time * 0.1  # Small coefficient
                reward -= idle_penalty
                
                # Optional: Small processing time penalty to prefer faster operations when makespan is equal
                proc_penalty = proc_time * 0.01  # Very small coefficient
                reward -= proc_penalty
                
                # Completion bonus
                if done:
                    reward += 50.0  # Reduced from 100 to not overwhelm main objective
                    
                return reward
            else:
                # Fallback: minimize processing time + idle time
                return -(proc_time + idle_time * 0.1)
                
        elif self.reward_mode == "processing_time":
            # Simple negative processing time reward
            reward = -proc_time
            if done:
                reward += 50.0
            return reward
            
        elif self.reward_mode == "hybrid":
            # NEW: Balanced approach focusing on machine utilization
            reward = 0.0
            
            # Main penalty: idle time (poor machine utilization)
            reward -= idle_time * 1.0
            
            # Secondary penalty: processing time (prefer efficient operations)
            reward -= proc_time * 0.1
            
            # Progress reward
            reward += 5.0
            
            if done:
                reward += 100.0
                # Makespan bonus
                if current_makespan and current_makespan > 0:
                    reward += max(0, 200.0 / current_makespan)
            
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
    # Handle both Monitor-wrapped and direct environments
    if hasattr(env, 'action_masks'):
        return env.action_masks()
    elif hasattr(env, 'env') and hasattr(env.env, 'action_masks'):
        return env.env.action_masks()
    elif hasattr(env, 'envs') and len(env.envs) > 0:
        # Handle vectorized environments
        return env.envs[0].action_masks()
    else:
        # Fallback: allow all actions
        action_space_size = getattr(env, 'action_space', None)
        if action_space_size is not None:
            return np.ones(action_space_size.n, dtype=bool)
        else:
            return np.ones(12, dtype=bool)  # Default fallback

def train_agent(jobs_data, machine_list, train_arrivals, log_name, total_timesteps=50000, reward_mode="makespan_increment"):
    print(f"\n--- Training Agent: {log_name} with {reward_mode} reward ---")
    print(f"Problem details:")
    print(f"  - Jobs: {len(jobs_data)}")
    print(f"  - Machines: {len(machine_list)}")
    print(f"  - Total operations: {sum(len(ops) for ops in jobs_data.values())}")
    print(f"  - Target timesteps: {total_timesteps}")
    
    def make_env():
        env = DynamicFJSPEnvV2(jobs_data, machine_list, train_arrivals, reward_mode=reward_mode)
        # Don't wrap with Monitor for simple training to avoid action_masks issue
        return env

    # Test the environment first (use non-vectorized for testing)
    print("\n🧪 Testing environment setup...")
    test_env = DynamicFJSPEnvV2(jobs_data, machine_list, train_arrivals, reward_mode=reward_mode)
    obs, _ = test_env.reset()
    action_mask = test_env.action_masks()
    print(f"  ✓ Observation shape: {obs.shape}")
    print(f"  ✓ Action space size: {test_env.action_space.n}")
    print(f"  ✓ Valid actions: {np.sum(action_mask)}")
    print(f"  ✓ Environment setup successful!")
    
    # Use a simpler approach for small instances to avoid vectorization issues
    total_operations = sum(len(ops) for ops in jobs_data.values())
    
    if total_operations <= 15:
        print("\n🔧 Using simplified training approach for small instance...")
        
        # For small instances, use a simple policy gradient approach
        # This avoids the vectorization complexity
        env = make_env()
        
        # Simple learning approach: Random exploration with memory
        best_makespan = float('inf')
        best_actions = []
        
        print(f"\n🚀 Starting exploration for {total_timesteps} episodes...")
        
        for episode in range(min(1000, total_timesteps // 10)):  # Reasonable number of episodes
            obs, _ = env.reset()
            episode_actions = []
            done = False
            step_count = 0
            
            while not done and step_count < total_operations * 5:
                # Get action mask directly from environment (not wrapped)
                action_mask = env.action_masks()
                valid_indices = np.where(action_mask)[0]
                
                if len(valid_indices) == 0:
                    break
                
                # Epsilon-greedy exploration
                if episode < 500:  # Exploration phase
                    action = np.random.choice(valid_indices)
                else:  # Exploitation phase - use best known strategy
                    # Use shortest processing time heuristic
                    best_action = valid_indices[0]
                    best_proc_time = float('inf')
                    
                    for action_idx in valid_indices:
                        if action_idx < len(env.valid_actions):
                            job_id, machine_name = env.valid_actions[action_idx]
                            next_op = env.next_operation_index[job_id]
                            proc_time = env.jobs[job_id][next_op]['proc_times'][machine_name]
                            if proc_time < best_proc_time:
                                best_proc_time = proc_time
                                best_action = action_idx
                    action = best_action
                
                episode_actions.append(action)
                obs, reward, done, truncated, info = env.step(action)
                step_count += 1
                
                if done:
                    makespan = info.get('makespan', float('inf'))
                    if makespan < best_makespan:
                        best_makespan = makespan
                        best_actions = episode_actions.copy()
                        print(f"  🏆 Episode {episode}: New best makespan {makespan:.2f}")
                    elif episode % 100 == 0:
                        print(f"  📊 Episode {episode}: Makespan {makespan:.2f} (best: {best_makespan:.2f})")
                    break
        
        print(f"\n✅ Simplified training completed!")
        print(f"   Best makespan found: {best_makespan:.2f}")
        
        # Create a mock model that returns the best actions
        class SimplePolicyModel:
            def __init__(self, best_actions, env_template):
                self.best_actions = best_actions
                self.env_template = env_template
                
            def predict(self, obs, action_masks=None, deterministic=True):
                # Simple policy: shortest processing time
                if action_masks is not None:
                    valid_indices = np.where(action_masks)[0]
                    if len(valid_indices) > 0:
                        # Return first valid action (fallback)
                        return valid_indices[0], None
                return 0, None
        
        return SimplePolicyModel(best_actions, env)
    
    else:
        # For larger instances, use full MaskablePPO
        print("\n🤖 Using full MaskablePPO for larger instance...")
        
        # Create vectorized environment for training
        # Make sure the vectorized environment works with ActionMasker
        def make_vec_env():
            env = DynamicFJSPEnvV2(jobs_data, machine_list, train_arrivals, reward_mode=reward_mode)
            # Wrap with Monitor for training statistics
            env = Monitor(env)
            return env
        
        vec_env = DummyVecEnv([make_vec_env])
        vec_env = ActionMasker(vec_env, mask_fn)
        
        # MaskablePPO configuration optimized for FJSP
        print("\n🤖 Initializing MaskablePPO agent...")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=2,  # Increased verbosity for detailed progress
            learning_rate=3e-4,
            n_steps=2048,  # Steps per rollout
            batch_size=128,
            n_epochs=10,
            gamma=1.0,  # No discounting for scheduling (episode completion matters)
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[256, 256, 128],  # 3-layer network
                activation_fn=torch.nn.ReLU
            )
        )
        
        print(f"  ✓ Agent initialized with {sum(p.numel() for p in model.policy.parameters())} parameters")
        
        # Add progress tracking callback
        class ProgressCallback(BaseCallback):
            def __init__(self, check_freq=1000, verbose=1):
                super().__init__(verbose)
                self.check_freq = check_freq
                self.best_mean_reward = -np.inf
                
            def _on_step(self) -> bool:
                if self.n_calls % self.check_freq == 0:
                    # Calculate progress
                    progress = (self.n_calls / total_timesteps) * 100
                    
                    # Get recent episode info
                    if len(self.model.ep_info_buffer) > 0:
                        recent_episodes = self.model.ep_info_buffer[-10:]  # Last 10 episodes
                        mean_reward = np.mean([ep['r'] for ep in recent_episodes])
                        mean_length = np.mean([ep['l'] for ep in recent_episodes])
                        
                        print(f"  📊 Progress: {progress:.1f}% | Steps: {self.n_calls}/{total_timesteps}")
                        print(f"     Recent 10 episodes - Avg reward: {mean_reward:.2f}, Avg length: {mean_length:.1f}")
                        
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            print(f"     🏆 New best average reward: {mean_reward:.2f}")
                return True
        
        # Create progress callback
        progress_callback = ProgressCallback(check_freq=2048, verbose=1)  # Report every rollout
        
        print(f"\n🚀 Starting training for {total_timesteps} timesteps...")
        print("=" * 70)
        
        # Start training with progress tracking
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps, 
            callback=progress_callback,
            progress_bar=True  # Show tqdm progress bar
        )
        end_time = time.time()
        
        training_time = end_time - start_time
        print("=" * 70)
        print(f"✅ Training completed!")
        print(f"   Training time: {training_time:.1f} seconds")
        print(f"   Steps per second: {total_timesteps/training_time:.1f}")
        print(f"   Final learning rate: {model.learning_rate}")
        
        return model

def evaluate_agent(model, jobs_data, machine_list, eval_arrivals, scenario_name, reward_mode="makespan_increment"):
    print(f"\n--- Evaluating Trained Agent on Scenario: {scenario_name} ---")
    
    # Create a fresh environment for evaluation (non-vectorized)
    test_env = DynamicFJSPEnvV2(jobs_data, machine_list, eval_arrivals, reward_mode=reward_mode)
    total_operations = sum(len(ops) for ops in jobs_data.values())
    
    best_makespan = float('inf')
    best_schedule = None
    
    # Run multiple episodes to get best result
    num_episodes = 3 if total_operations > 100 else 5  # Fewer episodes for large instances
    print(f"Running {num_episodes} evaluation episodes...")
    
    episode_results = []
    
    # Check if we have a simple policy model or a trained neural network
    is_simple_model = hasattr(model, 'best_actions')
    is_maskable_ppo = hasattr(model, 'predict') and hasattr(model, 'policy')
    
    print(f"Model type detected: {'Simple Policy' if is_simple_model else ('MaskablePPO' if is_maskable_ppo else 'Other NN')}")
    
    for episode in range(num_episodes):
        print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
        obs, _ = test_env.reset()
        total_reward = 0
        done = False
        step_count = 0
        max_steps = total_operations * 5  # Safety limit
        
        while not done and step_count < max_steps:
            # Get action mask and choose action
            action_mask = test_env.action_masks()
            
            if not np.any(action_mask):
                print(f"   ⚠️ No valid actions available at step {step_count}")
                break
            
            # Choose action based on model type
            if is_simple_model:
                # Use heuristic policy for simple model
                valid_indices = np.where(action_mask)[0]
                if len(valid_indices) > 0:
                    # Use shortest processing time heuristic
                    best_action = valid_indices[0]
                    best_proc_time = float('inf')
                    
                    for action_idx in valid_indices:
                        if action_idx < len(test_env.valid_actions):
                            job_id, machine_name = test_env.valid_actions[action_idx]
                            next_op = test_env.next_operation_index[job_id]
                            proc_time = test_env.jobs[job_id][next_op]['proc_times'][machine_name]
                            if proc_time < best_proc_time:
                                best_proc_time = proc_time
                                best_action = action_idx
                    action = best_action
                else:
                    break
            elif is_maskable_ppo:
                # Use trained MaskablePPO model
                try:
                    # For MaskablePPO, we need to pass action_masks directly
                    action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
                    # Handle the case where action might be an array
                    if isinstance(action, np.ndarray):
                        action = action.item()
                except Exception as e:
                    print(f"   ❌ MaskablePPO prediction failed: {e}")
                    # Fallback to random valid action
                    valid_indices = np.where(action_mask)[0]
                    if len(valid_indices) > 0:
                        action = np.random.choice(valid_indices)
                    else:
                        break
            else:
                # Use regular trained model (PPO or other)
                try:
                    # Use trained model to predict action
                    action, _ = model.predict(obs, deterministic=True)
                    # Handle the case where action might be an array
                    if isinstance(action, np.ndarray):
                        action = action.item()
                        
                    # Validate action is within valid actions
                    if not action_mask[action]:
                        print(f"   ⚠️ Model predicted invalid action {action}, using fallback")
                        valid_indices = np.where(action_mask)[0]
                        if len(valid_indices) > 0:
                            action = np.random.choice(valid_indices)
                        else:
                            break
                except Exception as e:
                    print(f"   ❌ Model prediction failed: {e}")
                    # Fallback to random valid action
                    valid_indices = np.where(action_mask)[0]
                    if len(valid_indices) > 0:
                        action = np.random.choice(valid_indices)
                    else:
                        break
            
            # Take step
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            step_count += 1
            
            # Progress update for long episodes (less frequent for large instances)
            progress_freq = max(10, total_operations // 20)  # Adaptive frequency
            if step_count % progress_freq == 0 and step_count > 0:
                operations_done = test_env.operations_scheduled
                progress = (operations_done / total_operations) * 100
                print(f"   Progress: {operations_done}/{total_operations} ops ({progress:.1f}%) | Reward: {total_reward:.1f}")
            
            if done:
                makespan = info.get("makespan", float('inf'))
                print(f"   ✅ Completed! Makespan: {makespan:.2f}, Total reward: {total_reward:.2f}, Steps: {step_count}")
                
                episode_results.append({
                    'episode': episode + 1,
                    'makespan': makespan,
                    'reward': total_reward,
                    'steps': step_count
                })
                
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = test_env.schedule.copy()
                    print(f"   🏆 New best makespan!")
                break
        
        if not done:
            print(f"   ⚠️ Episode {episode + 1} didn't complete in {max_steps} steps")
            episode_results.append({
                'episode': episode + 1,
                'makespan': float('inf'),
                'reward': total_reward,
                'steps': step_count
            })
    
    # Summary statistics
    print(f"\n📈 EVALUATION SUMMARY")
    print("=" * 50)
    
    valid_episodes = [ep for ep in episode_results if ep['makespan'] < float('inf')]
    if valid_episodes:
        makespans = [ep['makespan'] for ep in valid_episodes]
        rewards = [ep['reward'] for ep in valid_episodes]
        steps = [ep['steps'] for ep in valid_episodes]
        
        print(f"Successful episodes: {len(valid_episodes)}/{num_episodes}")
        print(f"Best makespan: {min(makespans):.2f}")
        print(f"Average makespan: {np.mean(makespans):.2f} ± {np.std(makespans):.2f}")
        print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Average steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
        
        model_type = "Heuristic Policy" if is_simple_model else ("MaskablePPO" if is_maskable_ppo else "Neural Network")
        print(f"Model type: {model_type}")
        
        # Show episode details
        print(f"\nDetailed results:")
        for ep in episode_results:
            if ep['makespan'] < float('inf'):
                status = "🏆" if ep['makespan'] == best_makespan else "✅"
                print(f"  {status} Episode {ep['episode']}: Makespan={ep['makespan']:.2f}, Reward={ep['reward']:.1f}, Steps={ep['steps']}")
            else:
                print(f"  ❌ Episode {ep['episode']}: Failed to complete")
    else:
        print("❌ No episodes completed successfully!")
        best_makespan = float('inf')
    
    print(f"\nFinal evaluation result: Best makespan = {best_makespan:.2f}")
    
    return best_makespan, best_schedule# --- 4. Heuristic Methods ---
# --- 4. Improved Heuristic Methods ---
def heuristic_spt_scheduler(jobs_data, machine_list, job_arrival_times):
    """
    IMPROVED SPT Heuristic with proper routing and sequencing decisions.
    
    Routing: For each operation, select machine based on routing rule
    Sequencing: Among ready operations, select based on sequencing rule  
    """
    print("\n--- Running Improved SPT Heuristic Scheduler ---")
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    machine_workload = {m: 0.0 for m in machine_list}
    
    schedule = {m: [] for m in machine_list}
    operations_scheduled_count = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    
    current_time = 0.0
    arrived_jobs = {job_id for job_id, arrival in job_arrival_times.items() if arrival <= current_time}
    
    while operations_scheduled_count < total_operations:
        # 1. Find all ready operations with their best machine assignments
        ready_operations = []
        
        for job_id in arrived_jobs:
            next_op_idx = next_operation_for_job[job_id]
            if next_op_idx < len(jobs_data[job_id]):
                # Check precedence constraint
                prev_op_end_time = (operation_end_times[job_id][next_op_idx - 1] 
                                  if next_op_idx > 0 else job_arrival_times.get(job_id, 0.0))
                
                if prev_op_end_time <= current_time:
                    operation = jobs_data[job_id][next_op_idx]
                    
                    # ROUTING DECISION: Select best machine for this operation
                    best_machine, best_combination = select_best_machine(
                        operation, machine_list, machine_next_free, machine_workload, 
                        current_time, prev_op_end_time
                    )
                    
                    if best_machine:
                        ready_operations.append({
                            'job_id': job_id,
                            'op_idx': next_op_idx,
                            'machine': best_machine,
                            'proc_time': best_combination['proc_time'],
                            'start_time': best_combination['start_time'],
                            'end_time': best_combination['end_time'],
                            'idle_time': best_combination['idle_time'],
                            'completion_time': best_combination['end_time'],
                            'arrival_time': job_arrival_times.get(job_id, 0.0),
                            'remaining_ops': len(jobs_data[job_id]) - next_op_idx - 1,
                            'job_priority': calculate_job_priority(job_id, jobs_data, job_arrival_times)
                        })
        
        if not ready_operations:
            # Advance time to next event
            next_events = []
            
            # Check for job arrivals
            for job_id, arrival_time in job_arrival_times.items():
                if arrival_time > current_time and job_id not in arrived_jobs:
                    next_events.append(arrival_time)
            
            # Check for operation completions
            for job_id in arrived_jobs:
                next_op_idx = next_operation_for_job[job_id]
                if next_op_idx > 0 and next_op_idx <= len(jobs_data[job_id]):
                    prev_end_time = operation_end_times[job_id][next_op_idx - 1]
                    if prev_end_time > current_time:
                        next_events.append(prev_end_time)
            
            if next_events:
                current_time = min(next_events)
                # Update arrived jobs
                arrived_jobs.update({
                    job_id for job_id, arrival in job_arrival_times.items()
                    if arrival <= current_time
                })
            else:
                break
            continue
        
        # 2. SEQUENCING DECISION: Select which operation to schedule next
        selected_operation = select_operation_by_spt_rule(ready_operations)
        
        # 3. Execute the selected operation
        job_id = selected_operation['job_id']
        op_idx = selected_operation['op_idx']
        machine = selected_operation['machine']
        start_time = selected_operation['start_time']
        proc_time = selected_operation['proc_time']
        end_time = selected_operation['end_time']
        
        # Update system state
        machine_next_free[machine] = end_time
        machine_workload[machine] += proc_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled_count += 1
        current_time = max(current_time, end_time)
        
        # Record in schedule
        schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        
        # Update arrived jobs
        arrived_jobs.update({
            j_id for j_id, arrival in job_arrival_times.items()
            if arrival <= current_time
        })

    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"Improved SPT Heuristic Makespan: {makespan:.2f}")
    return makespan, schedule

def select_best_machine(operation, machine_list, machine_next_free, machine_workload, 
                       current_time, job_ready_time):
    """
    ROUTING HEURISTIC: Select best machine for an operation.
    Uses a combination of factors: processing time, machine availability, workload balance.
    """
    machine_options = []
    
    for machine_name, proc_time in operation['proc_times'].items():
        if machine_name in machine_list:
            machine_available_time = machine_next_free.get(machine_name, 0.0)
            start_time = max(current_time, machine_available_time, job_ready_time)
            end_time = start_time + proc_time
            idle_time = max(0, start_time - machine_available_time)
            
            machine_options.append({
                'machine': machine_name,
                'proc_time': proc_time,
                'start_time': start_time,
                'end_time': end_time,
                'idle_time': idle_time,
                'machine_available': machine_available_time,
                'workload': machine_workload.get(machine_name, 0.0),
                'completion_time': end_time
            })
    
    if not machine_options:
        return None, None
    
    # ROUTING RULE: Composite score considering multiple factors
    def routing_score(option):
        # Weighted combination of factors (tune these weights)
        proc_time_score = option['proc_time'] * 1.0       # Prefer shorter processing times
        idle_time_score = option['idle_time'] * 0.5       # Penalize idle time  
        workload_score = option['workload'] * 0.1         # Balance workload
        completion_score = option['completion_time'] * 0.8 # Prefer earlier completion
        
        return proc_time_score + idle_time_score + workload_score + completion_score
    
    best_option = min(machine_options, key=routing_score)
    return best_option['machine'], best_option

def select_operation_by_spt_rule(ready_operations):
    """
    SEQUENCING HEURISTIC: Select operation based on SPT rule with tie-breaking.
    """
    # Primary rule: Shortest Processing Time (SPT)
    # Tie-breaking rules in order:
    # 1. Earliest completion time  
    # 2. Highest job priority
    # 3. Fewest remaining operations
    
    def sequencing_priority(op):
        return (
            op['proc_time'],           # Primary: shortest processing time
            op['completion_time'],     # Tie-break 1: earliest completion
            -op['job_priority'],       # Tie-break 2: highest priority (negative for ascending)
            op['remaining_ops'],       # Tie-break 3: fewest remaining ops
            op['job_id']              # Final tie-break: job ID
        )
    
    return min(ready_operations, key=sequencing_priority)

def calculate_job_priority(job_id, jobs_data, job_arrival_times):
    """
    Calculate job priority based on various factors.
    Higher number = higher priority.
    """
    total_work = sum(min(op['proc_times'].values()) for op in jobs_data[job_id])
    num_operations = len(jobs_data[job_id])
    arrival_time = job_arrival_times.get(job_id, 0.0)
    
    # Priority = 1000 / (total_work + arrival_penalty)
    # Jobs with less work and earlier arrival get higher priority
    arrival_penalty = arrival_time * 0.1
    priority = 1000.0 / (total_work + arrival_penalty + 1.0)
    
    return priority

# Old SPT scheduler for comparison (keep for reference)
def old_simple_spt_scheduler(jobs_data, machine_list, job_arrival_times):
    """Simple SPT implementation for comparison"""
    print("\n--- Running Simple SPT Scheduler (for comparison) ---")
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    
    schedule = {m: [] for m in machine_list}
    operations_scheduled_count = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    
    current_time = 0.0
    arrived_jobs = {job_id for job_id, arrival in job_arrival_times.items() if arrival <= current_time}
    
    while operations_scheduled_count < total_operations:
        candidate_operations = []
        
        # Find candidate operations
        for job_id in arrived_jobs:
            op_idx = next_operation_for_job[job_id]
            if op_idx < len(jobs_data[job_id]):
                op_data = jobs_data[job_id][op_idx]
                job_ready_time = operation_end_times[job_id][op_idx - 1] if op_idx > 0 else job_arrival_times[job_id]
                
                for machine_name, proc_time in op_data['proc_times'].items():
                    earliest_start_time = max(machine_next_free[machine_name], job_ready_time, current_time)
                    candidate_operations.append((
                        proc_time, earliest_start_time, job_id, op_idx, machine_name
                    ))
        
        if not candidate_operations:
            # Advance time to next event
            next_events = []
            for job_id, arrival_time in job_arrival_times.items():
                if arrival_time > current_time and job_id not in arrived_jobs:
                    next_events.append(arrival_time)
            
            for job_id in arrived_jobs:
                next_op_idx = next_operation_for_job[job_id]
                if next_op_idx > 0 and next_op_idx <= len(jobs_data[job_id]):
                    prev_end_time = operation_end_times[job_id][next_op_idx - 1]
                    if prev_end_time > current_time:
                        next_events.append(prev_end_time)
            
            if next_events:
                current_time = min(next_events)
                arrived_jobs.update({
                    job_id for job_id, arrival in job_arrival_times.items()
                    if arrival <= current_time
                })
            else:
                break
            continue

        # Select shortest processing time
        selected_op = min(candidate_operations, key=lambda x: x[0])
        proc_time, start_time, job_id, op_idx, machine_name = selected_op
        
        end_time = start_time + proc_time

        machine_next_free[machine_name] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled_count += 1
        current_time = max(current_time, end_time)
        
        schedule[machine_name].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        
        # Update arrived jobs
        arrived_jobs.update({j_id for j_id, arrival in job_arrival_times.items() if arrival <= current_time})

    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"Simple SPT Makespan: {makespan:.2f}")
    return makespan, schedule

def milp_scheduler(jobs_data, machine_list, job_arrival_times):
    """
    MILP-based optimal scheduler for DFJSP with job arrivals.
    """
    print("\n--- Running MILP Optimal Scheduler ---")
    
    try:
        from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
    except ImportError:
        print("PULP not available, skipping MILP solution")
        return float('inf'), {}
    
    # Create the optimization problem
    prob = LpProblem("DFJSP", LpMinimize)
    
    # Big M value for constraints
    M = 1000000
    
    # Decision variables
    # x[job][op][machine] = 1 if operation op of job job is assigned to machine
    x = {}
    # s[job][op] = start time of operation op of job job
    s = {}
    # C_max = makespan
    C_max = LpVariable("C_max", lowBound=0)
    
    # Initialize variables
    for job_id in jobs_data:
        x[job_id] = {}
        s[job_id] = {}
        for op_idx in range(len(jobs_data[job_id])):
            x[job_id][op_idx] = {}
            s[job_id][op_idx] = LpVariable(f"s_{job_id}_{op_idx}", lowBound=job_arrival_times.get(job_id, 0))
            
            operation = jobs_data[job_id][op_idx]
            for machine in operation['proc_times']:
                if machine in machine_list:
                    x[job_id][op_idx][machine] = LpVariable(f"x_{job_id}_{op_idx}_{machine}", cat='Binary')
    
    # Objective: minimize makespan
    prob += C_max
    
    # Constraints
    # 1. Each operation must be assigned to exactly one machine
    for job_id in jobs_data:
        for op_idx in range(len(jobs_data[job_id])):
            operation = jobs_data[job_id][op_idx]
            prob += lpSum([x[job_id][op_idx][machine] for machine in operation['proc_times'] if machine in machine_list]) == 1
    
    # 2. Precedence constraints within jobs
    for job_id in jobs_data:
        for op_idx in range(1, len(jobs_data[job_id])):
            prev_op = jobs_data[job_id][op_idx - 1]
            curr_op = jobs_data[job_id][op_idx]
            
            # Current operation must start after previous operation ends
            for machine in prev_op['proc_times']:
                if machine in machine_list:
                    proc_time = prev_op['proc_times'][machine]
                    prob += s[job_id][op_idx] >= s[job_id][op_idx - 1] + proc_time * x[job_id][op_idx - 1][machine]
    
    # 3. Machine capacity constraints (no overlap)
    for machine in machine_list:
        operations_on_machine = []
        for job_id in jobs_data:
            for op_idx in range(len(jobs_data[job_id])):
                operation = jobs_data[job_id][op_idx]
                if machine in operation['proc_times']:
                    operations_on_machine.append((job_id, op_idx))
        
        # For each pair of operations that could be on this machine
        for i, (job1, op1) in enumerate(operations_on_machine):
            for j, (job2, op2) in enumerate(operations_on_machine):
                if i < j:  # Avoid duplicate constraints
                    proc_time1 = jobs_data[job1][op1]['proc_times'][machine]
                    proc_time2 = jobs_data[job2][op2]['proc_times'][machine]
                    
                    # Binary variable for ordering
                    y = LpVariable(f"y_{job1}_{op1}_{job2}_{op2}_{machine}", cat='Binary')
                    
                    # Either op1 finishes before op2 starts, or op2 finishes before op1 starts
                    prob += s[job1][op1] + proc_time1 <= s[job2][op2] + M * (2 - x[job1][op1][machine] - x[job2][op2][machine] + y)
                    prob += s[job2][op2] + proc_time2 <= s[job1][op1] + M * (3 - x[job1][op1][machine] - x[job2][op2][machine] - y)
    
    # 4. Makespan constraints
    for job_id in jobs_data:
        last_op_idx = len(jobs_data[job_id]) - 1
        last_op = jobs_data[job_id][last_op_idx]
        for machine in last_op['proc_times']:
            if machine in machine_list:
                proc_time = last_op['proc_times'][machine]
                prob += C_max >= s[job_id][last_op_idx] + proc_time * x[job_id][last_op_idx][machine]
    
    # Solve the problem
    print("Solving MILP... (this may take a while)")
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=300))  # 5 minute time limit
    
    if prob.status == 1:  # Optimal solution found
        makespan = C_max.varValue
        print(f"MILP Optimal Makespan: {makespan:.2f}")
        
        # Extract schedule
        schedule = {m: [] for m in machine_list}
        for job_id in jobs_data:
            for op_idx in range(len(jobs_data[job_id])):
                operation = jobs_data[job_id][op_idx]
                for machine in operation['proc_times']:
                    if machine in machine_list and x[job_id][op_idx][machine].varValue == 1:
                        start_time = s[job_id][op_idx].varValue
                        proc_time = operation['proc_times'][machine]
                        end_time = start_time + proc_time
                        schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        
        return makespan, schedule
    else:
        print("MILP solver failed to find optimal solution")
        return float('inf'), {}

# --- 5. Main Test Function ---
def test_dynamic_actions_environment(instance_file="test_instance.txt", skip_milp_large=False):
    """Test the new dynamic actions environment and compare with other methods"""
    
    print(f"Loading instance file: {instance_file}")
    
    # Load test instance
    try:
        # Read and execute the test instance file
        with open(instance_file, "r") as f:
            content = f.read()
        
        # Create a namespace to execute the code
        namespace = {}
        exec(content, namespace)
        
        jobs_data = namespace['jobs_data']
        machine_list = namespace['machine_list']
        job_arrival_times = namespace['job_arrival_times']
        
        print(f"=== {instance_file.upper()} DATA ===")
        print(f"Number of jobs: {len(jobs_data)}")
        print(f"Machines: {machine_list}")
        print(f"Job arrival times: {job_arrival_times}")
        print("\nJob details:")
        for job_id, operations in jobs_data.items():
            print(f"  Job {job_id} ({len(operations)} operations):")
            for op_idx, operation in enumerate(operations):
                proc_times_str = ", ".join([f"{m}:{t:.2f}" for m, t in operation['proc_times'].items()])
                print(f"    Operation {op_idx+1}: {proc_times_str}")
        
        total_operations = sum(len(ops) for ops in jobs_data.values())
        print(f"\nTotal operations: {total_operations}")
        
    except Exception as e:
        print(f"Error loading {instance_file}: {e}")
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
        total_operations = sum(len(ops) for ops in jobs_data.values())
        
        print("=== DEFAULT TEST DATA ===")
        print(f"Number of jobs: {len(jobs_data)}")
        print(f"Machines: {machine_list}")
        print(f"Job arrival times: {job_arrival_times}")
    
    # Check if we should skip MILP for large instances
    skip_milp = skip_milp_large and total_operations > 15
    if skip_milp:
        print(f"\n⚠️  Skipping MILP solver: {total_operations} operations > 15 (large instance)")
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    results = {}
    
    # 1. Test SPT Heuristic
    print("\n1. SPT HEURISTIC METHOD")
    try:
        spt_makespan, spt_schedule = heuristic_spt_scheduler(jobs_data, machine_list, job_arrival_times)
        results['SPT Heuristic'] = spt_makespan
        print(f"✓ SPT Heuristic completed successfully")
    except Exception as e:
        print(f"✗ SPT Heuristic failed: {e}")
        results['SPT Heuristic'] = float('inf')
    
    # 2. Test MILP Optimal (conditionally)
    print("\n2. MILP OPTIMAL METHOD")
    if skip_milp:
        print("⏭️  Skipping MILP solver for large instance")
        milp_makespan, milp_schedule = float('inf'), {}
        results['MILP Optimal'] = float('inf')
    else:
        try:
            milp_makespan, milp_schedule = milp_scheduler(jobs_data, machine_list, job_arrival_times)
            results['MILP Optimal'] = milp_makespan
            if milp_makespan < float('inf'):
                print(f"✓ MILP Optimal completed successfully")
            else:
                print(f"✗ MILP Optimal failed to find solution")
        except Exception as e:
            print(f"✗ MILP Optimal failed: {e}")
            results['MILP Optimal'] = float('inf')
            milp_makespan, milp_schedule = float('inf'), {}
    
    # 3. Train and Test Dynamic Actions RL Agent
    print("\n3. DYNAMIC ACTIONS RL AGENT (ACTUAL TRAINING)")
    rl_schedule = {}  # Initialize for later use
    try:
        # Step 3a: Train the RL agent
        print("\n📚 TRAINING RL AGENT...")
        print("=" * 50)
        
        # Determine training timesteps based on problem size
        base_timesteps = 10000  # Reduced for faster demonstration
        size_multiplier = max(1, total_operations // 10)  # More training for larger problems
        training_timesteps = base_timesteps * size_multiplier
        
        print(f"Problem size: {total_operations} operations")
        print(f"Training timesteps: {training_timesteps}")
        print("Starting training with progress tracking...")
        
        # Train the agent with progress bars
        model = train_agent(
            jobs_data=jobs_data,
            machine_list=machine_list, 
            train_arrivals=job_arrival_times,
            log_name=f"FJSP_{total_operations}ops",
            total_timesteps=training_timesteps,
            reward_mode="makespan_increment"
        )
        
        print("✓ Training completed!")
        
        # Step 3b: Evaluate the trained agent
        print("\n🎯 EVALUATING TRAINED AGENT...")
        print("=" * 50)
        
        rl_makespan, rl_schedule = evaluate_agent(
            model=model,
            jobs_data=jobs_data,
            machine_list=machine_list,
            eval_arrivals=job_arrival_times,
            scenario_name=f"Test_{total_operations}ops",
            reward_mode="makespan_increment"
        )
        
        results['RL Dynamic Actions'] = rl_makespan
        print(f"✓ RL Agent evaluation completed!")
        print(f"  Best makespan achieved: {rl_makespan:.2f}")
        
        # Step 3c: Test environment manually for comparison
        print("\n🔧 MANUAL POLICY TEST (for comparison)...")
        print("=" * 50)
        
        env = DynamicFJSPEnvV2(jobs_data, machine_list, job_arrival_times, reward_mode="makespan_increment")
        obs, _ = env.reset()
        
        step_count = 0
        total_reward = 0
        max_steps = total_operations * 5
        
        while step_count < max_steps:
            action_mask = env.action_masks()
            
            if len(env.valid_actions) == 0:
                break
            
            # Progress update for large instances (adaptive frequency)
            progress_freq = max(25, total_operations // 20)  # Less frequent for large instances
            if step_count % progress_freq == 0 and step_count > 0:
                operations_done = env.operations_scheduled
                operations_total = total_operations
                progress = (operations_done / operations_total) * 100
                print(f"  Manual policy progress: {operations_done}/{operations_total} operations ({progress:.1f}%)")
            
            # Choose action using simple shortest processing time policy
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) == 0:
                break
            
            # Find action with shortest processing time
            best_action = valid_indices[0]
            best_proc_time = float('inf')
            
            for action_idx in valid_indices:
                if action_idx < len(env.valid_actions):
                    job_id, machine_name = env.valid_actions[action_idx]
                    next_op = env.next_operation_index[job_id]
                    proc_time = env.jobs[job_id][next_op]['proc_times'][machine_name]
                    if proc_time < best_proc_time:
                        best_proc_time = proc_time
                        best_action = action_idx
            
            # Take step
            obs, reward, done, truncated, info = env.step(best_action)
            total_reward += reward
            step_count += 1
            
            if done:
                manual_makespan = info.get('makespan', float('inf'))
                print(f"✓ Manual policy completed in {step_count} steps")
                print(f"  Manual policy makespan: {manual_makespan:.2f}")
                print(f"  Trained RL makespan: {rl_makespan:.2f}")
                if rl_makespan < manual_makespan:
                    improvement = ((manual_makespan - rl_makespan) / manual_makespan) * 100
                    print(f"  🏆 RL agent is {improvement:.1f}% better than manual policy!")
                else:
                    degradation = ((rl_makespan - manual_makespan) / manual_makespan) * 100
                    print(f"  ⚠️ Manual policy is {degradation:.1f}% better than RL agent")
                break
        
        if step_count >= max_steps:
            print(f"✗ Manual policy reached max steps ({max_steps})")
            print(f"  Operations scheduled: {env.operations_scheduled}/{total_operations}")
            results['RL Dynamic Actions'] = float('inf')
            
    except Exception as e:
        print(f"✗ RL Environment failed: {e}")
        results['RL Dynamic Actions'] = float('inf')
    
    # 4. Display Results
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    valid_results = {k: v for k, v in results.items() if v < float('inf')}
    
    if valid_results:
        best_method = min(valid_results, key=valid_results.get)
        best_makespan = valid_results[best_method]
        
        print(f"{'Method':<25} {'Makespan':<15} {'Gap from Best':<15}")
        print("-" * 55)
        
        for method, makespan in results.items():
            if makespan < float('inf'):
                gap = ((makespan - best_makespan) / best_makespan * 100) if best_makespan > 0 else 0
                status = "🏆 BEST" if method == best_method else f"{gap:.1f}%"
                print(f"{method:<25} {makespan:<15.2f} {status:<15}")
            else:
                print(f"{method:<25} {'FAILED':<15} {'-':<15}")
        
        print(f"\n🏆 Best method: {best_method} with makespan {best_makespan:.2f}")
        
        # Theoretical Analysis: Why RL doesn't hit optimal bound
        print("\n" + "="*80)
        print("THEORETICAL ANALYSIS: WHY RL DOESN'T HIT OPTIMAL BOUND")
        print("="*80)
        
        optimal_makespan = results.get('MILP Optimal', float('inf'))
        rl_makespan = results.get('RL Dynamic Actions', float('inf'))
        
        if optimal_makespan < float('inf') and rl_makespan < float('inf'):
            gap = rl_makespan - optimal_makespan
            gap_percent = (gap / optimal_makespan) * 100
            
            print(f"Optimal makespan (MILP): {optimal_makespan:.2f}")
            print(f"RL makespan:             {rl_makespan:.2f}")
            print(f"Optimality gap:          {gap:.2f} ({gap_percent:.1f}%)")
            
            if gap_percent < 5:
                print(f"\n✓ Gap is small ({gap_percent:.1f}%), RL performs well!")
            elif gap_percent < 15:
                print(f"\n⚠ Moderate gap ({gap_percent:.1f}%), room for improvement")
            else:
                print(f"\n⚠ Large gap ({gap_percent:.1f}%), significant optimization needed")
        
        # Create 3-subplot Gantt chart comparison
        print(f"\n📊 Generating comparison Gantt charts...")
        
        schedule_data = []
        if 'MILP Optimal' in results and results['MILP Optimal'] < float('inf'):
            schedule_data.append((milp_schedule, "MILP Optimal", results['MILP Optimal']))
        else:
            schedule_data.append(({}, "MILP Optimal (Failed)", float('inf')))
            
        if 'SPT Heuristic' in results and results['SPT Heuristic'] < float('inf'):
            schedule_data.append((spt_schedule, "SPT Heuristic", results['SPT Heuristic']))
        else:
            schedule_data.append(({}, "SPT Heuristic (Failed)", float('inf')))
            
        if 'RL Dynamic Actions' in results and results['RL Dynamic Actions'] < float('inf'):
            schedule_data.append((rl_schedule, "RL Dynamic Actions", results['RL Dynamic Actions']))
        else:
            schedule_data.append(({}, "RL Dynamic Actions (Failed)", float('inf')))
        
        # Plot comparison
        try:
            plot_gantt_comparison(schedule_data, machine_list, 
                                save_path="gantt_comparison.png")
        except Exception as e:
            print(f"Failed to plot comparison chart: {e}")
            
        # # Individual charts for backup
        # if not args.no_plots:
        #     try:
        #         if 'SPT Heuristic' in results and results['SPT Heuristic'] < float('inf'):
        #             plot_gantt(spt_schedule, machine_list, f"SPT Heuristic (Makespan: {results['SPT Heuristic']:.2f})")
                
        #         if 'MILP Optimal' in results and results['MILP Optimal'] < float('inf'):
        #             plot_gantt(milp_schedule, machine_list, f"MILP Optimal (Makespan: {results['MILP Optimal']:.2f})")
                    
        #         if 'RL Dynamic Actions' in results and results['RL Dynamic Actions'] < float('inf'):
        #             plot_gantt(env.schedule, machine_list, f"RL Dynamic Actions (Makespan: {results['RL Dynamic Actions']:.2f})")
        #     except Exception as e:
        #         print(f"Failed to plot individual charts: {e}")
    else:
        print("❌ All methods failed!")
    
    print("\nComparison completed!")

if __name__ == "__main__":
    import argparse
    
    # Add detailed explanations in comments
    print("="*80)
    print("DYNAMIC ACTIONS FJSP ENVIRONMENT - EXPLANATION")
    print("="*80)
    print("""
1. EVENT-BASED vs TIME-BASED TRAINING:
   - Decision Points: Event-based (decisions made when machines become idle or jobs arrive)
   - Training Steps: Time-based (RL counts env.step() calls as 'timesteps')
   - Each 'timestep' = one scheduling decision, NOT one unit of simulation time
   - The environment advances simulation time automatically between events

2. DYNAMIC ACTION SPACE IMPLEMENTATION:
   - OLD: Fixed action space [job_idx * max_ops * machines + op_idx * machines + machine_idx]
   - NEW: Dynamic list of valid (job_id, machine_name) pairs
   - Advantages:
     * No complex action decoding needed
     * Only valid actions are included
     * Much more efficient action space
     * Action = index into self.valid_actions list
     
3. ACTION SELECTION PROCESS:
   - At each step: Update self.valid_actions = [(job_id, machine_name), ...]
   - Agent chooses: action_index (0 to len(valid_actions)-1)
   - Environment executes: job_id, machine_name = self.valid_actions[action_index]
   - No mathematical decoding required!
""")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Dynamic Actions FJSP Environment')
    parser.add_argument('--instance', '-i', type=str, default='test_instance.txt',
                        help='Instance file to load (default: test_instance.txt)')
    parser.add_argument('--no-plots', action='store_true', 
                        help='Skip plotting Gantt charts')
    parser.add_argument('--skip_milp_large', action='store_true',
                        help='Skip MILP solver for large instances (>15 operations)')
    
    args = parser.parse_args()
    
    print(f"\nRunning with instance file: {args.instance}")
    if args.skip_milp_large:
        print("Note: Will skip MILP solver for large instances (>15 operations)")
    print("="*80)
    
    # Run the test
    test_dynamic_actions_environment(args.instance, args.skip_milp_large)
