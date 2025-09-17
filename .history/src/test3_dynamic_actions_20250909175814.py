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
        gamma=1,
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

# --- 4. Heuristic Methods ---
def heuristic_spt_scheduler(jobs_data, machine_list, job_arrival_times):
    """
    Schedules jobs based on the Shortest Processing Time (SPT) heuristic,
    considering dynamic job arrivals.
    """
    print("\n--- Running SPT Heuristic Scheduler ---")
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    
    schedule = {m: [] for m in machine_list}
    operations_scheduled_count = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    
    arrived_jobs = {job_id for job_id, arrival in job_arrival_times.items() if arrival <= 0}
    current_time = 0.0
    
    while operations_scheduled_count < total_operations:
        # Find all ready operations
        ready_operations = []
        
        for job_id in arrived_jobs:
            next_op_idx = next_operation_for_job[job_id]
            if next_op_idx < len(jobs_data[job_id]):
                # Check precedence constraint
                if next_op_idx == 0 or operation_end_times[job_id][next_op_idx - 1] <= current_time:
                    operation = jobs_data[job_id][next_op_idx]
                    
                    # Find best machine for this operation (shortest processing time)
                    best_machine = None
                    best_proc_time = float('inf')
                    
                    for machine_name in operation['proc_times']:
                        if machine_name in machine_list:
                            proc_time = operation['proc_times'][machine_name]
                            if proc_time < best_proc_time:
                                best_proc_time = proc_time
                                best_machine = machine_name
                    
                    if best_machine:
                        ready_operations.append((job_id, next_op_idx, best_machine, best_proc_time))
        
        if not ready_operations:
            # Advance time to next job arrival or operation completion
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
        
        # Sort by processing time (SPT rule)
        ready_operations.sort(key=lambda x: x[3])
        
        # Schedule the operation with shortest processing time
        job_id, op_idx, machine_name, proc_time = ready_operations[0]
        
        # Calculate start time
        machine_available_time = machine_next_free.get(machine_name, 0.0)
        precedence_time = operation_end_times[job_id][op_idx - 1] if op_idx > 0 else job_arrival_times.get(job_id, 0.0)
        start_time = max(current_time, machine_available_time, precedence_time)
        end_time = start_time + proc_time
        
        # Update state
        machine_next_free[machine_name] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled_count += 1
        current_time = max(current_time, end_time)
        
        # Record in schedule
        schedule[machine_name].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        
        # Update arrived jobs
        arrived_jobs.update({
            job_id for job_id, arrival in job_arrival_times.items()
            if arrival <= current_time
        })

    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"SPT Heuristic Makespan: {makespan:.2f}")
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
def test_dynamic_actions_environment(instance_file="test_instance.txt"):
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
        
        print(f"\nTotal operations: {sum(len(ops) for ops in jobs_data.values())}")
        
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
        
        print("=== DEFAULT TEST DATA ===")
        print(f"Number of jobs: {len(jobs_data)}")
        print(f"Machines: {machine_list}")
        print(f"Job arrival times: {job_arrival_times}")
    
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
    
    # 2. Test MILP Optimal
    print("\n2. MILP OPTIMAL METHOD")
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
    
    # 3. Test Dynamic Actions RL Environment (manual policy)
    print("\n3. DYNAMIC ACTIONS RL ENVIRONMENT (Manual Policy)")
    try:
        env = DynamicFJSPEnvV2(jobs_data, machine_list, job_arrival_times, reward_mode="makespan_increment")
        
        obs, _ = env.reset()
        print(f"Environment initialized - Action space: {env.action_space.n}, Obs space: {env.observation_space.shape}")
        
        step_count = 0
        total_reward = 0
        max_steps = 100  # Safety limit
        
        while step_count < max_steps:
            # Get valid actions
            valid_actions_info = env.get_current_valid_actions_info()
            action_mask = env.action_masks()
            
            if len(env.valid_actions) == 0:
                print("No more valid actions available")
                break
            
            # Choose action using a simple policy (shortest processing time)
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) == 0:
                print("No valid actions in mask!")
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
                rl_makespan = info.get('makespan', float('inf'))
                results['RL Dynamic Actions'] = rl_makespan
                print(f"✓ RL Environment completed in {step_count} steps")
                print(f"  Final makespan: {rl_makespan:.2f}")
                print(f"  Total reward: {total_reward:.2f}")
                break
        
        if step_count >= max_steps:
            print(f"✗ RL Environment reached max steps ({max_steps})")
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
        
        # Show Gantt charts for successful methods
        print("\nGenerating Gantt charts...")
        
        if 'SPT Heuristic' in results and results['SPT Heuristic'] < float('inf'):
            try:
                plot_gantt(spt_schedule, machine_list, f"SPT Heuristic (Makespan: {results['SPT Heuristic']:.2f})")
            except:
                print("Failed to plot SPT Gantt chart")
        
        if 'MILP Optimal' in results and results['MILP Optimal'] < float('inf'):
            try:
                plot_gantt(milp_schedule, machine_list, f"MILP Optimal (Makespan: {results['MILP Optimal']:.2f})")
            except:
                print("Failed to plot MILP Gantt chart")
                
        if 'RL Dynamic Actions' in results and results['RL Dynamic Actions'] < float('inf'):
            try:
                plot_gantt(env.schedule, machine_list, f"RL Dynamic Actions (Makespan: {results['RL Dynamic Actions']:.2f})")
            except:
                print("Failed to plot RL Gantt chart")
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
    
    args = parser.parse_args()
    
    print(f"\nRunning with instance file: {args.instance}")
    print("="*80)
    
    # Run the test
    test_dynamic_actions_environment(args.instance)
