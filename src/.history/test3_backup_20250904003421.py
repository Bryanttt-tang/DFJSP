import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
import random
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
import os
import collections
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import argparse
import importlib.util
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- Default Instance Data ---
DEFAULT_JOBS_DATA = collections.OrderedDict({
    0: [{'proc_times': {'M0': 5, 'M1': 7}}, {'proc_times': {'M1': 6, 'M2': 4}}, {'proc_times': {'M0': 3}}],
    1: [{'proc_times': {'M1': 8, 'M2': 6}}, {'proc_times': {'M0': 5}}, {'proc_times': {'M1': 4, 'M2': 5}}],
    2: [{'proc_times': {'M0': 6, 'M2': 7}}, {'proc_times': {'M0': 4, 'M1': 5}}, {'proc_times': {'M2': 8}}],
    3: [{'proc_times': {'M1': 9}}, {'proc_times': {'M2': 3}}, {'proc_times': {'M0': 6, 'M1': 7}}]
})
DEFAULT_MACHINE_LIST = ['M0', 'M1', 'M2']
DEFAULT_ARRIVAL_TIMES = {0: 0, 1: 0, 2: 10, 3: 15}

# --- Extended 7-Job Instance Data ---
EXTENDED_JOBS_DATA = collections.OrderedDict({
    0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}, {'proc_times': {'M0': 2}}],
    1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M1': 3, 'M2': 4}}],
    2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M0': 3, 'M1': 4}}, {'proc_times': {'M2': 7}}],
    3: [{'proc_times': {'M1': 8}}, {'proc_times': {'M2': 2}}, {'proc_times': {'M0': 5, 'M1': 6}}],
    4: [{'proc_times': {'M0': 6, 'M1': 9}}, {'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M2': 6}}],
    5: [{'proc_times': {'M1': 5, 'M2': 8}}, {'proc_times': {'M0': 6}}, {'proc_times': {'M1': 4, 'M2': 3}}],
    6: [{'proc_times': {'M0': 7, 'M2': 4}}, {'proc_times': {'M0': 5, 'M1': 6}}, {'proc_times': {'M1': 3}}, {'proc_times': {'M0': 2, 'M2': 5}}]
})
EXTENDED_MACHINE_LIST = ['M0', 'M1', 'M2']
EXTENDED_ARRIVAL_TIMES = {0: 0, 1: 0, 2: 0, 3: 10, 4: 15, 5: 25, 6: 35}

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

# --- 2. Dynamic RL Environment ---
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()

# ...existing code...

class DynamicFJSPEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, job_arrival_times=None):
        super().__init__()
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        
        if job_arrival_times is None:
            self.job_arrival_times = {job_id: 0 for job_id in self.job_ids}
        else:
            self.job_arrival_times = job_arrival_times

        self.action_space = spaces.Discrete(
            self.num_jobs * self.max_ops_per_job * len(self.machines)
        )
        
        obs_size = (
            len(self.machines) +  # Machine availability
            self.num_jobs * self.max_ops_per_job + # Operation completion
            self.num_jobs + # Job progress
            self.num_jobs + # Job arrival status
            1 # Current makespan
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        self.current_makespan = 0.0
        self.operations_scheduled = 0
        
        # Handle job arrivals at time 0
        self.arrived_jobs = {
            job_id for job_id, arrival_time in self.job_arrival_times.items()
            if arrival_time <= 0
        }
        
        return self._get_observation(), {}

    def _decode_action(self, action):
        num_machines = len(self.machines)
        ops_per_job = self.max_ops_per_job
        
        job_idx = action // (ops_per_job * num_machines)
        op_idx = (action % (ops_per_job * num_machines)) // num_machines
        machine_idx = action % num_machines
        
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        # Check basic bounds
        if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
            return False
        
        job_id = self.job_ids[job_idx]
        
        # Check if job has arrived
        if job_id not in self.arrived_jobs:
            return False
        
        # Check if operation index is valid for this specific job
        if not (0 <= op_idx < len(self.jobs[job_id])):
            return False
            
        # Check if this is the next operation to be scheduled for this job
        if op_idx != self.next_operation[job_id]:
            return False
            
        # Check if the machine can process this operation
        machine_name = self.machines[machine_idx]
        if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
            return False
            
        return True

    def action_masks(self):
        mask = np.full(self.action_space.n, False, dtype=bool)
        
        if self.operations_scheduled >= self.total_operations:
            return mask

        for job_idx, job_id in enumerate(self.job_ids):
            if job_id not in self.arrived_jobs:
                continue

            op_idx = self.next_operation[job_id]
            
            # Only consider valid operation indices for this job
            if op_idx < len(self.jobs[job_id]):
                op_data = self.jobs[job_id][op_idx]
                
                for machine_idx, machine_name in enumerate(self.machines):
                    if machine_name in op_data['proc_times']:
                        action = job_idx * (self.max_ops_per_job * len(self.machines)) + \
                                 op_idx * len(self.machines) + machine_idx
                        
                        if action < self.action_space.n:
                            mask[action] = True
        
        # Debug: Print mask information
        valid_actions = np.sum(mask)
        if valid_actions == 0:
            print(f"WARNING: No valid actions available!")
            print(f"Operations scheduled: {self.operations_scheduled}/{self.total_operations}")
            print(f"Arrived jobs: {self.arrived_jobs}")
            print(f"Next operations: {self.next_operation}")
            for job_id in self.job_ids:
                if job_id in self.arrived_jobs:
                    next_op = self.next_operation[job_id]
                    if next_op < len(self.jobs[job_id]):
                        print(f"Job {job_id}, Op {next_op}: {self.jobs[job_id][next_op]['proc_times']}")
        
        return mask

    def step(self, action):
        job_idx, op_idx, machine_idx = self._decode_action(action)

        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            print(f"Invalid action: job_idx={job_idx}, op_idx={op_idx}, machine_idx={machine_idx}")
            print(f"Available jobs: {list(self.arrived_jobs)}")
            print(f"Next operations: {self.next_operation}")
            return self._get_observation(), -100.0, True, False, {"error": "Invalid action taken"}

        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        
        machine_available_time = self.machine_next_free[machine]
        job_ready_time = self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 else self.job_arrival_times[job_id]
        
        start_time = max(machine_available_time, job_ready_time)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time

        self.machine_next_free[machine] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        
        # Update makespan and check for new arrivals
        previous_makespan = self.current_makespan
        self.current_makespan = max(self.current_makespan, end_time)
        
        newly_arrived = {
            j_id for j_id, arrival in self.job_arrival_times.items()
            if previous_makespan < arrival <= self.current_makespan
        }
        self.arrived_jobs.update(newly_arrived)

        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        terminated = self.operations_scheduled == self.total_operations
        reward = self._calculate_reward(proc_time, start_time - machine_available_time, terminated)
        
        info = {"makespan": self.current_makespan}
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done):
        reward = -proc_time - 2 * idle_time
        if done:
            reward += - self.current_makespan  
        return reward

    def _get_observation(self):
        norm_factor = max(self.current_makespan, 100.0)
        
        obs = []
        
        # Machine availability (normalized)
        obs.extend([self.machine_next_free[m] / norm_factor for m in self.machines])
        
        # Operation completion status (padded to max_ops_per_job)
        for job_id in self.job_ids:
            for op_idx in range(self.max_ops_per_job):
                if op_idx < len(self.jobs[job_id]):
                    completed = 1.0 if self.completed_ops[job_id][op_idx] else 0.0
                else:
                    completed = 1.0  # Non-existent operations are considered "completed"
                obs.append(completed)
        
        # Job progress (proportion of operations completed)
        for job_id in self.job_ids:
            total_ops = len(self.jobs[job_id])
            progress = self.next_operation[job_id] / total_ops if total_ops > 0 else 1.0
            obs.append(progress)
            
        # Job arrival status
        for job_id in self.job_ids:
            obs.append(1.0 if job_id in self.arrived_jobs else 0.0)
            
        # Current makespan (normalized)
        obs.append(self.current_makespan / norm_factor)
        
        return np.array(obs, dtype=np.float32)

class GraphDFJSPEnv(gym.Env):
    """
    Graph-based DFJSP environment using node features and adjacency matrices.
    More scalable and expressive than fixed-size observation spaces.
    """
    
    def __init__(self, jobs_data, machine_list, job_arrival_times=None):
        super().__init__()
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        self.num_machines = len(self.machines)
        self.job_arrival_times = job_arrival_times or {j: 0 for j in self.job_ids}
        
        # Create operation list and mappings
        self.operations = []  # [(job_id, op_idx), ...]
        self.job_to_ops = {}  # job_id -> [op_indices]
        
        op_idx = 0
        for job_id in self.job_ids:
            self.job_to_ops[job_id] = []
            for op_pos in range(len(self.jobs[job_id])):
                self.operations.append((job_id, op_pos))
                self.job_to_ops[job_id].append(op_idx)
                op_idx += 1
        
        self.num_operations = len(self.operations)
        
        # Action space: select (operation, machine) pair
        valid_pairs = []
        for op_idx, (job_id, op_pos) in enumerate(self.operations):
            for machine in self.jobs[job_id][op_pos]['proc_times']:
                valid_pairs.append((op_idx, machine))
        
        self.valid_action_pairs = valid_pairs
        self.action_space = spaces.Discrete(len(valid_pairs))
        
        # Graph-based observation space with enhanced features
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(low=0, high=1, 
                                      shape=(self.num_operations + self.num_machines, 8), 
                                      dtype=np.float32),
            'adjacency': spaces.Box(low=0, high=1, 
                                  shape=(self.num_operations + self.num_machines, 
                                        self.num_operations + self.num_machines),
                                  dtype=np.float32),
            'global_features': spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        })
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            
        # State tracking
        self.current_time = 0.0
        self.machine_available_time = {m: 0.0 for m in self.machines}
        self.operation_status = ['waiting'] * self.num_operations  # 'waiting', 'ready', 'running', 'done'
        self.operation_start_time = [None] * self.num_operations
        self.operation_end_time = [None] * self.num_operations
        self.job_next_op = {job_id: 0 for job_id in self.job_ids}
        self.schedule = {m: [] for m in self.machines}
        
        # Initialize arrived jobs (including jobs arriving at time 0)
        self.arrived_jobs = {
            job_id for job_id, arrival_time in self.job_arrival_times.items()
            if arrival_time <= self.current_time
        }
        
        # Update initial ready operations
        self._update_ready_operations()
        
        return self._get_observation(), {}
    
    def _update_ready_operations(self):
        """Update which operations are ready to be scheduled"""
        for job_id in self.job_ids:
            # Check if job has arrived
            if self.job_arrival_times[job_id] <= self.current_time:
                next_op_pos = self.job_next_op[job_id]
                if next_op_pos < len(self.jobs[job_id]):
                    op_idx = self.job_to_ops[job_id][next_op_pos]
                    
                    if self.operation_status[op_idx] == 'waiting':
                        # Check if previous operation is done
                        if next_op_pos == 0:  # First operation
                            self.operation_status[op_idx] = 'ready'
                        elif next_op_pos > 0:
                            prev_op_idx = self.job_to_ops[job_id][next_op_pos - 1]
                            if self.operation_status[prev_op_idx] == 'done':
                                self.operation_status[op_idx] = 'ready'
    
    def _get_valid_actions(self):
        """Get mask of valid actions"""
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # If no operations are ready, advance time to next arrival
        if not any(self.operation_status[i] == 'ready' for i in range(self.num_operations)):
            # Find next job arrival
            future_arrivals = [
                arrival for arrival in self.job_arrival_times.values() 
                if arrival > self.current_time
            ]
            
            if future_arrivals:
                next_time = min(future_arrivals)
                self.current_time = next_time
                
                # Update arrived jobs
                self.arrived_jobs.update({
                    job_id for job_id, arrival_time in self.job_arrival_times.items()
                    if arrival_time <= self.current_time
                })
                
                # Update ready operations
                self._update_ready_operations()
        
        # Mark valid actions
        for action_idx, (op_idx, machine) in enumerate(self.valid_action_pairs):
            job_id, op_pos = self.operations[op_idx]
            
            # Check if operation is ready and machine is available
            if (self.operation_status[op_idx] == 'ready' and
                machine in self.jobs[job_id][op_pos]['proc_times']):
                mask[action_idx] = True
                
        return mask
    
    def step(self, action):
        if not self._get_valid_actions()[action]:
            return self._get_observation(), -1000, True, False, {"error": "Invalid action"}
        
        op_idx, machine = self.valid_action_pairs[action]
        job_id, op_pos = self.operations[op_idx]
        
        # Schedule the operation
        proc_time = self.jobs[job_id][op_pos]['proc_times'][machine]
        start_time = max(self.current_time, self.machine_available_time[machine])
        
        # Handle precedence constraints
        if op_pos > 0:
            prev_op_idx = self.job_to_ops[job_id][op_pos - 1]
            if self.operation_end_time[prev_op_idx] is not None:
                start_time = max(start_time, self.operation_end_time[prev_op_idx])
        
        end_time = start_time + proc_time
        
        # Update state
        self.operation_status[op_idx] = 'done'
        self.operation_start_time[op_idx] = start_time
        self.operation_end_time[op_idx] = end_time
        self.machine_available_time[machine] = end_time
        self.current_time = max(self.current_time, end_time)
        self.job_next_op[job_id] += 1
        
        self.schedule[machine].append((f"J{job_id}-O{op_pos+1}", start_time, end_time))
        
        # Update arrived jobs based on current time
        self.arrived_jobs.update({
            job_id for job_id, arrival_time in self.job_arrival_times.items()
            if arrival_time <= self.current_time
        })
        
        # Update ready operations
        self._update_ready_operations()
        
        # Check if done
        done = all(status == 'done' for status in self.operation_status)
        
        # Calculate reward (fixed idle time calculation)
        idle_time = start_time - self.machine_available_time.get(machine, 0)
        reward = self._calculate_reward(proc_time, idle_time, done)
        
        info = {"makespan": self.current_time}
        return self._get_observation(), reward, done, False, info
    
    def _get_observation(self):
        """Create graph-based observation"""
        num_nodes = self.num_operations + self.num_machines
        node_features = np.zeros((num_nodes, 8), dtype=np.float32)
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # Operation node features (8 features per operation)
        for op_idx, (job_id, op_pos) in enumerate(self.operations):
            status = self.operation_status[op_idx]
            
            # Status indicators
            node_features[op_idx, 0] = 1.0 if status == 'waiting' else 0.0
            node_features[op_idx, 1] = 1.0 if status == 'ready' else 0.0
            node_features[op_idx, 2] = 1.0 if status == 'done' else 0.0
            
            # Processing time features
            proc_times = list(self.jobs[job_id][op_pos]['proc_times'].values())
            min_proc_time = min(proc_times)
            avg_proc_time = np.mean(proc_times)
            node_features[op_idx, 3] = min_proc_time / 30.0  # Normalized min processing time
            node_features[op_idx, 4] = len(proc_times) / self.num_machines  # Machine flexibility
            
            # Job and position features
            node_features[op_idx, 5] = op_pos / max(1, len(self.jobs[job_id]) - 1)  # Position in job
            node_features[op_idx, 6] = 1.0 if self.job_arrival_times[job_id] <= self.current_time else 0.0  # Job arrived
            
            # Urgency/priority features
            remaining_ops = len(self.jobs[job_id]) - op_pos - 1
            node_features[op_idx, 7] = 1.0 / (remaining_ops + 1)  # Inverse remaining operations (urgency)
        
        # Machine node features (8 features per machine)
        for m_idx, machine in enumerate(self.machines):
            machine_node_idx = self.num_operations + m_idx
            
            # Availability and load features
            node_features[machine_node_idx, 0] = 1.0 if self.machine_available_time[machine] <= self.current_time else 0.0
            node_features[machine_node_idx, 1] = self.machine_available_time[machine] / max(1.0, self.current_time + 1)
            
            # Workload comparison
            all_loads = [self.machine_available_time[m] for m in self.machines]
            max_load = max(all_loads) if max(all_loads) > 0 else 1.0
            min_load = min(all_loads)
            node_features[machine_node_idx, 2] = self.machine_available_time[machine] / max_load  # Relative load
            
            # Machine utilization and capability features
            scheduled_ops = len(self.schedule[machine])
            node_features[machine_node_idx, 3] = scheduled_ops / max(1, self.num_operations // self.num_machines)
            
            # Compatibility count (how many operations can this machine process)
            compatible_ops = 0
            for job_ops in self.jobs.values():
                for op_data in job_ops:
                    if machine in op_data['proc_times']:
                        compatible_ops += 1
            node_features[machine_node_idx, 4] = compatible_ops / max(1, self.num_operations)
            
            # Time-based features
            node_features[machine_node_idx, 5] = (self.machine_available_time[machine] - self.current_time) / max(1.0, self.current_time + 1)
            node_features[machine_node_idx, 6] = 1.0 if self.machine_available_time[machine] == min_load else 0.0  # Is least loaded
            node_features[machine_node_idx, 7] = 1.0 if self.machine_available_time[machine] == max_load else 0.0  # Is most loaded
            
        # Build adjacency matrix
        # Precedence edges (operation to next operation in same job)
        for job_id in self.job_ids:
            job_ops = self.job_to_ops[job_id]
            for i in range(len(job_ops) - 1):
                adjacency[job_ops[i], job_ops[i + 1]] = 1.0
        
        # Machine compatibility edges (operation to compatible machines)
        for op_idx, (job_id, op_pos) in enumerate(self.operations):
            for machine in self.jobs[job_id][op_pos]['proc_times']:
                machine_idx = self.machines.index(machine)
                machine_node_idx = self.num_operations + machine_idx
                adjacency[op_idx, machine_node_idx] = 1.0
        
        # Enhanced global features (8 features for comprehensive context)
        completed_ops = sum(1 for s in self.operation_status if s == 'done')
        ready_ops = sum(1 for s in self.operation_status if s == 'ready')
        
        machine_loads = [self.machine_available_time[m] for m in self.machines]
        max_load = max(machine_loads) if machine_loads else 1.0
        min_load = min(machine_loads) if machine_loads else 0.0
        
        arrived_job_count = len([j for j, t in self.job_arrival_times.items() if t <= self.current_time])
        
        global_features = np.array([
            self.current_time / 100.0,  # Normalized current time
            completed_ops / max(1, self.num_operations),  # Completion progress
            ready_ops / max(1, self.num_operations - completed_ops),  # Ready operation ratio
            (max_load - min_load) / max(1.0, max_load),  # Machine load imbalance
            arrived_job_count / max(1, self.num_jobs),  # Job arrival progress
            np.mean(machine_loads) / max(1.0, self.current_time + 1),  # Average machine utilization
            len([job_id for job_id in self.job_ids if self.job_next_op[job_id] >= len(self.jobs[job_id])]) / max(1, self.num_jobs),  # Job completion ratio
            min(ready_ops, len(self.valid_action_pairs)) / max(1, len(self.valid_action_pairs))  # Action space utilization
        ], dtype=np.float32)
        
        return {
            'node_features': node_features,
            'adjacency': adjacency,
            'global_features': global_features
        }
    
    def _calculate_reward(self, proc_time, idle_time, done):
        # Enhanced reward function for better GraphDFJSPEnv performance
        reward = 0.0
        
        # 1. Processing time penalty (small, focus on efficiency)
        reward -= proc_time * 0.05
        
        # 2. Idle time penalty (larger, focus on utilization)
        reward -= idle_time * 0.3
        
        # 3. Progress reward (encourage completing operations)
        reward += 5.0
        
        # 4. Machine load balancing reward
        machine_loads = [self.machine_available_time[m] for m in self.machines]
        if max(machine_loads) > 0:
            load_balance = 1.0 - (np.std(machine_loads) / max(machine_loads))
            reward += load_balance * 2.0
        
        # 5. Job completion bonus (encourage finishing jobs completely)
        completed_jobs = sum(1 for job_id in self.job_ids 
                           if self.job_next_op[job_id] >= len(self.jobs[job_id]))
        reward += completed_jobs * 10.0
        
        # 6. Final completion reward/penalty
        if done:
            # Large completion bonus
            reward += 200.0
            
            # Makespan penalty (encourage shorter schedules)
            reward -= self.current_time * 0.1
            
            # Perfect balance bonus (if all machines finish at similar times)
            makespan_balance = 1.0 - (np.std(machine_loads) / max(machine_loads)) if max(machine_loads) > 0 else 1.0
            reward += makespan_balance * 50.0
        
        return reward

# Enhanced Graph Neural Network Features Extractor for GraphDFJSPEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F

class EnhancedGraphFeaturesExtractor(BaseFeaturesExtractor):
    """
    Advanced Graph Neural Network feature extractor optimized for GraphDFJSPEnv.
    Includes multi-layer GNN with attention mechanisms and residual connections.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Get dimensions from observation space
        node_features_shape = observation_space['node_features'].shape
        global_features_shape = observation_space['global_features'].shape
        
        self.num_nodes = node_features_shape[0]
        self.node_feature_dim = node_features_shape[1]
        self.global_feature_dim = global_features_shape[0]
        
        # Enhanced node embedding layers with residual connections
        self.node_embedding = nn.Sequential(
            nn.Linear(self.node_feature_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # Multi-layer Graph Neural Network with attention
        self.gnn_layers = nn.ModuleList([
            nn.Linear(128, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 128)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(256),
            nn.LayerNorm(256),
            nn.LayerNorm(128)
        ])
        
        # Multi-head attention for graph pooling
        self.attention = nn.MultiheadAttention(128, num_heads=8, dropout=0.1, batch_first=True)
        
        # Global feature processing with enhanced architecture
        self.global_fc = nn.Sequential(
            nn.Linear(self.global_feature_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Final feature combination with larger capacity
        self.final_fc = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        node_features = observations['node_features']
        adjacency = observations['adjacency'] 
        global_features = observations['global_features']
        
        # Node embedding with residual connection
        h = self.node_embedding(node_features)
        original_h = h.clone()
        
        # Multi-layer GNN with residual connections
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            # Message passing: aggregate neighbor features
            neighbor_features = torch.matmul(adjacency, h)
            
            # Combine self features with neighbor features
            if i == 0:
                combined = torch.cat([h, neighbor_features], dim=-1)
                h_new = gnn_layer(combined)
            else:
                combined = h + neighbor_features
                h_new = gnn_layer(combined)
            
            # Apply normalization and activation
            h_new = layer_norm(h_new)
            h = F.relu(h_new)
            
            # Residual connection for the last layer
            if i == len(self.gnn_layers) - 1 and h.shape == original_h.shape:
                h = h + original_h
            
            # Apply dropout for regularization
            if self.training:
                h = F.dropout(h, p=0.1)
        
        # Multi-head attention pooling for graph representation
        h_expanded = h.unsqueeze(0)  # Add batch dimension for attention
        attn_output, _ = self.attention(h_expanded, h_expanded, h_expanded)
        graph_repr = torch.mean(attn_output.squeeze(0), dim=0)  # Global average pooling
        
        # Process global features
        global_repr = self.global_fc(global_features)
        
        # Combine representations
        combined = torch.cat([graph_repr, global_repr], dim=-1)
        features = self.final_fc(combined)
        
        return features

class EnhancedGraphMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """Custom policy using Enhanced Graph Neural Network features extractor."""
    def __init__(self, *args, **kwargs):
        kwargs['features_extractor_class'] = EnhancedGraphFeaturesExtractor
        kwargs['features_extractor_kwargs'] = {'features_dim': 512}
        super().__init__(*args, **kwargs)

# --- 3. Training and Evaluation ---
def train_agent(jobs_data, machine_list, train_arrivals, log_name, total_timesteps=150000):
    print(f"\n--- Training Agent: {log_name} ---")
    
    def make_train_env():
        env = DynamicFJSPEnv(jobs_data, machine_list, train_arrivals)
        env = ActionMasker(env, mask_fn)
        return Monitor(env)

    vec_env = DummyVecEnv([make_train_env])
    
    # Enhanced hyperparameters for better GraphDFJSPEnv training
    model = MaskablePPO(
        MaskableActorCriticPolicy, 
        vec_env, 
        verbose=1,
        # Optimized hyperparameters for graph-based environment
        n_steps=2048,  # Larger rollout buffer for better sample efficiency
        batch_size=128,  # Larger batch size for more stable gradients
        n_epochs=10,  # Moderate number of epochs to prevent overfitting
        gamma=0.995,  # Higher discount factor for long-term planning
        gae_lambda=0.95,  # Generalized Advantage Estimation
        ent_coef=0.02,  # Moderate entropy for exploration vs exploitation balance
        learning_rate=2e-4,  # Lower learning rate for more stable convergence
        clip_range=0.15,  # Moderate clip range
        clip_range_vf=None,  # No value function clipping
        vf_coef=0.25,  # Lower value function coefficient
        max_grad_norm=0.8,  # Higher gradient clipping threshold
        target_kl=0.01,  # Early stopping based on KL divergence
        # Additional stability improvements
        normalize_advantage=True,
        use_sde=False,  # No state-dependent exploration for discrete actions
        sde_sample_freq=-1,
        policy_kwargs=dict(
            net_arch=[512, 512, 256],  # Larger network for complex graph processing
            activation_fn=nn.ReLU,
            ortho_init=True,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs=dict(weight_decay=1e-4, eps=1e-5)
        )
    )
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    return model

def evaluate_agent(model, jobs_data, machine_list, eval_arrivals, scenario_name):
    print(f"\n--- Evaluating Agent on Scenario: {scenario_name} ---")
    
    def make_eval_env():
        return DynamicFJSPEnv(jobs_data, machine_list, eval_arrivals)

    eval_env = make_eval_env()
    obs, _ = eval_env.reset()
    done = False
    
    while not done:
        action_masks = get_action_masks(eval_env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, _, done, _, info = eval_env.step(action)

    final_makespan = info.get("makespan", float('inf'))
    print(f"Evaluation complete. Final Makespan: {final_makespan:.2f}")
    
    return final_makespan, eval_env.schedule

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
    
    while operations_scheduled_count < total_operations:
        candidate_operations = []
        
        # Find the earliest time a machine becomes free to advance time
        if not any(next_operation_for_job[job_id] < len(jobs_data[job_id]) for job_id in arrived_jobs):
             # If no available jobs have pending operations, advance time to next arrival
            upcoming_arrivals = [arr for arr in job_arrival_times.values() if arr > min(machine_next_free.values())]
            if not upcoming_arrivals: break # No more jobs will arrive
            
            next_arrival_time = min(upcoming_arrivals)
            for m in machine_list:
                if machine_next_free[m] < next_arrival_time:
                    machine_next_free[m] = next_arrival_time
            
            arrived_jobs.update({job_id for job_id, arrival in job_arrival_times.items() if arrival <= next_arrival_time})

        for job_id in arrived_jobs:
            op_idx = next_operation_for_job[job_id]
            if op_idx < len(jobs_data[job_id]):
                op_data = jobs_data[job_id][op_idx]
                job_ready_time = operation_end_times[job_id][op_idx - 1] if op_idx > 0 else job_arrival_times[job_id]
                
                for machine_name, proc_time in op_data['proc_times'].items():
                    earliest_start_time = max(machine_next_free[machine_name], job_ready_time)
                    candidate_operations.append((
                        proc_time,
                        earliest_start_time,
                        job_id, 
                        op_idx, 
                        machine_name
                    ))
        
        if not candidate_operations:
            break

        # Select the operation with the shortest processing time
        selected_op = min(candidate_operations, key=lambda x: x[0])
        proc_time, start_time, job_id, op_idx, machine_name = selected_op
        
        end_time = start_time + proc_time

        machine_next_free[machine_name] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled_count += 1
        
        schedule[machine_name].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        
        # Update arrived jobs based on new machine free times
        current_time = min(t for t in machine_next_free.values() if t > 0)
        arrived_jobs.update({j_id for j_id, arrival in job_arrival_times.items() if arrival <= current_time})

    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"SPT Heuristic Makespan: {makespan:.2f}")
    return makespan, schedule

def heuristic_ra_sa_scheduler(jobs_data, machine_list, job_arrival_times, 
                              routing_rule="EAM", sequencing_rule="SPT"):
    """
    Clean RA-SA scheduler implementing routing agent + sequencing agent approach.
    
    Args:
        routing_rule: "EAM", "LLM", "SPTM", "BEST", "GLOBAL_SPT"
        sequencing_rule: "SPT", "FCFS", "LPT", "EDD"
    """
    print(f"\n--- Running RA-SA Scheduler (RA: {routing_rule}, SA: {sequencing_rule}) ---")
    
    # Initialize state
    machine_next_free = {m: 0.0 for m in machine_list}
    machine_workload = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    schedule = {m: [] for m in machine_list}
    
    operations_scheduled_count = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    arrived_jobs = {job_id for job_id, arrival in job_arrival_times.items() if arrival <= 0}
    
    while operations_scheduled_count < total_operations:
        # Build all candidate (operation, machine) combinations
        all_combinations = []
        
        # Check if we need to advance time
        if not any(next_operation_for_job[job_id] < len(jobs_data[job_id]) for job_id in arrived_jobs):
            upcoming_arrivals = [arr for arr in job_arrival_times.values() 
                               if arr > min(machine_next_free.values())]
            if not upcoming_arrivals:
                break
            
            next_arrival_time = min(upcoming_arrivals)
            for m in machine_list:
                if machine_next_free[m] < next_arrival_time:
                    machine_next_free[m] = next_arrival_time
            
            arrived_jobs.update({job_id for job_id, arrival in job_arrival_times.items() 
                               if arrival <= next_arrival_time})
        
        # Generate all valid (operation, machine) combinations
        for job_id in arrived_jobs:
            op_idx = next_operation_for_job[job_id]
            if op_idx < len(jobs_data[job_id]):
                op_data = jobs_data[job_id][op_idx]
                job_ready_time = (operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                else job_arrival_times[job_id])
                
                for machine_name, proc_time in op_data['proc_times'].items():
                    earliest_start_time = max(machine_next_free[machine_name], job_ready_time)
                    
                    all_combinations.append({
                        'job_id': job_id,
                        'op_idx': op_idx,
                        'machine': machine_name,
                        'proc_time': proc_time,
                        'start_time': earliest_start_time,
                        'end_time': earliest_start_time + proc_time,
                        'ready_time': job_ready_time,
                        'machine_available': machine_next_free[machine_name],
                        'workload': machine_workload[machine_name],
                        'arrival_time': job_arrival_times[job_id]
                    })
        
        if not all_combinations:
            break

        # Select operation based on routing rule
        if routing_rule == "GLOBAL_SPT":
            # Special case: behave exactly like Global SPT
            selected_combination = min(all_combinations, key=lambda x: x['proc_time'])
        else:
            # Two-stage RA-SA approach
            # Stage 1: Routing Agent - group by operation, select best machine for each
            operation_groups = {}
            for combo in all_combinations:
                op_key = (combo['job_id'], combo['op_idx'])
                if op_key not in operation_groups:
                    operation_groups[op_key] = []
                operation_groups[op_key].append(combo)
            
            operations_with_best_machine = []
            for op_key, machine_options in operation_groups.items():
                best_machine_combo = routing_agent_decision_improved(machine_options, routing_rule)
                operations_with_best_machine.append(best_machine_combo)
            
            # Stage 2: Sequencing Agent - select which operation to schedule next
            selected_combination = sequencing_agent_decision(operations_with_best_machine, sequencing_rule)
        
        # Execute the selected combination
        job_id = selected_combination['job_id']
        op_idx = selected_combination['op_idx']
        machine = selected_combination['machine']
        start_time = selected_combination['start_time']
        proc_time = selected_combination['proc_time']
        end_time = start_time + proc_time
        
        # Update system state
        machine_next_free[machine] = end_time
        machine_workload[machine] += proc_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled_count += 1
        
        schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        
        # Update arrived jobs
        current_time = min(t for t in machine_next_free.values() if t > 0)
        arrived_jobs.update({j_id for j_id, arrival in job_arrival_times.items() 
                           if arrival <= current_time})
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"RA-SA Makespan: {makespan:.2f}")
    return makespan, schedule


def routing_agent_decision_improved(machine_options, rule):
    """
    Clean routing agent implementation with improved decision logic.
    
    Args:
        machine_options: List of machine option dictionaries
        rule: Routing rule ("EAM", "LLM", "SPTM", "BEST", "RANDOM")
    """
    if rule == "EAM":  # Earliest Available Machine
        return min(machine_options, key=lambda m: m['machine_available'])
    elif rule == "LLM":  # Least Loaded Machine  
        return min(machine_options, key=lambda m: m['workload'])
    elif rule == "SPTM":  # Shortest Processing Time Machine
        return min(machine_options, key=lambda m: m['proc_time'])
    elif rule == "BEST":  # Best combination of factors
        def composite_score(m):
            # Weighted combination: completion time + workload balance
            return m['end_time'] + 0.1 * m['workload']
        return min(machine_options, key=composite_score)
    elif rule == "RANDOM":
        return np.random.choice(machine_options)
    else:
        # Default to first available
        return machine_options[0] if machine_options else None


def routing_agent_decision(eligible_machines, proc_times, machine_next_free, machine_workload, rule):
    """Original routing agent decision (kept for compatibility)"""
    if rule == "EAM":  # Earliest Available Machine
        return min(eligible_machines, key=lambda m: machine_next_free[m])
    elif rule == "LLM":  # Least Loaded Machine
        return min(eligible_machines, key=lambda m: machine_workload[m])
    elif rule == "SPTM":  # Shortest Processing Time Machine
        return min(eligible_machines, key=lambda m: proc_times[m])
    elif rule == "RANDOM":
        return np.random.choice(eligible_machines)
    else:
        return eligible_machines[0]  # Default to first available


def sequencing_agent_decision(operation_queue, rule):
    """
    Clean sequencing agent implementation.
    
    Args:
        operation_queue: List of operation dictionaries ready for scheduling
        rule: Sequencing rule ("SPT", "LPT", "FCFS", "EDD")
    """
    if not operation_queue:
        return None
    
    if rule == "SPT":  # Shortest Processing Time
        return min(operation_queue, key=lambda op: op['proc_time'])
    elif rule == "LPT":  # Longest Processing Time
        return max(operation_queue, key=lambda op: op['proc_time'])
    elif rule == "FCFS":  # First Come First Served
        return min(operation_queue, key=lambda op: op['arrival_time'])
    elif rule == "EDD":  # Earliest Due Date (using end time as proxy)
        return min(operation_queue, key=lambda op: op.get('end_time', op['proc_time']))
    else:
        # Default to first operation in queue
        return operation_queue[0]


def compare_heuristics(jobs_data, machine_list, job_arrival_times):
    """Compare different heuristic approaches"""
    print("\n" + "="*60)
    print("HEURISTIC COMPARISON")
    print("="*60)
    
    results = {}
    
    # Global SPT
    spt_makespan, spt_schedule = heuristic_spt_scheduler(jobs_data, machine_list, job_arrival_times)
    results['Global SPT'] = spt_makespan
    
    # RA-SA variants
    ra_sa_combinations = [
        ("EAM", "SPT"),
        ("LLM", "SPT"),  
        ("SPTM", "SPT"),
        ("BEST", "SPT"),
        ("EAM", "FCFS"),
        ("BEST", "FCFS")
    ]
    
    for routing_rule, sequencing_rule in ra_sa_combinations:
        makespan, _ = heuristic_ra_sa_scheduler(jobs_data, machine_list, job_arrival_times, 
                                                routing_rule, sequencing_rule)
        results[f"RA-SA ({routing_rule}-{sequencing_rule})"] = makespan
    
    # Print comparison
    print("\nPerformance Comparison:")
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for method, makespan in sorted_results:
        print(f"{method:25s}: {makespan:6.2f}")
    
    print(f"\nBest method: {sorted_results[0][0]} with makespan {sorted_results[0][1]:.2f}")
    return results

def detailed_heuristic_comparison(jobs_data, machine_list, job_arrival_times):
    """
    Comprehensive comparison of routing and sequencing strategies to understand
    why Global SPT performs better than traditional RA-SA approaches.
    """
    print("\n" + "="*80)
    print("DETAILED HEURISTIC ANALYSIS")
    print("="*80)
    
    results = {}
    
    # Global SPT (the benchmark)
    spt_makespan, spt_schedule = heuristic_spt_scheduler(jobs_data, machine_list, job_arrival_times)
    results['Global SPT (benchmark)'] = spt_makespan
    
    # RA-SA with GLOBAL_SPT routing (should match Global SPT exactly)
    ra_sa_global_makespan, ra_sa_global_schedule = heuristic_ra_sa_scheduler(
        jobs_data, machine_list, job_arrival_times, "GLOBAL_SPT", "SPT"
    )
    results['RA-SA (GLOBAL_SPT-SPT)'] = ra_sa_global_makespan
    
    # Traditional RA-SA combinations
    routing_rules = ["EAM", "LLM", "SPTM", "BEST"]
    sequencing_rules = ["SPT", "FCFS", "LPT"]
    
    for routing_rule in routing_rules:
        for sequencing_rule in sequencing_rules:
            try:
                makespan, _ = heuristic_ra_sa_scheduler(
                    jobs_data, machine_list, job_arrival_times, 
                    routing_rule, sequencing_rule
                )
                results[f"RA-SA ({routing_rule}-{sequencing_rule})"] = makespan
            except Exception as e:
                print(f"Error with {routing_rule}-{sequencing_rule}: {e}")
                results[f"RA-SA ({routing_rule}-{sequencing_rule})"] = float('inf')
    
    # Analysis and comparison
    print("\n" + "-"*80)
    print("PERFORMANCE COMPARISON:")
    print("-"*80)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    for i, (method, makespan) in enumerate(sorted_results):
        if makespan == float('inf'):
            continue
        gap_from_best = makespan - sorted_results[0][1] if i > 0 else 0.0
        gap_percent = (gap_from_best / sorted_results[0][1] * 100) if sorted_results[0][1] > 0 else 0.0
        print(f"{i+1:2d}. {method:35s}: {makespan:6.2f} (+{gap_from_best:5.2f}, +{gap_percent:4.1f}%)")
    
    # Key insights
    print("\n" + "-"*80)
    print("KEY INSIGHTS:")
    print("-"*80)
    
    global_spt = results['Global SPT (benchmark)']
    ra_sa_global = results['RA-SA (GLOBAL_SPT-SPT)']
    
    print(f"1. Global SPT makespan: {global_spt:.2f}")
    print(f"2. RA-SA with GLOBAL_SPT routing: {ra_sa_global:.2f}")
    
    if abs(global_spt - ra_sa_global) < 0.01:
        print("   ✓ CONFIRMED: RA-SA with GLOBAL_SPT routing matches Global SPT exactly!")
        print("   This proves that Global SPT is essentially using global routing.")
    else:
        print("   ✗ Something is different - there might be a bug in the implementation.")
    
    # Find best traditional RA-SA
    traditional_methods = {k: v for k, v in results.items() 
                          if k.startswith('RA-SA') and 'GLOBAL_SPT' not in k and v != float('inf')}
    if traditional_methods:
        best_traditional = min(traditional_methods.items(), key=lambda x: x[1])
        gap = best_traditional[1] - global_spt
        gap_percent = gap / global_spt * 100
        
        print(f"3. Best traditional RA-SA: {best_traditional[0]} with {best_traditional[1]:.2f}")
        print(f"   Gap from Global SPT: +{gap:.2f} (+{gap_percent:.1f}%)")
        print(f"   This gap shows the cost of two-step decision making vs. global optimization.")
    
    return results

# --- 5. MILP Optimal Scheduler ---
def milp_scheduler(jobs, machines, arrival_times):
    """MILP approach for optimal dynamic scheduling."""
    print("\n--- Running MILP Optimal Scheduler ---")
    prob = LpProblem("DynamicFJSP_Optimal", LpMinimize)
    
    ops = [(j, oi) for j in jobs for oi in range(len(jobs[j]))]
    BIG_M = 1000 

    x = LpVariable.dicts("x", (ops, machines), cat="Binary")
    s = LpVariable.dicts("s", ops, lowBound=0)
    c = LpVariable.dicts("c", ops, lowBound=0)
    y = LpVariable.dicts("y", (ops, ops, machines), cat="Binary")
    Cmax = LpVariable("Cmax", lowBound=0)

    prob += Cmax

    for j, oi in ops:
        # Assignment constraint
        prob += lpSum(x[j, oi][m] for m in jobs[j][oi]['proc_times']) == 1
        # Completion time
        prob += c[j, oi] == s[j, oi] + lpSum(x[j, oi][m] * jobs[j][oi]['proc_times'][m] for m in jobs[j][oi]['proc_times'])
        # Precedence within a job
        if oi > 0:
            prob += s[j, oi] >= c[j, oi - 1]
        # Arrival time constraint
        else:
            prob += s[j, oi] >= arrival_times[j]
        # Makespan definition
        prob += Cmax >= c[j, oi]

    for m in machines:
        ops_on_m = [op for op in ops if m in jobs[op[0]][op[1]]['proc_times']]
        for i in range(len(ops_on_m)):
            for k in range(i + 1, len(ops_on_m)):
                op1, op2 = ops_on_m[i], ops_on_m[k]
                # Disjunctive constraints
                prob += s[op1] >= c[op2] - BIG_M * (1 - y[op1][op2][m]) - BIG_M * (2 - x[op1][m] - x[op2][m])
                prob += s[op2] >= c[op1] - BIG_M * y[op1][op2][m] - BIG_M * (2 - x[op1][m] - x[op2][m])

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=120)) # 2-minute time limit

    schedule = {m: [] for m in machines}
    if prob.status == 1 and Cmax.varValue is not None:  # Optimal solution found
        for (j, oi), m in ((op, m) for op in ops for m in jobs[op[0]][op[1]]['proc_times']):
            if x[j, oi][m].varValue > 0.5:
                schedule[m].append((f"J{j}-O{oi+1}", s[j, oi].varValue, c[j, oi].varValue))
        
        # Sort operations by start time
        for m in machines:
            schedule[m].sort(key=lambda x: x[1])
        
        print(f"MILP (optimal) Makespan: {Cmax.varValue:.2f}")
        return Cmax.varValue, schedule
    else:
        print("MILP solver failed to find optimal solution")
        return float('inf'), schedule

# --- 6. Main Execution Block ---
def plot_gantt_charts(figure_num, schedules, makespans, titles, machine_list, arrival_times, save_path=None):
    """Plot multiple Gantt charts with arrival indicators and optional saving."""
    # Set font for poster presentation
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 16
    })
    
    num_charts = len(schedules)
    # Increase figure height and adjust spacing for better readability
    fig = plt.figure(figure_num, figsize=(18, num_charts * 3.5))
    
    colors = plt.cm.Set3.colors  # Changed to Set3 for better visual distinction
    
    # Calculate the maximum time across all schedules to ensure consistent x-axis scale
    max_time = 0
    for schedule in schedules:
        for machine_ops in schedule.values():
            for op_data in machine_ops:
                if len(op_data) >= 3:
                    _, _, end_time = op_data
                    max_time = max(max_time, end_time)
    
    # Add some padding to the max time for better visualization
    max_time = max_time * 1.05

    for i, (schedule, makespan, title) in enumerate(zip(schedules, makespans, titles)):
        ax = fig.add_subplot(num_charts, 1, i + 1)
        for idx, m in enumerate(machine_list):
            for op_data in schedule.get(m, []):
                job_id_str, start, end = op_data
                j = int(job_id_str.split('-')[0][1:])
                ax.broken_barh([(start, end - start)], (idx * 10, 8),
                               facecolors=colors[j % len(colors)], edgecolor='black', linewidth=1.5, alpha=0.8)
                ax.text(start + (end - start) / 2, idx * 10 + 4, job_id_str,
                        color='black', ha='center', va='center', weight='bold', fontsize=10,
                        fontfamily='serif')
        
        # Use the correct arrival times for the specific scenario being plotted
        current_arrival_times = arrival_times[i] if isinstance(arrival_times, list) else arrival_times
        for job_id, arrival in current_arrival_times.items():
            if arrival > 0:
                ax.axvline(x=arrival, color='r', linestyle='--', linewidth=2)
                ax.annotate(f'J{job_id} Arrives', xy=(arrival, len(machine_list) * 10),
                            xytext=(arrival + 1, len(machine_list) * 10 + 4),
                            arrowprops=dict(facecolor='red', shrink=0.05, width=2),
                            color='darkred', ha='left', fontsize=10, weight='bold',
                            fontfamily='serif')

        ax.set_yticks([k * 10 + 4 for k in range(len(machine_list))])
        ax.set_yticklabels(machine_list)
        ax.set_ylabel("Machines", fontsize=12, fontfamily='serif')
        # Adjust title formatting and size
        ax.set_title(f"{title}\nMakespan: {makespan:.2f}", fontsize=14, pad=20, 
                    fontfamily='serif', weight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set consistent x-axis limits for all subplots
        ax.set_xlim(0, max_time)
        
        if i < num_charts - 1:
            ax.tick_params(labelbottom=False)
    
    plt.xlabel("Time", fontsize=14, fontfamily='serif', weight='bold')
    # Adjust layout with more spacing
    plt.tight_layout(rect=[0, 0.02, 1, 0.98], h_pad=2.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()

def plot_job_structure_table(jobs_data, machine_list, arrival_times, save_path=None):
    """Create a table figure showing the job data structure."""
    # Set font for poster presentation
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 12,
        'axes.titlesize': 20,
        'figure.titlesize': 22
    })
    
    # Create figure with white background to stand out on dark blue poster
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('white')  # White background for contrast against dark blue poster
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Job', 'Arrival Time', 'Operation', 'Available Machines', 'Processing Times']
    
    # High contrast color scheme for dark blue poster background
    job_colors = [
        '#FFFFFF',  # Pure white
        '#F5F5F5',  # Light gray
        '#E8E8E8',  # Medium light gray
        '#DDDDDD',  # Light gray
        '#D0D0D0',  # Medium gray
        '#C8C8C8'   # Darker gray
    ]
    
    for job_id, operations in jobs_data.items():
        arrival_time = arrival_times.get(job_id, 'N/A')
        
        for op_idx, operation in enumerate(operations):
            machines = list(operation['proc_times'].keys())
            # Clean processing times format
            proc_times = [f"{m}: {operation['proc_times'][m]}" for m in machines]
            
            row = [
                f'Job {job_id}' if op_idx == 0 else '',
                str(arrival_time) if op_idx == 0 else '',
                f'{op_idx + 1}',
                ', '.join(machines),
                ', '.join(proc_times)
            ]
            table_data.append(row)
    
    # Create table with optimal proportions
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2.5)  # Larger scale for better visibility
    
    # Strong header styling - lighter color for better contrast on dark blue poster
    header_color = '#E3F2FD'  # Light blue instead of dark navy
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color='#1A237E', fontsize=14, fontweight='bold', fontfamily='serif')  # Dark blue text
        cell.set_height(0.18)  # Taller headers
        cell.set_edgecolor('#333333')
        cell.set_linewidth(2)
    
    # Enhanced row styling with high contrast
    current_job = None
    color_idx = 0
    for i, row in enumerate(table_data):
        job_str = row[0]
        if job_str:  # New job
            current_job = int(job_str.split()[1])
            color_idx = current_job % len(job_colors)
        
        # Apply high contrast background colors
        bg_color = job_colors[color_idx]
        for j in range(len(headers)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(bg_color)
            cell.set_text_props(fontfamily='serif', fontsize=12, color='#000000')  # Black text
            cell.set_height(0.14)  # Good row height
            
            # Dark borders for definition
            cell.set_edgecolor('#333333')
            cell.set_linewidth(1.5)
            
            # Make job names and arrival times stand out with bold and color
            if j == 0 and row[j]:  # Job column
                cell.set_text_props(fontweight='bold', fontsize=13, color='#1A237E', fontfamily='serif')
            elif j == 1 and row[j]:  # Arrival column
                cell.set_text_props(fontweight='bold', fontsize=12, color='#3F51B5', fontfamily='serif')
    
    # Add strong borders to all cells for definition
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#333333')
        cell.set_linewidth(1.2)
    
    # Adjust layout to prevent clipping and optimize for poster
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])  # Use full space without title
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Job structure table saved to {save_path}")
    
    plt.show()

def visualize_action_mask(env, mask, jobs_data, machine_list, save_path=None):
    """
    Create a comprehensive visualization of the action mask showing which actions are valid.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 14))
    
    # 1. Binary mask visualization
    ax1.imshow(mask.reshape(1, -1), cmap='RdYlGn', aspect='auto', alpha=0.8)
    ax1.set_title('Action Mask - Valid Actions (Green=Valid, Red=Invalid)', fontsize=12, pad=10)
    ax1.set_xlabel('Action Index')
    ax1.set_ylabel('Mask')
    ax1.set_yticks([0])
    ax1.set_yticklabels(['Valid'])
    
    # Add text annotations for some key actions
    valid_indices = np.where(mask)[0]
    for i, idx in enumerate(valid_indices[:10]):  # Show first 10 valid actions
        ax1.annotate(f'{idx}', xy=(idx, 0), xytext=(idx, 0.3),
                    ha='center', fontsize=8, color='black',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    # 2. Action distribution by operation-machine pairs
    if hasattr(env, 'valid_action_pairs'):
        action_data = []
        for i, (op_idx, machine) in enumerate(env.valid_action_pairs):
            if mask[i]:
                job_id, op_pos = env.operations[op_idx]
                action_data.append((job_id, op_pos, machine, i))
        
        if action_data:
            # Group by job
            job_colors = plt.cm.Set3.colors
            job_data = {}
            for job_id, op_pos, machine, action_idx in action_data:
                if job_id not in job_data:
                    job_data[job_id] = []
                job_data[job_id].append((op_pos, machine, action_idx))
            
            y_pos = 0
            y_labels = []
            for job_id in sorted(job_data.keys()):
                for op_pos, machine, action_idx in sorted(job_data[job_id]):
                    ax2.barh(y_pos, 1, left=action_idx, height=0.8, 
                            color=job_colors[job_id % len(job_colors)], 
                            alpha=0.7, edgecolor='black', linewidth=0.5)
                    ax2.text(action_idx + 0.5, y_pos, f'J{job_id}-O{op_pos+1}\n{machine}', 
                            ha='center', va='center', fontsize=8, fontweight='bold')
                    y_labels.append(f'J{job_id}-O{op_pos+1}-{machine}')
                    y_pos += 1
            
            ax2.set_title('Valid Actions by Job-Operation-Machine', fontsize=12, pad=10)
            ax2.set_xlabel('Action Index')
            ax2.set_ylabel('Job-Operation-Machine')
            ax2.set_yticks(range(len(y_labels)))
            ax2.set_yticklabels(y_labels, fontsize=8)
            ax2.grid(True, alpha=0.3)
    
    # 3. Summary statistics
    total_actions = len(mask)
    valid_actions = np.sum(mask)
    invalid_actions = total_actions - valid_actions
    
    # Create pie chart of valid vs invalid
    sizes = [valid_actions, invalid_actions]
    labels = [f'Valid ({valid_actions})', f'Invalid ({invalid_actions})']
    colors = ['lightgreen', 'lightcoral']
    explode = (0.1, 0)  # Explode the valid slice
    
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
           explode=explode, shadow=True, startangle=90)
    ax3.set_title(f'Action Space Utilization\n(Total Actions: {total_actions})', fontsize=12, pad=10)
    
    # Add summary text
    summary_text = f"""
    Environment State:
    • Total Actions: {total_actions}
    • Valid Actions: {valid_actions} ({valid_actions/total_actions*100:.1f}%)
    • Action Space Efficiency: {'High' if valid_actions/total_actions > 0.1 else 'Low'}
    • Current Time: {getattr(env, 'current_time', 'N/A')}
    • Operations Scheduled: {getattr(env, 'operations_scheduled', 'N/A')}/{getattr(env, 'total_operations', 'N/A')}
    """
    
    plt.figtext(0.02, 0.02, summary_text, fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Action mask visualization saved to {save_path}")
    
    plt.show()

    """
    Detailed explanation of Dynamic FJSP as a Markov Decision Process
    """
    print("\n" + "="*80)
    print("DYNAMIC FJSP AS MARKOV DECISION PROCESS (MDP)")
    print("="*80)
    
    mdp_explanation = """
    The Dynamic Flexible Job Shop Scheduling Problem (D-FJSP) can be formulated as a 
    Markov Decision Process with the following components:

    CRITICAL MDP ANALYSIS:
    =====================
    
    Q: Is the transition probability always 1?
    A: YES! This is a DETERMINISTIC MDP. Once we choose an action in a given state,
       the next state is completely determined. There's no randomness in:
       - Processing times (they're fixed)
       - Machine availability updates
       - Job arrival times (predetermined)
       - Precedence constraints (deterministic)
       
       Therefore: P(s'|s,a) ∈ {0, 1} for all s, s', a
    
    Q: Is this fully or partially observed?
    A: This is a FULLY OBSERVED MDP (not a POMDP). The agent has complete access to:
       ✓ All machine availability times
       ✓ All job states and progress  
       ✓ All arrival times and job specifications
       ✓ Complete scheduling history
       ✓ Current time and makespan
       
       There are NO hidden variables or uncertain information.

    1. STATE SPACE (S):
    ==================
    FORMAL STATE DEFINITION:
    s = (M, J, A, t) where:
    - M = (m₁_free, m₂_free, ..., mₖ_free) : Machine availability times
    - J = (j₁_progress, j₂_progress, ..., jₙ_progress) : Job completion status  
    - A = {arrived_job_ids} : Set of arrived jobs
    - t = current_makespan : Current scheduling time
    
    STATE PROPERTIES:
    - Finite (though large): |S| = O(T^k × 2^(n×max_ops) × 2^n)
    - Deterministic transitions: Each (s,a) pair leads to exactly one s'
    - Markovian: P(s_{t+1}|s_t, a_t, s_{t-1}, ...) = P(s_{t+1}|s_t, a_t)
    
    OBSERVABILITY: FULL
    The agent observes the complete state s directly. No hidden information.

    2. ACTION SPACE (A):
    ===================
    FORMAL ACTION DEFINITION:
    a = (job_id, operation_id, machine_id) 
    
    ACTION CONSTRAINTS (Action Masking):
    - Arrival constraint: job_id ∈ arrived_jobs(t)
    - Precedence constraint: operation_id = next_operation[job_id]  
    - Capability constraint: machine_id ∈ capable_machines[job_id][operation_id]
    - Completion constraint: operation not already completed
    
    VALID ACTION SET:
    A(s) = {a | constraints satisfied in state s}
    
    |A(s)| is typically 5-20 actions (much smaller than total action space)

    3. TRANSITION FUNCTION (T):
    ===========================
    DETERMINISTIC TRANSITIONS: P(s'|s,a) ∈ {0,1}
    
    For valid action a = (j, op, m) in state s:
    
    TRANSITION MECHANICS:
    1. start_time = max(machine_free[m], job_ready_time[j], arrival_time[j])
    2. end_time = start_time + processing_time[j][op][m]
    3. s'.machine_free[m] = end_time
    4. s'.job_progress[j][op] = COMPLETED
    5. s'.next_operation[j] += 1
    6. s'.current_makespan = max(s.current_makespan, end_time)
    7. s'.arrived_jobs = s.arrived_jobs ∪ {new arrivals based on s'.current_makespan}
    
    DETERMINISM PROOF:
    Given (s,a), each step above is deterministic calculation → unique s'
    Therefore: P(s'|s,a) = 1 for the computed s', and P(s''|s,a) = 0 for all s'' ≠ s'

    4. REWARD FUNCTION (R):
    =======================
    DETERMINISTIC REWARDS: R(s,a) is deterministically computed
    
    R(s,a) = r_immediate + r_terminal
    
    IMMEDIATE REWARDS:
    - Processing penalty: -α × processing_time
    - Idle penalty: -β × idle_time (β > α)
    - Progress reward: +γ (constant per operation)
    
    TERMINAL REWARDS (when all operations complete):
    - Completion bonus: +C
    - Makespan penalty: -δ × final_makespan
    
    REWARD DETERMINISM:
    All reward components are deterministic functions of (s,a) and resulting s'.

    5. MDP CLASSIFICATION:
    =====================
    
    ✓ FINITE MDP: State and action spaces are finite (though large)
    ✓ DETERMINISTIC MDP: P(s'|s,a) ∈ {0,1}
    ✓ FULLY OBSERVABLE: Agent observes complete state s
    ✓ EPISODIC: Clear terminal states (all operations completed)
    ✓ STATIONARY: Transition probabilities don't change over episodes
    
    ✗ NOT STOCHASTIC: No randomness in transitions
    ✗ NOT POMDP: No hidden information
    ✗ NOT INFINITE: State space is finite (though exponentially large)

    6. BELLMAN EQUATIONS:
    ====================
    
    VALUE FUNCTION (State Value):
    V*(s) = max_a ∑_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]
    
    Since P(s'|s,a) ∈ {0,1}, this simplifies to:
    V*(s) = max_a [R(s,a) + γV*(s')] where s' is the unique next state
    
    Q-FUNCTION (Action Value):  
    Q*(s,a) = R(s,a) + γV*(s') where s' is the unique next state
    
    OPTIMAL POLICY:
    π*(s) = argmax_a Q*(s,a) = argmax_a [R(s,a) + γV*(s')]

    7. SOLUTION METHODS:
    ===================
    
    Due to deterministic transitions and full observability:
    
    ✓ VALUE ITERATION: V_{k+1}(s) = max_a [R(s,a) + γV_k(s')]
    ✓ POLICY ITERATION: Alternate between policy evaluation and improvement
    ✓ FINITE HORIZON DP: Backward induction from terminal states
    ✓ A* SEARCH: With admissible heuristics (e.g., lower bound on remaining makespan)
    ✓ BRANCH AND BOUND: Systematic tree search with bounds
    
    COMPUTATIONAL COMPLEXITY:
    - State space: Exponential in problem size
    - For small instances: DP is optimal
    - For large instances: Approximation methods (RL, heuristics)

    8. WHY REINFORCEMENT LEARNING?
    =============================
    
    Even though this is a deterministic, fully observable MDP, RL is still valuable:
    
    CURSE OF DIMENSIONALITY:
    - State space grows exponentially: O(T^k × 2^(n×max_ops) × 2^n)  
    - For n=10 jobs, k=5 machines: ~10^15 states
    - Exact DP becomes computationally intractable
    
    RL ADVANTAGES:
    ✓ Sample-based learning: Don't need to enumerate all states
    ✓ Function approximation: Learn compact state representations
    ✓ Generalization: Trained policy works on unseen problem instances
    ✓ Scalability: Can handle larger problem sizes
    ✓ Online learning: Can adapt to new problem variations
    
    EXACT DP ADVANTAGES:
    ✓ Guaranteed optimality (for small instances)
    ✓ Complete policy specification
    ✓ Interpretable solution process
    ✓ Theoretical guarantees
    """
    
    print(mdp_explanation)

if __name__ == "__main__":
    """
    Main execution demonstrating Policy Iteration for Dynamic FJSP
    """
    print("Dynamic FJSP - Policy Iteration Demonstration")
    print("This demonstrates solving D-FJSP using exact dynamic programming methods")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Use smaller problem instance for policy iteration
    jobs_data = DEFAULT_JOBS_DATA
    machine_list = DEFAULT_MACHINE_LIST
    arrival_times = DEFAULT_ARRIVAL_TIMES
    
    print("\n1. PROBLEM INSTANCE DETAILS:")
    print("-" * 50)
    print(f"Jobs: {len(jobs_data)}")
    print(f"Machines: {machine_list}")
    print(f"Arrival times: {arrival_times}")
    print(f"Total operations: {sum(len(ops) for ops in jobs_data.values())}")
    
    for job_id, operations in jobs_data.items():
        print(f"\nJob {job_id} (arrives at time {arrival_times[job_id]}):")
        for op_idx, op_data in enumerate(operations):
            proc_times = op_data['proc_times']
            print(f"  Operation {op_idx+1}: {proc_times}")
    
    
    print("\n2. SOLVING WITH POLICY ITERATION:")
    print("-" * 50)
    
    # Create and solve with Policy Iteration
    solver = StepByStepPolicyIterationSolver(jobs_data, machine_list, arrival_times, gamma=0.95)
    optimal_policy = solver.solve(max_iterations=20)
    
    # Execute the learned policy
    pi_makespan, pi_schedule, execution_trace = solver.execute_policy()
    
    # Visualize convergence
    visualize_policy_iteration_convergence(solver)
    
    print("\n3. COMPARISON WITH OPTIMAL SOLUTION:")
    print("-" * 50)
    
    # Get optimal solution using MILP
    try:
        optimal_makespan, optimal_schedule = milp_scheduler(jobs_data, machine_list, arrival_times)
    except Exception as e:
        print(f"MILP solver error: {e}")
        print("Using heuristic SPT as comparison baseline...")
        optimal_makespan, optimal_schedule = heuristic_spt_scheduler(jobs_data, machine_list, arrival_times)
    
    # Compare results
    print("\n4. RESULTS COMPARISON:")
    print("-" * 50)
    print(f"Policy Iteration Makespan: {pi_makespan:.2f}")
    print(f"Optimal (MILP) Makespan:   {optimal_makespan:.2f}")
    
    if optimal_makespan != float('inf'):
        gap = pi_makespan - optimal_makespan
        gap_percent = (gap / optimal_makespan * 100) if optimal_makespan > 0 else 0
        print(f"Optimality Gap:           +{gap:.2f} (+{gap_percent:.1f}%)")
        
        if abs(gap) < 0.01:
            print("🎉 PERFECT! Policy Iteration found the optimal solution!")
        elif gap_percent <= 5:
            print("✅ EXCELLENT! Policy Iteration is very close to optimal!")
        elif gap_percent <= 15:
            print("👍 GOOD! Policy Iteration found a competitive solution!")
        else:
            print("📊 Policy Iteration completed but with some optimality gap.")
    
    # Detailed schedule comparison
    print("\n5. SCHEDULE DETAILS:")
    print("-" * 50)
    
    print("Policy Iteration Schedule:")
    for machine, jobs in pi_schedule.items():
        if jobs:
            print(f"  {machine}: {jobs}")
    
    print("\nOptimal Schedule:")
    for machine, jobs in optimal_schedule.items():
        if jobs:
            print(f"  {machine}: {jobs}")
    
    # Visualize both schedules
    compare_schedules_visualization(pi_makespan, pi_schedule, 
                                  optimal_makespan, optimal_schedule, 
                                  machine_list, arrival_times)
    
    print("\n6. EXECUTION TRACE ANALYSIS:")
    print("-" * 50)
    if execution_trace:
        print("Step-by-step execution of learned policy:")
        for step_info in execution_trace[:10]:  # Show first 10 steps
            print(f"  Step {step_info['step']}: J{step_info['job_id']}-O{step_info['operation']} "
                  f"on {step_info['machine']} (proc_time={step_info['proc_time']}, "
                  f"reward={step_info['reward']:.2f})")
        
        if len(execution_trace) > 10:
            print(f"  ... and {len(execution_trace) - 10} more steps")
    
    print("\n7. ALGORITHM INSIGHTS:")
    print("-" * 50)
    
    print("Policy Iteration Performance:")
    if solver.iteration_data:
        converged = solver.iteration_data[-1]['policy_stable']
        iterations = len(solver.iteration_data)
        final_value = solver.iteration_data[-1]['avg_value']
        
        print(f"• Convergence: {'✓ Converged' if converged else '✗ Did not converge'}")
        print(f"• Iterations: {iterations}")
        print(f"• States explored: {len(solver.states)}")
        print(f"• Final average state value: {final_value:.4f}")
    
    insights = """
    Why Policy Iteration works well for this problem:
    • Deterministic transitions (P(s'|s,a) = 1)
    • Finite state space (manageable for small instances)  
    • Fully observable (complete state information)
    • Clear reward structure guiding toward optimal makespan
    
    Computational complexity:
    • State space size: Limited by job combinations and machine states
    • Action space per state: Typically 5-20 valid actions due to constraints
    • Memory usage: Polynomial in state space size
    • Time complexity: O(|S|²|A|) per iteration
    """
    
    print(insights)
    
    print("\n" + "="*80)
    print("POLICY ITERATION DEMONSTRATION COMPLETED")
    print("="*80)
    
    # Provide detailed MDP explanation
    explain_dynamic_fjsp_mdp()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED")
    print("="*80)
