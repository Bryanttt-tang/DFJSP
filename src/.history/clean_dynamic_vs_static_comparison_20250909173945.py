"""
Clean Dynamic vs Static RL Comparison for Poisson FJSP
=====================================================

This script compares:
1. Dynamic RL (trained on Poisson job arrivals) vs Static RL (trained on all jobs at t=0) 
2. Dynamic RL vs SPT Heuristic

All evaluations are done on the same Poisson job arrival test cases.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import gymnasium as gym
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- Job Data ---
ENHANCED_JOBS_DATA = collections.OrderedDict({
    0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}, {'proc_times': {'M0': 2}}],
    1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M1': 3, 'M2': 4}}],
    2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M0': 3, 'M1': 4}}, {'proc_times': {'M2': 7}}],
    3: [{'proc_times': {'M1': 8}}, {'proc_times': {'M2': 2}}, {'proc_times': {'M0': 5, 'M1': 6}}],
    4: [{'proc_times': {'M0': 6, 'M1': 9}}, {'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M2': 6}}],
    5: [{'proc_times': {'M1': 5, 'M2': 8}}, {'proc_times': {'M0': 6}}, {'proc_times': {'M1': 4, 'M2': 3}}],
    6: [{'proc_times': {'M0': 7, 'M2': 4}}, {'proc_times': {'M0': 5, 'M1': 6}}, {'proc_times': {'M1': 3}}, {'proc_times': {'M0': 2, 'M2': 5}}]
})

MACHINE_LIST = ['M0', 'M1', 'M2']


class StaticFJSPEnv(gym.Env):
    """
    Static FJSP Environment where all jobs are available at time 0.
    Used for training a baseline RL agent for comparison with dynamic agent.
    """
    
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, reward_mode="makespan_increment", seed=None):
        """Initialize the Static FJSP Environment."""
        super().__init__()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.reward_mode = reward_mode
        
        # Environment parameters
        self.num_jobs = len(self.job_ids)
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        
        # Action space: (job_idx, operation_idx, machine_idx)
        self.action_space = spaces.Discrete(
            min(self.num_jobs * self.max_ops_per_job * len(self.machines), 2000)
        )
        
        # Static observation space (simpler than dynamic)
        obs_size = (
            len(self.machines) +                      # Machine availability times
            self.num_jobs * self.max_ops_per_job +    # Operation completion status
            self.num_jobs +                           # Job progress ratios
            len(self.machines) +                      # Machine workloads
            1 +                                       # Current time/makespan
            1                                         # Number of completed jobs
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state variables
        self._reset_state()

    def _reset_state(self):
        """Reset all environment state variables."""
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.machine_workload = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        self.current_time = 0.0
        self.operations_scheduled = 0
        self.episode_step = 0
        self.max_episode_steps = self.total_operations * 3
        self.num_completed_jobs = 0

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        if seed is not None:
            super().reset(seed=seed, options=options)
            random.seed(seed)
            np.random.seed(seed)
        
        self._reset_state()
        return self._get_observation(), {}

    def _decode_action(self, action):
        """Decode action into job, operation, and machine indices."""
        action = int(action) % self.action_space.n
        
        num_machines = len(self.machines)
        ops_per_job = self.max_ops_per_job
        
        job_idx = action // (ops_per_job * num_machines)
        op_idx = (action % (ops_per_job * num_machines)) // num_machines
        machine_idx = action % num_machines
        
        # Ensure indices are within bounds
        job_idx = min(job_idx, self.num_jobs - 1)
        machine_idx = min(machine_idx, len(self.machines) - 1)
        
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        """Check if the action is valid."""
        if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
            return False
        
        job_id = self.job_ids[job_idx]
        
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
        """Generate action masks for valid actions."""
        mask = np.full(self.action_space.n, False, dtype=bool)
        
        if self.operations_scheduled >= self.total_operations:
            return mask

        for job_idx, job_id in enumerate(self.job_ids):
            op_idx = self.next_operation[job_id]
            
            if op_idx < len(self.jobs[job_id]):
                op_data = self.jobs[job_id][op_idx]
                
                for machine_idx, machine_name in enumerate(self.machines):
                    if machine_name in op_data['proc_times']:
                        action = job_idx * (self.max_ops_per_job * len(self.machines)) + \
                                 op_idx * len(self.machines) + machine_idx
                        
                        if action < self.action_space.n:
                            mask[action] = True
        
        return mask

    def step(self, action):
        """Execute one step in the environment."""
        self.episode_step += 1
        
        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        job_idx, op_idx, machine_idx = self._decode_action(action)

        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            return self._get_observation(), -100.0, False, False, {"error": "Invalid action"}

        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        
        # Calculate timing
        machine_available_time = self.machine_next_free.get(machine, 0.0)
        job_ready_time = self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 else 0.0
        
        start_time = max(machine_available_time, job_ready_time, self.current_time)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time
        
        # Calculate idle time
        idle_time = max(0, start_time - machine_available_time)

        # Update state
        previous_time = self.current_time
        self.machine_next_free[machine] = end_time
        self.machine_workload[machine] += proc_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        self.current_time = max(self.current_time, end_time)
        
        # Check if job is completed
        if self.next_operation[job_id] >= len(self.jobs[job_id]):
            self.num_completed_jobs += 1

        # Record in schedule
        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        # Check termination
        terminated = self.operations_scheduled >= self.total_operations

        # Calculate reward
        reward = self._calculate_reward(proc_time, idle_time, terminated, previous_time, self.current_time)
        
        info = {
            "makespan": self.current_time,
            "completed_operations": self.operations_scheduled,
            "idle_time": idle_time
        }
        
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_time, current_time):
        """Calculate reward for static environment."""
        if self.reward_mode == "makespan_increment":
            makespan_increment = current_time - previous_time
            reward = -makespan_increment - idle_time * 0.5 + 1.0
            
            if done:
                reward += 100.0
                if current_time > 0:
                    reward += max(0, 200.0 / current_time)
            
            return reward
        else:
            reward = 10.0 - proc_time * 0.1 - idle_time
            if done:
                reward += 100.0
            return reward

    def _get_observation(self):
        """Generate observation vector for static scheduling."""
        norm_factor = max(self.current_time, 1.0)
        obs = []
        
        # Machine availability (normalized by current time)
        for m in self.machines:
            value = float(self.machine_next_free.get(m, 0.0)) / norm_factor
            obs.append(max(0.0, min(1.0, value)))
        
        # Operation completion status (padded to max_ops_per_job)
        for job_id in self.job_ids:
            for op_idx in range(self.max_ops_per_job):
                if op_idx < len(self.jobs[job_id]):
                    completed = 1.0 if self.completed_ops[job_id][op_idx] else 0.0
                else:
                    completed = 1.0  # Non-existent operations considered completed
                obs.append(float(completed))
        
        # Job progress (proportion of operations completed)
        for job_id in self.job_ids:
            total_ops = len(self.jobs[job_id])
            if total_ops > 0:
                progress = float(self.next_operation[job_id]) / float(total_ops)
            else:
                progress = 1.0
            obs.append(max(0.0, min(1.0, progress)))
        
        # Machine workloads (normalized)
        max_workload = max(self.machine_workload.values()) if self.machine_workload.values() else 1.0
        for m in self.machines:
            workload_norm = float(self.machine_workload[m]) / max(max_workload, 1.0)
            obs.append(max(0.0, min(1.0, workload_norm)))
            
        # Current time (normalized)
        time_norm = float(self.current_time) / 100.0  # Assume max time around 100
        obs.append(max(0.0, min(1.0, time_norm)))
        
        # Number of completed jobs (normalized)
        completed_ratio = self.num_completed_jobs / len(self.job_ids)
        obs.append(max(0.0, min(1.0, completed_ratio)))
        
        # Pad or truncate to match observation space
        target_size = self.observation_space.shape[0]
        if len(obs) < target_size:
            obs.extend([0.0] * (target_size - len(obs)))
        elif len(obs) > target_size:
            obs = obs[:target_size]
        
        # Ensure proper format
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array

    def render(self, mode='human'):
        """Render the current state (optional)."""
        if mode == 'human':
            print(f"\n=== Time: {self.current_time:.2f} ===")
            print(f"Completed operations: {self.operations_scheduled}")
            print(f"Machine status:")
            for m in self.machines:
                print(f"  {m}: next free at {self.machine_next_free[m]:.2f}")


class PoissonDynamicFJSPEnv(gym.Env):
    """
    Dynamic FJSP Environment with Poisson-distributed job arrivals.
    
    Key features:
    - Initial jobs available at start
    - Remaining jobs arrive according to Poisson process
    - RL agent must adapt to unexpected job arrivals
    - No MILP solution possible due to dynamic nature
    """
    
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05, 
                 max_time_horizon=200, reward_mode="makespan_increment", seed=None):
        """
        Initialize the Poisson Dynamic FJSP Environment.
        
        Args:
            jobs_data: Dictionary of all possible jobs
            machine_list: List of available machines
            initial_jobs: Number of jobs available at start (default: 5)
            arrival_rate: Poisson rate parameter (jobs per time unit, default: 0.1)
            max_time_horizon: Maximum simulation time
            reward_mode: Reward function type
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        
        # Handle initial_jobs as either integer or list
        if isinstance(initial_jobs, list):
            self.initial_job_ids = initial_jobs
            self.dynamic_job_ids = [j for j in self.job_ids if j not in initial_jobs]
            self.initial_jobs = len(initial_jobs)
        else:
            self.initial_jobs = min(initial_jobs, len(self.job_ids))
            self.initial_job_ids = self.job_ids[:self.initial_jobs]
            self.dynamic_job_ids = self.job_ids[self.initial_jobs:]
        
        self.arrival_rate = arrival_rate
        self.max_time_horizon = max_time_horizon
        self.reward_mode = reward_mode
        
        # Environment parameters
        self.num_jobs = len(self.job_ids)
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        
        # Action space: (job_idx, operation_idx, machine_idx)
        self.action_space = spaces.Discrete(
            min(self.num_jobs * self.max_ops_per_job * len(self.machines), 2000)
        )
        
        # Enhanced observation space for dynamic environment
        obs_size = (
            len(self.machines) +          # Machine availability times
            self.num_jobs * self.max_ops_per_job +  # Operation completion status
            self.num_jobs +               # Job progress ratios
            self.num_jobs +               # Job arrival status
            len(self.machines) +          # Machine workloads
            1 +                           # Current time/makespan
            1 +                           # Number of arrived jobs
            1 +                           # Number of completed jobs
            self.initial_jobs +           # Initial job completion status
            len(self.dynamic_job_ids)     # Dynamic job arrival indicators
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state variables
        self._reset_state()

    def _reset_state(self):
        """Reset all environment state variables."""
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.machine_workload = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        self.current_time = 0.0
        self.operations_scheduled = 0
        self.episode_step = 0
        self.max_episode_steps = self.total_operations * 3
        
        # Job arrival management
        self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
        self.arrival_times = {}
        self.next_arrival_events = []
        
        # Generate Poisson arrival times for dynamic jobs
        self._generate_poisson_arrivals()
        
        # Performance tracking
        self.total_idle_time = 0.0
        self.total_tardiness = 0.0
        self.num_completed_jobs = 0

    def _generate_poisson_arrivals(self):
        """Generate arrival times for dynamic jobs using Poisson process."""
        self.arrival_times = {job_id: 0.0 for job_id in self.initial_job_ids}
        
        # Generate inter-arrival times using exponential distribution
        current_time = 0.0
        for job_id in self.dynamic_job_ids:
            inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
            current_time += inter_arrival_time
            
            if current_time <= self.max_time_horizon:
                self.arrival_times[job_id] = current_time
                self.next_arrival_events.append((current_time, job_id))
            else:
                self.arrival_times[job_id] = float('inf')  # Won't arrive in this episode
        
        # Sort arrival events by time
        self.next_arrival_events.sort(key=lambda x: x[0])

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        if seed is not None:
            super().reset(seed=seed, options=options)
            random.seed(seed)
            np.random.seed(seed)
        
        self._reset_state()
        return self._get_observation(), {}

    def _decode_action(self, action):
        """Decode action into job, operation, and machine indices."""
        action = int(action) % self.action_space.n
        
        num_machines = len(self.machines)
        ops_per_job = self.max_ops_per_job
        
        job_idx = action // (ops_per_job * num_machines)
        op_idx = (action % (ops_per_job * num_machines)) // num_machines
        machine_idx = action % num_machines
        
        # Ensure indices are within bounds
        job_idx = min(job_idx, self.num_jobs - 1)
        machine_idx = min(machine_idx, len(self.machines) - 1)
        
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        """Check if the action is valid."""
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
        """Enhanced action masks for dynamic scheduling with future-aware filtering."""
        mask = np.full(self.action_space.n, False, dtype=bool)
        
        # Early termination if all operations scheduled
        if self.operations_scheduled >= self.total_operations:
            return mask

        valid_actions = []
        
        # Collect all valid actions first
        for job_idx, job_id in enumerate(self.job_ids):
            if job_id not in self.arrived_jobs:
                continue

            op_idx = self.next_operation[job_id]
            
            if op_idx < len(self.jobs[job_id]):
                op_data = self.jobs[job_id][op_idx]
                
                for machine_idx, machine_name in enumerate(self.machines):
                    if machine_name in op_data['proc_times']:
                        action = job_idx * (self.max_ops_per_job * len(self.machines)) + \
                                 op_idx * len(self.machines) + machine_idx
                        
                        if action < self.action_space.n:
                            proc_time = op_data['proc_times'][machine_name]
                            valid_actions.append((action, job_id, machine_name, proc_time))
        
        # ENHANCED: Apply dynamic scheduling heuristics
        if valid_actions and len(self.next_arrival_events) > 0:
            current_time = self.current_time
            next_arrival_time = self.next_arrival_events[0][0]
            time_to_next_arrival = next_arrival_time - current_time
            
            # Smart filtering based on arrival timing
            filtered_actions = []
            
            for action_data in valid_actions:
                action, job_id, machine_name, proc_time = action_data
                
                # Calculate when this operation would complete
                machine_available_time = max(current_time, self.machine_next_free[machine_name])
                completion_time = machine_available_time + proc_time
                
                # Dynamic scheduling logic:
                include_action = False
                
                # 1. If arrival is imminent (≤ 3 time units), prioritize short operations
                if time_to_next_arrival <= 3.0:
                    if proc_time <= 5.0 or len(filtered_actions) < 2:
                        include_action = True
                
                # 2. If arrival is soon (3-8 time units), balance workload
                elif time_to_next_arrival <= 8.0:
                    machine_loads = [self.machine_next_free[m] for m in self.machines]
                    current_machine_load = self.machine_next_free[machine_name]
                    median_load = np.median(machine_loads) if machine_loads else 0
                    
                    # Prefer less loaded machines
                    if current_machine_load <= median_load or len(filtered_actions) < 3:
                        include_action = True
                
                # 3. For distant arrivals, allow all actions
                else:
                    include_action = True
                
                if include_action:
                    filtered_actions.append(action_data)
            
            # Use filtered actions if we have any
            if filtered_actions:
                for action_data in filtered_actions:
                    mask[action_data[0]] = True
            else:
                # Fallback to first valid action
                mask[valid_actions[0][0]] = True
        else:
            # No future arrivals or no valid actions - use all valid actions
            for action_data in valid_actions:
                mask[action_data[0]] = True
        
        # Fallback safety: ensure at least one action if jobs are available
        if not np.any(mask) and valid_actions:
            mask[valid_actions[0][0]] = True
        
        return mask

    def _update_arrivals(self, current_time):
        """Update job arrivals based on current time."""
        newly_arrived = []
        
        while (self.next_arrival_events and 
               self.next_arrival_events[0][0] <= current_time):
            arrival_time, job_id = self.next_arrival_events.pop(0)
            if job_id not in self.arrived_jobs:
                self.arrived_jobs.add(job_id)
                newly_arrived.append(job_id)
        
        return newly_arrived

    def step(self, action):
        """Execute one step in the environment."""
        self.episode_step += 1
        
        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        job_idx, op_idx, machine_idx = self._decode_action(action)

        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            return self._get_observation(), -100.0, False, False, {"error": "Invalid action"}

        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        
        # CRITICAL BUG CHECK: Ensure job has actually arrived
        job_arrival_time = self.arrival_times.get(job_id, 0.0)
        if self.current_time < job_arrival_time:
            print(f"ERROR: Trying to schedule Job {job_id} at time {self.current_time:.2f} but it arrives at {job_arrival_time:.2f}")
            return self._get_observation(), -1000.0, False, False, {"error": "Scheduling before arrival"}
        
        # Calculate timing
        machine_available_time = self.machine_next_free.get(machine, 0.0)
        job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                         else self.arrival_times.get(job_id, 0.0))
        
        start_time = max(machine_available_time, job_ready_time, self.current_time)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time
        
        # Calculate idle time for this machine
        idle_time = max(0, start_time - machine_available_time)
        self.total_idle_time += idle_time

        # Update state
        previous_time = self.current_time
        self.machine_next_free[machine] = end_time
        self.machine_workload[machine] += proc_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        self.current_time = max(self.current_time, end_time)
        
        # Check for newly arrived jobs
        newly_arrived = self._update_arrivals(self.current_time)
        
        # Check if job is completed
        if self.next_operation[job_id] >= len(self.jobs[job_id]):
            self.num_completed_jobs += 1

        # Record in schedule
        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        # Check termination - all arrived jobs completed
        arrived_ops = sum(len(self.jobs[job_id]) for job_id in self.arrived_jobs)
        terminated = self.operations_scheduled >= arrived_ops
        
        # If all current jobs done but more may arrive, continue
        if terminated and self.next_arrival_events:
            # Advance time to next arrival if no current work
            if not any(self.next_operation[job_id] < len(self.jobs[job_id]) for job_id in self.arrived_jobs):
                next_arrival_time = self.next_arrival_events[0][0]
                if next_arrival_time <= self.max_time_horizon:
                    self.current_time = next_arrival_time
                    self._update_arrivals(self.current_time)
                    terminated = False

        # Calculate reward
        reward = self._calculate_reward(
            proc_time, idle_time, terminated, 
            len(newly_arrived), previous_time, self.current_time
        )
        
        info = {
            "makespan": self.current_time,
            "newly_arrived_jobs": len(newly_arrived),
            "total_arrived_jobs": len(self.arrived_jobs),
            "completed_operations": self.operations_scheduled,
            "idle_time": idle_time
        }
        
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, num_new_arrivals, 
                         previous_time, current_time):
        """Enhanced reward function for dynamic scheduling that encourages adaptation."""
        
        if self.reward_mode == "makespan_increment":
            # Base reward: minimize makespan increment
            makespan_increment = current_time - previous_time
            reward = -makespan_increment
            
            # Efficiency rewards
            reward -= idle_time * 0.5  # Penalty for idle time
            reward += 1.0  # Small step reward for progress
            
            # ENHANCED: Dynamic adaptation rewards
            
            # 1. Future-awareness reward: bonus for leaving machines available when arrivals are imminent
            if self.next_arrival_events:
                next_arrival_time = self.next_arrival_events[0][0]
                time_to_next_arrival = next_arrival_time - current_time
                
                if 0 < time_to_next_arrival <= 10.0:  # Next arrival is soon
                    # Check machine availability
                    available_machines = sum(1 for m in self.machines 
                                           if self.machine_next_free[m] <= current_time + 2.0)
                    if available_machines > 0:
                        reward += 2.0 * available_machines  # Bonus for keeping machines available
            
            # 2. Load balancing reward: encourage balanced machine usage
            machine_loads = [self.machine_next_free[m] for m in self.machines]
            if max(machine_loads) > 0:
                load_balance = 1.0 - (np.std(machine_loads) / max(machine_loads))
                reward += load_balance * 1.0
            
            # 3. Arrival adaptation reward: bonus for scheduling efficiently after new arrivals
            if num_new_arrivals > 0:
                # Check if we're making good use of newly arrived jobs
                recently_arrived_scheduled = 0
                for arrival_time, job_id in self.next_arrival_events:
                    if arrival_time <= previous_time and self.next_operation[job_id] > 0:
                        recently_arrived_scheduled += 1
                
                if recently_arrived_scheduled > 0:
                    reward += 3.0 * recently_arrived_scheduled  # Reward quick utilization of new jobs
            
            # 4. Long-term planning reward: penalty for poor decisions when more work is coming
            future_work_ratio = len([j for j in self.job_ids if j not in self.arrived_jobs]) / len(self.job_ids)
            if future_work_ratio > 0.3:  # Significant future work remaining
                # Penalty for creating large idle times when more work is coming
                if idle_time > 5.0:
                    reward -= 2.0 * future_work_ratio
            
            # Completion bonus
            if done:
                reward += 100.0
                # Enhanced final bonus based on how well we used dynamic information
                final_makespan = current_time
                if final_makespan > 0:
                    efficiency_bonus = max(0, 200.0 / final_makespan)
                    reward += efficiency_bonus
                    
                    # Extra bonus for good load balancing at completion
                    final_loads = [self.machine_next_free[m] for m in self.machines]
                    balance_bonus = 1.0 - (np.std(final_loads) / max(final_loads)) if max(final_loads) > 0 else 1.0
                    reward += balance_bonus * 50.0
            
            return reward
            
        elif self.reward_mode == "dynamic_adaptation":
            reward = 0.0
            
            # Base reward for completing operation
            reward += 10.0
            
            # Efficiency rewards
            reward -= proc_time * 0.1  # Penalty for long operations
            reward -= idle_time * 2.0  # Strong penalty for machine idle time
            
            # Time progression penalty (encourage faster completion)
            time_penalty = (current_time - previous_time) * 0.5
            reward -= time_penalty
            
            # Adaptability bonus for handling new arrivals
            if num_new_arrivals > 0:
                reward += num_new_arrivals * 5.0  # Bonus for adapting to new jobs
            
            # Completion bonus
            if done:
                # Base completion bonus
                reward += 100.0
                
                # Bonus based on final makespan (shorter is better)
                if current_time > 0:
                    efficiency_bonus = max(0, 200.0 / current_time)
                    reward += efficiency_bonus
                
                # Bonus for completing all available jobs quickly
                if self.num_completed_jobs == len(self.arrived_jobs):
                    reward += 50.0
            
            return reward
            
        elif self.reward_mode == "makespan_minimization":
            # Simple makespan-focused reward
            reward = -proc_time  # Encourage shorter operations
            
            if done:
                reward += max(0, 500.0 / current_time) if current_time > 0 else 100.0
            
            return reward
            
        else:  # Default basic reward
            reward = 10.0 - proc_time * 0.1 - idle_time
            if done:
                reward += 100.0
            return reward

    def _get_observation(self):
        """Generate enhanced observation vector for dynamic scheduling."""
        norm_factor = max(self.current_time, 1.0)
        obs = []
        
        # Machine availability (normalized by current time)
        for m in self.machines:
            value = float(self.machine_next_free.get(m, 0.0)) / norm_factor
            obs.append(max(0.0, min(1.0, value)))
        
        # Operation completion status (padded to max_ops_per_job)
        for job_id in self.job_ids:
            for op_idx in range(self.max_ops_per_job):
                if op_idx < len(self.jobs[job_id]):
                    completed = 1.0 if self.completed_ops[job_id][op_idx] else 0.0
                else:
                    completed = 1.0  # Non-existent operations considered completed
                obs.append(float(completed))
        
        # Job progress (proportion of operations completed)
        for job_id in self.job_ids:
            total_ops = len(self.jobs[job_id])
            if total_ops > 0:
                progress = float(self.next_operation[job_id]) / float(total_ops)
            else:
                progress = 1.0
            obs.append(max(0.0, min(1.0, progress)))
            
        # ENHANCED: Job arrival status with timing information
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                # Job has arrived
                obs.append(1.0)  # Arrival status
                obs.append(0.0)  # Time until arrival (0 since already arrived)
            else:
                # Job hasn't arrived yet
                obs.append(0.0)  # Arrival status
                arrival_time = self.arrival_times.get(job_id, float('inf'))
                if arrival_time == float('inf'):
                    time_until_arrival = 1.0  # Won't arrive this episode
                else:
                    time_until_arrival = max(0.0, arrival_time - self.current_time)
                    time_until_arrival = min(1.0, time_until_arrival / self.max_time_horizon)
                obs.append(float(time_until_arrival))
        
        # ENHANCED: Future workload information
        # Calculate expected workload from jobs that will arrive soon
        future_workload = 0.0
        time_window = 20.0  # Look ahead 20 time units
        for job_id in self.job_ids:
            if job_id not in self.arrived_jobs:
                arrival_time = self.arrival_times.get(job_id, float('inf'))
                if arrival_time != float('inf') and arrival_time <= self.current_time + time_window:
                    # Add workload from this future job
                    job_workload = sum(min(op['proc_times'].values()) for op in self.jobs[job_id])
                    future_workload += job_workload
        
        future_workload_norm = min(1.0, future_workload / 50.0)  # Normalize
        obs.append(float(future_workload_norm))
        
        # ENHANCED: Time until next arrival
        if self.next_arrival_events:
            next_arrival_time = self.next_arrival_events[0][0]
            time_to_next = max(0.0, next_arrival_time - self.current_time)
            time_to_next_norm = min(1.0, time_to_next / 30.0)  # Normalize to 30 time units
        else:
            time_to_next_norm = 1.0  # No more arrivals
        obs.append(float(time_to_next_norm))
        
        # Machine workloads (normalized)
        max_workload = max(self.machine_workload.values()) if self.machine_workload.values() else 1.0
        for m in self.machines:
            workload_norm = float(self.machine_workload[m]) / max(max_workload, 1.0)
            obs.append(max(0.0, min(1.0, workload_norm)))
            
        # Current time (normalized)
        time_norm = float(self.current_time) / max(self.max_time_horizon, 1.0)
        obs.append(max(0.0, min(1.0, time_norm)))
        
        # Number of arrived jobs (normalized)
        arrived_ratio = len(self.arrived_jobs) / len(self.job_ids)
        obs.append(max(0.0, min(1.0, arrived_ratio)))
        
        # ENHANCED: Urgency indicators
        # How much work is currently available vs how much is coming
        current_available_ops = sum(
            len(self.jobs[job_id]) - self.next_operation[job_id]
            for job_id in self.arrived_jobs
        )
        future_ops = sum(
            len(self.jobs[job_id])
            for job_id in self.job_ids if job_id not in self.arrived_jobs
        )
        
        if current_available_ops + future_ops > 0:
            current_vs_future = current_available_ops / (current_available_ops + future_ops)
        else:
            current_vs_future = 1.0
        obs.append(float(current_vs_future))
        
        # Number of completed jobs (normalized)
        completed_ratio = self.num_completed_jobs / len(self.job_ids)
        obs.append(max(0.0, min(1.0, completed_ratio)))
        
        # Initial job completion status
        for job_id in self.initial_job_ids:
            completed = 1.0 if self.next_operation[job_id] >= len(self.jobs[job_id]) else 0.0
            obs.append(float(completed))
        
        # Dynamic job arrival indicators (has this dynamic job arrived?)
        for job_id in self.dynamic_job_ids:
            arrived = 1.0 if job_id in self.arrived_jobs else 0.0
            obs.append(float(arrived))
        
        # Pad or truncate to match observation space
        target_size = self.observation_space.shape[0]
        if len(obs) < target_size:
            obs.extend([0.0] * (target_size - len(obs)))
        elif len(obs) > target_size:
            obs = obs[:target_size]
        
        # Ensure proper format
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array

    def debug_step(self, action):
        """Debug version of step function to identify issues."""
        print(f"\n=== DEBUG STEP ===")
        job_idx, op_idx, machine_idx = self._decode_action(action)
        print(f"Action: {action} -> Job {job_idx}, Op {op_idx}, Machine {machine_idx}")
        
        if job_idx < len(self.job_ids):
            job_id = self.job_ids[job_idx]
            print(f"Job ID: {job_id}")
            print(f"Job arrived: {job_id in self.arrived_jobs}")
            print(f"Next operation: {self.next_operation[job_id]}")
            print(f"Job operations: {len(self.jobs[job_id])}")
            
            if op_idx < len(self.jobs[job_id]):
                print(f"Operation data: {self.jobs[job_id][op_idx]}")
                machine_name = self.machines[machine_idx]
                print(f"Machine {machine_name} can process: {machine_name in self.jobs[job_id][op_idx]['proc_times']}")
        
        print(f"Valid action: {self._is_valid_action(job_idx, op_idx, machine_idx)}")
        print(f"Arrived jobs: {sorted(self.arrived_jobs)}")
        print(f"Current time: {self.current_time}")
        print("================")
        
        return self.step(action)

    def render(self, mode='human'):
        """Render the current state (optional)."""
        if mode == 'human':
            print(f"\n=== Time: {self.current_time:.2f} ===")
            print(f"Arrived jobs: {sorted(self.arrived_jobs)}")
            print(f"Completed operations: {self.operations_scheduled}")
            print(f"Machine status:")
            for m in self.machines:
                print(f"  {m}: next free at {self.machine_next_free[m]:.2f}")


def mask_fn(env):
    """Mask function for ActionMasker wrapper"""
    return env.action_masks()

# --- Import Environment Classes ---
# Import the already working environment classes
# exec(open('dynamic_poisson_fjsp.py').read())

def train_static_agent(jobs_data, machine_list, total_timesteps=150000):
    """Train a static RL agent where all jobs are available at t=0."""
    print(f"\n--- Training Static RL Agent ---")
    print(f"Timesteps: {total_timesteps}")
    print("Training scenario: All jobs available at time 0")
    
    def make_static_env():
        env = StaticFJSPEnv(jobs_data, machine_list, reward_mode="makespan_increment")
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_static_env])
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        )
    )
    
    # Show a training episode every 50,000 timesteps
    for phase in range(3):
        phase_timesteps = total_timesteps // 3
        print(f"\nTraining Phase {phase + 1}/3 ({phase_timesteps} timesteps)")
        
        if phase == 0:
            print("Training episodes sample:")
            # Test current model performance
            test_env = StaticFJSPEnv(jobs_data, machine_list, reward_mode="makespan_increment")
            obs, _ = test_env.reset()
            total_reward = 0
            step_count = 0
            
            while step_count < 100:  # Quick test
                action_masks = test_env.action_masks()
                if not any(action_masks):
                    break
                    
                # Random action for initial demonstration
                valid_actions = [i for i, valid in enumerate(action_masks) if valid]
                if valid_actions:
                    action = np.random.choice(valid_actions)
                    obs, reward, done, truncated, info = test_env.step(action)
                    total_reward += reward
                    step_count += 1
                    
                    if step_count <= 5:  # Show first 5 steps
                        print(f"  Step {step_count}: Action {action}, Reward {reward:.2f}, Time {test_env.current_time:.2f}")
                    
                    if done or truncated:
                        break
            
            print(f"  Sample episode: {step_count} steps, Total reward: {total_reward:.2f}, Final makespan: {test_env.current_time:.2f}")
        
        model.learn(total_timesteps=phase_timesteps)
    
    print("Static RL agent training completed!")
    return model

def train_dynamic_agent(jobs_data, machine_list, initial_jobs=3, arrival_rate=0.08, total_timesteps=150000, reward_mode="utilization_maximization"):
    """Train a dynamic RL agent on Poisson job arrivals with improved reward function."""
    print(f"\n--- Training Dynamic RL Agent ---")
    print(f"Initial jobs: {initial_jobs}, Arrival rate: {arrival_rate}, Timesteps: {total_timesteps}")
    print(f"Reward mode: {reward_mode} (optimized for machine utilization)")
    print(f"Training scenario: Jobs {initial_jobs} at t=0, remaining jobs with Poisson arrivals")
    
    # Show sample arrival times from training environment
    print(f"\n--- Sample Training Episode Arrival Patterns ---")
    print(f"Initial jobs: {initial_jobs}, Arrival rate: {arrival_rate}")
    for sample in range(5):
        test_env = PoissonDynamicFJSPEnv(
            jobs_data, machine_list,
            initial_jobs=initial_jobs,
            arrival_rate=arrival_rate,
            reward_mode=reward_mode,
            seed=42 + sample  # Different seeds for variety
        )
        obs, _ = test_env.reset()
        print(f"Training Sample {sample + 1}: {test_env.arrival_times}")
    
    print(f"\nNote: During training, each episode generates DIFFERENT random arrival times")
    print(f"This teaches the agent to adapt to various Poisson arrival patterns.")
    print(f"The agent learns to handle uncertainty in job arrival timing.")
    
    def make_dynamic_env():
        env = PoissonDynamicFJSPEnv(
            jobs_data, machine_list, 
            initial_jobs=initial_jobs,
            arrival_rate=arrival_rate,
            reward_mode=reward_mode
        )
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_dynamic_env])
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.1,
        ent_coef=0.005,
        policy_kwargs=dict(
            net_arch=[512, 512, 256, 128],
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"\nTraining for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    print("Dynamic RL agent training completed!")
    return model

def generate_test_scenarios(jobs_data, initial_jobs=[0, 1, 2], arrival_rate=0.08, num_scenarios=5):
    """Generate test scenarios with fixed Poisson arrival times."""
    print(f"\n--- Generating {num_scenarios} Test Scenarios ---")
    print(f"Initial jobs: {initial_jobs}, Arrival rate: {arrival_rate}")
    
    scenarios = []
    for i in range(num_scenarios):
        np.random.seed(i)  # Fixed seeds for reproducibility
        arrival_times = {}
        
        # Initial jobs arrive at t=0
        for job_id in initial_jobs:
            arrival_times[job_id] = 0.0
        
        # Generate Poisson arrivals for remaining jobs
        remaining_jobs = [j for j in jobs_data.keys() if j not in initial_jobs]
        current_time = 0.0
        
        for job_id in remaining_jobs:
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            # print(f"  Job {job_id} inter-arrival time: {inter_arrival_time:.2f}")
            current_time += inter_arrival_time
            if current_time <= 200:  # Max simulation time
                arrival_times[job_id] = current_time
            else:
                arrival_times[job_id] = float('inf')  # Won't arrive
        
        scenarios.append({
            'scenario_id': i,
            'arrival_times': arrival_times,
            'seed': 100 + i
        })
        print(f"Scenario {i+1}: {arrival_times}")
    
    return scenarios

def evaluate_on_test_scenarios(model, test_scenarios, jobs_data, machine_list, method_name="Model", is_static_model=False):
    """Evaluate a model on multiple predefined test scenarios."""
    print(f"\n--- Evaluating {method_name} on {len(test_scenarios)} Test Scenarios ---")
    
    results = []
    
    for scenario in test_scenarios:
        scenario_id = scenario['scenario_id']
        arrival_times = scenario['arrival_times']
        seed = scenario['seed']
        
        if is_static_model:
            # For static models, use the special evaluation function
            makespan, schedule = evaluate_static_on_dynamic(
                model, jobs_data, machine_list, arrival_times)
            
            results.append({
                'scenario_id': scenario_id,
                'makespan': makespan,
                'schedule': schedule,
                'arrival_times': arrival_times,
                'steps': 0,  # Not tracked for this evaluation
                'reward': 0  # Not tracked for this evaluation
            })
        else:
            # For dynamic models, use the existing evaluation
            makespan, schedule = evaluate_dynamic_on_dynamic(
                model, jobs_data, machine_list, arrival_times)
            
            results.append({
                'scenario_id': scenario_id,
                'makespan': makespan,
                'schedule': schedule,
                'arrival_times': arrival_times,
                'steps': 0,
                'reward': 0
            })
        
        print(f"  Scenario {scenario_id+1}: Makespan = {makespan:.2f}")
    
    # Calculate statistics
    makespans = [r['makespan'] for r in results]
    avg_makespan = np.mean(makespans)
    std_makespan = np.std(makespans)
    min_makespan = np.min(makespans)
    max_makespan = np.max(makespans)
    
    print(f"Results for {method_name}:")
    print(f"  Average Makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
    print(f"  Best Makespan: {min_makespan:.2f}")
    print(f"  Worst Makespan: {max_makespan:.2f}")
    
    # Return best result for visualization
    best_result = min(results, key=lambda x: x['makespan'])
    return best_result['makespan'], best_result['schedule'], best_result['arrival_times']

def evaluate_static_on_dynamic(static_model, jobs_data, machine_list, arrival_times):
    """Evaluate static model on dynamic scenario with observation space mapping."""
    print(f"  Static RL using arrival times: {arrival_times}")
    
    # Create static environment to get expected observation size
    static_env = StaticFJSPEnv(jobs_data, machine_list)
    static_obs_size = static_env.observation_space.shape[0]
    
    # Create dynamic environment for testing
    test_env = PoissonDynamicFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=[k for k, v in arrival_times.items() if v == 0],
        arrival_rate=0.1,
        reward_mode="makespan_increment"
    )
    
    # Override with fixed arrival times
    test_env.arrival_times = arrival_times.copy()
    test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
    test_env.next_arrival_events.sort(key=lambda x: x[0])
    
    obs, _ = test_env.reset()
    
    # Re-override after reset
    test_env.arrival_times = arrival_times.copy()
    test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
    test_env.next_arrival_events.sort(key=lambda x: x[0])
    test_env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    
    def map_dynamic_to_static_obs(dynamic_obs):
        """Map dynamic observation to static observation space."""
        static_obs = []
        
        # Machine availability (first 3 elements)
        static_obs.extend(dynamic_obs[:len(machine_list)])
        
        # Operation completion status
        num_jobs = len(jobs_data)
        max_ops = max(len(ops) for ops in jobs_data.values())
        start_idx = len(machine_list)
        end_idx = start_idx + num_jobs * max_ops
        static_obs.extend(dynamic_obs[start_idx:end_idx])
        
        # Job progress ratios
        start_idx = end_idx
        end_idx = start_idx + num_jobs
        static_obs.extend(dynamic_obs[start_idx:end_idx])
        
        # Machine workloads (skip job arrival info in dynamic obs)
        machine_workloads_start = len(machine_list) + num_jobs * max_ops + num_jobs + num_jobs
        static_obs.extend(dynamic_obs[machine_workloads_start:machine_workloads_start + len(machine_list)])
        
        # Current time
        time_idx = machine_workloads_start + len(machine_list)
        static_obs.append(dynamic_obs[time_idx])
        
        # Number of completed jobs
        completed_jobs_idx = time_idx + 2  # Skip arrived jobs count
        static_obs.append(dynamic_obs[completed_jobs_idx])
        
        # Pad or truncate to match static observation space
        if len(static_obs) < static_obs_size:
            static_obs.extend([0.0] * (static_obs_size - len(static_obs)))
        elif len(static_obs) > static_obs_size:
            static_obs = static_obs[:static_obs_size]
        
        return np.array(static_obs, dtype=np.float32)
    
    step_count = 0
    invalid_actions = 0
    
    while step_count < 1000:
        action_masks = test_env.action_masks()
        
        if not any(action_masks):
            break
        
        # Map observation and predict action
        static_obs = map_dynamic_to_static_obs(obs)
        action, _ = static_model.predict(static_obs, action_masks=action_masks, deterministic=True)
        
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if info.get("error") == "Invalid action":
            invalid_actions += 1
        
        if done or truncated:
            break
    
    makespan = test_env.current_time
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    for machine_ops in test_env.schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    print(f"  Static RL scheduled jobs: {sorted(scheduled_jobs)}")
    
    return makespan, test_env.schedule

def evaluate_dynamic_on_dynamic(dynamic_model, jobs_data, machine_list, arrival_times):
    """Evaluate dynamic model on dynamic scenario."""
    print(f"  Dynamic RL using arrival times: {arrival_times}")
    
    test_env = PoissonDynamicFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=[k for k, v in arrival_times.items() if v == 0],
        arrival_rate=0.1,
        reward_mode="dynamic_adaptation"
    )
    
    # Override with fixed arrival times
    test_env.arrival_times = arrival_times.copy()
    test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
    test_env.next_arrival_events.sort(key=lambda x: x[0])
    
    obs, _ = test_env.reset()
    
    # Re-override after reset
    test_env.arrival_times = arrival_times.copy()
    test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
    test_env.next_arrival_events.sort(key=lambda x: x[0])
    test_env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    
    print(f"  Dynamic RL initial jobs: {test_env.arrived_jobs}")
    print(f"  Dynamic RL future arrivals: {test_env.next_arrival_events}")
    
    step_count = 0
    invalid_actions = 0
    
    while step_count < 1000:
        action_masks = test_env.action_masks()
        
        if not any(action_masks):
            break
        
        action, _ = dynamic_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if info.get("error") == "Invalid action":
            invalid_actions += 1
        
        if done or truncated:
            break
    
    makespan = test_env.current_time
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    for machine_ops in test_env.schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    print(f"  Dynamic RL scheduled jobs: {sorted(scheduled_jobs)}")
    
    return makespan, test_env.schedule

def spt_heuristic_poisson(jobs_data, machine_list, arrival_times):
    """SPT (Shortest Processing Time) heuristic for dynamic scheduling."""
    print(f"  SPT Heuristic using arrival times: {arrival_times}")
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    schedule = {m: [] for m in machine_list}
    
    arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    operations_scheduled = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    sim_time = 0.0
    
    print(f"  SPT initial jobs: {arrived_jobs}")
    future_arrivals = {job_id: arr_time for job_id, arr_time in arrival_times.items() if arr_time > 0}
    print(f"  SPT future arrivals: {future_arrivals}")
    
    while operations_scheduled < total_operations:
        # Update arrivals
        new_arrivals = []
        for job_id, arr_time in arrival_times.items():
            if job_id not in arrived_jobs and arr_time <= sim_time:
                arrived_jobs.add(job_id)
                new_arrivals.append(job_id)
        
        # Collect available operations
        available_ops = []
        for job_id in arrived_jobs:
            next_op = next_operation_for_job[job_id]
            if next_op < len(jobs_data[job_id]):
                # Check if job is ready (previous operation completed)
                if next_op == 0 or operation_end_times[job_id][next_op - 1] <= sim_time:
                    op_data = jobs_data[job_id][next_op]
                    for machine, proc_time in op_data['proc_times'].items():
                        available_ops.append((job_id, next_op, machine, proc_time))
        
        if not available_ops:
            # No operations available, advance time to next event
            next_events = []
            
            # Next machine available time
            for m, next_free in machine_next_free.items():
                if next_free > sim_time:
                    next_events.append(next_free)
            
            # Next job arrival
            for job_id, arr_time in arrival_times.items():
                if job_id not in arrived_jobs and arr_time > sim_time:
                    next_events.append(arr_time)
            
            # Next operation ready time
            for job_id in arrived_jobs:
                next_op = next_operation_for_job[job_id]
                if next_op > 0 and next_op < len(jobs_data[job_id]):
                    ready_time = operation_end_times[job_id][next_op - 1]
                    if ready_time > sim_time:
                        next_events.append(ready_time)
            
            if next_events:
                sim_time = min(next_events)
            else:
                break
            continue
        
        # Apply SPT: select operation with shortest processing time
        available_ops.sort(key=lambda x: x[3])  # Sort by processing time
        job_id, op_idx, machine, proc_time = available_ops[0]
        
        # Calculate timing
        machine_available_time = machine_next_free[machine]
        job_ready_time = operation_end_times[job_id][op_idx - 1] if op_idx > 0 else arrival_times[job_id]
        
        start_time = max(machine_available_time, job_ready_time, sim_time)
        end_time = start_time + proc_time
        
        # Update state
        machine_next_free[machine] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled += 1
        sim_time = max(sim_time, end_time)
        
        # Record in schedule
        schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    for machine_ops in schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    print(f"  SPT Heuristic scheduled jobs: {sorted(scheduled_jobs)}")
    
    return makespan, schedule

def plot_gantt(schedule, machines, title="Schedule", save_path=None):
    """Plot Gantt chart for the schedule."""
    if not schedule or all(len(ops) == 0 for ops in schedule.values()):
        print("No schedule to plot - schedule is empty")
        return

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(14, len(machines) * 0.8 + 1))

    for idx, m in enumerate(machines):
        machine_ops = schedule.get(m, [])
        machine_ops.sort(key=lambda x: x[1])  # Sort by start time

        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op, start_time, end_time = op_data[:3]
                duration = end_time - start_time
                
                # Extract job number for coloring
                job_num = 0
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                    except:
                        job_num = 0
                
                color = colors[job_num % len(colors)]
                
                ax.barh(idx, duration, left=start_time, height=0.6, 
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add operation label
                if duration > 1:  # Only add text if bar is wide enough
                    ax.text(start_time + duration/2, idx, job_op, 
                           ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gantt chart saved to {save_path}")
    
    plt.show()

def main():
 
    print("=" * 80)
    print("DYNAMIC vs STATIC RL COMPARISON FOR POISSON FJSP")
    print("=" * 80)
    print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
    print("Research Question: Does Dynamic RL outperform Static RL on Poisson arrivals?")
    print("=" * 80)
    arrival_rate = 0.1

        # Step 2: Generate test scenarios
    print("\n2. TEST SCENARIO GENERATION")
    print("-" * 40)
    test_scenarios = generate_test_scenarios(ENHANCED_JOBS_DATA, 
                                           initial_jobs=[0, 1, 2], 
                                           arrival_rate=arrival_rate, 
                                           num_scenarios=5)
    
    # Step 1: Train both agents
    print("\n1. TRAINING PHASE")
    print("-" * 40)
    
    # Train dynamic RL agent (Poisson job arrivals)
    dynamic_model = train_dynamic_agent(ENHANCED_JOBS_DATA, MACHINE_LIST, 
                                       initial_jobs=[0, 1, 2], arrival_rate=arrival_rate, total_timesteps=300000)

    # Train static RL agent (all jobs available at t=0)
    static_model = train_static_agent(ENHANCED_JOBS_DATA, MACHINE_LIST, total_timesteps=100000)
    
    
    
    # Step 3: Evaluate all methods on the same test scenarios
    print("\n3. EVALUATION PHASE")
    print("-" * 40)
    print("Using FIRST test scenario for fair comparison...")
    
    # Use only the first test scenario for all evaluations
    first_scenario = test_scenarios[0]
    first_scenario_arrivals = first_scenario['arrival_times']
    
    print(f"Test scenario arrival times: {first_scenario_arrivals}")
    
    # Static RL on dynamic scenario (first scenario only)
    print("Evaluating Static RL on first test scenario...")
    static_makespan, static_schedule = evaluate_static_on_dynamic(
        static_model, ENHANCED_JOBS_DATA, MACHINE_LIST, first_scenario_arrivals)
    
    # Dynamic RL on dynamic scenario (first scenario only)
    print("Evaluating Dynamic RL on first test scenario...")
    dynamic_makespan, dynamic_schedule = evaluate_dynamic_on_dynamic(
        dynamic_model, ENHANCED_JOBS_DATA, MACHINE_LIST, first_scenario_arrivals)
    
    # SPT Heuristic on dynamic scenario (first scenario only)
    print("Evaluating SPT Heuristic on first test scenario...")
    spt_makespan, spt_schedule = spt_heuristic_poisson(ENHANCED_JOBS_DATA, MACHINE_LIST, first_scenario_arrivals)
    
    # Step 4: Results Analysis
    print("\n4. RESULTS ANALYSIS")
    print("=" * 50)
    print(f"Static RL    - Best Makespan: {static_makespan:.2f}")
    print(f"Dynamic RL   - Best Makespan: {dynamic_makespan:.2f}")
    print(f"SPT Heuristic- Best Makespan: {spt_makespan:.2f}")
    
    # Performance comparisons
    print("\n5. PERFORMANCE COMPARISON")
    print("-" * 40)
    
    # Dynamic RL vs Static RL
    if dynamic_makespan < static_makespan:
        improvement = ((static_makespan - dynamic_makespan) / static_makespan) * 100
        print(f"✓ Dynamic RL outperforms Static RL by {improvement:.1f}%")
    else:
        gap = ((dynamic_makespan - static_makespan) / static_makespan) * 100
        print(f"✗ Dynamic RL underperforms Static RL by {gap:.1f}%")
    
    # Dynamic RL vs SPT
    if dynamic_makespan < spt_makespan:
        improvement = ((spt_makespan - dynamic_makespan) / spt_makespan) * 100
        print(f"✓ Dynamic RL outperforms SPT Heuristic by {improvement:.1f}%")
    else:
        gap = ((dynamic_makespan - spt_makespan) / spt_makespan) * 100
        print(f"✗ Dynamic RL underperforms SPT Heuristic by {gap:.1f}%")
    
    # Step 6: Generate Gantt Charts for Comparison
    print(f"\n6. GANTT CHART COMPARISON")
    print("-" * 60)
    
    # Create a three-subplot comparison
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Scheduling Comparison: Dynamic RL vs Static RL vs SPT Heuristic\n' + 
                 f'Test Scenario: Jobs 0-2 at t=0, Jobs 3-6 via Poisson arrivals', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab20.colors
    schedules_data = [
        {'schedule': dynamic_schedule, 'makespan': dynamic_makespan, 'title': 'Dynamic RL', 'arrival_times': first_scenario_arrivals},
        {'schedule': static_schedule, 'makespan': static_makespan, 'title': 'Static RL', 'arrival_times': first_scenario_arrivals},
        {'schedule': spt_schedule, 'makespan': spt_makespan, 'title': 'SPT Heuristic', 'arrival_times': first_scenario_arrivals}
    ]
    
    for plot_idx, data in enumerate(schedules_data):
        schedule = data['schedule']
        makespan = data['makespan']
        title = data['title']
        arrival_times = data['arrival_times']
        
        ax = axes[plot_idx]
        
        if not schedule or all(len(ops) == 0 for ops in schedule.values()):
            ax.text(0.5, 0.5, 'No valid schedule', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{title} - No Solution")
            continue
        
        # Plot operations for each machine
        for idx, machine in enumerate(MACHINE_LIST):
            machine_ops = schedule.get(machine, [])
            machine_ops.sort(key=lambda x: x[1])  # Sort by start time
            
            for op_data in machine_ops:
                if len(op_data) >= 3:
                    job_op, start_time, end_time = op_data[:3]
                    duration = end_time - start_time
                    
                    # Extract job number for coloring
                    job_num = 0
                    if 'J' in job_op:
                        try:
                            job_num = int(job_op.split('J')[1].split('-')[0])
                        except (ValueError, IndexError):
                            job_num = 0
                    
                    color = colors[job_num % len(colors)]
                    
                    ax.barh(idx, duration, left=start_time, height=0.6, 
                           color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                    
                    # Add operation label
                    if duration > 1:  # Only add text if bar is wide enough
                        ax.text(start_time + duration/2, idx, job_op, 
                               ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Add red arrows for job arrivals (only for dynamic jobs that arrive > 0)
        if arrival_times:
            arrow_y_position = len(MACHINE_LIST) + 0.3  # Position above all machines
            for job_id, arrival_time in arrival_times.items():
                if arrival_time > 0 and arrival_time < 200:  # Only show arrows for jobs that don't start at t=0 and arrive within time horizon
                    # Draw vertical line for arrival
                    ax.axvline(x=arrival_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
                    
                    # Add arrow and label
                    ax.annotate(f'Job {job_id} arrives', 
                               xy=(arrival_time, arrow_y_position), 
                               xytext=(arrival_time, arrow_y_position + 0.5),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2),
                               ha='center', va='bottom', color='red', fontweight='bold', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))
        
        # Formatting
        ax.set_yticks(range(len(MACHINE_LIST)))
        ax.set_yticklabels(MACHINE_LIST)
        ax.set_xlabel("Time" if plot_idx == 2 else "")
        ax.set_ylabel("Machines")
        ax.set_title(f"{title} (Makespan: {makespan:.2f})", fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set consistent x-axis limits with space for arrows
        if schedule and any(len(ops) > 0 for ops in schedule.values()):
            max_time = max([max([op[2] for op in ops]) for ops in schedule.values() if ops])
            ax.set_xlim(0, max_time * 1.05)
        else:
            ax.set_xlim(0, 100)  # Default range if no schedule
        ax.set_ylim(-0.5, len(MACHINE_LIST) + 2.0)  # Extra space for arrival arrows and labels
    
    # Add legend
    legend_elements = []
    for i in range(len(ENHANCED_JOBS_DATA)):
        color = colors[i % len(colors)]
        initial_or_poisson = ' (Initial)' if i < 3 else ' (Poisson)'
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                          alpha=0.8, label=f'Job {i}{initial_or_poisson}'))
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(ENHANCED_JOBS_DATA), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig('dynamic_vs_static_gantt_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED!")
    print("Generated files:")
    print("- dynamic_vs_static_gantt_comparison.png: Three-method Gantt chart comparison")
    print("=" * 80)

if __name__ == "__main__":
    main()
