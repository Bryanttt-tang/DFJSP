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

# --- Enhanced Job Data for Dynamic Environment (from test3_backup.py) ---
ENHANCED_JOBS_DATA = collections.OrderedDict({
    # Initial jobs (available at start)
    0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}, {'proc_times': {'M0': 2}}],
    1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M1': 3, 'M2': 4}}],
    2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M0': 3, 'M1': 4}}, {'proc_times': {'M2': 7}}],
    
    # Dynamic jobs (arrive according to Poisson process) 
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
                 max_time_horizon=200, reward_mode="makespan increment", seed=None):
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


def train_static_agent(jobs_data, machine_list, total_timesteps=200000, reward_mode="makespan_increment"):
    """Train a static RL agent on FJSP where all jobs are available at t=0."""
    print(f"\n--- Training Static FJSP Agent ---")
    print(f"All jobs available at t=0, Timesteps: {total_timesteps}")
    print(f"Reward mode: {reward_mode}")
    
    def make_static_env():
        env = StaticFJSPEnv(jobs_data, machine_list, reward_mode=reward_mode)
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_static_env])
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"Training static agent for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    return model


def evaluate_static_agent_on_dynamic(model, jobs_data, machine_list, arrival_times, debug=False):
    """
    Evaluate a static-trained agent on dynamic scenarios by creating a wrapper
    that maps dynamic observations to static observations.
    """
    print(f"\n--- Evaluating Static Agent on Dynamic Scenario ---")
    print(f"Using arrival times: {arrival_times}")
    
    # Create a static environment to get the observation space the model expects
    static_env = StaticFJSPEnv(jobs_data, machine_list, reward_mode="makespan_increment")
    static_obs_size = static_env.observation_space.shape[0]
    
    # Create dynamic environment for actual simulation
    test_env = PoissonDynamicFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=[k for k, v in arrival_times.items() if v == 0],
        arrival_rate=0.1,
        reward_mode="makespan_increment",
        seed=42
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
        # Extract the components that match static observation
        norm_factor = max(test_env.current_time, 1.0)
        static_obs = []
        
        # Machine availability (first 3 elements)
        static_obs.extend(dynamic_obs[:len(machine_list)])
        
        # Operation completion status (jobs * max_ops_per_job elements)
        num_jobs = len(jobs_data)
        max_ops = max(len(ops) for ops in jobs_data.values())
        start_idx = len(machine_list)
        end_idx = start_idx + num_jobs * max_ops
        static_obs.extend(dynamic_obs[start_idx:end_idx])
        
        # Job progress ratios (next num_jobs elements)  
        start_idx = end_idx
        end_idx = start_idx + num_jobs
        static_obs.extend(dynamic_obs[start_idx:end_idx])
        
        # Skip job arrival status in dynamic obs (not in static)
        
        # Machine workloads - find them in dynamic obs
        # They should be after job arrivals
        machine_workloads_start = len(machine_list) + num_jobs * max_ops + num_jobs + num_jobs
        static_obs.extend(dynamic_obs[machine_workloads_start:machine_workloads_start + len(machine_list)])
        
        # Current time (1 element)
        time_idx = machine_workloads_start + len(machine_list)
        static_obs.append(dynamic_obs[time_idx])
        
        # Number of completed jobs (1 element) - use from dynamic obs
        completed_jobs_idx = time_idx + 2  # Skip arrived jobs count
        static_obs.append(dynamic_obs[completed_jobs_idx])
        
        # Pad or truncate to match static observation space
        if len(static_obs) < static_obs_size:
            static_obs.extend([0.0] * (static_obs_size - len(static_obs)))
        elif len(static_obs) > static_obs_size:
            static_obs = static_obs[:static_obs_size]
        
        return np.array(static_obs, dtype=np.float32)
    
    episode_reward = 0.0
    step_count = 0
    invalid_actions = 0
    
    if debug:
        print(f"Static obs size: {static_obs_size}")
        print(f"Dynamic obs size: {len(obs)}")
        print(f"Initial arrived jobs: {test_env.arrived_jobs}")
    
    while True:
        action_masks = test_env.action_masks()
        
        if not any(action_masks):
            if debug:
                print(f"  Step {step_count}: No valid actions available")
            break
        
        # Map dynamic observation to static observation space
        static_obs = map_dynamic_to_static_obs(obs)
        
        action, _ = model.predict(static_obs, action_masks=action_masks, deterministic=True)
        
        if debug and step_count < 10:
            valid_actions = sum(action_masks)
            print(f"  Step {step_count}: {valid_actions} valid actions, chose action {action}, time={test_env.current_time:.2f}")
        
        obs, reward, done, truncated, info = test_env.step(action)
        episode_reward += reward
        step_count += 1
        
        if info.get("error") == "Invalid action":
            invalid_actions += 1
        
        if done or truncated or step_count > 2000:
            break
    
    makespan = test_env.current_time
    schedule = test_env.schedule
    
    print(f"Static agent on dynamic scenario makespan: {makespan:.2f}, invalid actions: {invalid_actions}")
    return makespan, schedule
    print(f"Invalid actions: {invalid_actions}, Steps: {step_count}")
    
    return makespan, schedule


def evaluate_static_agent_on_static(model, jobs_data, machine_list, num_episodes=10, debug=False):
    """Evaluate static agent on static scenarios (baseline performance)."""
    print(f"\n--- Evaluating Static Agent on Static Scenarios ---")
    
    results = []
    
    for episode in range(num_episodes):
        test_env = StaticFJSPEnv(
            jobs_data, machine_list,
            reward_mode="makespan_increment",
            seed=episode + 200
        )
        
        obs, _ = test_env.reset()
        episode_reward = 0.0
        step_count = 0
        invalid_actions = 0
        
        while True:
            action_masks = test_env.action_masks()
            
            if not any(action_masks):
                break
                
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            episode_reward += reward
            step_count += 1
            
            if info.get("error") == "Invalid action":
                invalid_actions += 1
            
            if done or truncated or step_count > 2000:
                break
        
        makespan = test_env.current_time
        
        results.append({
            'episode': episode,
            'makespan': makespan,
            'total_reward': episode_reward,
            'steps': step_count,
            'invalid_actions': invalid_actions,
            'schedule': {m: list(ops) for m, ops in test_env.schedule.items()}
        })
        
        print(f"Episode {episode+1}: Makespan={makespan:.2f}, Steps={step_count}, Invalid={invalid_actions}")
    
    # Calculate statistics
    valid_results = [r for r in results if r['makespan'] > 0 and r['makespan'] != float('inf')]
    
    if valid_results:
        makespans = [r['makespan'] for r in valid_results]
        avg_makespan = np.mean(makespans)
        std_makespan = np.std(makespans)
        min_makespan = np.min(makespans)
        
        print(f"Static agent on static scenarios:")
        print(f"Average makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
        print(f"Best makespan: {min_makespan:.2f}")
        
        best_episode = min(valid_results, key=lambda x: x['makespan'])
        return best_episode['makespan'], best_episode['schedule']
    else:
        print("No valid results obtained!")
        return float('inf'), {m: [] for m in machine_list}


def train_poisson_agent(jobs_data, machine_list, initial_jobs=5, arrival_rate=0.01, 
                       total_timesteps=100000, reward_mode="makespan_increment"):
    """Train an RL agent on the Poisson Dynamic FJSP environment with improved strategy."""
    print(f"\n--- Training Poisson Dynamic FJSP Agent ---")
    print(f"Initial jobs: {initial_jobs}, Arrival rate: {arrival_rate}")
    print(f"Reward mode: {reward_mode}, Timesteps: {total_timesteps}")
    
    def make_env():
        env = PoissonDynamicFJSPEnv(
            jobs_data, machine_list, 
            initial_jobs=initial_jobs, 
            arrival_rate=arrival_rate,
            reward_mode=reward_mode
        )
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_env])
    
    # Improved hyperparameters for better learning
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=5e-4,        # Slightly higher learning rate
        n_steps=4096,              # More steps for better experience collection
        batch_size=256,            # Larger batch size for stable updates
        n_epochs=10,               # Fewer epochs to prevent overfitting
        gamma=0.995,               # Higher discount factor for long-term planning
        gae_lambda=0.98,           # Higher GAE lambda for better advantage estimation
        clip_range=0.1,            # Smaller clip range for more conservative updates
        ent_coef=0.001,            # Lower entropy for more exploitation
        vf_coef=0.5,               # Value function coefficient
        max_grad_norm=0.5,         # Gradient clipping
        policy_kwargs=dict(
            net_arch=[512, 512, 256, 128],  # Deeper network for complex scheduling
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    return model


def evaluate_poisson_agent(model, jobs_data, machine_list, initial_jobs=5, 
                          arrival_rate=0.1, num_episodes=20, reward_mode="makespan_increment", debug=False):
    """Evaluate the trained agent on multiple episodes with detailed analysis."""
    print(f"\n--- Evaluating Poisson Dynamic FJSP Agent ---")
    print(f"Episodes: {num_episodes}, Initial jobs: {initial_jobs}, Arrival rate: {arrival_rate}")
    
    results = []
    
    for episode in range(num_episodes):
        test_env = PoissonDynamicFJSPEnv(
            jobs_data, machine_list,
            initial_jobs=initial_jobs,
            arrival_rate=arrival_rate,
            reward_mode=reward_mode,
            seed=episode + 100  # Different seed for each episode
        )
        
        obs, _ = test_env.reset()
        episode_reward = 0.0
        step_count = 0
        invalid_actions = 0
        
        if debug and episode == 0:
            print(f"\nDEBUG Episode {episode+1}:")
            print(f"Initial jobs: {test_env.initial_job_ids}")
            print(f"Dynamic jobs: {test_env.dynamic_job_ids}")
            print(f"Arrival times: {test_env.arrival_times}")
        
        while True:
            action_masks = test_env.action_masks()
            
            # Check if any valid actions are available
            if not any(action_masks):
                if debug and episode == 0:
                    print(f"  Step {step_count}: No valid actions available")
                break
                
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            if debug and episode == 0 and step_count < 10:
                valid_actions = sum(action_masks)
                print(f"  Step {step_count}: {valid_actions} valid actions, chose action {action}")
            
            obs, reward, done, truncated, info = test_env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Track invalid actions
            if info.get("error") == "Invalid action":
                invalid_actions += 1
            
            if done or truncated or step_count > 2000:  # Increased safety limit
                break
        
        makespan = info.get("makespan", 0.0)
        total_jobs = info.get("total_arrived_jobs", 0)
        
        results.append({
            'episode': episode,
            'makespan': makespan,
            'total_reward': episode_reward,
            'steps': step_count,
            'invalid_actions': invalid_actions,
            'arrived_jobs': total_jobs,
            'schedule': {m: list(ops) for m, ops in test_env.schedule.items()},
            'arrival_times': test_env.arrival_times.copy()
        })
        
        print(f"Episode {episode+1}: Makespan={makespan:.2f}, "
              f"Jobs={total_jobs}, Steps={step_count}, "
              f"Invalid={invalid_actions}, Reward={episode_reward:.1f}")
    
    # Calculate statistics
    valid_results = [r for r in results if r['makespan'] > 0 and r['makespan'] != float('inf')]
    
    if valid_results:
        makespans = [r['makespan'] for r in valid_results]
        avg_makespan = np.mean(makespans)
        std_makespan = np.std(makespans)
        min_makespan = np.min(makespans)
        max_makespan = np.max(makespans)
        
        # Performance analysis
        avg_invalid = np.mean([r['invalid_actions'] for r in valid_results])
        avg_steps = np.mean([r['steps'] for r in valid_results])
        
        print(f"\nEvaluation Results ({len(valid_results)}/{num_episodes} valid episodes):")
        print(f"Average makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
        print(f"Min makespan: {min_makespan:.2f}")
        print(f"Max makespan: {max_makespan:.2f}")
        print(f"Average invalid actions: {avg_invalid:.1f}")
        print(f"Average steps per episode: {avg_steps:.1f}")
        
        # Return best result
        best_episode = min(valid_results, key=lambda x: x['makespan'])
        return best_episode['makespan'], best_episode['schedule']
    else:
        print("No valid results obtained!")
        return float('inf'), {m: [] for m in machine_list}


def heuristic_spt_poisson(jobs_data, machine_list, initial_jobs=5, arrival_rate=0.1, 
                         max_time=200, seed=42):
    """SPT heuristic for Poisson dynamic environment."""
    print(f"\n--- SPT Heuristic for Poisson Dynamic FJSP ---")
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Handle initial_jobs as either integer or list
    if isinstance(initial_jobs, list):
        initial_job_ids = initial_jobs
        num_initial = len(initial_jobs)
        remaining_job_ids = [j for j in jobs_data.keys() if j not in initial_jobs]
    else:
        initial_job_ids = list(range(initial_jobs))
        num_initial = initial_jobs
        remaining_job_ids = list(range(initial_jobs, len(jobs_data)))
    
    # Generate arrival times
    arrival_times = {}
    
    # Initial jobs arrive at t=0
    for job_id in initial_job_ids:
        arrival_times[job_id] = 0.0
    
    # Generate Poisson arrivals for remaining jobs
    current_time = 0.0
    for job_id in remaining_job_ids:
        inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
        current_time += inter_arrival_time
        if current_time <= max_time:
            arrival_times[job_id] = current_time
        else:
            break
    
    print(f"Generated arrivals: {arrival_times}")
    
    # Run SPT with dynamic arrivals
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    schedule = {m: [] for m in machine_list}
    
    arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    operations_scheduled = 0
    sim_time = 0.0
    
    while operations_scheduled < sum(len(jobs_data[job_id]) for job_id in arrival_times):
        # Update arrivals
        current_min_time = min(machine_next_free.values())
        newly_arrived = {job_id for job_id, arr_time in arrival_times.items() 
                        if current_min_time < arr_time <= sim_time and job_id not in arrived_jobs}
        arrived_jobs.update(newly_arrived)
        
        # Find available operations
        candidates = []
        for job_id in arrived_jobs:
            op_idx = next_operation_for_job[job_id]
            if op_idx < len(jobs_data[job_id]):
                op_data = jobs_data[job_id][op_idx]
                job_ready_time = (operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                else arrival_times[job_id])
                
                for machine, proc_time in op_data['proc_times'].items():
                    earliest_start = max(machine_next_free[machine], job_ready_time)
                    candidates.append((proc_time, earliest_start, job_id, op_idx, machine))
        
        if not candidates:
            # Advance time to next arrival
            next_arrivals = [arr for arr in arrival_times.values() if arr > sim_time]
            if not next_arrivals:
                break
            sim_time = min(next_arrivals)
            continue
        
        # Select shortest processing time
        selected = min(candidates, key=lambda x: x[0])
        proc_time, start_time, job_id, op_idx, machine = selected
        end_time = start_time + proc_time
        
        # Update state
        machine_next_free[machine] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled += 1
        sim_time = max(sim_time, end_time)
        
        schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"SPT Heuristic makespan: {makespan:.2f}")
    return makespan, schedule, arrival_times


def heuristic_spt_poisson_fixed(jobs_data, machine_list, arrival_times):
    """SPT heuristic using pre-generated arrival times."""
    print(f"\n--- SPT Heuristic with Fixed Arrival Times ---")
    print(f"Using arrival times: {arrival_times}")
    
    # Run SPT with given arrival times
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    schedule = {m: [] for m in machine_list}
    
    operations_scheduled = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    sim_time = 0.0
    
    arrived_jobs = {job_id for job_id, arrival in arrival_times.items() if arrival <= 0}
    
    while operations_scheduled < total_operations:
        # Find available operations
        candidates = []
        for job_id in arrived_jobs:
            op_idx = next_operation_for_job[job_id]
            if op_idx < len(jobs_data[job_id]):
                op_data = jobs_data[job_id][op_idx]
                job_ready_time = (operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                else arrival_times[job_id])
                
                for machine, proc_time in op_data['proc_times'].items():
                    earliest_start = max(machine_next_free[machine], job_ready_time)
                    candidates.append((proc_time, earliest_start, job_id, op_idx, machine))
        
        if not candidates:
            # Advance time to next arrival
            next_arrivals = [arr for arr in arrival_times.values() if arr > sim_time]
            if not next_arrivals:
                break
            sim_time = min(next_arrivals)
            arrived_jobs.update({j_id for j_id, arrival in arrival_times.items() 
                               if arrival <= sim_time})
            continue
        
        # Select shortest processing time
        candidates.sort(key=lambda x: (x[0], x[1]))
        proc_time, start_time, job_id, op_idx, machine = candidates[0]
        end_time = start_time + proc_time
        
        # Update state
        machine_next_free[machine] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled += 1
        sim_time = max(sim_time, end_time)
        
        schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        
        # Check for new arrivals
        arrived_jobs.update({j_id for j_id, arrival in arrival_times.items() 
                           if arrival <= sim_time})
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"SPT Heuristic makespan: {makespan:.2f}")
    return makespan, schedule


def generate_test_scenarios(num_scenarios=10, initial_jobs=[0, 1, 2], arrival_rate=0.05, base_seed=1000):
    """Generate multiple test scenarios with different Poisson arrivals for fair comparison."""
    test_scenarios = []
    
    for scenario_id in range(num_scenarios):
        # Use different seed for each scenario
        np.random.seed(base_seed + scenario_id)
        random.seed(base_seed + scenario_id)
        
        # Generate arrival times for this scenario
        arrival_times = {}
        
        # Initial jobs arrive at t=0
        for job_id in initial_jobs:
            arrival_times[job_id] = 0.0
        
        # Generate Poisson arrivals for remaining jobs
        remaining_jobs = [j for j in ENHANCED_JOBS_DATA.keys() if j not in initial_jobs]
        current_time = 0.0
        for job_id in remaining_jobs:
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival_time
            arrival_times[job_id] = current_time
        
        test_scenarios.append({
            'scenario_id': scenario_id,
            'arrival_times': arrival_times,
            'seed': base_seed + scenario_id
        })
    
    return test_scenarios


def evaluate_on_test_scenarios(model, test_scenarios, method_name="Model", is_static_model=False):
    """Evaluate a model on multiple predefined test scenarios."""
    print(f"\n--- Evaluating {method_name} on {len(test_scenarios)} Test Scenarios ---")
    
    results = []
    
    for scenario in test_scenarios:
        scenario_id = scenario['scenario_id']
        arrival_times = scenario['arrival_times']
        seed = scenario['seed']
        
        if is_static_model:
            # For static models, use the special evaluation function
            makespan, schedule = evaluate_static_agent_on_dynamic(
                model, ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_times, debug=False
            )
            
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
            # Create environment with fixed arrival times
            test_env = PoissonDynamicFJSPEnv(
                ENHANCED_JOBS_DATA, MACHINE_LIST,
                initial_jobs=[k for k, v in arrival_times.items() if v == 0],
                arrival_rate=0.1,  # This will be overridden
                reward_mode="makespan_increment",
                seed=seed
            )
            
            # Override the generated arrival times with our fixed ones
            test_env.arrival_times = arrival_times.copy()
            test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
            test_env.next_arrival_events.sort(key=lambda x: x[0])
            
            obs, _ = test_env.reset()
            
            # Re-override after reset (since reset might regenerate)
            test_env.arrival_times = arrival_times.copy()
            test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
            test_env.next_arrival_events.sort(key=lambda x: x[0])
            test_env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
            
            episode_reward = 0.0
            step_count = 0
            
            while True:
                action_masks = test_env.action_masks()
                
                if not any(action_masks):
                    break
                    
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                episode_reward += reward
                step_count += 1
                
                if done or truncated or step_count > 2000:
                    break
            
            makespan = test_env.current_time
            schedule = test_env.schedule
            
            results.append({
                'scenario_id': scenario_id,
                'makespan': makespan,
                'schedule': schedule,
                'arrival_times': arrival_times,
                'steps': step_count,
                'reward': episode_reward
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


def evaluate_spt_on_test_scenarios(test_scenarios):
    """Evaluate SPT heuristic on predefined test scenarios."""
    print(f"\n--- Evaluating SPT Heuristic on {len(test_scenarios)} Test Scenarios ---")
    
    results = []
    
    for scenario in test_scenarios:
        scenario_id = scenario['scenario_id']
        arrival_times = scenario['arrival_times']
        
        makespan, schedule = heuristic_spt_poisson_fixed(
            ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_times
        )
        
        results.append({
            'scenario_id': scenario_id,
            'makespan': makespan,
            'schedule': schedule,
            'arrival_times': arrival_times
        })
        
        print(f"  Scenario {scenario_id+1}: Makespan = {makespan:.2f}")
    
    # Calculate statistics
    makespans = [r['makespan'] for r in results]
    avg_makespan = np.mean(makespans)
    std_makespan = np.std(makespans)
    min_makespan = np.min(makespans)
    max_makespan = np.max(makespans)
    
    print(f"Results for SPT Heuristic:")
    print(f"  Average Makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
    print(f"  Best Makespan: {min_makespan:.2f}")
    print(f"  Worst Makespan: {max_makespan:.2f}")
    
    # Return best result for visualization
    best_result = min(results, key=lambda x: x['makespan'])
    return best_result['makespan'], best_result['schedule'], best_result['arrival_times']


def evaluate_poisson_agent_fixed(model, jobs_data, machine_list, arrival_times, debug=False):
    """Evaluate agent using pre-generated arrival times."""
    print(f"\n--- Evaluating with Fixed Arrival Times ---")
    print(f"Using arrival times: {arrival_times}")
    
    # Create environment with fixed arrival times by manually setting them
    test_env = PoissonDynamicFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=[k for k, v in arrival_times.items() if v == 0],
        arrival_rate=0.1,  # This will be overridden
        reward_mode="makespan_increment",
        seed=42
    )
    
    # Override the generated arrival times with our fixed ones
    test_env.arrival_times = arrival_times.copy()
    test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
    test_env.next_arrival_events.sort(key=lambda x: x[0])
    
    obs, _ = test_env.reset()
    
    # Re-override after reset (since reset might regenerate)
    test_env.arrival_times = arrival_times.copy()
    test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
    test_env.next_arrival_events.sort(key=lambda x: x[0])
    test_env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    
    episode_reward = 0.0
    step_count = 0
    
    if debug:
        print(f"\nDEBUG Fixed Evaluation:")
        print(f"Initial jobs: {test_env.initial_job_ids}")
        print(f"Dynamic jobs: {test_env.dynamic_job_ids}")
        print(f"Fixed arrival times: {test_env.arrival_times}")
        print(f"Initial arrived jobs: {test_env.arrived_jobs}")
    
    while True:
        action_masks = test_env.action_masks()
        
        if not any(action_masks):
            if debug:
                print(f"  Step {step_count}: No valid actions available")
            break
            
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        if debug and step_count < 20:
            valid_actions = sum(action_masks)
            print(f"  Step {step_count}: {valid_actions} valid actions, chose action {action}, time={test_env.current_time:.2f}")
        
        obs, reward, done, truncated, info = test_env.step(action)
        episode_reward += reward
        step_count += 1
        
        if done or truncated or step_count > 2000:
            break
    
    makespan = test_env.current_time
    schedule = test_env.schedule
    
    print(f"Fixed evaluation makespan: {makespan:.2f}")
    return makespan, schedule


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


def plot_three_method_comparison(schedules_data, arrival_times=None, save_path=None):
    """
    Create a 3-subplot figure comparing Dynamic RL, SPT Heuristic, and Static RL
    
    Args:
        schedules_data: Dict with keys 'dynamic_rl', 'spt', 'static_rl'
                       Each containing {'schedule', 'makespan', 'title'}
        arrival_times: Dict of job_id -> arrival_time for showing arrival arrows
    """
    plt.rcParams.update({'font.size': 12})
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Scheduling Comparison: Dynamic RL vs SPT Heuristic vs Static RL\n' + 
                 'Dynamic: Jobs 0-2 at t=0, Jobs 3-6 Poisson | Static: All jobs at t=0', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab20.colors
    methods = ['dynamic_rl', 'spt', 'static_rl']
    
    for plot_idx, method in enumerate(methods):
        if method not in schedules_data:
            continue
            
        data = schedules_data[method]
        schedule = data['schedule']
        makespan = data['makespan']
        title = data['title']
        
        # Get arrival times (same for all methods now)
        # arrival_times parameter is used directly for all methods
        
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
                if len(op_data) == 3:
                    job_id_str, start, end = op_data
                    try:
                        # Extract job number from string like "J0-O1"
                        j = int(job_id_str.split('-')[0][1:])
                    except (ValueError, IndexError):
                        j = hash(job_id_str) % len(colors)
                    
                    # Use different shading for initial vs dynamic jobs
                    alpha = 0.8 if j < 3 else 0.6  # Initial jobs more opaque
                    edge_style = 'solid' if j < 3 else 'dashed'
                    
                    ax.broken_barh(
                        [(start, end - start)],
                        (idx * 10, 8),
                        facecolors=colors[j % len(colors)],
                        edgecolor='black',
                        alpha=alpha,
                        linewidth=2 if j < 3 else 1
                    )
                    
                    # Add operation label
                    label = job_id_str
                    ax.text(start + (end - start) / 2, idx * 10 + 4,
                           label, color='white', fontsize=9,
                           ha='center', va='center', weight='bold')
        
        # Add red arrows for job arrivals (only for dynamic jobs that arrive > 0)
        if arrival_times:
            for job_id, arrival_time in arrival_times.items():
                if arrival_time > 0:  # Only show arrows for jobs that don't start at t=0
                    # Draw red arrow at arrival time
                    arrow_y = len(MACHINE_LIST) * 10 + 5  # Above all machines
                    ax.annotate(f'J{job_id}↓', xy=(arrival_time, arrow_y), xytext=(arrival_time, arrow_y + 15),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2),
                               ha='center', va='bottom', color='red', fontweight='bold',
                               fontsize=10)
        
        # Formatting
        ax.set_yticks([i * 10 + 4 for i in range(len(MACHINE_LIST))])
        ax.set_yticklabels(MACHINE_LIST)
        ax.set_xlabel("Time" if plot_idx == 2 else "")
        ax.set_ylabel("Machines")
        ax.set_title(f"{title} (Makespan: {makespan:.2f})", fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set consistent x-axis limits with space for arrows
        max_time = max([max([op[2] for op in ops]) for ops in schedule.values() if ops])
        ax.set_xlim(0, max_time * 1.05)
        ax.set_ylim(-5, len(MACHINE_LIST) * 10 + 25)  # Extra space for arrival arrows
    
    # Add legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i % len(colors)], 
                          alpha=0.8, label=f'Job {i}' + (' (Initial)' if i < 3 else ' (Poisson)'))
        for i in range(len(ENHANCED_JOBS_DATA))
    ]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(ENHANCED_JOBS_DATA), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Three-method comparison saved to {save_path}")
    
    plt.show()


def comprehensive_dynamic_static_comparison():
    """
    Comprehensive comparison between Dynamic RL, Static RL, and SPT heuristic.
    
    This function addresses the key research question:
    - Dynamic RL should outperform Static RL when jobs arrive dynamically according to Poisson process
    - Both RL methods should be compared against classical SPT heuristic
    - Fair comparison using identical test scenarios for all methods
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE DYNAMIC vs STATIC RL COMPARISON")
    print("Research Question: Does Dynamic RL outperform Static RL on Poisson arrivals?")
    print("="*80)
    
    # Configuration
    initial_jobs = [0, 1, 2]      # Jobs available at start
    arrival_rate = 0.01           # Moderate Poisson arrival rate
    num_test_scenarios = 5        # Fair comparison scenarios
    training_timesteps = 150000   # Training budget for each method
    
    print(f"Experimental Setup:")
    print(f"- Initial jobs (t=0): {initial_jobs}")
    print(f"- Dynamic jobs (Poisson): {[j for j in ENHANCED_JOBS_DATA.keys() if j not in initial_jobs]}")
    print(f"- Arrival rate: {arrival_rate} jobs/time unit")
    print(f"- Test scenarios: {num_test_scenarios}")
    print(f"- Training timesteps per method: {training_timesteps}")
    
    # STEP 1: Generate Test Scenarios for Fair Comparison
    print(f"\n1. GENERATING {num_test_scenarios} TEST SCENARIOS...")
    print("-" * 50)
    test_scenarios = generate_test_scenarios(
        num_scenarios=num_test_scenarios,
        initial_jobs=initial_jobs,
        arrival_rate=arrival_rate,
        base_seed=3000  # Fixed seed for reproducibility
    )
    
    print("Test scenarios generated:")
    for i, scenario in enumerate(test_scenarios):
        arr_times = scenario['arrival_times']
        dynamic_arrivals = {k: v for k, v in arr_times.items() if v > 0}
        print(f"  Scenario {i+1}: Dynamic arrivals = {dynamic_arrivals}")
    
    # STEP 2: Train Dynamic RL Agent
    print(f"\n2. TRAINING DYNAMIC RL AGENT...")
    print("-" * 50)
    print("Training with curriculum learning on Poisson arrival scenarios")
    
    # Progressive training phases
    def make_dynamic_env(complexity_factor=1.0):
        rate = arrival_rate * complexity_factor
        env = PoissonDynamicFJSPEnv(
            ENHANCED_JOBS_DATA, MACHINE_LIST,
            initial_jobs=initial_jobs,
            arrival_rate=rate,
            reward_mode="makespan_increment"
        )
        return ActionMasker(env, mask_fn)
    
    # Phase 1: Easier scenarios (lower arrival rate)
    print("  Phase 1/3: Training on easier scenarios (50% arrival rate)...")
    dynamic_env = DummyVecEnv([lambda: make_dynamic_env()])
    dynamic_rl_model = MaskablePPO(
        "MlpPolicy", dynamic_env, verbose=1,
        learning_rate=3e-4, n_steps=512, batch_size=128, n_epochs=15,
        gamma=1, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256, 128], activation_fn=torch.nn.ReLU)
    )
    dynamic_rl_model.learn(total_timesteps=training_timesteps)
    
    # # Phase 2: Medium scenarios
    # print("  Phase 2/3: Training on medium scenarios (75% arrival rate)...")
    # dynamic_env = DummyVecEnv([lambda: make_dynamic_env(0.75)])
    # dynamic_rl_model.set_env(dynamic_env)
    # dynamic_rl_model.learn(total_timesteps=training_timesteps // 3)
    
    # # Phase 3: Full complexity
    # print("  Phase 3/3: Training on full complexity scenarios...")
    # dynamic_env = DummyVecEnv([lambda: make_dynamic_env(1.0)])
    # dynamic_rl_model.set_env(dynamic_env)
    # dynamic_rl_model.learn(total_timesteps=training_timesteps // 3)
    
    # STEP 3: Train Static RL Agent
    print(f"\n3. TRAINING STATIC RL AGENT...")
    print("-" * 50)
    print("Training on static scenarios (all jobs available at t=0)")
    
    static_rl_model = train_static_agent(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        total_timesteps=training_timesteps,
        reward_mode="makespan_increment"
    )
    
    # STEP 4: Evaluate All Methods on Same Test Scenarios
    print(f"\n4. EVALUATION ON IDENTICAL TEST SCENARIOS...")
    print("="*60)
    
    # Storage for results
    all_results = {}
    
    # Evaluate SPT Heuristic
    print("Evaluating SPT Heuristic...")
    spt_makespan, spt_schedule, spt_arrival_times = evaluate_spt_on_test_scenarios(test_scenarios)
    all_results['SPT Heuristic'] = {
        'avg_makespan': spt_makespan,
        'schedule': spt_schedule,
        'arrival_times': spt_arrival_times,
        'method_type': 'heuristic'
    }
    
    # Evaluate Dynamic RL
    print("Evaluating Dynamic RL...")
    dynamic_rl_makespan, dynamic_rl_schedule, dynamic_rl_arrival_times = evaluate_on_test_scenarios(
        dynamic_rl_model, test_scenarios, "Dynamic RL", is_static_model=False
    )
    all_results['Dynamic RL'] = {
        'avg_makespan': dynamic_rl_makespan,
        'schedule': dynamic_rl_schedule,
        'arrival_times': dynamic_rl_arrival_times,
        'method_type': 'dynamic_rl'
    }
    
    # Evaluate Static RL on Dynamic Scenarios
    print("Evaluating Static RL on Dynamic Scenarios...")
    static_rl_makespan, static_rl_schedule, static_rl_arrival_times = evaluate_on_test_scenarios(
        static_rl_model, test_scenarios, "Static RL", is_static_model=True
    )
    all_results['Static RL'] = {
        'avg_makespan': static_rl_makespan,
        'schedule': static_rl_schedule,
        'arrival_times': static_rl_arrival_times,
        'method_type': 'static_rl'
    }
    
    # STEP 5: Baseline Performance (Static RL on Static Problems)
    print(f"\n5. BASELINE: STATIC RL ON STATIC PROBLEMS...")
    print("-" * 50)
    static_baseline_makespan, static_baseline_schedule = evaluate_static_agent_on_static(
        static_rl_model, ENHANCED_JOBS_DATA, MACHINE_LIST, num_episodes=5
    )
    all_results['Static RL (Baseline)'] = {
        'avg_makespan': static_baseline_makespan,
        'schedule': static_baseline_schedule,
        'arrival_times': {j: 0.0 for j in ENHANCED_JOBS_DATA.keys()},  # All jobs at t=0
        'method_type': 'static_baseline'
    }
    
    # STEP 6: Results Analysis
    print(f"\n6. COMPREHENSIVE RESULTS ANALYSIS...")
    print("="*70)
    
    # Sort results by performance
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['avg_makespan'])
    
    print("Performance Ranking (Lower makespan = Better):")
    print("-" * 50)
    for i, (method, data) in enumerate(sorted_results):
        makespan = data['avg_makespan']
        if i == 0:
            print(f"{i+1}. {method:20s}: {makespan:6.2f} (BEST)")
        else:
            gap = ((makespan - sorted_results[0][1]['avg_makespan']) / sorted_results[0][1]['avg_makespan'] * 100)
            print(f"{i+1}. {method:20s}: {makespan:6.2f} (+{gap:5.1f}%)")
    
    # Key comparisons
    print(f"\nKey Comparisons:")
    print("-" * 30)
    
    dynamic_makespan = all_results['Dynamic RL']['avg_makespan']
    static_makespan = all_results['Static RL']['avg_makespan']
    static_baseline = all_results['Static RL (Baseline)']['avg_makespan']
    spt_makespan = all_results['SPT Heuristic']['avg_makespan']
    
    # Dynamic vs Static RL on dynamic scenarios
    if dynamic_makespan < static_makespan:
        improvement = ((static_makespan - dynamic_makespan) / static_makespan * 100)
        print(f"✓ Dynamic RL vs Static RL: {improvement:5.1f}% improvement")
        print(f"  → Dynamic RL successfully adapts to Poisson arrivals")
    else:
        degradation = ((dynamic_makespan - static_makespan) / static_makespan * 100)
        print(f"✗ Dynamic RL vs Static RL: {degradation:5.1f}% worse")
        print(f"  → Dynamic RL failed to learn adaptation")
    
    # Static RL: Dynamic vs Static scenarios
    adaptation_penalty = ((static_makespan - static_baseline) / static_baseline * 100)
    print(f"  Static RL adaptation penalty: {adaptation_penalty:5.1f}%")
    print(f"  → Cost of not being trained for dynamic arrivals")
    
    # Both RL vs SPT
    dynamic_vs_spt = ((dynamic_makespan - spt_makespan) / spt_makespan * 100)
    static_vs_spt = ((static_makespan - spt_makespan) / spt_makespan * 100)
    print(f"  Dynamic RL vs SPT: {dynamic_vs_spt:+5.1f}%")
    print(f"  Static RL vs SPT:  {static_vs_spt:+5.1f}%")
    
    # STEP 7: Visualization
    print(f"\n7. CREATING COMPARISON VISUALIZATION...")
    print("-" * 40)
    
    # Prepare data for visualization
    schedules_data = {
        'dynamic_rl': {
            'schedule': dynamic_rl_schedule,
            'makespan': dynamic_makespan,
            'title': f'Dynamic RL (Makespan: {dynamic_makespan:.2f})'
        },
        'spt': {
            'schedule': spt_schedule,
            'makespan': spt_makespan,
            'title': f'SPT Heuristic (Makespan: {spt_makespan:.2f})'
        },
        'static_rl': {
            'schedule': static_rl_schedule,
            'makespan': static_makespan,
            'title': f'Static RL on Dynamic (Makespan: {static_makespan:.2f})'
        }
    }
    
    # Create comparison plot
    plot_three_method_comparison(
        schedules_data, 
        arrival_times=dynamic_rl_arrival_times,
        save_path="dynamic_vs_static_rl_comparison.png"
    )
    
    # STEP 8: Summary and Conclusions
    print(f"\n8. SUMMARY AND CONCLUSIONS...")
    print("="*50)
    
    best_method = sorted_results[0][0]
    print(f"Best performing method: {best_method}")
    
    if 'Dynamic RL' in [result[0] for result in sorted_results[:2]]:
        print("✓ Dynamic RL is competitive with best methods")
    else:
        print("✗ Dynamic RL needs improvement")
    
    if dynamic_makespan < static_makespan:
        print("✓ Dynamic RL successfully outperforms Static RL on dynamic scenarios")
        print("  Research hypothesis CONFIRMED: Dynamic training helps with Poisson arrivals")
    else:
        print("✗ Dynamic RL failed to outperform Static RL")
        print("  Research hypothesis REJECTED: Need to improve dynamic training")
    
    # Recommendations
    print(f"\nRecommendations:")
    if dynamic_makespan > static_makespan:
        print("- Increase dynamic training complexity")
        print("- Improve reward function for dynamic adaptation")
        print("- Consider different arrival rate during training")
    if min(dynamic_makespan, static_makespan) > spt_makespan:
        print("- Both RL methods lag behind SPT heuristic")
        print("- Consider hybrid RL-heuristic approaches")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON COMPLETED")
    print("="*80)
    
    return all_results


def compare_poisson_methods():
    """Compare RL vs SPT heuristic on Poisson dynamic FJSP."""
    print("\n" + "="*80)
    print("POISSON DYNAMIC FJSP COMPARISON")
    print("="*80)
    
    # Environment parameters
    initial_jobs = 5
    arrival_rate = 0.01  # jobs per time unit
    
    print(f"Environment setup:")
    print(f"- Initial jobs available: {initial_jobs}")
    print(f"- Dynamic jobs: {len(ENHANCED_JOBS_DATA) - initial_jobs}")
    print(f"- Poisson arrival rate: {arrival_rate} jobs/time unit")
    print(f"- Average inter-arrival time: {1.0/arrival_rate:.1f} time units")
    print(f"- Reward mode: makespan_increment (like in test3_backup.py)")
    
    # 1. Train RL agent with makespan increment reward
    print("\n1. Training RL Agent with makespan increment reward...")
    rl_model = train_poisson_agent(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=initial_jobs,
        arrival_rate=arrival_rate,
        total_timesteps=200000,  # More training steps
        reward_mode="makespan_increment"
    )
    
    # 2. Evaluate RL agent
    print("\n2. Evaluating RL Agent...")
    rl_makespan, rl_schedule = evaluate_poisson_agent(
        rl_model, ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=initial_jobs,
        arrival_rate=arrival_rate,
        num_episodes=10,
        reward_mode="makespan_increment"
    )
    
    # 3. SPT Heuristic (multiple runs for fair comparison)
    print("\n3. Running SPT Heuristic (multiple runs)...")
    spt_results = []
    
    for run in range(10):
        makespan, schedule = heuristic_spt_poisson(
            ENHANCED_JOBS_DATA, MACHINE_LIST,
            initial_jobs=initial_jobs,
            arrival_rate=arrival_rate,
            seed=42 + run  # Different seeds for fair comparison
        )
        spt_results.append(makespan)
    
    spt_avg = np.mean(spt_results)
    spt_std = np.std(spt_results)
    spt_min = np.min(spt_results)
    
    print(f"SPT Heuristic Results (10 runs):")
    print(f"Average makespan: {spt_avg:.2f} ± {spt_std:.2f}")
    print(f"Best makespan: {spt_min:.2f}")
    
    # 4. Results comparison
    print("\n4. DETAILED COMPARISON")
    print("="*50)
    print(f"RL Agent (best):          {rl_makespan:.2f}")
    print(f"SPT Heuristic (average):  {spt_avg:.2f}")
    print(f"SPT Heuristic (best):     {spt_min:.2f}")
    
    if rl_makespan != float('inf') and spt_avg != float('inf'):
        if rl_makespan < spt_avg:
            improvement = ((spt_avg - rl_makespan) / spt_avg) * 100
            print(f"RL improvement over SPT average: {improvement:.1f}%")
        else:
            gap = ((rl_makespan - spt_avg) / spt_avg) * 100
            print(f"RL gap from SPT average: {gap:.1f}%")
            
        if rl_makespan < spt_min:
            improvement_best = ((spt_min - rl_makespan) / spt_min) * 100
            print(f"RL improvement over SPT best: {improvement_best:.1f}%")
        else:
            gap_best = ((rl_makespan - spt_min) / spt_min) * 100
            print(f"RL gap from SPT best: {gap_best:.1f}%")
    
    # 5. Analysis and recommendations
    print(f"\n5. ANALYSIS:")
    if rl_makespan < spt_avg * 1.1:  # Within 10% of SPT
        print("✓ RL performance is competitive with SPT heuristic")
    else:
        print("✗ RL performance needs improvement")
        print("  Recommendations:")
        print("  - Increase training timesteps")
        print("  - Tune reward function")
        print("  - Check environment observation space")
        print("  - Verify action masking correctness")
    
    # 6. Use best SPT run for visualization
    best_spt_run = np.argmin(spt_results)
    spt_makespan, spt_schedule = heuristic_spt_poisson(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=initial_jobs,
        arrival_rate=arrival_rate,
        seed=42 + best_spt_run
    )
    
    print("\n6. Plotting Gantt Charts...")
    
    # RL Gantt chart
    if rl_makespan != float('inf'):
        plot_gantt(rl_schedule, MACHINE_LIST, 
                  title=f"RL Agent - Poisson Dynamic FJSP (Makespan: {rl_makespan:.2f})",
                  save_path="rl_poisson_dynamic_gantt.png")
    
    # SPT Gantt chart
    plot_gantt(spt_schedule, MACHINE_LIST,
              title=f"SPT Heuristic - Poisson Dynamic FJSP (Makespan: {spt_makespan:.2f})", 
              save_path="spt_poisson_dynamic_gantt.png")
    
    print("\n" + "="*80)
    print("POISSON DYNAMIC FJSP ANALYSIS COMPLETED")
    print("="*80)
    
    return {
        'rl_makespan': rl_makespan,
        'spt_avg': spt_avg,
        'spt_best': spt_min,
        'rl_schedule': rl_schedule,
        'spt_schedule': spt_schedule
    }


if __name__ == "__main__":
    print("Starting Comprehensive Dynamic vs Static RL Comparison...")
    print("="*80)
    print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
    print("Research Question: Does Dynamic RL outperform Static RL on Poisson arrivals?")
    print("="*80)
    
    # Run the comprehensive comparison
    results = comprehensive_dynamic_static_comparison()
    
    print("\nExecution completed. Check the results above and the generated plots.")
    print("Generated files:")
    print("- dynamic_vs_static_rl_comparison.png: Three-method comparison plot")
    print("="*80)
