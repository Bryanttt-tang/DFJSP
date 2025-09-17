import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import collections
import time
import os
import hashlib
import pickle
import torch
from tqdm import tqdm
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD

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


class StaticFJSPEnv(gym.Env):
    """
    Static FJSP Environment where all jobs are available at time 0.
    Uses the same structure as PerfectKnowledgeFJSPEnv but with all arrival times = 0.
    """
    
    def __init__(self, jobs_data, machine_list, reward_mode="makespan_increment", seed=None):
        """Initialize with all jobs arriving at t=0."""
        super().__init__()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.reward_mode = reward_mode
        
        # Create arrival times dict with all jobs at t=0 (static scenario)
        self.job_arrival_times = {job_id: 0.0 for job_id in self.job_ids}
        
        # Environment parameters
        self.num_jobs = len(self.job_ids)
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        
        # Use SAME action and observation spaces as PerfectKnowledgeFJSPEnv
        self.action_space = spaces.Discrete(
            min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
        )
        
        # SAME observation space as PerfectKnowledgeFJSPEnv
        obs_size = (
            len(self.machines) +                    # Machine availability
            self.num_jobs * self.max_ops_per_job +  # Operation completion
            self.num_jobs +                         # Job progress  
            self.num_jobs +                         # Job arrival status (all arrived at t=0)
            1                                       # Current makespan
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        self._reset_state()
    
    def _reset_state(self):
        """Reset the environment state - same as PerfectKnowledgeFJSPEnv."""
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        self.current_makespan = 0.0
        self.operations_scheduled = 0
        self.episode_step = 0
        self.max_episode_steps = self.total_operations * 2
        
        # Handle job arrivals - all jobs arrive at t=0 for static scenario
        self.arrived_jobs = set(self.job_ids)  # All jobs available immediately
        
    def reset(self, seed=None, options=None):
        """Reset environment - same as PerfectKnowledgeFJSPEnv."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self._reset_state()
        return self._get_observation(), {}

    def _decode_action(self, action):
        """Decode action - same as PerfectKnowledgeFJSPEnv."""
        action = int(action) % self.action_space.n
        num_machines = len(self.machines)
        ops_per_job = self.max_ops_per_job
        
        job_idx = action // (ops_per_job * num_machines)
        op_idx = (action % (ops_per_job * num_machines)) // num_machines
        machine_idx = action % num_machines
        
        job_idx = min(job_idx, self.num_jobs - 1)
        machine_idx = min(machine_idx, len(self.machines) - 1)
        
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        """Check if action is valid - same as PerfectKnowledgeFJSPEnv."""
        if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
            return False
        
        job_id = self.job_ids[job_idx]
        
        # Check if job has arrived (all jobs arrive at t=0 for static)
        if job_id not in self.arrived_jobs:
            return False
            
        # Check operation index validity
        if not (0 <= op_idx < len(self.jobs[job_id])):
            return False
            
        # Check if this is the next operation
        if op_idx != self.next_operation[job_id]:
            return False
            
        # Check machine compatibility
        machine_name = self.machines[machine_idx]
        if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
            return False
            
        return True

    def action_masks(self):
        """Generate action masks - same as PerfectKnowledgeFJSPEnv."""
        mask = np.full(self.action_space.n, False, dtype=bool)
        
        if self.operations_scheduled >= self.total_operations:
            return mask

        valid_action_count = 0
        for job_idx, job_id in enumerate(self.job_ids):
            if job_id not in self.arrived_jobs:
                continue
                
            next_op_idx = self.next_operation[job_id]
            if next_op_idx >= len(self.jobs[job_id]):
                continue
                
            for machine_idx, machine in enumerate(self.machines):
                if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                    action = job_idx * (self.max_ops_per_job * len(self.machines)) + next_op_idx * len(self.machines) + machine_idx
                    if action < self.action_space.n:
                        mask[action] = True
                        valid_action_count += 1
        
        if valid_action_count == 0:
            mask.fill(True)
            
        return mask
    
    def step(self, action):
        """Step function - same as PerfectKnowledgeFJSPEnv."""
        self.episode_step += 1
        
        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        job_idx, op_idx, machine_idx = self._decode_action(action)

        # Use softer invalid action handling like PerfectKnowledgeFJSPEnv
        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        
        # Calculate timing using PerfectKnowledgeFJSPEnv approach
        machine_available_time = self.machine_next_free.get(machine, 0.0)
        job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                         else self.job_arrival_times.get(job_id, 0.0))  # All jobs at t=0
        
        start_time = max(machine_available_time, job_ready_time)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time

        # Update state
        self.machine_next_free[machine] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        
        # Update makespan - no need to check for new arrivals in static environment
        previous_makespan = self.current_makespan
        self.current_makespan = max(self.current_makespan, end_time)

        # Record in schedule
        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        # Check termination
        terminated = self.operations_scheduled >= self.total_operations
        
        # Calculate reward using PerfectKnowledgeFJSPEnv style
        idle_time = max(0, start_time - machine_available_time)
        reward = self._calculate_reward(proc_time, idle_time, terminated, previous_makespan, self.current_makespan)
        
        info = {"makespan": self.current_makespan}
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan=None, current_makespan=None):
        """Reward calculation - same as PerfectKnowledgeFJSPEnv."""
        if self.reward_mode == "makespan_increment":
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment
                
                # Add completion bonus
                if done:
                    reward += 50.0
                    
                return reward
            else:
                return -proc_time
        else:
            # Default reward function
            reward = 10.0 - proc_time * 0.1 - idle_time
            if done:
                reward += 100.0
            return reward

    def _get_observation(self):
        """Generate observation - same structure as PerfectKnowledgeFJSPEnv."""
        norm_factor = max(self.current_makespan, 1.0)
        obs = []
        
        # Machine availability (normalized by current makespan)
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
        
        # Job arrival status (all jobs arrived at t=0 for static environment)
        for job_id in self.job_ids:
            obs.append(1.0)  # All jobs are available (arrived at t=0)
            
        # Current makespan (normalized)
        makespan_norm = float(self.current_makespan) / 100.0  # Assume max makespan around 100
        obs.append(max(0.0, min(1.0, makespan_norm)))
        
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
            print(f"\n=== Static Environment - Makespan: {self.current_makespan:.2f} ===")
            print(f"Completed operations: {self.operations_scheduled}")
            print(f"Machine status:")
            for m in self.machines:
                print(f"  {m}: next free at {self.machine_next_free[m]:.2f}")


class PoissonDynamicFJSPEnv(gym.Env):
    """
    Dynamic FJSP Environment with Poisson-distributed job arrivals.
    FIXED to use the SAME structure as successful StaticFJSPEnv and PerfectKnowledgeFJSPEnv.
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
        
        # USE SAME ACTION SPACE as successful environments (FIXED, not dynamic)
        self.action_space = spaces.Discrete(
            min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
        )
        
        # ENHANCED OBSERVATION SPACE for better dynamic scheduling
        obs_size = (
            len(self.machines) +                    # Machine availability
            1 +                                     # Machine load balance
            self.num_jobs * self.max_ops_per_job +  # Operation completion status
            self.num_jobs +                         # Job progress ratios  
            self.num_jobs +                         # Job arrival status with timing hints
            2 +                                     # Work urgency indicators
            1                                       # Current makespan
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state variables
        self._reset_state()

    def _reset_state(self):
        """Reset all environment state variables - SAME as successful environments."""
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        self.current_makespan = 0.0
        self.operations_scheduled = 0
        self.episode_step = 0
        self.max_episode_steps = self.total_operations * 2
        
        # Job arrival management - simplified
        self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
        self.job_arrival_times = {}
        
        # Generate Poisson arrival times for dynamic jobs
        self._generate_poisson_arrivals()

    def _generate_poisson_arrivals(self):
        """Generate arrival times for dynamic jobs using enhanced Poisson process."""
        # Initialize arrival times
        for job_id in self.initial_job_ids:
            self.job_arrival_times[job_id] = 0.0
        
        # Generate inter-arrival times using exponential distribution
        current_time = 0.0
        for job_id in self.dynamic_job_ids:
            # Add some variability but ensure reasonable spread
            base_inter_arrival = np.random.exponential(1.0 / self.arrival_rate)
            
            # Add slight job-dependent timing to create more realistic patterns
            job_factor = 1.0 + 0.2 * (job_id % 3 - 1)  # ±20% variation based on job ID
            inter_arrival_time = base_inter_arrival * job_factor
            
            current_time += inter_arrival_time
            
            # Use more strategic rounding - favor early arrivals for better learning
            if current_time <= 15:  # Early in schedule
                integer_arrival_time = max(1, int(current_time))  # Round down, min 1
            else:
                integer_arrival_time = round(current_time)  # Normal rounding
            
            if integer_arrival_time <= self.max_time_horizon:
                self.job_arrival_times[job_id] = float(integer_arrival_time)
            else:
                # Instead of infinite, set to a large but finite value
                self.job_arrival_times[job_id] = float(self.max_time_horizon + 1)

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode - SAME structure as successful environments."""
        global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
        
        if seed is not None:
            super().reset(seed=seed, options=options)
            random.seed(seed)
            np.random.seed(seed)
        
        self._reset_state()
        
        # Track arrival times for analysis
        TRAINING_EPISODE_COUNT += 1
        episode_arrivals = []
        for job_id, arr_time in self.job_arrival_times.items():
            if arr_time != float('inf') and arr_time > 0:  # Only dynamic arrivals
                episode_arrivals.append(arr_time)
        
        if episode_arrivals:
            TRAINING_ARRIVAL_TIMES.extend(episode_arrivals)
        
        return self._get_observation(), {}

    def _decode_action(self, action):
        """Decode action - SAME as successful environments."""
        action = int(action) % self.action_space.n
        num_machines = len(self.machines)
        ops_per_job = self.max_ops_per_job
        
        job_idx = action // (ops_per_job * num_machines)
        op_idx = (action % (ops_per_job * num_machines)) // num_machines
        machine_idx = action % num_machines
        
        job_idx = min(job_idx, self.num_jobs - 1)
        machine_idx = min(machine_idx, len(self.machines) - 1)
        
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        """Check if action is valid - SAME as successful environments."""
        if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
            return False
        
        job_id = self.job_ids[job_idx]
        
        # Check if job has arrived
        if job_id not in self.arrived_jobs:
            return False
            
        # Check operation index validity
        if not (0 <= op_idx < len(self.jobs[job_id])):
            return False
            
        # Check if this is the next operation
        if op_idx != self.next_operation[job_id]:
            return False
            
        # Check machine compatibility
        machine_name = self.machines[machine_idx]
        if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
            return False
            
        return True

    def action_masks(self):
        """Generate action masks with enhanced efficiency for better learning."""
        mask = np.full(self.action_space.n, False, dtype=bool)
        
        if self.operations_scheduled >= self.total_operations:
            return mask

        valid_action_count = 0
        priority_actions = []  # Track high-priority actions
        
        for job_idx, job_id in enumerate(self.job_ids):
            if job_id not in self.arrived_jobs:
                continue
                
            next_op_idx = self.next_operation[job_id]
            if next_op_idx >= len(self.jobs[job_id]):
                continue
                
            for machine_idx, machine in enumerate(self.machines):
                if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                    action = job_idx * (self.max_ops_per_job * len(self.machines)) + next_op_idx * len(self.machines) + machine_idx
                    if action < self.action_space.n:
                        mask[action] = True
                        valid_action_count += 1
                        
                        # Track priority for efficient actions (short processing times)
                        proc_time = self.jobs[job_id][next_op_idx]['proc_times'][machine]
                        machine_available = self.machine_next_free.get(machine, 0.0)
                        if proc_time <= 5 or machine_available <= self.current_makespan:
                            priority_actions.append(action)
        
        # If we have priority actions, slightly bias towards them by ensuring they're always valid
        # This helps the agent learn to prefer efficient actions during training
        for action in priority_actions:
            if action < self.action_space.n:
                mask[action] = True
        
        if valid_action_count == 0:
            mask.fill(True)
            
        return mask

    def step(self, action):
        """Step function - SIMPLIFIED to match successful environments."""
        self.episode_step += 1
        
        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        job_idx, op_idx, machine_idx = self._decode_action(action)

        # Use softer invalid action handling like successful environments
        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        
        # Calculate timing using successful environments' approach
        machine_available_time = self.machine_next_free.get(machine, 0.0)
        job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                         else self.job_arrival_times.get(job_id, 0.0))
        
        start_time = max(machine_available_time, job_ready_time)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time

        # Update state
        previous_makespan = self.current_makespan
        self.machine_next_free[machine] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        
        # Update makespan and check for new arrivals (key improvement)
        self.current_makespan = max(self.current_makespan, end_time)
        
        # Check for newly arrived jobs (deterministic based on current makespan)
        newly_arrived = []
        for job_id_check, arrival_time in self.job_arrival_times.items():
            if (job_id_check not in self.arrived_jobs and 
                arrival_time <= self.current_makespan and 
                arrival_time != float('inf')):
                self.arrived_jobs.add(job_id_check)
                newly_arrived.append(job_id_check)

        # Record in schedule
        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        # Check termination
        terminated = self.operations_scheduled >= self.total_operations
        
        # SIMPLIFIED reward calculation matching successful environments
        idle_time = max(0, start_time - machine_available_time)
        reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                      previous_makespan, self.current_makespan, len(newly_arrived))
        
        info = {
            "makespan": self.current_makespan,
            "newly_arrived_jobs": len(newly_arrived),
            "total_arrived_jobs": len(self.arrived_jobs)
        }
        
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan, num_new_arrivals):
        """ENHANCED reward function optimized for dynamic scheduling performance."""
        
        if self.reward_mode == "makespan_increment":
            # Enhanced reward structure for better dynamic performance
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Base negative increment
                
                # ENHANCED: Larger bonus for utilizing newly arrived jobs (encourages anticipation)
                if num_new_arrivals > 0:
                    reward += 15.0 * num_new_arrivals
                
                # ENHANCED: Machine utilization bonus (reduces idle time penalty)
                if idle_time > 0:
                    reward -= idle_time * 0.5  # Penalty for machine idle time
                else:
                    reward += 2.0  # Bonus for no idle time
                
                # ENHANCED: Processing time efficiency bonus
                reward += max(0, 10.0 - proc_time)  # Bonus for shorter operations
                
                # ENHANCED: Load balancing reward
                machine_loads = [self.machine_next_free[m] for m in self.machines]
                if len(machine_loads) > 1:
                    load_balance = 1.0 - (max(machine_loads) - min(machine_loads)) / max(max(machine_loads), 1.0)
                    reward += load_balance * 3.0
                
                # ENHANCED: Completion bonus with makespan quality
                if done:
                    completion_bonus = 100.0
                    # Extra bonus for good final makespan
                    if current_makespan <= 50.0:  # Target makespan
                        completion_bonus += 50.0
                    elif current_makespan <= 60.0:
                        completion_bonus += 25.0
                    reward += completion_bonus
                    
                return reward
            else:
                return -proc_time
        else:
            # Enhanced default reward function
            reward = 20.0 - proc_time * 0.2 - idle_time * 0.8
            
            # New arrival utilization bonus
            if num_new_arrivals > 0:
                reward += 10.0 * num_new_arrivals
            
            if done:
                reward += 150.0 - current_makespan * 0.5  # Makespan-dependent completion bonus
            return reward

    def _get_observation(self):
        """
        ENHANCED observation with better anticipatory information for dynamic scheduling.
        """
        norm_factor = max(self.current_makespan, 100.0)  # Better normalization
        obs = []
        
        # Machine availability (normalized by current makespan)
        for m in self.machines:
            value = float(self.machine_next_free.get(m, 0.0)) / norm_factor
            obs.append(max(0.0, min(1.0, value)))
        
        # ENHANCED: Machine load balance information
        machine_loads = [self.machine_next_free.get(m, 0.0) for m in self.machines]
        if max(machine_loads) > 0:
            load_balance = 1.0 - (max(machine_loads) - min(machine_loads)) / max(machine_loads)
        else:
            load_balance = 1.0
        obs.append(load_balance)
        
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
        
        # ENHANCED: Job arrival information with timing hints
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                obs.append(1.0)  # Job is available
            else:
                # Provide hint about when job might arrive
                arrival_time = self.job_arrival_times.get(job_id, float('inf'))
                if arrival_time == float('inf'):
                    obs.append(0.0)  # Job never arrives
                else:
                    # Normalized arrival time hint (how soon the job arrives)
                    time_until_arrival = max(0, arrival_time - self.current_makespan) / norm_factor
                    obs.append(max(0.0, min(0.9, 0.5 - time_until_arrival)))  # Scale to [0, 0.9]
        
        # ENHANCED: Work urgency indicators
        total_remaining_work = 0
        urgent_jobs = 0
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs and self.next_operation[job_id] < len(self.jobs[job_id]):
                remaining_ops = len(self.jobs[job_id]) - self.next_operation[job_id]
                total_remaining_work += remaining_ops
                if remaining_ops >= 2:  # Jobs with 2+ operations are urgent
                    urgent_jobs += 1
        
        obs.append(total_remaining_work / max(1.0, self.total_operations))  # Normalized remaining work
        obs.append(urgent_jobs / max(1.0, len(self.arrived_jobs)))  # Proportion of urgent jobs
            
        # Current makespan (normalized)
        makespan_norm = float(self.current_makespan) / norm_factor
        obs.append(max(0.0, min(1.0, makespan_norm)))
        
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

    # def debug_step(self, action):
    #     """Debug version of step function to identify issues."""
    #     print(f"\n=== DEBUG STEP ===")
    #     job_idx, op_idx, machine_idx = self._decode_action(action)
    #     print(f"Action: {action} -> Job {job_idx}, Op {op_idx}, Machine {machine_idx}")
        
    #     if job_idx < len(self.job_ids):
    #         job_id = self.job_ids[job_idx]
    #         print(f"Job ID: {job_id}")
    #         print(f"Job arrived: {job_id in self.arrived_jobs}")
    #         print(f"Next operation: {self.next_operation[job_id]}")
    #         print(f"Job operations: {len(self.jobs[job_id])}")
            
    #         if op_idx < len(self.jobs[job_id]):
    #             print(f"Operation data: {self.jobs[job_id][op_idx]}")
    #             machine_name = self.machines[machine_idx]
    #             print(f"Machine {machine_name} can process: {machine_name in self.jobs[job_id][op_idx]['proc_times']}")
        
    #     print(f"Valid action: {self._is_valid_action(job_idx, op_idx, machine_idx)}")
    #     print(f"Arrived jobs: {sorted(self.arrived_jobs)}")
    #     print(f"Current time: {self.current_time}")
    #     print("================")
        
    #     return self.step(action)

    def render(self, mode='human'):
        """Render the current state (optional)."""
        if mode == 'human':
            print(f"\n=== Time: {self.current_makespan:.2f} ===")
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

def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, reward_mode="makespan_increment"):
    """
    Train a perfect knowledge RL agent using the same approach as test3_backup.py.
    
    Key insight: Train on deterministic arrival times (like the test scenario)
    rather than trying to create a complex "perfect knowledge" environment.
    This matches the working approach from test3_backup.py.
    """
    print(f"\n--- Training Perfect Knowledge RL Agent (test3_backup.py approach) ---")
    print(f"Training arrival times: {arrival_times}")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_perfect_env():
        # Use PerfectKnowledgeFJSPEnv for both training and evaluation consistency
        class PoissonDynamicFJSPEnv(gym.Env):
            """
            SIMPLIFIED Dynamic FJSP Environment with Poisson-distributed job arrivals.
            Uses the SAME structure as StaticFJSPEnv and PerfectKnowledgeFJSPEnv for consistency.
            """
            
            metadata = {"render.modes": ["human"]}

            def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05, 
                         max_time_horizon=200, reward_mode="makespan_increment", seed=None):
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
                
                # USE SAME ACTION SPACE as successful environments (FIXED, not dynamic)
                self.action_space = spaces.Discrete(
                    min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
                )
                
                # USE SAME OBSERVATION SPACE as successful environments
                obs_size = (
                    len(self.machines) +                    # Machine availability
                    self.num_jobs * self.max_ops_per_job +  # Operation completion status
                    self.num_jobs +                         # Job progress ratios  
                    self.num_jobs +                         # Job arrival status
                    1                                       # Current makespan
                )
                
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
                )
                
                self._reset_state()

            def _reset_state(self):
                """Reset all environment state variables - SAME as successful environments."""
                self.machine_next_free = {m: 0.0 for m in self.machines}
                self.schedule = {m: [] for m in self.machines}
                self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.next_operation = {job_id: 0 for job_id in self.job_ids}
                
                self.current_makespan = 0.0
                self.operations_scheduled = 0
                self.episode_step = 0
                self.max_episode_steps = self.total_operations * 2
                
                # Job arrival management - simplified
                self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
                self.job_arrival_times = {}
                
                # Generate Poisson arrival times for dynamic jobs
                self._generate_poisson_arrivals()

            def _generate_poisson_arrivals(self):
                """Generate arrival times for dynamic jobs using Poisson process."""
                # Initialize arrival times
                for job_id in self.initial_job_ids:
                    self.job_arrival_times[job_id] = 0.0
                
                # Generate inter-arrival times using exponential distribution
                current_time = 0.0
                for job_id in self.dynamic_job_ids:
                    inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
                    current_time += inter_arrival_time
                    
                    # Round to nearest integer for simplicity
                    integer_arrival_time = round(current_time)
                    
                    if integer_arrival_time <= self.max_time_horizon:
                        self.job_arrival_times[job_id] = float(integer_arrival_time)
                    else:
                        self.job_arrival_times[job_id] = float('inf')  # Won't arrive in this episode

            def reset(self, seed=None, options=None):
                """Reset the environment for a new episode - SAME structure as successful environments."""
                global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
                
                if seed is not None:
                    super().reset(seed=seed, options=options)
                    random.seed(seed)
                    np.random.seed(seed)
                
                self._reset_state()
                
                # Track arrival times for analysis
                TRAINING_EPISODE_COUNT += 1
                episode_arrivals = []
                for job_id, arr_time in self.job_arrival_times.items():
                    if arr_time != float('inf') and arr_time > 0:  # Only dynamic arrivals
                        episode_arrivals.append(arr_time)
                
                if episode_arrivals:
                    TRAINING_ARRIVAL_TIMES.extend(episode_arrivals)
                
                return self._get_observation(), {}

            def _decode_action(self, action):
                """Decode action - SAME as successful environments."""
                action = int(action) % self.action_space.n
                num_machines = len(self.machines)
                ops_per_job = self.max_ops_per_job
                
                job_idx = action // (ops_per_job * num_machines)
                op_idx = (action % (ops_per_job * num_machines)) // num_machines
                machine_idx = action % num_machines
                
                job_idx = min(job_idx, self.num_jobs - 1)
                machine_idx = min(machine_idx, len(self.machines) - 1)
                
                return job_idx, op_idx, machine_idx

            def _is_valid_action(self, job_idx, op_idx, machine_idx):
                """Check if action is valid - SAME as successful environments."""
                if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
                    return False
                
                job_id = self.job_ids[job_idx]
                
                # Check if job has arrived
                if job_id not in self.arrived_jobs:
                    return False
                    
                # Check operation index validity
                if not (0 <= op_idx < len(self.jobs[job_id])):
                    return False
                    
                # Check if this is the next operation
                if op_idx != self.next_operation[job_id]:
                    return False
                    
                # Check machine compatibility
                machine_name = self.machines[machine_idx]
                if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
                    return False
                    
                return True

            def action_masks(self):
                """Generate action masks - SAME as successful environments."""
                mask = np.full(self.action_space.n, False, dtype=bool)
                
                if self.operations_scheduled >= self.total_operations:
                    return mask

                valid_action_count = 0
                for job_idx, job_id in enumerate(self.job_ids):
                    if job_id not in self.arrived_jobs:
                        continue
                        
                    next_op_idx = self.next_operation[job_id]
                    if next_op_idx >= len(self.jobs[job_id]):
                        continue
                        
                    for machine_idx, machine in enumerate(self.machines):
                        if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                            action = job_idx * (self.max_ops_per_job * len(self.machines)) + next_op_idx * len(self.machines) + machine_idx
                            if action < self.action_space.n:
                                mask[action] = True
                                valid_action_count += 1
                
                if valid_action_count == 0:
                    mask.fill(True)
                    
                return mask

            def step(self, action):
                """Step function - SIMPLIFIED to match successful environments."""
                self.episode_step += 1
                
                # Safety check for infinite episodes
                if self.episode_step >= self.max_episode_steps:
                    return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
                
                job_idx, op_idx, machine_idx = self._decode_action(action)

                # Use softer invalid action handling like successful environments
                if not self._is_valid_action(job_idx, op_idx, machine_idx):
                    return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

                job_id = self.job_ids[job_idx]
                machine = self.machines[machine_idx]
                
                # Calculate timing using successful environments' approach
                machine_available_time = self.machine_next_free.get(machine, 0.0)
                job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                 else self.job_arrival_times.get(job_id, 0.0))
                
                start_time = max(machine_available_time, job_ready_time)
                proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
                end_time = start_time + proc_time

                # Update state
                previous_makespan = self.current_makespan
                self.machine_next_free[machine] = end_time
                self.operation_end_times[job_id][op_idx] = end_time
                self.completed_ops[job_id][op_idx] = True
                self.next_operation[job_id] += 1
                self.operations_scheduled += 1
                
                # Update makespan and check for new arrivals (key improvement)
                self.current_makespan = max(self.current_makespan, end_time)
                
                # Check for newly arrived jobs (deterministic based on current makespan)
                newly_arrived = []
                for job_id_check, arrival_time in self.job_arrival_times.items():
                    if (job_id_check not in self.arrived_jobs and 
                        arrival_time <= self.current_makespan and 
                        arrival_time != float('inf')):
                        self.arrived_jobs.add(job_id_check)
                        newly_arrived.append(job_id_check)

                # Record in schedule
                self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

                # Check termination
                terminated = self.operations_scheduled >= self.total_operations
                
                # SIMPLIFIED reward calculation matching successful environments
                idle_time = max(0, start_time - machine_available_time)
                reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                              previous_makespan, self.current_makespan, len(newly_arrived))
                
                info = {
                    "makespan": self.current_makespan,
                    "newly_arrived_jobs": len(newly_arrived),
                    "total_arrived_jobs": len(self.arrived_jobs)
                }
                
                return self._get_observation(), reward, terminated, False, info

            def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan, num_new_arrivals):
                """SIMPLIFIED reward function matching successful environments."""
                
                if self.reward_mode == "makespan_increment":
                    # Use SAME reward structure as successful environments
                    if previous_makespan is not None and current_makespan is not None:
                        makespan_increment = current_makespan - previous_makespan
                        reward = -makespan_increment  # Negative increment
                        
                        # Small bonus for utilizing newly arrived jobs (dynamic advantage)
                        if num_new_arrivals > 0:
                            reward += 5.0 * num_new_arrivals
                        
                        # Add completion bonus
                        if done:
                            reward += 50.0
                            
                        return reward
                    else:
                        return -proc_time
                else:
                    # Default reward function matching successful environments
                    reward = 10.0 - proc_time * 0.1 - idle_time
                    if done:
                        reward += 100.0
                    return reward

            def _get_observation(self):
                """Generate observation - SAME structure as successful environments."""
                norm_factor = max(self.current_makespan, 1.0)
                obs = []
                
                # Machine availability (normalized by current makespan)
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
                
                # Job arrival status (arrived or not)
                for job_id in self.job_ids:
                    if job_id in self.arrived_jobs:
                        obs.append(1.0)  # Job is available
                    else:
                        obs.append(0.0)  # Job not yet arrived
                    
                # Current makespan
                obs.append(self.current_makespan / norm_factor)
                
                # Ensure correct size
                target_size = self.observation_space.shape[0]
                if len(obs) < target_size:
                    obs.extend([0.0] * (target_size - len(obs)))
                elif len(obs) > target_size:
                    obs = obs[:target_size]
                
                obs_array = np.array(obs, dtype=np.float32)
                obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
                
                return obs_array

            def render(self, mode='human'):
                """Render the current state (optional)."""
                if mode == 'human':
                    print(f"\n=== Time: {self.current_makespan:.2f} ===")
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

def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, reward_mode="makespan_increment"):
    """
    Train a perfect knowledge RL agent using the same approach as test3_backup.py.
    
    Key insight: Train on deterministic arrival times (like the test scenario)
    rather than trying to create a complex "perfect knowledge" environment.
    This matches the working approach from test3_backup.py.
    """
    print(f"\n--- Training Perfect Knowledge RL Agent (test3_backup.py approach) ---")
    print(f"Training arrival times: {arrival_times}")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_perfect_env():
        # Use PerfectKnowledgeFJSPEnv for both training and evaluation consistency
        class PoissonDynamicFJSPEnv(gym.Env):
            """
            SIMPLIFIED Dynamic FJSP Environment with Poisson-distributed job arrivals.
            Uses the SAME structure as StaticFJSPEnv and PerfectKnowledgeFJSPEnv for consistency.
            """
            
            metadata = {"render.modes": ["human"]}

            def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05, 
                         max_time_horizon=200, reward_mode="makespan_increment", seed=None):
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
                
                # USE SAME ACTION SPACE as successful environments (FIXED, not dynamic)
                self.action_space = spaces.Discrete(
                    min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
                )
                
                # USE SAME OBSERVATION SPACE as successful environments
                obs_size = (
                    len(self.machines) +                    # Machine availability
                    self.num_jobs * self.max_ops_per_job +  # Operation completion status
                    self.num_jobs +                         # Job progress ratios  
                    self.num_jobs +                         # Job arrival status
                    1                                       # Current makespan
                )
                
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
                )
                
                self._reset_state()

            def _reset_state(self):
                """Reset all environment state variables - SAME as successful environments."""
                self.machine_next_free = {m: 0.0 for m in self.machines}
                self.schedule = {m: [] for m in self.machines}
                self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.next_operation = {job_id: 0 for job_id in self.job_ids}
                
                self.current_makespan = 0.0
                self.operations_scheduled = 0
                self.episode_step = 0
                self.max_episode_steps = self.total_operations * 2
                
                # Job arrival management - simplified
                self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
                self.job_arrival_times = {}
                
                # Generate Poisson arrival times for dynamic jobs
                self._generate_poisson_arrivals()

            def _generate_poisson_arrivals(self):
                """Generate arrival times for dynamic jobs using Poisson process."""
                # Initialize arrival times
                for job_id in self.initial_job_ids:
                    self.job_arrival_times[job_id] = 0.0
                
                # Generate inter-arrival times using exponential distribution
                current_time = 0.0
                for job_id in self.dynamic_job_ids:
                    inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
                    current_time += inter_arrival_time
                    
                    # Round to nearest integer for simplicity
                    integer_arrival_time = round(current_time)
                    
                    if integer_arrival_time <= self.max_time_horizon:
                        self.job_arrival_times[job_id] = float(integer_arrival_time)
                    else:
                        self.job_arrival_times[job_id] = float('inf')  # Won't arrive in this episode

            def reset(self, seed=None, options=None):
                """Reset the environment for a new episode - SAME structure as successful environments."""
                global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
                
                if seed is not None:
                    super().reset(seed=seed, options=options)
                    random.seed(seed)
                    np.random.seed(seed)
                
                self._reset_state()
                
                # Track arrival times for analysis
                TRAINING_EPISODE_COUNT += 1
                episode_arrivals = []
                for job_id, arr_time in self.job_arrival_times.items():
                    if arr_time != float('inf') and arr_time > 0:  # Only dynamic arrivals
                        episode_arrivals.append(arr_time)
                
                if episode_arrivals:
                    TRAINING_ARRIVAL_TIMES.extend(episode_arrivals)
                
                return self._get_observation(), {}

            def _decode_action(self, action):
                """Decode action - SAME as successful environments."""
                action = int(action) % self.action_space.n
                num_machines = len(self.machines)
                ops_per_job = self.max_ops_per_job
                
                job_idx = action // (ops_per_job * num_machines)
                op_idx = (action % (ops_per_job * num_machines)) // num_machines
                machine_idx = action % num_machines
                
                job_idx = min(job_idx, self.num_jobs - 1)
                machine_idx = min(machine_idx, len(self.machines) - 1)
                
                return job_idx, op_idx, machine_idx

            def _is_valid_action(self, job_idx, op_idx, machine_idx):
                """Check if action is valid - SAME as successful environments."""
                if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
                    return False
                
                job_id = self.job_ids[job_idx]
                
                # Check if job has arrived
                if job_id not in self.arrived_jobs:
                    return False
                    
                # Check operation index validity
                if not (0 <= op_idx < len(self.jobs[job_id])):
                    return False
                    
                # Check if this is the next operation
                if op_idx != self.next_operation[job_id]:
                    return False
                    
                # Check machine compatibility
                machine_name = self.machines[machine_idx]
                if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
                    return False
                    
                return True

            def action_masks(self):
                """Generate action masks - SAME as successful environments."""
                mask = np.full(self.action_space.n, False, dtype=bool)
                
                if self.operations_scheduled >= self.total_operations:
                    return mask

                valid_action_count = 0
                for job_idx, job_id in enumerate(self.job_ids):
                    if job_id not in self.arrived_jobs:
                        continue
                        
                    next_op_idx = self.next_operation[job_id]
                    if next_op_idx >= len(self.jobs[job_id]):
                        continue
                        
                    for machine_idx, machine in enumerate(self.machines):
                        if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                            action = job_idx * (self.max_ops_per_job * len(self.machines)) + next_op_idx * len(self.machines) + machine_idx
                            if action < self.action_space.n:
                                mask[action] = True
                                valid_action_count += 1
                
                if valid_action_count == 0:
                    mask.fill(True)
                    
                return mask

            def step(self, action):
                """Step function - SIMPLIFIED to match successful environments."""
                self.episode_step += 1
                
                # Safety check for infinite episodes
                if self.episode_step >= self.max_episode_steps:
                    return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
                
                job_idx, op_idx, machine_idx = self._decode_action(action)

                # Use softer invalid action handling like successful environments
                if not self._is_valid_action(job_idx, op_idx, machine_idx):
                    return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

                job_id = self.job_ids[job_idx]
                machine = self.machines[machine_idx]
                
                # Calculate timing using successful environments' approach
                machine_available_time = self.machine_next_free.get(machine, 0.0)
                job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                 else self.job_arrival_times.get(job_id, 0.0))
                
                start_time = max(machine_available_time, job_ready_time)
                proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
                end_time = start_time + proc_time

                # Update state
                previous_makespan = self.current_makespan
                self.machine_next_free[machine] = end_time
                self.operation_end_times[job_id][op_idx] = end_time
                self.completed_ops[job_id][op_idx] = True
                self.next_operation[job_id] += 1
                self.operations_scheduled += 1
                
                # Update makespan and check for new arrivals (key improvement)
                self.current_makespan = max(self.current_makespan, end_time)
                
                # Check for newly arrived jobs (deterministic based on current makespan)
                newly_arrived = []
                for job_id_check, arrival_time in self.job_arrival_times.items():
                    if (job_id_check not in self.arrived_jobs and 
                        arrival_time <= self.current_makespan and 
                        arrival_time != float('inf')):
                        self.arrived_jobs.add(job_id_check)
                        newly_arrived.append(job_id_check)

                # Record in schedule
                self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

                # Check termination
                terminated = self.operations_scheduled >= self.total_operations
                
                # SIMPLIFIED reward calculation matching successful environments
                idle_time = max(0, start_time - machine_available_time)
                reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                              previous_makespan, self.current_makespan, len(newly_arrived))
                
                info = {
                    "makespan": self.current_makespan,
                    "newly_arrived_jobs": len(newly_arrived),
                    "total_arrived_jobs": len(self.arrived_jobs)
                }
                
                return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan, num_new_arrivals):
        """SIMPLIFIED reward function matching successful environments."""
        
        if self.reward_mode == "makespan_increment":
            # Use SAME reward structure as successful environments
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment
                
                # Small bonus for utilizing newly arrived jobs (dynamic advantage)
                if num_new_arrivals > 0:
                    reward += 5.0 * num_new_arrivals
                
                # Add completion bonus
                if done:
                    reward += 50.0
                    
                return reward
            else:
                return -proc_time
        else:
            # Default reward function matching successful environments
            reward = 10.0 - proc_time * 0.1 - idle_time
            if done:
                reward += 100.0
            return reward

    def _get_observation(self):
        """Generate observation - SAME structure as successful environments."""
        norm_factor = max(self.current_makespan, 1.0)
        obs = []
        
        # Machine availability (normalized by current makespan)
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
        
        # Job arrival status (arrived or not)
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                obs.append(1.0)  # Job is available
            else:
                obs.append(0.0)  # Job not yet arrived
            
        # Current makespan
        obs.append(self.current_makespan / norm_factor)
        
        # Ensure correct size
        target_size = self.observation_space.shape[0]
        if len(obs) < target_size:
            obs.extend([0.0] * (target_size - len(obs)))
        elif len(obs) > target_size:
            obs = obs[:target_size]
        
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array

    def render(self, mode='human'):
        """Render the current state (optional)."""
        if mode == 'human':
            print(f"\n=== Time: {self.current_makespan:.2f} ===")
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

def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, reward_mode="makespan_increment"):
    """
    Train a perfect knowledge RL agent using the same approach as test3_backup.py.
    
    Key insight: Train on deterministic arrival times (like the test scenario)
    rather than trying to create a complex "perfect knowledge" environment.
    This matches the working approach from test3_backup.py.
    """
    print(f"\n--- Training Perfect Knowledge RL Agent (test3_backup.py approach) ---")
    print(f"Training arrival times: {arrival_times}")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_perfect_env():
        # Use PerfectKnowledgeFJSPEnv for both training and evaluation consistency
        class PoissonDynamicFJSPEnv(gym.Env):
            """
            SIMPLIFIED Dynamic FJSP Environment with Poisson-distributed job arrivals.
            Uses the SAME structure as StaticFJSPEnv and PerfectKnowledgeFJSPEnv for consistency.
            """
            
            metadata = {"render.modes": ["human"]}

            def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05, 
                         max_time_horizon=200, reward_mode="makespan_increment", seed=None):
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
                
                # USE SAME ACTION SPACE as successful environments (FIXED, not dynamic)
                self.action_space = spaces.Discrete(
                    min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
                )
                
                # USE SAME OBSERVATION SPACE as successful environments
                obs_size = (
                    len(self.machines) +                    # Machine availability
                    self.num_jobs * self.max_ops_per_job +  # Operation completion status
                    self.num_jobs +                         # Job progress ratios  
                    self.num_jobs +                         # Job arrival status
                    1                                       # Current makespan
                )
                
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
                )
                
                self._reset_state()

            def _reset_state(self):
                """Reset all environment state variables - SAME as successful environments."""
                self.machine_next_free = {m: 0.0 for m in self.machines}
                self.schedule = {m: [] for m in self.machines}
                self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.next_operation = {job_id: 0 for job_id in self.job_ids}
                
                self.current_makespan = 0.0
                self.operations_scheduled = 0
                self.episode_step = 0
                self.max_episode_steps = self.total_operations * 2
                
                # Job arrival management - simplified
                self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
                self.job_arrival_times = {}
                
                # Generate Poisson arrival times for dynamic jobs
                self._generate_poisson_arrivals()

            def _generate_poisson_arrivals(self):
                """Generate arrival times for dynamic jobs using Poisson process."""
                # Initialize arrival times
                for job_id in self.initial_job_ids:
                    self.job_arrival_times[job_id] = 0.0
                
                # Generate inter-arrival times using exponential distribution
                current_time = 0.0
                for job_id in self.dynamic_job_ids:
                    inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
                    current_time += inter_arrival_time
                    
                    # Round to nearest integer for simplicity
                    integer_arrival_time = round(current_time)
                    
                    if integer_arrival_time <= self.max_time_horizon:
                        self.job_arrival_times[job_id] = float(integer_arrival_time)
                    else:
                        self.job_arrival_times[job_id] = float('inf')  # Won't arrive in this episode

            def reset(self, seed=None, options=None):
                """Reset the environment for a new episode - SAME structure as successful environments."""
                global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
                
                if seed is not None:
                    super().reset(seed=seed, options=options)
                    random.seed(seed)
                    np.random.seed(seed)
                
                self._reset_state()
                
                # Track arrival times for analysis
                TRAINING_EPISODE_COUNT += 1
                episode_arrivals = []
                for job_id, arr_time in self.job_arrival_times.items():
                    if arr_time != float('inf') and arr_time > 0:  # Only dynamic arrivals
                        episode_arrivals.append(arr_time)
                
                if episode_arrivals:
                    TRAINING_ARRIVAL_TIMES.extend(episode_arrivals)
                
                return self._get_observation(), {}

            def _decode_action(self, action):
                """Decode action - SAME as successful environments."""
                action = int(action) % self.action_space.n
                num_machines = len(self.machines)
                ops_per_job = self.max_ops_per_job
                
                job_idx = action // (ops_per_job * num_machines)
                op_idx = (action % (ops_per_job * num_machines)) // num_machines
                machine_idx = action % num_machines
                
                job_idx = min(job_idx, self.num_jobs - 1)
                machine_idx = min(machine_idx, len(self.machines) - 1)
                
                return job_idx, op_idx, machine_idx

            def _is_valid_action(self, job_idx, op_idx, machine_idx):
                """Check if action is valid - SAME as successful environments."""
                if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
                    return False
                
                job_id = self.job_ids[job_idx]
                
                # Check if job has arrived
                if job_id not in self.arrived_jobs:
                    return False
                    
                # Check operation index validity
                if not (0 <= op_idx < len(self.jobs[job_id])):
                    return False
                    
                # Check if this is the next operation
                if op_idx != self.next_operation[job_id]:
                    return False
                    
                # Check machine compatibility
                machine_name = self.machines[machine_idx]
                if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
                    return False
                    
                return True

            def action_masks(self):
                """Generate action masks - SAME as successful environments."""
                mask = np.full(self.action_space.n, False, dtype=bool)
                
                if self.operations_scheduled >= self.total_operations:
                    return mask

                valid_action_count = 0
                for job_idx, job_id in enumerate(self.job_ids):
                    if job_id not in self.arrived_jobs:
                        continue
                        
                    next_op_idx = self.next_operation[job_id]
                   
                    if next_op_idx >= len(self.jobs[job_id]):
                        continue
                        
                    for machine_idx, machine in enumerate(self.machines):
                        if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                            action = job_idx * (self.max_ops_per_job * len(self.machines)) + next_op_idx * len(self.machines) + machine_idx
                            if action < self.action_space.n:
                                mask[action] = True
                                valid_action_count += 1
                
                if valid_action_count == 0:
                    mask.fill(True)
                    
                return mask

            def step(self, action):
                """Step function - SIMPLIFIED to match successful environments."""
                self.episode_step += 1
                
                # Safety check for infinite episodes
                if self.episode_step >= self.max_episode_steps:
                    return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
                
                job_idx, op_idx, machine_idx = self._decode_action(action)

                # Use softer invalid action handling like successful environments
                if not self._is_valid_action(job_idx, op_idx, machine_idx):
                    return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

                job_id = self.job_ids[job_idx]
                machine = self.machines[machine_idx]
                
                # Calculate timing using successful environments' approach
                machine_available_time = self.machine_next_free.get(machine, 0.0)
                job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                 else self.job_arrival_times.get(job_id, 0.0))
                
                start_time = max(machine_available_time, job_ready_time)
                proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
                end_time = start_time + proc_time

                # Update state
                previous_makespan = self.current_makespan
                self.machine_next_free[machine] = end_time
                self.operation_end_times[job_id][op_idx] = end_time
                self.completed_ops[job_id][op_idx] = True
                self.next_operation[job_id] += 1
                self.operations_scheduled += 1
                
                # Update makespan and check for new arrivals (key improvement)
                self.current_makespan = max(self.current_makespan, end_time)
                
                # Check for newly arrived jobs (deterministic based on current makespan)
                newly_arrived = []
                for job_id_check, arrival_time in self.job_arrival_times.items():
                    if (job_id_check not in self.arrived_jobs and 
                        arrival_time <= self.current_makespan and 
                        arrival_time != float('inf')):
                        self.arrived_jobs.add(job_id_check)
                        newly_arrived.append(job_id_check)

                # Record in schedule
                self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

                # Check termination
                terminated = self.operations_scheduled >= self.total_operations
                
                # SIMPLIFIED reward calculation matching successful environments
                idle_time = max(0, start_time - machine_available_time)
                reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                              previous_makespan, self.current_makespan, len(newly_arrived))
                
                info = {
                    "makespan": self.current_makespan,
                    "newly_arrived_jobs": len(newly_arrived),
                    "total_arrived_jobs": len(self.arrived_jobs)
                }
                
                return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan, num_new_arrivals):
        """SIMPLIFIED reward function matching successful environments."""
        
        if self.reward_mode == "makespan_increment":
            # Use SAME reward structure as successful environments
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment
                
                # Small bonus for utilizing newly arrived jobs (dynamic advantage)
                if num_new_arrivals > 0:
                    reward += 5.0 * num_new_arrivals
                
                # Add completion bonus
                if done:
                    reward += 50.0
                    
                return reward
            else:
                return -proc_time
        else:
            # Default reward function matching successful environments
            reward = 10.0 - proc_time * 0.1 - idle_time
            if done:
                reward += 100.0
            return reward

    def _get_observation(self):
        """Generate observation - SAME structure as successful environments."""
        norm_factor = max(self.current_makespan, 1.0)
        obs = []
        
        # Machine availability (normalized by current makespan)
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
        
        # Job arrival status (arrived or not)
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                obs.append(1.0)  # Job is available
            else:
                obs.append(0.0)  # Job not yet arrived
            
        # Current makespan
        obs.append(self.current_makespan / norm_factor)
        
        # Ensure correct size
        target_size = self.observation_space.shape[0]
        if len(obs) < target_size:
            obs.extend([0.0] * (target_size - len(obs)))
        elif len(obs) > target_size:
            obs = obs[:target_size]
        
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array

    def render(self, mode='human'):
        """Render the current state (optional)."""
        if mode == 'human':
            print(f"\n=== Time: {self.current_makespan:.2f} ===")
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

def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, reward_mode="makespan_increment"):
    """
    Train a perfect knowledge RL agent using the same approach as test3_backup.py.
    
    Key insight: Train on deterministic arrival times (like the test scenario)
    rather than trying to create a complex "perfect knowledge" environment.
    This matches the working approach from test3_backup.py.
    """
    print(f"\n--- Training Perfect Knowledge RL Agent (test3_backup.py approach) ---")
    print(f"Training arrival times: {arrival_times}")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_perfect_env():
        # Use PerfectKnowledgeFJSPEnv for both training and evaluation consistency
        class PoissonDynamicFJSPEnv(gym.Env):
            """
            SIMPLIFIED Dynamic FJSP Environment with Poisson-distributed job arrivals.
            Uses the SAME structure as StaticFJSPEnv and PerfectKnowledgeFJSPEnv for consistency.
            """
            
            metadata = {"render.modes": ["human"]}

            def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05, 
                         max_time_horizon=200, reward_mode="makespan_increment", seed=None):
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
                
                # USE SAME ACTION SPACE as successful environments (FIXED, not dynamic)
                self.action_space = spaces.Discrete(
                    min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
                )
                
                # USE SAME OBSERVATION SPACE as successful environments
                obs_size = (
                    len(self.machines) +                    # Machine availability
                    self.num_jobs * self.max_ops_per_job +  # Operation completion status
                    self.num_jobs +                         # Job progress ratios  
                    self.num_jobs +                         # Job arrival status
                    1                                       # Current makespan
                )
                
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
                )
                
                self._reset_state()

            def _reset_state(self):
                """Reset all environment state variables - SAME as successful environments."""
                self.machine_next_free = {m: 0.0 for m in self.machines}
                self.schedule = {m: [] for m in self.machines}
                self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.next_operation = {job_id: 0 for job_id in self.job_ids}
                
                self.current_makespan = 0.0
                self.operations_scheduled = 0
                self.episode_step = 0
                self.max_episode_steps = self.total_operations * 2
                
                # Job arrival management - simplified
                self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
                self.job_arrival_times = {}
                
                # Generate Poisson arrival times for dynamic jobs
                self._generate_poisson_arrivals()

            def _generate_poisson_arrivals(self):
                """Generate arrival times for dynamic jobs using Poisson process."""
                # Initialize arrival times
                for job_id in self.initial_job_ids:
                    self.job_arrival_times[job_id] = 0.0
                
                # Generate inter-arrival times using exponential distribution
                current_time = 0.0
                for job_id in self.dynamic_job_ids:
                    inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
                    current_time += inter_arrival_time
                    
                    # Round to nearest integer for simplicity
                    integer_arrival_time = round(current_time)
                    
                    if integer_arrival_time <= self.max_time_horizon:
                        self.job_arrival_times[job_id] = float(integer_arrival_time)
                    else:
                        self.job_arrival_times[job_id] = float('inf')  # Won't arrive in this episode

            def reset(self, seed=None, options=None):
                """Reset the environment for a new episode - SAME structure as successful environments."""
                global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
                
                if seed is not None:
                    super().reset(seed=seed, options=options)
                    random.seed(seed)
                    np.random.seed(seed)
                
                self._reset_state()
                
                # Track arrival times for analysis
                TRAINING_EPISODE_COUNT += 1
                episode_arrivals = []
                for job_id, arr_time in self.job_arrival_times.items():
                    if arr_time != float('inf') and arr_time > 0:  # Only dynamic arrivals
                        episode_arrivals.append(arr_time)
                
                if episode_arrivals:
                    TRAINING_ARRIVAL_TIMES.extend(episode_arrivals)
                
                return self._get_observation(), {}

            def _decode_action(self, action):
                """Decode action - SAME as successful environments."""
                action = int(action) % self.action_space.n
                num_machines = len(self.machines)
                ops_per_job = self.max_ops_per_job
                
                job_idx = action // (ops_per_job * num_machines)
                op_idx = (action % (ops_per_job * num_machines)) // num_machines
                machine_idx = action % num_machines
                
                job_idx = min(job_idx, self.num_jobs - 1)
                machine_idx = min(machine_idx, len(self.machines) - 1)
                
                return job_idx, op_idx, machine_idx

            def _is_valid_action(self, job_idx, op_idx, machine_idx):
                """Check if action is valid - SAME as successful environments."""
                if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
                    return False
                
                job_id = self.job_ids[job_idx]
                
                # Check if job has arrived
                if job_id not in self.arrived_jobs:
                    return False
                    
                # Check operation index validity
                if not (0 <= op_idx < len(self.jobs[job_id])):
                    return False
                    
                # Check if this is the next operation
                if op_idx != self.next_operation[job_id]:
                    return False
                    
                # Check machine compatibility
                machine_name = self.machines[machine_idx]
                if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
                    return False
                    
                return True

            def action_masks(self):
                """Generate action masks - SAME as successful environments."""
                mask = np.full(self.action_space.n, False, dtype=bool)
                
                if self.operations_scheduled >= self.total_operations:
                    return mask

                valid_action_count = 0
                for job_idx, job_id in enumerate(self.job_ids):
                    if job_id not in self.arrived_jobs:
                        continue
                        
                    next_op_idx = self.next_operation[job_id]
                    if next_op_idx >= len(self.jobs[job_id]):
                        continue
                        
                    for machine_idx, machine in enumerate(self.machines):
                        if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                            action = job_idx * (self.max_ops_per_job * len(self.machines)) + next_op_idx * len(self.machines) + machine_idx
                            if action < self.action_space.n:
                                mask[action] = True
                                valid_action_count += 1
                
                if valid_action_count == 0:
                    mask.fill(True)
                    
                return mask

            def step(self, action):
                """Step function - SIMPLIFIED to match successful environments."""
                self.episode_step += 1
                
                # Safety check for infinite episodes
                if self.episode_step >= self.max_episode_steps:
                    return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
                
                job_idx, op_idx, machine_idx = self._decode_action(action)

                # Use softer invalid action handling like successful environments
                if not self._is_valid_action(job_idx, op_idx, machine_idx):
                    return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

                job_id = self.job_ids[job_idx]
                machine = self.machines[machine_idx]
                
                # Calculate timing using successful environments' approach
                machine_available_time = self.machine_next_free.get(machine, 0.0)
                job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                 else self.job_arrival_times.get(job_id, 0.0))
                
                start_time = max(machine_available_time, job_ready_time)
                proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
                end_time = start_time + proc_time

                # Update state
                previous_makespan = self.current_makespan
                self.machine_next_free[machine] = end_time
                self.operation_end_times[job_id][op_idx] = end_time
                self.completed_ops[job_id][op_idx] = True
                self.next_operation[job_id] += 1
                self.operations_scheduled += 1
                
                # Update makespan and check for new arrivals (key improvement)
                self.current_makespan = max(self.current_makespan, end_time)
                
                # Check for newly arrived jobs (deterministic based on current makespan)
                newly_arrived = []
                for job_id_check, arrival_time in self.job_arrival_times.items():
                    if (job_id_check not in self.arrived_jobs and 
                        arrival_time <= self.current_makespan and 
                        arrival_time != float('inf')):
                        self.arrived_jobs.add(job_id_check)
                        newly_arrived.append(job_id_check)

                # Record in schedule
                self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

                # Check termination
                terminated = self.operations_scheduled >= self.total_operations
                
                # SIMPLIFIED reward calculation matching successful environments
                idle_time = max(0, start_time - machine_available_time)
                reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                              previous_makespan, self.current_makespan, len(newly_arrived))
                
                info = {
                    "makespan": self.current_makespan,
                    "newly_arrived_jobs": len(newly_arrived),
                    "total_arrived_jobs": len(self.arrived_jobs)
                }
                
                return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan, num_new_arrivals):
        """SIMPLIFIED reward function matching successful environments."""
        
        if self.reward_mode == "makespan_increment":
            # Use SAME reward structure as successful environments
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment
                
                # Small bonus for utilizing newly arrived jobs (dynamic advantage)
                if num_new_arrivals > 0:
                    reward += 5.0 * num_new_arrivals
                
                # Add completion bonus
                if done:
                    reward += 50.0
                    
                return reward
            else:
                return -proc_time
        else:
            # Default reward function matching successful environments
            reward = 10.0 - proc_time * 0.1 - idle_time
            if done:
                reward += 100.0
            return reward

    def _get_observation(self):
        """Generate observation - SAME structure as successful environments."""
        norm_factor = max(self.current_makespan, 1.0)
        obs = []
        
        # Machine availability (normalized by current makespan)
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
        
        # Job arrival status (arrived or not)
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                obs.append(1.0)  # Job is available
            else:
                obs.append(0.0)  # Job not yet arrived
            
        # Current makespan (normalized)
        makespan_norm = float(self.current_makespan) / 100.0  # Assume max makespan around 100
        obs.append(max(0.0, min(1.0, makespan_norm)))
        
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
            print(f"\n=== Time: {self.current_makespan:.2f} ===")
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

def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, reward_mode="makespan_increment"):
    """
    Train a perfect knowledge RL agent using the same approach as test3_backup.py.
    
    Key insight: Train on deterministic arrival times (like the test scenario)
    rather than trying to create a complex "perfect knowledge" environment.
    This matches the working approach from test3_backup.py.
    """
    print(f"\n--- Training Perfect Knowledge RL Agent (test3_backup.py approach) ---")
    print(f"Training arrival times: {arrival_times}")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_perfect_env():
        # Use PerfectKnowledgeFJSPEnv for both training and evaluation consistency
        class PoissonDynamicFJSPEnv(gym.Env):
            """
            SIMPLIFIED Dynamic FJSP Environment with Poisson-distributed job arrivals.
            Uses the SAME structure as StaticFJSPEnv and PerfectKnowledgeFJSPEnv for consistency.
            """
            
            metadata = {"render.modes": ["human"]}

            def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05, 
                         max_time_horizon=200, reward_mode="makespan_increment", seed=None):
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
                
                # USE SAME ACTION SPACE as successful environments (FIXED, not dynamic)
                self.action_space = spaces.Discrete(
                    min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
                )
                
                # USE SAME OBSERVATION SPACE as successful environments
                obs_size = (
                    len(self.machines) +                    # Machine availability
                    self.num_jobs * self.max_ops_per_job +  # Operation completion status
                    self.num_jobs +                         # Job progress ratios  
                    self.num_jobs +                         # Job arrival status
                    1                                       # Current makespan
                )
                
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
                )
                
                self._reset_state()

            def _reset_state(self):
                """Reset all environment state variables - SAME as successful environments."""
                self.machine_next_free = {m: 0.0 for m in self.machines}
                self.schedule = {m: [] for m in self.machines}
                self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.next_operation = {job_id: 0 for job_id in self.job_ids}
                
                self.current_makespan = 0.0
                self.operations_scheduled = 0
                self.episode_step = 0
                self.max_episode_steps = self.total_operations * 2
                
                # Job arrival management - simplified
                self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
                self.job_arrival_times = {}
                
                # Generate Poisson arrival times for dynamic jobs
                self._generate_poisson_arrivals()

            def _generate_poisson_arrivals(self):
                """Generate arrival times for dynamic jobs using Poisson process."""
                # Initialize arrival times
                for job_id in self.initial_job_ids:
                    self.job_arrival_times[job_id] = 0.0
                
                # Generate inter-arrival times using exponential distribution
                current_time = 0.0
                for job_id in self.dynamic_job_ids:
                    inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
                    current_time += inter_arrival_time
                    
                    # Round to nearest integer for simplicity
                    integer_arrival_time = round(current_time)
                    
                    if integer_arrival_time <= self.max_time_horizon:
                        self.job_arrival_times[job_id] = float(integer_arrival_time)
                    else:
                        self.job_arrival_times[job_id] = float('inf')  # Won't arrive in this episode

            def reset(self, seed=None, options=None):
                """Reset the environment for a new episode - SAME structure as successful environments."""
                global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
                
                if seed is not None:
                    super().reset(seed=seed, options=options)
                    random.seed(seed)
                    np.random.seed(seed)
                
                self._reset_state()
                
                # Track arrival times for analysis
                TRAINING_EPISODE_COUNT += 1
                episode_arrivals = []
                for job_id, arr_time in self.job_arrival_times.items():
                    if arr_time != float('inf') and arr_time > 0:  # Only dynamic arrivals
                        episode_arrivals.append(arr_time)
                
                if episode_arrivals:
                    TRAINING_ARRIVAL_TIMES.extend(episode_arrivals)
                
                return self._get_observation(), {}

            def _decode_action(self, action):
                """Decode action - SAME as successful environments."""
                action = int(action) % self.action_space.n
                num_machines = len(self.machines)
                ops_per_job = self.max_ops_per_job
                
                job_idx = action // (ops_per_job * num_machines)
                op_idx = (action % (ops_per_job * num_machines)) // num_machines
                machine_idx = action % num_machines
                
                job_idx = min(job_idx, self.num_jobs - 1)
                machine_idx = min(machine_idx, len(self.machines) - 1)
                
                return job_idx, op_idx, machine_idx

            def _is_valid_action(self, job_idx, op_idx, machine_idx):
                """Check if action is valid - SAME as successful environments."""
                if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
                    return False
                
                job_id = self.job_ids[job_idx]
                
                # Check if job has arrived
                if job_id not in self.arrived_jobs:
                    return False
                    
                # Check operation index validity
                if not (0 <= op_idx < len(self.jobs[job_id])):
                    return False
                    
                # Check if this is the next operation
                if op_idx != self.next_operation[job_id]:
                    return False
                    
                # Check machine compatibility
                machine_name = self.machines[machine_idx]
                if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
                    return False
                    
                return True

            def action_masks(self):
                """Generate action masks - SAME as successful environments."""
                mask = np.full(self.action_space.n, False, dtype=bool)
                
                if self.operations_scheduled >= self.total_operations:
                    return mask

                valid_action_count = 0
                for job_idx, job_id in enumerate(self.job_ids):
                    if job_id not in self.arrived_jobs:
                        continue
                        
                    next_op_idx = self.next_operation[job_id]
                    if next_op_idx >= len(self.jobs[job_id]):
                        continue
                        
                    for machine_idx, machine in enumerate(self.machines):
                        if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                            action = job_idx * (self.max_ops_per_job * len(self.machines)) + next_op_idx * len(self.machines) + machine_idx
                            if action < self.action_space.n:
                                mask[action] = True
                                valid_action_count += 1
                
                if valid_action_count == 0:
                    mask.fill(True)
                    
                return mask

            def step(self, action):
                """Step function - SIMPLIFIED to match successful environments."""
                self.episode_step += 1
                
                # Safety check for infinite episodes
                if self.episode_step >= self.max_episode_steps:
                    return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
                
                job_idx, op_idx, machine_idx = self._decode_action(action)

                # Use softer invalid action handling like successful environments
                if not self._is_valid_action(job_idx, op_idx, machine_idx):
                    return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

                job_id = self.job_ids[job_idx]
                machine = self.machines[machine_idx]
                
                # Calculate timing using successful environments' approach
                machine_available_time = self.machine_next_free.get(machine, 0.0)
                job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                 else self.job_arrival_times.get(job_id, 0.0))
                
                start_time = max(machine_available_time, job_ready_time)
                proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
                end_time = start_time + proc_time

                # Update state
                previous_makespan = self.current_makespan
                self.machine_next_free[machine] = end_time
                self.operation_end_times[job_id][op_idx] = end_time
                self.completed_ops[job_id][op_idx] = True
                self.next_operation[job_id] += 1
                self.operations_scheduled += 1
                
                # Update makespan and check for new arrivals (key improvement)
                self.current_makespan = max(self.current_makespan, end_time)
                
                # Check for newly arrived jobs (deterministic based on current makespan)
                newly_arrived = []
                for job_id_check, arrival_time in self.job_arrival_times.items():
                    if (job_id_check not in self.arrived_jobs and 
                        arrival_time <= self.current_makespan and 
                        arrival_time != float('inf')):
                        self.arrived_jobs.add(job_id_check)
                        newly_arrived.append(job_id_check)

                # Record in schedule
                self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

                # Check termination
                terminated = self.operations_scheduled >= self.total_operations
                
                # SIMPLIFIED reward calculation matching successful environments
                idle_time = max(0, start_time - machine_available_time)
                reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                              previous_makespan, self.current_makespan, len(newly_arrived))
                
                info = {
                    "makespan": self.current_makespan,
                    "newly_arrived_jobs": len(newly_arrived),
                    "total_arrived_jobs": len(self.arrived_jobs)
                }
                
                return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan, num_new_arrivals):
        """SIMPLIFIED reward function matching successful environments."""
        
        if self.reward_mode == "makespan_increment":
            # Use SAME reward structure as successful environments
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment
                
                # Small bonus for utilizing newly arrived jobs (dynamic advantage)
                if num_new_arrivals > 0:
                    reward += 5.0 * num_new_arrivals
                
                # Add completion bonus
                if done:
                    reward += 50.0
                    
                return reward
            else:
                return -proc_time
        else:
            # Default reward function matching successful environments
            reward = 10.0 - proc_time * 0.1 - idle_time
            if done:
                reward += 100.0
            return reward

    def _get_observation(self):
        """Generate observation - SAME structure as successful environments."""
        norm_factor = max(self.current_makespan, 1.0)
        obs = []
        
        # Machine availability (normalized by current makespan)
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
        
        # Job arrival status (arrived or not)
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                obs.append(1.0)  # Job is available
            else:
                obs.append(0.0)  # Job not yet arrived
            
        # Current makespan (normalized)
        makespan_norm = float(self.current_makespan) / 100.0  # Assume max makespan around 100
        obs.append(max(0.0, min(1.0, makespan_norm)))
        
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
            print(f"\n=== Time: {self.current_makespan:.2f} ===")
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

def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, reward_mode="makespan_increment"):
    """
    Train a perfect knowledge RL agent using the same approach as test3_backup.py.
    
    Key insight: Train on deterministic arrival times (like the test scenario)
    rather than trying to create a complex "perfect knowledge" environment.
    This matches the working approach from test3_backup.py.
    """
    print(f"\n--- Training Perfect Knowledge RL Agent (test3_backup.py approach) ---")
    print(f"Training arrival times: {arrival_times}")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_perfect_env():
        # Use PerfectKnowledgeFJSPEnv for both training and evaluation consistency
        class PoissonDynamicFJSPEnv(gym.Env):
            """
            SIMPLIFIED Dynamic FJSP Environment with Poisson-distributed job arrivals.
            Uses the SAME structure as StaticFJSPEnv and PerfectKnowledgeFJSPEnv for consistency.
            """
            
            metadata = {"render.modes": ["human"]}

            def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05, 
                         max_time_horizon=200, reward_mode="makespan_increment", seed=None):
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
                
                # USE SAME ACTION SPACE as successful environments (FIXED, not dynamic)
                self.action_space = spaces.Discrete(
                    min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
                )
                
                # USE SAME OBSERVATION SPACE as successful environments
                obs_size = (
                    len(self.machines) +                    # Machine availability
                    self.num_jobs * self.max_ops_per_job +  # Operation completion status
                    self.num_jobs +                         # Job progress ratios  
                    self.num_jobs +                         # Job arrival status
                    1                                       # Current makespan
                )
                
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
                )
                
                self._reset_state()

            def _reset_state(self):
                """Reset all environment state variables - SAME as successful environments."""
                self.machine_next_free = {m: 0.0 for m in self.machines}
                self.schedule = {m: [] for m in self.machines}
                self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.next_operation = {job_id: 0 for job_id in self.job_ids}
                
                self.current_makespan = 0.0
                self.operations_scheduled = 0
                self.episode_step = 0
                self.max_episode_steps = self.total_operations * 2
                
                # Job arrival management - simplified
                self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
                self.job_arrival_times = {}
                
                # Generate Poisson arrival times for dynamic jobs
                self._generate_poisson_arrivals()

            def _generate_poisson_arrivals(self):
                """Generate arrival times for dynamic jobs using Poisson process."""
                # Initialize arrival times
                for job_id in self.initial_job_ids:
                    self.job_arrival_times[job_id] = 0.0
                
                # Generate inter-arrival times using exponential distribution
                current_time = 0.0
                for job_id in self.dynamic_job_ids:
                    inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
                    current_time += inter_arrival_time
                    
                    # Round to nearest integer for simplicity
                    integer_arrival_time = round(current_time)
                    
                    if integer_arrival_time <= self.max_time_horizon:
                        self.job_arrival_times[job_id] = float(integer_arrival_time)
                    else:
                        self.job_arrival_times[job_id] = float('inf')  # Won't arrive in this episode

            def reset(self, seed=None, options=None):
                """Reset the environment for a new episode - SAME structure as successful environments."""
                global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
                
                if seed is not None:
                    super().reset(seed=seed, options=options)
                    random.seed(seed)
                    np.random.seed(seed)
                
                self._reset_state()
                
                # Track arrival times for analysis
                TRAINING_EPISODE_COUNT += 1
                episode_arrivals = []
                for job_id, arr_time in self.job_arrival_times.items():
                    if arr_time != float('inf') and arr_time > 0:  # Only dynamic arrivals
                        episode_arrivals.append(arr_time)
                
                if episode_arrivals:
                    TRAINING_ARRIVAL_TIMES.extend(episode_arrivals)
                
                return self._get_observation(), {}

            def _decode_action(self, action):
                """Decode action - SAME as successful environments."""
                action = int(action) % self.action_space.n
                num_machines = len(self.machines)
                ops_per_job = self.max_ops_per_job
                
                job_idx = action // (ops_per_job * num_machines)
                op_idx = (action % (ops_per_job * num_machines)) // num_machines
                machine_idx = action % num_machines
                
                job_idx = min(job_idx, self.num_jobs - 1)
                machine_idx = min(machine_idx, len(self.machines) - 1)
                
                return job_idx, op_idx, machine_idx

            def _is_valid_action(self, job_idx, op_idx, machine_idx):
                """Check if action is valid - SAME as successful environments."""
                if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
                    return False
                
                job_id = self.job_ids[job_idx]
                
                # Check if job has arrived
                if job_id not in self.arrived_jobs:
                    return False
                    
                # Check operation index validity
                if not (0 <= op_idx < len(self.jobs[job_id])):
                    return False
                    
                # Check if this is the next operation
                if op_idx != self.next_operation[job_id]:
                    return False
                    
                # Check machine compatibility
                machine_name = self.machines[machine_idx]
                if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
                    return False
                    
                return True

            def action_masks(self):
                """Generate action masks - SAME as successful environments."""
                mask = np.full(self.action_space.n, False, dtype=bool)
                
                if self.operations_scheduled >= self.total_operations:
                    return mask

                valid_action_count = 0
                for job_idx, job_id in enumerate(self.job_ids):
                    if job_id not in self.arrived_jobs:
                        continue
                        
                    next_op_idx = self.next_operation[job_id]
                    if next_op_idx >= len(self.jobs[job_id]):
                        continue
                        
                    for machine_idx, machine in enumerate(self.machines):
                        if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                            action = job_idx * (self.max_ops_per_job * len(self.machines)) + next_op_idx * len(self.machines) + machine_idx
                            if action < self.action_space.n:
                                mask[action] = True
                                valid_action_count += 1
                
                if valid_action_count == 0:
                    mask.fill(True)
                    
                return mask

            def step(self, action):
                """Step function - SIMPLIFIED to match successful environments."""
                self.episode_step += 1
                
                # Safety check for infinite episodes
                if self.episode_step >= self.max_episode_steps:
                    return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
                
                job_idx, op_idx, machine_idx = self._decode_action(action)

                # Use softer invalid action handling like successful environments
                if not self._is_valid_action(job_idx, op_idx, machine_idx):
                    return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

                job_id = self.job_ids[job_idx]
                machine = self.machines[machine_idx]
                
                # Calculate timing using successful environments' approach
                machine_available_time = self.machine_next_free.get(machine, 0.0)
                job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                 else self.job_arrival_times.get(job_id, 0.0))
                
                start_time = max(machine_available_time, job_ready_time)
                proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
                end_time = start_time + proc_time

                # Update state
                previous_makespan = self.current_makespan
                self.machine_next_free[machine] = end_time
                self.operation_end_times[job_id][op_idx] = end_time
                self.completed_ops[job_id][op_idx] = True
                self.next_operation[job_id] += 1
                self.operations_scheduled += 1
                
                # Update makespan and check for new arrivals (key improvement)
                self.current_makespan = max(self.current_makespan, end_time)
                
                # Check for newly arrived jobs (deterministic based on current makespan)
                newly_arrived = []
                for job_id_check, arrival_time in self.job_arrival_times.items():
                    if (job_id_check not in self.arrived_jobs and 
                        arrival_time <= self.current_makespan and 
                        arrival_time != float('inf')):
                        self.arrived_jobs.add(job_id_check)
                        newly_arrived.append(job_id_check)

                # Record in schedule
                self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

                # Check termination
                terminated = self.operations_scheduled >= self.total_operations
                
                # SIMPLIFIED reward calculation matching successful environments
                idle_time = max(0, start_time - machine_available_time)
                reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                              previous_makespan, self.current_makespan, len(newly_arrived))
                
                info = {
                    "makespan": self.current_makespan,
                    "newly_arrived_jobs": len(newly_arrived),
                    "total_arrived_jobs": len(self.arrived_jobs)
                }
                
                return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan, num_new_arrivals):
        """SIMPLIFIED reward function matching successful environments."""
        
        if self.reward_mode == "makespan_increment":
            # Use SAME reward structure as successful environments
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment
                
                # Small bonus for utilizing newly arrived jobs (dynamic advantage)
                if num_new_arrivals > 0:
                    reward += 5.0 * num_new_arrivals
                
                # Add completion bonus
                if done:
                    reward += 50.0
                    
                return reward
            else:
                return -proc_time
        else:
            # Default reward function matching successful environments
            reward = 10.0 - proc_time * 0.1 - idle_time
            if done:
                reward += 100.0
            return reward

    def _get_observation(self):
        """Generate observation - SAME structure as successful environments."""
        norm_factor = max(self.current_makespan, 1.0)
        obs = []
        
        # Machine availability (normalized by current makespan)
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
        
        # Job arrival status (arrived or not)
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                obs.append(1.0)  # Job is available
            else:
                obs.append(0.0)  # Job not yet arrived
            
        # Current makespan (normalized)
        makespan_norm = float(self.current_makespan) / 100.0  # Assume max makespan around 100
        obs.append(max(0.0, min(1.0, makespan_norm)))
        
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
            print(f"\n=== Time: {self.current_makespan:.2f} ===")
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

def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, reward_mode="makespan_increment"):
    """
    Train a perfect knowledge RL agent using the same approach as test3_backup.py.
    
    Key insight: Train on deterministic arrival times (like the test scenario)
    rather than trying to create a complex "perfect knowledge" environment.
    This matches the working approach from test3_backup.py.
    """
    print(f"\n--- Training Perfect Knowledge RL Agent (test3_backup.py approach) ---")
    print(f"Training arrival times: {arrival_times}")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_perfect_env():
        # Use PerfectKnowledgeFJSPEnv for both training and evaluation consistency
        class PoissonDynamicFJSPEnv(gym.Env):
            """
            SIMPLIFIED Dynamic FJSP Environment with Poisson-distributed job arrivals.
            Uses the SAME structure as StaticFJSPEnv and PerfectKnowledgeFJSPEnv for consistency.
            """
            
            metadata = {"render.modes": ["human"]}

            def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05, 
                         max_time_horizon=200, reward_mode="makespan_increment", seed=None):
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
                
                # USE SAME ACTION SPACE as successful environments (FIXED, not dynamic)
                self.action_space = spaces.Discrete(
                    min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
                )
                
                # USE SAME OBSERVATION SPACE as successful environments
                obs_size = (
                    len(self.machines) +                    # Machine availability
                    self.num_jobs * self.max_ops_per_job +  # Operation completion status
                    self.num_jobs +                         # Job progress ratios  
                    self.num_jobs +                         # Job arrival status
                    1                                       # Current makespan
                )
                
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
                )
                
                self._reset_state()

            def _reset_state(self):
                """Reset all environment state variables - SAME as successful environments."""
                self.machine_next_free = {m: 0.0 for m in self.machines}
                self.schedule = {m: [] for m in self.machines}
                self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.next_operation = {job_id: 0 for job_id in self.job_ids}
                
                self.current_makespan = 0.0
                self.operations_scheduled = 0
                self.episode_step = 0
                self.max_episode_steps = self.total_operations * 2
                
                # Job arrival management - simplified
                self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
                self.job_arrival_times = {}
                
                # Generate Poisson arrival times for dynamic jobs
                self._generate_poisson_arrivals()

            def _generate_poisson_arrivals(self):
                """Generate arrival times for dynamic jobs using Poisson process."""
                # Initialize arrival times
                for job_id in self.initial_job_ids:
                    self.job_arrival_times[job_id] = 0.0
                
                # Generate inter-arrival times using exponential distribution
                current_time = 0.0
                for job_id in self.dynamic_job_ids:
                    inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
                    current_time += inter_arrival_time
                    
                    # Round to nearest integer for simplicity
                    integer_arrival_time = round(current_time)
                    
                    if integer_arrival_time <= self.max_time_horizon:
                        self.job_arrival_times[job_id] = float(integer_arrival_time)
                    else:
                        self.job_arrival_times[job_id] = float('inf')  # Won't arrive in this episode

            def reset(self, seed=None, options=None):
                """Reset the environment for a new episode - SAME structure as successful environments."""
                global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
                
                if seed is not None:
                    super().reset(seed=seed, options=options)
                    random.seed(seed)
                    np.random.seed(seed)
                
                self._reset_state()
                
                # Track arrival times for analysis
                TRAINING_EPISODE_COUNT += 1
                episode_arrivals = []
                for job_id, arr_time in self.job_arrival_times.items():
                    if arr_time != float('inf') and arr_time > 0:  # Only dynamic arrivals
                        episode_arrivals.append(arr_time)
                
                if episode_arrivals:
                    TRAINING_ARRIVAL_TIMES.extend(episode_arrivals)
                
                return self._get_observation(), {}

            def _decode_action(self, action):
                """Decode action - SAME as successful environments."""
                action = int(action) % self.action_space.n
                num_machines = len(self.machines)
                ops_per_job = self.max_ops_per_job
                
                job_idx = action // (ops_per_job * num_machines)
                op_idx = (action % (ops_per_job * num_machines)) // num_machines
                machine_idx = action % num_machines
                
                job_idx = min(job_idx, self.num_jobs - 1)
                machine_idx = min(machine_idx, len(self.machines) - 1)
                
                return job_idx, op_idx, machine_idx

            def _is_valid_action(self, job_idx, op_idx, machine_idx):
                """Check if action is valid - SAME as successful environments."""
                if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
                    return False
                
                job_id = self.job_ids[job_idx]
                
                # Check if job has arrived
                if job_id not in self.arrived_jobs:
                    return False
                    
                # Check operation index validity
                if not (0 <= op_idx < len(self.jobs[job_id])):
                    return False
                    
                # Check if this is the next operation
                if op_idx != self.next_operation[job_id]:
                    return False
                    
                # Check machine compatibility
                machine_name = self.machines[machine_idx]
                if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
                    return False
                    
                return True

            def action_masks(self):
                """Generate action masks - SAME as successful environments."""
                mask = np.full(self.action_space.n, False, dtype=bool)
                
                if self.operations_scheduled >= self.total_operations:
                    return mask

                valid_action_count = 0
                for job_idx, job_id in enumerate(self.job_ids):
                    if job_id not in self.arrived_jobs:
                        continue
                        
                    next_op_idx = self.next_operation[job_id]
                    if next_op_idx >= len(self.jobs[job_id]):
                        continue
                        
                    for machine_idx, machine in enumerate(self.machines):
                        if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                            action = job_idx * (self.max_ops_per_job * len(self.machines)) + next_op_idx * len(self.machines) + machine_idx
                            if action < self.action_space.n:
                                mask[action] = True
                                valid_action_count += 1
                
                if valid_action_count == 0:
                    mask.fill(True)
                    
                return mask

            def step(self, action):
                """Step function - SIMPLIFIED to match successful environments."""
                self.episode_step += 1
                
                # Safety check for infinite episodes
                if self.episode_step >= self.max_episode_steps:
                    return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
                
                job_idx, op_idx, machine_idx = self._decode_action(action)

                # Use softer invalid action handling like successful environments
                if not self._is_valid_action(job_idx, op_idx, machine_idx):
                    return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

                job_id = self.job_ids[job_idx]
                machine = self.machines[machine_idx]
                
                # Calculate timing using successful environments' approach
                machine_available_time = self.machine_next_free.get(machine, 0.0)
                job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                 else self.job_arrival_times.get(job_id, 0.0))
                
                start_time = max(machine_available_time, job_ready_time)
                proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
                end_time = start_time + proc_time

                # Update state
                previous_makespan = self.current_makespan
                self.machine_next_free[machine] = end_time
                self.operation_end_times[job_id][op_idx] = end_time
                self.completed_ops[job_id][op_idx] = True
                self.next_operation[job_id] += 1
                self.operations_scheduled += 1
                
                # Update makespan and check for new arrivals (key improvement)
                self.current_makespan = max(self.current_makespan, end_time)
                
                # Check for newly arrived jobs (deterministic based on current makespan)
                newly_arrived = []
                for job_id_check, arrival_time in self.job_arrival_times.items():
                    if (job_id_check not in self.arrived_jobs and 
                        arrival_time <= self.current_makespan and 
                        arrival_time != float('inf')):
                        self.arrived_jobs.add(job_id_check)
                        newly_arrived.append(job_id_check)

                # Record in schedule
                self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

                # Check termination
                terminated = self.operations_scheduled >= self.total_operations
                
                # SIMPLIFIED reward calculation matching successful environments
                idle_time = max(0, start_time - machine_available_time)
                reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                              previous_makespan, self.current_makespan, len(newly_arrived))
                
                info = {
                    "makespan": self.current_makespan,
                    "newly_arrived_jobs": len(newly_arrived),
                    "total_arrived_jobs": len(self.arrived_jobs)
                }
                
                return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan, num_new_arrivals):
        """SIMPLIFIED reward function matching successful environments."""
        
        if self.reward_mode == "makespan_increment":
            # Use SAME reward structure as successful environments
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment
                
                # Small bonus for utilizing newly arrived jobs (dynamic advantage)
                if num_new_arrivals > 0:
                    reward += 5.0 * num_new_arrivals
                
                # Add completion bonus
                if done:
                    reward += 50.0
                    
                return reward
            else:
                return -proc_time
        else:
            # Default reward function matching successful environments
            reward = 10.0 - proc_time * 0.1 - idle_time
            if done:
                reward += 100.0
            return reward

    def _get_observation(self):
        """Generate observation - SAME structure as successful environments."""
        norm_factor = max(self.current_makespan, 1.0)
        obs = []
        
        # Machine availability (normalized by current makespan)
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
        
        # Job arrival status (arrived or not)
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                obs.append(1.0)  # Job is available
            else:
                obs.append(0.0)  # Job not yet arrived
            
        # Current makespan (normalized)
        makespan_norm = float(self.current_makespan) / 100.0  # Assume max makespan around 100
        obs.append(max(0.0, min(1.0, makespan_norm)))
        
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
            print(f"\n=== Time: {self.current_makespan:.2f} ===")
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

def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, reward_mode="makespan_increment"):
    """
    Train a perfect knowledge RL agent using the same approach as test3_backup.py.
    
    Key insight: Train on deterministic arrival times (like the test scenario)
    rather than trying to create a complex "perfect knowledge" environment.
    This matches the working approach from test3_backup.py.
    """
    print(f"\n--- Training Perfect Knowledge RL Agent (test3_backup.py approach) ---")
    print(f"Training arrival times: {arrival_times}")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_perfect_env():
        # Use PerfectKnowledgeFJSPEnv for both training and evaluation consistency
        class PoissonDynamicFJSPEnv(gym.Env):
            """
            SIMPLIFIED Dynamic FJSP Environment with Poisson-distributed job arrivals.
            Uses the SAME structure as StaticFJSPEnv and PerfectKnowledgeFJSPEnv for consistency.
            """
            
            metadata = {"render.modes": ["human"]}

            def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05, 
                         max_time_horizon=200, reward_mode="makespan_increment", seed=None):
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
                
                # USE SAME ACTION SPACE as successful environments (FIXED, not dynamic)
                self.action_space = spaces.Discrete(
                    min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
                )
                
                # USE SAME OBSERVATION SPACE as successful environments
                obs_size = (
                    len(self.machines) +                    # Machine availability
                    self.num_jobs * self.max_ops_per_job +  # Operation completion status
                    self.num_jobs +                         # Job progress ratios  
                    self.num_jobs +                         # Job arrival status
                    1                                       # Current makespan
                )
                
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
                )
                
                self._reset_state()

            def _reset_state(self):
                """Reset all environment state variables - SAME as successful environments."""
                self.machine_next_free = {m: 0.0 for m in self.machines}
                self.schedule = {m: [] for m in self.machines}
                self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
                self.next_operation = {job_id: 0 for job_id in self.job_ids}
                
                self.current_makespan = 0.0
                self.operations_scheduled = 0
                self.episode_step = 0
                self.max_episode_steps = self.total_operations * 2
                
                # Job arrival management - simplified
                self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
                self.job_arrival_times = {}
                
                # Generate Poisson arrival times for dynamic jobs
                self._generate_poisson_arrivals()

            def _generate_poisson_arrivals(self):
                """Generate arrival times for dynamic jobs using Poisson process."""
                # Initialize arrival times
                for job_id in self.initial_job_ids:
                    self.job_arrival_times[job_id] = 0.0
                
                # Generate inter-arrival times using exponential distribution
                current_time = 0.0
                for job_id in self.dynamic_job_ids:
                    inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
                    current_time += inter_arrival_time
                    
                    # Round to nearest integer for simplicity
                    integer_arrival_time = round(current_time)
                    
                    if integer_arrival_time <= self.max_time_horizon:
                        self.job_arrival_times[job_id] = float(integer_arrival_time)
                    else:
                        self.job_arrival_times[job_id] = float('inf