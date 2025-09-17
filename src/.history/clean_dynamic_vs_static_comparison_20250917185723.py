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
            self.num_jobs * self.max_ops_per_job * len(self.machines)
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
        """
        Simplified observation similar to successful PerfectKnowledgeFJSPEnv.
        Focus on essential information without over-complication.
        """
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
        
        # Job arrival status (simple binary: arrived or not)
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
        # class PoissonDynamicFJSPEnv(gym.Env):
        #     """
        #     SIMPLIFIED Dynamic FJSP Environment with Poisson-distributed job arrivals.
        #     Uses the SAME structure as StaticFJSPEnv and PerfectKnowledgeFJSPEnv for consistency.
        #     """
            
        #     metadata = {"render.modes": ["human"]}

        #     def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05, 
        #                  max_time_horizon=200, reward_mode="makespan_increment", seed=None):
        #         super().__init__()
                
        #         if seed is not None:
        #             random.seed(seed)
        #             np.random.seed(seed)
                
        #         self.jobs = jobs_data
        #         self.machines = machine_list
        #         self.job_ids = list(self.jobs.keys())
                
        #         # Handle initial_jobs as either integer or list
        #         if isinstance(initial_jobs, list):
        #             self.initial_job_ids = initial_jobs
        #             self.dynamic_job_ids = [j for j in self.job_ids if j not in initial_jobs]
        #             self.initial_jobs = len(initial_jobs)
        #         else:
        #             self.initial_jobs = min(initial_jobs, len(self.job_ids))
        #             self.initial_job_ids = self.job_ids[:self.initial_jobs]
        #             self.dynamic_job_ids = self.job_ids[self.initial_jobs:]
                
        #         self.arrival_rate = arrival_rate
        #         self.max_time_horizon = max_time_horizon
        #         self.reward_mode = reward_mode
                
        #         # Environment parameters
        #         self.num_jobs = len(self.job_ids)
        #         self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        #         self.total_operations = sum(len(ops) for ops in self.jobs.values())
                
        #         # USE SAME ACTION SPACE as successful environments (FIXED, not dynamic)
        #         self.action_space = spaces.Discrete(
        #             min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
        #         )
                
        #         # USE SAME OBSERVATION SPACE as successful environments
        #         obs_size = (
        #             len(self.machines) +                    # Machine availability
        #             self.num_jobs * self.max_ops_per_job +  # Operation completion status
        #             self.num_jobs +                         # Job progress ratios  
        #             self.num_jobs +                         # Job arrival status
        #             1                                       # Current makespan
        #         )
                
        #         self.observation_space = spaces.Box(
        #             low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        #         )
                
        #         self._reset_state()

        #     def _reset_state(self):
        #         """Reset all environment state variables - SAME as successful environments."""
        #         self.machine_next_free = {m: 0.0 for m in self.machines}
        #         self.schedule = {m: [] for m in self.machines}
        #         self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        #         self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        #         self.next_operation = {job_id: 0 for job_id in self.job_ids}
                
        #         self.current_makespan = 0.0
        #         self.operations_scheduled = 0
        #         self.episode_step = 0
        #         self.max_episode_steps = self.total_operations * 2
                
        #         # Job arrival management - simplified
        #         self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
        #         self.job_arrival_times = {}
                
        #         # Generate Poisson arrival times for dynamic jobs
        #         self._generate_poisson_arrivals()

        #     def _generate_poisson_arrivals(self):
        #         """Generate arrival times for dynamic jobs using Poisson process."""
        #         # Initialize arrival times
        #         for job_id in self.initial_job_ids:
        #             self.job_arrival_times[job_id] = 0.0
                
        #         # Generate inter-arrival times using exponential distribution
        #         current_time = 0.0
        #         for job_id in self.dynamic_job_ids:
        #             inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
        #             current_time += inter_arrival_time
                    
        #             # Round to nearest integer for simplicity
        #             integer_arrival_time = round(current_time)
                    
        #             if integer_arrival_time <= self.max_time_horizon:
        #                 self.job_arrival_times[job_id] = float(integer_arrival_time)
        #             else:
        #                 self.job_arrival_times[job_id] = float('inf')  # Won't arrive in this episode

        #     def reset(self, seed=None, options=None):
        #         """Reset the environment for a new episode - SAME structure as successful environments."""
        #         global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
                
        #         if seed is not None:
        #             super().reset(seed=seed, options=options)
        #             random.seed(seed)
        #             np.random.seed(seed)
                
        #         self._reset_state()
                
        #         # Track arrival times for analysis
        #         TRAINING_EPISODE_COUNT += 1
        #         episode_arrivals = []
        #         for job_id, arr_time in self.job_arrival_times.items():
        #             if arr_time != float('inf') and arr_time > 0:  # Only dynamic arrivals
        #                 episode_arrivals.append(arr_time)
                
        #         if episode_arrivals:
        #             TRAINING_ARRIVAL_TIMES.extend(episode_arrivals)
                
        #         return self._get_observation(), {}

        #     def _decode_action(self, action):
        #         """Decode action - SAME as successful environments."""
        #         action = int(action) % self.action_space.n
        #         num_machines = len(self.machines)
        #         ops_per_job = self.max_ops_per_job
                
        #         job_idx = action // (ops_per_job * num_machines)
        #         op_idx = (action % (ops_per_job * num_machines)) // num_machines
        #         machine_idx = action % num_machines
                
        #         job_idx = min(job_idx, self.num_jobs - 1)
        #         machine_idx = min(machine_idx, len(self.machines) - 1)
                
        #         return job_idx, op_idx, machine_idx

        #     def _is_valid_action(self, job_idx, op_idx, machine_idx):
        #         """Check if action is valid - SAME as successful environments."""
        #         if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
        #             return False
                
        #         job_id = self.job_ids[job_idx]
                
        #         # Check if job has arrived
        #         if job_id not in self.arrived_jobs:
        #             return False
                    
        #         # Check operation index validity
        #         if not (0 <= op_idx < len(self.jobs[job_id])):
        #             return False
                    
        #         # Check if this is the next operation
        #         if op_idx != self.next_operation[job_id]:
        #             return False
                    
        #         # Check machine compatibility
        #         machine_name = self.machines[machine_idx]
        #         if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
        #             return False
                    
        #         return True

        #     def action_masks(self):
        #         """Generate action masks - SAME as successful environments."""
        #         mask = np.full(self.action_space.n, False, dtype=bool)
                
        #         if self.operations_scheduled >= self.total_operations:
        #             return mask

        #         valid_action_count = 0
        #         for job_idx, job_id in enumerate(self.job_ids):
        #             if job_id not in self.arrived_jobs:
        #                 continue
                        
        #             next_op_idx = self.next_operation[job_id]
        #             if next_op_idx >= len(self.jobs[job_id]):
        #                 continue
                        
        #             for machine_idx, machine in enumerate(self.machines):
        #                 if machine in self.jobs[job_id][next_op_idx]['proc_times']:
        #                     action = job_idx * (self.max_ops_per_job * len(self.machines)) + next_op_idx * len(self.machines) + machine_idx
        #                     if action < self.action_space.n:
        #                         mask[action] = True
        #                         valid_action_count += 1
                
        #         if valid_action_count == 0:
        #             mask.fill(True)
                
        #         return mask

        #     def step(self, action):
        #         """Step function - SIMPLIFIED to match successful environments."""
        #         self.episode_step += 1
                
        #         # Safety check for infinite episodes
        #         if self.episode_step >= self.max_episode_steps:
        #             return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
                
        #         job_idx, op_idx, machine_idx = self._decode_action(action)

        #         # Use softer invalid action handling like successful environments
        #         if not self._is_valid_action(job_idx, op_idx, machine_idx):
        #             return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

        #         job_id = self.job_ids[job_idx]
        #         machine = self.machines[machine_idx]
                
        #         # Calculate timing using successful environments' approach
        #         machine_available_time = self.machine_next_free.get(machine, 0.0)
        #         job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
        #                          else self.job_arrival_times.get(job_id, 0.0))
                
        #         start_time = max(machine_available_time, job_ready_time)
        #         proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        #         end_time = start_time + proc_time

        #         # Update state
        #         previous_makespan = self.current_makespan
        #         self.machine_next_free[machine] = end_time
        #         self.operation_end_times[job_id][op_idx] = end_time
        #         self.completed_ops[job_id][op_idx] = True
        #         self.next_operation[job_id] += 1
        #         self.operations_scheduled += 1
                
        #         # Update makespan and check for new arrivals (key improvement)
        #         self.current_makespan = max(self.current_makespan, end_time)
                
        #         # Check for newly arrived jobs (deterministic based on current makespan)
        #         newly_arrived = []
        #         for job_id_check, arrival_time in self.job_arrival_times.items():
        #             if (job_id_check not in self.arrived_jobs and 
        #                 arrival_time <= self.current_makespan and 
        #                 arrival_time != float('inf')):
        #                 self.arrived_jobs.add(job_id_check)
        #                 newly_arrived.append(job_id_check)

        #         # Record in schedule
        #         self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        #         # Check termination
        #         terminated = self.operations_scheduled >= self.total_operations
                
        #         # SIMPLIFIED reward calculation matching successful environments
        #         idle_time = max(0, start_time - machine_available_time)
        #         reward = self._calculate_reward(proc_time, idle_time, terminated, 
        #                                       previous_makespan, self.current_makespan, len(newly_arrived))
                
        #         info = {
        #             "makespan": self.current_makespan,
        #             "newly_arrived_jobs": len(newly_arrived),
        #             "total_arrived_jobs": len(self.arrived_jobs)
        #         }
                
        #         return self._get_observation(), reward, terminated, False, info

        #     def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan, num_new_arrivals):
        #         """SIMPLIFIED reward function matching successful environments."""
                
        #         if self.reward_mode == "makespan_increment":
        #             # Use SAME reward structure as successful environments
        #             if previous_makespan is not None and current_makespan is not None:
        #                 makespan_increment = current_makespan - previous_makespan
        #                 reward = -makespan_increment  # Negative increment
                
        #                 # Small bonus for utilizing newly arrived jobs (dynamic advantage)
        #                 if num_new_arrivals > 0:
        #                     reward += 5.0 * num_new_arrivals
                
        #                 # Add completion bonus
        #                 if done:
        #                     reward += 50.0
                    
        #                 return reward
        #             else:
        #                 return -proc_time
        #         else:
        #             # Default reward function matching successful environments
        #             reward = 10.0 - proc_time * 0.1 - idle_time
        #             if done:
        #                 reward += 100.0
        #             return reward

        #     def _get_observation(self):
        #         """Generate observation - SAME structure as successful environments."""
        #         norm_factor = max(self.current_makespan, 1.0)
        #         obs = []
                
        #         # Machine availability (normalized by current makespan)
        #         for m in self.machines:
        #             value = float(self.machine_next_free.get(m, 0.0)) / norm_factor
        #             obs.append(max(0.0, min(1.0, value)))
                
        #         # Operation completion status (padded to max_ops_per_job)
        #         for job_id in self.job_ids:
        #             for op_idx in range(self.max_ops_per_job):
        #                 if op_idx < len(self.jobs[job_id]):
        #                     completed = 1.0 if self.completed_ops[job_id][op_idx] else 0.0
        #                 else:
        #                     completed = 1.0  # Non-existent operations considered completed
        #                 obs.append(float(completed))
                
        #         # Job progress (proportion of operations completed)
        #         for job_id in self.job_ids:
        #             total_ops = len(self.jobs[job_id])
        #             if total_ops > 0:
        #                 progress = float(self.next_operation[job_id]) / float(total_ops)
        #             else:
        #                 progress = 1.0
        #             obs.append(max(0.0, min(1.0, progress)))
                
        #         # Job arrival status (arrived or not)
        #         for job_id in self.job_ids:
        #             if job_id in self.arrived_jobs:
        #                 obs.append(1.0)  # Job is available
        #             else:
        #                 obs.append(0.0)  # Job not yet arrived
                    
        #         # Current makespan (normalized)
        #         makespan_norm = float(self.current_makespan) / 100.0  # Assume max makespan around 100
        #         obs.append(max(0.0, min(1.0, makespan_norm)))
                
        #         # Pad or truncate to match observation space
        #         target_size = self.observation_space.shape[0]
        #         if len(obs) < target_size:
        #             obs.extend([0.0] * (target_size - len(obs)))
        #         elif len(obs) > target_size:
        #             obs = obs[:target_size]
                
        #         # Ensure proper format
        #         obs_array = np.array(obs, dtype=np.float32)
        #         obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
                
        #         return obs_array

        #     def render(self, mode='human'):
        #         """Render the current state (optional)."""
        #         if mode == 'human':
        #             print(f"\n=== Time: {self.current_makespan:.2f} ===")
        #             print(f"Arrived jobs: {sorted(self.arrived_jobs)}")
        #             print(f"Completed operations: {self.operations_scheduled}")
        #             print(f"Machine status:")
        #             for m in self.machines:
        #                 print(f"  {m}: next free at {self.machine_next_free[m]:.2f}")
        env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times, reward_mode=reward_mode)
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_perfect_env])
    
    # Use similar hyperparameters as in test3_backup.py
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,        # Matches test3_backup.py
        n_steps=2048,              # Matches test3_backup.py
        batch_size=128,            # Matches test3_backup.py  
        n_epochs=10,               # Matches test3_backup.py
        gamma=1,                # Matches test3_backup.py
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[512, 512, 256],  # Matches test3_backup.py
            activation_fn=torch.nn.ReLU
        )
    )
    
    # Training with progress bar
    print("Training perfect knowledge agent (deterministic arrival times)...")
    with tqdm(total=total_timesteps, desc="Perfect Knowledge Training") as pbar:
        def callback(locals, globals):
            pbar.update(model.n_steps)
            return True
        
        model.learn(total_timesteps=total_timesteps, callback=callback)
    
    print(f"Perfect knowledge training completed!")
    return model


def train_static_agent(jobs_data, machine_list, total_timesteps=300000, reward_mode="makespan_increment"):
    """Train a static RL agent where all jobs are available at t=0."""
    print(f"\n--- Training Static RL Agent on {len(jobs_data)} jobs ---")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_static_env():
        env = StaticFJSPEnv(jobs_data, machine_list, reward_mode=reward_mode)
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_static_env])
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,  # Minimal output
        learning_rate=1e-4,
        n_steps=4096,          # Increased for larger job set
        batch_size=512,        # Increased for larger job set
        n_epochs=10,           # More epochs for complex patterns
        gamma=1,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=[256, 128, 64],  # Smaller network for 7-job dataset
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"Training Static RL for {total_timesteps:,} timesteps...")
    
    # Train with tqdm progress bar
    start_time = time.time()
    
    with tqdm(total=total_timesteps, desc="Static RL", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} timesteps [{elapsed}<{remaining}]') as pbar:
        
        # Break training into chunks for progress updates
        chunk_size = total_timesteps // 30  # 30 chunks
        remaining_timesteps = total_timesteps
        
        while remaining_timesteps > 0:
            current_chunk = min(chunk_size, remaining_timesteps)
            model.learn(total_timesteps=current_chunk)
            pbar.update(current_chunk)
            remaining_timesteps -= current_chunk
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Static RL training completed in {training_time:.1f}s!")
    
    return model

class PerfectKnowledgeFJSPEnv(gym.Env):
    """
    Dynamic FJSP Environment with PERFECT knowledge of specific arrival times.
    
    This environment is based on the working DynamicFJSPEnv from test3_backup.py
    but uses deterministic arrival times instead of dynamic arrivals.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment", seed=None):
        super().__init__()
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        self.reward_mode = reward_mode
        self.job_arrival_times = arrival_times.copy()
        
        # Action space - similar to test3_backup.py
        self.action_space = spaces.Discrete(
            min(self.num_jobs * self.max_ops_per_job * len(self.machines), 1000)
        )
        
        # Observation space - similar to test3_backup.py
        obs_size = (
            len(self.machines) +                    # Machine availability
            self.num_jobs * self.max_ops_per_job +  # Operation completion
            self.num_jobs +                         # Job progress  
            self.num_jobs +                         # Job arrival status
            1                                       # Current makespan
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        """Reset environment - based on test3_backup.py approach."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize state similar to test3_backup.py DynamicFJSPEnv
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        self.current_makespan = 0.0
        self.operations_scheduled = 0
        self.episode_step = 0
        self.max_episode_steps = self.total_operations * 2
        
        # Handle job arrivals at time 0 - deterministic
        self.arrived_jobs = {
            job_id for job_id, arrival_time in self.job_arrival_times.items()
            if arrival_time <= 0
        }
        
        return self._get_observation(), {}

    def _decode_action(self, action):
        """Decode action - same as test3_backup.py"""
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
        """Check if action is valid - same as test3_backup.py"""
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
        """Generate action masks - same as test3_backup.py"""
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
                    action = (job_idx * self.max_ops_per_job * len(self.machines) + 
                             next_op_idx * len(self.machines) + machine_idx)
                    if action < self.action_space.n:
                        mask[action] = True
                        valid_action_count += 1
        
        if valid_action_count == 0:
            mask.fill(True)
            
        return mask

    def step(self, action):
        """Step function - improved version based on test3_backup.py"""
        self.episode_step += 1
        
        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        job_idx, op_idx, machine_idx = self._decode_action(action)

        # Use softer invalid action handling like test3_backup.py
        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            # Give a negative reward but don't terminate - helps learning
            return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        
        # Calculate timing using test3_backup.py approach
        machine_available_time = self.machine_next_free.get(machine, 0.0)
        job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                         else self.job_arrival_times.get(job_id, 0.0))
        
        start_time = max(machine_available_time, job_ready_time)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time

        # Update state
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
        
        # Calculate reward using test3_backup.py style
        idle_time = max(0, start_time - machine_available_time)
        reward = self._calculate_reward(proc_time, idle_time, terminated, previous_makespan, self.current_makespan)
        
        info = {"makespan": self.current_makespan}
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
        """
        Simplified observation similar to successful PerfectKnowledgeFJSPEnv.
        Focus on essential information without over-complication.
        """
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
        
        # Job arrival status (simple binary: arrived or not)
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


def train_dynamic_agent(jobs_data, machine_list, initial_jobs=5, arrival_rate=0.08, total_timesteps=500000, reward_mode="makespan_increment"):
    """
    Train a dynamic RL agent on Poisson job arrivals with EXPANDED DATASET.
    """
    print(f"\n--- Training Dynamic RL Agent on {len(jobs_data)} jobs ---")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_dynamic_env():
        # ENHANCED randomization for better generalization
        # available_jobs = list(jobs_data.keys())
        
        # if isinstance(initial_jobs, int):
        #     # Vary the number of initial jobs (2-4) for more diversity
        #     num_initial = random.randint(2, min(4, len(available_jobs)))
        #     random_initial = random.sample(available_jobs, num_initial)
        # else:
        #     # Still randomize even with fixed initial jobs list
        #     random_initial = random.sample(initial_jobs, len(initial_jobs))
            
        # # Vary arrival rate slightly for robustness (±20%)
        # varied_arrival_rate = arrival_rate * random.uniform(0.8, 1.2)
            
        # env = PoissonDynamicFJSPEnv(
        #     jobs_data, machine_list, 
        #     initial_jobs=random_initial,
        #     arrival_rate=varied_arrival_rate,
        #     reward_mode=reward_mode
        # )
        env = PoissonDynamicFJSPEnv(
            jobs_data, machine_list, 
            initial_jobs=initial_jobs,
            arrival_rate=arrival_rate,
            reward_mode=reward_mode
        )
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_dynamic_env])
    
    # Simplified hyperparameters matching successful environments
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,        # Match PerfectKnowledgeFJSPEnv
        n_steps=2048,              # Match PerfectKnowledgeFJSPEnv  
        batch_size=128,            # Match PerfectKnowledgeFJSPEnv
        n_epochs=10,               # Match PerfectKnowledgeFJSPEnv
        gamma=1,                   # Match PerfectKnowledgeFJSPEnv
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[512, 512, 256],  # Match PerfectKnowledgeFJSPEnv
            activation_fn=torch.nn.ReLU
        )
    )
    
    # Calculate expected training diversity
    avg_episode_length = 25  # Estimated for 7-job problem
    total_episodes = total_timesteps // avg_episode_length
    batches = total_timesteps // model.n_steps
    episodes_per_batch = model.n_steps // avg_episode_length
    
    print(f"Training Dynamic RL for {total_timesteps:,} timesteps...")
    print(f"Using simplified approach matching successful environments")
    
    # Train with progress bar like PerfectKnowledgeFJSPEnv
    start_time = time.time()
    
    with tqdm(total=total_timesteps, desc="Dynamic RL Training") as pbar:
        def callback(locals, globals):
            if hasattr(model, 'num_timesteps'):
                pbar.n = model.num_timesteps
                pbar.refresh()
            return True
        
        model.learn(total_timesteps=total_timesteps, callback=callback)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Dynamic RL training completed in {training_time:.1f}s!")
    
    return model

def fifo_heuristic(jobs_data, machine_list, arrival_times):
    """FIFO (First In First Out) - Process jobs in arrival order."""
    return basic_greedy_scheduler(jobs_data, machine_list, arrival_times)


def spt_heuristic_simple(jobs_data, machine_list, arrival_times):
    """SPT (Shortest Processing Time) - Process shortest operations first."""
    return simple_spt_heuristic(jobs_data, machine_list, arrival_times)


def run_heuristic_comparison(jobs_data, machine_list, arrival_times):
    """Run all heuristics and return the best result."""
    print(f"  Comparing all optimized heuristics...")
    
    # Test all our heuristics
    results = []
    
    spt_makespan, spt_schedule = simple_spt_heuristic(jobs_data, machine_list, arrival_times)
    results.append(("Simple SPT", spt_makespan, spt_schedule))
    
    greedy_makespan, greedy_schedule = basic_greedy_scheduler(jobs_data, machine_list, arrival_times)
    results.append(("Basic Greedy", greedy_makespan, greedy_schedule))
    
    opt_makespan, opt_schedule = optimized_spt_scheduler(jobs_data, machine_list, arrival_times)
    results.append(("Optimized SPT", opt_makespan, opt_schedule))
    
    ec_makespan, ec_schedule = earliest_completion_scheduler(jobs_data, machine_list, arrival_times)
    results.append(("Earliest Completion", ec_makespan, ec_schedule))
    
    # Find the best
    best_result = min(results, key=lambda x: x[1])
    best_name, best_makespan, best_schedule = best_result
    
    print(f"  Heuristic comparison results:")
    for name, makespan, _ in sorted(results, key=lambda x: x[1]):
        status = "✅ BEST" if makespan == best_makespan else ""
        print(f"    {name}: {makespan:.2f} {status}")
    
    print(f"  Selected: {best_name} Heuristic (makespan: {best_makespan:.2f})")
    return best_makespan, best_schedule





def simple_spt_heuristic(jobs_data, machine_list, arrival_times):
    """
    Fixed SPT heuristic using proper list scheduling approach.
    Key insight: Don't advance ALL machines when waiting for arrivals.
    """
    print("\n--- Running Fixed SPT Heuristic ---")
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    schedule = {m: [] for m in machine_list}
    
    operations_scheduled = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    arrived_jobs = {job_id for job_id, arrival in arrival_times.items() if arrival <= 0}
    
    while operations_scheduled < total_operations:
        candidate_operations = []
        
        # Find current simulation time (earliest machine available time)
        current_time = min(machine_next_free.values())
        
        # Update job arrivals based on current time
        for job_id, arrival_time in arrival_times.items():
            if job_id not in arrived_jobs and arrival_time <= current_time:
                arrived_jobs.add(job_id)
        
        # Collect all ready operations
        for job_id in arrived_jobs:
            op_idx = next_operation_for_job[job_id]
            if op_idx < len(jobs_data[job_id]):
                op_data = jobs_data[job_id][op_idx]
                job_ready_time = (operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                else arrival_times[job_id])
                
                # Check if operation is ready (precedence satisfied)
                if job_ready_time <= current_time:
                    for machine_name, proc_time in op_data['proc_times'].items():
                        earliest_start_time = max(machine_next_free[machine_name], job_ready_time)
                        candidate_operations.append((
                            proc_time,           # SPT criterion
                            earliest_start_time,
                            job_id, 
                            op_idx, 
                            machine_name
                        ))
        
        if not candidate_operations:
            # No operations ready - advance time to next event
            next_events = []
            
            # Next job arrival
            for job_id, arrival_time in arrival_times.items():
                if job_id not in arrived_jobs and arrival_time > current_time:
                    next_events.append(arrival_time)
            
            # Next operation becomes ready due to precedence
            for job_id in arrived_jobs:
                op_idx = next_operation_for_job[job_id]
                if op_idx > 0 and op_idx < len(jobs_data[job_id]):
                    ready_time = operation_end_times[job_id][op_idx - 1]
                    if ready_time > current_time:
                        next_events.append(ready_time)
            
            if not next_events:
                break  # No more events
            
            # Advance time to next event (don't advance all machines!)
            next_time = min(next_events)
            # Only advance machines that are free before this time
            for m in machine_list:
                if machine_next_free[m] < next_time:
                    machine_next_free[m] = next_time
            
            continue
        
        # SPT Rule: Select operation with shortest processing time
        selected_op = min(candidate_operations, key=lambda x: x[0])
        proc_time, start_time, job_id, op_idx, machine_name = selected_op
        
        end_time = start_time + proc_time
        
        # Update state
        machine_next_free[machine_name] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled += 1
        
        schedule[machine_name].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"Fixed SPT Makespan: {makespan:.2f}")
    return makespan, schedule


def spt_heuristic_poisson(jobs_data, machine_list, arrival_times):
    """
    Run comparison of different dispatching heuristics and return the best one.
    Properly separates job sequencing rules from machine selection rules.
    """
    print(f"  Testing improved two-stage heuristics with arrival times: {arrival_times}")
    
    # First try the simple SPT (should perform much better)
    simple_spt_makespan, simple_spt_schedule = simple_spt_heuristic(jobs_data, machine_list, arrival_times)
    print(f"  Simple SPT makespan: {simple_spt_makespan:.2f}")

    
    # Also test our other optimized heuristics for comparison
    print(f"  Testing other heuristics for comparison...")
    greedy_makespan, _ = basic_greedy_scheduler(jobs_data, machine_list, arrival_times)
    print(f"  Basic Greedy makespan: {greedy_makespan:.2f}")
    
    opt_spt_makespan, _ = optimized_spt_scheduler(jobs_data, machine_list, arrival_times)
    print(f"  Optimized SPT makespan: {opt_spt_makespan:.2f}")
    
    ec_makespan, _ = earliest_completion_scheduler(jobs_data, machine_list, arrival_times)  
    print(f"  Earliest Completion makespan: {ec_makespan:.2f}")
    
    # Return the best result
    all_results = [
        (simple_spt_makespan, simple_spt_schedule, "Simple SPT"),
        (greedy_makespan, _, "Basic Greedy"),
        (opt_spt_makespan, _, "Optimized SPT"), 
        (ec_makespan, _, "Earliest Completion")
    ]
    
    best_result = min(all_results, key=lambda x: x[0])
    best_makespan, best_schedule, best_name = best_result
    
    print(f"  Best heuristic: {best_name} (makespan: {best_makespan:.2f})")
    return best_makespan, best_schedule


def milp_optimal_scheduler(jobs_data, machine_list, arrival_times):
    """
    MILP approach for optimal dynamic scheduling with perfect knowledge.
    
    This provides the theoretical optimal solution for the perfect knowledge case,
    serving as the benchmark for regret calculation and verification of the
    perfect knowledge RL agent performance.
    
    Args:
        jobs_data: Dictionary of job data with processing times
        machine_list: List of available machines
        arrival_times: Dictionary of exact arrival times for each job
        
    Returns:
        tuple: (optimal_makespan, optimal_schedule) or (float('inf'), empty_schedule) if failed
    """
    import pickle
    import os
    import hashlib
    
    # Create cache key based on problem instance
    problem_key = str(sorted(arrival_times.items())) + str(sorted(jobs_data.items())) + str(sorted(machine_list))
    cache_key = hashlib.md5(problem_key.encode()).hexdigest()
    cache_file = f"milp_cache_{cache_key}.pkl"
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
            print("\n--- Loading MILP Solution from Cache ---")
            print(f"✅ MILP OPTIMAL SOLUTION (CACHED)!")
            print(f"   Optimal Makespan: {cached_result[0]:.2f}")
            return cached_result
        except:
            print("Cache file corrupted, recomputing...")
    
    print("\n--- Running MILP Optimal Scheduler (Perfect Knowledge Benchmark) ---")
    print(f"Jobs: {len(jobs_data)}, Machines: {len(machine_list)}")
    print(f"Arrival times: {arrival_times}")
    
    try:
        prob = LpProblem("PerfectKnowledge_FJSP_Optimal", LpMinimize)
        
        # Generate all operations
        ops = [(j, oi) for j in jobs_data for oi in range(len(jobs_data[j]))]
        BIG_M = 1000  # Large constant for disjunctive constraints

        # Decision variables
        x = LpVariable.dicts("x", (ops, machine_list), cat="Binary")  # Assignment variables
        s = LpVariable.dicts("s", ops, lowBound=0)                    # Start time variables
        c = LpVariable.dicts("c", ops, lowBound=0)                    # Completion time variables
        y = LpVariable.dicts("y", (ops, ops, machine_list), cat="Binary")  # Sequencing variables
        Cmax = LpVariable("Cmax", lowBound=0)                         # Makespan variable

        # Objective: minimize makespan
        prob += Cmax

        # Constraints
        for j, oi in ops:
            # 1. Assignment constraint: each operation must be assigned to exactly one compatible machine
            compatible_machines = [m for m in machine_list if m in jobs_data[j][oi]['proc_times']]
            prob += lpSum(x[j, oi][m] for m in compatible_machines) == 1
            
            # 2. Completion time definition
            prob += c[j, oi] == s[j, oi] + lpSum(
                x[j, oi][m] * jobs_data[j][oi]['proc_times'][m] 
                for m in compatible_machines
            )
            
            # 3. Precedence constraints within jobs
            if oi > 0:
                prob += s[j, oi] >= c[j, oi - 1]
            else:
                # 4. Arrival time constraint for first operation of each job
                prob += s[j, oi] >= arrival_times.get(j, 0)
            
            # 5. Makespan definition
            prob += Cmax >= c[j, oi]

        # 6. Disjunctive constraints for machine capacity
        for m in machine_list:
            ops_on_m = [op for op in ops if m in jobs_data[op[0]][op[1]]['proc_times']]
            for i in range(len(ops_on_m)):
                for k in range(i + 1, len(ops_on_m)):
                    op1, op2 = ops_on_m[i], ops_on_m[k]
                    # Either op1 before op2 or op2 before op1 (if both assigned to machine m)
                    prob += s[op1] >= c[op2] - BIG_M * (1 - y[op1][op2][m]) - BIG_M * (2 - x[op1][m] - x[op2][m])
                    prob += s[op2] >= c[op1] - BIG_M * y[op1][op2][m] - BIG_M * (2 - x[op1][m] - x[op2][m])

        # Solve with time limit
        print("Solving MILP optimization problem...")
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=300))  # 5-minute time limit
        
        # Extract solution
        schedule = {m: [] for m in machine_list}
        
        if prob.status == 1 and Cmax.varValue is not None:  # Optimal solution found
            optimal_makespan = Cmax.varValue
            
            # Extract schedule from solution
            for (j, oi), m in ((op, m) for op in ops for m in machine_list):
                if m in jobs_data[j][oi]['proc_times'] and x[j, oi][m].varValue > 0.5:
                    start_time = s[j, oi].varValue
                    end_time = c[j, oi].varValue
                    schedule[m].append((f"J{j}-O{oi+1}", start_time, end_time))
            
            # Sort operations by start time for each machine
            for m in machine_list:
                schedule[m].sort(key=lambda x: x[1])
            
            print(f"✅ MILP OPTIMAL SOLUTION FOUND!")
            print(f"   Optimal Makespan: {optimal_makespan:.2f}")
            print(f"   This represents the THEORETICAL BEST possible performance")
            print(f"   with perfect knowledge of arrival times: {arrival_times}")
            
            # Cache the result for future runs
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump((optimal_makespan, schedule), f)
                print(f"   Result cached to: {cache_file}")
            except:
                print("   Warning: Could not cache result")
            
            return optimal_makespan, schedule
            
        else:
            print(f"❌ MILP solver failed to find optimal solution (status: {prob.status})")
            print("   Possible reasons: problem too complex, time limit exceeded, or infeasible")
            return float('inf'), schedule
            
    except Exception as e:
        print(f"❌ MILP solver error: {e}")
        return float('inf'), {m: [] for m in machine_list}

def basic_greedy_scheduler(jobs_data, machine_list, arrival_times):
    """
    Basic greedy scheduler that just tries to minimize completion time.
    Simple baseline to verify our data and approach.
    """
    print("\n--- Running Basic Greedy Scheduler ---")
    
    # Create a list of all operations with their job/operation indices
    all_operations = []
    for job_id, operations in jobs_data.items():
        for op_idx, op_data in enumerate(operations):
            all_operations.append((job_id, op_idx, op_data))
    
    # Sort by arrival time, then by operation index (FIFO with precedence)
    all_operations.sort(key=lambda x: (arrival_times[x[0]], x[1]))
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    schedule = {m: [] for m in machine_list}
    
    for job_id, op_idx, op_data in all_operations:
        # Wait for job arrival
        job_arrival_time = arrival_times[job_id]
        
        # Wait for precedence (previous operation to complete)
        precedence_time = (operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                          else job_arrival_time)
        
        # Find the best machine (earliest completion time)
        best_machine = None
        best_completion_time = float('inf')
        
        for machine, proc_time in op_data['proc_times'].items():
            start_time = max(machine_next_free[machine], precedence_time)
            completion_time = start_time + proc_time
            
            if completion_time < best_completion_time:
                best_completion_time = completion_time
                best_machine = machine
                best_start_time = start_time
        
        # Schedule on best machine
        machine_next_free[best_machine] = best_completion_time
        operation_end_times[job_id][op_idx] = best_completion_time
        
        schedule[best_machine].append((f"J{job_id}-O{op_idx+1}", best_start_time, best_completion_time))
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"Basic Greedy Makespan: {makespan:.2f}")
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
                job_op = op_data[0]
                start_time = op_data[1]
                end_time = op_data[2]
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

def optimized_spt_scheduler(jobs_data, machine_list, arrival_times):
    """
    Optimized SPT scheduler with load balancing.
    Key improvements:
    1. When selecting machines, consider both processing time AND machine workload
    2. Use a weighted score: 0.7 * processing_time + 0.3 * machine_finish_time
    3. This balances SPT rule with load balancing
    """
    print("\n--- Running Optimized SPT with Load Balancing ---")
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    schedule = {m: [] for m in machine_list}
    
    operations_scheduled = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    arrived_jobs = {job_id for job_id, arrival in arrival_times.items() if arrival <= 0}
    
    while operations_scheduled < total_operations:
        # Find current simulation time
        current_time = min(machine_next_free.values())
        
        # Update job arrivals
        for job_id, arrival_time in arrival_times.items():
            if job_id not in arrived_jobs and arrival_time <= current_time:
                arrived_jobs.add(job_id)
        
        # Collect all ready operations
        ready_operations = []
        for job_id in arrived_jobs:
            op_idx = next_operation_for_job[job_id]
            if op_idx < len(jobs_data[job_id]):
                op_data = jobs_data[job_id][op_idx]
                job_ready_time = (operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                else arrival_times[job_id])
                
                if job_ready_time <= current_time:
                    ready_operations.append((job_id, op_idx, op_data, job_ready_time))
        
        if not ready_operations:
            # Advance time to next event
            next_events = []
            
            for job_id, arrival_time in arrival_times.items():
                if job_id not in arrived_jobs and arrival_time > current_time:
                    next_events.append(arrival_time)
            
            for job_id in arrived_jobs:
                op_idx = next_operation_for_job[job_id]
                if op_idx > 0 and op_idx < len(jobs_data[job_id]):
                    ready_time = operation_end_times[job_id][op_idx - 1]
                    if ready_time > current_time:
                        next_events.append(ready_time)
            
            if not next_events:
                break  # No more events
            
            next_time = min(next_events)
            for m in machine_list:
                if machine_next_free[m] < next_time:
                    machine_next_free[m] = next_time
            continue
        
        # Select operation and machine using optimized scoring
        best_score = float('inf')
        best_assignment = None
        
        for job_id, op_idx, op_data, job_ready_time in ready_operations:
            for machine_name, proc_time in op_data['proc_times'].items():
                machine_available_time = machine_next_free[machine_name]
                start_time = max(current_time, machine_available_time, job_ready_time)
                completion_time = start_time + proc_time
                
                # Optimized scoring: balance processing time with machine finish time
                # This encourages both short operations AND load balancing
                score = 0.7 * proc_time + 0.3 * machine_available_time
                
                if score < best_score:
                    best_score = score
                    best_assignment = (job_id, op_idx, machine_name, proc_time, start_time, completion_time)
        
        if not best_assignment:
            break
        
        job_id, op_idx, machine_name, proc_time, start_time, end_time = best_assignment
        
        # Update state
        machine_next_free[machine_name] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled += 1
        
        schedule[machine_name].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"Optimized SPT Makespan: {makespan:.2f}")
    return makespan, schedule

def earliest_completion_scheduler(jobs_data, machine_list, arrival_times):
    """
    Simplest possible scheduler: For each operation, assign it to the machine
    that can complete it earliest. Process operations in order of readiness.
    """
    print("\n--- Running Earliest Completion Time Scheduler ---")
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    schedule = {m: [] for m in machine_list}
    
    operations_scheduled = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    
    while operations_scheduled < total_operations:
        # Find the earliest operation that can be scheduled
        best_completion_time = float('inf')
        best_assignment = None
        
        for job_id, operations in jobs_data.items():
            op_idx = next_operation_for_job[job_id]
            if op_idx < len(operations):
                op_data = operations[op_idx]
                
                # Check if job has arrived
                job_arrival_time = arrival_times[job_id]
                
                # Check precedence constraint
                precedence_time = (operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                                 else job_arrival_time)
                
                # Find best machine for this operation
                for machine_name, proc_time in op_data['proc_times'].items():
                    earliest_start = max(machine_next_free[machine_name], precedence_time)
                    completion_time = earliest_start + proc_time
                    
                    if completion_time < best_completion_time:
                        best_completion_time = completion_time
                        best_assignment = (job_id, op_idx, machine_name, proc_time, earliest_start, completion_time)
        
        if not best_assignment:
            break
        
        job_id, op_idx, machine_name, proc_time, start_time, end_time = best_assignment
        
        # Update state
        machine_next_free[machine_name] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled += 1
        
        schedule[machine_name].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"Earliest Completion Makespan: {makespan:.2f}")
    return makespan, schedule

def generate_test_scenarios(jobs_data, initial_jobs, arrival_rate, num_scenarios=5):
    """Generate test scenarios with Poisson arrivals for evaluation"""
    scenarios = []
    
    for i in range(num_scenarios):
        # Set seed for reproducibility
        np.random.seed(42 + i)
        
        arrival_times = {}
        
        # Initial jobs arrive at t=0
        if isinstance(initial_jobs, list):
            for job_id in initial_jobs:
                arrival_times[job_id] = 0.0
        else:
            # If initial_jobs is an integer, use first N jobs
            all_jobs = list(jobs_data.keys())
            for j in range(min(initial_jobs, len(all_jobs))):
                arrival_times[all_jobs[j]] = 0.0
        
        # Generate Poisson arrivals for remaining jobs
        remaining_jobs = [j for j in jobs_data.keys() if j not in arrival_times]
        
        current_time = 0.0
        for job_id in remaining_jobs:
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival_time
            arrival_times[job_id] = round(current_time)
        
        scenarios.append({
            'scenario_id': i + 1,
            'arrival_times': arrival_times,
            'initial_jobs': initial_jobs,
            'arrival_rate': arrival_rate
        })
    
    return scenarios

def analyze_training_arrival_distribution():
    """Analyze the arrival time distribution during dynamic RL training"""
    global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
    
    if not TRAINING_ARRIVAL_TIMES:
        print("No training arrival data collected")
        return
    
    print(f"Analyzed {len(TRAINING_ARRIVAL_TIMES)} arrival events over {TRAINING_EPISODE_COUNT} episodes")
    print(f"Average arrivals per episode: {len(TRAINING_ARRIVAL_TIMES) / max(TRAINING_EPISODE_COUNT, 1):.1f}")
    
    if len(TRAINING_ARRIVAL_TIMES) >= 10:
        arrival_array = np.array(TRAINING_ARRIVAL_TIMES)
        print(f"Arrival time statistics:")
        print(f"  Mean: {np.mean(arrival_array):.2f}")
        print(f"  Std: {np.std(arrival_array):.2f}")
        print(f"  Range: {np.min(arrival_array):.2f} - {np.max(arrival_array):.2f}")

def evaluate_perfect_knowledge_on_scenario(model, jobs_data, machine_list, arrival_times):
    """Evaluate perfect knowledge RL agent on a specific scenario"""
    env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times)
    env = ActionMasker(env, mask_fn)
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        if truncated:
            break
    
    return info.get('makespan', env.current_makespan), env.schedule

def evaluate_dynamic_on_dynamic(model, jobs_data, machine_list, arrival_times):
    """Evaluate dynamic RL agent on a dynamic scenario"""
    # Create initial jobs list from arrival times
    initial_jobs = [job_id for job_id, arr_time in arrival_times.items() if arr_time == 0]
    
    env = PoissonDynamicFJSPEnv(jobs_data, machine_list, initial_jobs=initial_jobs, arrival_rate=0.5)
    # Override arrival times with specific scenario
    env.job_arrival_times = arrival_times.copy()
    env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    env = ActionMasker(env, mask_fn)
    
    obs, _ = env.reset()
    # Override again after reset
    env.job_arrival_times = arrival_times.copy()
    env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    
    done = False
    
    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        if truncated:
            break
    
    return info.get('makespan', env.current_makespan), env.schedule

def evaluate_static_on_dynamic(model, jobs_data, machine_list, arrival_times):
    """Evaluate static RL agent on a dynamic scenario (challenging case)"""
    # Create a modified environment that reveals jobs as they arrive
    # but the agent was trained assuming all jobs at t=0
    
    env = StaticFJSPEnv(jobs_data, machine_list)
    
    # Manually handle arrivals during evaluation
    machine_next_free = {m: 0.0 for m in machine_list}
    schedule = {m: [] for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation = {job_id: 0 for job_id in jobs_data}
    
    # Process operations respecting arrival times
    operations_scheduled = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    
    current_time = 0.0
    
    while operations_scheduled < total_operations:
        # Update arrivals
        for job_id, arr_time in arrival_times.items():
            if job_id not in arrived_jobs and arr_time <= current_time:
                arrived_jobs.add(job_id)
        
        # Create a temporary environment state for the static agent
        temp_env = StaticFJSPEnv(jobs_data, machine_list)
        temp_env.arrived_jobs = arrived_jobs.copy()
        temp_env.next_operation = next_operation.copy()
        temp_env.machine_next_free = machine_next_free.copy()
        temp_env.operation_end_times = operation_end_times.copy()
        temp_env.operations_scheduled = operations_scheduled
        temp_env.current_makespan = current_time
        temp_env = ActionMasker(temp_env, mask_fn)
        
        obs = temp_env._get_observation()
        action_masks = temp_env.action_masks()
        
        if not np.any(action_masks):
            # No valid actions, advance time
            next_arrival = min([arr_time for job_id, arr_time in arrival_times.items() 
                              if job_id not in arrived_jobs and arr_time > current_time], 
                             default=float('inf'))
            if next_arrival != float('inf'):
                current_time = next_arrival
                continue
            else:
                break
        
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        # Execute action manually
        job_idx, op_idx, machine_idx = temp_env._decode_action(action)
        
        if temp_env._is_valid_action(job_idx, op_idx, machine_idx):
            job_id = temp_env.job_ids[job_idx]
            machine = temp_env.machines[machine_idx]
            
            machine_available_time = machine_next_free.get(machine, 0.0)
            job_ready_time = (operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                             else arrival_times.get(job_id, 0.0))
            
            start_time = max(current_time, machine_available_time, job_ready_time)
            proc_time = jobs_data[job_id][op_idx]['proc_times'][machine]
            end_time = start_time + proc_time
            
            # Update state
            machine_next_free[machine] = end_time
            operation_end_times[job_id][op_idx] = end_time
            next_operation[job_id] += 1
            operations_scheduled += 1
            current_time = max(current_time, end_time)
            
            schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        else:
            # Invalid action, advance time
            current_time += 1
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    return makespan, schedule

def evaluate_static_on_static(model, jobs_data, machine_list):
    """Evaluate static RL agent on a static scenario (all jobs at t=0)"""
    env = StaticFJSPEnv(jobs_data, machine_list)
    env = ActionMasker(env, mask_fn)
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        if truncated:
            break
    
    return info.get('makespan', env.current_makespan), env.schedule
