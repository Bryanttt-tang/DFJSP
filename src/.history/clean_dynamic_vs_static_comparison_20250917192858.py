
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
        previous_makespan = self.current_makespan
        self.current_makespan = max(self.current_makespan, end_time)
        
        # Check for newly arrived jobs (deterministic)
        newly_arrived = {
            j_id for j_id, arrival in self.job_arrival_times.items()
            if previous_makespan < arrival <= self.current_makespan
        }
        self.arrived_jobs.update(newly_arrived)

        # Record in schedule
        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        # Check termination
        terminated = self.operations_scheduled >= self.total_operations
        
        # Calculate reward using test3_backup.py style
        idle_time = max(0, start_time - machine_available_time)
        reward = self._calculate_reward(proc_time, idle_time, terminated, previous_makespan, self.current_makespan)
        
        info = {"makespan": self.current_makespan}
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan=None, current_makespan=None):
        """Reward calculation based on test3_backup.py approach"""
        if self.reward_mode == "makespan_increment":
            # R(s_t, a_t) = E(t) - E(t+1) = negative increment in makespan
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment (reward for not increasing makespan)
                
                # Add small completion bonus
                if done:
                    reward += 50.0
                    
                return reward
            else:
                # Fallback if makespan values not provided
                return -proc_time
        else:
            # Improved reward function with better guidance
            reward = 0.0
            
            # Strong positive reward for completing an operation
            reward += 20.0
            
            # Small penalty for processing time (encourage shorter operations)
            reward -= proc_time * 0.1
            
            # Penalty for idle time (encourage efficiency)  
            reward -= idle_time * 1.0
            
            # Large completion bonus
            if done:
                reward += 200.0
                # Bonus for shorter makespan
                if current_makespan and current_makespan > 0:
                    reward += max(0, 500.0 / current_makespan)
            
            return reward
    
    def _get_observation(self):
        """Generate observation - same structure as test3_backup.py"""
        norm_factor = max(self.current_makespan, 1.0)
        obs = []
        
        # Machine availability
        for m in self.machines:
            obs.append(self.machine_next_free[m] / norm_factor)
        
        # Operation completion status
        for job_id in self.job_ids:
            ops_status = self.completed_ops[job_id]
            while len(ops_status) < self.max_ops_per_job:
                ops_status.append(True)  # Pad with completed
            for status in ops_status[:self.max_ops_per_job]:
                obs.append(1.0 if status else 0.0)
        
        # Job progress
        for job_id in self.job_ids:
            completed = sum(self.completed_ops[job_id])
            total = len(self.jobs[job_id])
            obs.append(completed / max(1, total))
            
        # Job arrival status
        for job_id in self.job_ids:
            obs.append(1.0 if job_id in self.arrived_jobs else 0.0)
            
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
    
    # OPTIMIZED hyperparameters for dynamic scheduling
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=1e-4,        # Lower LR for more stable learning
        n_steps=4096,              # More steps for better experience collection
        batch_size=256,            # Larger batch for stable updates
        n_epochs=15,               # More epochs to learn complex patterns
        gamma=0.99,                # Standard discount factor for long-term planning
        gae_lambda=0.95,
        clip_range=0.15,           # Slightly tighter clipping for stability
        ent_coef=0.02,             # Higher entropy for better exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[1024, 512, 256, 128],  # Deeper network for complex dynamics
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

def generate_test_scenarios(jobs_data, initial_jobs=[0, 1, 2, 3, 4], arrival_rate=0.08, num_scenarios=10):
    """
    Generate diverse test scenarios with expanded job set.
    """
    print(f"Generating {num_scenarios} test scenarios from {len(jobs_data)} total jobs...")
    
    scenarios = []
    for i in range(num_scenarios):
        np.random.seed(42 + i)  # Fixed but different seeds for reproducibility
        arrival_times = {}
        
        # Use different initial job combinations for diversity
        if i < 5:
            # First 5 scenarios: use provided initial jobs
            current_initial = initial_jobs
        else:
            # Last 5 scenarios: randomize initial jobs for generalization testing
            all_jobs = list(jobs_data.keys())
            current_initial = random.sample(all_jobs, 5)
        
        # Initial jobs arrive at t=0
        for job_id in current_initial:
            arrival_times[job_id] = 0.0
        
        # Generate Poisson arrivals for remaining jobs
        remaining_jobs = [j for j in jobs_data.keys() if j not in current_initial]
        current_time = 0.0
        
        # Vary arrival rate slightly for each scenario to test robustness
        # scenario_arrival_rate = arrival_rate * (0.8 + 0.4 * i / num_scenarios)  # 80% to 120% of base rate
        
        for job_id in remaining_jobs:
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival_time
            
            # Round to nearest integer for simplicity
            integer_arrival_time = round(current_time)
            
            if integer_arrival_time <= 300:  # Extended time horizon for larger job set
                arrival_times[job_id] = float(integer_arrival_time)
            else:
                arrival_times[job_id] = float('inf')  # Won't arrive
        
        scenarios.append({
            'scenario_id': i,
            'arrival_times': arrival_times,
            'initial_jobs': current_initial,
            'arrival_rate':arrival_rate,
            'seed': 100 + i
        })
        
        arrived_jobs = [j for j, t in arrival_times.items() if t < float('inf')]
        print(f"  Scenario {i+1}: {len(arrived_jobs)} jobs, rate={arrival_rate:.3f}")
        print(f"    Initial: {current_initial}")
        print(f"    Arrivals: {len(arrived_jobs) - len(current_initial)} jobs")
    
    return scenarios


def analyze_training_arrival_distribution():
    """
    Analyze and plot the distribution of arrival times during training.
    This helps identify if the dynamic RL is seeing diverse enough scenarios.
    """
    global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
    
    if not TRAINING_ARRIVAL_TIMES:
        print("No arrival times recorded during training!")
        return
    
    print(f"\n=== TRAINING ARRIVAL DISTRIBUTION ANALYSIS ===")
    print(f"Total episodes: {TRAINING_EPISODE_COUNT}")
    print(f"Total dynamic arrivals recorded: {len(TRAINING_ARRIVAL_TIMES)}")
    print(f"Average arrivals per episode: {len(TRAINING_ARRIVAL_TIMES)/max(TRAINING_EPISODE_COUNT,1):.2f}")
    
    # Statistics
    arrival_times = np.array(TRAINING_ARRIVAL_TIMES)
    print(f"Arrival time statistics:")
    print(f"  Min: {np.min(arrival_times):.2f}")
    print(f"  Max: {np.max(arrival_times):.2f}")
    print(f"  Mean: {np.mean(arrival_times):.2f}")
    print(f"  Std: {np.std(arrival_times):.2f}")
    
    # Create distribution plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Histogram of arrival times
    plt.subplot(2, 2, 1)
    plt.hist(arrival_times, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Arrival Time')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Job Arrival Times During Training\n({len(TRAINING_ARRIVAL_TIMES)} arrivals across {TRAINING_EPISODE_COUNT} episodes)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(arrival_times, vert=True)
    plt.ylabel('Arrival Time')
    plt.title('Box Plot of Arrival Times')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative distribution
    plt.subplot(2, 2, 3)
    sorted_arrivals = np.sort(arrival_times)
    y_vals = np.arange(1, len(sorted_arrivals) + 1) / len(sorted_arrivals)
    plt.plot(sorted_arrivals, y_vals, linewidth=2)
    plt.xlabel('Arrival Time')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Arrival Times')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Inter-arrival times
    if len(arrival_times) > 1:
        plt.subplot(2, 2, 4)
        # Group by episodes and calculate inter-arrival times within episodes
        inter_arrivals = []
        episode_arrivals = []
        current_episode_times = []
        
        # Simple approximation: assume arrivals are chronological within batches
        sorted_times = np.sort(arrival_times)
        for i in range(1, len(sorted_times)):
            inter_arrival = sorted_times[i] - sorted_times[i-1]
            if inter_arrival > 0 and inter_arrival < 100:  # Filter reasonable inter-arrivals
                inter_arrivals.append(inter_arrival)
        
        if inter_arrivals:
            plt.hist(inter_arrivals, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Inter-arrival Time')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Inter-arrival Times\n(Mean: {np.mean(inter_arrivals):.2f})')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_arrival_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Check if distribution is diverse enough
    unique_times = len(np.unique(np.round(arrival_times, 1)))
    print(f"\nDiversity Analysis:")
    print(f"  Unique arrival times (rounded to 0.1): {unique_times}")
    print(f"  Time span: {np.max(arrival_times) - np.min(arrival_times):.2f}")
    
    if unique_times < 20:
        print("⚠️  WARNING: Low diversity in arrival times may limit learning")
    else:
        print("✅ Good diversity in arrival times")


def create_perfect_knowledge_scenario(base_scenario):
    """
    Create a scenario where dynamic RL has PERFECT knowledge of arrival times.
    This simulates the advantage of knowing exactly when jobs will arrive,
    rather than just knowing the arrival distribution (Poisson rate).
    
    Args:
        base_scenario: A Poisson-generated scenario with specific arrival times
    
    Returns:
        A scenario where the environment can train with perfect arrival knowledge
    """
    return {
        'scenario_id': f'perfect_knowledge_{base_scenario["scenario_id"]}',
        'arrival_times': base_scenario['arrival_times'].copy(),
        'initial_jobs': base_scenario['initial_jobs'].copy(),
        'arrival_rate': None,  # Not stochastic - exact times known
        'seed': base_scenario['seed'],
        'is_perfect_knowledge': True
    }


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

def evaluate_static_on_dynamic(static_model, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment"):
    """Evaluate static model on dynamic scenario with actual arrival times."""
    print(f"  Static RL evaluation on dynamic scenario (arrival times: {arrival_times})...")
    
    # Create PerfectKnowledgeFJSPEnv to properly handle arrival times for static agent
    test_env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times, reward_mode=reward_mode)
    test_env = ActionMasker(test_env, mask_fn)
    
    obs, _ = test_env.reset()
    
    # Run evaluation
    done = False
    step_count = 0
    max_steps = len(jobs_data) * max(len(ops) for ops in jobs_data.values()) * 3
    
    while not done and step_count < max_steps:
        action_masks = test_env.action_masks()
        if not np.any(action_masks):
            print(f"    No valid actions at step {step_count}")
            break
            
        action, _ = static_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if step_count % 15 == 0:
            print(f"    Step {step_count}: current_makespan = {test_env.env.current_makespan:.2f}")
    
    makespan = test_env.env.current_makespan
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    for machine_ops in test_env.env.schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    print(f"  Static RL on dynamic scenario scheduled jobs: {sorted(scheduled_jobs)}")
    
    return makespan, test_env.env.schedule


def evaluate_static_on_static(static_model, jobs_data, machine_list, reward_mode="makespan_increment"):
    """Evaluate static model on static scenario (all jobs at t=0)."""
    print(f"  Static RL evaluation on static scenario (all jobs at t=0)...")
    
    # Create static environment for evaluation (all jobs at t=0)
    test_env = StaticFJSPEnv(jobs_data, machine_list, reward_mode=reward_mode)
    test_env = ActionMasker(test_env, mask_fn)
    
    obs, _ = test_env.reset()
    
    # Run evaluation
    done = False
    step_count = 0
    max_steps = len(jobs_data) * max(len(ops) for ops in jobs_data.values()) * 2
    
    while not done and step_count < max_steps:
        action_masks = test_env.action_masks()
        if not np.any(action_masks):
            print(f"    No valid actions at step {step_count}")
            break
            
        action, _ = static_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"    Step {step_count}: current_makespan = {test_env.env.current_makespan:.2f}")
    
    makespan = test_env.env.current_makespan
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    for machine_ops in test_env.env.schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    print(f"  Static RL on static scenario scheduled jobs: {sorted(scheduled_jobs)}")
    
    return makespan, test_env.env.schedule

def evaluate_dynamic_on_dynamic(dynamic_model, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment" ):
    """Evaluate dynamic model on dynamic scenario."""
    print(f"  Dynamic RL using arrival times: {arrival_times}")
    
    test_env = PoissonDynamicFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=[k for k, v in arrival_times.items() if v == 0],
        arrival_rate=0.2,
        reward_mode=reward_mode
    )
    
    # Create ActionMasker wrapper
    test_env = ActionMasker(test_env, mask_fn)
    
    # Override with fixed arrival times
    test_env.env.job_arrival_times = arrival_times.copy()
    
    obs, _ = test_env.reset()
    
    # Re-override after reset
    test_env.env.job_arrival_times = arrival_times.copy()
    test_env.env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    
    print(f"  Dynamic RL initial jobs: {test_env.env.arrived_jobs}")
    
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
    
    makespan = test_env.env.current_makespan
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    for machine_ops in test_env.env.schedule.values():
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
    
    return makespan, test_env.env.schedule


def evaluate_perfect_knowledge_on_scenario(perfect_model, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment"):
    """Evaluate perfect knowledge agent using the simple deterministic environment."""
    print(f"  Perfect Knowledge RL evaluation (deterministic environment)...")
    
    # Create the corrected PerfectKnowledgeFJSPEnv for evaluation
    test_env = PerfectKnowledgeFJSPEnv(
        jobs_data, machine_list, 
        arrival_times=arrival_times,
        reward_mode=reward_mode
    )
    test_env = ActionMasker(test_env, mask_fn)
    
    # Run evaluation
    obs, _ = test_env.reset()
    done = False
    step_count = 0
    max_steps = len(jobs_data) * max(len(ops) for ops in jobs_data.values()) * 2
    
    while not done and step_count < max_steps:
        action_masks = test_env.action_masks()
        if not np.any(action_masks):
            print(f"    No valid actions at step {step_count}")
            break
            
        action, _ = perfect_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"    Step {step_count}: current_makespan = {test_env.env.current_makespan:.2f}")
    
    makespan = test_env.env.current_makespan
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    for machine_ops in test_env.env.schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    print(f"  Perfect Knowledge RL scheduled jobs: {sorted(scheduled_jobs)}")
    
    return makespan, test_env.env.schedule





def simple_list_scheduling(jobs_data, machine_list, arrival_times, rule):
    """
    Correct list scheduling implementation for FJSP with proper dispatching rules.
    """
    machine_next_free = {m: 0.0 for m in machine_list}
    job_next_op = {job_id: 0 for job_id in jobs_data.keys()}
    job_op_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data.keys()}
    schedule = {m: [] for m in machine_list}
    
    completed_operations = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    sim_time = 0.0
    
    while completed_operations < total_operations:
        # Find ready operations
        ready_operations = []
        
        for job_id in jobs_data.keys():
            if job_next_op[job_id] < len(jobs_data[job_id]):  # Job not finished
                op_idx = job_next_op[job_id]
                
                # Check if job has arrived and previous operation is complete
                job_ready_time = arrival_times[job_id]
                if op_idx > 0:
                    job_ready_time = max(job_ready_time, job_op_end_times[job_id][op_idx - 1])
                
                if sim_time >= job_ready_time:
                    op_data = jobs_data[job_id][op_idx]
                    
                    # Find best machine assignment (SPT for machine selection)
                    best_machine = min(op_data['proc_times'].keys(), 
                                     key=lambda m: op_data['proc_times'][m])
                    proc_time = op_data['proc_times'][best_machine]
                    
                    ready_operations.append({
                        'job_id': job_id,
                        'op_idx': op_idx,
                        'machine': best_machine,
                        'proc_time': proc_time,
                        'arrival_time': arrival_times[job_id],
                        'job_ready_time': job_ready_time
                    })
        
        if not ready_operations:
            # Advance time to next event
            next_time = float('inf')
            for job_id in jobs_data.keys():
                if job_next_op[job_id] < len(jobs_data[job_id]):
                    op_idx = job_next_op[job_id]
                    job_ready_time = arrival_times[job_id]
                    if op_idx > 0:
                        job_ready_time = max(job_ready_time, job_op_end_times[job_id][op_idx - 1])
                    next_time = min(next_time, job_ready_time)
            
            if next_time == float('inf'):
                break
            sim_time = next_time
            continue
        
        # Select operation based on dispatching rule
        if rule == "FIFO":
            selected_op = min(ready_operations, key=lambda x: (x['arrival_time'], x['job_id'], x['op_idx']))
        elif rule == "LIFO":
            selected_op = max(ready_operations, key=lambda x: (x['arrival_time'], x['job_id'], x['op_idx']))
        elif rule == "SPT":
            selected_op = min(ready_operations, key=lambda x: (x['proc_time'], x['arrival_time'], x['job_id']))
        elif rule == "LPT":
            selected_op = max(ready_operations, key=lambda x: (x['proc_time'], -x['arrival_time'], -x['job_id']))
        elif rule == "EDD":
            def due_date(op):
                total_work = sum(min(jobs_data[op['job_id']][i]['proc_times'].values()) 
                               for i in range(len(jobs_data[op['job_id']])))
                return op['arrival_time'] + total_work * 1.5
            selected_op = min(ready_operations, key=lambda x: (due_date(x), x['arrival_time'], x['job_id']))
        else:
            selected_op = ready_operations[0]  # Default to first
        
        # Schedule the selected operation
        job_id = selected_op['job_id']
        op_idx = selected_op['op_idx']
        machine = selected_op['machine']
        proc_time = selected_op['proc_time']
        
        # Calculate start time
        machine_avail = machine_next_free[machine]
        job_ready = selected_op['job_ready_time']
        start_time = max(sim_time, machine_avail, job_ready)
        end_time = start_time + proc_time
        
        # Update state
        machine_next_free[machine] = end_time
        job_op_end_times[job_id][op_idx] = end_time
        job_next_op[job_id] += 1
        schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        
        completed_operations += 1
        sim_time = start_time  # Move simulation time forward
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    return makespan, schedule


def run_heuristic_comparison(jobs_data, machine_list, arrival_times):
    """
    Compare different dispatching rules and return the best one.
    Tests FIFO, LIFO, SPT, LPT, and EDD heuristics.
    """
    heuristics = {
        'FIFO': lambda ops: fifo_heuristic(jobs_data, machine_list, arrival_times),
        'LIFO': lambda ops: lifo_heuristic(jobs_data, machine_list, arrival_times), 
        'SPT': lambda ops: spt_heuristic_simple(jobs_data, machine_list, arrival_times),
        'LPT': lambda ops: lpt_heuristic(jobs_data, machine_list, arrival_times),
        'EDD': lambda ops: edd_heuristic(jobs_data, machine_list, arrival_times)
    }
    
    results = {}
    for name, heuristic_func in heuristics.items():
        try:
            makespan, schedule = heuristic_func(None)
            results[name] = (makespan, schedule)
            print(f"    {name} completed with makespan: {makespan:.2f}")
        except Exception as e:
            print(f"    {name} failed: {e}")
            results[name] = (float('inf'), {})
    
    # Find best heuristic
    valid_results = {k: v for k, v in results.items() if v[0] != float('inf')}
    if not valid_results:
        print("    All heuristics failed! Using fallback.")
        return 999.0, {m: [] for m in machine_list}
    
    best_name = min(valid_results.keys(), key=lambda k: valid_results[k][0])
    best_makespan, best_schedule = valid_results[best_name]
    
    print(f"  Heuristic comparison results:")
    for name, (makespan, _) in results.items():
        if makespan == float('inf'):
            print(f"    {name}: FAILED")
        else:
            status = "✅ BEST" if name == best_name else ""
            print(f"    {name}: {makespan:.2f} {status}")
    
    print(f"  Selected: {best_name} Heuristic (makespan: {best_makespan:.2f})")
    return best_makespan, best_schedule








def fifo_heuristic(jobs_data, machine_list, arrival_times):
    """FIFO (First In First Out) - Process jobs in arrival order."""
    return simple_list_scheduling(jobs_data, machine_list, arrival_times, "FIFO")


def lifo_heuristic(jobs_data, machine_list, arrival_times):
    """LIFO (Last In First Out) - Process newest jobs first.""" 
    return simple_list_scheduling(jobs_data, machine_list, arrival_times, "LIFO")


def spt_heuristic_simple(jobs_data, machine_list, arrival_times):
    """SPT (Shortest Processing Time) - Process shortest operations first."""
    return simple_list_scheduling(jobs_data, machine_list, arrival_times, "SPT")


def lpt_heuristic(jobs_data, machine_list, arrival_times): 
    """LPT (Longest Processing Time) - Process longest operations first."""
    return simple_list_scheduling(jobs_data, machine_list, arrival_times, "LPT")


def edd_heuristic(jobs_data, machine_list, arrival_times):
    """EDD (Earliest Due Date) - Simple version using job completion time estimates."""
    return simple_list_scheduling(jobs_data, machine_list, arrival_times, "EDD")


def _generic_heuristic(jobs_data, machine_list, arrival_times, heuristic_name, priority_func):
    """
    Improved generic heuristic implementation for different dispatching rules.
    
    Args:
        priority_func: Function that takes (job_id, op_idx, machine, proc_time) and returns priority value.
                      Lower values = higher priority.
    """
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    schedule = {m: [] for m in machine_list}
    
    arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    operations_scheduled = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    sim_time = 0.0
    
    while operations_scheduled < total_operations:
        # Update arrivals based on current simulation time
        for job_id, arr_time in arrival_times.items():
            if job_id not in arrived_jobs and arr_time <= sim_time:
                arrived_jobs.add(job_id)
        
        # Collect available operations (with all machine options)
        available_ops = []
        for job_id in arrived_jobs:
            next_op = next_operation_for_job[job_id]
            if next_op < len(jobs_data[job_id]):
                # Check if job is ready (previous operation completed)
                job_ready_time = (operation_end_times[job_id][next_op - 1] 
                                if next_op > 0 else arrival_times[job_id])
                
                if job_ready_time <= sim_time:
                    op_data = jobs_data[job_id][next_op]
                    # Consider ALL compatible machines, not just the best one
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
        
        # Enhanced priority function that considers machine availability
        def enhanced_priority(op_data):
            job_id, op_idx, machine, proc_time = op_data
            
            # Base priority from the heuristic rule
            base_priority = priority_func(op_data)
            
            # Machine availability factor - prefer machines that are available sooner
            machine_available_time = machine_next_free[machine]
            job_ready_time = (operation_end_times[job_id][op_idx - 1] 
                            if op_idx > 0 else arrival_times[job_id])
            
            earliest_start = max(machine_available_time, job_ready_time, sim_time)
            
            # Combine base priority with machine availability
            # For SPT/LPT: mainly processing time, with slight preference for available machines
            # For FIFO/LIFO: mainly arrival order, with machine availability as tiebreaker
            if heuristic_name in ['SPT', 'LPT']:
                # Processing time is primary, machine availability is secondary
                return base_priority + (earliest_start - sim_time) * 0.1
            else:  # FIFO, LIFO
                # Arrival order is primary, processing time and availability are secondary
                return base_priority + proc_time * 0.1 + (earliest_start - sim_time) * 0.05
        
        # Sort operations by enhanced priority (lower is better)
        available_ops.sort(key=enhanced_priority)
        job_id, op_idx, machine, proc_time = available_ops[0]
        
        # Calculate timing
        machine_available_time = machine_next_free[machine]
        job_ready_time = (operation_end_times[job_id][op_idx - 1] 
                         if op_idx > 0 else arrival_times[job_id])
        
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
    
    return makespan, schedule


def spt_heuristic_poisson(jobs_data, machine_list, arrival_times):
    """
    Run comparison of simple dispatching heuristics and return the best one.
    Uses SPT for machine selection and compares FIFO, LIFO, SPT, LPT for job sequencing.
    """
    print(f"  Comparing FIFO, LIFO, SPT, LPT heuristics with arrival times: {arrival_times}")
    return run_heuristic_comparison(jobs_data, machine_list, arrival_times)


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

def diagnose_performance_similarity(perfect_makespan, dynamic_makespan, static_makespan, spt_makespan):
    """
    Diagnose why the different methods might have similar performance.
    This helps identify if the problem setup is creating meaningful differences.
    """
    print(f"\n=== PERFORMANCE SIMILARITY DIAGNOSIS ===")
    
    results = [
        ("Perfect Knowledge RL", perfect_makespan),
        ("Dynamic RL", dynamic_makespan), 
        ("Static RL", static_makespan),
        ("Best Heuristic", spt_makespan)
    ]
    
    makespans = [makespan for _, makespan in results]
    
    # Calculate performance spread
    max_makespan = max(makespans)
    min_makespan = min(makespans)
    spread = max_makespan - min_makespan
    
    # Handle division by zero if min_makespan is 0
    if min_makespan > 0:
        relative_spread = spread / min_makespan * 100
    else:
        # Find the smallest non-zero makespan for relative comparison
        non_zero_makespans = [m for m in makespans if m > 0]
        if non_zero_makespans:
            relative_spread = spread / min(non_zero_makespans) * 100
        else:
            relative_spread = 0.0
    
    print(f"Performance spread: {spread:.2f} time units ({relative_spread:.1f}%)")
    
    if relative_spread < 5:
        print("🔴 ISSUE: Very small performance differences (<5%)")
        print("   Possible causes:")
        print("   - Arrival rate too low (jobs arrive too late to matter)")
        print("   - Test scenario too easy (all methods find similar solutions)")
        print("   - State representation not sufficiently informative")
        print("   - Training not sufficient to learn anticipatory behavior")
    elif relative_spread < 15:
        print("🟡 MODERATE: Small but measurable differences (5-15%)")
        print("   This suggests some advantage but limited differentiation")
    else:
        print("🟢 GOOD: Clear performance differences (>15%)")
        print("   Methods are showing distinct capabilities")
    
    # Check if hierarchy is as expected
    expected_order = perfect_makespan <= dynamic_makespan <= static_makespan
    if expected_order:
        print("✅ Expected performance hierarchy maintained")
    else:
        print("❌ Unexpected performance hierarchy - investigate training issues")
    
    # Recommendations
    print(f"\nRecommendations:")
    if relative_spread < 5:
        print("- Increase arrival rate (try λ=1.0 or higher)")
        print("- Use longer training episodes") 
        print("- Add more complex job structures")
        print("- Increase reward differentiation for anticipatory actions")


def calculate_regret_analysis(optimal_makespan, methods_results):
    """
    Calculate regret (performance gap from optimal) for all methods.
    
    Regret provides a normalized measure of how far each method is from
    the theoretical optimum, helping understand the relative performance
    and the room for improvement.
    
    Args:
        optimal_makespan: The MILP optimal makespan (benchmark)
        methods_results: Dict with method names as keys and makespans as values
        
    Returns:
        dict: Regret analysis results
    """
    print("\n" + "="*60)
    print("REGRET ANALYSIS (Gap from MILP Optimal)")
    print("="*60)
    
    if optimal_makespan == float('inf') or optimal_makespan <= 0:
        print("❌ No valid optimal solution available for regret calculation")
        return None
    
    print(f"📊 MILP Optimal Makespan (Benchmark): {optimal_makespan:.2f}")
    print("-" * 60)
    
    regret_results = {}
    
    # Calculate regret for each method
    for method_name, makespan in methods_results.items():
        if makespan == float('inf'):
            regret_abs = float('inf')
            regret_rel = float('inf')
        else:
            regret_abs = makespan - optimal_makespan  # Absolute regret
            regret_rel = (regret_abs / optimal_makespan) * 100  # Relative regret (%)
        
        regret_results[method_name] = {
            'makespan': makespan,
            'absolute_regret': regret_abs,
            'relative_regret_percent': regret_rel
        }
        
        # Display results with status indicators
        if regret_abs == 0:
            status = "🎯 OPTIMAL"
        elif regret_rel <= 5:
            status = "🟢 EXCELLENT"
        elif regret_rel <= 15:
            status = "🟡 GOOD"
        elif regret_rel <= 30:
            status = "🟠 ACCEPTABLE"
        else:
            status = "🔴 POOR"
        
        print(f"{method_name:25s}: {makespan:6.2f} | Regret: +{regret_abs:5.2f} (+{regret_rel:5.1f}%) {status}")
    
    # Analysis and insights
    print("\n" + "-" * 60)
    print("KEY INSIGHTS:")
    
    # Find best and worst performing methods
    valid_methods = {k: v for k, v in regret_results.items() 
                    if v['absolute_regret'] != float('inf')}
    
    if valid_methods:
        best_method = min(valid_methods.items(), key=lambda x: x[1]['absolute_regret'])
        worst_method = max(valid_methods.items(), key=lambda x: x[1]['absolute_regret'])
        
        print(f"🏆 Best Method: {best_method[0]} (regret: +{best_method[1]['relative_regret_percent']:.1f}%)")
        print(f"⚠️  Worst Method: {worst_method[0]} (regret: +{worst_method[1]['relative_regret_percent']:.1f}%)")
        
        # Calculate performance gap between best and worst
        performance_gap = worst_method[1]['absolute_regret'] - best_method[1]['absolute_regret']
        print(f"📈 Performance Gap: {performance_gap:.2f} time units between best and worst")
        
        # Perfect Knowledge RL validation
        if 'Perfect Knowledge RL' in valid_methods:
            pk_regret = valid_methods['Perfect Knowledge RL']['relative_regret_percent']
            if pk_regret <= 10:
                print(f"✅ Perfect Knowledge RL is performing well (regret: {pk_regret:.1f}%)")
                print("   This validates that the RL agent can effectively use arrival information")
            else:
                print(f"❌ Perfect Knowledge RL has high regret ({pk_regret:.1f}%)")
                print("   Consider: longer training, better hyperparameters, or state representation issues")
    
    return regret_results


def main():
 
    print("=" * 80)
    print("DYNAMIC vs STATIC RL COMPARISON FOR POISSON FJSP")
    print("=" * 80)
    print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
    print("Research Question: Does Dynamic RL outperform Static RL on Poisson arrivals?")
    print("=" * 80)
    arrival_rate = 0.5  # HIGHER arrival rate to create more dynamic scenarios
    # With λ=0.5, expected inter-arrival = 2 time units (faster than most job operations)
    
    # Step 1: Training Setup
    print("\n1. TRAINING SETUP")
    print("-" * 50)
    perfect_timesteps = 50000    # Perfect knowledge needs less training
    dynamic_timesteps = 50000   # Increased for better learning with integer timing  
    static_timesteps = 50000    # Increased for better learning
    
    print(f"Perfect RL: {perfect_timesteps:,} | Dynamic RL: {dynamic_timesteps:,} | Static RL: {static_timesteps:,} timesteps")
    print(f"Arrival rate: {arrival_rate} (expected inter-arrival: {1/arrival_rate:.1f} time units)")

    # Step 2: Generate test scenarios (Poisson arrivals)
    print("\n2. GENERATING TEST SCENARIOS")
    print("-" * 40)
    print("Expected: Dynamic RL (knows arrival distribution) > Static RL (assumes all jobs at t=0)")
    print("Performance should be: Deterministic(~43) > Poisson Dynamic > Static(~50)")
    test_scenarios = generate_test_scenarios(ENHANCED_JOBS_DATA, 
                                           initial_jobs=[0, 1, 2], 
                                           arrival_rate=arrival_rate, 
                                           num_scenarios=5)
    
    # Print all test scenario arrival times
    print("\nALL TEST SCENARIO ARRIVAL TIMES:")
    print("-" * 50)
    for i, scenario in enumerate(test_scenarios):
        print(f"Scenario {i+1}: {scenario['arrival_times']}")
        arrived_jobs = [j for j, t in scenario['arrival_times'].items() if t < float('inf')]
        print(f"  Jobs arriving: {len(arrived_jobs)} ({sorted(arrived_jobs)})")
        print()
    
    # Step 3: Train all three agents
    print("\n3. TRAINING PHASE")
    print("-" * 40)
    
    # Use first test scenario for perfect knowledge training
    first_scenario = test_scenarios[0]
    first_scenario_arrivals = first_scenario['arrival_times']
    print(f"Using scenario 1 for training: {first_scenario_arrivals}")
    
    # Train perfect knowledge RL agent (knows exact arrival times)

    perfect_model = train_perfect_knowledge_agent(ENHANCED_JOBS_DATA, MACHINE_LIST, 
                                                 arrival_times=first_scenario_arrivals, 
                                                 total_timesteps=perfect_timesteps,
                                                 reward_mode="makespan_increment")
    
    # Train dynamic RL agent (knows arrival distribution only)
    dynamic_model = train_dynamic_agent(ENHANCED_JOBS_DATA, MACHINE_LIST, 
                                       initial_jobs=[0, 1, 2], arrival_rate=arrival_rate, 
                                       total_timesteps=dynamic_timesteps,reward_mode="makespan_increment")

    # Train static RL agent (assumes all jobs at t=0)
    static_model = train_static_agent(ENHANCED_JOBS_DATA, MACHINE_LIST, total_timesteps=static_timesteps, 
                                     reward_mode="makespan_increment")

    # Analyze arrival time distribution during training
    print("\n3.5. TRAINING ARRIVAL DISTRIBUTION ANALYSIS")
    analyze_training_arrival_distribution()
    
    # Step 4: Evaluate all methods on the same test scenario
    print("\n4. EVALUATION PHASE")
    print("-" * 40)
    print("Comparing three levels of arrival information:")
    print("1. Perfect Knowledge RL (knows exact arrival times)")
    print("2. Dynamic RL (knows arrival distribution)")  
    print("3. Static RL (assumes all jobs at t=0)")
    print(f"Test scenario arrivals: {first_scenario_arrivals}")
    
    # Perfect Knowledge RL (upper bound)
    print("Evaluating Perfect Knowledge RL...")
    perfect_makespan, perfect_schedule = evaluate_perfect_knowledge_on_scenario(
        perfect_model, ENHANCED_JOBS_DATA, MACHINE_LIST, first_scenario_arrivals)
    
    # Dynamic RL (knows distribution only)
    print("Evaluating Dynamic RL...")
    dynamic_makespan, dynamic_schedule = evaluate_dynamic_on_dynamic(
        dynamic_model, ENHANCED_JOBS_DATA, MACHINE_LIST, first_scenario_arrivals)
    
    # Static RL (no arrival info) - evaluate on both dynamic and static scenarios
    print("Evaluating Static RL on dynamic scenario...")
    static_dynamic_makespan, static_dynamic_schedule = evaluate_static_on_dynamic(
        static_model, ENHANCED_JOBS_DATA, MACHINE_LIST, first_scenario_arrivals)
    
    print("Evaluating Static RL on static scenario (all jobs at t=0)...")
    static_static_makespan, static_static_schedule = evaluate_static_on_static(
        static_model, ENHANCED_JOBS_DATA, MACHINE_LIST)
    
    # Define static arrivals for plotting (all jobs at t=0)
    static_arrivals = {job_id: 0.0 for job_id in ENHANCED_JOBS_DATA.keys()}
    
    # Best Heuristic (from FIFO, LIFO, SPT, LPT comparison)
    print("Evaluating Best Heuristic (comparing FIFO, LIFO, SPT, LPT)...")
    spt_makespan, spt_schedule = spt_heuristic_poisson(ENHANCED_JOBS_DATA, MACHINE_LIST, first_scenario_arrivals)
    
    # MILP Optimal Solution (Benchmark)
    print("Computing MILP Optimal Solution...")
    milp_makespan, milp_schedule = milp_optimal_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, first_scenario_arrivals)
    
    # Step 5: Results Analysis
    print("\n5. RESULTS ANALYSIS")
    print("=" * 60)
    print(f"MILP Optimal              - Makespan: {milp_makespan:.2f} (THEORETICAL BEST)")
    print(f"Perfect Knowledge RL      - Makespan: {perfect_makespan:.2f}")
    print(f"Dynamic RL (Poisson)      - Makespan: {dynamic_makespan:.2f}")  
    print(f"Static RL (on dynamic)    - Makespan: {static_dynamic_makespan:.2f}")
    print(f"Static RL (on static)     - Makespan: {static_static_makespan:.2f}")
    print(f"Best Heuristic            - Makespan: {spt_makespan:.2f}")
    
    print("\nPerformance Ranking:")
    results = [
        ("MILP Optimal", milp_makespan),
        ("Perfect Knowledge RL", perfect_makespan),
        ("Dynamic RL", dynamic_makespan), 
        ("Static RL (dynamic)", static_dynamic_makespan),
        ("Static RL (static)", static_static_makespan),
        ("Best Heuristic", spt_makespan)
    ]
    results.sort(key=lambda x: x[1])
    for i, (method, makespan) in enumerate(results, 1):
        if makespan == float('inf'):
            print(f"{i}. {method}: Failed")
        else:
            print(f"{i}. {method}: {makespan:.2f}")
    
    print(f"\nExpected Performance Hierarchy:")
    print(f"MILP Optimal ≤ Perfect Knowledge ≤ Dynamic RL ≤ Static RL")
    if milp_makespan != float('inf'):
        print(f"Actual: {milp_makespan:.2f} ≤ {perfect_makespan:.2f} ≤ {dynamic_makespan:.2f} ≤ {static_dynamic_makespan:.2f}")
    else:
        print(f"Actual (no MILP): {perfect_makespan:.2f} ≤ {dynamic_makespan:.2f} ≤ {static_dynamic_makespan:.2f}")
    
    # Step 5.5: Regret Analysis (Gap from Optimal)
    if milp_makespan != float('inf'):
        methods_results = {
            "Perfect Knowledge RL": perfect_makespan,
            "Dynamic RL": dynamic_makespan,
            "Static RL (dynamic)": static_dynamic_makespan,
            "Static RL (static)": static_static_makespan,
            "Best Heuristic": spt_makespan
        }
        regret_results = calculate_regret_analysis(milp_makespan, methods_results)
    
    # Validate expected ordering
    if perfect_makespan <= dynamic_makespan <= static_dynamic_makespan:
        print("✅ EXPECTED: Perfect knowledge outperforms distribution knowledge outperforms no knowledge")
    else:
        print("❌ UNEXPECTED: Performance doesn't follow expected hierarchy")
    
    # Diagnose performance similarity issues
    diagnose_performance_similarity(perfect_makespan, dynamic_makespan, static_dynamic_makespan, spt_makespan)
    
    # Performance comparisons
    print("\n6. PERFORMANCE COMPARISON")
    print("-" * 40)
    
    # Perfect Knowledge vs Dynamic RL
    if perfect_makespan < dynamic_makespan:
        perfect_advantage = ((dynamic_makespan - perfect_makespan) / dynamic_makespan) * 100
        print(f"Perfect Knowledge advantage over Dynamic RL: {perfect_advantage:.1f}%")
    
    # Dynamic RL vs Static RL (on dynamic scenario)
    if dynamic_makespan < static_dynamic_makespan:
        improvement = ((static_dynamic_makespan - dynamic_makespan) / static_dynamic_makespan) * 100
        print(f"✓ Dynamic RL outperforms Static RL (dynamic) by {improvement:.1f}%")
    else:
        gap = ((dynamic_makespan - static_dynamic_makespan) / static_dynamic_makespan) * 100
        print(f"✗ Dynamic RL underperforms Static RL (dynamic) by {gap:.1f}%")
    
    # Static RL comparison: dynamic vs static scenarios
    if static_static_makespan < static_dynamic_makespan:
        improvement = ((static_dynamic_makespan - static_static_makespan) / static_dynamic_makespan) * 100
        print(f"✓ Static RL performs {improvement:.1f}% better on static scenarios (as expected)")
    else:
        gap = ((static_static_makespan - static_dynamic_makespan) / static_static_makespan) * 100
        print(f"⚠️ Unexpected: Static RL performs {gap:.1f}% worse on static scenarios")
    
    # Dynamic RL vs Best Heuristic
    if dynamic_makespan < spt_makespan:
        improvement = ((spt_makespan - dynamic_makespan) / spt_makespan) * 100
        print(f"✓ Dynamic RL outperforms Best Heuristic by {improvement:.1f}%")
    else:
        gap = ((dynamic_makespan - spt_makespan) / spt_makespan) * 100
        print(f"✗ Dynamic RL underperforms Best Heuristic by {gap:.1f}%")
    
    # Step 7: Generate Gantt Charts for Comparison
    print(f"\n7. GANTT CHART COMPARISON")
    print("-" * 60)
    
    # Main comparison with 4 plots (remove static RL on static from main plot)
    num_plots = 5 if milp_makespan != float('inf') else 4
    fig, axes = plt.subplots(num_plots, 1, figsize=(18, num_plots * 3.5))
    
    if milp_makespan != float('inf'):
        fig.suptitle('Main Scheduling Comparison: MILP vs Perfect Knowledge vs Dynamic vs Static RL vs Best Heuristic\n' + 
                     f'Test Scenario: Jobs 0-2 at t=0, Jobs 3-6 via Poisson arrivals\n' +
                     f'Static RL evaluated on dynamic scenario with arrivals', 
                     fontsize=16, fontweight='bold')
        schedules_data = [
            {'schedule': milp_schedule, 'makespan': milp_makespan, 'title': 'MILP Optimal (Benchmark)', 'arrival_times': first_scenario_arrivals},
            {'schedule': perfect_schedule, 'makespan': perfect_makespan, 'title': 'Perfect Knowledge RL', 'arrival_times': first_scenario_arrivals},
            {'schedule': dynamic_schedule, 'makespan': dynamic_makespan, 'title': 'Dynamic RL', 'arrival_times': first_scenario_arrivals},
            {'schedule': static_dynamic_schedule, 'makespan': static_dynamic_makespan, 'title': 'Static RL (on dynamic scenario)', 'arrival_times': first_scenario_arrivals},
            {'schedule': spt_schedule, 'makespan': spt_makespan, 'title': 'Best Heuristic', 'arrival_times': first_scenario_arrivals}
        ]
    else:
        fig.suptitle('Main Scheduling Comparison: Perfect Knowledge vs Dynamic vs Static RL vs Best Heuristic\n' + 
                     f'Test Scenario: Jobs 0-2 at t=0, Jobs 3-6 via Poisson arrivals\n' +
                     f'Static RL evaluated on dynamic scenario with arrivals', 
                     fontsize=16, fontweight='bold')
        schedules_data = [
            {'schedule': perfect_schedule, 'makespan': perfect_makespan, 'title': 'Perfect Knowledge RL', 'arrival_times': first_scenario_arrivals},
            {'schedule': dynamic_schedule, 'makespan': dynamic_makespan, 'title': 'Dynamic RL', 'arrival_times': first_scenario_arrivals},
            {'schedule': static_dynamic_schedule, 'makespan': static_dynamic_makespan, 'title': 'Static RL (on dynamic scenario)', 'arrival_times': first_scenario_arrivals},
            {'schedule': spt_schedule, 'makespan': spt_makespan, 'title': 'Best Heuristic', 'arrival_times': first_scenario_arrivals}
        ]
    
    colors = plt.cm.tab20.colors
    
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
    
    # Save with appropriate filename based on MILP availability
    if milp_makespan != float('inf'):
        filename = 'complete_scheduling_comparison_with_milp_optimal.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comprehensive comparison with MILP optimal: {filename}")
    else:
        filename = 'dynamic_vs_static_gantt_comparison-7jobs.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison without MILP: {filename}")
    
    plt.show()
    
    # Step 8: Create separate plot for Static RL comparison (dynamic vs static scenarios)
    print(f"\n8. SEPARATE STATIC RL COMPARISON PLOT")
    print("-" * 60)
    
    fig_static, axes_static = plt.subplots(2, 1, figsize=(18, 8))
    fig_static.suptitle('Static RL Performance Comparison: Dynamic Scenario vs Static Scenario\n' + 
                       f'Static RL trained on static cases but evaluated on both arrival patterns\n' +
                       f'Dynamic: Jobs 0-2 at t=0, Jobs 3-6 via Poisson | Static: All jobs at t=0', 
                       fontsize=14, fontweight='bold')
    
    static_comparison_data = [
        {'schedule': static_dynamic_schedule, 'makespan': static_dynamic_makespan, 'title': 'Static RL on Dynamic Scenario', 'arrival_times': first_scenario_arrivals},
        {'schedule': static_static_schedule, 'makespan': static_static_makespan, 'title': 'Static RL on Static Scenario (All jobs at t=0)', 'arrival_times': static_arrivals}
    ]
    
    for plot_idx, data in enumerate(static_comparison_data):
        schedule = data['schedule']
        makespan = data['makespan']
        title = data['title']
        arrival_times = data['arrival_times']
        
        ax = axes_static[plot_idx]
        
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
        
        # Add red arrows for job arrivals (only for dynamic scenario)
        if plot_idx == 0 and arrival_times:  # Only for dynamic scenario
            arrow_y_position = len(MACHINE_LIST) + 0.3  # Position above all machines
            for job_id, arrival_time in arrival_times.items():
                if arrival_time > 0 and arrival_time < 200:  # Only show arrows for jobs that don't start at t=0
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
        ax.set_xlabel("Time" if plot_idx == 1 else "")
        ax.set_ylabel("Machines")
        ax.set_title(f"{title} (Makespan: {makespan:.2f})", fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set consistent x-axis limits
        if schedule and any(len(ops) > 0 for ops in schedule.values()):
            max_time = max([max([op[2] for op in ops]) for ops in schedule.values() if ops])
            ax.set_xlim(0, max_time * 1.05)
        else:
            ax.set_xlim(0, 100)  # Default range if no schedule
        ax.set_ylim(-0.5, len(MACHINE_LIST) + 2.0)  # Extra space for arrival arrows and labels
    
    # Add legend for static comparison
    legend_elements_static = []
    for i in range(len(ENHANCED_JOBS_DATA)):
        color = colors[i % len(colors)]
        initial_or_poisson = ' (Initial)' if i < 3 else ' (Poisson)'
        legend_elements_static.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                          alpha=0.8, label=f'Job {i}{initial_or_poisson}'))
    
    fig_static.legend(handles=legend_elements_static, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(ENHANCED_JOBS_DATA), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    
    # Save the separate static comparison plot
    static_filename = 'static_rl_dynamic_vs_static_comparison.png'
    plt.savefig(static_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved separate Static RL comparison: {static_filename}")
    
    # Analysis of Static RL performance difference
    print(f"\nStatic RL Performance Analysis:")
    print(f"• Static scenario makespan: {static_static_makespan:.2f}")
    print(f"• Dynamic scenario makespan: {static_dynamic_makespan:.2f}")
    
    if static_static_makespan < static_dynamic_makespan:
        improvement = ((static_dynamic_makespan - static_static_makespan) / static_dynamic_makespan) * 100
        print(f"✅ Static RL performs {improvement:.1f}% better on static scenarios (expected)")
    else:
        degradation = ((static_static_makespan - static_dynamic_makespan) / static_static_makespan) * 100
        print(f"❌ UNEXPECTED: Static RL performs {degradation:.1f}% worse on static scenarios")
        print("   This suggests issues with the Static RL training or environment")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED!")
    print("Generated files:")
    if milp_makespan != float('inf'):
        print("- complete_scheduling_comparison_with_milp_optimal.png: Six-method comprehensive comparison with MILP benchmark")
        print("- static_rl_dynamic_vs_static_comparison.png: Separate Static RL comparison (dynamic vs static scenarios)")
        print("- arrival_distribution_analysis.png: Training arrival time analysis")
        print(f"\nKey Findings:")
        print(f"• MILP Optimal (Benchmark): {milp_makespan:.2f}")
        print(f"• Perfect Knowledge RL: {perfect_makespan:.2f} (regret: +{((perfect_makespan-milp_makespan)/milp_makespan*100):.1f}%)")
        print(f"• Dynamic RL: {dynamic_makespan:.2f} (regret: +{((dynamic_makespan-milp_makespan)/milp_makespan*100):.1f}%)")
        print(f"• Static RL (dynamic): {static_dynamic_makespan:.2f} (regret: +{((static_dynamic_makespan-milp_makespan)/milp_makespan*100):.1f}%)")
        print(f"• Static RL (static): {static_static_makespan:.2f} (regret: +{((static_static_makespan-milp_makespan)/milp_makespan*100):.1f}%)")
        print(f"• Perfect Knowledge RL validation: {'✅ Working well' if perfect_makespan <= milp_makespan * 1.15 else '❌ Needs improvement'}")
    else:
        print("- dynamic_vs_static_gantt_comparison-7jobs.png: Five-method comparison")
        print("- static_rl_dynamic_vs_static_comparison.png: Separate Static RL comparison (dynamic vs static scenarios)")
        print("- arrival_distribution_analysis.png: Training arrival time analysis")
        print(f"\nKey Findings (no MILP benchmark available):")
        print(f"• Perfect Knowledge RL: {perfect_makespan:.2f}")
        print(f"• Dynamic RL: {dynamic_makespan:.2f}")
        print(f"• Static RL (dynamic): {static_dynamic_makespan:.2f}")
        print(f"• Static RL (static): {static_static_makespan:.2f}")
        print(f"• Performance hierarchy: {'✅ Expected' if perfect_makespan <= dynamic_makespan <= static_dynamic_makespan else '❌ Unexpected'}")
        print(f"• Static RL scenario comparison: {'✅ Better on static' if static_static_makespan < static_dynamic_makespan else '❌ Needs investigation'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
