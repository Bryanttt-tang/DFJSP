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
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)

# Global tracking for arrival time distribution analysis
TRAINING_ARRIVAL_TIMES = []  # Track all arrival times during training
TRAINING_EPISODE_COUNT = 0   # Track episode count
DEBUG_EPISODE_ARRIVALS = []  # Track first 10 episodes' arrival details

# Training metrics tracking - ENHANCED for detailed loss visualization
TRAINING_METRICS = {
    'episode_rewards': [],
    'episode_lengths': [],
    'action_entropy': [],
    'policy_loss': [],
    'value_loss': [],
    'timesteps': [],
    'episodes': [],  # Track episode numbers
    'learning_rate': [],  # Track learning rate changes
    'clip_fraction': []   # Track clipping fraction
}

# Agent-specific training metrics for comparison
AGENT_TRAINING_METRICS = {
    'Perfect Knowledge RL': {'policy_loss': [], 'value_loss': [], 'episodes': [], 'timesteps': []},
    'Dynamic RL': {'policy_loss': [], 'value_loss': [], 'episodes': [], 'timesteps': []},
    'Static RL': {'policy_loss': [], 'value_loss': [], 'episodes': [], 'timesteps': []}
}

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
        
        # Minimal observation space for fully observed MDP
        obs_size = (
            len(self.machines) +                    # Machine time-until-free
            self.num_jobs +                         # Next operation for each job
            self.num_jobs +                         # Job next-op-ready-time
            1 +                                     # Current makespan/time
            self.num_jobs                           # Job arrival times (static: all 0)
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
            super().reset(seed=seed, options=options)
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
                
                # # Add completion bonus
                # if done:
                #     reward += 50.0
                    
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
        """
        Generate minimal observation for fully observed MDP.
        
        Components:
        1. Machine time-until-free (relative to current time)
        2. Next operation for each job
        3. Job next-operation-ready-time (relative to current time)
        4. Current makespan/time (normalized)
        5. Job arrival times (static: all 0)
        """
        obs = []
        
        # Estimate time scale for consistent normalization
        all_proc_times = []
        for job_data in self.jobs.values():
            for op in job_data:
                all_proc_times.extend(op['proc_times'].values())
        
        if all_proc_times:
            time_scale = max(np.mean(all_proc_times) * 10, 1.0)  # 10x avg proc time as scale
        else:
            time_scale = 50.0  # Fallback
        
        current_time = self.current_makespan
        
        # 1. Machine time-until-free (relative timing, normalized)
        for machine in self.machines:
            time_until_free = max(0.0, self.machine_next_free[machine] - current_time)
            normalized_time = min(1.0, time_until_free / time_scale)
            obs.append(normalized_time)
        
        # 2. Next operation for each job (normalized by max_ops_per_job)
        for job_id in self.job_ids:
            next_op = self.next_operation[job_id]
            total_ops = len(self.jobs[job_id])
            if total_ops > 0:
                normalized_next_op = float(next_op) / float(self.max_ops_per_job)
            else:
                normalized_next_op = 1.0  # Job has no operations
            obs.append(min(1.0, normalized_next_op))
        
        # 3. Job next-operation-ready-time (relative to current time, normalized)
        for job_id in self.job_ids:
            next_op_idx = self.next_operation[job_id]
            
            if next_op_idx < len(self.jobs[job_id]):
                # Job has more operations
                if next_op_idx == 0:
                    # First operation: ready when job arrives
                    job_ready_time = self.job_arrival_times.get(job_id, 0.0)
                else:
                    # Later operation: ready when previous operation completes
                    job_ready_time = self.operation_end_times[job_id][next_op_idx - 1]
                
                time_until_ready = max(0.0, job_ready_time - current_time)
                normalized_ready_time = min(1.0, time_until_ready / time_scale)
                obs.append(normalized_ready_time)
            else:
                # Job completed: no more operations
                obs.append(0.0)
        
        # 4. Current makespan/time (normalized)
        normalized_makespan = min(1.0, current_time / time_scale)
        obs.append(normalized_makespan)
        
        # 5. Job arrival times (static: all jobs arrive at t=0)
        for job_id in self.job_ids:
            # For static environment, all arrival times are 0
            obs.append(0.0)
        
        # Ensure correct size and format
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
        
        # Minimal observation space for fully observed MDP
        obs_size = (
            len(self.machines) +                    # Machine time-until-free
            self.num_jobs +                         # Next operation for each job
            self.num_jobs +                         # Job next-op-ready-time
            1 +                                     # Current makespan/time
            self.num_jobs                           # Job arrival times (dynamic)
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
        """Generate arrival times for dynamic jobs using Poisson process - BACK TO INTEGERS."""
        # Initialize arrival times
        for job_id in self.initial_job_ids:
            self.job_arrival_times[job_id] = 0.0
        
        # Generate inter-arrival times using exponential distribution
        current_time = 0.0
        for job_id in self.dynamic_job_ids:
            inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
            current_time += inter_arrival_time
            
            # CHANGED BACK: Round to integers for simple, discrete time
            if current_time <= self.max_time_horizon:
                self.job_arrival_times[job_id] = float(round(current_time))  # Back to integer arrivals
            else:
                self.job_arrival_times[job_id] = float('inf')  # Won't arrive in this episode

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode - SAME structure as successful environments."""
        global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT, DEBUG_EPISODE_ARRIVALS
        
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
        
        # Debug: Track first 10 episodes in detail (silently)
        if TRAINING_EPISODE_COUNT <= 10:
            episode_debug_info = {
                'episode': TRAINING_EPISODE_COUNT,
                'initial_jobs': sorted(self.initial_job_ids),
                'dynamic_jobs': sorted(self.dynamic_job_ids),
                'arrival_times': dict(self.job_arrival_times),
                'arrived_at_reset': sorted(self.arrived_jobs),
                'dynamic_arrivals': sorted(episode_arrivals) if episode_arrivals else []
            }
            DEBUG_EPISODE_ARRIVALS.append(episode_debug_info)
        
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
                
                # # Small bonus for utilizing newly arrived jobs (dynamic advantage)
                # if num_new_arrivals > 0:
                #     reward += 5.0 * num_new_arrivals
                
                # # Add completion bonus
                # if done:
                #     reward += 50.0
                    
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
        Generate minimal observation for fully observed MDP.
        
        Components:
        1. Machine time-until-free (relative to current time)
        2. Next operation for each job
        3. Job next-operation-ready-time (relative to current time)
        4. Current makespan/time (normalized)
        5. Job arrival times (dynamic: actual arrival times)
        """
        obs = []
        
        # Estimate time scale for consistent normalization
        all_proc_times = []
        for job_data in self.jobs.values():
            for op in job_data:
                all_proc_times.extend(op['proc_times'].values())
        
        if all_proc_times:
            time_scale = max(np.mean(all_proc_times) * 10, 1.0)  # 10x avg proc time as scale
        else:
            time_scale = 50.0  # Fallback
        
        current_time = self.current_makespan
        
        # 1. Machine time-until-free (relative timing, normalized)
        for machine in self.machines:
            time_until_free = max(0.0, self.machine_next_free[machine] - current_time)
            normalized_time = min(1.0, time_until_free / time_scale)
            obs.append(normalized_time)
        
        # 2. Next operation for each job (normalized by max_ops_per_job)
        for job_id in self.job_ids:
            next_op = self.next_operation[job_id]
            total_ops = len(self.jobs[job_id])
            if total_ops > 0:
                normalized_next_op = float(next_op) / float(self.max_ops_per_job)
            else:
                normalized_next_op = 1.0  # Job has no operations
            obs.append(min(1.0, normalized_next_op))
        
        # 3. Job next-operation-ready-time (relative to current time, normalized)
        for job_id in self.job_ids:
            next_op_idx = self.next_operation[job_id]
            
            if next_op_idx < len(self.jobs[job_id]):
                # Job has more operations
                if next_op_idx == 0:
                    # First operation: ready when job arrives
                    job_ready_time = self.job_arrival_times.get(job_id, 0.0)
                else:
                    # Later operation: ready when previous operation completes
                    job_ready_time = self.operation_end_times[job_id][next_op_idx - 1]
                
                time_until_ready = max(0.0, job_ready_time - current_time)
                normalized_ready_time = min(1.0, time_until_ready / time_scale)
                obs.append(normalized_ready_time)
            else:
                # Job completed: no more operations
                obs.append(0.0)
        
        # 4. Current makespan/time (normalized)
        normalized_makespan = min(1.0, current_time / time_scale)
        obs.append(normalized_makespan)
        
        # 5. Job arrival times (dynamic: normalized arrival times relative to current time)
        for job_id in self.job_ids:
            arrival_time = self.job_arrival_times.get(job_id, 0.0)
            if arrival_time == float('inf'):
                # Job won't arrive in this episode
                obs.append(1.0)  # Use max value to indicate "very far future"
            else:
                # Time until job arrives (or 0 if already arrived)
                time_until_arrival = max(0.0, arrival_time - current_time)
                normalized_arrival = min(1.0, time_until_arrival / time_scale)
                obs.append(normalized_arrival)
        
        # Ensure correct size and format
        target_size = self.observation_space.shape[0]
        if len(obs) < target_size:
            obs.extend([0.0] * (target_size - len(obs)))
        elif len(obs) > target_size:
            obs = obs[:target_size]
        
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

class TrainingCallback:
    """Enhanced callback to track detailed training metrics including losses."""
    
    def __init__(self, method_name):
        self.method_name = method_name
        self.step_count = 0
        self.episode_count = 0
        self.last_logged_episode = -1
        
    def __call__(self, locals_dict, globals_dict):
        global TRAINING_METRICS, AGENT_TRAINING_METRICS
        
        # Extract PPO model from locals
        model = locals_dict.get('self')
        
        if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
            log_data = model.logger.name_to_value
            
            # Track episode count
            if 'rollout/ep_len_mean' in log_data:
                current_episode = model.num_timesteps // (model.n_steps if hasattr(model, 'n_steps') else 2048)
                if current_episode > self.last_logged_episode:
                    self.episode_count = current_episode
                    self.last_logged_episode = current_episode
            
            # Log losses with episode tracking
            if 'train/policy_gradient_loss' in log_data:
                policy_loss = log_data['train/policy_gradient_loss']
                TRAINING_METRICS['policy_loss'].append(policy_loss)
                AGENT_TRAINING_METRICS[self.method_name]['policy_loss'].append(policy_loss)
                AGENT_TRAINING_METRICS[self.method_name]['episodes'].append(self.episode_count)
                AGENT_TRAINING_METRICS[self.method_name]['timesteps'].append(model.num_timesteps)
                
            if 'train/value_loss' in log_data:
                value_loss = log_data['train/value_loss']
                TRAINING_METRICS['value_loss'].append(value_loss)
                AGENT_TRAINING_METRICS[self.method_name]['value_loss'].append(value_loss)
            
            # Log additional metrics
            if 'train/entropy_loss' in log_data:
                TRAINING_METRICS['action_entropy'].append(log_data['train/entropy_loss'])
            
            if 'train/learning_rate' in log_data:
                TRAINING_METRICS['learning_rate'].append(log_data['train/learning_rate'])
                
            if 'train/clip_fraction' in log_data:
                TRAINING_METRICS['clip_fraction'].append(log_data['train/clip_fraction'])
            
            TRAINING_METRICS['timesteps'].append(model.num_timesteps)
            TRAINING_METRICS['episodes'].append(self.episode_count)
        
        return True

def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, reward_mode="makespan_increment", learning_rate=3e-4):
    """
    Train a perfect knowledge RL agent using the same approach as test3_backup.py.
    
    Key insight: Train on deterministic arrival times (like the test scenario)
    rather than trying to create a complex "perfect knowledge" environment.
    This matches the working approach from test3_backup.py.
    """
    print(f"\n--- Training Perfect Knowledge RL Agent (test3_backup.py approach) ---")
    print(f"Training arrival times: {arrival_times}")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    print(f"Fixed seed: {GLOBAL_SEED} (for reproducibility)")
    
    def make_perfect_env():
        # Use PerfectKnowledgeFJSPEnv for both training and evaluation consistency
    
        env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times, reward_mode=reward_mode)
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_perfect_env])
    
    # Use similar hyperparameters as in test3_backup.py
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=learning_rate,        # Matches test3_backup.py
        n_steps=2048,              # Matches test3_backup.py
        batch_size=128,            # Matches test3_backup.py  
        n_epochs=10,               # Matches test3_backup.py
        gamma=1,                # Matches test3_backup.py
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=GLOBAL_SEED,          # Ensure reproducibility
        policy_kwargs=dict(
            net_arch=[512, 512, 256],  # Matches test3_backup.py
            activation_fn=torch.nn.ReLU
        )
    )
    
    # Training with progress bar and entropy tracking
    print("Training perfect knowledge agent (deterministic arrival times)...")
    callback = TrainingCallback("Perfect Knowledge RL")
    
    with tqdm(total=total_timesteps, desc="Perfect Knowledge Training") as pbar:
        def combined_callback(locals_dict, globals_dict):
            callback(locals_dict, globals_dict)
            pbar.update(model.n_steps)
            return True
        
        model.learn(total_timesteps=total_timesteps, callback=combined_callback)
    
    print(f"Perfect knowledge training completed!")
    return model

def train_static_agent(jobs_data, machine_list, total_timesteps=300000, reward_mode="makespan_increment", learning_rate=3e-4):
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
        learning_rate=learning_rate,
        n_steps=4096,          # Increased for larger job set
        batch_size=512,        # Increased for larger job set
        n_epochs=10,           # More epochs for complex patterns
        gamma=1,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=GLOBAL_SEED,      # Ensure reproducibility
        policy_kwargs=dict(
            net_arch=[256, 128, 64],  # Smaller network for 7-job dataset
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"Training Static RL for {total_timesteps:,} timesteps with seed {GLOBAL_SEED}...")
    
    # Train with tqdm progress bar and entropy tracking
    start_time = time.time()
    callback = TrainingCallback("Static RL")
    
    with tqdm(total=total_timesteps, desc="Static RL", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} timesteps [{elapsed}<{remaining}]') as pbar:
        
        def combined_callback(locals_dict, globals_dict):
            callback(locals_dict, globals_dict)
            return True
        
        # Break training into chunks for progress updates
        chunk_size = total_timesteps // 30  # 30 chunks
        remaining_timesteps = total_timesteps
        
        while remaining_timesteps > 0:
            current_chunk = min(chunk_size, remaining_timesteps)
            model.learn(total_timesteps=current_chunk, callback=combined_callback)
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
        
        # Minimal observation space matching other environments
        obs_size = (
            len(self.machines) +                    # Machine time-until-free
            self.num_jobs +                         # Next operation for each job
            self.num_jobs +                         # Job next-op-ready-time
            1 +                                     # Current makespan/time
            self.num_jobs                           # Job arrival times (perfect knowledge)
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        """Reset environment - based on test3_backup.py approach."""
        if seed is not None:
            super().reset(seed=seed, options=options)
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
        previous_makespan = self.current_makespan
        self.machine_next_free[machine] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        
        # Update makespan and check for new arrivals (key improvement)
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

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan):
        """Reward calculation based on test3_backup.py approach"""
        if self.reward_mode == "makespan_increment":
            # R(s_t, a_t) = E(t) - E(t+1) = negative increment in makespan
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment (reward for not increasing makespan)
                
                # # Add small completion bonus
                # if done:
                #     reward += 50.0
                    
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
        """
        Generate minimal observation for fully observed MDP.
        
        Components:
        1. Machine time-until-free (relative to current time)
        2. Next operation for each job
        3. Job next-operation-ready-time (relative to current time)
        4. Current makespan/time (normalized)
        5. Job arrival times (perfect knowledge: exact arrival times)
        """
        obs = []
        
        # Estimate time scale for consistent normalization
        all_proc_times = []
        for job_data in self.jobs.values():
            for op in job_data:
                all_proc_times.extend(op['proc_times'].values())
        
        if all_proc_times:
            time_scale = max(np.mean(all_proc_times) * 10, 1.0)  # 10x avg proc time as scale
        else:
            time_scale = 50.0  # Fallback
        
        current_time = self.current_makespan
        
        # 1. Machine time-until-free (relative timing, normalized)
        for machine in self.machines:
            time_until_free = max(0.0, self.machine_next_free[machine] - current_time)
            normalized_time = min(1.0, time_until_free / time_scale)
            obs.append(normalized_time)
        
        # 2. Next operation for each job (normalized by max_ops_per_job)
        for job_id in self.job_ids:
            next_op = self.next_operation[job_id]
            total_ops = len(self.jobs[job_id])
            if total_ops > 0:
                normalized_next_op = float(next_op) / float(self.max_ops_per_job)
            else:
                normalized_next_op = 1.0  # Job has no operations
            obs.append(min(1.0, normalized_next_op))
        
        # 3. Job next-operation-ready-time (relative to current time, normalized)
        for job_id in self.job_ids:
            next_op_idx = self.next_operation[job_id]
            
            if next_op_idx < len(self.jobs[job_id]):
                # Job has more operations
                if next_op_idx == 0:
                    # First operation: ready when job arrives
                    job_ready_time = self.job_arrival_times.get(job_id, 0.0)
                else:
                    # Later operation: ready when previous operation completes
                    job_ready_time = self.operation_end_times[job_id][next_op_idx - 1]
                
                time_until_ready = max(0.0, job_ready_time - current_time)
                normalized_ready_time = min(1.0, time_until_ready / time_scale)
                obs.append(normalized_ready_time)
            else:
                # Job completed: no more operations
                obs.append(0.0)
        
        # 4. Current makespan/time (normalized)
        normalized_makespan = min(1.0, current_time / time_scale)
        obs.append(normalized_makespan)
        
        # 5. Job arrival times (perfect knowledge: normalized arrival times relative to current time)
        for job_id in self.job_ids:
            arrival_time = self.job_arrival_times.get(job_id, 0.0)
            # Time until job arrives (or 0 if already arrived)
            time_until_arrival = max(0.0, arrival_time - current_time)
            normalized_arrival = min(1.0, time_until_arrival / time_scale)
            obs.append(normalized_arrival)
        
        # Ensure correct size and format
        target_size = self.observation_space.shape[0]
        if len(obs) < target_size:
            obs.extend([0.0] * (target_size - len(obs)))
        elif len(obs) > target_size:
            obs = obs[:target_size]
        
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array


def train_dynamic_agent(jobs_data, machine_list, initial_jobs=5, arrival_rate=0.08, total_timesteps=500000, reward_mode="makespan_increment", learning_rate=3e-4):
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
        verbose=1,
        learning_rate=learning_rate,        # Match PerfectKnowledgeFJSPEnv
        n_steps=2048,              # Match PerfectKnowledgeFJSPEnv
        batch_size=128,            # Match PerfectKnowledgeFJSPEnv
        n_epochs=10,               # Match PerfectKnowledgeFJSPEnv
        gamma=1,                   # Match PerfectKnowledgeFJSPEnv
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=GLOBAL_SEED,          # Ensure reproducibility
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
    
    print(f"Training Dynamic RL for {total_timesteps:,} timesteps with seed {GLOBAL_SEED}...")
    print(f"Using simplified approach matching successful environments")
    
    # Train with progress bar like PerfectKnowledgeFJSPEnv
    start_time = time.time()
    callback = TrainingCallback("Dynamic RL")
    
    with tqdm(total=total_timesteps, desc="Dynamic RL Training") as pbar:
        def combined_callback(locals_dict, globals_dict):
            callback(locals_dict, globals_dict)
            if hasattr(model, 'num_timesteps'):
                pbar.n = model.num_timesteps
                pbar.refresh()
            return True
        
        model.learn(total_timesteps=total_timesteps, callback=combined_callback)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Dynamic RL training completed in {training_time:.1f}s!")
    
    return model

def generate_test_scenarios(jobs_data, initial_jobs=[0, 1, 2, 3, 4], arrival_rate=0.08, num_scenarios=10):
    """
    Generate diverse test scenarios with expanded job set.
    Uses different seeds from training to test generalizability.
    """
    print(f"Generating {num_scenarios} test scenarios from {len(jobs_data)} total jobs...")
    print(f"Using test seeds 5000-{5000+num_scenarios-1} (different from training seed {GLOBAL_SEED})")
    
    scenarios = []
    for i in range(num_scenarios):
        test_seed = GLOBAL_SEED+1 + i  # Changed from 1000 to 5000 range for fresh test scenarios
        np.random.seed(test_seed)  # Different from GLOBAL_SEED=12345 used in training
        random.seed(test_seed)
        arrival_times = {}
        
        # Use consistent initial jobs across all test scenarios
        # Only vary the dynamic job arrival times to isolate the impact of arrivals
        current_initial = initial_jobs
        
        # Initial jobs arrive at t=0
        for job_id in current_initial:
            arrival_times[job_id] = 0.0
        
        # Generate Poisson arrivals for remaining jobs
        remaining_jobs = [j for j in jobs_data.keys() if j not in current_initial]
        current_time = 0.0
        
        for job_id in remaining_jobs:
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival_time
            
            # CHANGED BACK: Use integer values for arrival times instead of float values
            # This provides simpler, discrete time scheduling for better learning
            if current_time <= 300:  # Extended time horizon for larger job set
                arrival_times[job_id] = round(current_time)  # Back to integer arrivals
            else:
                arrival_times[job_id] = float('inf')  # Won't arrive
        
        scenarios.append({
            'scenario_id': i,
            'arrival_times': arrival_times,
            'initial_jobs': current_initial,
            'arrival_rate':arrival_rate,
            'seed': test_seed
        })
        
        arrived_jobs = [j for j, t in arrival_times.items() if t < float('inf')]
        print(f"  Scenario {i+1}: {len(arrived_jobs)} jobs, rate={arrival_rate:.3f}")
        print(f"    Initial: {current_initial}")
        print(f"    Arrivals: {len(arrived_jobs) - len(current_initial)} jobs")
    
    return scenarios


def plot_training_metrics():
    """
    Plot training metrics including action entropy, policy loss, and value loss.
    This helps debug PPO exploration and convergence issues.
    """
    global TRAINING_METRICS
    
    if not TRAINING_METRICS['timesteps']:
        print("No training metrics recorded!")
        return
    
    print(f"\n=== TRAINING METRICS ANALYSIS ===")
    print(f"Total training steps recorded: {len(TRAINING_METRICS['timesteps'])}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PPO Training Metrics Analysis - Debugging Exploration & Convergence', fontsize=16, fontweight='bold')
    
    timesteps = TRAINING_METRICS['timesteps']
    
    # Plot 1: Action Entropy over time
    if TRAINING_METRICS['action_entropy']:
        axes[0, 0].plot(timesteps[:len(TRAINING_METRICS['action_entropy'])], TRAINING_METRICS['action_entropy'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Action Entropy')
        axes[0, 0].set_title('Action Entropy (Exploration Level)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add interpretation
        final_entropy = TRAINING_METRICS['action_entropy'][-1]
        if final_entropy > 0.5:
            axes[0, 0].text(0.02, 0.98, '✅ High exploration', transform=axes[0, 0].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        elif final_entropy > 0.1:
            axes[0, 0].text(0.02, 0.98, '🟡 Moderate exploration', transform=axes[0, 0].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        else:
            axes[0, 0].text(0.02, 0.98, '🔴 Low exploration', transform=axes[0, 0].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    else:
        axes[0, 0].text(0.5, 0.5, 'No entropy data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Action Entropy (No Data)')
    
    # Plot 2: Policy Loss over time
    if TRAINING_METRICS['policy_loss']:
        axes[0, 1].plot(timesteps[:len(TRAINING_METRICS['policy_loss'])], TRAINING_METRICS['policy_loss'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Policy Loss')
        axes[0, 1].set_title('Policy Gradient Loss')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No policy loss data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Policy Loss (No Data)')
    
    # Plot 3: Value Loss over time
    if TRAINING_METRICS['value_loss']:
        axes[1, 0].plot(timesteps[:len(TRAINING_METRICS['value_loss'])], TRAINING_METRICS['value_loss'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Value Loss')
        axes[1, 0].set_title('Value Function Loss')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No value loss data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Value Loss (No Data)')
    
    # Plot 4: Combined losses
    axes[1, 1].set_title('Combined Training Losses')
    if TRAINING_METRICS['policy_loss'] and TRAINING_METRICS['value_loss']:
        min_len = min(len(TRAINING_METRICS['policy_loss']), len(TRAINING_METRICS['value_loss']))
        axes[1, 1].plot(timesteps[:min_len], TRAINING_METRICS['policy_loss'][:min_len], 'r-', linewidth=2, label='Policy Loss')
        axes[1, 1].plot(timesteps[:min_len], TRAINING_METRICS['value_loss'][:min_len], 'g-', linewidth=2, label='Value Loss')
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient loss data', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig('ppo_training_metrics.png', dpi=300, bbox_inches='tight')
    print("✅ Training metrics plot saved: ppo_training_metrics.png")
    plt.show()
    
    # Print summary statistics
    if TRAINING_METRICS['action_entropy']:
        entropy_data = TRAINING_METRICS['action_entropy']
        print(f"\nAction Entropy Statistics:")
        print(f"  Initial: {entropy_data[0]:.4f}")
        print(f"  Final: {entropy_data[-1]:.4f}")
        print(f"  Mean: {np.mean(entropy_data):.4f}")
        print(f"  Std: {np.std(entropy_data):.4f}")
        
        # Entropy trend analysis
        if len(entropy_data) > 10:
            early_entropy = np.mean(entropy_data[:len(entropy_data)//4])
            late_entropy = np.mean(entropy_data[-len(entropy_data)//4:])
            entropy_change = ((late_entropy - early_entropy) / early_entropy) * 100
            
            print(f"  Early training avg: {early_entropy:.4f}")
            print(f"  Late training avg: {late_entropy:.4f}")
            print(f"  Change: {entropy_change:+.1f}%")
            
            if entropy_change < -50:
                print("  🔴 WARNING: Entropy dropped significantly - may indicate premature convergence")
            elif entropy_change < -20:
                print("  🟡 CAUTION: Entropy decreased - normal but monitor for exploitation vs exploration balance")
            else:
                print("  ✅ Entropy maintained reasonably well")
    
    print("=" * 50)

def analyze_first_10_episodes():
    """
    Analyze and display the dynamic job arrivals for the first 10 episodes.
    This helps understand the variation in Poisson arrival patterns during early training.
    """
    global DEBUG_EPISODE_ARRIVALS
    
    print(f"\n" + "="*80)
    print("FIRST 10 EPISODES - DYNAMIC JOB ARRIVAL ANALYSIS")
    print("="*80)
    print(f"Debug episodes recorded: {len(DEBUG_EPISODE_ARRIVALS)}")
    
    if not DEBUG_EPISODE_ARRIVALS:
        print("❌ No debug episode data recorded!")
        print("   This might be because:")
        print("   - Dynamic RL training didn't complete")
        print("   - Episode tracking wasn't properly initialized")  
        print("   - All jobs arrived beyond time horizon")
        return
    
    for episode_info in DEBUG_EPISODE_ARRIVALS:
        ep_num = episode_info['episode'] 
        arrival_times = episode_info['arrival_times']
        dynamic_arrivals = episode_info['dynamic_arrivals']
        
        print(f"\nEpisode {ep_num}:")
        print(f"  Jobs arriving dynamically: {len(dynamic_arrivals)} jobs")
        if dynamic_arrivals:
            print(f"  Arrival times: {dynamic_arrivals}")
            print(f"  Time span: {min(dynamic_arrivals):.1f} - {max(dynamic_arrivals):.1f}")
            print(f"  Average inter-arrival: {(max(dynamic_arrivals) - min(dynamic_arrivals)) / max(1, len(dynamic_arrivals)-1):.1f}")
        else:
            print(f"  No dynamic arrivals (all jobs beyond time horizon)")
            
        # Show which jobs will never arrive
        no_arrival_jobs = [j for j, t in arrival_times.items() if t == float('inf')]
        if no_arrival_jobs:
            print(f"  Jobs not arriving: {sorted(no_arrival_jobs)}")
    
    # Summary statistics
    all_dynamic_arrivals = []
    episodes_with_arrivals = 0
    for ep_info in DEBUG_EPISODE_ARRIVALS:
        if ep_info['dynamic_arrivals']:
            all_dynamic_arrivals.extend(ep_info['dynamic_arrivals'])
            episodes_with_arrivals += 1
    
    print(f"\n" + "-"*80)
    print("SUMMARY (First 10 Episodes):")
    print(f"Episodes with dynamic arrivals: {episodes_with_arrivals}/{len(DEBUG_EPISODE_ARRIVALS)}")
    if all_dynamic_arrivals:
        print(f"Total dynamic arrivals: {len(all_dynamic_arrivals)}")
        print(f"Arrival time range: {min(all_dynamic_arrivals):.1f} - {max(all_dynamic_arrivals):.1f}")
        print(f"Average arrival time: {np.mean(all_dynamic_arrivals):.1f}")
        print(f"Std deviation: {np.std(all_dynamic_arrivals):.1f}")
    else:
        print("No dynamic arrivals recorded in first 10 episodes!")
    print("="*80)

def plot_training_losses_comparison():
    """
    Plot training losses for all three RL agents in comparison.
    Shows how well each agent is learning during training.
    """
    global AGENT_TRAINING_METRICS
    
    print(f"\n=== TRAINING LOSS COMPARISON ACROSS ALL RL AGENTS ===")
    
    # Check if we have data for all agents
    agents_with_data = []
    for agent_name, metrics in AGENT_TRAINING_METRICS.items():
        if metrics['policy_loss'] and metrics['value_loss']:
            agents_with_data.append(agent_name)
            print(f"✅ {agent_name}: {len(metrics['policy_loss'])} loss measurements")
        else:
            print(f"❌ {agent_name}: No loss data recorded")
    
    if len(agents_with_data) < 2:
        print("❌ Insufficient data for comparison plot!")
        return
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Loss Comparison: Perfect Knowledge vs Dynamic vs Static RL\n' + 
                 'Good training should show decreasing losses over episodes', 
                 fontsize=16, fontweight='bold')
    
    colors = {'Perfect Knowledge RL': 'blue', 'Dynamic RL': 'green', 'Static RL': 'red'}
    
    # Plot 1: Policy Loss Comparison
    ax1 = axes[0, 0]
    for agent_name in agents_with_data:
        metrics = AGENT_TRAINING_METRICS[agent_name]
        if metrics['episodes'] and metrics['policy_loss']:
            episodes = metrics['episodes'][:len(metrics['policy_loss'])]
            policy_losses = metrics['policy_loss']
            
            ax1.plot(episodes, policy_losses, 
                    color=colors.get(agent_name, 'black'), 
                    label=agent_name, linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Policy Loss')
    ax1.set_title('Policy Gradient Loss Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Value Loss Comparison
    ax2 = axes[0, 1]
    for agent_name in agents_with_data:
        metrics = AGENT_TRAINING_METRICS[agent_name]
        if metrics['episodes'] and metrics['value_loss']:
            episodes = metrics['episodes'][:len(metrics['value_loss'])]
            value_losses = metrics['value_loss']
            
            ax2.plot(episodes, value_losses, 
                    color=colors.get(agent_name, 'black'), 
                    label=agent_name, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Value Loss')
    ax2.set_title('Value Function Loss Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization
    
    # Plot 3: Combined Loss Trends (normalized)
    ax3 = axes[1, 0]
    for agent_name in agents_with_data:
        metrics = AGENT_TRAINING_METRICS[agent_name]
        if metrics['episodes'] and metrics['policy_loss'] and metrics['value_loss']:
            episodes = metrics['episodes'][:min(len(metrics['policy_loss']), len(metrics['value_loss']))]
            
            # Normalize losses to [0, 1] for comparison
            policy_losses = np.array(metrics['policy_loss'][:len(episodes)])
            value_losses = np.array(metrics['value_loss'][:len(episodes)])
            
            if len(policy_losses) > 0 and len(value_losses) > 0:
                # Normalize by first value to show relative improvement
                normalized_policy = policy_losses / (policy_losses[0] if policy_losses[0] != 0 else 1)
                normalized_value = value_losses / (value_losses[0] if value_losses[0] != 0 else 1)
                
                ax3.plot(episodes, normalized_policy, 
                        color=colors.get(agent_name, 'black'), 
                        linestyle='-', label=f'{agent_name} Policy', linewidth=2, alpha=0.8)
                ax3.plot(episodes, normalized_value, 
                        color=colors.get(agent_name, 'black'), 
                        linestyle='--', label=f'{agent_name} Value', linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Training Episodes')
    ax3.set_ylabel('Normalized Loss (relative to initial)')
    ax3.set_title('Loss Improvement Over Training (Normalized)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Initial Loss Level')
    
    # Plot 4: Training Progress Analysis
    ax4 = axes[1, 1]
    
    # Calculate final vs initial loss ratios for analysis
    improvement_data = []
    for agent_name in agents_with_data:
        metrics = AGENT_TRAINING_METRICS[agent_name]
        if len(metrics['policy_loss']) > 10:  # Need sufficient data
            initial_policy = np.mean(metrics['policy_loss'][:5])  # First 5 measurements
            final_policy = np.mean(metrics['policy_loss'][-5:])   # Last 5 measurements
            
            initial_value = np.mean(metrics['value_loss'][:5]) if len(metrics['value_loss']) > 10 else 1
            final_value = np.mean(metrics['value_loss'][-5:]) if len(metrics['value_loss']) > 10 else 1
            
            policy_improvement = (initial_policy - final_policy) / initial_policy * 100
            value_improvement = (initial_value - final_value) / initial_value * 100
            
            improvement_data.append({
                'agent': agent_name,
                'policy_improvement': policy_improvement,
                'value_improvement': value_improvement
            })
    
    if improvement_data:
        agents = [d['agent'] for d in improvement_data]
        policy_improvements = [d['policy_improvement'] for d in improvement_data]
        value_improvements = [d['value_improvement'] for d in improvement_data]
        
        x = np.arange(len(agents))
        width = 0.35
        
        ax4.bar(x - width/2, policy_improvements, width, label='Policy Loss Improvement (%)', 
               color='lightblue', alpha=0.8)
        ax4.bar(x + width/2, value_improvements, width, label='Value Loss Improvement (%)', 
               color='lightcoral', alpha=0.8)
        
        ax4.set_xlabel('RL Agents')
        ax4.set_ylabel('Loss Improvement (%)')
        ax4.set_title('Training Effectiveness: % Loss Reduction')
        ax4.set_xticks(x)
        ax4.set_xticklabels(agents, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add percentage labels on bars
        for i, (policy_imp, value_imp) in enumerate(zip(policy_improvements, value_improvements)):
            ax4.text(i - width/2, policy_imp + 1, f'{policy_imp:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
            ax4.text(i + width/2, value_imp + 1, f'{value_imp:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for improvement analysis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Training Effectiveness Analysis - No Data')
    
    plt.tight_layout()
    plt.savefig('ppo_training_metrics.png', dpi=300, bbox_inches='tight')
    print("✅ Training metrics plot saved: ppo_training_metrics.png")
    plt.show()
    
    # Print analysis summary
    print("\n📊 TRAINING ANALYSIS SUMMARY:")
    for agent_name in agents_with_data:
        metrics = AGENT_TRAINING_METRICS[agent_name]
        if len(metrics['policy_loss']) > 10:
            initial_loss = np.mean(metrics['policy_loss'][:5])
            final_loss = np.mean(metrics['policy_loss'][-5:])
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            status = "✅ Good" if improvement > 20 else "🟡 Moderate" if improvement > 5 else "🔴 Poor"
            print(f"{agent_name:25s}: {improvement:6.1f}% policy loss reduction {status}")
            
            if improvement < 5:
                print(f"  ⚠️  Consider: longer training, lower learning rate, or different hyperparameters")
        else:
            print(f"{agent_name:25s}: Insufficient data for analysis")
    
    print("=" * 80)

def plot_individual_agent_training(agent_name):
    """
    Plot detailed training metrics for a single agent.
    """
    global AGENT_TRAINING_METRICS
    
    if agent_name not in AGENT_TRAINING_METRICS:
        print(f"❌ No data for agent: {agent_name}")
        return
    
    metrics = AGENT_TRAINING_METRICS[agent_name]
    if not metrics['policy_loss'] or not metrics['value_loss']:
        print(f"❌ No loss data for agent: {agent_name}")
        return
    
    print(f"\n=== DETAILED TRAINING ANALYSIS: {agent_name} ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Detailed Training Analysis: {agent_name}\n' + 
                 'Monitoring learning progress and convergence', 
                 fontsize=16, fontweight='bold')
    
    episodes = metrics['episodes'][:len(metrics['policy_loss'])]
    policy_losses = metrics['policy_loss']
    value_losses = metrics['value_loss'][:len(episodes)]
    
    # Plot 1: Policy Loss Over Time
    axes[0, 0].plot(episodes, policy_losses, 'b-', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Training Episodes')
    axes[0, 0].set_ylabel('Policy Loss')
    axes[0, 0].set_title('Policy Gradient Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Add trend line
    if len(episodes) > 5:
        z = np.polyfit(episodes, np.log(policy_losses), 1)
        p = np.poly1d(z)
        axes[0, 0].plot(episodes, np.exp(p(episodes)), "r--", alpha=0.8, label='Trend')
        axes[0, 0].legend()
    
    # Plot 2: Value Loss Over Time
    axes[0, 1].plot(episodes, value_losses, 'g-', linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Training Episodes')
    axes[0, 1].set_ylabel('Value Loss')
    axes[0, 1].set_title('Value Function Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Add trend line
    if len(episodes) > 5:
        z = np.polyfit(episodes, np.log(value_losses), 1)
        p = np.poly1d(z)
        axes[0, 1].plot(episodes, np.exp(p(episodes)), "r--", alpha=0.8, label='Trend')
        axes[0, 1].legend()
    
    # Plot 3: Moving Average (Smoothed Losses)
    if len(policy_losses) > 10:
        window_size = max(5, len(policy_losses) // 20)
        policy_ma = np.convolve(policy_losses, np.ones(window_size)/window_size, mode='valid')
        value_ma = np.convolve(value_losses[:len(policy_ma)], np.ones(window_size)/window_size, mode='valid')
        episodes_ma = episodes[window_size-1:len(policy_ma)+window_size-1]
        
        axes[1, 0].plot(episodes_ma, policy_ma, 'b-', linewidth=2, label='Policy Loss (smoothed)')
        axes[1, 0].plot(episodes_ma, value_ma, 'g-', linewidth=2, label='Value Loss (smoothed)')
        axes[1, 0].set_xlabel('Training Episodes')
        axes[1, 0].set_ylabel('Loss (Moving Average)')
        axes[1, 0].set_title(f'Smoothed Training Progress (window={window_size})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient data for\nmoving average', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Smoothed Training Progress - No Data')
    
    # Plot 4: Training Statistics
    axes[1, 1].axis('off')
    
    # Calculate statistics
    initial_policy = np.mean(policy_losses[:5]) if len(policy_losses) > 5 else policy_losses[0]
    final_policy = np.mean(policy_losses[-5:]) if len(policy_losses) > 5 else policy_losses[-1]
    policy_improvement = (initial_policy - final_policy) / initial_policy * 100
    
    initial_value = np.mean(value_losses[:5]) if len(value_losses) > 5 else value_losses[0]
    final_value = np.mean(value_losses[-5:]) if len(value_losses) > 5 else value_losses[-1]
    value_improvement = (initial_value - final_value) / initial_value * 100
    
    # Display statistics
    stats_text = f"""
    TRAINING STATISTICS:
    
    📈 Policy Loss:
       Initial:     {initial_policy:.4f}
       Final:       {final_policy:.4f}
       Improvement: {policy_improvement:.1f}%
    
    📈 Value Loss:
       Initial:     {initial_value:.4f}
       Final:       {final_value:.4f}
       Improvement: {value_improvement:.1f}%
    
    📊 Training Progress:
       Episodes:    {len(episodes)}
       Data Points: {len(policy_losses)}
       
    🎯 Convergence:
       Policy: {'✅ Good' if policy_improvement > 20 else '🟡 Moderate' if policy_improvement > 5 else '🔴 Poor'}
       Value:  {'✅ Good' if value_improvement > 20 else '🟡 Moderate' if value_improvement > 5 else '🔴 Poor'}
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save individual agent plot
    filename = f'training_analysis_{agent_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Individual training analysis saved: {filename}")
    plt.show()
    
    return policy_improvement, value_improvement

def main():
 
    print("=" * 80)
    print("DYNAMIC vs STATIC RL COMPARISON FOR POISSON FJSP")
    print("=" * 80)
    print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
    print("Research Question: Does Dynamic RL outperform Static RL on Poisson arrivals?")
    print(f"🔧 REPRODUCIBILITY: Fixed seed {GLOBAL_SEED} for all random components")
    print("📊 DEBUGGING: Action entropy & training metrics tracking enabled")
    print("=" * 80)
    arrival_rate = 0.5  # HIGHER arrival rate to create more dynamic scenarios
    # With λ=0.5, expected inter-arrival = 2 time units (faster than most job operations)
    
    # Step 1: Training Setup
    print("\n1. TRAINING SETUP")
    print("-" * 50)
    perfect_timesteps = 100000    # Perfect knowledge needs less training
    dynamic_timesteps = 100000   # Increased for better learning with integer timing  
    static_timesteps = 100000    # Increased for better learning
    learning_rate = 1e-3       # Standard learning rate for PPO
    
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
                                                 reward_mode="makespan_increment", learning_rate=learning_rate)
    
    # Train dynamic RL agent (knows arrival distribution only)
    dynamic_model = train_dynamic_agent(ENHANCED_JOBS_DATA, MACHINE_LIST, 
                                       initial_jobs=[0, 1, 2], arrival_rate=arrival_rate, 
                                       total_timesteps=dynamic_timesteps,reward_mode="makespan_increment",learning_rate=learning_rate)

    # Train static RL agent (assumes all jobs at t=0)
    static_model = train_static_agent(ENHANCED_JOBS_DATA, MACHINE_LIST, total_timesteps=static_timesteps, 
                                     reward_mode="makespan_increment", learning_rate=learning_rate)

    # Analyze arrival time distribution during training
    print("\n3.5. TRAINING ANALYSIS")
    analyze_first_10_episodes()  # Show detailed first 10 episodes
    
    # NEW: Plot training losses for all agents
    print("\n3.6. TRAINING LOSS ANALYSIS")
    plot_training_losses_comparison()  # Compare all three agents
    
    # Plot individual agent analyses
    for agent_name in ['Perfect Knowledge RL', 'Dynamic RL', 'Static RL']:
        if agent_name in AGENT_TRAINING_METRICS and AGENT_TRAINING_METRICS[agent_name]['policy_loss']:
            policy_imp, value_imp = plot_individual_agent_training(agent_name)
            print(f"{agent_name}: Policy improvement: {policy_imp:.1f}%, Value improvement: {value_imp:.1f}%")
    
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
    
    # Calculate the maximum makespan across all schedules for consistent scaling
    max_makespan_for_scaling = 0
    for data in schedules_data:
        schedule = data['schedule']
        if schedule and any(len(ops) > 0 for ops in schedule.values()):
            schedule_max_time = max([max([op[2] for op in ops]) for ops in schedule.values() if ops])
            max_makespan_for_scaling = max(max_makespan_for_scaling, schedule_max_time)
    
    # Add some padding (10%) for visual clarity
    consistent_x_limit = max_makespan_for_scaling * 1.1 if max_makespan_for_scaling > 0 else 100
    
    # Plot each schedule
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
            # Still apply consistent scaling even for failed schedules
            ax.set_xlim(0, consistent_x_limit)
            ax.set_ylim(-0.5, len(MACHINE_LIST) + 2.0)
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
        ax.set_xlim(0, consistent_x_limit)
        ax.set_ylim(-0.5, len(MACHINE_LIST) + 2.0)  # Extra space for arrival arrows and labels
        ax.grid(True, alpha=0.3)
        
        # Add red arrows for job arrivals (only for dynamic jobs that arrive > 0)
        if arrival_times:
            arrow_y_position = len(MACHINE_LIST) + 0.3  # Position above all machines
            for job_id, arrival_time in arrival_times.items():
                if arrival_time > 0 and arrival_time < consistent_x_limit:  # Only show arrows for jobs that don't start at t=0 and arrive within time horizon
                    # Draw vertical line for arrival
                    ax.axvline(x=arrival_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
                    
                    # Add arrow and label
                    ax.annotate(f'Job {job_id} arrives', 
                               xy=(arrival_time, arrow_y_position), 
                               xytext=(arrival_time, arrow_y_position + 0.5),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2),
                               ha='center', va='bottom', color='red', fontweight='bold', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))
    
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
    
    # Calculate consistent scaling for static comparison plots
    static_max_makespan = 0
    for data in static_comparison_data:
        schedule = data['schedule']
        if schedule and any(len(ops) > 0 for ops in schedule.values()):
            schedule_max_time = max([max([op[2] for op in ops]) for ops in schedule.values() if ops])
            static_max_makespan = max(static_max_makespan, schedule_max_time)
    
    static_consistent_x_limit = static_max_makespan * 1.1 if static_max_makespan > 0 else 100
    
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
            # Still apply consistent scaling even for failed schedules
            ax.set_xlim(0, static_consistent_x_limit)
            ax.set_ylim(-0.5, len(MACHINE_LIST) + 2.0)
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
                if arrival_time > 0 and arrival_time < static_consistent_x_limit:  # Only show arrows for jobs that don't start at t=0
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
        
        # Apply consistent x-axis limits across both static comparison plots
        ax.set_xlim(0, static_consistent_x_limit)
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
