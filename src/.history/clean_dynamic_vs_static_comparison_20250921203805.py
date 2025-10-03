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

# Training metrics tracking
TRAINING_METRICS = {
    'episode_rewards': [],
    'episode_lengths': [],
    'action_entropy': [],
    'policy_loss': [],
    'value_loss': [],
    'timesteps': []
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
        1. Machine time-until-free (relative to current decision time)
        2. Next operation for each job
        3. Job next-operation-ready-time (relative to current decision time)
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
        
        # CORRECTED: Current decision time is the earliest we can schedule next operation
        # This is the minimum of all machine free times and job ready times for available operations
        current_decision_time = self._get_current_decision_time()
        
        # 1. Machine time-until-free (relative timing, normalized)
        for machine in self.machines:
            time_until_free = max(0.0, self.machine_next_free[machine] - current_decision_time)
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
        
        # 3. Job next-operation-ready-time (relative to current decision time, normalized)
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
                
                time_until_ready = max(0.0, job_ready_time - current_decision_time)
                normalized_ready_time = min(1.0, time_until_ready / time_scale)
                obs.append(normalized_ready_time)
            else:
                # Job completed: no more operations
                obs.append(0.0)
        
        # 4. Current makespan/time (normalized)
        normalized_makespan = min(1.0, current_decision_time / time_scale)
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

    def _get_current_decision_time(self):
        """
        Calculate the current decision time - the earliest time when we can make a scheduling decision.
        This considers:
        1. When machines become available
        2. When jobs are ready for their next operation
        3. When jobs have arrived
        
        Returns the time at which the next scheduling decision should be made.
        """
        candidate_times = []
        
        # Consider machine availability times
        for machine in self.machines:
            candidate_times.append(self.machine_next_free.get(machine, 0.0))
        
        # Consider when jobs are ready for their next operation
        for job_id in self.arrived_jobs:
            next_op_idx = self.next_operation[job_id]
            if next_op_idx < len(self.jobs[job_id]):  # Job has more operations
                if next_op_idx == 0:
                    # First operation: ready when job arrives (already arrived since job is in arrived_jobs)
                    job_ready_time = self.job_arrival_times.get(job_id, 0.0)
                else:
                    # Later operation: ready when previous operation completes
                    job_ready_time = self.operation_end_times[job_id][next_op_idx - 1]
                candidate_times.append(job_ready_time)
        
        # The decision time is the minimum of all these times (earliest we can act)
        # But it should be at least as large as the current makespan
        if candidate_times:
            decision_time = max(min(candidate_times), self.current_makespan)
        else:
            decision_time = self.current_makespan
        
        return decision_time

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
            self.num_jobs                           # Job arrival status (0/1)
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
        1. Machine time-until-free (relative to current decision time)
        2. Next operation for each job
        3. Job next-operation-ready-time (relative to current decision time)
        4. Current makespan/time (normalized)
        5. Job arrival status (dynamic: actual arrival times)
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
        
        # CORRECTED: Use proper decision time calculation
        current_decision_time = self._get_current_decision_time()
        
        # 1. Machine time-until-free (relative timing, normalized)
        for machine in self.machines:
            time_until_free = max(0.0, self.machine_next_free[machine] - current_decision_time)
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
        
        # 3. Job next-operation-ready-time (relative to current decision time, normalized)
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
                
                time_until_ready = max(0.0, job_ready_time - current_decision_time)
                normalized_ready_time = min(1.0, time_until_ready / time_scale)
                obs.append(normalized_ready_time)
            else:
                # Job completed: no more operations
                obs.append(0.0)
        
        # 4. Current makespan/time (normalized)
        normalized_makespan = min(1.0, current_decision_time / time_scale)
        obs.append(normalized_makespan)
        
        # 5. Job arrival status (dynamic: binary arrived/not-arrived, NOT specific times)
        # CRITICAL FIX: Dynamic RL should NOT know future arrival times!
        # It should only know which jobs have already arrived, maintaining generalizability
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                obs.append(0.0)  # Job has arrived (available for scheduling)
            else:
                obs.append(1.0)  # Job has not arrived yet (not available)
        
        # Ensure correct size and format
        target_size = self.observation_space.shape[0]
        if len(obs) < target_size:
            obs.extend([0.0] * (target_size - len(obs)))
        elif len(obs) > target_size:
            obs = obs[:target_size]
        
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array

    def _get_current_decision_time(self):
        """
        Calculate the current decision time - the earliest time when we can make a scheduling decision.
        This considers:
        1. When machines become available
        2. When jobs are ready for their next operation
        3. When jobs have arrived
        
        Returns the time at which the next scheduling decision should be made.
        """
        candidate_times = []
        
        # Consider machine availability times
        for machine in self.machines:
            candidate_times.append(self.machine_next_free.get(machine, 0.0))
        
        # Consider when jobs are ready for their next operation
        for job_id in self.arrived_jobs:
            next_op_idx = self.next_operation[job_id]
            if next_op_idx < len(self.jobs[job_id]):  # Job has more operations
                if next_op_idx == 0:
                    # First operation: ready when job arrives (already arrived since job is in arrived_jobs)
                    job_ready_time = self.job_arrival_times.get(job_id, 0.0)
                else:
                    # Later operation: ready when previous operation completes
                    job_ready_time = self.operation_end_times[job_id][next_op_idx - 1]
                candidate_times.append(job_ready_time)
        
        # The decision time is the minimum of all these times (earliest we can act)
        # But it should be at least as large as the current makespan
        if candidate_times:
            decision_time = max(min(candidate_times), self.current_makespan)
        else:
            decision_time = self.current_makespan
        
        return decision_time

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
    """Callback to track training metrics including action entropy."""
    
    def __init__(self, method_name):
        self.method_name = method_name
        self.step_count = 0
        
    def __call__(self, locals_dict, globals_dict):
        global TRAINING_METRICS
        
        # Extract PPO model from locals
        model = locals_dict.get('self')
        
        if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
            log_data = model.logger.name_to_value
            
            # Log entropy if available
            if 'train/entropy_loss' in log_data:
                TRAINING_METRICS['action_entropy'].append(log_data['train/entropy_loss'])
            
            # Log policy and value losses
            if 'train/policy_gradient_loss' in log_data:
                TRAINING_METRICS['policy_loss'].append(log_data['train/policy_gradient_loss'])
            if 'train/value_loss' in log_data:
                TRAINING_METRICS['value_loss'].append(log_data['train/value_loss'])
            
            TRAINING_METRICS['timesteps'].append(model.num_timesteps)
            
            # Suppress periodic entropy updates during training for cleaner output
            # if len(TRAINING_METRICS['action_entropy']) > 0 and len(TRAINING_METRICS['action_entropy']) % 10 == 0:
            #     recent_entropy = TRAINING_METRICS['action_entropy'][-1]
            #     print(f"  {self.method_name} - Step {model.num_timesteps}: Action Entropy = {recent_entropy:.4f}")
        
        return True

def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, reward_mode="makespan_increment", learning_rate=3e-4):
    """
    Train a perfect knowledge RL agent using the same approach as test3_backup.py.
    
    Key insight: Train on deterministic arrival times (like the test scenario)
    rather than trying to create a complex "perfect knowledge" environment.
    This matches the working approach from test3_backup.py.
    
    IMPORTANT: Each Perfect Knowledge RL is trained specifically for ONE scenario
    with the exact arrival times, making it the optimal RL benchmark for that scenario.
    """
    print(f"    Training arrival times: {arrival_times}")
    print(f"    Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    # Use DIFFERENT seed for each Perfect Knowledge training to ensure diversity
    # But make it deterministic based on arrival pattern
    arrival_seed = hash(str(sorted(arrival_times.items()))) % 10000
    print(f"    Using scenario-specific seed: {arrival_seed}")
    
    def make_perfect_env():
        # Use PerfectKnowledgeFJSPEnv for both training and evaluation consistency
        env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times, reward_mode=reward_mode, seed=arrival_seed)
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_perfect_env])
    
    # Use similar hyperparameters as in test3_backup.py
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
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
        seed=arrival_seed,         # Use scenario-specific seed
        policy_kwargs=dict(
            net_arch=[512, 512, 256],  # Matches test3_backup.py
            activation_fn=torch.nn.ReLU
        )
    )
    
    # Training with progress bar and entropy tracking (silent)
    print(f"    Training perfect knowledge agent for this specific scenario...")
    callback = TrainingCallback("Perfect Knowledge RL")
    
    # Training with progress tracking for Perfect Knowledge
    print(f"    Starting training...")
    
    # Do a quick evaluation before training to establish baseline
    test_env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times, reward_mode=reward_mode, seed=arrival_seed)
    test_env = ActionMasker(test_env, mask_fn)
    obs, _ = test_env.reset()
    
    # Quick random baseline test
    random_makespan = 0
    try:
        random_steps = 0
        done = False
        while not done and random_steps < 100:
            action_masks = test_env.action_masks()
            if not any(action_masks):
                break
            # Random valid action
            valid_actions = [i for i, mask in enumerate(action_masks) if mask]
            if valid_actions:
                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = test_env.step(action)
                random_steps += 1
        random_makespan = test_env.env.current_makespan
        print(f"    Random baseline makespan: {random_makespan:.2f}")
    except:
        print(f"    Could not establish random baseline")
    
    # Train with progress tracking
    callback = TrainingCallback("Perfect Knowledge RL")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Quick post-training evaluation
    test_env.reset()
    post_training_makespan = 999.0
    try:
        obs, _ = test_env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action_masks = test_env.action_masks()
            if not any(action_masks):
                break
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            steps += 1
        post_training_makespan = test_env.env.current_makespan
        print(f"    Post-training makespan: {post_training_makespan:.2f}")
        
        if post_training_makespan < random_makespan:
            print(f"    ✅ Training improved by {random_makespan - post_training_makespan:.2f}")
        else:
            print(f"    ⚠️  Training did not improve (random: {random_makespan:.2f}, trained: {post_training_makespan:.2f})")
    except Exception as e:
        print(f"    ⚠️  Could not evaluate post-training: {e}")
    
    print(f"    ✅ Perfect knowledge training completed for this scenario!")
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
        reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                      previous_makespan, self.current_makespan)
        
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
        1. Machine time-until-free (relative to current decision time)
        2. Next operation for each job
        3. Job next-operation-ready-time (relative to current decision time)
        4. Current makespan/time (normalized)
        5. Job arrival times (perfect knowledge: knows exact arrival times)
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
        
        current_decision_time = self._get_current_decision_time()
        
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
        
        # 5. Job arrival times (perfect knowledge: knows exact future arrival times)
        # Perfect Knowledge RL SHOULD know exact arrival times - this is its advantage
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
        verbose=0,
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
    print(f"Using test seeds 1000-{1000+num_scenarios-1} (different from training seed {GLOBAL_SEED})")
    
    scenarios = []
    for i in range(num_scenarios):
        test_seed = 1000 + i  # Use completely different seed range from training
        np.random.seed(test_seed)  # Different from GLOBAL_SEED=42 used in training
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
            'seed': test_seed
        })
        
        arrived_jobs = [j for j, t in arrival_times.items() if t < float('inf')]
        print(f"  Scenario {i+1}: {len(arrived_jobs)} jobs, rate={arrival_rate:.3f}")
        print(f"    Initial: {current_initial}")
        print(f"    Arrivals: {len(arrived_jobs) - len(current_initial)} jobs")
    
    return scenarios


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


def spt_heuristic_poisson(jobs_data, machine_list, arrival_times):
    """
    Shortest Processing Time (SPT) heuristic for Poisson FJSP.
    This is a simple greedy heuristic that prioritizes jobs with shorter processing times.
    """
    try:
        # Create schedule structure
        schedule = {m: [] for m in machine_list}
        machine_next_free = {m: 0.0 for m in machine_list}
        
        # Track job progress
        job_next_op = {job_id: 0 for job_id in jobs_data.keys()}
        job_op_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data.keys()}
        
        # Get all operations with their details
        all_operations = []
        for job_id, job_ops in jobs_data.items():
            for op_idx, op_data in enumerate(job_ops):
                min_proc_time = min(op_data['proc_times'].values())
                all_operations.append((job_id, op_idx, min_proc_time))
        
        # Sort by shortest processing time
        all_operations.sort(key=lambda x: x[2])
        
        # Schedule operations respecting precedence and arrivals
        operations_scheduled = 0
        max_iterations = len(all_operations) * 10
        
        while operations_scheduled < sum(len(ops) for ops in jobs_data.values()) and max_iterations > 0:
            max_iterations -= 1
            progress_made = False
            
            for job_id, op_idx, _ in all_operations:
                # Check if this operation is the next one for this job
                if job_next_op[job_id] != op_idx:
                    continue
                
                # Check if job has arrived
                job_arrival_time = arrival_times.get(job_id, 0.0)
                
                # Check precedence constraint
                if op_idx > 0:
                    prev_op_end_time = job_op_end_times[job_id][op_idx - 1]
                else:
                    prev_op_end_time = job_arrival_time
                
                # Find best machine for this operation
                op_data = jobs_data[job_id][op_idx]
                best_machine = None
                best_end_time = float('inf')
                
                for machine, proc_time in op_data['proc_times'].items():
                    machine_available = machine_next_free[machine]
                    start_time = max(machine_available, prev_op_end_time, job_arrival_time)
                    end_time = start_time + proc_time
                    
                    if end_time < best_end_time:
                        best_end_time = end_time
                        best_machine = machine
                        best_start_time = start_time
                        best_proc_time = proc_time
                
                if best_machine is not None:
                    # Schedule the operation
                    schedule[best_machine].append((f"J{job_id}-O{op_idx+1}", best_start_time, best_end_time))
                    machine_next_free[best_machine] = best_end_time
                    job_op_end_times[job_id][op_idx] = best_end_time
                    job_next_op[job_id] += 1
                    operations_scheduled += 1
                    progress_made = True
            
            if not progress_made:
                break
        
        # Calculate makespan
        if any(len(ops) > 0 for ops in schedule.values()):
            makespan = max([max([op[2] for op in ops]) for ops in schedule.values() if ops])
        else:
            makespan = float('inf')
        
        return makespan, schedule
        
    except Exception as e:
        print(f"SPT Heuristic failed: {e}")
        return float('inf'), {m: [] for m in machine_list}


def milp_optimal_scheduler(jobs_data, machine_list, arrival_times, time_limit=300):
    """
    MILP-based optimal scheduler for FJSP with job arrivals.
    Uses PuLP to formulate and solve the MILP problem.
    """
    try:
        from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
        
        # Create the problem
        prob = LpProblem("FJSP_Optimal", LpMinimize)
        
        # Calculate big M (upper bound on completion times)
        total_proc_time = sum(sum(op['proc_times'].values()) for job_ops in jobs_data.values() for op in job_ops)
        big_m = total_proc_time + max(arrival_times.values()) + 100
        
        # Decision variables
        # x[j,o,m,t] = 1 if job j operation o starts on machine m at time t
        jobs = list(jobs_data.keys())
        time_horizon = int(big_m)
        
        # Simplified model: Use start time variables instead of binary time variables
        # s[j,o] = start time of job j operation o
        start_times = {}
        machine_assignment = {}  # Which machine processes each operation
        
        for job_id in jobs:
            for op_idx in range(len(jobs_data[job_id])):
                start_times[(job_id, op_idx)] = LpVariable(f"start_{job_id}_{op_idx}", 0, big_m)
                
                # Machine assignment variables
                op_data = jobs_data[job_id][op_idx]
                for machine in op_data['proc_times'].keys():
                    machine_assignment[(job_id, op_idx, machine)] = LpVariable(
                        f"assign_{job_id}_{op_idx}_{machine}", cat='Binary')
        
        # Makespan variable
        makespan = LpVariable("makespan", 0, big_m)
        
        # Objective: minimize makespan
        prob += makespan
        
        # Constraints
        
        # 1. Each operation assigned to exactly one compatible machine
        for job_id in jobs:
            for op_idx in range(len(jobs_data[job_id])):
                op_data = jobs_data[job_id][op_idx]
                prob += lpSum([machine_assignment[(job_id, op_idx, machine)] 
                              for machine in op_data['proc_times'].keys()]) == 1
        
        # 2. Precedence constraints within jobs
        for job_id in jobs:
            job_ops = jobs_data[job_id]
            for op_idx in range(1, len(job_ops)):
                # Current operation starts after previous operation completes
                prev_op_duration = lpSum([machine_assignment[(job_id, op_idx-1, machine)] * 
                                        job_ops[op_idx-1]['proc_times'][machine]
                                        for machine in job_ops[op_idx-1]['proc_times'].keys()])
                
                prob += (start_times[(job_id, op_idx)] >= 
                        start_times[(job_id, op_idx-1)] + prev_op_duration)
        
        # 3. Job arrival time constraints
        for job_id in jobs:
            arrival_time = arrival_times.get(job_id, 0.0)
            prob += start_times[(job_id, 0)] >= arrival_time
        
        # 4. Machine capacity constraints (no overlapping operations on same machine)
        for machine in machine_list:
            # Get all operations that can use this machine
            machine_ops = []
            for job_id in jobs:
                for op_idx in range(len(jobs_data[job_id])):
                    if machine in jobs_data[job_id][op_idx]['proc_times']:
                        machine_ops.append((job_id, op_idx))
            
            # For each pair of operations that can use this machine
            for i, (job1, op1) in enumerate(machine_ops):
                for j, (job2, op2) in enumerate(machine_ops[i+1:], i+1):
                    # If both operations are assigned to this machine, they cannot overlap
                    proc_time_1 = jobs_data[job1][op1]['proc_times'][machine]
                    proc_time_2 = jobs_data[job2][op2]['proc_times'][machine]
                    
                    # Binary variables for ordering
                    order_var = LpVariable(f"order_{job1}_{op1}_{job2}_{op2}_{machine}", cat='Binary')
                    
                    # If both assigned to this machine, one must complete before other starts
                    assigned_both = (machine_assignment[(job1, op1, machine)] + 
                                   machine_assignment[(job2, op2, machine)] - 1)
                    
                    # Either op1 finishes before op2 starts, or op2 finishes before op1 starts
                    prob += (start_times[(job1, op1)] + proc_time_1 <= 
                            start_times[(job2, op2)] + big_m * (1 - order_var) + 
                            big_m * (2 - machine_assignment[(job1, op1, machine)] - 
                                   machine_assignment[(job2, op2, machine)]))
                    
                    prob += (start_times[(job2, op2)] + proc_time_2 <= 
                            start_times[(job1, op1)] + big_m * order_var + 
                            big_m * (2 - machine_assignment[(job1, op1, machine)] - 
                                   machine_assignment[(job2, op2, machine)]))
        
        # 5. Makespan constraints
        for job_id in jobs:
            job_ops = jobs_data[job_id]
            last_op_idx = len(job_ops) - 1
            
            # Makespan >= completion time of last operation of each job
            last_op_duration = lpSum([machine_assignment[(job_id, last_op_idx, machine)] * 
                                    job_ops[last_op_idx]['proc_times'][machine]
                                    for machine in job_ops[last_op_idx]['proc_times'].keys()])
            
            prob += makespan >= start_times[(job_id, last_op_idx)] + last_op_duration
        
        # Solve the problem
        solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        prob.solve(solver)
        
        if prob.status != 1:  # Not optimal
            print(f"MILP solver status: {prob.status} (not optimal)")
            return float('inf'), {m: [] for m in machine_list}
        
        # Extract solution
        schedule = {m: [] for m in machine_list}
        
        for job_id in jobs:
            for op_idx in range(len(jobs_data[job_id])):
                op_start = start_times[(job_id, op_idx)].value()
                
                # Find which machine this operation is assigned to
                assigned_machine = None
                for machine in jobs_data[job_id][op_idx]['proc_times'].keys():
                    if machine_assignment[(job_id, op_idx, machine)].value() > 0.5:
                        assigned_machine = machine
                        break
                
                if assigned_machine is not None:
                    proc_time = jobs_data[job_id][op_idx]['proc_times'][assigned_machine]
                    op_end = op_start + proc_time
                    
                    schedule[assigned_machine].append((f"J{job_id}-O{op_idx+1}", op_start, op_end))
        
        # Sort operations by start time for each machine
        for machine in schedule:
            schedule[machine].sort(key=lambda x: x[1])
        
        optimal_makespan = makespan.value()
        return optimal_makespan, schedule
        
    except ImportError:
        print("PuLP not available - cannot solve MILP")
        return float('inf'), {m: [] for m in machine_list}
    except Exception as e:
        print(f"MILP solver failed: {e}")
        return float('inf'), {m: [] for m in machine_list}


def calculate_regret_analysis(optimal_makespan, methods_results):
    """Calculate regret (gap from optimal) for each method."""
    regret_results = {}
    
    for method, makespan in methods_results.items():
        if makespan != float('inf') and optimal_makespan != float('inf'):
            regret = ((makespan - optimal_makespan) / optimal_makespan) * 100
            regret_results[method] = regret
            
            if regret < 5:
                status = "✅ Excellent"
            elif regret < 15:
                status = "🟢 Good"
            elif regret < 30:
                status = "🟡 Acceptable"
            else:
                status = "🔴 Poor"
            
            print(f"{method:25s}: +{regret:5.1f}% above optimal ({status})")
        else:
            regret_results[method] = float('inf')
            print(f"{method:25s}: Failed to find solution")
    
    return regret_results


def diagnose_performance_similarity(perfect_makespan, dynamic_makespan, static_makespan, heuristic_makespan):
    """Diagnose why different methods might be giving similar results."""
    print(f"\n🔍 PERFORMANCE SIMILARITY DIAGNOSIS:")
    
    makespans = [perfect_makespan, dynamic_makespan, static_makespan, heuristic_makespan]
    methods = ["Perfect RL", "Dynamic RL", "Static RL", "Heuristic"]
    
    # Check if all methods give very similar results
    valid_makespans = [m for m in makespans if m != float('inf')]
    if len(valid_makespans) > 1:
        max_span = max(valid_makespans)
        min_span = min(valid_makespans)
        relative_diff = ((max_span - min_span) / min_span) * 100
        
        if relative_diff < 5:
            print(f"⚠️  All methods very similar (±{relative_diff:.1f}%)")
            print("   Possible reasons:")
            print("   1. Problem is easy - little room for optimization")
            print("   2. All methods converge to similar local optima")
            print("   3. Arrival pattern doesn't create significant dynamic advantage")
            print("   4. RL agents haven't learned problem-specific patterns")
        else:
            print(f"✅ Methods show meaningful differences (±{relative_diff:.1f}%)")
    
    # Check specific pairs
    if abs(perfect_makespan - dynamic_makespan) < 0.1:
        print(f"🚨 Perfect Knowledge RL ≈ Dynamic RL: Perfect knowledge not being utilized")
    
    if abs(dynamic_makespan - static_makespan) < 0.1:
        print(f"🚨 Dynamic RL ≈ Static RL: Dynamic information not providing advantage")
    
    if abs(dynamic_makespan - heuristic_makespan) < 0.1:
        print(f"🚨 Dynamic RL ≈ Heuristic: RL not outperforming simple heuristic")
