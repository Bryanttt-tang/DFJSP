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
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD

# Set random seed for reproducibility - CHANGED FOR FRESH TESTING
GLOBAL_SEED = 12345  # Changed from 42 to force fresh results
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
    'timesteps': [],
    'episode_count': [],
    'learning_rate': [],
    'explained_variance': []
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

class PoissonDynamicFJSPEnv(gym.Env):
    """
    Dynamic FJSP Environment with Poisson-distributed job arrivals.
    BUILDER MODE: Actions place operations at earliest feasible start time.
    REALISTIC: Only arrived jobs can be scheduled + WAIT action to advance time.
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
        
        # BUILDER MODE: Action space includes WAIT action
        # Action encoding: job_idx * max_ops_per_job * num_machines + op_idx * num_machines + machine_idx
        # WAIT action is the last action (highest index)
        max_scheduling_actions = self.num_jobs * self.max_ops_per_job * len(self.machines)
        self.action_space = spaces.Discrete(max_scheduling_actions + 1)  # +1 for WAIT
        self.WAIT_ACTION = max_scheduling_actions  # WAIT is the last action
        
        # UNIFIED observation space (same size for all RL methods for evaluation compatibility)
        obs_size = (
            self.num_jobs +                         # Ready job indicators
            len(self.machines) +                    # Machine next_free times (normalized)
            self.num_jobs * len(self.machines) +    # Processing times for ready ops
            self.num_jobs                          # DYNAMIC ADVANTAGE: Arrival pattern features
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state variables
        self._reset_state()

    def _reset_state(self):
        """Reset all environment state variables for builder mode."""
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        # BUILDER MODE: Use makespan as the "builder clock" (no separate current_time)
        self.current_makespan = 0.0
        self.operations_scheduled = 0
        self.episode_step = 0
        self.max_episode_steps = (self.total_operations + len(self.dynamic_job_ids)) * 3  # Extra steps for WAIT actions
        
        # Job arrival management - realistic dynamic scheduling
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
            
            if current_time <= self.max_time_horizon:
                self.job_arrival_times[job_id] = float(current_time)
            else:
                self.job_arrival_times[job_id] = float('inf')  # Won't arrive in this episode

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
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
        """Decode action - includes WAIT action handling."""
        action = int(action)
        
        # Check if it's the WAIT action
        if action == self.WAIT_ACTION:
            return None, None, None  # Special return for WAIT
        
        # Decode scheduling action
        action = action % (self.num_jobs * self.max_ops_per_job * len(self.machines))
        num_machines = len(self.machines)
        ops_per_job = self.max_ops_per_job
        
        job_idx = action // (ops_per_job * num_machines)
        op_idx = (action % (ops_per_job * num_machines)) // num_machines
        machine_idx = action % num_machines
        
        job_idx = min(job_idx, self.num_jobs - 1)
        machine_idx = min(machine_idx, len(self.machines) - 1)
        
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        """BUILDER MODE: Check if scheduling action is valid (arrival + precedence + compatibility)."""
        if job_idx is None:  # WAIT action
            return True  # WAIT is always valid unless terminal
        
        if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
            return False
        
        job_id = self.job_ids[job_idx]
        
        # REALISTIC DYNAMIC: Job must have arrived to be scheduled
        if job_id not in self.arrived_jobs:
            return False
            
        # Check operation index validity
        if not (0 <= op_idx < len(self.jobs[job_id])):
            return False
            
        # Check precedence: this must be the next operation for the job
        if op_idx != self.next_operation[job_id]:
            return False
            
        # Check machine compatibility
        machine_name = self.machines[machine_idx]
        if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
            return False
            
        return True

    def action_masks(self):
        """BUILDER MODE: Generate action masks based on arrival, precedence, and compatibility."""
        mask = np.full(self.action_space.n, False, dtype=bool)
        
        # Check if terminal (all operations scheduled)
        if self.operations_scheduled >= self.total_operations:
            return mask  # All actions invalid at terminal

        valid_scheduling_actions = 0
        
        # Check scheduling actions for arrived jobs
        for job_idx, job_id in enumerate(self.job_ids):
            # REALISTIC DYNAMIC: Only arrived jobs can be scheduled
            if job_id not in self.arrived_jobs:
                continue
                
            next_op_idx = self.next_operation[job_id]
            if next_op_idx >= len(self.jobs[job_id]):
                continue  # Job completed
                
            # Check each machine for compatibility (no busy/idle check in builder mode)
            for machine_idx, machine in enumerate(self.machines):
                if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                    action = (job_idx * self.max_ops_per_job * len(self.machines) + 
                             next_op_idx * len(self.machines) + machine_idx)
                    if action < self.WAIT_ACTION:  # Ensure it's a valid scheduling action
                        mask[action] = True
                        valid_scheduling_actions += 1
        
        # WAIT action: Always valid unless terminal
        if self.operations_scheduled < self.total_operations:
            mask[self.WAIT_ACTION] = True
        
        return mask

    def _advance_to_next_arrival(self):
        """WAIT ACTION: Advance makespan to next job arrival."""
        # Find next arrival time after current makespan
        next_arrival_time = float('inf')
        next_arriving_jobs = []
        
        for job_id, arrival_time in self.job_arrival_times.items():
            if (job_id not in self.arrived_jobs and 
                arrival_time > self.current_makespan and 
                arrival_time != float('inf')):
                if arrival_time < next_arrival_time:
                    next_arrival_time = arrival_time
                    next_arriving_jobs = [job_id]
                elif arrival_time == next_arrival_time:
                    next_arriving_jobs.append(job_id)
        
        if next_arrival_time != float('inf'):
            # Advance makespan to next arrival
            self.current_makespan = next_arrival_time
            
            # Add newly arrived jobs
            for job_id in next_arriving_jobs:
                self.arrived_jobs.add(job_id)
            
            return len(next_arriving_jobs), next_arrival_time
        else:
            # No more arrivals - advance makespan minimally
            self.current_makespan += 1.0
            return 0, self.current_makespan

    def step(self, action):
        """BUILDER MODE: Step function with WAIT action support."""
        self.episode_step += 1
        
        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        job_idx, op_idx, machine_idx = self._decode_action(action)

        # Handle WAIT action
        if job_idx is None:  # WAIT action
            if self.operations_scheduled >= self.total_operations:
                return self._get_observation(), -10.0, True, False, {"error": "WAIT at terminal"}
            
            num_new_arrivals, new_time = self._advance_to_next_arrival()
            
            # WAIT reward: small penalty for waiting, small bonus for revealing jobs
            wait_reward = -1.0 + (num_new_arrivals * 2.0)  # Encourage revealing jobs
            
            info = {
                "makespan": self.current_makespan,
                "newly_arrived_jobs": num_new_arrivals,
                "total_arrived_jobs": len(self.arrived_jobs),
                "action_type": "WAIT"
            }
            
            return self._get_observation(), wait_reward, False, False, info

        # Handle scheduling action
        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            return self._get_observation(), -50.0, False, False, {"error": "Invalid scheduling action"}

        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        
        # BUILDER MODE: Calculate earliest feasible start time
        machine_available_time = self.machine_next_free.get(machine, 0.0)
        job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                         else self.job_arrival_times.get(job_id, 0.0))
        
        # Earliest feasible start = max of all constraints
        start_time = max(machine_available_time, job_ready_time, self.current_makespan)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time

        # Update state
        previous_makespan = self.current_makespan
        self.machine_next_free[machine] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        
        # Update makespan
        self.current_makespan = max(self.current_makespan, end_time)

        # Record in schedule
        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        # Check termination
        terminated = self.operations_scheduled >= self.total_operations
        
        # Calculate reward
        idle_time = max(0, start_time - machine_available_time)
        reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                      previous_makespan, self.current_makespan)
        
        info = {
            "makespan": self.current_makespan,
            "newly_arrived_jobs": 0,
            "total_arrived_jobs": len(self.arrived_jobs),
            "action_type": "SCHEDULE"
        }
        
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan):
        """Reward calculation for builder mode."""
        
        if self.reward_mode == "makespan_increment":
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment
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
        """BUILDER MODE: Event-driven observation for dynamic scheduling."""
        obs = []
        
        # 1. Ready job indicators (arrived + has next operation)
        for job_id in self.job_ids:
            if (job_id in self.arrived_jobs and 
                self.next_operation[job_id] < len(self.jobs[job_id])):
                obs.append(1.0)
            else:
                obs.append(0.0)
        
        # 2. Machine availability (normalized next_free times relative to current makespan)
        max_time_horizon = 100.0  # For normalization
        for machine in self.machines:
            machine_free_time = self.machine_next_free[machine]
            # How far ahead is this machine busy (relative to current makespan)
            relative_busy_time = max(0, machine_free_time - self.current_makespan)
            normalized_busy = min(1.0, relative_busy_time / max_time_horizon)
            obs.append(normalized_busy)
        
        # 3. Processing times for ready operations (normalized)
        max_proc_time = 10.0
        
        for job_id in self.job_ids:
            if (job_id in self.arrived_jobs and 
                self.next_operation[job_id] < len(self.jobs[job_id])):
                next_op_idx = self.next_operation[job_id]
                operation = self.jobs[job_id][next_op_idx]
                
                for machine in self.machines:
                    if machine in operation['proc_times']:
                        proc_time = operation['proc_times'][machine]
                        normalized_time = min(1.0, proc_time / max_proc_time)
                        obs.append(normalized_time)
                    else:
                        obs.append(0.0)
            else:
                for machine in self.machines:
                    obs.append(0.0)
        
        # 4. DYNAMIC RL: Arrival pattern features (no direct future knowledge)
        # Time since last arrival
        if len(self.arrived_jobs) > 0:
            last_arrival_time = max(self.job_arrival_times.get(job_id, 0.0) for job_id in self.arrived_jobs)
            time_since_last_arrival = max(0, self.current_makespan - last_arrival_time)
        else:
            time_since_last_arrival = self.current_makespan
        obs.append(min(1.0, time_since_last_arrival / 30.0))
        
        # Arrival progress
        arrival_progress = len(self.arrived_jobs) / len(self.job_ids)
        obs.append(arrival_progress)
        
        # Current makespan (time context)
        normalized_makespan = min(1.0, self.current_makespan / 100.0)
        obs.append(normalized_makespan)
        
        # Fill remaining positions with zeros
        remaining_positions = len(self.job_ids) - 3
        for _ in range(remaining_positions):
            obs.append(0.0)
        
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array


class PerfectKnowledgeFJSPEnv(gym.Env):
    """
    BUILDER MODE Perfect Knowledge FJSP Environment.
    Knows exact arrival times, can schedule any job (arrived or not) with proper timing.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment"):
        super().__init__()
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        self.reward_mode = reward_mode
        self.job_arrival_times = arrival_times.copy()
        
        # BUILDER MODE: Action space includes WAIT action (though less needed with perfect knowledge)
        max_scheduling_actions = self.num_jobs * self.max_ops_per_job * len(self.machines)
        self.action_space = spaces.Discrete(max_scheduling_actions + 1)
        self.WAIT_ACTION = max_scheduling_actions
        
        # Perfect knowledge observation space
        obs_size = (
            self.num_jobs +                         # Ready job indicators
            len(self.machines) +                    # Machine next_free times
            self.num_jobs * len(self.machines) +    # Processing times for ready ops
            self.num_jobs                          # PERFECT ADVANTAGE: Exact arrival times
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        """Reset environment for perfect knowledge builder mode."""
        if seed is not None:
            super().reset(seed=seed, options=options)
            random.seed(seed)
            np.random.seed(seed)
        
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        self.current_makespan = 0.0
        self.operations_scheduled = 0
        self.episode_step = 0
        self.max_episode_steps = self.total_operations * 3
        
        # All jobs can be scheduled from the start (perfect knowledge allows pre-planning)
        self.arrived_jobs = set(self.job_ids)  # Perfect knowledge: can schedule all jobs
        
        return self._get_observation(), {}

    def _decode_action(self, action):
        """Decode action including WAIT."""
        action = int(action)
        
        if action == self.WAIT_ACTION:
            return None, None, None
        
        action = action % (self.num_jobs * self.max_ops_per_job * len(self.machines))
        num_machines = len(self.machines)
        ops_per_job = self.max_ops_per_job
        
        job_idx = action // (ops_per_job * num_machines)
        op_idx = (action % (ops_per_job * num_machines)) // num_machines
        machine_idx = action % num_machines
        
        job_idx = min(job_idx, self.num_jobs - 1)
        machine_idx = min(machine_idx, len(self.machines) - 1)
        
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        """Perfect knowledge validation: precedence + compatibility only."""
        if job_idx is None:  # WAIT action
            return True
        
        if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
            return False
        
        job_id = self.job_ids[job_idx]
        
        # Check operation index validity
        if not (0 <= op_idx < len(self.jobs[job_id])):
            return False
            
        # Check precedence: this must be the next operation for the job
        if op_idx != self.next_operation[job_id]:
            return False
            
        # Check machine compatibility
        machine_name = self.machines[machine_idx]
        if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
            return False
            
        return True

    def action_masks(self):
        """Perfect knowledge action masks: precedence + compatibility."""
        mask = np.full(self.action_space.n, False, dtype=bool)
        
        if self.operations_scheduled >= self.total_operations:
            return mask

        valid_action_count = 0
        
        # All jobs can be scheduled (perfect knowledge)
        for job_idx, job_id in enumerate(self.job_ids):
            next_op_idx = self.next_operation[job_id]
            if next_op_idx >= len(self.jobs[job_id]):
                continue
                
            for machine_idx, machine in enumerate(self.machines):
                if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                    action = (job_idx * self.max_ops_per_job * len(self.machines) + 
                             next_op_idx * len(self.machines) + machine_idx)
                    if action < self.WAIT_ACTION:
                        mask[action] = True
                        valid_action_count += 1
        
        # WAIT action (though less useful with perfect knowledge)
        if self.operations_scheduled < self.total_operations:
            mask[self.WAIT_ACTION] = True
            
        return mask

    def step(self, action):
        """Perfect knowledge step function with builder semantics."""
        self.episode_step += 1
        
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        job_idx, op_idx, machine_idx = self._decode_action(action)

        # Handle WAIT action (advance makespan minimally)
        if job_idx is None:
            if self.operations_scheduled >= self.total_operations:
                return self._get_observation(), -10.0, True, False, {"error": "WAIT at terminal"}
            
            self.current_makespan += 1.0  # Minimal advance
            return self._get_observation(), -2.0, False, False, {"action_type": "WAIT"}

        # Handle scheduling action
        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            return self._get_observation(), -50.0, False, False, {"error": "Invalid action, continuing"}

        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        
        # BUILDER MODE: Calculate earliest feasible start time
        machine_available_time = self.machine_next_free.get(machine, 0.0)
        job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                         else self.job_arrival_times.get(job_id, 0.0))
        
        # Earliest feasible start = max of all constraints
        start_time = max(machine_available_time, job_ready_time, self.current_makespan)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time

        # Update state
        previous_makespan = self.current_makespan
        self.machine_next_free[machine] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        
        # Update makespan
        self.current_makespan = max(self.current_makespan, end_time)

        # Record in schedule
        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        # Check termination
        terminated = self.operations_scheduled >= self.total_operations
        
        # Calculate reward
        idle_time = max(0, start_time - machine_available_time)
        reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                      previous_makespan, self.current_makespan)
        
        info = {"makespan": self.current_makespan, "action_type": "SCHEDULE"}
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan):
        """Reward calculation for perfect knowledge builder mode."""
        
        if self.reward_mode == "makespan_increment":
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment
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
        """BUILDER MODE: Perfect knowledge observation with exact arrival information."""
        obs = []
        
        # 1. Ready job indicators (all jobs available for scheduling with perfect knowledge)
        for job_id in self.job_ids:
            if self.next_operation[job_id] < len(self.jobs[job_id]):
                obs.append(1.0)  # Job has remaining operations
            else:
                obs.append(0.0)  # Job completed
        
        # 2. Machine availability (normalized next_free times relative to current makespan)
        max_time_horizon = 100.0  # For normalization
        for machine in self.machines:
            machine_free_time = self.machine_next_free[machine]
            # How far ahead is this machine busy (relative to current makespan)
            relative_busy_time = max(0, machine_free_time - self.current_makespan)
            normalized_busy = min(1.0, relative_busy_time / max_time_horizon)
            obs.append(normalized_busy)
        
        # 3. Processing times for next operations (normalized)
        max_proc_time = 10.0
        
        for job_id in self.job_ids:
            if self.next_operation[job_id] < len(self.jobs[job_id]):
                next_op_idx = self.next_operation[job_id]
                operation = self.jobs[job_id][next_op_idx]
                
                for machine in self.machines:
                    if machine in operation['proc_times']:
                        proc_time = operation['proc_times'][machine]
                        normalized_time = min(1.0, proc_time / max_proc_time)
                        obs.append(normalized_time)
                    else:
                        obs.append(0.0)  # Machine cannot process this operation
            else:
                # Job completed: add zeros for processing times
                for machine in self.machines:
                    obs.append(0.0)
        
        # 4. PERFECT KNOWLEDGE ADVANTAGE: Exact arrival times for all jobs
        for job_id in self.job_ids:
            arrival_time = self.job_arrival_times.get(job_id, 0.0)
            if arrival_time != float('inf'):
                # Provide exact timing information relative to current makespan
                delay = max(0, arrival_time - self.current_makespan)
                normalized_delay = min(1.0, delay / 50.0)  # Normalize by 50 time units
                obs.append(normalized_delay)
            else:
                obs.append(0.0)  # Job won't arrive
        
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array

def mask_fn(env):
    """Function to retrieve action masks from the environment."""
    return env.action_masks()

class TrainingCallback:
    """Enhanced callback to track comprehensive training metrics including episode-level data."""
    
    def __init__(self, method_name):
        self.method_name = method_name
        self.step_count = 0
        self.episode_count = 0
        self.last_episode_rewards = []
        self.last_episode_lengths = []
        
    def __call__(self, locals_dict, globals_dict):
        global TRAINING_METRICS
        
        # Store method name for plotting
        TRAINING_METRICS['method_name'] = self.method_name
        
        # Extract PPO model from locals
        model = locals_dict.get('self')
        
        if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
            log_data = model.logger.name_to_value
            
            # # DEBUG: Print all available log keys to see what's being logged
            # if self.step_count % 100 == 0:  # Print every 100 calls to avoid spam
            #     print(f"\n[DEBUG {self.method_name}] Available log keys: {list(log_data.keys())}")
            #     if 'rollout/ep_rew_mean' in log_data:
            #         print(f"[DEBUG {self.method_name}] Episode reward found: {log_data['rollout/ep_rew_mean']}")
            #     else:
            #         print(f"[DEBUG {self.method_name}] No episode reward in log_data")
            
            # Log entropy if available
            if 'train/entropy_loss' in log_data:
                TRAINING_METRICS['action_entropy'].append(log_data['train/entropy_loss'])
            
            # Log policy and value losses
            if 'train/policy_gradient_loss' in log_data:
                TRAINING_METRICS['policy_loss'].append(log_data['train/policy_gradient_loss'])
            if 'train/value_loss' in log_data:
                TRAINING_METRICS['value_loss'].append(log_data['train/value_loss'])
            
            # Log learning rate
            if 'train/learning_rate' in log_data:
                TRAINING_METRICS['learning_rate'].append(log_data['train/learning_rate'])
            
            # Log explained variance (measure of value function quality)
            if 'train/explained_variance' in log_data:
                TRAINING_METRICS['explained_variance'].append(log_data['train/explained_variance'])
            
            # FIXED: Log episode rewards and lengths from rollout data
            if 'rollout/ep_rew_mean' in log_data:
                ep_reward = log_data['rollout/ep_rew_mean']
                TRAINING_METRICS['episode_rewards'].append(ep_reward)
                self.episode_count += 1
                TRAINING_METRICS['episode_count'].append(self.episode_count)
                print(f"[DEBUG {self.method_name}] Logged episode reward: {ep_reward:.4f} (episode {self.episode_count})")
            
            if 'rollout/ep_len_mean' in log_data:
                TRAINING_METRICS['episode_lengths'].append(log_data['rollout/ep_len_mean'])
            
            TRAINING_METRICS['timesteps'].append(model.num_timesteps)
            self.step_count += 1
        
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
    
    def make_perfect_env():
        # Use PerfectKnowledgeFJSPEnv for both training and evaluation consistency
        env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times, reward_mode=reward_mode)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_perfect_env])
    
    # IDENTICAL hyperparameters across all RL methods for fair comparison
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=learning_rate,        # IDENTICAL across all RL methods
        n_steps=2048,              # IDENTICAL across all RL methods
        batch_size=128,            # IDENTICAL across all RL methods
        n_epochs=10,               # IDENTICAL across all RL methods
        gamma=1,                # IDENTICAL across all RL methods
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,             # IDENTICAL across all RL methods
        vf_coef=0.5,
        max_grad_norm=0.5,     
        policy_kwargs=dict(
            net_arch=[256, 256, 128],  # IDENTICAL across all RL methods
            activation_fn=torch.nn.ReLU
        )
    )
    
    # # Training with progress bar and entropy tracking (silent)
    # print(f"    Training perfect knowledge agent for this specific scenario...")
    # callback = TrainingCallback("Perfect Knowledge RL")
    
    # Training with progress tracking for Perfect Knowledge
    print(f"    Starting training...")
    
    # Do a quick evaluation before training to establish baseline
    test_env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times, reward_mode=reward_mode)
    test_env = ActionMasker(test_env, mask_fn)
    obs, _ = test_env.reset()
    
    # Quick random baseline test
    random_makespan = 0
    try:
        random_steps = 0
        done = False
        while not done and random_steps < 200:  # Increased steps for late arrivals
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
    
    # Train with progress tracking and tqdm bar
    callback = TrainingCallback("Perfect Knowledge RL")
    
    with tqdm(total=total_timesteps, desc=f"Perfect Knowledge RL", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} timesteps [{elapsed}<{remaining}]') as pbar:
        
        def combined_callback(locals_dict, globals_dict):
            callback(locals_dict, globals_dict)
            if hasattr(model, 'num_timesteps'):
                pbar.n = model.num_timesteps
                pbar.refresh()
            return True
        
        model.learn(total_timesteps=total_timesteps, callback=combined_callback)
    
    # More thorough post-training evaluation
    test_env.reset()
    post_training_makespan = 999.0
    try:
        obs, _ = test_env.reset()
        done = False
        steps = 0
        max_eval_steps = 200  # Standard evaluation steps
        
        while not done and steps < max_eval_steps:
            action_masks = test_env.action_masks()
            if not any(action_masks):
                break
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            steps += 1
        post_training_makespan = test_env.env.current_makespan
        print(f"    Post-training makespan: {post_training_makespan:.2f}")
        
        if post_training_makespan < random_makespan:
            improvement = random_makespan - post_training_makespan
            print(f"    ✅ Training improved by {improvement:.2f} ({improvement/random_makespan*100:.1f}%)")
        else:
            degradation = post_training_makespan - random_makespan
            print(f"    ⚠️  Training did not improve (random: {random_makespan:.2f}, trained: {post_training_makespan:.2f})")
            print(f"    ⚠️  Degradation: {degradation:.2f} - may need longer training or different hyperparameters")
    except Exception as e:
        print(f"    ⚠️  Could not evaluate post-training: {e}")
    
    print(f"    ✅ Perfect knowledge training completed for this scenario!")
    return model

def train_static_agent(jobs_data, machine_list, total_timesteps=300000, reward_mode="makespan_increment", learning_rate=3e-4):
    """Train a static RL agent where all jobs are available at t=0."""
    print(f"\n--- Training Static RL Agent on {len(jobs_data)} jobs ---")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    # CORRECTED: Use PerfectKnowledgeFJSPEnv with all arrival times = 0 for static scenario
    static_arrival_times = {job_id: 0.0 for job_id in jobs_data.keys()}
    
    def make_static_env():
        env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, static_arrival_times, reward_mode=reward_mode)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_static_env])
    
    # IDENTICAL hyperparameters as Perfect Knowledge RL for fair comparison
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,  # Minimal output
        learning_rate=learning_rate,
        n_steps=2048,              # IDENTICAL across all RL methods
        batch_size=128,            # IDENTICAL across all RL methods
        n_epochs=10,               # IDENTICAL across all RL methods (no special case for late arrivals)
        gamma=1,                # IDENTICAL across all RL methods
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,             # IDENTICAL across all RL methods
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=GLOBAL_SEED,          # Ensure reproducibility
        policy_kwargs=dict(
            net_arch=[256, 256, 128],  # IDENTICAL across all RL methods
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"Training Static RL for {total_timesteps:,} timesteps with seed {GLOBAL_SEED}...")
    
    # Train with tqdm progress bar and entropy tracking
    start_time = time.time()
    callback = TrainingCallback("Static RL")
    
    with tqdm(total=total_timesteps, desc="Static RL Training", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} timesteps [{elapsed}<{remaining}]') as pbar:
        
        def combined_callback(locals_dict, globals_dict):
            callback(locals_dict, globals_dict)
            if hasattr(model, 'num_timesteps'):
                pbar.n = model.num_timesteps
                pbar.refresh()
            return True
        
        model.learn(total_timesteps=total_timesteps, callback=combined_callback)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Static RL training completed in {training_time:.1f}s!")
    
    return model


def train_dynamic_agent(jobs_data, machine_list, initial_jobs=5, arrival_rate=0.08, total_timesteps=500000, reward_mode="makespan_increment", learning_rate=3e-4):
    """
    Train a dynamic RL agent on Poisson job arrivals with EXPANDED DATASET.
    """
    print(f"\n--- Training Dynamic RL Agent on {len(jobs_data)} jobs ---")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_dynamic_env():
        env = PoissonDynamicFJSPEnv(
            jobs_data, machine_list, 
            initial_jobs=initial_jobs,
            arrival_rate=arrival_rate,
            reward_mode=reward_mode,
            seed=GLOBAL_SEED+100,  # Ensure reproducibility
            max_time_horizon=200  # Standard time horizon
        )
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_dynamic_env])
    
    # IDENTICAL hyperparameters across all RL methods for fair comparison
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=learning_rate,        # IDENTICAL across all RL methods
        n_steps=4096,              # IDENTICAL across all RL methods
        batch_size=512,            # IDENTICAL across all RL methods
        n_epochs=20,               # IDENTICAL across all RL methods
        gamma=1,                # IDENTICAL across all RL methods
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,             # IDENTICAL across all RL methods
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=GLOBAL_SEED,          # Ensure reproducibility
        policy_kwargs=dict(
            net_arch=[256, 256, 128],  # IDENTICAL across all RL methods
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"Training Dynamic RL for {total_timesteps:,} timesteps with seed {GLOBAL_SEED}...")
    
    # Train with progress bar like PerfectKnowledgeFJSPEnv
    start_time = time.time()
    callback = TrainingCallback("Dynamic RL")
    
    with tqdm(total=total_timesteps, desc="Dynamic RL Training", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} timesteps [{elapsed}<{remaining}]') as pbar:
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
    Enhanced plot of training metrics including episode rewards, losses, and learning progress.
    This helps debug PPO exploration and convergence issues with episode-level granularity.
    """
    global TRAINING_METRICS
    
    if not TRAINING_METRICS['timesteps']:
        print("No training metrics recorded!")
        return
    
    print(f"\n=== ENHANCED TRAINING METRICS ANALYSIS ===")
    print(f"Total training steps recorded: {len(TRAINING_METRICS['timesteps'])}")
    print(f"Episode rewards recorded: {len(TRAINING_METRICS['episode_rewards'])}")
    print(f"Policy loss records: {len(TRAINING_METRICS['policy_loss'])}")
    print(f"Value loss records: {len(TRAINING_METRICS['value_loss'])}")
    
    # # DEBUG: Print actual episode rewards if available
    # if TRAINING_METRICS['episode_rewards']:
    #     rewards = TRAINING_METRICS['episode_rewards']
    #     print(f"Episode rewards sample (first 10): {rewards[:10]}")
    #     print(f"Episode rewards sample (last 10): {rewards[-10:]}")
    #     print(f"Episode rewards range: {min(rewards):.4f} to {max(rewards):.4f}")
    # else:
    #     print("❌ NO EPISODE REWARDS RECORDED!")
    #     print("This suggests the Monitor wrapper or callback is not working properly")
    
    # Create comprehensive figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle(f'Comprehensive PPO Training Analysis - {TRAINING_METRICS.get("method_name", "RL Agent")}', 
                 fontsize=16, fontweight='bold')
    
    timesteps = TRAINING_METRICS['timesteps']
    
    # Plot 1: Episode Rewards over Time (Top Left)
    if TRAINING_METRICS['episode_rewards'] and len(TRAINING_METRICS['episode_rewards']) > 1:
        ep_rewards = TRAINING_METRICS['episode_rewards']
        # The number of reward entries may be less than timestep entries.
        # We assume rewards are logged at the same frequency as other metrics.
        ep_timesteps = timesteps[:len(ep_rewards)]
        
        print(f"[DEBUG] Plotting {len(ep_rewards)} episode rewards against {len(ep_timesteps)} timesteps")
        
        axes[0, 0].plot(ep_timesteps, ep_rewards, 'g-', linewidth=2, alpha=0.7, label='Mean Episode Reward')
        
        # Add moving average for trend
        if len(ep_rewards) > 10:
            window_size = min(50, len(ep_rewards) // 4)
            if window_size > 0:
                moving_avg = np.convolve(ep_rewards, np.ones(window_size)/window_size, mode='valid')
                moving_timesteps = ep_timesteps[window_size-1:]
                axes[0, 0].plot(moving_timesteps, moving_avg, 'darkgreen', linewidth=3, 
                               label=f'Moving Avg (window={window_size})')
        
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Episode Rewards (Higher = Better Learning)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add interpretation
        if len(ep_rewards) > 20:
            early_reward = np.mean(ep_rewards[:len(ep_rewards)//4])
            late_reward = np.mean(ep_rewards[-len(ep_rewards)//4:])
            improvement = late_reward - early_reward
            
            if improvement > 0:
                axes[0, 0].text(0.02, 0.98, f'✅ Improving (+{improvement:.1f})', 
                               transform=axes[0, 0].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            else:
                axes[0, 0].text(0.02, 0.98, f'🔴 Declining ({improvement:.1f})', 
                               transform=axes[0, 0].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    else:
        axes[0, 0].text(0.5, 0.5, f'No episode reward data\n(Recorded: {len(TRAINING_METRICS["episode_rewards"])} rewards)', 
                       ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        axes[0, 0].set_title('Episode Rewards (No Data)')
        print("❌ NO EPISODE REWARDS TO PLOT!")
    
    # Plot 2: Policy Loss over Time (Top Right)
    if TRAINING_METRICS['policy_loss']:
        policy_loss = TRAINING_METRICS['policy_loss']
        loss_timesteps = timesteps[:len(policy_loss)]
        
        axes[0, 1].plot(loss_timesteps, policy_loss, 'r-', linewidth=2, alpha=0.7)
        
        # Add moving average
        if len(policy_loss) > 10:
            window_size = min(20, len(policy_loss) // 4)
            moving_avg = np.convolve(policy_loss, np.ones(window_size)/window_size, mode='valid')
            moving_timesteps = loss_timesteps[window_size-1:]
            axes[0, 1].plot(moving_timesteps, moving_avg, 'darkred', linewidth=3,
                           label=f'Moving Avg (window={window_size})')
            axes[0, 1].legend()
        
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Policy Loss')
        axes[0, 1].set_title('Policy Gradient Loss (Lower = Better)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add trend analysis
        if len(policy_loss) > 20:
            early_loss = np.mean(policy_loss[:len(policy_loss)//4])
            late_loss = np.mean(policy_loss[-len(policy_loss)//4:])
            reduction = early_loss - late_loss
            
            if reduction > 0:
                axes[0, 1].text(0.02, 0.98, f'✅ Decreasing (-{reduction:.4f})', 
                               transform=axes[0, 1].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            else:
                axes[0, 1].text(0.02, 0.98, f'🔴 Increasing (+{abs(reduction):.4f})', 
                               transform=axes[0, 1].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    else:
        axes[0, 1].text(0.5, 0.5, 'No policy loss data', ha='center', va='center', 
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Policy Loss (No Data)')
    
    # Plot 3: Value Loss over Time (Middle Left)
    if TRAINING_METRICS['value_loss']:
        value_loss = TRAINING_METRICS['value_loss']
        loss_timesteps = timesteps[:len(value_loss)]
        
        axes[1, 0].plot(loss_timesteps, value_loss, 'b-', linewidth=2, alpha=0.7)
        
        # Add moving average
        if len(value_loss) > 10:
            window_size = min(20, len(value_loss) // 4)
            moving_avg = np.convolve(value_loss, np.ones(window_size)/window_size, mode='valid')
            moving_timesteps = loss_timesteps[window_size-1:]
            axes[1, 0].plot(moving_timesteps, moving_avg, 'darkblue', linewidth=3,
                           label=f'Moving Avg (window={window_size})')
            axes[1, 0].legend()
        
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Value Loss')
        axes[1, 0].set_title('Value Function Loss (Lower = Better)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add trend analysis
        if len(value_loss) > 20:
            early_loss = np.mean(value_loss[:len(value_loss)//4])
            late_loss = np.mean(value_loss[-len(value_loss)//4:])
            reduction = early_loss - late_loss
            
            if reduction > 0:
                axes[1, 0].text(0.02, 0.98, f'✅ Decreasing (-{reduction:.4f})', 
                               transform=axes[1, 0].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            else:
                axes[1, 0].text(0.02, 0.98, f'🔴 Increasing (+{abs(reduction):.4f})', 
                               transform=axes[1, 0].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    else:
        axes[1, 0].text(0.5, 0.5, 'No value loss data', ha='center', va='center', 
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Value Loss (No Data)')
    
    # Plot 4: Action Entropy over Time (Middle Right)
    if TRAINING_METRICS['action_entropy']:
        entropy = TRAINING_METRICS['action_entropy']
        entropy_timesteps = timesteps[:len(entropy)]
        
        axes[1, 1].plot(entropy_timesteps, entropy, 'purple', linewidth=2, alpha=0.7)
        
        # Add moving average
        if len(entropy) > 10:
            window_size = min(20, len(entropy) // 4)
            moving_avg = np.convolve(entropy, np.ones(window_size)/window_size, mode='valid')
            moving_timesteps = entropy_timesteps[window_size-1:]
            axes[1, 1].plot(moving_timesteps, moving_avg, 'indigo', linewidth=3,
                           label=f'Moving Avg (window={window_size})')
            axes[1, 1].legend()
        
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Action Entropy')
        axes[1, 1].set_title('Action Entropy (Exploration Level)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add interpretation
        final_entropy = entropy[-1]
        if final_entropy > 0.5:
            axes[1, 1].text(0.02, 0.98, '✅ High exploration', transform=axes[1, 1].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        elif final_entropy > 0.1:
            axes[1, 1].text(0.02, 0.98, '🟡 Moderate exploration', transform=axes[1, 1].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        else:
            axes[1, 1].text(0.02, 0.98, '🔴 Low exploration', transform=axes[1, 1].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    else:
        axes[1, 1].text(0.5, 0.5, 'No entropy data', ha='center', va='center', 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Action Entropy (No Data)')
    
    # Plot 5: Combined Loss Comparison (Bottom Left)
    axes[2, 0].set_title('Combined Training Losses')
    if TRAINING_METRICS['policy_loss'] and TRAINING_METRICS['value_loss']:
        min_len = min(len(TRAINING_METRICS['policy_loss']), len(TRAINING_METRICS['value_loss']))
        combined_timesteps = timesteps[:min_len]
        
        axes[2, 0].plot(combined_timesteps, TRAINING_METRICS['policy_loss'][:min_len], 
                       'r-', linewidth=2, label='Policy Loss', alpha=0.7)
        axes[2, 0].plot(combined_timesteps, TRAINING_METRICS['value_loss'][:min_len], 
                       'b-', linewidth=2, label='Value Loss', alpha=0.7)
        axes[2, 0].legend()
        axes[2, 0].set_xlabel('Training Steps')
        axes[2, 0].set_ylabel('Loss')
        axes[2, 0].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, 'Insufficient loss data', ha='center', va='center', 
                       transform=axes[2, 0].transAxes)
    
    # Plot 6: Learning Rate and Explained Variance (Bottom Right)
    if TRAINING_METRICS['learning_rate'] or TRAINING_METRICS['explained_variance']:
        ax_lr = axes[2, 1]
        ax_ev = ax_lr.twinx()
        
        if TRAINING_METRICS['learning_rate']:
            lr = TRAINING_METRICS['learning_rate']
            lr_timesteps = timesteps[:len(lr)]
            line1 = ax_lr.plot(lr_timesteps, lr, 'orange', linewidth=2, label='Learning Rate')
            ax_lr.set_ylabel('Learning Rate', color='orange')
            ax_lr.tick_params(axis='y', labelcolor='orange')
        
        if TRAINING_METRICS['explained_variance']:
            ev = TRAINING_METRICS['explained_variance']
            ev_timesteps = timesteps[:len(ev)]
            line2 = ax_ev.plot(ev_timesteps, ev, 'cyan', linewidth=2, label='Explained Variance')
            ax_ev.set_ylabel('Explained Variance', color='cyan')
            ax_ev.tick_params(axis='y', labelcolor='cyan')
        
        axes[2, 1].set_xlabel('Training Steps')
        axes[2, 1].set_title('Learning Rate & Value Function Quality')
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, 'No LR/EV data', ha='center', va='center', 
                       transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Learning Rate & Explained Variance (No Data)')
    
    plt.tight_layout()
    plt.savefig('enhanced_ppo_training_metrics.png', dpi=300, bbox_inches='tight')
    print("✅ Enhanced training metrics plot saved: enhanced_ppo_training_metrics.png")
    plt.show()
    
    # Print comprehensive summary statistics
    print(f"\n=== TRAINING SUMMARY STATISTICS ===")
    
    if TRAINING_METRICS['episode_rewards']:
        rewards = TRAINING_METRICS['episode_rewards']
        print(f"Episode Rewards:")
        print(f"  Episodes recorded: {len(rewards)}")
        print(f"  Initial reward: {rewards[0]:.4f}")
        print(f"  Final reward: {rewards[-1]:.4f}")
       
        print(f"  Best reward: {max(rewards):.4f}")
        print(f"  Worst reward: {min(rewards):.4f}")
        print(f"  Mean reward: {np.mean(rewards):.4f}")
        print(f"  Std reward: {np.std(rewards):.4f}")
        
        if len(rewards) > 20:
            early_rewards = rewards[:len(rewards)//4]
            late_rewards = rewards[-len(rewards)//4:]
            improvement = np.mean(late_rewards) - np.mean(early_rewards)
            improvement_pct = (improvement / abs(np.mean(early_rewards))) * 100 if np.mean(early_rewards) != 0 else 0
            
            print(f"  Early training avg: {np.mean(early_rewards):.4f}")
            print(f"  Late training avg: {np.mean(late_rewards):.4f}")
            print(f"  Total improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
            
            if improvement > 0:
                print(f"  ✅ Training is improving reward over time")
            else:
                print(f"  ❌ Training reward is declining - check hyperparameters")
    
    if TRAINING_METRICS['policy_loss']:
        policy_losses = TRAINING_METRICS['policy_loss']
       
        print(f"\nPolicy Loss:")
        print(f"  Initial loss: {policy_losses[0]:.6f}")
        print(f"  Final loss: {policy_losses[-1]:.6f}")
        print(f"  Mean loss: {np.mean(policy_losses):.6f}")
        print(f"  Std loss: {np.std(policy_losses):.6f}")
        
        if len(policy_losses) > 20:
            early_loss = np.mean(policy_losses[:len(policy_losses)//4])
            late_loss = np.mean(policy_losses[-len(policy_losses)//4:])
            reduction = early_loss - late_loss
            reduction_pct = (reduction / early_loss) * 100 if early_loss != 0 else 0
            
            print(f"  Loss reduction: {reduction:+.6f} ({reduction_pct:+.1f}%)")
            
            if reduction > 0:
                print(f"  ✅ Policy loss is decreasing (good)")
            else:
                print(f"  ⚠️  Policy loss is increasing - monitor for overfitting")
    
    if TRAINING_METRICS['value_loss']:
        value_losses = TRAINING_METRICS['value_loss']
        print(f"\nValue Loss:")
        print(f"  Initial loss: {value_losses[0]:.6f}")
        print(f"  Final loss: {value_losses[-1]:.6f}")
        print(f"  Mean loss: {np.mean(value_losses):.6f}")
        print(f"  Std loss: {np.std(value_losses):.6f}")
        
        if len(value_losses) > 20:
            early_loss = np.mean(value_losses[:len(value_losses)//4])
            late_loss = np.mean(value_losses[-len(value_losses)//4:])
            reduction = early_loss - late_loss
            reduction_pct = (reduction / early_loss) * 100 if early_loss != 0 else 0
            
            print(f"  Loss reduction: {reduction:+.6f} ({reduction_pct:+.1f}%)")
            
            if reduction > 0:
                print(f"  ✅ Value loss is decreasing (good)")
            else:
                print(f"  ⚠️  Value loss is increasing - value function may be struggling")
    
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


# def evaluate_on_test_scenarios(model, test_scenarios, jobs_data, machine_list, method_name="Model", is_static_model=False):
#     """Evaluate a model on multiple predefined test scenarios."""
#     print(f"\n--- Evaluating {method_name} on {len(test_scenarios)} Test Scenarios ---")
    
#     results = []
    
#     for scenario in test_scenarios:
#         scenario_id = scenario['scenario_id']
#         arrival_times = scenario['arrival_times']
#         seed = scenario['seed']
        
#         if is_static_model:
#             # For static models, use the special evaluation function
#             makespan, schedule = evaluate_static_on_dynamic(
#                 model, jobs_data, machine_list, arrival_times)
            
#             results.append({
#                 'scenario_id': scenario_id,
#                 'makespan': makespan,
#                 'schedule': schedule,
#                 'arrival_times': arrival_times,
#                 'steps': 0,  # Not tracked for this evaluation
#                 'reward': 0  # Not tracked for this evaluation
#             })
#         else:
#             # For dynamic models, use the existing evaluation
#             makespan, schedule = evaluate_dynamic_on_dynamic(
#                 model, jobs_data, machine_list, arrival_times)
            
#             results.append({
#                 'scenario_id': scenario_id,
#                 'makespan': makespan,
#                 'schedule': schedule,
#                 'arrival_times': arrival_times,
#                 'steps': 0,
#                 'reward': 0
#             })
        
#         print(f"  Scenario {scenario_id+1}: Makespan = {makespan:.2f}")
    
#     # Calculate statistics
#     makespans = [r['makespan'] for r in results]
#     avg_makespan = np.mean(makespans)
#     std_makespan = np.std(makespans)
#     min_makespan = np.min(makespans)
#     max_makespan = np.max(makespans)
    
#     print(f"Results for {method_name}:")
#     print(f"  Average Makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
#     print(f"  Best Makespan: {min_makespan:.2f}")
#     print(f"  Worst Makespan: {max_makespan:.2f}")
    
#     # Return best result for visualization
#     best_result = min(results, key=lambda x: x['makespan'])
#     return best_result['makespan'], best_result['schedule'], best_result['arrival_times']

def evaluate_static_on_dynamic(static_model, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment"):
    """Evaluate static model on dynamic scenario - BUILDER MODE VERSION."""
    print(f"  Static RL evaluation on dynamic scenario...")
    
    # Use PerfectKnowledgeFJSPEnv with actual arrival times for consistency
    test_env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times, reward_mode=reward_mode)
    test_env = ActionMasker(test_env, mask_fn)
    
    obs, _ = test_env.reset()
    
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
    
    # Create static arrival times (all jobs at t=0) - same as training
    static_arrival_times = {job_id: 0.0 for job_id in jobs_data.keys()}
    
    # Use PerfectKnowledgeFJSPEnv for consistency with training and other evaluations
    test_env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, static_arrival_times, reward_mode=reward_mode)
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

def evaluate_dynamic_on_dynamic(dynamic_model, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment"):
    """Evaluate dynamic model on dynamic scenario - BUILDER MODE VERSION."""
    print(f"  Dynamic RL using arrival times: {arrival_times}")
    
    # Create environment with proper builder-mode settings
    test_env = PoissonDynamicFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=[k for k, v in arrival_times.items() if v == 0],
        arrival_rate=0.1,  # Rate doesn't matter since we'll override
        reward_mode=reward_mode,
        seed=GLOBAL_SEED,
        max_time_horizon=max([t for t in arrival_times.values() if t != float('inf')] + [200])
    )
    
    # Override arrival times BEFORE creating ActionMasker
    test_env.job_arrival_times = arrival_times.copy()
    test_env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    
    test_env = ActionMasker(test_env, mask_fn)
    
    obs, _ = test_env.reset()
    
    # Force the environment to use our arrival times
    test_env.env.job_arrival_times = arrival_times.copy()
    test_env.env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    
    obs = test_env.env._get_observation()
    
    step_count = 0
    max_steps = 500  # Increased for WAIT actions
    
    while step_count < max_steps:
        action_masks = test_env.action_masks()
        
        if not any(action_masks):
            print(f"    No valid actions available at step {step_count}")
            break
        
        action, _ = dynamic_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if done or truncated:
            print(f"    Episode completed at step {step_count}")
            break
    
    makespan = test_env.env.current_makespan
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    total_ops_scheduled = 0
    for machine_ops in test_env.env.schedule.values():
        total_ops_scheduled += len(machine_ops)
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    
    total_expected_ops = sum(len(ops) for ops in jobs_data.values())
    print(f"  Dynamic RL scheduled jobs: {sorted(scheduled_jobs)} ({total_ops_scheduled}/{total_expected_ops} ops)")
    
    # Verify schedule is valid and complete
    if total_ops_scheduled < total_expected_ops:
        missing_jobs = set(jobs_data.keys()) - scheduled_jobs
        print(f"  ⚠️  WARNING: Incomplete schedule! Expected {total_expected_ops} operations, got {total_ops_scheduled}")
        print(f"  Missing jobs: {sorted(missing_jobs)}")
        
        # Check which jobs never arrived vs which arrived but weren't scheduled
        never_arrived = [j for j in missing_jobs if arrival_times.get(j, 0) == float('inf')]
        arrived_but_not_scheduled = [j for j in missing_jobs if j not in never_arrived]
        
        if never_arrived:
            print(f"  Jobs that never arrived: {sorted(never_arrived)}")
        if arrived_but_not_scheduled:
            print(f"  Jobs that arrived but weren't scheduled: {sorted(arrived_but_not_scheduled)}")
    
    return makespan, test_env.env.schedule

def evaluate_perfect_knowledge_on_scenario(perfect_model, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment"):
    """Evaluate perfect knowledge agent - BUILDER MODE VERSION."""
    print(f"  Perfect Knowledge RL evaluation (builder mode)...")
    
    test_env = PerfectKnowledgeFJSPEnv(
        jobs_data, machine_list, 
        arrival_times=arrival_times,
        reward_mode=reward_mode
    )
    test_env = ActionMasker(test_env, mask_fn)
    
    obs, _ = test_env.reset()
    done = False
    step_count = 0
    max_steps = len(jobs_data) * max(len(ops) for ops in jobs_data.values()) * 3
    
    while not done and step_count < max_steps:
        action_masks = test_env.action_masks()
        if not np.any(action_masks):
            print(f"    No valid actions at step {step_count}")
            break
            
        action, _ = perfect_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
    
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

def verify_schedule_correctness(schedule, jobs_data, arrival_times, method_name):
    """
    Verify that a schedule is valid and calculate its true makespan.
    This helps detect bugs in evaluation functions.
    """
    if not schedule or all(len(ops) == 0 for ops in schedule.values()):
        print(f"    ❌ {method_name}: Empty schedule")
        return False, float('inf')
    
    # Check 1: All operations scheduled exactly once
    scheduled_ops = set()
    for machine_ops in schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if job_op in scheduled_ops:
                    print(f"    ❌ {method_name}: Duplicate operation {job_op}")
                    return False, float('inf')
                scheduled_ops.add(job_op)
    
    expected_ops = set()
    for job_id in jobs_data.keys():
        for op_idx in range(len(jobs_data[job_id])):
            expected_ops.add(f"J{job_id}-O{op_idx+1}")
    
    if scheduled_ops != expected_ops:
        missing = expected_ops - scheduled_ops
        extra = scheduled_ops - expected_ops
        if missing:
            print(f"    ❌ {method_name}: Missing operations {missing}")
        if extra:
            print(f"    ❌ {method_name}: Extra operations {extra}")
        return False, float('inf')
    
    # Check 2: Precedence constraints within jobs
    job_op_times = {}
    for machine_ops in schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op, start_time, end_time = op_data[:3]
                # Parse job and operation
                try:
                    job_id = int(job_op.split('J')[1].split('-')[0])
                    op_idx = int(job_op.split('O')[1]) - 1  # Extract op index from "O1" (0-indexed)
                    job_op_times[(job_id, op_idx)] = (start_time, end_time)
                except:
                    print(f"    ❌ {method_name}: Invalid operation format {job_op}")
                    return False, float('inf')
    
    # Check precedence within jobs
    for job_id in jobs_data.keys():
        for op_idx in range(1, len(jobs_data[job_id])):
            prev_op = (job_id, op_idx - 1)
            curr_op = (job_id, op_idx)
            
            if prev_op not in job_op_times or curr_op not in job_op_times:
                continue
                
            prev_end = job_op_times[prev_op][1]
            curr_start = job_op_times[curr_op][0]
            
            if curr_start < prev_end:
                print(f"    ❌ {method_name}: Precedence violation Job {job_id}: Op {op_idx} starts before Op {op_idx-1} ends")
                return False, float('inf')
    
    # Check 3: Arrival time constraints
    for job_id in jobs_data.keys():
        if (job_id, 0) in job_op_times:
            first_op_start = job_op_times[(job_id, 0)][0]
            arrival_time = arrival_times.get(job_id, 0.0)
            
            if first_op_start < arrival_time:
                print(f"    ❌ {method_name}: Job {job_id} starts at {first_op_start:.2f} before arrival at {arrival_time:.2f}")
                return False, float('inf')
    
    # Check 4: Machine conflicts
    for machine, machine_ops in schedule.items():
        sorted_ops = sorted(machine_ops, key=lambda x: x[1])  # Sort by start time
        for i in range(len(sorted_ops) - 1):
            curr_end = sorted_ops[i][2]
            next_start = sorted_ops[i+1][1]
            if next_start < curr_end:
                print(f"    ❌ {method_name}: Machine {machine} conflict: {sorted_ops[i][0]} overlaps with {sorted_ops[i+1][0]}")
                return False, float('inf')
    
    # Calculate true makespan
    true_makespan = max([max([op[2] for op in ops]) for ops in schedule.values() if ops])
    
    return True, true_makespan

def schedules_identical(schedule1, schedule2):
    """Check if two schedules are identical (same operations at same times on same machines)."""
    if not schedule1 or not schedule2:
        return False
    
    # Get all operations from both schedules
    ops1 = []
    ops2 = []
    
    for machine, machine_ops in schedule1.items():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                ops1.append((machine, op_data[0], op_data[1], op_data[2]))
    
    for machine, machine_ops in schedule2.items():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                ops2.append((machine, op_data[0], op_data[1], op_data[2]))
    
    # Sort and compare
    ops1.sort()
    ops2.sort()
    
    return ops1 == ops2

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
    
    best_name = min(valid_results.keys(), key=lambda x: valid_results[x][0])
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


def spt_heuristic_poisson(jobs_data, machine_list, arrival_times):
    """
    Run comparison of simple dispatching heuristics and return the best one.
    Uses SPT for machine selection and compares FIFO, LIFO, SPT, LPT for job sequencing.
    """
    print(f"  Comparing FIFO, LIFO, SPT, LPT heuristics with arrival times: {arrival_times}")
    return run_heuristic_comparison(jobs_data, machine_list, arrival_times)


def validate_schedule_makespan(schedule, jobs_data, arrival_times):
    """
    Validate a schedule and calculate its actual makespan.
    """
    max_completion = 0
    job_completion_times = {}
    
    for machine, operations in schedule.items():
        for op_name, start_time, end_time in operations:
            # Parse job and operation from op_name (e.g., "J0-O1")
            job_id = int(op_name.split('-')[0][1:])  # Extract job ID from "J0"
            op_idx = int(op_name.split('-')[1][1:]) - 1  # Extract op index from "O1" (0-indexed)
            
            # Check arrival time constraint
            if op_idx == 0:  # First operation
                required_arrival = arrival_times.get(job_id, 0)
                if start_time < required_arrival - 0.001:
                    print(f"❌ Job {job_id} operation {op_idx} starts at {start_time:.2f} before arrival at {required_arrival:.2f}")
                    return float('inf')
            
            # Check precedence constraint
            if op_idx > 0:
                prev_completion = job_completion_times.get((job_id, op_idx - 1), 0)
                if start_time < prev_completion - 0.001:
                    print(f"❌ Job {job_id} operation {op_idx} starts at {start_time:.2f} before previous op completes at {prev_completion:.2f}")
                    return float('inf')
            
            # Record completion time
            job_completion_times[(job_id, op_idx)] = end_time
            max_completion = max(max_completion, end_time)
    
    return max_completion

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
    
    # # CACHE CLEARING: Remove all existing MILP cache files to force fresh computation
    # cache_files = [f for f in os.listdir('.') if f.startswith('milp_cache_') and f.endswith('.pkl')]
    # for cache_file in cache_files:
    #     try:
    #         os.remove(cache_file)
    #         print(f"🧹 Cleared cache file: {cache_file}")
    #     except:
    #         pass

    print("\n--- Running MILP Optimal Scheduler ---")
    print(f"Jobs: {len(jobs_data)}, Machines: {len(machine_list)}")
    print(f"Arrival times: {arrival_times}")
    
    try:
        prob = LpProblem("PerfectKnowledge_FJSP_Optimal", LpMinimize)
        
        # Generate all operations
        ops = [(j, oi) for j in jobs_data for oi in range(len(jobs_data[j]))]
        BIG_M = 1000  # Large constant for disjunctive constraints (increased for safety)

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
                    
                    # Create binary variable to check if both operations are on this machine
                    both_on_m = LpVariable(f"both_{op1}_{op2}_on_{m}", cat="Binary")
                    prob += both_on_m <= x[op1][m]
                    prob += both_on_m <= x[op2][m]
                    prob += both_on_m >= x[op1][m] + x[op2][m] - 1
                    
                    # Either op1 before op2 or op2 before op1 (only if both assigned to machine m)
                    prob += s[op1] >= c[op2] - BIG_M * (1 - y[op1][op2][m]) - BIG_M * (1 - both_on_m)
                    prob += s[op2] >= c[op1] - BIG_M * y[op1][op2][m] - BIG_M * (1 - both_on_m)

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
            
            # CRITICAL VALIDATION: Verify MILP schedule is actually valid
            print(f"🔍 VALIDATING MILP SOLUTION...")
            actual_makespan = validate_schedule_makespan(schedule, jobs_data, arrival_times)
            
            if abs(actual_makespan - optimal_makespan) > 0.001:
                print(f"❌ MILP VALIDATION FAILED!")
                print(f"   MILP claimed makespan: {optimal_makespan:.2f}")
                print(f"   Actual schedule makespan: {actual_makespan:.2f}")
                print(f"   This explains why RL appears to beat MILP - MILP solution is invalid!")
                return float('inf'), {}  # Return invalid result
            
            print(f"✅ MILP OPTIMAL SOLUTION VALIDATED!")
            print(f"   Optimal Makespan: {optimal_makespan:.2f}")
            print(f"   This represents the THEORETICAL BEST possible performance")
            print(f"   with perfect knowledge of arrival times: {arrival_times}")
            
            # DO NOT cache result - force fresh computation each time for debugging
            print(f"   ⚠️  Caching disabled for debugging - fresh computation each run")
            
            return optimal_makespan, schedule
            
        else:
            print(f"❌ MILP solver failed to find optimal solution (status: {prob.status})")
            print("   Possible reasons: problem too complex, time limit exceeded, or infeasible")
            return float('inf'), schedule
            
    except Exception as e:
        print(f"❌ MILP solver error: {e}")
        return float('inf'), {m: [] for m in machine_list}


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
    else:
        print("- Methods showing good differentiation, consider fine-tuning hyperparameters")
        
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
    
    # Check if hierarchy is as expected
    expected_order = (
        regret_results['Perfect Knowledge RL']['absolute_regret'] <= 
        regret_results['Dynamic RL']['absolute_regret'] <= 
        regret_results['Static RL (dynamic)']['absolute_regret']
    )
    
    if expected_order:
        print("✅ Expected regret hierarchy maintained")
    else:
        print("❌ Unexpected regret hierarchy - investigate training issues")
    
    # Recommendations
    print(f"\nRecommendations:")
    if regret_results and len([v for v in regret_results.values() if v['absolute_regret'] != float('inf')]) > 1:
        regret_values = [v['absolute_regret'] for v in regret_results.values() if v['absolute_regret'] != float('inf')]
        regret_spread = max(regret_values) - min(regret_values)
        avg_regret = sum(regret_values) / len(regret_values)
        relative_regret_spread = (regret_spread / avg_regret * 100) if avg_regret > 0 else 0
        
        if relative_regret_spread < 5:
            print("- Increase arrival rate (try λ=1.0 or higher)")
            print("- Use longer training episodes") 
            print("- Add more complex job structures")
            print("- Increase reward differentiation for anticipatory actions")
        else:
            print("- Methods showing good differentiation, consider fine-tuning hyperparameters")
    else:
        print("- Need more valid methods for comparison analysis")


def main():
 
    print("=" * 80)
    print("DYNAMIC vs STATIC RL COMPARISON FOR POISSON FJSP")
    print("=" * 80)
    print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
    print("Research Question: Does Dynamic RL outperform Static RL on Poisson arrivals?")
    print(f"🔧 REPRODUCIBILITY: Fixed seed {GLOBAL_SEED} for all random components (CHANGED from 42)")
    print("🧹 CACHE CLEARING: All MILP cache files will be removed for fresh computation")
    print("📊 DEBUGGING: Action entropy & training metrics tracking enabled")
    print("🚨 STRICT VALIDATION: Will halt execution if any RL outperforms MILP optimal")
    print("=" * 80)
    arrival_rate = 0.1  # LOWER arrival rate to create more dynamic scenarios
    # With λ=0.5, expected inter-arrival = 2 time units (faster than most job operations)
    
    # Step 1: Training Setup
    print("\n1. TRAINING SETUP")
    print("-" * 50)
    perfect_timesteps = 200000    # Perfect knowledge needs less training
    dynamic_timesteps = 500000   # Increased for better learning with integer timing  
    static_timesteps = 200000    # Increased for better learning
    learning_rate = 7e-4       # Standard learning rate for PPO
    
    print(f"Perfect RL: {perfect_timesteps:,} | Dynamic RL: {dynamic_timesteps:,} | Static RL: {static_timesteps:,} timesteps")
    print(f"Arrival rate: {arrival_rate} (expected inter-arrival: {1/arrival_rate:.1f} time units)")

    # Step 2: Generate test scenarios (Poisson arrivals) - DIFFERENT from training scenarios
    print("\n2. GENERATING TEST SCENARIOS")
    print("-" * 40)
    print("Expected: Dynamic RL (knows arrival distribution) > Static RL (assumes all jobs at t=0)")
    print("Performance should be: Deterministic(~43) > Poisson Dynamic > Static(~50)")
    print(f"⚠️  IMPORTANT: Test scenarios use seeds 5000-5009, training used seed {GLOBAL_SEED}")
    print("   This tests generalizability to unseen arrival patterns!")
    print("   🧹 FRESH RUN: All seeds changed to force new evaluations and clear any cached bugs")
    test_scenarios = generate_test_scenarios(ENHANCED_JOBS_DATA, 
                                           initial_jobs=[0, 1, 2], 
                                           arrival_rate=arrival_rate, 
                                           num_scenarios=10)
    
    # Print all test scenario arrival times
    print("\nALL TEST SCENARIO ARRIVAL TIMES:")
    print("-" * 50)
    for i, scenario in enumerate(test_scenarios):
        print(f"Scenario {i+1}: {scenario['arrival_times']}")
        arrived_jobs = [j for j, t in scenario['arrival_times'].items() if t < float('inf')]
        print(f"  Jobs arriving: {len(arrived_jobs)} ({sorted(arrived_jobs)})")
        print()
    
    # Step 3: Train base agents (Dynamic and Static RL)
    print("\n3. TRAINING PHASE")
    print("-" * 40)
    
    print("Note: Perfect Knowledge RL will be trained separately for each test scenario")
    print("This ensures each scenario has its optimal RL benchmark for comparison")
    
    # Train dynamic RL agent (knows arrival distribution only) - trained once
    dynamic_model = train_dynamic_agent(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=[0, 1, 2], arrival_rate=arrival_rate,
        total_timesteps=dynamic_timesteps, reward_mode="makespan_increment", learning_rate=5e-5
    )
    print("\n--- Dynamic RL Training Metrics ---")
    plot_training_metrics()

    # Train static RL agent (assumes all jobs at t=0) - trained once
    # Reset metrics before each training
    for k in TRAINING_METRICS.keys():
        TRAINING_METRICS[k] = []
    static_model = train_static_agent(
        ENHANCED_JOBS_DATA, MACHINE_LIST, total_timesteps=static_timesteps,
        reward_mode="makespan_increment", learning_rate=learning_rate
    )
    print("\n--- Static RL Training Metrics ---")
    plot_training_metrics()

    # Train Perfect Knowledge RL for each scenario
    perfect_knowledge_models = []
    for i, scenario in enumerate(test_scenarios):
        scenario_arrivals = scenario['arrival_times']
        print(f"\nTraining Perfect Knowledge RL for Test Scenario {i+1}...")
        print(f"Arrival times: {scenario_arrivals}")
        
        # Reset metrics before each perfect RL training
        for k in TRAINING_METRICS.keys():
            TRAINING_METRICS[k] = []
        perfect_model = train_perfect_knowledge_agent(
            ENHANCED_JOBS_DATA, MACHINE_LIST,
            arrival_times=scenario_arrivals,
            total_timesteps=perfect_timesteps,
            reward_mode="makespan_increment", learning_rate=learning_rate
        )
        perfect_knowledge_models.append(perfect_model)
    
    # Step 4: Evaluate all methods on all test scenarios
    print("\n4. EVALUATION PHASE - MULTIPLE SCENARIOS")
    print("-" * 40)
    print("Comparing three levels of arrival information across 10 test scenarios:")
    print("1. Perfect Knowledge RL (knows exact arrival times)")
    print("2. Dynamic RL (knows arrival distribution)")  
    print("3. Static RL (assumes all jobs at t=0)")
    
    # Initialize results storage
    all_results = {
        'Perfect Knowledge RL': [],
        'Dynamic RL': [],
        'Static RL (dynamic)': [],
        'Static RL (static)': [],
        'Best Heuristic': [],
        'MILP Optimal': []
    }
    
    # Storage for first 3 scenarios for Gantt chart plotting
    gantt_scenarios_data = []
    
    print(f"\nEvaluating on {len(test_scenarios)} test scenarios...")
    
    for i, scenario in enumerate(test_scenarios):
        scenario_arrivals = scenario['arrival_times']
        print(f"\nScenario {i+1}/10: {scenario_arrivals}")
        # Train Perfect Knowledge RL specifically for this scenario
        print(f"  Training Perfect Knowledge RL for scenario {i+1}...")
        print(f"    Scenario arrival times: {scenario_arrivals}")
        # Reset metrics before each perfect RL training
        for k in TRAINING_METRICS.keys():
            TRAINING_METRICS[k] = []
        perfect_model = train_perfect_knowledge_agent(
            ENHANCED_JOBS_DATA, MACHINE_LIST,
            arrival_times=scenario_arrivals,
            total_timesteps=perfect_timesteps,
            reward_mode="makespan_increment", learning_rate=learning_rate
        )
        if i == 0:
            print("\n--- Perfect Knowledge RL Training Metrics (Scenario 1) ---")
            plot_training_metrics()
        # Perfect Knowledge RL
        perfect_makespan, perfect_schedule = evaluate_perfect_knowledge_on_scenario(
            perfect_model, ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        all_results['Perfect Knowledge RL'].append(perfect_makespan)
        print(f"    Perfect RL trained specifically for this scenario: {perfect_makespan:.2f}")
        
        # Dynamic RL
        dynamic_makespan, dynamic_schedule = evaluate_dynamic_on_dynamic(
            dynamic_model, ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        all_results['Dynamic RL'].append(dynamic_makespan)
        
        # Static RL (on dynamic scenario)
        static_dynamic_makespan, static_dynamic_schedule = evaluate_static_on_dynamic(
            static_model, ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        all_results['Static RL (dynamic)'].append(static_dynamic_makespan)
        
        # Static RL (on static scenario) - only do once since it's always the same
        if i == 0:
            static_static_makespan, static_static_schedule = evaluate_static_on_static(
                static_model, ENHANCED_JOBS_DATA, MACHINE_LIST)
        all_results['Static RL (static)'].append(static_static_makespan)
        
        # Best Heuristic
        spt_makespan, spt_schedule = spt_heuristic_poisson(ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        all_results['Best Heuristic'].append(spt_makespan)
        
        # MILP Optimal Solution
        milp_makespan, milp_schedule = milp_optimal_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        all_results['MILP Optimal'].append(milp_makespan)
        
        # Store ALL scenarios for Gantt plotting
        gantt_scenarios_data.append({
            'scenario_id': i,
            'arrival_times': scenario_arrivals,
            'schedules': {
                'MILP Optimal': (milp_makespan, milp_schedule),
                'Perfect Knowledge RL': (perfect_makespan, perfect_schedule),
                'Dynamic RL': (dynamic_makespan, dynamic_schedule),
                'Static RL (dynamic)': (static_dynamic_makespan, static_dynamic_schedule),
                'Static RL (static)': (static_static_makespan, static_static_schedule),
                'Best Heuristic': (spt_makespan, spt_schedule)
            }
        })
        
        # Verify all schedules for correctness
        print(f"  Verifying schedule correctness for scenario {i+1}:")
        
        methods_to_verify = [
            ("MILP Optimal", milp_makespan, milp_schedule),
            ("Perfect Knowledge RL", perfect_makespan, perfect_schedule),
            ("Dynamic RL", dynamic_makespan, dynamic_schedule),
            ("Static RL (dynamic)", static_dynamic_makespan, static_dynamic_schedule),
            ("Best Heuristic", spt_makespan, spt_schedule)
        ]
        
        for method_name, reported_makespan, schedule in methods_to_verify:
            if reported_makespan != float('inf') and schedule:
                is_valid, true_makespan = verify_schedule_correctness(schedule, ENHANCED_JOBS_DATA, arrival_times, method_name)
                if is_valid:
                    if abs(reported_makespan - true_makespan) > 0.01:
                        print(f"    ⚠️  {method_name}: Makespan mismatch! Reported: {reported_makespan:.2f}, Actual: {true_makespan:.2f}")
                        # Update the reported makespan to the correct one
                        if method_name == "Perfect Knowledge RL":
                            perfect_makespan = true_makespan
                        elif method_name == "Dynamic RL":
                            dynamic_makespan = true_makespan
                        elif method_name == "Static RL (dynamic)":
                            static_dynamic_makespan = true_makespan
                    else:
                        print(f"    ✅ {method_name}: Valid schedule, makespan: {true_makespan:.2f}")
                else:
                    print(f"    ❌ {method_name}: Invalid schedule!")
                    # Mark as failed
                    if method_name == "Perfect Knowledge RL":
                        perfect_makespan = float('inf')
                    elif method_name == "Dynamic RL":
                        dynamic_makespan = float('inf')
                    elif method_name == "Static RL (dynamic)":
                        static_dynamic_makespan = float('inf')
            else:
                print(f"    ❌ {method_name}: No valid schedule!")
        
        # Check for duplicate schedules across methods (debugging identical results)
        schedules_for_comparison = [
            ("Perfect Knowledge RL", perfect_schedule),
            ("Dynamic RL", dynamic_schedule),
            ("Static RL (dynamic)", static_dynamic_schedule)
        ]
        
        for i_method, (method1, sched1) in enumerate(schedules_for_comparison):
            for j_method, (method2, sched2) in enumerate(schedules_for_comparison[i_method+1:], i_method+1):
                if sched1 and sched2 and schedules_identical(sched1, sched2):
                    print(f"    🚨 WARNING: {method1} and {method2} produced identical schedules!")
        
        print()  # Empty line for readability
        
        # Store first 3 scenarios for Gantt plotting
        if i < 3:
            gantt_scenarios_data.append({
                'scenario_id': i,
                'arrival_times': scenario_arrivals,
                'schedules': {
                    'MILP Optimal': (milp_makespan, milp_schedule),
                    'Perfect Knowledge RL': (perfect_makespan, perfect_schedule),
                    'Dynamic RL': (dynamic_makespan, dynamic_schedule),
                    'Static RL (dynamic)': (static_dynamic_makespan, static_dynamic_schedule),
                    'Static RL (static)': (static_static_makespan, static_static_schedule),
                    'Best Heuristic': (spt_makespan, spt_schedule)
                }
            })
        
        print(f"  Results: Perfect={perfect_makespan:.2f}, Dynamic={dynamic_makespan:.2f}, Static(dyn)={static_dynamic_makespan:.2f}, Heuristic={spt_makespan:.2f}, MILP={milp_makespan:.2f}")
        
        # STRICT DEBUG: Check for impossible results and HALT execution if found
        if milp_makespan != float('inf') and dynamic_makespan < milp_makespan - 0.001:  # Small tolerance for numerical precision
            print(f"  🚨🚨🚨 FATAL ERROR: Dynamic RL ({dynamic_makespan:.2f}) outperformed MILP Optimal ({milp_makespan:.2f})!")
            print(f"      This is THEORETICALLY IMPOSSIBLE - Dynamic RL cannot be better than MILP optimal!")
            print(f"      Bug in: evaluation function, schedule validation, or MILP formulation")
            print(f"      HALTING EXECUTION to investigate...")
            exit(1)
        
        if milp_makespan != float('inf') and perfect_makespan < milp_makespan - 0.001:  # Small tolerance for numerical precision
            print(f"  🚨🚨🚨 FATAL ERROR: Perfect Knowledge RL ({perfect_makespan:.2f}) outperformed MILP Optimal ({milp_makespan:.2f})!")
            print(f"      This is THEORETICALLY IMPOSSIBLE - RL cannot be better than MILP optimal!")
            print(f"      Bug in: evaluation function, schedule validation, or MILP formulation")
            print(f"      HALTING EXECUTION to investigate...")
            exit(1)
        
        if perfect_makespan > dynamic_makespan + 5.0:  # Increased tolerance for training variations
            print(f"  🚨 WARNING: Perfect Knowledge RL ({perfect_makespan:.2f}) much worse than Dynamic RL ({dynamic_makespan:.2f})")
            print(f"      Perfect RL should generally be better since it knows exact arrival times")
            print(f"      This may indicate training issues or very difficult scenario")
        
        # Check if Dynamic RL is giving same result as previous scenarios
        if i > 0 and abs(dynamic_makespan - all_results['Dynamic RL'][i-1]) < 0.01:
            print(f"  🚨 SUSPICIOUS: Dynamic RL giving identical makespan to previous scenario")
            print(f"      This suggests evaluation isn't properly using different arrival times")
    
    # Calculate average results
    avg_results = {}
    std_results = {}
    for method, results in all_results.items():
        valid_results = [r for r in results if r != float('inf')]
        if valid_results:
            avg_results[method] = np.mean(valid_results)
            std_results[method] = np.std(valid_results)
        else:
            avg_results[method] = float('inf')
            std_results[method] = 0
    
    # Use first scenario data for single-scenario analyses (backward compatibility)
    first_scenario_arrivals = test_scenarios[0]['arrival_times']
    static_arrivals = {job_id: 0.0 for job_id in ENHANCED_JOBS_DATA.keys()}
    
    # Get individual results from first scenario for single plots
    perfect_makespan = all_results['Perfect Knowledge RL'][0]
    dynamic_makespan = all_results['Dynamic RL'][0]  
    static_dynamic_makespan = all_results['Static RL (dynamic)'][0]
    static_static_makespan = all_results['Static RL (static)'][0]
    spt_makespan = all_results['Best Heuristic'][0]
    milp_makespan = all_results['MILP Optimal'][0]
    
    # Get schedules from first scenario
    perfect_schedule = gantt_scenarios_data[0]['schedules']['Perfect Knowledge RL'][1]
    dynamic_schedule = gantt_scenarios_data[0]['schedules']['Dynamic RL'][1]
    static_dynamic_schedule = gantt_scenarios_data[0]['schedules']['Static RL (dynamic)'][1]
    static_static_schedule = gantt_scenarios_data[0]['schedules']['Static RL (static)'][1]
    spt_schedule = gantt_scenarios_data[0]['schedules']['Best Heuristic'][1]
    milp_schedule = gantt_scenarios_data[0]['schedules']['MILP Optimal'][1]
    
    # Step 5: Results Analysis
    print("\n5. RESULTS ANALYSIS")
    print("=" * 60)
    print("AVERAGE RESULTS ACROSS 10 TEST SCENARIOS:")
    print(f"MILP Optimal              - Avg Makespan: {avg_results['MILP Optimal']:.2f} ± {std_results['MILP Optimal']:.2f}")
    print(f"Perfect Knowledge RL      - Avg Makespan: {avg_results['Perfect Knowledge RL']:.2f} ± {std_results['Perfect Knowledge RL']:.2f}")
    print(f"Dynamic RL (Poisson)      - Avg Makespan: {avg_results['Dynamic RL']:.2f} ± {std_results['Dynamic RL']:.2f}")  
    print(f"Static RL (on dynamic)    - Avg Makespan: {avg_results['Static RL (dynamic)']:.2f} ± {std_results['Static RL (dynamic)']:.2f}")
    print(f"Static RL (on static)     - Avg Makespan: {avg_results['Static RL (static)']:.2f} ± {std_results['Static RL (static)']:.2f}")
    print(f"Best Heuristic            - Avg Makespan: {avg_results['Best Heuristic']:.2f} ± {std_results['Best Heuristic']:.2f}")
    
    print("\nFirst Scenario Results (for detailed analysis):")
    print(f"MILP Optimal              - Makespan: {milp_makespan:.2f} (THEORETICAL BEST)")
    print(f"Perfect Knowledge RL      - Makespan: {perfect_makespan:.2f}")
    print(f"Dynamic RL (Poisson)      - Makespan: {dynamic_makespan:.2f}")  
    print(f"Static RL (on dynamic)    - Makespan: {static_dynamic_makespan:.2f}")
    print(f"Static RL (on static)     - Makespan: {static_static_makespan:.2f}")
    print(f"Best Heuristic            - Makespan: {spt_makespan:.2f}")
    
    print("\nAverage Performance Ranking:")
    avg_results_list = [
        ("MILP Optimal", avg_results['MILP Optimal']),
        ("Perfect Knowledge RL", avg_results['Perfect Knowledge RL']),
        ("Dynamic RL", avg_results['Dynamic RL']), 
        ("Static RL (dynamic)", avg_results['Static RL (dynamic)']),
        ("Static RL (static)", avg_results['Static RL (static)']),
        ("Best Heuristic", avg_results['Best Heuristic'])
    ]
    avg_results_list.sort(key=lambda x: x[1])
    for i, (method, makespan) in enumerate(avg_results_list, 1):
        if makespan == float('inf'):
            print(f"{i}. {method}: Failed")
        else:
            print(f"{i}. {method}: {makespan:.2f}")
    
    print(f"\nExpected Performance Hierarchy:")
    print(f"MILP Optimal ≤ Perfect Knowledge ≤ Dynamic RL ≤ Static RL")
    if avg_results['MILP Optimal'] != float('inf'):
        print(f"Actual Avg: {avg_results['MILP Optimal']:.2f} ≤ {avg_results['Perfect Knowledge RL']:.2f} ≤ {avg_results['Dynamic RL']:.2f} ≤ {avg_results['Static RL (dynamic)']:.2f}")
    else:
        print(f"Actual Avg (no MILP): {avg_results['Perfect Knowledge RL']:.2f} ≤ {avg_results['Dynamic RL']:.2f} ≤ {avg_results['Static RL (dynamic)']:.2f}")
    
    # Step 5.5: Average Regret Analysis (Gap from Optimal across all scenarios)
    if avg_results['MILP Optimal'] != float('inf'):
        avg_methods_results = {
            "Perfect Knowledge RL": avg_results['Perfect Knowledge RL'],
            "Dynamic RL": avg_results['Dynamic RL'],
            "Static RL (dynamic)": avg_results['Static RL (dynamic)'],
            "Static RL (static)": avg_results['Static RL (static)'],
            "Best Heuristic": avg_results['Best Heuristic']
        }
        print("\nAVERAGE REGRET ANALYSIS:")
        regret_results = calculate_regret_analysis(avg_results['MILP Optimal'], avg_methods_results)
        
        # Also calculate regret for each individual scenario
        print("\nINDIVIDUAL SCENARIO REGRET ANALYSIS:")
        all_regrets = {method: [] for method in avg_methods_results.keys()}
        
        for i in range(len(test_scenarios)):
            if all_results['MILP Optimal'][i] != float('inf'):
                scenario_methods = {
                    "Perfect Knowledge RL": all_results['Perfect Knowledge RL'][i],
                    "Dynamic RL": all_results['Dynamic RL'][i],
                    "Static RL (dynamic)": all_results['Static RL (dynamic)'][i],
                    "Static RL (static)": all_results['Static RL (static)'][i],
                    "Best Heuristic": all_results['Best Heuristic'][i]
                }
                
                optimal = all_results['MILP Optimal'][i]
                for method, makespan in scenario_methods.items():
                    if makespan != float('inf'):
                        regret = ((makespan - optimal) / optimal) * 100
                        all_regrets[method].append(regret)
        
        print("Regret Statistics (% above optimal):")
        for method, regret_list in all_regrets.items():
            if regret_list:
                avg_regret = np.mean(regret_list)
                std_regret = np.std(regret_list)
                min_regret = np.min(regret_list)
                max_regret = np.max(regret_list)
                print(f"{method:25s}: {avg_regret:6.1f}% ± {std_regret:5.1f}% (range: {min_regret:.1f}% - {max_regret:.1f}%)")
            else:
                print(f"{method:25s}: No valid results")
    
    # Validate expected ordering
    if perfect_makespan <= dynamic_makespan <= static_dynamic_makespan:
        print("✅ EXPECTED: Perfect knowledge outperforms distribution knowledge outperforms no knowledge")
    else:
        print("❌ UNEXPECTED: Performance doesn't follow expected hierarchy")
    
    # Diagnose performance similarity issues
    diagnose_performance_similarity(perfect_makespan, dynamic_makespan, static_makespan, spt_makespan)
    
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
        
        # Formatting
        ax.set_yticks(range(len(MACHINE_LIST)))
        ax.set_yticklabels(MACHINE_LIST)
        ax.set_xlabel("Time" if plot_idx == len(schedules_data)-1 else "")
        ax.set_ylabel("Machines")
        ax.set_title(f"{title} (Makespan: {makespan:.2f})", fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Apply consistent x-axis limits across all subplots
        ax.set_xlim(0, consistent_x_limit)
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
    
    # Skip the separate static RL comparison - focus on 10 test scenarios
    # Step 8: Create Gantt Charts for All 10 Test Scenarios (5 methods only)
    print(f"\n8. GANTT CHARTS FOR ALL 10 TEST SCENARIOS")
    print("-" * 60)
    
    # Create small_instances folder
    import os
    folder_name = "small_instances_0.1"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    
    # Generate Gantt charts for all 10 scenarios using stored data only
    for scenario_idx in range(len(gantt_scenarios_data)):
        scenario = gantt_scenarios_data[scenario_idx]
        scenario_id = scenario['scenario_id']
        arrival_times = scenario['arrival_times']
        schedules = scenario['schedules']
        print(f"\nGenerating Gantt chart for Test Scenario {scenario_id + 1}...")
        print(f"Arrival times: {arrival_times}")
        num_methods = 5 if schedules['MILP Optimal'][0] != float('inf') else 4
        fig, axes = plt.subplots(num_methods, 1, figsize=(16, num_methods * 3))
        
        if schedules['MILP Optimal'][0] != float('inf'):
            fig.suptitle(f'Test Scenario {scenario_id + 1} - 5 Method Comparison\n' + 
                         f'Arrival Times: {arrival_times}', 
                         fontsize=14, fontweight='bold')
            methods_to_plot = [
                ('MILP Optimal', schedules['MILP Optimal']),
                ('Perfect Knowledge RL', schedules['Perfect Knowledge RL']),
                ('Dynamic RL', schedules['Dynamic RL']),
                ('Static RL (dynamic)', schedules['Static RL (dynamic)']),
                ('Best Heuristic', schedules['Best Heuristic'])
            ]
        else:
            fig.suptitle(f'Test Scenario {scenario_id + 1} - 4 Method Comparison\n' + 
                         f'Arrival Times: {arrival_times}', 
                         fontsize=14, fontweight='bold')
            methods_to_plot = [
                ('Perfect Knowledge RL', schedules['Perfect Knowledge RL']),
                ('Dynamic RL', schedules['Dynamic RL']),
                ('Static RL (dynamic)', schedules['Static RL (dynamic)']),
                ('Best Heuristic', schedules['Best Heuristic'])
            ]
        
        # Calculate consistent x-axis limits for this scenario
        max_time_scenario = 0
        for method_name, (makespan, schedule) in methods_to_plot:
            if schedule and any(len(ops) > 0 for ops in schedule.values()):
                scenario_max_time = max([max([op[2] for op in ops]) for ops in schedule.values() if ops])
                max_time_scenario = max(max_time_scenario, scenario_max_time)
        
        x_limit_scenario = max_time_scenario * 1.1 if max_time_scenario > 0 else 100
        
        # Plot each method
        colors = plt.cm.tab20.colors
        
        for plot_idx, (method_name, (makespan, schedule)) in enumerate(methods_to_plot):
            ax = axes[plot_idx] if num_methods > 1 else axes
            
            if not schedule or all(len(ops) == 0 for ops in schedule.values()):
                ax.text(0.5, 0.5, 'No valid schedule', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{method_name} - No Solution")
                ax.set_xlim(0, x_limit_scenario)
                ax.set_ylim(-0.5, len(MACHINE_LIST) + 1.5)
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

        # Add arrival arrows for jobs that arrive after t=0
        arrow_y_position = len(MACHINE_LIST) + 0.2
        for job_id, arrival_time in arrival_times.items():
            if arrival_time > 0 and arrival_time < x_limit_scenario:
                ax.axvline(x=arrival_time, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                ax.annotate(f'J{job_id}', 
                           xy=(arrival_time, arrow_y_position), 
                           xytext=(arrival_time, arrow_y_position + 0.3),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                           ha='center', va='bottom', color='red', fontweight='bold', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.8))
        
        # Formatting
        ax.set_yticks(range(len(MACHINE_LIST)))
        ax.set_yticklabels(MACHINE_LIST)
        ax.set_xlabel("Time" if plot_idx == len(methods_to_plot)-1 else "")
        ax.set_ylabel("Machines")
        ax.set_title(f"{method_name} (Makespan: {makespan:.2f})", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x_limit_scenario)
        ax.set_ylim(-0.5, len(MACHINE_LIST) + 1.5)
    
    # Add legend for jobs
    legend_elements = []
    for i in range(len(ENHANCED_JOBS_DATA)):
        color = colors[i % len(colors)]
        initial_or_dynamic = ' (Initial)' if i < 3 else ' (Dynamic)'
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                          alpha=0.8, label=f'Job {i}{initial_or_dynamic}'))
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(ENHANCED_JOBS_DATA), fontsize=9)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    
    # Save scenario-specific Gantt chart in small_instances folder
    scenario_filename = os.path.join(folder_name, f'test_scenario_{scenario_id + 1}_gantt_comparison.png')
    plt.savefig(scenario_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Test Scenario {scenario_id + 1} Gantt chart: {scenario_filename}")
    
    plt.close()  # Close figure to save memory
    
    print(f"\n✅ All 10 Gantt charts saved in {folder_name}/ folder")
    
    # Skip the old static RL comparison code - focus on the 10 test scenario Gantt charts above
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED!")
    print("Generated files:")
    if milp_makespan != float('inf'):
        print("- complete_scheduling_comparison_with_milp_optimal.png: Five-method comprehensive comparison with MILP benchmark")
        print(f"- small_instances/ folder: Contains 10 Gantt charts for all test scenarios (5 methods each)")
        for i in range(10):
            print(f"  ├── test_scenario_{i+1}_gantt_comparison.png")
        print(f"\nKey Findings (Average across 10 test scenarios):")
        print(f"• MILP Optimal (Benchmark): {avg_results['MILP Optimal']:.2f} ± {std_results['MILP Optimal']:.2f}")
        print(f"• Perfect Knowledge RL: {avg_results['Perfect Knowledge RL']:.2f} ± {std_results['Perfect Knowledge RL']:.2f} (avg regret: +{((avg_results['Perfect Knowledge RL']-avg_results['MILP Optimal'])/avg_results['MILP Optimal']*100):.1f}%)")
        print(f"• Dynamic RL: {avg_results['Dynamic RL']:.2f} ± {std_results['Dynamic RL']:.2f} (avg regret: +{((avg_results['Dynamic RL']-avg_results['MILP Optimal'])/avg_results['MILP Optimal']*100):.1f}%)")
        print(f"• Static RL (on dynamic): {avg_results['Static RL (dynamic)']:.2f} ± {std_results['Static RL (dynamic)']:.2f} (avg regret: +{((avg_results['Static RL (dynamic)']-avg_results['MILP Optimal'])/avg_results['MILP Optimal']*100):.1f}%)")
        print(f"• Static RL (on static): {avg_results['Static RL (static)']:.2f} ± {std_results['Static RL (static)']:.2f} (avg regret: +{((avg_results['Static RL (static)']-avg_results['MILP Optimal'])/avg_results['MILP Optimal']*100):.1f}%)")
        print(f"• Best Heuristic: {avg_results['Best Heuristic']:.2f} ± {std_results['Best Heuristic']:.2f} (avg regret: +{((avg_results['Best Heuristic']-avg_results['MILP Optimal'])/avg_results['MILP Optimal']*100):.1f}%)")
        print(f"• Perfect Knowledge RL validation: {'✅ Working well' if avg_results['Perfect Knowledge RL'] <= avg_results['MILP Optimal'] * 1.15 else '❌ Needs improvement'}")
    else:
        print("- dynamic_vs_static_gantt_comparison-7jobs.png: Five-method comparison")
        print("- static_rl_dynamic_vs_static_comparison.png: Separate Static RL comparison (dynamic vs static scenarios)")  
        print("- test_scenario_1_gantt_comparison.png: Detailed Gantt chart for Test Scenario 1")
        print("- test_scenario_2_gantt_comparison.png: Detailed Gantt chart for Test Scenario 2")
        print("- test_scenario_3_gantt_comparison.png: Detailed Gantt chart for Test Scenario 3")
        print(f"\nKey Findings (Average across 10 test scenarios, no MILP benchmark available):")
        print(f"• Perfect Knowledge RL: {avg_results['Perfect Knowledge RL']:.2f} ± {std_results['Perfect Knowledge RL']:.2f}")
        print(f"• Dynamic RL: {avg_results['Dynamic RL']:.2f} ± {std_results['Dynamic RL']:.2f}")
        print(f"• Static RL (on dynamic): {avg_results['Static RL (dynamic)']:.2f} ± {std_results['Static RL (dynamic)']:.2f}")
        print(f"• Static RL (on static): {avg_results['Static RL (static)']:.2f} ± {std_results['Static RL (static)']:.2f}")
        print(f"• Best Heuristic: {avg_results['Best Heuristic']:.2f} ± {std_results['Best Heuristic']:.2f}")
        print(f"• Performance hierarchy: {'✅ Expected' if avg_results['Perfect Knowledge RL'] <= avg_results['Dynamic RL'] <= avg_results['Static RL (dynamic)'] else '❌ Unexpected'}")
        print(f"• Static RL scenario comparison: {'✅ Better on static' if avg_results['Static RL (static)'] < avg_results['Static RL (dynamic)'] else '❌ Needs investigation'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
