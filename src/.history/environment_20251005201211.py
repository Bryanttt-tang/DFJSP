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
        
        # BUILDER MODE: Use makespan as the "builder clock" for final schedule width
        self.current_makespan = 0.0
        
        # EVENT-DRIVEN: Separate event time that controls arrival visibility
        self.event_time = 0.0  # Event frontier for revealing arrivals and WAIT processing
        
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
        """WAIT ACTION: Advance event_time to next job arrival or machine completion."""
        # Find next arrival time after current event_time
        next_arrival_time = float('inf')
        next_arriving_jobs = []
        
        for job_id, arrival_time in self.job_arrival_times.items():
            if (job_id not in self.arrived_jobs and 
                arrival_time > self.event_time and 
                arrival_time != float('inf')):
                if arrival_time < next_arrival_time:
                    next_arrival_time = arrival_time
                    next_arriving_jobs = [job_id]
                elif arrival_time == next_arrival_time:
                    next_arriving_jobs.append(job_id)
        
        # Find next machine completion time after current event_time
        next_machine_completion = float('inf')
        for machine, free_time in self.machine_next_free.items():
            if free_time > self.event_time:
                next_machine_completion = min(next_machine_completion, free_time)
        # Choose the earliest next event
        next_event_time = min(next_arrival_time, next_machine_completion)
        
        if next_event_time != float('inf'):
            # Advance event_time to next event
            self.event_time = next_event_time
            
            # Add newly arrived jobs if this was an arrival event
            if next_event_time == next_arrival_time:
                for job_id in next_arriving_jobs:
                    self.arrived_jobs.add(job_id)
                return len(next_arriving_jobs), next_event_time
            else:
                # This was a machine completion event
                return 0, next_event_time
        else:
            # No more events - advance event_time minimally
            self.event_time += 1.0
            return 0, self.event_time

    def step(self, action):
        """BUILDER MODE: Step function with proper event-driven WAIT semantics."""
        self.episode_step += 1
        
        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        job_idx, op_idx, machine_idx = self._decode_action(action)

        # Handle WAIT action
        if job_idx is None:  # WAIT action
            if self.operations_scheduled >= self.total_operations:
                return self._get_observation(), -10.0, True, False, {"error": "WAIT at terminal"}
            
            num_new_arrivals, new_event_time = self._advance_to_next_arrival()
            
            # WAIT reward: small penalty for waiting
            wait_reward = -1.0 
            info = {
                "makespan": self.current_makespan,
                "event_time": self.event_time,
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
        
        # Earliest feasible start = max of all constraints (including event_time for proper sequencing)
        start_time = max(machine_available_time, job_ready_time, self.event_time)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time

        # Update state
        previous_makespan = self.current_makespan
        self.machine_next_free[machine] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        
        # Update makespan (builder clock for final schedule width)
        self.current_makespan = max(self.current_makespan, end_time)
        
        # Advance event_time to at least the start of this operation
        self.event_time = max(self.event_time, start_time)

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
            "event_time": self.event_time,
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
        """BUILDER MODE: Event-driven observation using event_time for arrival visibility."""
        obs = []
        
        # 1. Ready job indicators (arrived based on event_time + has next operation)
        for job_id in self.job_ids:
            if (job_id in self.arrived_jobs and 
                self.next_operation[job_id] < len(self.jobs[job_id])):
                obs.append(1.0)
            else:
                obs.append(0.0)
        
        # 2. Machine availability (normalized next_free times relative to event_time)
        max_time_horizon = 100.0  # For normalization
        for machine in self.machines:
            machine_free_time = self.machine_next_free[machine]
            # How far ahead is this machine busy (relative to event_time)
            relative_busy_time = max(0, machine_free_time - self.event_time)
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
        # Time since last arrival (based on event_time)
        if len(self.arrived_jobs) > 0:
            last_arrival_time = max(self.job_arrival_times.get(job_id, 0.0) for job_id in self.arrived_jobs)
            time_since_last_arrival = max(0, self.event_time - last_arrival_time)
        else:
            time_since_last_arrival = self.event_time
        obs.append(min(1.0, time_since_last_arrival / 30.0))
        
        # Arrival progress
        arrival_progress = len(self.arrived_jobs) / len(self.job_ids)
        obs.append(arrival_progress)
        
        # Current event time (time context for event-driven decisions)
        normalized_event_time = min(1.0, self.event_time / 100.0)
        obs.append(normalized_event_time)
        
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
