
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

# Deterministic arrival times from successful test3_backup.py scenario
DETERMINISTIC_ARRIVAL_TIMES = {0: 0, 1: 0, 2: 0, 3: 10, 4: 15, 5: 20, 6: 25}

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
        
        # Dynamic action space: action = index into valid_actions list
        # Maximum possible actions = num_jobs * num_machines (each job can have at most one ready operation)
        max_possible_actions = self.num_jobs * len(self.machines)
        self.action_space = spaces.Discrete(max_possible_actions)
        
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
        
        # Dynamic action space: list of valid (job_id, machine_name) pairs
        self.valid_actions = []
        self._update_valid_actions()

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        if seed is not None:
            super().reset(seed=seed, options=options)
            random.seed(seed)
            np.random.seed(seed)
        
        self._reset_state()
        return self._get_observation(), {}

    def _update_valid_actions(self):
        """
        Update the list of valid actions based on current state.
        Each action is a (job_id, machine_name) pair for the next available operation.
        """
        self.valid_actions = []
        
        for job_id in self.job_ids:
            # Get the next operation index for this job
            next_op_idx = self.next_operation[job_id]
            
            # Skip if job is complete
            if next_op_idx >= len(self.jobs[job_id]):
                continue
                
            # Get the operation data
            operation = self.jobs[job_id][next_op_idx]
            
            # Add all valid (job, machine) pairs for this operation
            for machine_name in operation['proc_times'].keys():
                if machine_name in self.machines:
                    self.valid_actions.append((job_id, machine_name))

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
        """Generate action masks for valid actions using dynamic action space."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # Mark valid actions based on current valid_actions list
        for i in range(len(self.valid_actions)):
            if i < self.action_space.n:
                mask[i] = True
        
        # If no valid actions, enable the first action to prevent crashes
        if not np.any(mask) and self.action_space.n > 0:
            mask[0] = True
            
        return mask

    def step(self, action):
        """Execute one step in the environment."""
        self.episode_step += 1
        
        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        # Dynamic action: get job_id and machine directly from valid_actions list
        if action >= len(self.valid_actions):
            return self._get_observation(), -100.0, False, False, {"error": "Invalid action index"}
        
        job_id, machine = self.valid_actions[action]
        
        # Get the next operation index for this job
        op_idx = self.next_operation[job_id]
        
        # Validate that this operation exists and can be processed on this machine
        if op_idx >= len(self.jobs[job_id]):
            return self._get_observation(), -100.0, False, False, {"error": "Job already complete"}
        
        operation = self.jobs[job_id][op_idx]
        if machine not in operation['proc_times']:
            return self._get_observation(), -100.0, False, False, {"error": "Machine cannot process operation"}
        
        # Calculate timing
        machine_available_time = self.machine_next_free.get(machine, 0.0)
        job_ready_time = self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 else 0.0
        
        start_time = max(machine_available_time, job_ready_time, self.current_time)
        proc_time = operation['proc_times'][machine]
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
        
        # Update valid actions for next step
        self._update_valid_actions()
        
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
        if self.reward_mode == "combined_makespan_increment":
            makespan_increment = current_time - previous_time
            reward = -makespan_increment - idle_time * 0.5 + 1.0
            
            if done:
                reward += 100.0
                if current_time > 0:
                    reward += max(0, 200.0 / current_time)
            
            return reward
        
        elif self.reward_mode == "makespan_increment":
            makespan_increment = current_time - previous_time
            reward = -makespan_increment
            
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
        
        # Dynamic action space: action = index into valid_actions list
        # Maximum possible actions = num_jobs * num_machines (each job can have at most one ready operation)
        max_possible_actions = self.num_jobs * len(self.machines)
        self.action_space = spaces.Discrete(max_possible_actions)
        
        # Enhanced observation space with arrival distribution awareness
        obs_size = (
            len(self.machines) +                    # Machine relative availability
            self.num_jobs * self.max_ops_per_job +  # Operation completion status
            self.num_jobs +                         # Job progress ratios  
            self.num_jobs * 3 +                     # Job arrival urgency (3 categories per job)
            len(self.machines) +                    # Resource demand indicators
            1 +                                     # Overall completion progress
            1 +                                     # Future work indicator (binary)
            3                                       # ARRIVAL DISTRIBUTION INFO: rate, expected_next_arrival, arrival_intensity
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
        
        # Dynamic action space: list of valid (job_id, machine_name) pairs
        self.valid_actions = []
        
        # Generate Poisson arrival times for dynamic jobs
        self._generate_poisson_arrivals()
        
        # Update valid actions after initialization
        self._update_valid_actions()
        
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
        global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
        
        if seed is not None:
            super().reset(seed=seed, options=options)
            random.seed(seed)
            np.random.seed(seed)
        
        self._reset_state()
        
        # Track arrival times for analysis
        TRAINING_EPISODE_COUNT += 1
        episode_arrivals = []
        for job_id, arr_time in self.arrival_times.items():
            if arr_time != float('inf') and arr_time > 0:  # Only dynamic arrivals
                episode_arrivals.append(arr_time)
        
        if episode_arrivals:
            TRAINING_ARRIVAL_TIMES.extend(episode_arrivals)
        
        return self._get_observation(), {}

    def _update_valid_actions(self):
        """
        Update the list of valid actions based on current state.
        Each action is a (job_id, machine_name) pair for the next available operation.
        This handles both initial jobs and dynamically arriving jobs.
        """
        self.valid_actions = []
        
        for job_id in self.job_ids:
            # Skip if job hasn't arrived yet
            if job_id not in self.arrived_jobs:
                continue
                
            # Get the next operation index for this job
            next_op_idx = self.next_operation[job_id]
            
            # Skip if job is complete
            if next_op_idx >= len(self.jobs[job_id]):
                continue
                
            # Get the operation data
            operation = self.jobs[job_id][next_op_idx]
            
            # Add all valid (job, machine) pairs for this operation
            for machine_name in operation['proc_times'].keys():
                if machine_name in self.machines:
                    self.valid_actions.append((job_id, machine_name))

        # If no valid actions, we need to advance time to next job arrival
        if not self.valid_actions:
            self._advance_time_to_next_arrival()

    def _advance_time_to_next_arrival(self):
        """Advance time to the next job arrival if no operations are ready"""
        if self.next_arrival_events:
            next_arrival_time = self.next_arrival_events[0][0]
            if next_arrival_time <= self.max_time_horizon:
                self.current_time = next_arrival_time
                self._update_arrivals(self.current_time)
                # Update valid actions after new arrivals
                self._update_valid_actions()

    def action_masks(self):
        """Return action mask for MaskablePPO using dynamic action space."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # Mark valid actions based on current valid_actions list
        for i in range(len(self.valid_actions)):
            if i < self.action_space.n:
                mask[i] = True
        
        # If no valid actions, enable the first action to prevent crashes
        if not np.any(mask) and self.action_space.n > 0:
            mask[0] = True
            
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
        """Execute one step in the environment using dynamic actions."""
        self.episode_step += 1
        
        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        # Validate action (simple bounds check, no complex decoding!)
        if action >= len(self.valid_actions):
            # Invalid action - give negative reward but continue
            return self._get_observation(), -100.0, False, False, {"error": "Invalid action index"}
        
        # Get the job and machine from the action (NO DECODING NEEDED!)
        job_id, machine = self.valid_actions[action]
        next_op_idx = self.next_operation[job_id]
        
        # Double-check validity (should always be true)
        if next_op_idx >= len(self.jobs[job_id]):
            return self._get_observation(), -100.0, False, False, {"error": "Job already complete"}
        
        operation = self.jobs[job_id][next_op_idx]
        if machine not in operation['proc_times']:
            return self._get_observation(), -100.0, False, False, {"error": "Machine cannot process operation"}
        
        # CRITICAL BUG CHECK: Ensure job has actually arrived
        job_arrival_time = self.arrival_times.get(job_id, 0.0)
        if self.current_time < job_arrival_time:
            print(f"ERROR: Trying to schedule Job {job_id} at time {self.current_time:.2f} but it arrives at {job_arrival_time:.2f}")
            return self._get_observation(), -1000.0, False, False, {"error": "Scheduling before arrival"}
        
        # Calculate timing
        machine_available_time = self.machine_next_free.get(machine, 0.0)
        job_ready_time = (self.operation_end_times[job_id][next_op_idx - 1] if next_op_idx > 0 
                         else self.arrival_times.get(job_id, 0.0))
        
        start_time = max(machine_available_time, job_ready_time, self.current_time)
        proc_time = operation['proc_times'][machine]
        end_time = start_time + proc_time
        
        # Calculate idle time for this machine
        idle_time = max(0, start_time - machine_available_time)
        self.total_idle_time += idle_time

        # Update state
        previous_time = self.current_time
        self.machine_next_free[machine] = end_time
        self.machine_workload[machine] += proc_time
        self.operation_end_times[job_id][next_op_idx] = end_time
        self.completed_ops[job_id][next_op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        self.current_time = max(self.current_time, end_time)
        
        # Check for newly arrived jobs
        newly_arrived = self._update_arrivals(self.current_time)
        
        # Check if job is completed
        if self.next_operation[job_id] >= len(self.jobs[job_id]):
            self.num_completed_jobs += 1

        # Record in schedule
        self.schedule[machine].append((f"J{job_id}-O{next_op_idx+1}", start_time, end_time))

        # Update valid actions for next step (DYNAMIC ACTION SPACE UPDATE)
        self._update_valid_actions()

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
            len(newly_arrived), previous_time, self.current_time, job_id
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
                         previous_time, current_time, job_id):
        """Enhanced reward function for dynamic scheduling that encourages adaptation."""
        
        if self.reward_mode == "makespan_increment":
            # Simple, focused reward: minimize makespan increment and idle time
            makespan_increment = current_time - previous_time
            reward = -makespan_increment - idle_time * 0.5 + 1.0
            
            # Bonus for utilizing newly arrived jobs quickly
            if num_new_arrivals > 0:
                reward += 5.0 * num_new_arrivals
            
            # Completion bonus
            if done:
                reward += 100.0
                if current_time > 0:
                    reward += max(0, 200.0 / current_time)
            
            return reward
            
        elif self.reward_mode == "makespan_minimization":
            # Simple makespan-focused reward
            reward = -proc_time  # Encourage shorter operations
            
            if done:
                reward += max(0, 500.0 / current_time) if current_time > 0 else 100.0
            
            return reward
            
        elif self.reward_mode == "combined_makespan_utilization":
            """
            ENHANCED Combined reward mode that balances makespan minimization with machine utilization.
            This is the recommended mode for dynamic scheduling problems.
            
            Key improvements for dynamic scheduling:
            1. Stronger machine utilization penalties
            2. Better future-aware scheduling incentives  
            3. Dynamic adaptation bonuses
            4. Load balancing rewards
            """
            reward = 0.0
            
            # 1. Makespan increment penalty (primary objective)
            makespan_increment = current_time - previous_time
            reward -= makespan_increment * 3.0  # INCREASED penalty for time progression
            
            # 2. ENHANCED Machine utilization rewards (critical for reducing empty spaces)
            reward -= idle_time * 5.0  # STRONGER penalty for idle time
            reward += 3.0  # Base reward for making progress
            
            # 3. Processing efficiency incentives
            reward -= proc_time * 0.2  # Slight penalty for longer operations
            
            # 4. ENHANCED Dynamic adaptation bonuses
            if num_new_arrivals > 0:
                # Strong bonus for immediately utilizing newly arrived jobs
                reward += num_new_arrivals * 12.0  # INCREASED bonus
                
                # Extra bonus if we schedule new jobs on currently idle machines
                current_machine_loads = [max(0, self.machine_next_free[m] - current_time) for m in self.machines]
                idle_machines = sum(1 for load in current_machine_loads if load <= 0.1)
                if idle_machines > 0:
                    reward += idle_machines * 5.0  # INCREASED bonus for using idle machines
            
            # 5. ENHANCED Load balancing incentive (reduce empty spaces)
            machine_loads = [self.machine_next_free[m] for m in self.machines]
            if len(machine_loads) > 1 and max(machine_loads) > 0:
                load_std = np.std(machine_loads)
                load_mean = np.mean(machine_loads)
                if load_mean > 0:
                    load_balance_ratio = 1.0 - (load_std / load_mean)
                    reward += load_balance_ratio * 4.0  # INCREASED load balancing reward
            
            # 6. ENHANCED Future-aware scheduling
            if self.next_arrival_events:
                next_arrival_time = self.next_arrival_events[0][0]
                time_to_next_arrival = next_arrival_time - current_time
                
                # Bonus for keeping machines available for upcoming arrivals
                if 0 < time_to_next_arrival <= 10.0:
                    available_machines = sum(1 for m in self.machines 
                                           if self.machine_next_free[m] <= current_time + 2.0)
                    reward += available_machines * 3.0  # INCREASED future-awareness bonus
                
                # STRONGER penalty for creating long idle periods when more jobs are coming
                if idle_time > 3.0 and time_to_next_arrival > idle_time:
                    reward -= 5.0  # INCREASED penalty for poor timing decisions
                    
                # NEW: Bonus for scheduling operations that finish close to next arrival
                if 0 < time_to_next_arrival <= proc_time + 2.0:
                    reward += 2.0  # Reward for good timing coordination
            
            # 7. Job completion efficiency
            if self.next_operation[job_id] >= len(self.jobs[job_id]):
                # Job completed - bonus for quick completion
                job_arrival_time = self.arrival_times.get(job_id, 0.0)
                job_flow_time = current_time - job_arrival_time
                if job_flow_time > 0:
                    flow_time_bonus = min(20.0 / job_flow_time, 10.0)  # INCREASED flow time bonus
                    reward += flow_time_bonus
            
            # 8. ENHANCED Final completion rewards
            if done:
                # Base completion bonus
                reward += 200.0  # INCREASED completion bonus
                
                # Makespan efficiency bonus
                if current_time > 0:
                    makespan_bonus = min(400.0 / current_time, 100.0)  # INCREASED makespan bonus
                    reward += makespan_bonus
                
                # ENHANCED Machine utilization bonus
                total_workload = sum(self.machine_workload.values())
                total_capacity = current_time * len(self.machines)
                if total_capacity > 0:
                    utilization = total_workload / total_capacity
                    reward += utilization * 100.0  # INCREASED utilization bonus
                
                # ENHANCED Load balancing bonus
                final_loads = [self.machine_next_free[m] for m in self.machines]
                if len(final_loads) > 1 and max(final_loads) > 0:
                    final_balance = 1.0 - (np.std(final_loads) / max(final_loads))
                    reward += final_balance * 60.0  # INCREASED load balancing bonus
                    
                # NEW: Bonus for minimizing total machine idle time
                total_machine_time = current_time * len(self.machines)
                total_idle = total_machine_time - total_workload
                if total_machine_time > 0:
                    idle_ratio = total_idle / total_machine_time
                    idle_efficiency_bonus = (1.0 - idle_ratio) * 80.0
                    reward += idle_efficiency_bonus
            
            return reward
            
        else:  # Default basic reward
            reward = 10.0 - proc_time * 0.1 - idle_time
            if done:
                reward += 100.0
            return reward

    def _get_observation(self):
        """
        ENHANCED RELATIVE TIMING STATE REPRESENTATION
        
        Key insight: Include timing information but normalize it to be relative rather than absolute.
        This preserves scheduling urgency while enabling generalization across different scenarios.
        
        Core principle: Agent learns scheduling policies based on:
        1. Relative machine availability 
        2. Job arrival urgency categories
        3. Structural job/machine relationships
        4. Relative workload distribution
        """
        obs = []
        
        # 1. MACHINE RELATIVE AVAILABILITY (normalized busy times)
        machine_busy_times = [max(0.0, self.machine_next_free[m] - self.current_time) for m in self.machines]
        max_busy = max(machine_busy_times) if machine_busy_times else 1.0
        for busy_time in machine_busy_times:
            # Normalize: 1.0 = completely free, 0.0 = busiest machine
            relative_availability = 1.0 - (busy_time / max(max_busy, 1.0))
            obs.append(max(0.0, min(1.0, relative_availability)))
        
        # 2. OPERATION COMPLETION STATUS (structural state)
        for job_id in self.job_ids:
            for op_idx in range(self.max_ops_per_job):
                if op_idx < len(self.jobs[job_id]):
                    completed = 1.0 if self.completed_ops[job_id][op_idx] else 0.0
                else:
                    completed = 1.0  # Non-existent operations considered completed
                obs.append(float(completed))
        
        # 3. JOB PROGRESS (structural progress ratio)
        for job_id in self.job_ids:
            total_ops = len(self.jobs[job_id])
            if total_ops > 0:
                progress = float(self.next_operation[job_id]) / float(total_ops)
            else:
                progress = 1.0
            obs.append(max(0.0, min(1.0, progress)))
        
        # 4. JOB ARRIVAL URGENCY (relative timing categories)
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                # Job is available now - highest priority
                obs.append(1.0)  # Available now
                obs.append(0.0)  # Not arriving soon  
                obs.append(0.0)  # Not arriving later
            else:
                # Job will arrive later - categorize urgency
                arrival_time = self.arrival_times.get(job_id, float('inf'))
                if arrival_time == float('inf'):
                    # Won't arrive this episode
                    obs.append(0.0)  # Not available
                    obs.append(0.0)  # Not arriving soon
                    obs.append(0.0)  # Not arriving later
                else:
                    time_until_arrival = arrival_time - self.current_time
                    
                    # Normalize by typical processing times (10-15 time units)
                    typical_proc_time = 10.0
                    
                    if time_until_arrival <= typical_proc_time:
                        # Arriving very soon - high urgency  
                        obs.append(0.0)  # Not available now
                        obs.append(1.0)  # Arriving soon - consider in decisions
                        obs.append(0.0)  # Not arriving later
                    elif time_until_arrival <= 3 * typical_proc_time:
                        # Arriving later - medium urgency
                        obs.append(0.0)  # Not available now
                        obs.append(0.0)  # Not arriving soon
                        obs.append(1.0)  # Arriving later - plan for it
                    else:
                        # Arriving much later - low urgency
                        obs.append(0.0)  # Not available now
                        obs.append(0.0)  # Not arriving soon
                        obs.append(0.0)  # Too far to consider
        
        # 5. RESOURCE DEMAND (current bottlenecks)
        # Which machines are needed by available jobs right now
        machine_demand = {m: 0 for m in self.machines}
        for job_id in self.arrived_jobs:
            next_op_idx = self.next_operation[job_id]
            if next_op_idx < len(self.jobs[job_id]):
                operation = self.jobs[job_id][next_op_idx]
                for machine_name in operation['proc_times'].keys():
                    if machine_name in machine_demand:
                        machine_demand[machine_name] += 1
        
        # Normalize demand indicators
        max_demand = max(machine_demand.values()) if machine_demand.values() else 1
        for m in self.machines:
            demand_ratio = machine_demand[m] / max(max_demand, 1)
            obs.append(max(0.0, min(1.0, demand_ratio)))
        
        # 6. OVERALL PROGRESS (how much of available work is done)
        total_ops_completed = sum(
            sum(1 for completed in job_ops if completed)
            for job_ops in self.completed_ops.values()
        )
        total_ops_available = sum(
            len(self.jobs[job_id]) for job_id in self.arrived_jobs
        )
        
        if total_ops_available > 0:
            completion_ratio = total_ops_completed / total_ops_available
            obs.append(max(0.0, min(1.0, completion_ratio)))
        else:
            obs.append(0.0)
        
        # 7. FUTURE WORK INDICATOR (binary: more jobs coming?)
        future_jobs_count = len([j for j in self.job_ids if j not in self.arrived_jobs])
        has_future_work = 1.0 if future_jobs_count > 0 else 0.0
        obs.append(has_future_work)
        
        # 8. ARRIVAL DISTRIBUTION AWARENESS (KEY ADVANTAGE over static RL)
        # This is the crucial information that dynamic RL has but static RL doesn't!
        
        # 8a. Normalized arrival rate (helps agent understand arrival intensity)
        normalized_arrival_rate = min(1.0, self.arrival_rate * 10.0)  # Scale for [0,1] range
        obs.append(normalized_arrival_rate)
        
        # 8b. Expected time until next arrival (Poisson distribution knowledge)
        if future_jobs_count > 0:
            # Expected inter-arrival time = 1/λ for Poisson process
            expected_next_arrival = min(1.0, (1.0 / max(self.arrival_rate, 0.01)) / 20.0)  # Normalize
        else:
            expected_next_arrival = 0.0  # No more jobs coming
        obs.append(expected_next_arrival)
        
        # 8c. Current arrival intensity (how "busy" the arrival process is)
        # Higher values when we expect many arrivals soon
        if future_jobs_count > 0:
            arrival_intensity = min(1.0, self.arrival_rate * future_jobs_count / 10.0)
        else:
            arrival_intensity = 0.0
        obs.append(arrival_intensity)
        
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

def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, reward_mode="makespan_increment"):
    """
    Train a dynamic RL agent with PERFECT knowledge of specific arrival times.
    This represents the upper bound of what's possible with arrival time information.
    """
    print(f"\n--- Training Perfect Knowledge RL Agent ---")
    print(f"Arrival times: {arrival_times}")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_perfect_env():
        env = PerfectKnowledgeFJSPEnv(
            jobs_data, machine_list, 
            arrival_times=arrival_times,
            reward_mode=reward_mode
        )
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_perfect_env])
    
    # Use similar hyperparameters as dynamic agent
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=1,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[512, 512, 256],
            activation_fn=torch.nn.ReLU
        )
    )
    
    # Training with progress bar
    print("Training perfect knowledge agent...")
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

class PerfectKnowledgeFJSPEnv(PoissonDynamicFJSPEnv):
    """
    Dynamic FJSP Environment with PERFECT knowledge of specific arrival times.
    
    This environment represents the optimal case where the RL agent knows
    exactly when each job will arrive, rather than just the Poisson rate.
    It demonstrates the maximum advantage that arrival time knowledge can provide.
    
    Key differences from PoissonDynamicFJSPEnv:
    - Uses predetermined arrival times instead of stochastic Poisson process
    - Can include future arrival information in state representation
    - Serves as upper bound for dynamic RL performance
    """
    
    def __init__(self, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment", seed=None):
        """
        Initialize with perfect knowledge of arrival times.
        
        Args:
            jobs_data: Dictionary of all jobs
            machine_list: List of available machines  
            arrival_times: Dict {job_id: arrival_time} - EXACT arrival times
            reward_mode: Reward function type
            seed: Random seed
        """
        # Extract initial jobs from arrival times
        initial_jobs = [job_id for job_id, arr_time in arrival_times.items() if arr_time == 0]
        
        # Initialize parent class with dummy arrival rate (not used)
        super().__init__(jobs_data, machine_list, initial_jobs=initial_jobs, 
                        arrival_rate=0.1, reward_mode=reward_mode, seed=seed)
        
        # Override with exact arrival times
        self.exact_arrival_times = arrival_times.copy()
        self.is_perfect_knowledge = True
        
    def reset(self, seed=None, options=None):
        """Reset with predetermined arrival schedule."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Override the arrival schedule with exact times
        self.arrival_times = self.exact_arrival_times.copy()
        
        # Set up deterministic arrival events (no randomness)
        self.next_arrival_events = []
        for job_id, arr_time in self.arrival_times.items():
            if arr_time > 0 and arr_time != float('inf'):
                self.next_arrival_events.append((arr_time, job_id))
        
        # Sort by arrival time
        self.next_arrival_events.sort(key=lambda x: x[0])
        
        return obs, info
    
    def _get_observation(self):
        """Enhanced observation with perfect arrival knowledge."""
        # Get base observation from parent
        obs = super()._get_observation()
        
        # Convert to list for modification
        obs_list = obs.tolist()
        
        # Remove the last 3 elements (arrival distribution info from parent)
        obs_list = obs_list[:-3]
        
        # Add PERFECT KNOWLEDGE FEATURES instead of distribution estimates
        
        # 1. Time until next specific arrival (exact knowledge)
        if self.next_arrival_events:
            next_arrival_time = self.next_arrival_events[0][0]
            time_until_next = max(0, next_arrival_time - self.current_time)
            normalized_time_until_next = min(1.0, time_until_next / 30.0)  # Normalize
            obs_list.append(normalized_time_until_next)
        else:
            obs_list.append(0.0)  # No more arrivals
            
        # 2. Number of jobs arriving in near future (next 10 time units)
        near_future_arrivals = len([
            (t, j) for t, j in self.next_arrival_events 
            if 0 < t - self.current_time <= 10.0
        ])
        normalized_near_arrivals = min(1.0, near_future_arrivals / 5.0)
        obs_list.append(normalized_near_arrivals)
        
        # 3. Arrival density indicator (jobs per time unit in next 20 time units)
        future_horizon = 20.0
        future_arrivals = len([
            (t, j) for t, j in self.next_arrival_events 
            if 0 < t - self.current_time <= future_horizon
        ])
        arrival_density = min(1.0, future_arrivals / future_horizon)
        obs_list.append(arrival_density)
        
        # Ensure proper array format
        target_size = self.observation_space.shape[0]
        if len(obs_list) < target_size:
            obs_list.extend([0.0] * (target_size - len(obs_list)))
        elif len(obs_list) > target_size:
            obs_list = obs_list[:target_size]
            
        obs_array = np.array(obs_list, dtype=np.float32)
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
        available_jobs = list(jobs_data.keys())
        
        if isinstance(initial_jobs, int):
            # Vary the number of initial jobs (2-4) for more diversity
            num_initial = random.randint(2, min(4, len(available_jobs)))
            random_initial = random.sample(available_jobs, num_initial)
        else:
            # Still randomize even with fixed initial jobs list
            random_initial = random.sample(initial_jobs, len(initial_jobs))
            
        # Vary arrival rate slightly for robustness (±20%)
        varied_arrival_rate = arrival_rate * random.uniform(0.8, 1.2)
            
        env = PoissonDynamicFJSPEnv(
            jobs_data, machine_list, 
            initial_jobs=random_initial,
            arrival_rate=varied_arrival_rate,
            reward_mode=reward_mode
        )
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_dynamic_env])
    
    # ENHANCED hyperparameters for larger, more complex job set
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,  # Minimal output - just progress bar
        learning_rate=1e-4,            # Balanced learning rate for complex patterns
        n_steps=4096,                  # LARGE steps for diverse experience collection
        batch_size=512,                # LARGE batch for stable learning
        n_epochs=10,                   # MORE epochs to learn from complex scenarios
        gamma=1,                    # Standard discount for this problem type
        gae_lambda=0.95,               # Standard GAE
        clip_range=0.2,                # Standard clipping
        ent_coef=0.01,                 # Encourage exploration of diverse scenarios
        vf_coef=0.5,                   # Standard value function weight
        max_grad_norm=0.5,             # Prevent gradient explosion with complex patterns
        policy_kwargs=dict(
            net_arch=[256, 256, 128, 64],  # Moderately deeper for dynamic complexity
            activation_fn=torch.nn.ReLU,
            ortho_init=True  # Better initialization for deeper networks
        )
    )
    
    # Calculate expected training diversity
    avg_episode_length = 25  # Estimated for 7-job problem
    total_episodes = total_timesteps // avg_episode_length
    batches = total_timesteps // model.n_steps
    episodes_per_batch = model.n_steps // avg_episode_length
    
    print(f"Training Dynamic RL for {total_timesteps:,} timesteps...")
    print(f"Expected training episodes: ~{total_episodes}")
    print(f"Training batches: ~{batches}")
    print(f"Episodes per batch: ~{episodes_per_batch}")
    print(f"Arrival pattern diversity: Each episode generates new Poisson arrivals")
    print(f"Initial job combinations: Randomized from {len(jobs_data)} jobs")
    
    # Train with tqdm progress bar  
    start_time = time.time()
    
    with tqdm(total=total_timesteps, desc="Dynamic RL", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} timesteps [{elapsed}<{remaining}]') as pbar:
        
        # Break training into chunks for progress updates
        chunk_size = total_timesteps // 50  # 50 chunks = 2% updates
        remaining_timesteps = total_timesteps
        
        while remaining_timesteps > 0:
            current_chunk = min(chunk_size, remaining_timesteps)
            model.learn(total_timesteps=current_chunk)
            pbar.update(current_chunk)
            remaining_timesteps -= current_chunk
    
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
            if current_time <= 300:  # Extended time horizon for larger job set
                arrival_times[job_id] = current_time
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
        reward_mode=reward_mode
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
        """Map dynamic observation to static observation space with bounds checking."""
        static_obs = []
        
        try:
            # For simplified time-invariant observation, just take what we can and pad
            # Dynamic obs structure: machines + operations + job_progress + job_availability + demand + progress + future
            
            num_jobs = len(jobs_data)
            max_ops = max(len(ops) for ops in jobs_data.values())
            
            # Take available elements safely
            obs_idx = 0
            
            # 1. Machine availability (first 3 elements)
            if obs_idx + len(machine_list) <= len(dynamic_obs):
                static_obs.extend(dynamic_obs[obs_idx:obs_idx + len(machine_list)])
                obs_idx += len(machine_list)
            else:
                static_obs.extend([0.0] * len(machine_list))
            
            # 2. Operation completion status
            ops_needed = num_jobs * max_ops
            if obs_idx + ops_needed <= len(dynamic_obs):
                static_obs.extend(dynamic_obs[obs_idx:obs_idx + ops_needed])
                obs_idx += ops_needed
            else:
                static_obs.extend([0.0] * ops_needed)
            
            # 3. Job progress ratios
            if obs_idx + num_jobs <= len(dynamic_obs):
                static_obs.extend(dynamic_obs[obs_idx:obs_idx + num_jobs])
                obs_idx += num_jobs
            else:
                static_obs.extend([0.0] * num_jobs)
            
            # For remaining elements, just fill with zeros or use available data
            remaining_needed = static_obs_size - len(static_obs)
            remaining_available = len(dynamic_obs) - obs_idx
            
            if remaining_available > 0 and remaining_needed > 0:
                take_count = min(remaining_needed, remaining_available)
                static_obs.extend(dynamic_obs[obs_idx:obs_idx + take_count])
                remaining_needed -= take_count
            
            # Pad with zeros if needed
            if remaining_needed > 0:
                static_obs.extend([0.0] * remaining_needed)
            
            # Truncate if too long
            if len(static_obs) > static_obs_size:
                static_obs = static_obs[:static_obs_size]
                
        except Exception as e:
            print(f"Error in observation mapping: {e}")
            print(f"Dynamic obs size: {len(dynamic_obs)}, Static obs size needed: {static_obs_size}")
            # Return safe default observation
            static_obs = [0.0] * static_obs_size
        
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

def evaluate_dynamic_on_dynamic(dynamic_model, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment" ):
    """Evaluate dynamic model on dynamic scenario."""
    print(f"  Dynamic RL using arrival times: {arrival_times}")
    
    test_env = PoissonDynamicFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=[k for k, v in arrival_times.items() if v == 0],
        arrival_rate=0.1,
        reward_mode=reward_mode
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


def evaluate_perfect_knowledge_on_scenario(perfect_model, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment"):
    """Evaluate perfect knowledge agent on a specific scenario."""
    print(f"  Perfect Knowledge RL evaluation...")
    
    # Create environment with perfect knowledge of arrivals
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
            print(f"    Step {step_count}: current_time = {test_env.env.current_time:.2f}")
    
    makespan = test_env.env.current_time
    
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
        
        # Apply improved SPT with machine utilization priority
        # Sort by multiple criteria: 1) processing time, 2) machine availability, 3) load balancing
        def spt_priority(op_data):
            job_id, op_idx, machine, proc_time = op_data
            machine_available_time = machine_next_free[machine]
            job_ready_time = operation_end_times[job_id][op_idx - 1] if op_idx > 0 else arrival_times[job_id]
            
            start_time = max(machine_available_time, job_ready_time, sim_time)
            idle_time = max(0, start_time - machine_available_time)
            
            # Multi-criteria priority: prefer short operations on immediately available machines
            priority = (
                proc_time * 1.0,          # Primary: shortest processing time
                idle_time * 0.5,          # Secondary: minimize idle time
                machine_available_time * 0.1  # Tertiary: prefer available machines
            )
            return sum(priority)
        
        available_ops.sort(key=spt_priority)
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
        ("SPT Heuristic", spt_makespan)
    ]
    
    makespans = [makespan for _, makespan in results]
    
    # Calculate performance spread
    max_makespan = max(makespans)
    min_makespan = min(makespans)
    spread = max_makespan - min_makespan
    relative_spread = spread / min_makespan * 100
    
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
    dynamic_timesteps = 150000   # Dynamic RL needs more time to learn patterns
    static_timesteps = 100000    # Static RL baseline
    
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
    perfect_timesteps = 300000  # Less needed since it has perfect info
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
    
    # Static RL (no arrival info)
    print("Evaluating Static RL...")
    static_makespan, static_schedule = evaluate_static_on_dynamic(
        static_model, ENHANCED_JOBS_DATA, MACHINE_LIST, first_scenario_arrivals)
    
    # SPT Heuristic for comparison
    print("Evaluating SPT Heuristic...")
    spt_makespan, spt_schedule = spt_heuristic_poisson(ENHANCED_JOBS_DATA, MACHINE_LIST, first_scenario_arrivals)
    
    # MILP Optimal Solution (Benchmark)
    print("Computing MILP Optimal Solution...")
    milp_makespan, milp_schedule = milp_optimal_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, first_scenario_arrivals)
    
    # Step 5: Results Analysis
    print("\n5. RESULTS ANALYSIS")
    print("=" * 60)
    print(f"Perfect Knowledge RL - Best Makespan: {perfect_makespan:.2f}")
    print(f"Dynamic RL (Poisson) - Best Makespan: {dynamic_makespan:.2f}")  
    print(f"Static RL            - Best Makespan: {static_makespan:.2f}")
    print(f"SPT Heuristic        - Best Makespan: {spt_makespan:.2f}")
    
    print("\nPerformance Ranking:")
    results = [
        ("Perfect Knowledge RL", perfect_makespan),
        ("Dynamic RL", dynamic_makespan), 
        ("Static RL", static_makespan),
        ("SPT Heuristic", spt_makespan)
    ]
    results.sort(key=lambda x: x[1])
    for i, (method, makespan) in enumerate(results, 1):
        print(f"{i}. {method}: {makespan:.2f}")
    
    print(f"\nExpected Performance Hierarchy:")
    print(f"Perfect Knowledge ≤ Dynamic RL ≤ Static RL")
    print(f"Actual: {perfect_makespan:.2f} ≤ {dynamic_makespan:.2f} ≤ {static_makespan:.2f}")
    
    # Validate expected ordering
    if perfect_makespan <= dynamic_makespan <= static_makespan:
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
    
    # Step 7: Generate Gantt Charts for Comparison
    print(f"\n7. GANTT CHART COMPARISON")
    print("-" * 60)
    
    # Create a four-subplot comparison
    fig, axes = plt.subplots(4, 1, figsize=(16, 16))
    fig.suptitle('Scheduling Comparison: Perfect Knowledge vs Dynamic RL vs Static RL vs SPT\n' + 
                 f'Test Scenario: Jobs 0-2 at t=0, Jobs 3-6 via Poisson arrivals', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab20.colors
    schedules_data = [
        {'schedule': perfect_schedule, 'makespan': perfect_makespan, 'title': 'Perfect Knowledge RL', 'arrival_times': first_scenario_arrivals},
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
    plt.savefig('dynamic_vs_static_gantt_comparison-7jobs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED!")
    print("Generated files:")
    print("- dynamic_vs_static_gantt_comparison.png: Three-method Gantt chart comparison")
    print("=" * 80)

if __name__ == "__main__":
    main()
