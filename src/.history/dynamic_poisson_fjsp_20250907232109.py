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
                 max_time_horizon=200, reward_mode="dynamic_adaptation", seed=None):
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
        """Generate action masks for valid actions."""
        mask = np.full(self.action_space.n, False, dtype=bool)
        
        # Early termination if all operations scheduled
        if self.operations_scheduled >= self.total_operations:
            return mask

        valid_action_count = 0
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
                            mask[action] = True
                            valid_action_count += 1
        
        # Fallback if no valid actions
        if valid_action_count == 0:
            for job_idx, job_id in enumerate(self.job_ids):
                if job_id in self.arrived_jobs:
                    op_idx = self.next_operation[job_id]
                    if op_idx < len(self.jobs[job_id]):
                        for machine_idx, machine_name in enumerate(self.machines):
                            if machine_name in self.jobs[job_id][op_idx]['proc_times']:
                                action = job_idx * (self.max_ops_per_job * len(self.machines)) + \
                                        op_idx * len(self.machines) + machine_idx
                                if action < self.action_space.n:
                                    mask[action] = True
                                    break
                        break
        
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
        """Calculate reward based on multiple factors for dynamic environment."""
        
        if self.reward_mode == "makespan_increment":
            # Reward based on makespan increment (like in test3_backup.py)
            makespan_increment = current_time - previous_time
            reward = -makespan_increment  # Negative increment encourages shorter makespan
            
            # Small penalty for idle time to encourage efficiency
            reward -= idle_time * 0.5
            
            # Small step reward to encourage progress
            reward += 1.0
            
            # Completion bonus - important for learning!
            if done:
                reward += 100.0
                # Additional bonus for good final makespan (encourage shorter schedules)
                if current_time > 0:
                    reward += max(0, 200.0 / current_time)
            
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
        """Generate observation vector for current state."""
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
            
        # Job arrival status
        for job_id in self.job_ids:
            arrived = 1.0 if job_id in self.arrived_jobs else 0.0
            obs.append(float(arrived))
        
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


def train_poisson_agent(jobs_data, machine_list, initial_jobs=5, arrival_rate=0.1, 
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
                 'Jobs 0-2: Available at t=0, Jobs 3-6: Poisson Arrivals', 
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
        
        # Add red arrows for job arrivals (dynamic jobs only)
        if arrival_times:
            for job_id, arrival_time in arrival_times.items():
                if job_id >= 3 and arrival_time > 0:  # Dynamic jobs only
                    # Draw red arrow at arrival time
                    arrow_y = len(MACHINE_LIST) * 10 + 5  # Above all machines
                    ax.annotate(f'J{job_id}', xy=(arrival_time, arrow_y), xytext=(arrival_time, arrow_y + 15),
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


def compare_poisson_methods():
    """Compare RL vs SPT heuristic on Poisson dynamic FJSP."""
    print("\n" + "="*80)
    print("POISSON DYNAMIC FJSP COMPARISON")
    print("="*80)
    
    # Environment parameters
    initial_jobs = 5
    arrival_rate = 0.1  # jobs per time unit
    
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
    print("Starting Poisson Dynamic FJSP Analysis with 3-Method Comparison...")
    print("="*80)
    print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
    print("Initial jobs (t=0): Jobs 0-2")
    print("Dynamic jobs (Poisson): Jobs 3-6")
    print("="*80)
    
    # Configuration
    initial_jobs = [0, 1, 2]  # Jobs available at start
    arrival_rate = 0.05        # Poisson arrival rate for jobs 3-6
    
    # Results storage
    schedules_data = {}
    
    print("\n1. RUNNING SPT HEURISTIC...")
    print("-" * 50)
    spt_makespan, spt_schedule, spt_arrival_times = heuristic_spt_poisson(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=initial_jobs,
        arrival_rate=arrival_rate,
        seed=42
    )
    schedules_data['spt'] = {
        'schedule': spt_schedule,
        'makespan': spt_makespan,
        'title': 'SPT Heuristic (Poisson Dynamic)'
    }
    print(f"SPT Makespan: {spt_makespan:.2f}")
    
    print("\n2. TRAINING DYNAMIC RL AGENT...")
    print("-" * 50)
    print("Training on Poisson job arrivals...")
    dynamic_rl_model = train_poisson_agent(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=initial_jobs,
        arrival_rate=arrival_rate,
        total_timesteps=200000  # Increased for better learning
    )
    
    print("\n3. EVALUATING DYNAMIC RL AGENT...")
    print("-" * 50)
    dynamic_rl_makespan, dynamic_rl_schedule = evaluate_poisson_agent(
        dynamic_rl_model, ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=initial_jobs,
        arrival_rate=arrival_rate,
        num_episodes=10,
        debug=True  # Enable debug for first evaluation
    )
    schedules_data['dynamic_rl'] = {
        'schedule': dynamic_rl_schedule,
        'makespan': dynamic_rl_makespan,
        'title': 'Dynamic RL (Trained on Poisson)'
    }
    print(f"Dynamic RL Makespan: {dynamic_rl_makespan:.2f}")
    
    print("\n4. TRAINING STATIC RL AGENT...")
    print("-" * 50)
    print("Training on static arrivals (all jobs at t=0)...")
    
    # Create static arrival times (all jobs available at t=0)
    static_arrivals = {job_id: 0 for job_id in ENHANCED_JOBS_DATA.keys()}
    
    # Train static RL agent
    def make_static_env():
        env = PoissonDynamicFJSPEnv(
            ENHANCED_JOBS_DATA, MACHINE_LIST,
            initial_jobs=list(ENHANCED_JOBS_DATA.keys()),  # All jobs initial
            arrival_rate=0.0,  # No Poisson arrivals
            reward_mode="makespan_increment"
        )
        return ActionMasker(env, mask_fn)
    
    static_vec_env = DummyVecEnv([make_static_env])
    static_rl_model = MaskablePPO(
        "MlpPolicy", static_vec_env, verbose=1,
        learning_rate=5e-4, n_steps=4096, batch_size=256,
        gamma=0.995, policy_kwargs=dict(net_arch=[512, 512, 256])
    )
    static_rl_model.learn(total_timesteps=200000)
    
    print("\n5. EVALUATING STATIC RL ON POISSON ARRIVALS...")
    print("-" * 50)
    print("Testing static RL agent on Poisson job arrivals...")
    static_rl_makespan, static_rl_schedule = evaluate_poisson_agent(
        static_rl_model, ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=initial_jobs,
        arrival_rate=arrival_rate,
        num_episodes=10
    )
    schedules_data['static_rl'] = {
        'schedule': static_rl_schedule,
        'makespan': static_rl_makespan,
        'title': 'Static RL (Trained on Static, Tested on Poisson)'
    }
    print(f"Static RL Makespan: {static_rl_makespan:.2f}")
    
    print("\n6. RESULTS SUMMARY...")
    print("="*80)
    results = [
        ("SPT Heuristic", spt_makespan),
        ("Dynamic RL", dynamic_rl_makespan),
        ("Static RL", static_rl_makespan)
    ]
    
    # Sort by performance
    results.sort(key=lambda x: x[1] if x[1] != float('inf') else 999999)
    
    print("Performance Ranking:")
    for i, (method, makespan) in enumerate(results):
        if makespan == float('inf'):
            print(f"{i+1}. {method}: FAILED")
        else:
            gap = ((makespan - results[0][1]) / results[0][1] * 100) if i > 0 else 0
            print(f"{i+1}. {method}: {makespan:.2f} (+{gap:.1f}%)")
    
    print("\n7. CREATING 3-METHOD COMPARISON PLOT...")
    print("-" * 50)
    plot_three_method_comparison(schedules_data, arrival_times=spt_arrival_times, 
                               save_path="three_method_comparison.png")
    
    print("\n" + "="*80)
    print("POISSON DYNAMIC FJSP 3-METHOD COMPARISON COMPLETED")
    print("="*80)
