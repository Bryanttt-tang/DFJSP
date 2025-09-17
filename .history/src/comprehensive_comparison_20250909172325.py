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
import torch.nn.functional as F
import os
import collections
# Skip PULP import if it causes issues
try:
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
    PULP_AVAILABLE = True
except ImportError:
    print("Warning: PULP not available, MILP solver will not work")
    PULP_AVAILABLE = False
import sys
import math
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Test instance data
TEST_JOBS_DATA = collections.OrderedDict({
    0: [
        {'proc_times': {'M0': 7.85, 'M2': 8.07, 'M1': 2.15}},
        {'proc_times': {'M2': 6.79, 'M0': 8.4, 'M1': 4.99}},
    ],
    1: [
        {'proc_times': {'M2': 4.19, 'M0': 9.74, 'M1': 9.04}},
        {'proc_times': {'M1': 5.2, 'M0': 1.39, 'M2': 2.39}},
        {'proc_times': {'M1': 3.93, 'M0': 4.33, 'M2': 5.23}},
        {'proc_times': {'M2': 5.28, 'M0': 3.04, 'M1': 7.03}},
    ],
    2: [
        {'proc_times': {'M1': 7.14, 'M0': 2.26, 'M2': 2.8}},
        {'proc_times': {'M0': 7.35, 'M1': 8.03, 'M2': 5.13}},
        {'proc_times': {'M2': 2.03, 'M1': 7.02}},
    ],
    3: [
        {'proc_times': {'M1': 6.03, 'M2': 3.74}},
        {'proc_times': {'M1': 2.93, 'M0': 4.68}},
    ],
})
TEST_MACHINE_LIST = ['M0', 'M1', 'M2']
TEST_JOB_ARRIVAL_TIMES = {0: 0.0, 1: 0.0, 2: 13.34, 3: 27.25}

def print_instance_data():
    """Print the test instance data"""
    print("=" * 60)
    print("TEST INSTANCE DATA")
    print("=" * 60)
    print(f"Number of jobs: {len(TEST_JOBS_DATA)}")
    print(f"Machines: {TEST_MACHINE_LIST}")
    print(f"Job arrival times: {TEST_JOB_ARRIVAL_TIMES}")
    print("\nJob Details:")
    
    for job_id, operations in TEST_JOBS_DATA.items():
        print(f"  Job {job_id} (arrives at {TEST_JOB_ARRIVAL_TIMES[job_id]}):")
        for op_idx, operation in enumerate(operations):
            proc_times = operation['proc_times']
            print(f"    Operation {op_idx + 1}: {proc_times}")
    
    total_ops = sum(len(ops) for ops in TEST_JOBS_DATA.values())
    print(f"\nTotal operations: {total_ops}")
    print("=" * 60)

# --- 1. Gantt Chart Plotter ---
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
                    # Extract job number from "J0-O1"
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

# --- 2. Dynamic RL Environment with Dynamic Action Space ---
class DynamicFJSPEnvV2(gym.Env):
    """
    Improved Dynamic FJSP Environment with dynamic action space.
    Actions are (job_id, machine_id) pairs that are valid at the current step.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, job_arrival_times=None, reward_mode="makespan_increment"):
        super().__init__()
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        self.reward_mode = reward_mode
        
        if job_arrival_times is None:
            self.job_arrival_times = {job_id: 0 for job_id in self.job_ids}
        else:
            self.job_arrival_times = job_arrival_times

        # Calculate maximum possible actions
        max_possible_actions = self.num_jobs * len(self.machines)
        self.action_space = spaces.Discrete(max_possible_actions)
        
        # Observation space
        obs_size = (
            len(self.machines) +  # Machine availability
            self.num_jobs +       # Job progress
            self.num_jobs +       # Job arrival status
            1                     # Current makespan
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed, options=options)
            random.seed(seed)
            np.random.seed(seed)
        
        # State tracking
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation_index = {job_id: 0 for job_id in self.job_ids}
        
        self.current_time = 0.0
        self.operations_scheduled = 0
        self.episode_step = 0
        self.max_episode_steps = self.total_operations * 3
        
        # Handle job arrivals at time 0
        self.arrived_jobs = {
            job_id for job_id, arrival_time in self.job_arrival_times.items()
            if arrival_time <= self.current_time
        }
        
        # Current valid actions
        self.valid_actions = []
        self._update_valid_actions()
        
        return self._get_observation(), {}

    def _update_valid_actions(self):
        """Update the list of valid actions based on current state"""
        self.valid_actions = []
        
        for job_id in self.job_ids:
            if job_id not in self.arrived_jobs:
                continue
                
            next_op_idx = self.next_operation_index[job_id]
            if next_op_idx >= len(self.jobs[job_id]):
                continue
                
            operation = self.jobs[job_id][next_op_idx]
            
            for machine_name in operation['proc_times'].keys():
                if machine_name in self.machines:
                    self.valid_actions.append((job_id, machine_name))
        
        if not self.valid_actions:
            self._advance_time_to_next_arrival()

    def _advance_time_to_next_arrival(self):
        """Advance time to the next job arrival if no operations are ready"""
        future_arrivals = [
            arrival for arrival in self.job_arrival_times.values() 
            if arrival > self.current_time
        ]
        
        if future_arrivals:
            next_arrival_time = min(future_arrivals)
            self.current_time = next_arrival_time
            
            self.arrived_jobs.update({
                job_id for job_id, arrival_time in self.job_arrival_times.items()
                if arrival_time <= self.current_time
            })
            
            self._update_valid_actions()

    def action_masks(self):
        """Return action mask for MaskablePPO"""
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        for i in range(len(self.valid_actions)):
            if i < self.action_space.n:
                mask[i] = True
        
        if not np.any(mask) and self.action_space.n > 0:
            mask[0] = True
            
        return mask

    def step(self, action):
        self.episode_step += 1
        
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        if action >= len(self.valid_actions):
            return self._get_observation(), -100.0, False, False, {"error": "Invalid action index"}
        
        job_id, machine_name = self.valid_actions[action]
        next_op_idx = self.next_operation_index[job_id]
        
        if next_op_idx >= len(self.jobs[job_id]):
            return self._get_observation(), -100.0, False, False, {"error": "Job already complete"}
        
        operation = self.jobs[job_id][next_op_idx]
        if machine_name not in operation['proc_times']:
            return self._get_observation(), -100.0, False, False, {"error": "Machine cannot process operation"}
        
        # Schedule the operation
        proc_time = operation['proc_times'][machine_name]
        machine_available_time = self.machine_next_free.get(machine_name, 0.0)
        
        if next_op_idx > 0:
            prev_op_end_time = self.operation_end_times[job_id][next_op_idx - 1]
        else:
            prev_op_end_time = self.job_arrival_times.get(job_id, 0.0)
        
        start_time = max(self.current_time, machine_available_time, prev_op_end_time)
        end_time = start_time + proc_time
        
        # Update state
        previous_makespan = max(self.machine_next_free.values()) if self.machine_next_free else 0.0
        self.machine_next_free[machine_name] = end_time
        self.operation_end_times[job_id][next_op_idx] = end_time
        self.next_operation_index[job_id] += 1
        self.operations_scheduled += 1
        
        self.current_time = max(self.current_time, end_time)
        newly_arrived = {
            j_id for j_id, arrival in self.job_arrival_times.items()
            if previous_makespan < arrival <= self.current_time
        }
        self.arrived_jobs.update(newly_arrived)

        self.schedule[machine_name].append((f"J{job_id}-O{next_op_idx+1}", start_time, end_time))
        self._update_valid_actions()

        terminated = self.operations_scheduled >= self.total_operations
        
        current_makespan = max(self.machine_next_free.values()) if self.machine_next_free else 0.0
        idle_time = max(0, start_time - machine_available_time)
        reward = self._calculate_reward(proc_time, idle_time, terminated, previous_makespan, current_makespan)
        
        info = {
            "makespan": current_makespan,
            "valid_actions_count": len(self.valid_actions),
            "operations_scheduled": self.operations_scheduled
        }
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan=None, current_makespan=None):
        if self.reward_mode == "makespan_increment":
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment
                if done:
                    reward += 100.0
                return reward
            else:
                return -proc_time
        else:
            reward = 10.0
            reward -= proc_time * 0.1
            reward -= idle_time * 0.5
            if done:
                reward += 200.0
                if current_makespan and current_makespan > 0:
                    reward += max(0, 300.0 / current_makespan)
            return reward

    def _get_observation(self):
        """Generate observation vector"""
        obs = []
        max_time = max(max(self.machine_next_free.values(), default=0), self.current_time, 1.0)
        
        for machine in self.machines:
            availability = self.machine_next_free.get(machine, 0.0) / max_time
            obs.append(min(1.0, max(0.0, availability)))
        
        for job_id in self.job_ids:
            total_ops = len(self.jobs[job_id])
            progress = self.next_operation_index[job_id] / max(total_ops, 1)
            obs.append(min(1.0, max(0.0, progress)))
            
        for job_id in self.job_ids:
            arrived = 1.0 if job_id in self.arrived_jobs else 0.0
            obs.append(arrived)
            
        makespan_norm = self.current_time / max_time
        obs.append(min(1.0, max(0.0, makespan_norm)))
        
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array

# --- 3. SPT Heuristic Scheduler ---
def spt_heuristic_scheduler(jobs_data, machine_list, job_arrival_times):
    """SPT (Shortest Processing Time) heuristic scheduler"""
    print("\n--- Running SPT Heuristic Scheduler ---")
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    
    schedule = {m: [] for m in machine_list}
    operations_scheduled_count = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    
    arrived_jobs = {job_id for job_id, arrival in job_arrival_times.items() if arrival <= 0}
    current_time = 0.0
    
    while operations_scheduled_count < total_operations:
        # Find all ready operations
        ready_operations = []
        
        for job_id in jobs_data.keys():
            if job_id not in arrived_jobs:
                continue
                
            next_op = next_operation_for_job[job_id]
            if next_op < len(jobs_data[job_id]):
                operation = jobs_data[job_id][next_op]
                
                # Check precedence constraint
                if next_op == 0:
                    ready_time = job_arrival_times.get(job_id, 0.0)
                else:
                    ready_time = operation_end_times[job_id][next_op - 1]
                
                # Find shortest processing time for this operation
                min_proc_time = min(operation['proc_times'].values())
                best_machine = min(operation['proc_times'], key=operation['proc_times'].get)
                
                ready_operations.append((min_proc_time, job_id, next_op, best_machine, ready_time))
        
        if not ready_operations:
            # Advance time to next job arrival
            future_arrivals = [arrival for arrival in job_arrival_times.values() if arrival > current_time]
            if future_arrivals:
                next_arrival = min(future_arrivals)
                current_time = next_arrival
                arrived_jobs.update({
                    job_id for job_id, arrival_time in job_arrival_times.items()
                    if arrival_time <= current_time
                })
                continue
            else:
                break
        
        # Sort by shortest processing time
        ready_operations.sort(key=lambda x: x[0])
        
        # Schedule the operation with shortest processing time
        _, job_id, op_idx, machine, job_ready_time = ready_operations[0]
        proc_time = jobs_data[job_id][op_idx]['proc_times'][machine]
        
        machine_available_time = machine_next_free.get(machine, 0.0)
        start_time = max(current_time, machine_available_time, job_ready_time)
        end_time = start_time + proc_time
        
        # Update state
        machine_next_free[machine] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled_count += 1
        current_time = max(current_time, end_time)
        
        # Check for newly arrived jobs
        newly_arrived = {
            j_id for j_id, arrival in job_arrival_times.items()
            if arrival <= current_time and j_id not in arrived_jobs
        }
        arrived_jobs.update(newly_arrived)
        
        schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"SPT Heuristic Makespan: {makespan:.2f}")
    return makespan, schedule

# --- 4. MILP Optimal Scheduler ---
def milp_scheduler(jobs_data, machine_list, job_arrival_times):
    """MILP optimal scheduler using PuLP"""
    if not PULP_AVAILABLE:
        print("PULP not available, skipping MILP solver")
        return float('inf'), {}
    
    print("\n--- Running MILP Optimal Scheduler ---")
    
    try:
        # Problem setup
        prob = LpProblem("FJSP", LpMinimize)
        
        # Calculate big M
        max_proc_time = max(
            max(op['proc_times'].values()) 
            for job_ops in jobs_data.values() 
            for op in job_ops
        )
        total_proc_time = sum(
            sum(op['proc_times'].values()) 
            for job_ops in jobs_data.values() 
            for op in job_ops
        )
        big_M = total_proc_time + max(job_arrival_times.values()) + 1000
        
        # Variables
        # x[j, o, m] = 1 if operation o of job j is processed on machine m
        x = {}
        # s[j, o] = start time of operation o of job j
        s = {}
        # C_max = makespan
        C_max = LpVariable("C_max", lowBound=0)
        
        for job_id in jobs_data:
            for op_idx in range(len(jobs_data[job_id])):
                s[job_id, op_idx] = LpVariable(f"s_{job_id}_{op_idx}", lowBound=0)
                
                for machine in jobs_data[job_id][op_idx]['proc_times']:
                    x[job_id, op_idx, machine] = LpVariable(f"x_{job_id}_{op_idx}_{machine}", cat='Binary')
        
        # Objective: minimize makespan
        prob += C_max
        
        # Constraints
        # 1. Each operation must be assigned to exactly one machine
        for job_id in jobs_data:
            for op_idx in range(len(jobs_data[job_id])):
                prob += lpSum(
                    x[job_id, op_idx, machine] 
                    for machine in jobs_data[job_id][op_idx]['proc_times']
                ) == 1
        
        # 2. Precedence constraints
        for job_id in jobs_data:
            for op_idx in range(1, len(jobs_data[job_id])):
                prev_op_proc_time = lpSum(
                    x[job_id, op_idx-1, machine] * jobs_data[job_id][op_idx-1]['proc_times'][machine]
                    for machine in jobs_data[job_id][op_idx-1]['proc_times']
                )
                prob += s[job_id, op_idx] >= s[job_id, op_idx-1] + prev_op_proc_time
        
        # 3. Job arrival constraints
        for job_id in jobs_data:
            prob += s[job_id, 0] >= job_arrival_times.get(job_id, 0)
        
        # 4. Machine capacity constraints (no two operations on same machine at same time)
        for machine in machine_list:
            operations_on_machine = []
            for job_id in jobs_data:
                for op_idx in range(len(jobs_data[job_id])):
                    if machine in jobs_data[job_id][op_idx]['proc_times']:
                        operations_on_machine.append((job_id, op_idx))
            
            # For each pair of operations that could be on this machine
            for i, (job1, op1) in enumerate(operations_on_machine):
                for job2, op2 in operations_on_machine[i+1:]:
                    # Binary variable for ordering
                    y = LpVariable(f"y_{job1}_{op1}_{job2}_{op2}_{machine}", cat='Binary')
                    
                    proc_time_1 = jobs_data[job1][op1]['proc_times'][machine]
                    proc_time_2 = jobs_data[job2][op2]['proc_times'][machine]
                    
                    # If both operations are on this machine, they cannot overlap
                    prob += s[job1, op1] + proc_time_1 <= s[job2, op2] + big_M * (1 - y) + big_M * (2 - x[job1, op1, machine] - x[job2, op2, machine])
                    prob += s[job2, op2] + proc_time_2 <= s[job1, op1] + big_M * y + big_M * (2 - x[job1, op1, machine] - x[job2, op2, machine])
        
        # 5. Makespan constraints
        for job_id in jobs_data:
            last_op = len(jobs_data[job_id]) - 1
            last_op_proc_time = lpSum(
                x[job_id, last_op, machine] * jobs_data[job_id][last_op]['proc_times'][machine]
                for machine in jobs_data[job_id][last_op]['proc_times']
            )
            prob += C_max >= s[job_id, last_op] + last_op_proc_time
        
        # Solve the problem
        print("Solving MILP... (this may take a while)")
        prob.solve(PULP_CBC_CMD(msg=1, timeLimit=300))  # 5 minute time limit
        
        if prob.status != 1:  # Not optimal
            print(f"MILP solver status: {prob.status} (not optimal)")
            return float('inf'), {}
        
        # Extract solution
        makespan = C_max.varValue
        schedule = {m: [] for m in machine_list}
        
        for job_id in jobs_data:
            for op_idx in range(len(jobs_data[job_id])):
                start_time = s[job_id, op_idx].varValue
                
                for machine in jobs_data[job_id][op_idx]['proc_times']:
                    if x[job_id, op_idx, machine].varValue > 0.5:  # Machine is selected
                        proc_time = jobs_data[job_id][op_idx]['proc_times'][machine]
                        end_time = start_time + proc_time
                        schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
                        break
        
        print(f"MILP Optimal Makespan: {makespan:.2f}")
        return makespan, schedule
        
    except Exception as e:
        print(f"Error in MILP solver: {e}")
        return float('inf'), {}

# --- 5. RL Agent Functions ---
def mask_fn(env):
    """Mask function for ActionMasker wrapper"""
    return env.action_masks()

def train_rl_agent(jobs_data, machine_list, job_arrival_times, total_timesteps=10000):
    """Train RL agent"""
    print(f"\n--- Training RL Agent ({total_timesteps} timesteps) ---")
    
    def make_env():
        env = DynamicFJSPEnvV2(jobs_data, machine_list, job_arrival_times, reward_mode="makespan_increment")
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = ActionMasker(vec_env, mask_fn)
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[128, 128],
            activation_fn=torch.nn.ReLU
        )
    )
    
    model.learn(total_timesteps=total_timesteps)
    return model

def evaluate_rl_agent(model, jobs_data, machine_list, job_arrival_times, num_episodes=10):
    """Evaluate RL agent"""
    print(f"\n--- Evaluating RL Agent ({num_episodes} episodes) ---")
    
    test_env = DynamicFJSPEnvV2(jobs_data, machine_list, job_arrival_times, reward_mode="makespan_increment")
    
    best_makespan = float('inf')
    best_schedule = None
    makespans = []
    
    for episode in range(num_episodes):
        obs, _ = test_env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 1000:
            action_mask = test_env.action_masks()
            
            if not np.any(action_mask):
                break
            
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            step_count += 1
            
            if done:
                makespan = info.get("makespan", float('inf'))
                makespans.append(makespan)
                
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = test_env.schedule.copy()
                break
    
    avg_makespan = np.mean(makespans) if makespans else float('inf')
    std_makespan = np.std(makespans) if makespans else 0
    
    print(f"RL Agent - Best: {best_makespan:.2f}, Avg: {avg_makespan:.2f} ± {std_makespan:.2f}")
    return best_makespan, best_schedule

# --- 6. Main Comparison Function ---
def comprehensive_comparison():
    """Run comprehensive comparison of all methods"""
    print_instance_data()
    
    results = {}
    schedules = {}
    
    # 1. SPT Heuristic
    start_time = time.time()
    spt_makespan, spt_schedule = spt_heuristic_scheduler(TEST_JOBS_DATA, TEST_MACHINE_LIST, TEST_JOB_ARRIVAL_TIMES)
    spt_time = time.time() - start_time
    results['SPT'] = spt_makespan
    schedules['SPT'] = spt_schedule
    
    # 2. MILP Optimal
    start_time = time.time()
    milp_makespan, milp_schedule = milp_scheduler(TEST_JOBS_DATA, TEST_MACHINE_LIST, TEST_JOB_ARRIVAL_TIMES)
    milp_time = time.time() - start_time
    results['MILP'] = milp_makespan
    schedules['MILP'] = milp_schedule
    
    # 3. RL Agent
    start_time = time.time()
    rl_model = train_rl_agent(TEST_JOBS_DATA, TEST_MACHINE_LIST, TEST_JOB_ARRIVAL_TIMES, total_timesteps=5000)
    rl_makespan, rl_schedule = evaluate_rl_agent(rl_model, TEST_JOBS_DATA, TEST_MACHINE_LIST, TEST_JOB_ARRIVAL_TIMES)
    rl_time = time.time() - start_time
    results['RL'] = rl_makespan
    schedules['RL'] = rl_schedule
    
    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Method':<15} {'Makespan':<12} {'Time (s)':<10} {'Gap from Best (%)':<15}")
    print("-" * 60)
    
    best_makespan = min(results.values())
    
    print(f"{'SPT Heuristic':<15} {spt_makespan:<12.2f} {spt_time:<10.2f} {((spt_makespan - best_makespan) / best_makespan * 100):<15.2f}")
    if milp_makespan < float('inf'):
        print(f"{'MILP Optimal':<15} {milp_makespan:<12.2f} {milp_time:<10.2f} {((milp_makespan - best_makespan) / best_makespan * 100):<15.2f}")
    else:
        print(f"{'MILP Optimal':<15} {'Failed':<12} {milp_time:<10.2f} {'-':<15}")
    print(f"{'RL Agent':<15} {rl_makespan:<12.2f} {rl_time:<10.2f} {((rl_makespan - best_makespan) / best_makespan * 100):<15.2f}")
    print("=" * 60)
    
    # Plot Gantt charts for comparison
    if spt_schedule:
        plot_gantt(spt_schedule, TEST_MACHINE_LIST, f"SPT Heuristic (Makespan: {spt_makespan:.2f})")
    
    if milp_schedule and milp_makespan < float('inf'):
        plot_gantt(milp_schedule, TEST_MACHINE_LIST, f"MILP Optimal (Makespan: {milp_makespan:.2f})")
    
    if rl_schedule:
        plot_gantt(rl_schedule, TEST_MACHINE_LIST, f"RL Agent (Makespan: {rl_makespan:.2f})")

if __name__ == "__main__":
    comprehensive_comparison()
