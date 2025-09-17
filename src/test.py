import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
import random
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
import os
import collections
from tqdm import tqdm
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.wrappers import ActionMasker


# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- 1. Data Parser for New Format ---

# --- 1.2 Data Parser for Taillard JSP Format ---
def parse_data(data_string, format_type='JSP'):
    """
    Parses a string containing Taillard JSP data.
    
    Args:
        data_string (str): The raw string data of the JSP problem.

    Returns:
        tuple: (jobs_data, machine_list)
    """
    if format_type == 'JSP':
        lines = [line.strip() for line in data_string.strip().split('\n') if line.strip()]
        if not lines:
            return {}, []

        first_line_parts = list(map(int, lines[0].split()))
        num_jobs = first_line_parts[0]
        num_machines = first_line_parts[1]
        
        machine_list = [f"M{i}" for i in range(num_machines)]
        jobs_data = collections.OrderedDict()
        
        job_id_counter = 0
        for line in lines[1:]:
            if job_id_counter >= num_jobs:
                break
                
            parts = list(map(int, line.split()))
            jobs_data[job_id_counter] = []
            
            # In Taillard, the number of operations is fixed and implicitly num_machines
            num_operations_in_job = num_machines 
            
            current_idx = 0
            for _ in range(num_operations_in_job):
                op_data = {'proc_times': {}}
                
                # For JSP, each operation has only one machine choice
                machine_idx = parts[current_idx] # Taillard machines are 0-indexed
                processing_time = parts[current_idx + 1]
                machine_name = f'M{machine_idx}'
                
                op_data['proc_times'][machine_name] = processing_time
                jobs_data[job_id_counter].append(op_data)
                current_idx += 2
                
            job_id_counter += 1
                
        return jobs_data, machine_list
    
    elif format_type == 'FJSP':
        """
        Parses a string containing FJSP data in the specified format.
        
        Args:
            data_string (str): The raw string data of the FJSP problem.

        Returns:
            tuple: (jobs_data, machine_list)
                jobs_data: A dictionary where keys are job_ids (e.g., 0) and
                            values are lists of operations. Each operation is a dict:
                            {'proc_times': {'M_name': time, ...}}
                machine_list: A sorted list of unique machine names (e.g., ['M0', 'M1', ...]).
        """
        lines = [line.strip() for line in data_string.strip().split('\n') if line.strip()]
        if not lines:
            return {}, []

        first_line_parts = list(map(int, lines[0].split()))
        num_jobs = first_line_parts[0]
        num_machines = first_line_parts[1]
        
        machine_list = [f"M{i}" for i in range(num_machines)]
        jobs_data = collections.OrderedDict()
        
        job_id_counter = 0
        for line in lines[1:]:
            parts = list(map(int, line.split()))
            jobs_data[job_id_counter] = []
            
            num_operations_in_job = parts[0]
            current_idx = 1
            
            for op_idx in range(num_operations_in_job):
                op_data = {'proc_times': {}}
                num_machine_choices = parts[current_idx]
                current_idx += 1
                
                for _ in range(num_machine_choices):
                    machine_idx = parts[current_idx]
                    processing_time = parts[current_idx + 1]
                    machine_name = f'M{machine_idx}'
                    op_data['proc_times'][machine_name] = processing_time
                    current_idx += 2
                jobs_data[job_id_counter].append(op_data)
            
            job_id_counter += 1
            if job_id_counter == num_jobs:
                break
                
        return jobs_data, machine_list

# --- 2. Gantt Chart Plotter ---
def plot_gantt(schedule, machines, title="Schedule"):
    """Plot Gantt chart for the schedule"""
    if not schedule or all(len(ops) == 0 for ops in schedule.values()):
        print("No schedule to plot - schedule is empty")
        return

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(12, len(machines)*0.8))

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
                ax.text(start + (end-start)/2, idx * 10 + 4,
                       label, color='white', fontsize=10,
                       ha='center', va='center', weight='bold')

    ax.set_yticks([i * 10 + 4 for i in range(len(machines))])
    ax.set_yticklabels(machines)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- 3. Heuristic Scheduler ---
def heuristic_spt_scheduler(jobs_data, machine_list):
    """
    Implements a corrected heuristic that schedules the operation with the
    shortest processing time that can start earliest.
    """
    print("\nRunning Corrected Heuristic Scheduler...")
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    
    schedule = {m: [] for m in machine_list}
    operations_scheduled_count = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    current_makespan = 0.0

    while operations_scheduled_count < total_operations:
        candidate_operations = []
        for job_id in jobs_data:
            op_idx = next_operation_for_job[job_id]
            if op_idx < len(jobs_data[job_id]):
                op_data = jobs_data[job_id][op_idx]
                job_ready_time = operation_end_times[job_id][op_idx - 1] if op_idx > 0 else 0.0
                
                for machine_name, proc_time in op_data['proc_times'].items():
                    earliest_start_time = max(machine_next_free[machine_name], job_ready_time)
                    candidate_operations.append((
                        earliest_start_time + proc_time,  # Completion time as key
                        proc_time,
                        earliest_start_time,
                        job_id, 
                        op_idx, 
                        machine_name
                    ))
        
        if not candidate_operations:
            break

        # Select the operation that finishes earliest
        selected_op = min(candidate_operations, key=lambda x: x[0])
        completion_time, proc_time, start_time, job_id, op_idx, machine_name = selected_op
        
        end_time = start_time + proc_time

        machine_next_free[machine_name] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled_count += 1
        current_makespan = max(current_makespan, end_time)

        schedule[machine_name].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

    print(f"Corrected Heuristic Makespan: {current_makespan:.2f}")
    return schedule, current_makespan

# --- 4. RL Environment ---
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()

class FJSPEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, reward_mode="optimized"):
        super().__init__()
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        self.reward_mode = reward_mode
        self.optimal_makespan_lb = self._calculate_optimal_lower_bound()
        self.max_steps = self.total_operations * 3
        
        # --- MODIFIED: Action space now includes a special 'no-op' action ---
        self.action_space_size = self.num_jobs * self.max_ops_per_job * len(self.machines)
        self.action_space = spaces.Discrete(self.action_space_size + 1) # +1 for no-op
        self.NO_OP_ACTION = self.action_space_size
        
        obs_size = (
            len(self.machines) +
            self.num_jobs * self.max_ops_per_job + 
            self.num_jobs +
            len(self.machines) +
            1
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.reset()
    
    def _calculate_optimal_lower_bound(self):
        if not self.jobs:
            return 0
        total_min_time = sum(min(op['proc_times'].values()) for job_ops in self.jobs.values() for op in job_ops if op['proc_times'])
        if not self.machines: 
            return 0
        machine_load_lb = total_min_time / len(self.machines)
        max_job_length_lb = max(sum(min(op['proc_times'].values()) for op in job_ops if op['proc_times']) for job_ops in self.jobs.values()) if self.jobs else 0
        return max(machine_load_lb, max_job_length_lb) if self.jobs else 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        self.current_makespan = 0.0
        self.operations_scheduled = 0
        self.step_count = 0
        return self._get_observation(), {}

    def _decode_action(self, action):
        """
        MODIFIED: Decodes a flat action integer, now including a check for the no-op action.
        """
        if action == self.NO_OP_ACTION:
            return -1, -1, -1 # Use special values for no-op

        num_machine_choices = len(self.machines)
        num_op_choices = self.max_ops_per_job
        
        job_idx = action // (num_op_choices * num_machine_choices)
        remaining = action % (num_op_choices * num_machine_choices)
        op_idx = remaining // num_machine_choices
        machine_idx = remaining % num_machine_choices
        
        if job_idx >= self.num_jobs or op_idx >= len(self.jobs.get(self.job_ids[job_idx], [])):
            return -1, -1, -1
        
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        """
        Checks if a decoded action is valid for the current state.
        This method remains the same as its logic is sound.
        """
        if job_idx < 0 or job_idx >= self.num_jobs:
            return False
        
        job_id = self.job_ids[job_idx]
        
        if op_idx < 0 or op_idx >= len(self.jobs[job_id]):
            return False
        
        if machine_idx < 0 or machine_idx >= len(self.machines):
            return False
            
        if op_idx != self.next_operation[job_id]:
            return False
            
        machine_name = self.machines[machine_idx]
        op_data = self.jobs[job_id][op_idx]
        if machine_name not in op_data['proc_times']:
            return False
            
        return True

    def action_masks(self):
        """
        MODIFIED: Generate action mask for MaskablePPO, always allowing the no-op action.
        """
        mask = np.full(self.action_space.n, False, dtype=bool)

        # Always allow the no-op action
        mask[self.NO_OP_ACTION] = True

        if self.operations_scheduled >= self.total_operations:
            return mask

        # Check all other possible actions
        for action in range(self.action_space_size):
            job_idx, op_idx, machine_idx = self._decode_action(action)
            if self._is_valid_action(job_idx, op_idx, machine_idx):
                mask[action] = True
            
        return mask

    def step(self, action):
        self.step_count += 1
        
        # --- MODIFIED: Handle the no-op action ---
        if action == self.NO_OP_ACTION:
            terminated = self.operations_scheduled == self.total_operations
            # Advance time to the next scheduled event to avoid an infinite loop of no-ops
            next_event_time = min(t for t in self.machine_next_free.values()) if any(self.machine_next_free.values()) else 0.0
            
            # Find the next job ready time
            min_job_ready_time = float('inf')
            for job_id in self.job_ids:
                op_idx = self.next_operation[job_id]
                if op_idx < len(self.jobs[job_id]):
                    job_ready_time = self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 else 0.0
                    min_job_ready_time = min(min_job_ready_time, job_ready_time)
            
            if min_job_ready_time == float('inf'):
                # No more jobs, advance to the final completion
                next_event_time = max(self.current_makespan, next_event_time)
            else:
                next_event_time = max(next_event_time, min_job_ready_time)

            self.current_makespan = next_event_time
            reward = -0.1 # Small penalty for waiting
            info = {"no_op": True, "makespan": self.current_makespan}
            return self._get_observation(), reward, terminated, False, info
        
        # --- Handle normal actions ---
        job_idx, op_idx, machine_idx = self._decode_action(action)

        # Check for invalid action (should be rare with masks, but still a safeguard)
        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            terminated = True
            reward = -500.0  # Heavy penalty for invalid action
            info = {"invalid_action": True, "makespan": self.current_makespan}
            return self._get_observation(), reward, terminated, False, info
        
        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        machine_available_time = self.machine_next_free[machine]
        job_ready_time = self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 else 0.0
        start_time = max(machine_available_time, job_ready_time)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time

        # Update state
        self.machine_next_free[machine] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        self.current_makespan = max(self.current_makespan, end_time)
        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        terminated = self.operations_scheduled == self.total_operations
        idle_time_for_machine = start_time - machine_available_time
        reward = self._calculate_reward(proc_time, idle_time_for_machine, terminated)
        
        info = {"makespan": self.current_makespan, "operations_scheduled": self.operations_scheduled}
        
        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, proc_time, idle_time, done):
        if self.reward_mode == "optimized":
            progress_reward = 1.0
            efficiency_reward = 1.0 / (proc_time + 1e-6)
            idle_penalty = -0.1 * idle_time
            reward = progress_reward + efficiency_reward + idle_penalty
            if done:
                makespan_reward_scale = 100.0
                makespan_ratio = self.optimal_makespan_lb / max(self.current_makespan, 1e-6)
                terminal_reward = makespan_reward_scale * makespan_ratio
                if self.optimal_makespan_lb > 0 and self.current_makespan <= self.optimal_makespan_lb * 1.05:
                    terminal_reward += 20.0
                reward += terminal_reward
            return reward
        elif self.reward_mode == "sparse":
            return -self.current_makespan if done else 0.0
        return 0.0

    def _get_observation(self):
        obs = []
        norm_factor_time = self.optimal_makespan_lb * 2 if self.optimal_makespan_lb > 0 else 1000.0
        
        # Machine availability times
        for machine in self.machines: 
            obs.append(self.machine_next_free[machine] / norm_factor_time)
        
        # Operation completion status
        for job_id in self.job_ids:
            for op_idx in range(self.max_ops_per_job):
                obs.append(1.0 if op_idx < len(self.jobs[job_id]) and self.completed_ops[job_id][op_idx] else 0.0)
        
        # Job progress
        for job_id in self.job_ids:
            total_ops_in_job = len(self.jobs[job_id])
            obs.append(self.next_operation[job_id] / total_ops_in_job if total_ops_in_job > 0 else 0.0)
        
        # Machine next free times (normalized)
        for machine in self.machines:
            obs.append(min(self.machine_next_free[machine] / norm_factor_time, 1.0))
        
        # Current makespan
        obs.append(min(self.current_makespan / norm_factor_time, 1.0))
        
        return np.array(obs, dtype=np.float32)

    def render(self, mode="human"):
        print(f"Step: {self.step_count}, Makespan: {self.current_makespan:.2f}, Operations: {self.operations_scheduled}/{self.total_operations}")

# --- 5. Training and Evaluation ---
LOG_DIR = "./ppo_training_logs/"
os.makedirs(LOG_DIR, exist_ok=True)

def train_and_evaluate_rl(jobs_data, machine_list, total_timesteps, log_name):
    print(f"\n--- Training {log_name} Agent ---")
    log_path = os.path.join(LOG_DIR, log_name)
    os.makedirs(log_path, exist_ok=True)

    def make_env():
        env = FJSPEnv(jobs_data, machine_list)
        env = ActionMasker(env, mask_fn)
        monitor_filename = os.path.join(log_path, f"monitor_{log_name}")
        return Monitor(env, monitor_filename)

    vec_env = make_vec_env(make_env, n_envs=1)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=2048
    )
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    print(f"\n--- Evaluating {log_name} Agent using `evaluate_policy` ---")
    
    eval_env = make_env()
    
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, warn=False)
    print(f"Evaluation results (10 episodes): mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Run one final episode to get a deterministic schedule for plotting
    print("\n--- Generating Final Schedule for Plotting ---")
    obs, _ = eval_env.reset()
    done = False
    episode_makespan = 0.0
    
    while not done:
        action_masks = get_action_masks(eval_env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        if done:
            episode_makespan = info.get("makespan", float('inf'))

    print(f"Final schedule generated with makespan: {episode_makespan:.2f}")
    schedule = eval_env.unwrapped.schedule
    
    total_scheduled = sum(len(ops) for ops in schedule.values())
    print(f"Schedule contains {total_scheduled} operations across {len([m for m, ops in schedule.items() if ops])} machines")

    return episode_makespan, log_path, schedule

def plot_results(log_folder, title='Learning Curve'):
    """Plot training results from Monitor logs"""
    print(f"\n--- Plotting Results for {log_folder} ---")
    try:
        # Check if monitor file exists
        monitor_files = [f for f in os.listdir(log_folder) if f.endswith('.monitor.csv')]
        if not monitor_files:
            print(f"No monitor files found in {log_folder}")
            return
            
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        if len(x) == 0:
            print(f"No data to plot for {log_folder}")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(x, y)
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Reward')
        plt.title(f'{title}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting results for {log_folder}: {e}")

# --- 6. Sample data for testing ---
def create_sample_data():
    """Create a small sample FJSP instance for testing"""
    sample_data = """2 3
2 2 0 3 1 4 2 1 2 5
2 1 0 2 1 1 2 3"""
    return sample_data

def load_instance(instance_family, instance_id):
    """
    Constructs the file path and reads the FJSP data from a specified instance file.
    It intelligently handles whether the '.txt' extension is needed.
    
    Args:
        instance_family (str): The family of the instance (e.g., 'brandimarte', 'Taillard').
        instance_id (str): The specific instance ID (e.g., 'mk01', 'ta02').

    Returns:
        str: The raw string data of the FJSP problem.
        
    Raises:
        FileNotFoundError: If the specified file does not exist with or without the extension.
    """
    base_path = f"instances/{instance_family}/{instance_id}"
    
    # Try opening the file without the .txt extension first
    if os.path.exists(base_path):
        file_path = base_path
    # If that fails, try with the .txt extension
    elif os.path.exists(base_path + ".txt"):
        file_path = base_path + ".txt"
    # If neither exists, raise an error
    else:
        raise FileNotFoundError(f"Data file not found: '{base_path}' or '{base_path}.txt'")
    
    with open(file_path, 'r') as f:
        raw_data = f.read()
    return raw_data
# --- 7. Main Execution Block ---
if __name__ == "__main__":
    # For mk01 instance - you can replace this with your actual mk01 data
    # mk01 optimal makespan is 40 (known from literature)
    print("--- FJSP mk01 Instance Information ---")
    print("mk01 is a well-known benchmark instance:")
    print("- 10 jobs, 6 machines")
    print("- Known optimal makespan: 40")
    print("- This is a challenging flexible job shop problem")
    print("---------------------------------------")
    
    # Use sample data for testing (replace with mk01 data loading if you have the file)
    print("\n--- Using Sample FJSP Data for Testing ---")
    print("(Replace this section with mk01 data loading)")

    instance_family = 'Taillard'  # Example family
    instance_id = 'ta04'
    format_type = 'JSP'  # Specify the format type


    raw_data = load_instance(instance_family, instance_id)
    
    print(f"--- Loaded Dataset: {instance_family}/{instance_id} ---")
    print('\n'.join(raw_data.split('\n')[:5])) # Print only the first 5 lines for brevity
    print("-----------------------------------")
    jobs_data, machine_list = parse_data(raw_data, format_type)
    print('Jobs Data:', jobs_data)

    if not jobs_data:
        print("Failed to parse the dataset. Exiting.")
    else:
        print(f"Parsed Data: {len(jobs_data)} jobs, {len(machine_list)} machines, {sum(len(v) for v in jobs_data.values())} total operations.")

        # Show job structure
        for job_id, operations in jobs_data.items():
            print(f"Job {job_id}: {len(operations)} operations")
            for i, op in enumerate(operations):
                machines_available = list(op['proc_times'].keys())
                print(f"  Operation {i}: can run on {machines_available}")
        
        # Heuristic Method
        print("\n" + "="*50)
        heuristic_schedule, heuristic_makespan = heuristic_spt_scheduler(jobs_data, machine_list)
        plot_gantt(heuristic_schedule, machine_list, f"Heuristic (SPT) Makespan: {heuristic_makespan:.2f}")

        # PPO Agent - reduced timesteps for faster testing
        print("\n" + "="*50)
        ppo_makespan, ppo_log_path, ppo_schedule = train_and_evaluate_rl(
            jobs_data, machine_list, 10000, "ppo_custom"  # Increased timesteps for better learning
        )
        print(f"\nFinal Results:")
        print(f"- Heuristic (SPT) Makespan: {heuristic_makespan:.2f}")
        print(f"- PPO Agent Makespan: {ppo_makespan:.2f}")
        
        if ppo_makespan < heuristic_makespan:
            improvement = ((heuristic_makespan - ppo_makespan) / heuristic_makespan) * 100
            print(f"- PPO improved over heuristic by {improvement:.1f}%")
        else:
            degradation = ((ppo_makespan - heuristic_makespan) / heuristic_makespan) * 100
            print(f"- PPO performed {degradation:.1f}% worse than heuristic")
        
        if ppo_schedule and sum(len(ops) for ops in ppo_schedule.values()) > 0:
            plot_gantt(ppo_schedule, machine_list, f"PPO Makespan: {ppo_makespan:.2f}")
        else:
            print("Warning: PPO schedule is empty or invalid")

        # Plot PPO training results
        plot_results(ppo_log_path, "PPO Training Rewards")
