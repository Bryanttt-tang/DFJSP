import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import collections
import matplotlib.pyplot as plt

# Import stable-baselines3 and sb3-contrib
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# --- 1. Data Processing ---
def parse_fjsp_brandimarte(filepath):
    """
    Parses a Flexible Job Shop Scheduling Problem (FJSP) dataset file
    in Brandimarte (mk01.txt) format.
    """
    jobs_data = collections.OrderedDict()
    all_machines = set()

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            first_line_parts = list(map(int, lines[0].strip().split()))
            num_jobs = first_line_parts[0]
            num_machines = first_line_parts[1]

            job_counter = 0
            for line in lines[1:]:
                parts = list(map(int, line.strip().split()))
                if not parts:
                    continue
                
                job_id = f'J{job_counter}'
                jobs_data[job_id] = []

                num_operations_for_job = parts[0]
                current_idx = 1

                for op_idx in range(num_operations_for_job):
                    op_data = {'op_id': op_idx, 'proc_times': {}}
                    num_machines_for_op = parts[current_idx]
                    current_idx += 1
                    
                    for _ in range(num_machines_for_op):
                        machine_idx = parts[current_idx]
                        processing_time = parts[current_idx + 1]
                        machine_name = f'M{machine_idx - 1}'
                        
                        op_data['proc_times'][machine_name] = processing_time
                        all_machines.add(machine_name)
                        current_idx += 2
                    jobs_data[job_id].append(op_data)
                job_counter += 1
                if job_counter == num_jobs:
                    break
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return None, None
            
    machine_list = sorted(list(all_machines), key=lambda x: int(x[1:]))
    return jobs_data, machine_list

# --- 2. Fixed RL Environment (FJSPEnv) ---
class FJSPEnv(gym.Env):
    """Fixed FJSP Environment with proper action masking."""
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

        # Action space: for each ready operation, choose which machine to assign it to
        # We'll use a simpler action encoding: action = operation_index * num_machines + machine_index
        max_ready_ops = self.num_jobs  # At most one ready operation per job
        self.action_space = spaces.Discrete(max_ready_ops * len(self.machines))

        # Observation space (without action mask in observation)
        obs_size = (
            len(self.machines) +  # machine availability times
            self.num_jobs * self.max_ops_per_job +  # operation completion status
            self.num_jobs +  # job progress
            len(self.machines) +  # machine utilization
            1  # current makespan
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    def _calculate_optimal_lower_bound(self):
        """Calculate a lower bound on optimal makespan"""
        total_min_proc_time_all_ops = 0
        for job_ops in self.jobs.values():
            for op_data in job_ops:
                if op_data['proc_times']:
                    min_time_for_op = min(op_data['proc_times'].values())
                    total_min_proc_time_all_ops += min_time_for_op
                else:
                    return 1.0  # Avoid division by zero
        
        if not self.machines:
            return 1.0
        
        machine_load_lb = total_min_proc_time_all_ops / len(self.machines)
        
        max_job_length_lb = 0
        for job_ops in self.jobs.values():
            job_total_min_time = sum(min(op['proc_times'].values()) for op in job_ops if op['proc_times'])
            max_job_length_lb = max(max_job_length_lb, job_total_min_time)
            
        return max(machine_load_lb, max_job_length_lb, 1.0)

    def reset(self, seed=None, options=None):
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

    def _get_ready_operations(self):
        """Get list of ready operations that can be scheduled"""
        ready_ops = []
        for job_id in self.job_ids:
            op_idx = self.next_operation[job_id]
            if op_idx < len(self.jobs[job_id]):
                ready_ops.append((job_id, op_idx, self.jobs[job_id][op_idx]))
        return ready_ops

    def _decode_action(self, action):
        """Decode action into operation and machine choice"""
        ready_ops = self._get_ready_operations()
        
        if not ready_ops:
            return None, None, None
            
        num_machines = len(self.machines)
        op_choice = action // num_machines
        machine_choice = action % num_machines
        
        if op_choice >= len(ready_ops):
            return None, None, None
            
        job_id, op_idx, op_data = ready_ops[op_choice]
        machine_name = self.machines[machine_choice]
        
        # Check if this machine can process this operation
        if machine_name not in op_data['proc_times']:
            return None, None, None
            
        return job_id, op_idx, machine_name

    def action_masks(self):
        """Generate action mask for MaskablePPO"""
        ready_ops = self._get_ready_operations()
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        num_machines = len(self.machines)
        
        for op_idx, (job_id, operation_idx, op_data) in enumerate(ready_ops):
            for machine_idx, machine_name in enumerate(self.machines):
                if machine_name in op_data['proc_times']:
                    action = op_idx * num_machines + machine_idx
                    if action < self.action_space.n:
                        mask[action] = True
        
        return mask

    def step(self, action):
        self.step_count += 1
        
        job_id, op_idx, machine_name = self._decode_action(action)
        
        terminated = False
        truncated = False
        info = {}
        
        # Check for invalid action
        if job_id is None or op_idx is None or machine_name is None:
            reward = -10.0  # Penalty for invalid action
            info = {"invalid_action": True, "makespan": self.current_makespan}
            return self._get_observation(), reward, terminated, truncated, info

        # Schedule the operation
        machine_available = self.machine_next_free[machine_name]
        job_ready = self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 else 0.0
        
        start_time = max(machine_available, job_ready)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine_name]
        end_time = start_time + proc_time

        # Update state
        self.machine_next_free[machine_name] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        
        old_makespan = self.current_makespan
        self.current_makespan = max(self.current_makespan, end_time)

        self.schedule[machine_name].append((f"{job_id}-O{op_idx+1}", start_time, end_time))

        terminated = self.operations_scheduled == self.total_operations

        # Calculate reward
        idle_time = start_time - machine_available
        reward = self._calculate_reward(proc_time, idle_time, terminated, old_makespan)

        info = {
            "makespan": self.current_makespan,
            "operations_scheduled": self.operations_scheduled,
            "efficiency": self.optimal_makespan_lb / max(self.current_makespan, 0.1) if self.optimal_makespan_lb > 0 else 0,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _calculate_reward(self, proc_time, idle_time, done, old_makespan):
        """Calculate reward with better shaping"""
        if self.reward_mode == "optimized":
            # Base reward for making progress
            progress_reward = 1.0
            
            # Penalty for idle time
            idle_penalty = -0.5 * idle_time
            
            # Reward for efficiency (shorter processing times are better)
            efficiency_reward = 10.0 / max(proc_time, 1.0)
            
            reward = progress_reward + idle_penalty + efficiency_reward
            
            if done:
                # Terminal reward based on final makespan quality
                if self.optimal_makespan_lb > 0:
                    quality_ratio = self.optimal_makespan_lb / max(self.current_makespan, 0.1)
                    terminal_reward = 100.0 * quality_ratio
                    
                    # Bonus for very good solutions
                    if quality_ratio >= 0.95:
                        terminal_reward += 50.0
                    
                    reward += terminal_reward
                else:
                    reward += -100.0  # Penalty if something went wrong
            
            return reward
        
        elif self.reward_mode == "sparse":
            if done:
                return -self.current_makespan
            return 0.0
            
        return 0.0

    def _get_observation(self):
        """Build observation vector without action mask"""
        obs = []
        norm_factor_time = max(self.optimal_makespan_lb * 2, 100.0)

        # Machine availability times
        for machine in self.machines:
            obs.append(min(self.machine_next_free[machine] / norm_factor_time, 1.0))
        
        # Operation completion status
        for job_id in self.job_ids:
            for op_idx in range(self.max_ops_per_job):
                if op_idx < len(self.jobs[job_id]):
                    obs.append(1.0 if self.completed_ops[job_id][op_idx] else 0.0)
                else:
                    obs.append(0.0)
        
        # Job progress
        for job_id in self.job_ids:
            next_op = self.next_operation[job_id]
            total_ops_in_job = len(self.jobs[job_id])
            obs.append(next_op / total_ops_in_job if total_ops_in_job > 0 else 1.0)
        
        # Machine utilization
        for machine in self.machines:
            utilization_time = self.machine_next_free[machine]
            obs.append(min(utilization_time / norm_factor_time, 1.0))
        
        # Current makespan
        obs.append(min(self.current_makespan / norm_factor_time, 1.0))

        return np.array(obs, dtype=np.float32)

    def render(self, mode="human"):
        print(f"Step: {self.step_count}, Makespan: {self.current_makespan:.2f}")
        print(f"Operations Scheduled: {self.operations_scheduled}/{self.total_operations}")
        
        # Print ready operations
        ready_ops = self._get_ready_operations()
        print(f"Ready operations: {len(ready_ops)}")
        for job_id, op_idx, _ in ready_ops:
            print(f"  {job_id} operation {op_idx}")

    def close(self):
        pass

    def plot_gantt_chart(self):
        """Plot a Gantt chart of the current schedule"""
        if not any(self.schedule.values()):
            print("No operations scheduled yet.")
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.job_ids)))
        job_colors = {job_id: colors[i] for i, job_id in enumerate(self.job_ids)}
        
        y_pos = 0
        machine_positions = {}
        
        for machine in self.machines:
            machine_positions[machine] = y_pos
            operations = self.schedule[machine]
            
            for op_name, start_time, end_time in operations:
                job_id = op_name.split('-')[0]
                duration = end_time - start_time
                
                ax.barh(y_pos, duration, left=start_time, height=0.6, 
                       color=job_colors[job_id], alpha=0.8, 
                       edgecolor='black', linewidth=1)
                
                # Add operation label
                ax.text(start_time + duration/2, y_pos, op_name, 
                       ha='center', va='center', fontsize=8, fontweight='bold')
            
            y_pos += 1
        
        ax.set_yticks(list(machine_positions.values()))
        ax.set_yticklabels(list(machine_positions.keys()))
        ax.set_xlabel('Time')
        ax.set_ylabel('Machines')
        ax.set_title(f'FJSP Schedule - Makespan: {self.current_makespan:.2f}')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=job_colors[job_id], alpha=0.8) 
                  for job_id in self.job_ids]
        ax.legend(handles, self.job_ids, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        plt.show()

# --- 3. Heuristic Scheduler ---
class HeuristicScheduler:
    """Implements the Shortest Processing Time (SPT) rule for FJSP."""
    def __init__(self, jobs_data, machine_list):
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        self.schedule = {m: [] for m in self.machines}

    def run_spt(self):
        machine_next_free = {m: 0.0 for m in self.machines}
        operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        next_operation_for_job = {job_id: 0 for job_id in self.job_ids}
        
        operations_scheduled_count = 0
        current_makespan = 0.0

        while operations_scheduled_count < self.total_operations:
            candidate_operations = []

            # Find all ready operations
            for job_id in self.job_ids:
                op_idx = next_operation_for_job[job_id]
                if op_idx < len(self.jobs[job_id]):
                    op_data = self.jobs[job_id][op_idx]
                    
                    job_ready_time = operation_end_times[job_id][op_idx - 1] if op_idx > 0 else 0.0

                    # Find best machine for this operation
                    for machine_name, proc_time in op_data['proc_times'].items():
                        machine_ready_time = machine_next_free[machine_name]
                        start_time = max(machine_ready_time, job_ready_time)
                        
                        candidate_operations.append((
                            proc_time,  # SPT: shortest processing time first
                            start_time,  # Tie-breaker: earliest start time
                            job_id, 
                            op_idx, 
                            machine_name,
                            proc_time
                        ))
            
            if not candidate_operations:
                break

            # Select operation with shortest processing time
            selected_op = min(candidate_operations, key=lambda x: (x[0], x[1]))
            _, start_time, job_id, op_idx, machine_name, proc_time = selected_op
            
            end_time = start_time + proc_time

            # Update state
            machine_next_free[machine_name] = end_time
            operation_end_times[job_id][op_idx] = end_time
            next_operation_for_job[job_id] += 1
            operations_scheduled_count += 1
            current_makespan = max(current_makespan, end_time)
            
            # Record in schedule
            self.schedule[machine_name].append((f"{job_id}-O{op_idx+1}", start_time, end_time))

        return current_makespan

    def plot_gantt_chart(self, makespan):
        """Plot Gantt chart for heuristic solution"""
        if not any(self.schedule.values()):
            print("No operations in heuristic schedule.")
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.job_ids)))
        job_colors = {job_id: colors[i] for i, job_id in enumerate(self.job_ids)}
        
        y_pos = 0
        machine_positions = {}
        
        for machine in self.machines:
            machine_positions[machine] = y_pos
            operations = self.schedule[machine]
            
            for op_name, start_time, end_time in operations:
                job_id = op_name.split('-')[0]
                duration = end_time - start_time
                
                ax.barh(y_pos, duration, left=start_time, height=0.6, 
                       color=job_colors[job_id], alpha=0.8, 
                       edgecolor='black', linewidth=1)
                
                ax.text(start_time + duration/2, y_pos, op_name, 
                       ha='center', va='center', fontsize=8, fontweight='bold')
            
            y_pos += 1
        
        ax.set_yticks(list(machine_positions.values()))
        ax.set_yticklabels(list(machine_positions.keys()))
        ax.set_xlabel('Time')
        ax.set_ylabel('Machines')
        ax.set_title(f'Heuristic (SPT) Schedule - Makespan: {makespan:.2f}')
        ax.grid(True, alpha=0.3)
        
        handles = [plt.Rectangle((0,0),1,1, color=job_colors[job_id], alpha=0.8) 
                  for job_id in self.job_ids]
        ax.legend(handles, self.job_ids, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        plt.show()

# --- 4. Training and Evaluation ---
LOG_DIR = "./fjsp_rl_logs/"
os.makedirs(LOG_DIR, exist_ok=True)

def mask_fn(env):
    """Function to get action mask for ActionMasker wrapper"""
    return env.action_masks()

class SaveOnBestMakespanCallback(BaseCallback):
    """Callback for saving the best model during training"""
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_makespan = float('inf')
        self.eval_env = None

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate current model
            makespan = self._evaluate_model()
            
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                if self.verbose > 0:
                    print(f"New best makespan: {self.best_makespan:.2f}")
                self.model.save(os.path.join(self.save_path, "best_model"))
        return True

    def _evaluate_model(self):
        """Evaluate the current model"""
        obs = self.training_env.reset()
        done = False
        info = {}

        while not done:
            # Check if the model is a MaskablePPO model
            is_maskable = "Maskable" in self.model.__class__.__name__
            
            if is_maskable:
                # For MaskablePPO, get action masks from the vectorized environment
                action_masks = self.training_env.env_method("action_masks")
                action, _ = self.model.predict(obs, action_mask=action_masks[0], deterministic=True)
            else:
                # For standard PPO
                action, _ = self.model.predict(obs, deterministic=True)
            
            # The step function of a VecEnv returns arrays
            obs, _, dones, infos = self.training_env.step(action)
            
            # We are evaluating on a single environment, so we take the first element
            done = dones[0]
            info = infos[0]
        
        return info.get('makespan', float('inf'))

def train_rl_agent(env_class, jobs_data, machine_list, model_type='maskable', total_timesteps=20000):
    """Train RL agent with proper action masking"""
    print(f"\n--- Training {model_type.upper()} Agent ---")
    
    # Create environment
    env = env_class(jobs_data, machine_list, reward_mode="optimized")
    
    if model_type == 'maskable':
        # Wrap environment with ActionMasker for MaskablePPO
        env = ActionMasker(env, mask_fn)
        model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, 
                           gamma=0.99, ent_coef=0.01, learning_rate=3e-4)
    else:
        # Standard PPO
        model = PPO("MlpPolicy", env, verbose=1, 
                   gamma=0.99, ent_coef=0.01, learning_rate=3e-4)
    
    # Train
    log_path = os.path.join(LOG_DIR, f"{model_type}_training")
    callback = SaveOnBestMakespanCallback(check_freq=1000, log_dir=log_path)
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Evaluate trained model
    obs = env.reset()
    done = False
    episode_makespan = 0.0
    
    print(f"\n--- Evaluating {model_type.upper()} Agent ---")
    while not done:
        if model_type == 'maskable':
            action_mask = env.action_masks()
            action, _ = model.predict(obs, action_mask=action_mask, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        
        obs, _, done, _, info = env.step(action)
        episode_makespan = info.get('makespan', 0)
    
    return episode_makespan, env

def create_sample_mk01_data():
    """Create sample mk01 data if file doesn't exist"""
    mk01_content = """10 6
3 1 2 1 1 1 3 2 3 3 1 1
2 1 8 2 3
3 1 5 2 6 3 4
3 1 4 2 6 3 5
2 2 3 3 7
1 3 8
3 1 9 2 10 3 6
2 1 2 3 8
3 1 7 2 4 3 3
2 2 5 3 9
1 1 6"""
    
    return mk01_content

def main():
    # Create sample data or use existing file
    fjsp_file = os.path.join("instances", "brandimarte", "mk01.txt")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(fjsp_file), exist_ok=True)
    
    # Check if file exists, if not create sample data
    if not os.path.exists(fjsp_file):
        print(f"Creating sample mk01.txt file at {fjsp_file}")
        with open(fjsp_file, 'w') as f:
            f.write(create_sample_mk01_data())

    print(f"** Running comparison for FJSP problem: {fjsp_file} **")
    jobs_data, machine_list = parse_fjsp_brandimarte(fjsp_file)
    
    if jobs_data is None:
        return

    print(f"Parsed Data: {len(jobs_data)} jobs, {len(machine_list)} machines, {sum(len(v) for v in jobs_data.values())} total operations.")
    
    # Test heuristic method
    print("\n--- Heuristic (SPT) Solution ---")
    heuristic_scheduler = HeuristicScheduler(jobs_data, machine_list)
    spt_makespan = heuristic_scheduler.run_spt()
    print(f"Heuristic (SPT) Makespan: {spt_makespan:.2f}")
    heuristic_scheduler.plot_gantt_chart(spt_makespan)

    # Train and test RL agents
    TOTAL_TIMESTEPS = 15000
    
    # Standard PPO
    ppo_makespan, ppo_env = train_rl_agent(FJSPEnv, jobs_data, machine_list, 
                                          'standard', TOTAL_TIMESTEPS)
    print(f"Standard PPO Final Makespan: {ppo_makespan:.2f}")
    ppo_env.plot_gantt_chart()
    
    # Maskable PPO  
    maskable_makespan, maskable_env = train_rl_agent(FJSPEnv, jobs_data, machine_list, 
                                                    'maskable', TOTAL_TIMESTEPS)
    print(f"Maskable PPO Final Makespan: {maskable_makespan:.2f}")
    maskable_env.plot_gantt_chart()

    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS:")
    print("="*60)
    print(f"Heuristic (SPT) Makespan:  {spt_makespan:.2f}")
    print(f"Standard PPO Makespan:     {ppo_makespan:.2f}")
    print(f"Maskable PPO Makespan:     {maskable_makespan:.2f}")
    print("="*60)
    
    # Calculate improvements
    if spt_makespan > 0:
        ppo_improvement = ((spt_makespan - ppo_makespan) / spt_makespan) * 100
        maskable_improvement = ((spt_makespan - maskable_makespan) / spt_makespan) * 100
        
        print(f"Standard PPO vs Heuristic: {ppo_improvement:+.1f}%")
        print(f"Maskable PPO vs Heuristic: {maskable_improvement:+.1f}%")

if __name__ == "__main__":
    main()
