import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
import itertools
import random
from pulp import PULP_CBC_CMD, LpProblem, LpMinimize, LpVariable, lpSum
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import pandas as pd
import io # For simulating file reading
from tqdm import tqdm

# Custom callback to show training progress
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        
    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training PPO")
        
    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(1)
        return True
        
    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- Problem Definitions (now dynamic, for example only) ---
# Original small problem
original_jobs = {
    1: [
        {'proc_times': {'M1': 2, 'M2': 4, 'M3': 3}},
        {'proc_times': {'M1': 3, 'M2': 2, 'M3': 4}}
    ],
    2: [
        {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2}},
        {'proc_times': {'M1': 2, 'M2': 3, 'M3': 4}}
    ]
}
original_machines = ["M1", "M2", "M3"]

def plot_gantt(schedule, machines, title="Schedule"):
    """Plot Gantt chart for the schedule"""
    if not schedule or all(len(ops) == 0 for ops in schedule.values()):
        print("No schedule to plot - schedule is empty")
        return

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(12, len(machines)*0.8))

    for idx, m in enumerate(machines):
        machine_ops = schedule.get(m, [])
        # Sort operations by start time for proper plotting
        machine_ops.sort(key=lambda x: x[1])

        for op_data in machine_ops:
            if len(op_data) == 3:
                job_id_str, start, end = op_data
                if isinstance(job_id_str, str):
                    # Extract job number from string like "J1-O1"
                    try:
                        j = int(job_id_str.split('-')[0][1:])
                    except ValueError: # Handle cases where it might just be 'J1' or 'OpX'
                        j = hash(job_id_str) % len(colors) # Fallback to hash for color if not parseable as JX
                else:
                    j = job_id_str # If job_id is already an integer

                ax.broken_barh(
                    [(start, end - start)],
                    (idx * 10, 8),
                    facecolors=colors[j % len(colors)],
                    edgecolor='black',
                    alpha=0.8
                )
                # Add job label
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

# --- New: Taillard Data Parser ---
def parse_taillard_data(data_string):
    """
    Parses a string containing Taillard-like Flexible Job Shop Scheduling data.

    The format is expected to be:
    - First line: <num_jobs> <num_machines>
    - Subsequent lines (one per job):
      <num_operations_in_job> <machine_1> <time_1> <machine_2> <time_2> ...
      (where machine and time pairs are repeated for each operation,
       and for each operation, there are multiple machine-time options)

    Returns:
        tuple: (jobs_data, machine_list)
            jobs_data (dict): {job_id: [{proc_times: {machine_name: time, ...}}, ...]}
            machine_list (list): ["M1", "M2", ...]
    """
    lines = data_string.strip().split('\n')
    num_jobs, num_machines = map(int, lines[0].split())

    machine_list = [f"M{i+1}" for i in range(num_machines)]
    jobs_data = {}
    job_id_counter = 1

    for line in lines[1:]:
        parts = list(map(int, line.split()))
        num_operations_in_job = parts[0]
        current_part_idx = 1 # Start reading from index 1 for operation details

        operations_for_job = []
        op_idx = 0

        while op_idx < num_operations_in_job:
            num_choices = parts[current_part_idx]
            current_part_idx += 1 # Move past the 'num_choices' integer

            current_op_proc_times = {}
            for _ in range(num_choices):
                machine_id_val = parts[current_part_idx] # 1-indexed machine ID
                proc_time = parts[current_part_idx + 1]
                current_part_idx += 2 # Move past machine_id and proc_time
                current_op_proc_times[f"M{machine_id_val}"] = proc_time

            operations_for_job.append({'proc_times': current_op_proc_times})
            op_idx += 1

        jobs_data[job_id_counter] = operations_for_job
        job_id_counter += 1

    return jobs_data, machine_list


class FJSPEnv(gym.Env):
    """Optimized FJSP Environment, now adaptable to varying problem sizes."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, reward_mode="optimized"):
        super().__init__()

        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        # Dynamically determine max_ops_per_job, the observation space must have a fixed size for most RL libraries. 
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values()) # for termination condition
        self.reward_mode = reward_mode
        # rough estimate of the best possible solution and is used to normalize the makespan and shape the reward function.
        self.optimal_makespan_lb = self._calculate_optimal_lower_bound()

        # Action space: (job_idx, operation_idx, machine_idx)
        # Needs to represent the indices for the chosen job, its operation, and the machine.
        # Max values for each: num_jobs-1, max_ops_per_job-1, len(machines)-1
        # The action space size is the product of the number of choices for each dimension.
        self.action_space = spaces.Discrete(
            self.num_jobs * self.max_ops_per_job * len(self.machines)
        )

        # Enhanced observation space - dynamically sized
        # Machine next free times: len(self.machines)
        # Operation completion status: sum(len(job_ops) for job_ops in self.jobs.values()) -- simplified to max_ops_per_job * num_jobs for fixed size
        # Next operation index for each job: num_jobs
        # Machine utilization: len(self.machines)
        # Current makespan normalized: 1
        obs_size = (
            len(self.machines) +
            self.num_jobs * self.max_ops_per_job + # Fixed size for observation, even if some ops don't exist
            self.num_jobs +
            len(self.machines) +
            1
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    def _calculate_optimal_lower_bound(self):
        """Calculate a lower bound on optimal makespan"""
        total_min_time = 0
        for job_ops in self.jobs.values():
            for op in job_ops:
                # Ensure there are processing times and get the minimum
                if op['proc_times']:
                    min_time = min(op['proc_times'].values())
                    total_min_time += min_time
                else:
                    # Handle case where an operation might have no valid machine options (shouldn't happen with valid data)
                    return 0 # Or raise an error
        # If no machines, avoid division by zero
        if not self.machines:
            return total_min_time
        return total_min_time / len(self.machines)

    # Sets up the environment for a new episode
    def reset(self):
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}

        # Track operation completion
        self.completed_ops = {}
        self.operation_end_times = {} # for precedence constraints
        for job_id in self.job_ids:
            self.completed_ops[job_id] = [False] * len(self.jobs[job_id])
            self.operation_end_times[job_id] = [0.0] * len(self.jobs[job_id])

        # Track the index of the next operation to be scheduled for each job.
        self.next_operation = {job_id: 0 for job_id in self.job_ids}

        self.current_makespan = 0.0
        self.operations_scheduled = 0
        self.step_count = 0

        return self._get_observation()

    def _decode_action(self, action):
        """Decode action into job, operation, machine indices"""
        # action = job_idx * (max_ops * num_machines) + op_idx * num_machines + machine_idx
        # Ensure that action decoding is robust to dynamically sized spaces
        num_machine_choices = len(self.machines)
        num_op_choices = self.max_ops_per_job # This is the max possible op index + 1 for any job

        job_idx = action // (num_op_choices * num_machine_choices)
        remaining = action % (num_op_choices * num_machine_choices)
        op_idx = remaining // num_machine_choices
        machine_idx = remaining % num_machine_choices
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        """Check if action is valid"""
        if (job_idx >= self.num_jobs or machine_idx >= len(self.machines)):
            return False

        job_id = self.job_ids[job_idx]

        # Check if operation exists for this job
        if op_idx >= len(self.jobs[job_id]):
            return False

        # Check if operation is already completed
        if self.completed_ops[job_id][op_idx]:
            return False

        # Check if this is the next operation for the job (precedence)
        if op_idx != self.next_operation[job_id]:
            return False

        # Check if the chosen machine is actually capable of processing this operation
        machine_name = self.machines[machine_idx]
        if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
            return False

        return True
    
    # output one discrete action per step call.
    def step(self, action):
        self.step_count += 1
        job_idx, op_idx, machine_idx = self._decode_action(action)

        # Check validity
        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            # Penalize invalid actions
            return self._get_observation(), -5.0, False, {"invalid_action": True, "makespan": self.current_makespan}

        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]

        # Calculate start time
        machine_available_time = self.machine_next_free[machine]

        if op_idx == 0:
            job_ready_time = 0.0
        else:
            job_ready_time = self.operation_end_times[job_id][op_idx - 1]

        start_time = max(machine_available_time, job_ready_time)

        # Calculate processing time and end time
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time

        # Update state
        self.machine_next_free[machine] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        self.current_makespan = max(self.current_makespan, end_time)

        # Add to schedule
        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        # Calculate reward
        done = self.operations_scheduled == self.total_operations
        reward = self._calculate_reward(proc_time, start_time - machine_available_time, done)

        info = {
            "makespan": self.current_makespan,
            "operations_scheduled": self.operations_scheduled,
            "efficiency": self.optimal_makespan_lb / max(self.current_makespan, 0.1) if self.optimal_makespan_lb > 0 else 0
        }

        return self._get_observation(), reward, done, info

    def _calculate_reward(self, proc_time, idle_time, done):
        """Calculate reward with better shaping, adapted for general problems"""
        if self.reward_mode == "optimized":
            progress_reward = 1.0 # Basic reward for scheduling an operation

            # Efficiency reward: inversely proportional to processing time.
            # Max/Min proc times should ideally come from problem stats, but for general case,
            # we'll use a heuristic or average. Let's assume a range.
            # For a general solution, we might need to pre-compute or dynamically update bounds for scaling
            # For simplicity here, let's use a simple inverse:
            efficiency_reward = 1.0 / (proc_time + 1e-6) # Add small epsilon to avoid div by zero

            # Idle time penalty
            idle_penalty = -0.1 * idle_time # Penalize more for longer idle times

            reward = progress_reward + efficiency_reward + idle_penalty

            if done:
                makespan_reward_scale = 100.0
                # Use a logarithmic or inverse scaling for makespan, relative to lower bound
                if self.current_makespan > 0 and self.optimal_makespan_lb > 0:
                    makespan_ratio = self.optimal_makespan_lb / self.current_makespan
                    terminal_reward = makespan_reward_scale * makespan_ratio
                elif self.current_makespan == 0 and self.optimal_makespan_lb == 0: # Ideal case
                    terminal_reward = makespan_reward_scale
                else: # Problematic makespan or lower bound
                    terminal_reward = -makespan_reward_scale # Large penalty

                # Add a bonus for near-optimal makespan (heuristic threshold)
                # This threshold should ideally be dynamic or learned. For now, a small fixed value
                if self.optimal_makespan_lb > 0 and self.current_makespan <= self.optimal_makespan_lb * 1.1:
                    terminal_reward += 20.0 # Small bonus for being close to optimal

                reward += terminal_reward

            return reward

        elif self.reward_mode == "sparse":
            if done:
                return -self.current_makespan
            return 0.0

        return 0.0 # Default if reward_mode is unknown

    def _get_observation(self):
        """Build observation vector, dynamically sized"""
        obs = [] # shall be normalized and has fixed size

        # Machine next free times (normalized by estimated max makespan or current makespan)
        # Using current_makespan as norm factor can be unstable early on.
        # A fixed large number or pre-calculated max possible makespan could be better.
        # For now, current_makespan is kept for simplicity, as per original.
        norm_factor = max(float(self.current_makespan), 1.0)
        for machine in self.machines:
            obs.append(float(self.machine_next_free[machine]) / norm_factor)

        # Operation completion status (fixed size using max_ops_per_job)
        for job_id in self.job_ids:
            for op_idx in range(self.max_ops_per_job):
                if op_idx < len(self.jobs[job_id]): # Actual operations for this job
                    obs.append(1.0 if self.completed_ops[job_id][op_idx] else 0.0)
                else: # Pad with zeros for jobs with fewer operations than max_ops_per_job
                    obs.append(0.0)

        # Next operation index for each job ( Normalized by the total number of operations for that job)
        for job_id in self.job_ids:
            next_op = float(self.next_operation[job_id])
            total_ops_in_job = float(len(self.jobs[job_id]))
            obs.append(next_op / total_ops_in_job if total_ops_in_job > 0 else 0.0)

        # Machine utilization (normalized by current makespan)
        total_time = max(float(self.current_makespan), 1.0)
        for machine in self.machines:
            utilization = float(self.machine_next_free[machine]) / total_time
            obs.append(min(utilization, 1.0)) # Clip at 1.0

        # Current makespan (normalized by a rough upper bound)
        # Using optimal_makespan_lb * 3 as upper bound for normalization,
        # but a more robust upper bound could be sum of all max processing times.
        rough_max_makespan = float(self.optimal_makespan_lb) * 5.0 # Heuristic
        if rough_max_makespan == 0: rough_max_makespan = 100.0 # Avoid div by zero if lower bound is 0
        obs.append(min(float(self.current_makespan) / rough_max_makespan, 1.0))

        # Ensure all values are proper numpy float32
        obs_array = np.array(obs, dtype=np.float32)
        # Double check dtype
        if obs_array.dtype != np.float32:
            obs_array = obs_array.astype(np.float32)
        return obs_array

    def render(self, mode="human"):
        """Render current state"""
        print(f"Step: {self.step_count}, Makespan: {self.current_makespan:.2f}")
        print(f"Operations: {self.operations_scheduled}/{self.total_operations}")


# --- Exact Search Method (No changes needed, as it accepts generic jobs/machines) ---
def exact_search(jobs, machines):
    """Exhaustive search for optimal solution"""
    job_ids = list(jobs.keys())

    # Build list of all operations
    op_indices = []
    for j in job_ids:
        for oi in range(len(jobs[j])):
            op_indices.append((j, oi))

    # For each operation, possible machines
    op_choices = []
    for (j, oi) in op_indices:
        # Get machine options for the current operation
        possible_machines = list(jobs[j][oi]['proc_times'].keys())
        if not possible_machines:
            # This case should ideally not happen with valid input, but good to handle
            raise ValueError(f"Operation J{j}-O{oi+1} has no valid machine options.")
        op_choices.append(possible_machines)

    # Enumerate all possible assignments
    # Handle case with no operations
    if not op_choices:
        return {m: [] for m in machines}, 0.0

    all_assignments = list(itertools.product(*op_choices))
    best_makespan = float('inf')
    best_schedule = None

    for assignment in all_assignments:
        sch = {m: [] for m in machines}
        completion = {j: 0 for j in job_ids}

        assign_map = {op_indices[i]: assignment[i] for i in range(len(op_indices))}

        for j in job_ids:
            prev_end = 0
            for oi in range(len(jobs[j])):
                op_identifier = (j, oi)
                m = assign_map[op_identifier]
                op = jobs[j][oi]

                last_end = sch[m][-1][2] if sch[m] else 0
                start_time = max(last_end, prev_end)

                proc_time = op['proc_times'][m] # Use the processing time for the assigned machine
                end_time = start_time + proc_time

                sch[m].append((f"J{j}-O{oi+1}", start_time, end_time))
                prev_end = end_time
            completion[j] = prev_end

        mk = max(completion.values()) if completion else 0.0 # Handle case with no jobs/operations
        if mk < best_makespan:
            best_makespan = mk
            # Deep copy the schedule to prevent modification in subsequent iterations
            best_schedule = {m: list(ops) for m, ops in sch.items()}

    return best_schedule, best_makespan

# --- MILP Method (Modified for robust Big M constraints) ---
def milp_scheduler(jobs, machines):
    """Mixed Integer Linear Programming approach"""
    job_ids = list(jobs.keys())
    prob = LpProblem("FlexibleJSP", LpMinimize)

    # Operations list
    ops = []
    for j in job_ids:
        for oi in range(len(jobs[j])):
            ops.append((j, oi))

    # If no operations, return empty schedule and 0 makespan
    if not ops:
        return {m: [] for m in machines}, 0.0

    # Variables
    x = {(j, oi, m): LpVariable(f"x_{j}_{oi}_{m}", cat="Binary")
         for (j, oi) in ops for m in machines if m in jobs[j][oi]['proc_times']} # Only create x if machine is valid
    s = {(j, oi): LpVariable(f"s_{j}_{oi}", lowBound=0) for (j, oi) in ops}
    c = {(j, oi): LpVariable(f"c_{j}_{oi}", lowBound=0) for (j, oi) in ops}
    Cmax = LpVariable("Cmax", lowBound=0)

    # Objective
    prob += Cmax

    # Constraints
    # 1. Each operation assigned to exactly one machine AND calculate completion time
    for (j, oi) in ops:
        # Sum only over machines capable of processing this operation
        valid_machines_for_op = [m for m in machines if m in jobs[j][oi]['proc_times']]
        if not valid_machines_for_op:
            raise ValueError(f"Operation J{j}-O{oi+1} has no valid machine assignments. Cannot form MILP.")
        prob += lpSum(x[(j, oi, m)] for m in valid_machines_for_op) == 1
        prob += c[(j, oi)] == s[(j, oi)] + lpSum(jobs[j][oi]['proc_times'][m] * x[(j, oi, m)] for m in valid_machines_for_op)

    # 2. Precedence constraints
    for j in job_ids:
        for oi in range(len(jobs[j])-1):
            prob += s[(j, oi+1)] >= c[(j, oi)]

    # 3. Machine capacity constraints (no overlap) - MODIFIED ROBUST BIG M
    BIG = 1000000 # A sufficiently large number, must be larger than any possible makespan

    # Iterate over all machines
    for m in machines:
        # Get all operations that *can* be processed on machine 'm'
        ops_on_machine_m = [(j, oi) for (j, oi) in ops if m in jobs[j][oi]['proc_times']]

        # For every unique pair of operations that can be on this machine
        for i in range(len(ops_on_machine_m)):
            for k in range(i + 1, len(ops_on_machine_m)):
                op1 = ops_on_machine_m[i]
                op2 = ops_on_machine_m[k]

                # Introduce a binary variable 'y_op1_op2_m'
                # y_op1_op2_m = 1 if op1 precedes op2 on machine m
                # y_op1_op2_m = 0 if op2 precedes op1 on machine m
                y_var = LpVariable(f"y_{op1[0]}{op1[1]}_{op2[0]}{op2[1]}_{m}", cat="Binary")

                # Disjunctive constraints for processing on machine 'm':
                # These constraints ensure that if both op1 and op2 are assigned to machine m,
                # then one must finish before the other starts.
                # If x[op1,m] and x[op2,m] are 1:
                # Constraint 1: If op1 precedes op2 (y_var = 1), then c_op1 <= s_op2
                #   s[op2] >= c[op1] - BIG * (1 - y_var)
                # Constraint 2: If op2 precedes op1 (y_var = 0), then c_op2 <= s_op1
                #   s[op1] >= c[op2] - BIG * y_var

                # The challenge is to activate these disjunctive constraints ONLY IF both operations are on machine m.
                # A common way to do this is to add (1 - x_op1_m) * BIG and (1 - x_op2_m) * BIG to the right side,
                # which effectively relaxes the constraint if either operation is NOT on machine m.

                # s_op2 >= c_op1 - BIG * (1 - y_var) - BIG * (1 - x[op1[0], op1[1], m]) - BIG * (1 - x[op2[0], op2[1], m])
                prob += s[op2] >= c[op1] - BIG * (1 - y_var) - BIG * (1 - x[op1[0], op1[1], m]) - BIG * (1 - x[op2[0], op2[1], m])
                prob += s[op1] >= c[op2] - BIG * y_var - BIG * (1 - x[op1[0], op1[1], m]) - BIG * (1 - x[op2[0], op2[1], m])


    # 4. Makespan constraints
    for (j, oi) in ops:
        prob += Cmax >= c[(j, oi)]

    # Solve
    print("   Solving MILP optimization...")
    prob.solve(PULP_CBC_CMD(msg=False))

    # Extract schedule
    schedule = {m: [] for m in machines}
    for (j, oi) in ops:
        for m in machines:
            if m in jobs[j][oi]['proc_times'] and x[(j, oi, m)].varValue is not None and x[(j, oi, m)].varValue > 0.5:
                start = round(s[(j, oi)].varValue)
                end = round(c[(j, oi)].varValue)
                schedule[m].append((f"J{j}-O{oi+1}", start, end))

    # Sort by start time
    for m in schedule:
        schedule[m].sort(key=lambda trip: trip[1])

    # Get makespan from solver, handle potential None or unbound Cmax
    makespan_val = round(Cmax.varValue) if Cmax.varValue is not None else float('inf')

    return schedule, makespan_val

# RL Training and Execution
def train_and_test_rl(jobs_data, machine_list):
    """Train RL agent and test it"""
    print("Training RL agent...")

    def make_env():
        return FJSPEnv(jobs_data=jobs_data, machine_list=machine_list, reward_mode="optimized")

    # Create environments
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env]) # Used for evaluation, but not explicitly used in this snippet's callback

    # Create PPO model with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0, # Set to 0 for less output during training
        learning_rate=1e-3,
        n_steps=1024,
        batch_size=32,
        n_epochs=20,
        gamma=0.99,
        ent_coef=0.05,
        clip_range=0.1,
        policy_kwargs=dict(net_arch=[256, 256, 128])
    )

    # Train the model
    # Consider adjusting total_timesteps based on problem complexity
    # For small problems, 50000 might be enough. For larger, need more.
    total_timesteps = 50000
    progress_callback = ProgressBarCallback(total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=progress_callback)

    # Test the trained model
    test_env = FJSPEnv(jobs_data=jobs_data, machine_list=machine_list, reward_mode="optimized")
    
    best_makespan = float('inf')
    best_schedule = None

    # Run multiple episodes to get best result (RL is stochastic)
    num_episodes_test = 10
    print(f"Testing RL agent for {num_episodes_test} episodes...")
    
    # Add progress bar for testing
    for episode in tqdm(range(num_episodes_test), desc="Testing Episodes"):
        obs, _ = test_env.reset()  # Updated for gymnasium format
        episode_makespan = 0.0
        
        while True:
            action, _ = model.predict(obs, deterministic=True) # Use deterministic for evaluation
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_makespan = info["makespan"] # Get makespan from info
            done = terminated or truncated

            if done:
                # print(f"  Episode {episode+1} Makespan: {episode_makespan:.2f}")
                if episode_makespan < best_makespan:
                    best_makespan = episode_makespan
                    # Ensure deep copy of the schedule
                    best_schedule = {m: list(ops) for m, ops in test_env.schedule.items()}
                break
    
    print(f"RL Best makespan after {num_episodes_test} episodes: {best_makespan:.2f}")
    return best_schedule, best_makespan

# Main comparison function
def compare_methods():
    """Compare all three methods"""
    print("=== FJSP Method Comparison ===\n")

    # --- Use the original small problem for initial testing ---
    print("Using Original Small Problem Instance:")
    current_jobs = original_jobs
    current_machines = original_machines
    print("Jobs:", current_jobs)
    print("Machines:", current_machines)
    print()

    # Method 1: Exact Search
    print("1. Running Exact Search...")
    exact_schedule, exact_makespan = exact_search(current_jobs, current_machines)
    print(f"   Makespan: {exact_makespan:.2f}")

    # Method 2: MILP
    print("2. Running MILP...")
    milp_schedule, milp_makespan = milp_scheduler(current_jobs, current_machines)
    print(f"   Makespan: {milp_makespan:.2f}")

    # Method 3: RL
    print("3. Training and Testing RL...")
    rl_schedule, rl_makespan = train_and_test_rl(current_jobs, current_machines)
    print(f"   Makespan: {rl_makespan:.2f}")

    # Results comparison for original problem
    print("\n=== RESULTS COMPARISON (Original Problem) ===")
    print(f"Exact Search Makespan: {exact_makespan:.2f}")
    print(f"MILP Makespan:         {milp_makespan:.2f}")
    print(f"RL Makespan:           {rl_makespan:.2f}")
    if exact_makespan > 0:
        print(f"RL Gap from Optimal:   {((rl_makespan - exact_makespan) / exact_makespan * 100):.1f}%")
    else:
        print("Optimal makespan is 0, cannot calculate percentage gap.")

    # Plot Gantt charts for original problem
    plot_gantt(exact_schedule, current_machines, f"Original: Exact Search (Makespan: {exact_makespan:.2f})")
    plot_gantt(milp_schedule, current_machines, f"Original: MILP (Makespan: {milp_makespan:.2f})")
    plot_gantt(rl_schedule, current_machines, f"Original: RL - PPO (Makespan: {rl_makespan:.2f})")

    print("\n" + "="*50 + "\n")

    # --- Demonstrate with a larger, Taillard-like problem ---
    print("Demonstrating with a Taillard-like Problem Instance:")
    # This is a made-up example in Taillard-like format:
    # First line: num_jobs num_machines
    # Subsequent lines: num_operations_in_job (num_machine_options machine_id proc_time) ...
    taillard_sample_data = """
    3 4
    2 2 1 3 2 4 3 3 1
    3 1 3 4 5 2 2 4 1 3 2
    2 3 2 1 4 2 2 3 5
    """
    
    # Remove leading/trailing whitespace and empty lines
    taillard_sample_data = "\n".join([line.strip() for line in taillard_sample_data.strip().split('\n') if line.strip()])

    try:
        taillard_jobs, taillard_machines = parse_taillard_data(taillard_sample_data)
        print("Parsed Jobs:", taillard_jobs)
        print("Parsed Machines:", taillard_machines)
        print()

        # Exact search might be too slow for larger problems.
        # It's kept here for theoretical understanding, but for real Taillard, it won't finish.
        print("1. Running Exact Search (Taillard-like - may be very slow or unfeasible for larger instances)...")
        # exact_taillard_schedule, exact_taillard_makespan = exact_search(taillard_jobs, taillard_machines)
        # print(f"   Makespan: {exact_taillard_makespan:.2f}")
        print("   Skipping exact search for larger Taillard-like instance due to computational complexity.")
        exact_taillard_schedule, exact_taillard_makespan = {}, float('inf') # Placeholder

        print("2. Running MILP (Taillard-like - might be slow)...")
        milp_taillard_schedule, milp_taillard_makespan = milp_scheduler(taillard_jobs, taillard_machines)
        print(f"   Makespan: {milp_taillard_makespan:.2f}")

        print("3. Training and Testing RL (Taillard-like)...")
        rl_taillard_schedule, rl_taillard_makespan = train_and_test_rl(taillard_jobs, taillard_machines)
        print(f"   Makespan: {rl_taillard_makespan:.2f}")

        # Results comparison for Taillard-like problem
        print("\n=== RESULTS COMPARISON (Taillard-like Problem) ===")
        print(f"MILP Makespan:         {milp_taillard_makespan:.2f}")
        print(f"RL Makespan:           {rl_taillard_makespan:.2f}")
        # Cannot calculate gap from optimal without exact search result
        if milp_taillard_makespan > 0:
            print(f"RL Gap from MILP:      {((rl_taillard_makespan - milp_taillard_makespan) / milp_taillard_makespan * 100):.1f}%")

        # Plot Gantt charts for Taillard-like problem
        plot_gantt(milp_taillard_schedule, taillard_machines, f"Taillard-like: MILP (Makespan: {milp_taillard_makespan:.2f})")
        plot_gantt(rl_taillard_schedule, taillard_machines, f"Taillard-like: RL - PPO (Makespan: {rl_taillard_makespan:.2f})")

    except Exception as e:
        print(f"\nError processing Taillard-like data: {e}")
        print("Skipping Taillard-like problem due to parsing or processing issue.")


    return {
        'original': {
            'exact': (exact_schedule, exact_makespan),
            'milp': (milp_schedule, milp_makespan),
            'rl': (rl_schedule, rl_makespan)
        },
        'taillard_like': {
            # 'exact': (exact_taillard_schedule, exact_taillard_makespan), # exact often too slow
            'milp': (milp_taillard_schedule, milp_taillard_makespan),
            'rl': (rl_taillard_schedule, rl_taillard_makespan)
        }
    }

if __name__ == "__main__":
    results = compare_methods()