import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces

def plot_gantt(schedule, machines, title="Schedule"):
    """
    Plot Gantt chart for the schedule
    """
    if not schedule or all(len(ops) == 0 for ops in schedule.values()):
        print("No schedule to plot - schedule is empty")
        return
        
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(12, len(machines)*0.8))
    
    for idx, m in enumerate(machines):
        machine_ops = schedule.get(m, [])
        for (j, start, end) in machine_ops:
            ax.broken_barh(
                [(start, end - start)],
                (idx * 10, 8),
                facecolors=colors[j % len(colors)],
                edgecolor='black',
                alpha=0.8
            )
            # Add job label
            ax.text(start + (end-start)/2, idx * 10 + 4, 
                   f"J{j}", color='white', fontsize=10, 
                   ha='center', va='center', weight='bold')
    
    ax.set_yticks([i * 10 + 4 for i in range(len(machines))])
    ax.set_yticklabels(machines)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


class FJSPEnv(gym.Env):
    """
    Optimized FJSP Environment with better reward shaping
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data=None, machines=None, reward_mode="optimized"):
        super().__init__()
        
        # Default problem if none provided
        if jobs_data is None:
            self.jobs = {
                1: [
                    {'proc_times': {'M1': 2, 'M2': 4, 'M3': 3}},
                    {'proc_times': {'M1': 3, 'M2': 2, 'M3': 4}}
                ],
                2: [
                    {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2}},
                    {'proc_times': {'M1': 2, 'M2': 3, 'M3': 4}}
                ]
            }
        else:
            self.jobs = jobs_data
            
        self.machines = machines if machines else ["M1", "M2", "M3"]
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values())
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        
        self.reward_mode = reward_mode
        
        # Calculate optimal lower bound for makespan (critical path)
        self.optimal_makespan_lb = self._calculate_optimal_lower_bound()
        
        # Action space: (job_idx, operation_idx, machine_idx)
        self.action_space = spaces.Discrete(
            self.num_jobs * self.max_ops_per_job * len(self.machines)
        )
        
        # Enhanced observation space
        obs_size = (
            len(self.machines) +  # machine next free times (normalized)
            self.num_jobs * self.max_ops_per_job +  # operation completion status
            self.num_jobs +  # next operation index for each job
            len(self.machines) +  # machine utilization
            1  # current makespan normalized
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        self.reset()

    def _calculate_optimal_lower_bound(self):
        """Calculate a lower bound on optimal makespan"""
        # Sum of minimum processing times for each operation
        total_min_time = 0
        for job_ops in self.jobs.values():
            for op in job_ops:
                min_time = min(op['proc_times'].values())
                total_min_time += min_time
        
        # Divide by number of machines for theoretical minimum
        return total_min_time / len(self.machines)

    def reset(self):
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        
        # Track operation completion
        self.completed_ops = {}
        self.operation_end_times = {}
        for job_id in self.job_ids:
            self.completed_ops[job_id] = [False] * len(self.jobs[job_id])
            self.operation_end_times[job_id] = [0.0] * len(self.jobs[job_id])
        
        # Track job progress
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        self.current_makespan = 0.0
        self.operations_scheduled = 0
        self.step_count = 0
        
        return self._get_observation()

    def _decode_action(self, action):
        """Decode action into job, operation, machine indices"""
        job_idx = action // (self.max_ops_per_job * len(self.machines))
        remaining = action % (self.max_ops_per_job * len(self.machines))
        op_idx = remaining // len(self.machines)
        machine_idx = remaining % len(self.machines)
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        """Check if action is valid"""
        if (job_idx >= self.num_jobs or 
            machine_idx >= len(self.machines)):
            return False
            
        job_id = self.job_ids[job_idx]
        
        # Check if operation exists
        if op_idx >= len(self.jobs[job_id]):
            return False
            
        # Check if operation is already completed
        if self.completed_ops[job_id][op_idx]:
            return False
            
        # Check if this is the next operation for the job (precedence)
        if op_idx != self.next_operation[job_id]:
            return False
            
        return True

    def step(self, action):
        self.step_count += 1
        job_idx, op_idx, machine_idx = self._decode_action(action)
        
        # Check validity
        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            # Heavy penalty for invalid actions
            return self._get_observation(), -20.0, False, {"invalid_action": True}
        
        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        
        # Calculate start time
        machine_available = self.machine_next_free[machine]
        
        if op_idx == 0:
            job_ready = 0.0
        else:
            job_ready = self.operation_end_times[job_id][op_idx - 1]
        
        start_time = max(machine_available, job_ready)
        
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
        self.schedule[machine].append((job_id, start_time, end_time))
        
        # Calculate reward
        done = self.operations_scheduled == self.total_operations
        reward = self._calculate_reward(proc_time, start_time - machine_available, done)
        
        info = {
            "makespan": self.current_makespan,
            "operations_scheduled": self.operations_scheduled,
            "efficiency": self.optimal_makespan_lb / max(self.current_makespan, 0.1)
        }
        
        return self._get_observation(), reward, done, info

    def _calculate_reward(self, proc_time, idle_time, done):
        """Calculate reward with better shaping"""
        if self.reward_mode == "optimized":
            # Progress reward
            progress_reward = 5.0
            
            # Efficiency reward (prefer shorter processing times)
            min_proc_time = min(
                self.jobs[self.job_ids[0]][0]['proc_times'].values()
            )
            max_proc_time = max(
                max(op['proc_times'].values()) 
                for job_ops in self.jobs.values() 
                for op in job_ops
            )
            efficiency_reward = 2.0 * (max_proc_time - proc_time) / (max_proc_time - min_proc_time)
            
            # Idle time penalty
            idle_penalty = -1.0 * idle_time
            
            # Terminal reward based on makespan quality
            if done:
                makespan_ratio = self.optimal_makespan_lb / self.current_makespan
                terminal_reward = 20.0 * makespan_ratio
                return progress_reward + efficiency_reward + idle_penalty + terminal_reward
            else:
                return progress_reward + efficiency_reward + idle_penalty
                
        elif self.reward_mode == "sparse":
            if done:
                return -self.current_makespan
            return 0.0
            
        return 0.0

    def _get_observation(self):
        """Build observation vector"""
        obs = []
        
        # Machine next free times (normalized by current makespan + 1)
        norm_factor = max(self.current_makespan, 1.0)
        for machine in self.machines:
            obs.append(self.machine_next_free[machine] / norm_factor)
        
        # Operation completion status
        for job_id in self.job_ids:
            for op_idx in range(self.max_ops_per_job):
                if op_idx < len(self.jobs[job_id]):
                    obs.append(1.0 if self.completed_ops[job_id][op_idx] else 0.0)
                else:
                    obs.append(0.0)
        
        # Next operation index for each job (normalized)
        for job_id in self.job_ids:
            next_op = self.next_operation[job_id]
            total_ops = len(self.jobs[job_id])
            obs.append(next_op / total_ops)
        
        # Machine utilization
        total_time = max(self.current_makespan, 1.0)
        for machine in self.machines:
            utilization = self.machine_next_free[machine] / total_time
            obs.append(min(utilization, 1.0))
        
        # Current makespan (normalized by theoretical optimum)
        obs.append(min(self.current_makespan / (self.optimal_makespan_lb * 3), 1.0))
        
        return np.array(obs, dtype=np.float32)

    def render(self, mode="human"):
        """Render current state"""
        print(f"Step: {self.step_count}, Makespan: {self.current_makespan:.2f}")
        print(f"Operations: {self.operations_scheduled}/{self.total_operations}")
        
        for job_id in self.job_ids:
            next_op = self.next_operation[job_id]
            total_ops = len(self.jobs[job_id])
            print(f"Job {job_id}: {next_op}/{total_ops} operations completed")

    def get_schedule_for_comparison(self):
        """Get schedule in format compatible with comparison functions"""
        formatted_schedule = {m: [] for m in self.machines}
        
        for machine in self.machines:
            for job_id, start, end in self.schedule[machine]:
                # Find which operation this is for the job
                op_count = 0
                for other_machine in self.machines:
                    for other_job, other_start, other_end in self.schedule[other_machine]:
                        if other_job == job_id and other_end <= end:
                            if other_machine != machine or other_start != start:
                                op_count += 1
                
                formatted_schedule[machine].append((f"J{job_id}-O{op_count+1}", start, end))
        
        return formatted_schedule