import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import time

# =============================================================================
# ENHANCED FJSP POLICY ITERATION WITH TIMING STATE INFORMATION
# =============================================================================
# Simple problem from dynamic-tutorial.py with enhanced state representation

SIMPLE_JOBS_DATA = collections.OrderedDict({
    1: [
        {'proc_times': {'M1': 2, 'M2': 4, 'M3': 3}},  # J1-O1: can be processed on M1(2), M2(4), or M3(3)
        {'proc_times': {'M1': 3, 'M2': 2, 'M3': 4}}   # J1-O2: can be processed on M1(3), M2(2), or M3(4)
    ],
    2: [
        {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2}},  # J2-O1: can be processed on M1(4), M2(3), or M3(2)
        {'proc_times': {'M1': 2, 'M2': 3, 'M3': 4}}   # J2-O2: can be processed on M1(2), M2(3), or M3(4)
    ]
})
SIMPLE_MACHINES = ['M1', 'M2', 'M3']
SIMPLE_ARRIVALS = {1: 0, 2: 0}  # Both jobs arrive at t=0

print("=== ENHANCED MDP FORMULATION FOR SIMPLE FJSP PROBLEM ===")
print("Problem Instance (from dynamic-tutorial.py):")
print("Jobs and Operations:")
for job_id, operations in SIMPLE_JOBS_DATA.items():
    print(f"  Job {job_id}:")
    for op_idx, op in enumerate(operations):
        proc_times_str = ", ".join([f"{m}({t})" for m, t in op['proc_times'].items()])
        print(f"    Operation {op_idx+1}: {proc_times_str}")
print(f"Machines: {SIMPLE_MACHINES}")
print(f"Job arrivals: {SIMPLE_ARRIVALS}")
print()

print("ENHANCED MDP Components with Timing Information:")
print("1. STATE SPACE S:")
print("   Enhanced State = (J1_ops_done, J2_ops_done, M1_free_time, M2_free_time, M3_free_time, current_makespan)")
print("   - Operation counts: how many operations completed for each job")
print("   - Machine free times: when each machine becomes available")
print("   - Current makespan: maximum completion time so far")
print("   This gives more precise state representation for timing-dependent decisions")
print()

print("2. ACTION SPACE A(s):")
print("   Action = (job_id, operation_index, machine)")
print("   Valid actions depend on:")
print("   - Job precedence constraints (operations must be done in order)")
print("   - Machine capabilities (operation can only be done on compatible machines)")
print("   - Job arrival times (job must have arrived)")
print()

print("3. TRANSITION FUNCTION P(s'|s,a):")
print("   Enhanced deterministic transitions:")
print("   - Update operation count for the job")
print("   - Update machine free time = max(machine_free_time, job_ready_time) + processing_time")
print("   - Update current makespan = max(current_makespan, operation_end_time)")
print()

print("4. REWARD FUNCTION R(s,a):")
print("   Using MAKESPAN INCREMENT reward:")
print("   R(s,a) = Current_Makespan - New_Makespan")
print("   This encourages actions that minimize makespan growth")
print("   Negative values indicate makespan increased")
print()

print("5. POLICY π(s):")
print("   Maps each enhanced state to an action")
print("   Goal: Find optimal policy π* that minimizes expected cumulative reward")
print()

print("Let's trace through the enhanced MDP step by step...")
print("="*80)
print()

class EnhancedFJSPEnv:
    """
    Enhanced FJSP Environment for Policy Iteration with timing information in state
    Focuses on the simple 2-job, 3-machine example from dynamic-tutorial.py
    """
    def __init__(self, jobs_data, machine_list, reward_mode="makespan_increment", job_arrival_times=None):
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        self.num_machines = len(self.machines)
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        self.reward_mode = reward_mode
        
        # Set job arrival times - default to all jobs arriving at t=0
        if job_arrival_times is None:
            self.job_arrival_times = {job_id: 0 for job_id in self.job_ids}
        else:
            self.job_arrival_times = job_arrival_times
        
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        # Enhanced State: (J1_ops_done, J2_ops_done, M1_free_time, M2_free_time, M3_free_time, J1_ready_time, J2_ready_time, current_makespan)
        initial_ops = tuple(0 for _ in self.job_ids)
        initial_machine_times = tuple(0.0 for _ in self.machines)
        initial_job_ready_times = tuple(float(self.job_arrival_times[job_id]) for job_id in self.job_ids)
        initial_makespan = 0.0
        
        self.state = initial_ops + initial_machine_times + initial_job_ready_times + (initial_makespan,)
        
        # Internal tracking for validation
        self.machine_free_times = {m: 0.0 for m in self.machines}
        self.job_ready_times = {j: self.job_arrival_times[j] for j in self.job_ids}
        self.makespan = 0
        self.schedule_log = collections.defaultdict(list)
        return self.state

    def get_state_info(self, state):
        """Extract components from enhanced state for readability"""
        completed_ops = state[:self.num_jobs]
        machine_free_times = state[self.num_jobs:self.num_jobs+self.num_machines]
        job_ready_times = state[self.num_jobs+self.num_machines:self.num_jobs+self.num_machines+self.num_jobs]
        current_makespan = state[-1]
        
        return {
            'operations_completed': dict(zip(self.job_ids, completed_ops)),
            'machine_free_times': dict(zip(self.machines, machine_free_times)),
            'job_ready_times': dict(zip(self.job_ids, job_ready_times)),
            'current_makespan': current_makespan
        }

    def get_valid_actions(self, state, verbose=False):
        """
        Get valid actions for the current state with timing information
        Returns list of (action, processing_time) tuples
        """
        state_info = self.get_state_info(state)
        completed_ops = state[:self.num_jobs]
        machine_free_times = state[self.num_jobs:self.num_jobs+self.num_machines]
        job_ready_times = state[self.num_jobs+self.num_machines:self.num_jobs+self.num_machines+self.num_jobs]
        current_makespan = state[-1]
        
        valid_actions = []
        
        if verbose:
            print(f"    Getting valid actions for state {state}:")
            print(f"      Operations completed: {state_info['operations_completed']}")
            print(f"      Machine free times: {state_info['machine_free_times']}")
            print(f"      Job ready times: {state_info['job_ready_times']}")
            print(f"      Current makespan: {current_makespan}")
        
        for job_idx, job_id in enumerate(self.job_ids):
            # Check if job has remaining operations
            if completed_ops[job_idx] < len(self.jobs[job_id]):
                
                operation = self.jobs[job_id][completed_ops[job_idx]]
                
                for machine, proc_time in operation['proc_times'].items():
                    action = (job_id, completed_ops[job_idx], machine)
                    valid_actions.append((action, proc_time))
                    
                    if verbose:
                        machine_idx = self.machines.index(machine)
                        machine_free_time = machine_free_times[machine_idx]
                        job_ready_time = job_ready_times[job_idx]
                        
                        start_time = max(machine_free_time, job_ready_time)
                        end_time = start_time + proc_time
                        
                        print(f"      Action: J{job_id}-O{completed_ops[job_idx]+1} on {machine}")
                        print(f"        Processing time: {proc_time}")
                        print(f"        Would start at: {start_time}, end at: {end_time}")
        
        if verbose:
            print(f"    Total valid actions: {len(valid_actions)}")
        return valid_actions

    def transition(self, state, action, verbose=False):
        """
        Execute action from given state and return new state and reward
        """
        job_id, op_idx, machine = action
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        
        # Extract current state components
        completed_ops = list(state[:self.num_jobs])
        machine_free_times = list(state[self.num_jobs:self.num_jobs+self.num_machines])
        job_ready_times = list(state[self.num_jobs+self.num_machines:self.num_jobs+self.num_machines+self.num_jobs])
        current_makespan = state[-1]
        
        if verbose:
            print(f"    Executing action: J{job_id}-O{op_idx+1} on {machine}")
            print(f"      Processing time: {proc_time}")
            print(f"      Current state: {state}")
            print(f"      Current makespan: {current_makespan}")
        
        # Calculate timing
        job_idx = self.job_ids.index(job_id)
        machine_idx = self.machines.index(machine)
        
        # Job ready time is already tracked in the state
        job_ready_time = job_ready_times[job_idx]
        machine_free_time = machine_free_times[machine_idx]
        
        # Start time is when both machine and job are ready
        start_time = max(machine_free_time, job_ready_time)
        end_time = start_time + proc_time
        
        if verbose:
            print(f"      Machine {machine} free at: {machine_free_time}")
            print(f"      Job {job_id} ready at: {job_ready_time}")
            print(f"      Operation starts at: {start_time}, ends at: {end_time}")
        
        # Update state components
        completed_ops[job_idx] += 1
        machine_free_times[machine_idx] = end_time
        job_ready_times[job_idx] = end_time  # Job becomes ready for next operation when this one completes
        new_makespan = max(current_makespan, end_time)
        
        # Create new state
        new_state = tuple(completed_ops) + tuple(machine_free_times) + tuple(job_ready_times) + (new_makespan,)
        
        if verbose:
            print(f"      New state: {new_state}")
            print(f"      New makespan: {new_makespan}")
        
        # Calculate reward based on mode
        if self.reward_mode == "makespan_increment":
            # R(s,a) = current_makespan - new_makespan 
            # Negative values indicate makespan increased (bad)
            reward = current_makespan - new_makespan
            if verbose:
                print(f"      Reward = {current_makespan} - {new_makespan} = {reward}")
        else:  # processing_time mode
            reward = -proc_time
            if verbose:
                print(f"      Reward = -{proc_time} = {reward}")
        
        return new_state, reward

    def is_done(self, state):
        """Check if all operations are completed"""
        completed_ops = state[:self.num_jobs]
        terminal_ops = tuple(len(self.jobs[job_id]) for job_id in self.job_ids)
        return completed_ops == terminal_ops

    def get_all_reachable_states(self):
        """Generate all reachable states for policy iteration"""
        # This is a simplified version - in practice, this could be very large
        # For the simple 2-job, 3-machine problem, we'll enumerate key states
        
        states = set()
        states.add(self.reset())  # Initial state
        
        # Use BFS to find all reachable states
        queue = [self.reset()]
        visited = {self.reset()}
        
        while queue:
            current_state = queue.pop(0)
            
            if self.is_done(current_state):
                continue
                
            valid_actions = self.get_valid_actions(current_state)
            
            for action, _ in valid_actions:
                next_state, _ = self.transition(current_state, action)
                
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(next_state)
                    states.add(next_state)
        
        return list(states)

def enhanced_policy_iteration(env, gamma=0.9, theta=1e-6, max_iterations=100):
    """
    Enhanced Policy Iteration Algorithm with detailed step-by-step output
    """
    print("="*80)
    print("ENHANCED POLICY ITERATION ALGORITHM")
    print("="*80)
    
    # Get all reachable states
    print("Step 1: Generating all reachable states...")
    all_states = env.get_all_reachable_states()
    print(f"Found {len(all_states)} reachable states:")
    for i, state in enumerate(all_states):
        state_info = env.get_state_info(state)
        print(f"  State {i}: {state}")
        print(f"    Ops completed: {state_info['operations_completed']}")
        print(f"    Machine free times: {state_info['machine_free_times']}")
        print(f"    Makespan: {state_info['current_makespan']}")
        if env.is_done(state):
            print("    (Terminal state)")
    print()
    
    # Find terminal state
    terminal_states = [s for s in all_states if env.is_done(s)]
    print(f"Terminal states: {terminal_states}")
    print()
    
    # Initialize policy and value function
    policy = {}
    V = {s: 0.0 for s in all_states}
    
    print("Step 2: Initializing random policy...")
    for state in all_states:
        if env.is_done(state):
            policy[state] = None  # No action in terminal state
        else:
            valid_actions = env.get_valid_actions(state)
            if valid_actions:
                policy[state] = valid_actions[0][0]  # Choose first valid action
            else:
                policy[state] = None
    
    print("Initial policy:")
    for state in all_states:
        if policy[state] is not None:
            action = policy[state]
            state_info = env.get_state_info(state)
            print(f"  State {state}: J{action[0]}-O{action[1]+1} on {action[2]}")
            print(f"    (Ops: {state_info['operations_completed']}, Makespan: {state_info['current_makespan']})")
        else:
            print(f"  State {state}: Terminal (no action)")
    print()
    
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"="*60)
        print(f"POLICY ITERATION - ITERATION {iteration}")
        print(f"="*60)
        
        # Policy Evaluation
        print(f"POLICY EVALUATION (Iteration {iteration}):")
        print("Solving Bellman equations: V^π(s) = R(s,π(s)) + γ * V^π(s')")
        print()
        
        eval_iter = 0
        while True:
            eval_iter += 1
            delta = 0
            print(f"  Policy Evaluation Sub-iteration {eval_iter}:")
            
            for state in all_states:
                if env.is_done(state):
                    continue  # Terminal state value remains 0
                
                v_old = V[state]
                action = policy[state]
                
                if action is None:
                    continue
                
                # Calculate expected value
                next_state, reward = env.transition(state, action)
                V[state] = reward + gamma * V[next_state]
                delta = max(delta, abs(v_old - V[state]))
                
                state_info = env.get_state_info(state)
                next_state_info = env.get_state_info(next_state)
                
                print(f"    State {state}:")
                print(f"      Current ops: {state_info['operations_completed']}, makespan: {state_info['current_makespan']}")
                print(f"      Action: J{action[0]}-O{action[1]+1} on {action[2]}")
                print(f"      Next ops: {next_state_info['operations_completed']}, makespan: {next_state_info['current_makespan']}")
                print(f"      V(s): {v_old:.3f} → {V[state]:.3f}")
                print(f"      Calculation: V(s) = R + γ*V(s') = {reward:.3f} + {gamma}*{V[next_state]:.3f} = {V[state]:.3f}")
            
            print(f"    Max delta = {delta:.6f} (threshold: {theta})")
            if delta < theta:
                print(f"  Policy evaluation converged in {eval_iter} iterations!")
                break
            print()
        
        print()
        
        # Policy Improvement
        print(f"POLICY IMPROVEMENT (Iteration {iteration}):")
        print("For each state, finding action that maximizes Q(s,a) = R(s,a) + γ * V(s')")
        print()
        
        policy_stable = True
        
        for state in all_states:
            if env.is_done(state):
                continue  # Terminal state
            
            old_action = policy[state]
            best_q = float('-inf')
            best_action = None
            
            valid_actions = env.get_valid_actions(state)
            
            state_info = env.get_state_info(state)
            print(f"  State {state}:")
            print(f"    Current ops: {state_info['operations_completed']}, makespan: {state_info['current_makespan']}")
            print(f"    Evaluating actions:")
            
            for action, _ in valid_actions:
                next_state, reward = env.transition(state, action)
                q_value = reward + gamma * V[next_state]
                
                next_state_info = env.get_state_info(next_state)
                print(f"      J{action[0]}-O{action[1]+1} on {action[2]}: Q = {reward:.3f} + {gamma}*{V[next_state]:.3f} = {q_value:.3f}")
                print(f"        → Next: ops={next_state_info['operations_completed']}, makespan={next_state_info['current_makespan']}")
                
                if q_value > best_q:
                    best_q = q_value
                    best_action = action
            
            policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
                print(f"    Policy changed: J{old_action[0]}-O{old_action[1]+1} on {old_action[2]} → J{best_action[0]}-O{best_action[1]+1} on {best_action[2]}")
            else:
                print(f"    Policy unchanged: J{best_action[0]}-O{best_action[1]+1} on {best_action[2]}")
            print()
        
        if policy_stable:
            print(f"🎉 POLICY CONVERGED in {iteration} iterations!")
            break
        
        print(f"Policy updated, continuing to iteration {iteration + 1}...")
        print()
    
    print(f"="*60)
    print(f"FINAL OPTIMAL POLICY AND VALUES:")
    print(f"="*60)
    for state in all_states:
        if policy[state] is not None:
            action = policy[state]
            state_info = env.get_state_info(state)
            print(f"State {state}: J{action[0]}-O{action[1]+1} on {action[2]} (V = {V[state]:.3f})")
            print(f"  Ops: {state_info['operations_completed']}, Makespan: {state_info['current_makespan']}")
        else:
            print(f"State {state}: Terminal (V = {V[state]:.3f})")
    print()
    
    return policy, V

def execute_optimal_policy(env, policy):
    """Execute the optimal policy and show detailed trace"""
    print("="*80)
    print("EXECUTING OPTIMAL POLICY")
    print("="*80)
    
    env.reset()
    step = 0
    total_reward = 0
    
    print(f"Initial state: {env.state}")
    state_info = env.get_state_info(env.state)
    print(f"Initial state info: {state_info}")
    print()
    
    while not env.is_done(env.state):
        step += 1
        current_state = env.state
        action = policy[current_state]
        
        print(f"--- Step {step} ---")
        current_info = env.get_state_info(current_state)
        print(f"Current state: {current_state}")
        print(f"  Operations completed: {current_info['operations_completed']}")
        print(f"  Machine free times: {current_info['machine_free_times']}")
        print(f"  Current makespan: {current_info['current_makespan']}")
        print(f"Optimal action: J{action[0]}-O{action[1]+1} on {action[2]}")
        
        # Execute action
        next_state, reward = env.transition(current_state, action, verbose=True)
        env.state = next_state  # Update environment state
        total_reward += reward
        
        next_info = env.get_state_info(next_state)
        print(f"Next state: {next_state}")
        print(f"  Operations completed: {next_info['operations_completed']}")
        print(f"  Machine free times: {next_info['machine_free_times']}")
        print(f"  New makespan: {next_info['current_makespan']}")
        print(f"Step reward: {reward:.3f}")
        print(f"Cumulative reward: {total_reward:.3f}")
        print()
    
    final_info = env.get_state_info(env.state)
    print(f"🏁 FINAL RESULTS:")
    print(f"Total steps: {step}")
    print(f"Final makespan: {final_info['current_makespan']}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Final state: {env.state}")
    
    return final_info['current_makespan'], total_reward

def main():
    """Main function to run the enhanced policy iteration demonstration"""
    print("Starting Enhanced Policy Iteration for Simple FJSP Problem...")
    print()
    
    # Create environment
    env = EnhancedFJSPEnv(
        jobs_data=SIMPLE_JOBS_DATA,
        machine_list=SIMPLE_MACHINES,
        reward_mode="makespan_increment",
        job_arrival_times=SIMPLE_ARRIVALS
    )
    
    # Run policy iteration
    optimal_policy, optimal_values = enhanced_policy_iteration(env)
    
    # Execute optimal policy
    makespan, total_reward = execute_optimal_policy(env, optimal_policy)
    
    print(f"\n🎯 SUMMARY:")
    print(f"Optimal makespan: {makespan}")
    print(f"Total reward: {total_reward:.3f}")

if __name__ == "__main__":
    main()
