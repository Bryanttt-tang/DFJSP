import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import time

# =============================================================================
# SIMPLE FJSP EXAMPLE FROM DYNAMIC-TUTORIAL.PY
# =============================================================================
# This is the original small problem from dynamic-tutorial.py for detailed MDP demonstration

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

print("=== MDP FORMULATION FOR SIMPLE FJSP PROBLEM ===")
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

print("MDP Components:")
print("1. STATE SPACE S:")
print("   State = (J1_ops_completed, J2_ops_completed)")
print("   Where each job can have 0, 1, or 2 operations completed")
print("   Total states: 3 × 3 = 9 states")
print("   S = {(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)}")
print("   Terminal state: (2,2) - all operations completed")
print()

print("2. ACTION SPACE A(s):")
print("   Action = (job_id, operation_index, machine)")
print("   Valid actions depend on current state (precedence constraints)")
print("   From state (0,0): can do J1-O1 or J2-O1 on any compatible machine")
print("   From state (1,0): can do J1-O2 or J2-O1 on any compatible machine")
print("   etc.")
print()

print("3. TRANSITION FUNCTION P(s'|s,a):")
print("   Deterministic transitions based on operation completion")
print("   Action (job_id, op_idx, machine) in state (j1_ops, j2_ops)")
print("   → new state with incremented operation count for the job")
print()

print("4. REWARD FUNCTION R(s,a):")
print("   Using MAKESPAN INCREMENT reward:")
print("   R(s,a) = Current_Makespan - New_Makespan")
print("   This encourages actions that minimize makespan growth")
print("   Makespan = max(completion_time_of_all_operations)")
print()

print("5. POLICY π(s):")
print("   Maps each state to an action")
print("   Goal: Find optimal policy π* that minimizes expected cumulative reward")
print()

print("Let's trace through the MDP step by step...")
print("="*80)
print()

class SimplifiedFJSPEnv:
    """
    Simplified FJSP Environment for Policy Iteration with detailed MDP tracing
    Focuses on the simple 2-job, 3-machine example from dynamic-tutorial.py
    """
    def __init__(self, jobs_data, machine_list, reward_mode="makespan_increment", job_arrival_times=None):
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
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
        # State: operation counts for each job (J1_ops_done, J2_ops_done, ...)
        self.state = tuple(0 for _ in self.job_ids)
        
        # Environment tracks timing internally (handles constraints)
        self.machine_free_times = {m: 0.0 for m in self.machines}
        # Jobs are ready at their arrival times
        self.job_ready_times = {j: self.job_arrival_times[j] for j in self.job_ids}
        self.makespan = 0
        self.schedule_log = collections.defaultdict(list)
        return self.state

    def get_valid_actions(self, state, current_time=None, verbose=False):
        """
        Get valid actions for the current state
        Returns list of (action, processing_time) tuples
        """
        if current_time is None:
            # If no current_time provided, use max arrival time to ensure all jobs are available
            current_time = max(self.job_arrival_times.values())
            
        completed_ops = state  # Operation count state
        valid_actions = []
        
        if verbose:
            print(f"    Getting valid actions for state {state}:")
        
        for job_idx, job_id in enumerate(self.job_ids):
            # Check if job has arrived and has remaining operations
            if (self.job_arrival_times[job_id] <= current_time and 
                completed_ops[job_idx] < len(self.jobs[job_id])):
                
                operation = self.jobs[job_id][completed_ops[job_idx]]
                
                for machine, proc_time in operation['proc_times'].items():
                    action = (job_id, completed_ops[job_idx], machine)
                    valid_actions.append((action, proc_time))
                    if verbose:
                        print(f"      Action: J{job_id}-O{completed_ops[job_idx]+1} on {machine} (time={proc_time})")
        
        if verbose:
            print(f"    Total valid actions: {len(valid_actions)}")
        return valid_actions
    
    def _sync_environment_to_state(self, state):
        """
        Synchronize environment timing to match a given state
        This reconstructs the environment state by replaying optimal decisions
        """
        # Reset timing
        self.machine_free_times = {m: 0.0 for m in self.machines}
        self.job_ready_times = {j: self.job_arrival_times[j] for j in self.job_ids}
        self.makespan = 0
        self.schedule_log = collections.defaultdict(list)
        
        # For simplicity, we'll use a greedy reconstruction
        # In practice, this would need to track the exact sequence of decisions
        current_state = tuple(0 for _ in self.job_ids)
        
        while current_state != state:
            # Find the next operation to complete
            for job_idx, job_id in enumerate(self.job_ids):
                if current_state[job_idx] < state[job_idx]:
                    # This job needs more operations completed
                    op_idx = current_state[job_idx]
                    operation = self.jobs[job_id][op_idx]
                    
                    # Choose the machine with minimum processing time (greedy)
                    best_machine = min(operation['proc_times'].keys(), 
                                     key=lambda m: operation['proc_times'][m])
                    
                    # Execute this operation
                    self._execute_operation(job_id, op_idx, best_machine)
                    
                    # Update state
                    new_state = list(current_state)
                    new_state[job_idx] += 1
                    current_state = tuple(new_state)
                    break

    def _execute_operation(self, job_id, op_idx, machine):
        """Execute an operation and update environment timing"""
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        
        # Start time is when both machine and job are ready
        start_time = max(self.machine_free_times[machine], self.job_ready_times[job_id])
        end_time = start_time + proc_time
        
        # Update environment timing
        self.machine_free_times[machine] = end_time
        self.job_ready_times[job_id] = end_time
        self.makespan = max(self.makespan, end_time)
        
        # Log the operation
        self.schedule_log[machine].append({
            'job_id': job_id, 'op_idx': op_idx,
            'start': start_time, 'end': end_time
        })

    def transition(self, action, verbose=True):
        """
        Execute action and return new state and reward
        """
        job_id, op_idx, machine = action
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        
        # Store current makespan before transition
        current_makespan = self.makespan
        
        if verbose:
            print(f"    Executing action: J{job_id}-O{op_idx+1} on {machine}")
            print(f"      Processing time: {proc_time}")
            print(f"      Current makespan: {current_makespan}")
        
        # Calculate timing (environment handles constraints automatically)
        job_idx = self.job_ids.index(job_id)
        
        # Start time is when both machine and job are ready
        start_time = max(self.machine_free_times[machine], self.job_ready_times[job_id])
        end_time = start_time + proc_time
        
        if verbose:
            print(f"      Machine {machine} available at: {self.machine_free_times[machine]}")
            print(f"      Job {job_id} ready at: {self.job_ready_times[job_id]}")
            print(f"      Operation starts at: {start_time}, ends at: {end_time}")
        
        # Update environment timing
        self.machine_free_times[machine] = end_time
        self.job_ready_times[job_id] = end_time
        self.makespan = max(self.makespan, end_time)
        
        # Store new makespan after transition
        new_makespan = self.makespan
        
        if verbose:
            print(f"      New makespan: {new_makespan}")
        
        # Update operation count state
        new_state = list(self.state)
        new_state[job_idx] += 1
        self.state = tuple(new_state)
        
        # Log the operation
        self.schedule_log[machine].append({
            'job_id': job_id, 'op_idx': op_idx,
            'start': start_time, 'end': end_time
        })
        
        # Calculate reward based on mode
        if self.reward_mode == "makespan_increment":
            # R(s_t, a_t) = current_makespan - new_makespan 
            # Negative increment encourages minimizing makespan growth
            reward = current_makespan - new_makespan
            if verbose:
                print(f"      Reward = {current_makespan} - {new_makespan} = {reward}")
        else:  # processing_time mode
            reward = -proc_time
            if verbose:
                print(f"      Reward = -{proc_time} = {reward}")
        
        return self.state, reward
    
    def is_done(self):
        """Check if all operations are completed"""
        terminal_state = tuple(len(self.jobs[job_id]) for job_id in self.job_ids)
        return self.state == terminal_state

    def get_current_schedule_info(self):
        """Get current schedule information for display"""
        schedule_info = {}
        for machine in self.machines:
            schedule_info[machine] = list(self.schedule_log[machine])
        return schedule_info, self.makespan

def print_policy_step(iteration, V, policy, env):
    """Print detailed policy and value function information"""
    print(f"\n=== ITERATION {iteration} RESULTS ===")
    print("Value Function V(state):")
    for state in sorted(V.keys()):
        print(f"  V{state}: {V[state]:.3f}")
    
    print("\nPolicy π(state):")
    for state in sorted(policy.keys()):
        if policy[state]:
            action = policy[state]
            print(f"  π{state}: J{action[0]}-O{action[1]+1} on {action[2]}")
        else:
            print(f"  π{state}: Terminal state")
    print("-" * 60)

def detailed_policy_iteration(jobs_data, machines, arrival_times, gamma=1.0, theta=1e-6):
    """
    Detailed Policy Iteration for FJSP with step-by-step explanation
    """
    print("=== DETAILED POLICY ITERATION FOR SIMPLE FJSP ===\n")
    
    # Create environment
    env = SimplifiedFJSPEnv(jobs_data, machines, "makespan_increment", arrival_times)
    
    # Generate all possible states
    all_states = []
    job_op_counts = [len(env.jobs[job_id]) for job_id in env.job_ids]
    
    def generate_states(job_idx, current_state):
        if job_idx >= len(env.job_ids):
            all_states.append(tuple(current_state))
            return
        
        max_ops = job_op_counts[job_idx]
        for ops_done in range(max_ops + 1):
            generate_states(job_idx + 1, current_state + [ops_done])
    
    generate_states(0, [])
    
    # Determine terminal state
    terminal_state = tuple(job_op_counts)
    
    print(f"State space: {all_states}")
    print(f"Total states: {len(all_states)}")
    print(f"Terminal state: {terminal_state}")
    print()
    
    # Initialize policy and value function
    policy = {s: None for s in all_states}
    V = {s: 0.0 for s in all_states}
    
    # Initialize policy with first available action for each state
    print("STEP 1: POLICY INITIALIZATION")
    print("Initializing policy π₀(s) with first available action for each state...")
    
    for s in all_states:
        if s == terminal_state:
            continue  # Terminal state has no actions
        
        # Create temporary environment to get valid actions from this state
        temp_env = SimplifiedFJSPEnv(jobs_data, machines, "makespan_increment", arrival_times)
        temp_env.state = s
        temp_env._sync_environment_to_state(s)
        
        valid_actions = temp_env.get_valid_actions(s, max(arrival_times.values()), verbose=True)
        
        if valid_actions:
            policy[s] = valid_actions[0][0]  # Pick first action
            print(f"  π₀{s} = J{valid_actions[0][0][0]}-O{valid_actions[0][0][1]+1} on {valid_actions[0][0][2]}")
    
    print_policy_step(0, V, policy, env)
    
    iteration = 0
    
    while True:
        iteration += 1
        print(f"\n{'='*20} ITERATION {iteration} {'='*20}")
        
        # STEP 1: POLICY EVALUATION
        print(f"\nSTEP 1: POLICY EVALUATION")
        print("Computing value function V^π(s) for current policy π...")
        print("Solving: V^π(s) = R(s,π(s)) + γ ∑_{s'} P(s'|s,π(s)) V^π(s')")
        print("Since transitions are deterministic: V^π(s) = R(s,π(s)) + γ V^π(s')")
        
        eval_iteration = 0
        while True:
            eval_iteration += 1
            print(f"\n  Policy Evaluation Sub-iteration {eval_iteration}:")
            
            delta = 0
            V_old = V.copy()
            
            for s in all_states:
                if s == terminal_state:
                    continue  # Terminal state value remains 0
                
                action = policy[s]
                if action is None:
                    continue
                
                # Create environment to compute transition
                temp_env = SimplifiedFJSPEnv(jobs_data, machines, "makespan_increment", arrival_times)
                temp_env.state = s
                temp_env._sync_environment_to_state(s)
                
                # Execute action to get reward and next state
                next_state, reward = temp_env.transition(action, verbose=False)
                
                # Bellman equation
                old_value = V[s]
                V[s] = reward + gamma * V[next_state]
                
                delta = max(delta, abs(V[s] - old_value))
                
                print(f"    V{s}: R({s},{action}) + γV({next_state}) = {reward:.3f} + {gamma}×{V_old[next_state]:.3f} = {V[s]:.3f}")
            
            print(f"    Max value change (δ): {delta:.6f}")
            
            if delta < theta:
                print(f"    Policy evaluation converged after {eval_iteration} iterations")
                break
        
        # STEP 2: POLICY IMPROVEMENT
        print(f"\nSTEP 2: POLICY IMPROVEMENT")
        print("For each state, finding action that maximizes Q^π(s,a)...")
        print("Q^π(s,a) = R(s,a) + γ ∑_{s'} P(s'|s,a) V^π(s')")
        
        policy_stable = True
        new_policy = policy.copy()
        
        for s in all_states:
            if s == terminal_state:
                continue
            
            old_action = policy[s]
            
            # Create environment to evaluate all actions from this state
            temp_env = SimplifiedFJSPEnv(jobs_data, machines, "makespan_increment", arrival_times)
            temp_env.state = s
            temp_env._sync_environment_to_state(s)
            
            valid_actions = temp_env.get_valid_actions(s, max(arrival_times.values()))
            
            if not valid_actions:
                continue
            
            print(f"\n    Evaluating actions for state {s}:")
            
            best_action = None
            best_value = float('-inf')
            
            for action, _ in valid_actions:
                # Create fresh environment for this action evaluation
                action_env = SimplifiedFJSPEnv(jobs_data, machines, "makespan_increment", arrival_times)
                action_env.state = s
                action_env._sync_environment_to_state(s)
                
                # Execute action to get reward and next state
                next_state, reward = action_env.transition(action, verbose=False)
                
                # Calculate Q-value
                q_value = reward + gamma * V[next_state]
                
                print(f"      Q({s}, J{action[0]}-O{action[1]+1} on {action[2]}) = {reward:.3f} + {gamma}×{V[next_state]:.3f} = {q_value:.3f}")
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            new_policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
                print(f"      Policy changed: {old_action} → {best_action}")
            else:
                print(f"      Policy unchanged: {old_action}")
        
        policy = new_policy
        
        print_policy_step(iteration, V, policy, env)
        
        if policy_stable:
            print(f"\n🎉 POLICY ITERATION CONVERGED after {iteration} iterations!")
            print("Optimal policy found!")
            break
        
        if iteration > 10:  # Safety limit
            print("\nReached iteration limit")
            break
    
    return policy, V

def demonstrate_optimal_policy_execution(jobs_data, machines, arrival_times, optimal_policy):
    """
    Demonstrate executing the optimal policy step by step
    """
    print("\n" + "="*60)
    print("DEMONSTRATING OPTIMAL POLICY EXECUTION")
    print("="*60)
    
    env = SimplifiedFJSPEnv(jobs_data, machines, "makespan_increment", arrival_times)
    
    step = 0
    total_reward = 0
    
    print(f"\nInitial state: {env.state}")
    print(f"Initial makespan: {env.makespan}")
    
    while not env.is_done():
        step += 1
        current_state = env.state
        action = optimal_policy[current_state]
        
        print(f"\n--- Step {step} ---")
        print(f"Current state: {current_state}")
        print(f"Optimal action: J{action[0]}-O{action[1]+1} on {action[2]}")
        
        # Execute action
        next_state, reward = env.transition(action, verbose=True)
        total_reward += reward
        
        print(f"Next state: {next_state}")
        print(f"Step reward: {reward:.3f}")
        print(f"Cumulative reward: {total_reward:.3f}")
        
        # Show current schedule
        schedule_info, makespan = env.get_current_schedule_info()
        print(f"Current schedule:")
        for machine in machines:
            if schedule_info[machine]:
                ops = ", ".join([f"J{op['job_id']}-O{op['op_idx']+1}({op['start']:.1f}-{op['end']:.1f})" 
                               for op in schedule_info[machine]])
                print(f"  {machine}: {ops}")
            else:
                print(f"  {machine}: (empty)")
    
    print(f"\n🏁 FINAL RESULTS:")
    print(f"Total steps: {step}")
    print(f"Final makespan: {env.makespan}")
    print(f"Total reward: {total_reward:.3f}")
    
    return env.makespan, env.get_current_schedule_info()[0]

def main():
    """
    Main function to demonstrate policy iteration on the simple FJSP example
    """
    print("RUNNING POLICY ITERATION ON SIMPLE FJSP EXAMPLE")
    print("="*60)
    
    # Run policy iteration
    optimal_policy, optimal_values = detailed_policy_iteration(
        SIMPLE_JOBS_DATA, SIMPLE_MACHINES, SIMPLE_ARRIVALS
    )
    
    # Demonstrate optimal policy execution
    optimal_makespan, optimal_schedule = demonstrate_optimal_policy_execution(
        SIMPLE_JOBS_DATA, SIMPLE_MACHINES, SIMPLE_ARRIVALS, optimal_policy
    )
    
    print(f"\n🎯 OPTIMAL SOLUTION SUMMARY:")
    print(f"Optimal makespan: {optimal_makespan}")
    print(f"Optimal policy found through policy iteration")
    
    return optimal_policy, optimal_values, optimal_makespan

if __name__ == "__main__":
    main()
