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
    Enhanced FJSP Environment for Policy Iteration with timing information in state
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
        # Enhanced State: (J1_ops_done, J2_ops_done, M1_free_time, M2_free_time, M3_free_time, current_makespan)
        # This includes timing information as part of the state
        initial_ops = tuple(0 for _ in self.job_ids)
        initial_machine_times = tuple(0.0 for _ in self.machines)
        initial_makespan = 0.0
        
        self.state = initial_ops + initial_machine_times + (initial_makespan,)
        
        # Environment tracks timing internally (handles constraints)
        self.machine_free_times = {m: 0.0 for m in self.machines}
        # Jobs are ready at their arrival times
        self.job_ready_times = {j: self.job_arrival_times[j] for j in self.job_ids}
        self.makespan = 0
        self.schedule_log = collections.defaultdict(list)
        return self.state

    def get_valid_actions(self, state, current_time=None):
        """
        Get valid actions for the current state with timing information
        State format: (J1_ops_done, J2_ops_done, M1_free_time, M2_free_time, M3_free_time, current_makespan)
        Returns list of (action, processing_time) tuples
        """
        # Extract components from enhanced state
        num_jobs = len(self.job_ids)
        num_machines = len(self.machines)
        
        completed_ops = state[:num_jobs]  # Operation counts
        machine_free_times = state[num_jobs:num_jobs+num_machines]  # Machine free times
        current_makespan = state[-1]  # Current makespan
        
        if current_time is None:
            current_time = current_makespan
            
        valid_actions = []
        
        print(f"    Getting valid actions for state {state}:")
        print(f"      Operations completed: {dict(zip(self.job_ids, completed_ops))}")
        print(f"      Machine free times: {dict(zip(self.machines, machine_free_times))}")
        print(f"      Current makespan: {current_makespan}")
        
        for job_idx, job_id in enumerate(self.job_ids):
            # Check if job has arrived and has remaining operations
            if (self.job_arrival_times[job_id] <= current_time and 
                completed_ops[job_idx] < len(self.jobs[job_id])):
                
                operation = self.jobs[job_id][completed_ops[job_idx]]
                
                for machine, proc_time in operation['proc_times'].items():
                    action = (job_id, completed_ops[job_idx], machine)
                    valid_actions.append((action, proc_time))
                    machine_idx = self.machines.index(machine)
                    machine_free_time = machine_free_times[machine_idx]
                    job_ready_time = self.job_arrival_times[job_id]
                    
                    # For operations after the first, job ready time is when previous op completes
                    if completed_ops[job_idx] > 0:
                        # This is approximate - in real implementation we'd track job completion times
                        job_ready_time = current_makespan
                    
                    start_time = max(machine_free_time, job_ready_time)
                    end_time = start_time + proc_time
                    
                    print(f"      Action: J{job_id}-O{completed_ops[job_idx]+1} on {machine}")
                    print(f"        Processing time: {proc_time}")
                    print(f"        Machine {machine} free at: {machine_free_time}")
                    print(f"        Job {job_id} ready at: {job_ready_time}")
                    print(f"        Operation would start at: {start_time}, end at: {end_time}")
        
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
        
        # Environment tracks timing internally (handles constraints)
        self.machine_free_times = {m: 0.0 for m in self.machines}
        # Jobs are ready at their arrival times
        self.job_ready_times = {j: self.job_arrival_times[j] for j in self.job_ids}
        self.makespan = 0
        self.schedule_log = collections.defaultdict(list)
        return self.state

    def get_valid_actions(self, state, current_time=None):
        """Get valid actions for the current state"""
        if current_time is None:
            # If no current_time provided, use max arrival time to ensure all jobs are available
            current_time = max(self.job_arrival_times.values())
            
        completed_ops = state  # Operation count state
        valid_actions = []
        
        for job_idx, job_id in enumerate(self.job_ids):
            # Check if job has arrived and has remaining operations
            if (self.job_arrival_times[job_id] <= current_time and 
                completed_ops[job_idx] < len(self.jobs[job_id])):
                
                operation = self.jobs[job_id][completed_ops[job_idx]]
                
                for machine, proc_time in operation['proc_times'].items():
                    # Environment will handle timing constraints during transition
                    action = (job_id, completed_ops[job_idx], machine)
                    valid_actions.append((action, proc_time))
        
        return valid_actions
    
    def _sync_environment_to_state(self, state):
        """Synchronize environment timing to match a given state"""
        # This is a simplified synchronization - in practice this would be complex
        # For policy iteration, we assume environment can be reset to any state
        # Reset timing
        self.machine_free_times = {m: 0.0 for m in self.machines}
        self.job_ready_times = {j: self.job_arrival_times[j] for j in self.job_ids}
        self.makespan = 0
        self.schedule_log = collections.defaultdict(list)

    def transition(self, action):
        job_id, op_idx, machine = action
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        
        # Store current makespan before transition
        current_makespan = self.makespan
        
        # Calculate timing (environment handles constraints automatically)
        job_idx = self.job_ids.index(job_id)
        
        # Start time is when both machine and job are ready
        # Job ready time is max of its arrival time and when previous operation finished
        start_time = max(self.machine_free_times[machine], self.job_ready_times[job_id])
        end_time = start_time + proc_time
        
        # Update environment timing
        self.machine_free_times[machine] = end_time
        self.job_ready_times[job_id] = end_time
        self.makespan = max(self.makespan, end_time)
        
        # Store new makespan after transition
        new_makespan = self.makespan
        
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
            # R(s_t, a_t) = E(t) - E(t+1) = negative increment in makespan
            # This encourages actions that minimize makespan growth
            reward = current_makespan - new_makespan
        else:  # processing_time mode
            reward = -proc_time
        
        return self.state, reward
    
    def is_done(self):
        """Check if all operations are completed"""
        terminal_state = tuple(len(self.jobs[job_id]) for job_id in self.job_ids)
        return self.state == terminal_state

def milp_scheduler(jobs, machines, arrival_times):
    """MILP approach for optimal FJSP scheduling - adapted from test3.py"""
    print("\n--- Running MILP Optimal Scheduler ---")
    prob = LpProblem("FJSP_Optimal", LpMinimize)
    
    # Create operation list: (job_id, operation_index)
    ops = [(j, oi) for j in jobs for oi in range(len(jobs[j]))]
    BIG_M = 1000  # Large number for disjunctive constraints

    # Decision variables
    x = LpVariable.dicts("x", (ops, machines), cat="Binary")  # Assignment variables
    s = LpVariable.dicts("s", ops, lowBound=0)  # Start times
    c = LpVariable.dicts("c", ops, lowBound=0)  # Completion times
    y = LpVariable.dicts("y", (ops, ops, machines), cat="Binary")  # Precedence variables
    Cmax = LpVariable("Cmax", lowBound=0)  # Makespan

    # Objective: minimize makespan
    prob += Cmax

    # Constraints
    for j, oi in ops:
        # Each operation must be assigned to exactly one compatible machine
        prob += lpSum(x[j, oi][m] for m in jobs[j][oi]['proc_times']) == 1
        
        # Completion time definition
        prob += c[j, oi] == s[j, oi] + lpSum(x[j, oi][m] * jobs[j][oi]['proc_times'][m] 
                                           for m in jobs[j][oi]['proc_times'])
        
        # Precedence within a job (operation oi+1 can't start before oi completes)
        if oi > 0:
            prob += s[j, oi] >= c[j, oi - 1]
        
        # Job arrival time constraint (first operation can't start before job arrives)
        if oi == 0:
            prob += s[j, oi] >= arrival_times[j]
        
        # Makespan constraint
        prob += Cmax >= c[j, oi]

    # Machine capacity constraints (disjunctive constraints)
    for m in machines:
        ops_on_m = [op for op in ops if m in jobs[op[0]][op[1]]['proc_times']]
        for i in range(len(ops_on_m)):
            for k in range(i + 1, len(ops_on_m)):
                op1, op2 = ops_on_m[i], ops_on_m[k]
                # Either op1 finishes before op2 starts, or op2 finishes before op1 starts
                prob += s[op1] >= c[op2] - BIG_M * (1 - y[op1][op2][m]) - BIG_M * (2 - x[op1][m] - x[op2][m])
                prob += s[op2] >= c[op1] - BIG_M * y[op1][op2][m] - BIG_M * (2 - x[op1][m] - x[op2][m])

    # Solve the problem
    print("Solving MILP... (this may take a while for large instances)")
    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=300))  # 5-minute time limit

    # Extract solution
    schedule = {m: [] for m in machines}
    if prob.status == 1 and Cmax.varValue is not None:  # Optimal solution found
        for (j, oi), m in ((op, m) for op in ops for m in jobs[op[0]][op[1]]['proc_times']):
            if x[j, oi][m].varValue > 0.5:
                schedule[m].append({
                    'job_id': j, 'op_idx': oi,
                    'start': s[j, oi].varValue, 'end': c[j, oi].varValue
                })
        
        # Sort operations by start time
        for m in machines:
            schedule[m].sort(key=lambda x: x['start'])
        
        print(f"MILP (optimal) Makespan: {Cmax.varValue:.2f}")
        return Cmax.varValue, schedule
    else:
        print("MILP solver failed to find optimal solution (timeout or infeasible)")
        return float('inf'), schedule
    
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
        
        valid_actions = temp_env.get_valid_actions(s, max(arrival_times.values()))
        
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
        
        eval_iter = 0
        while True:
            eval_iter += 1
            delta = 0
            print(f"  Policy Evaluation Sub-iteration {eval_iter}:")
            
            for s in all_states:
                if s == terminal_state:
                    continue  # Terminal state value remains 0
                
                v_old = V[s]
                action = policy[s]
                
                if action is None:
                    continue
                
                # Simulate action from this state
                sim_env = SimplifiedFJSPEnv(env.jobs, env.machines, reward_mode, env.job_arrival_times)
                sim_env.state = s
                sim_env._sync_environment_to_state(s)
                
                # Store makespan before action for detailed output
                old_makespan = sim_env.makespan
                next_state, reward = sim_env.transition(action)
                new_makespan = sim_env.makespan
                
                V[s] = reward + gamma * V[next_state]
                delta = max(delta, np.abs(v_old - V[s]))
                
                if reward_mode == "makespan_increment":
                    print(f"    V{s}: {v_old:.3f} → {V[s]:.3f}")
                    print(f"           R = E(t) - E(t+1) = {old_makespan:.1f} - {new_makespan:.1f} = {reward:.3f}")
                    print(f"           V(s) = {reward:.3f} + {gamma} * {V[next_state]:.3f}")
                    print(f"           (action: J{action[0]}-Op{action[1]+1} on {action[2]}, next: {next_state})")
                else:
                    print(f"    V{s}: {v_old:.3f} → {V[s]:.3f} = {reward:.3f} + {gamma} * {V[next_state]:.3f}")
                    print(f"           (action: J{action[0]}-Op{action[1]+1} on {action[2]}, next: {next_state})")
            
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
        
        for s in all_states:
            if s == terminal_state:
                continue  # Terminal state
            
            old_action = policy[s]
            best_q = -np.inf
            best_action = None
            
            # Create temporary environment to get valid actions
            temp_env = SimplifiedFJSPEnv(env.jobs, env.machines, reward_mode, env.job_arrival_times)
            temp_env.state = s
            temp_env._sync_environment_to_state(s)
            max_arrival_time = max(temp_env.job_arrival_times.values())
            valid_actions = temp_env.get_valid_actions(s, max_arrival_time)
            
            print(f"  State {s} - Evaluating all possible actions:")
            
            q_values = []
            for action, proc_time in valid_actions:
                # Simulate this action
                sim_env = SimplifiedFJSPEnv(env.jobs, env.machines, reward_mode, env.job_arrival_times)
                sim_env.state = s
                sim_env._sync_environment_to_state(s)
                
                # Store makespan before action for detailed output
                old_makespan = sim_env.makespan
                next_state, reward = sim_env.transition(action)
                new_makespan = sim_env.makespan
                q_val = reward + gamma * V[next_state]
                q_values.append((action, q_val, reward, next_state))
                
                print(f"    Action J{action[0]}-Op{action[1]+1} on {action[2]}:")
                if reward_mode == "makespan_increment":
                    print(f"      R = E(t) - E(t+1) = {old_makespan:.1f} - {new_makespan:.1f} = {reward:.3f}")
                    print(f"      Q(s,a) = {reward:.3f} + {gamma} * {V[next_state]:.3f} = {q_val:.3f}")
                else:
                    print(f"      R = -proc_time = {reward:.3f}")
                    print(f"      Q(s,a) = {reward:.3f} + {gamma} * {V[next_state]:.3f} = {q_val:.3f}")
                print(f"      Next state: {next_state}")
                
                if q_val > best_q:
                    best_q = q_val
                    best_action = action
            
            # Show the selection process
            print(f"    Best Q-value: {best_q:.3f}")
            if best_action != old_action:
                policy_stable = False
                print(f"    Policy CHANGED: {old_action} → {best_action}")
            else:
                print(f"    Policy UNCHANGED: {best_action}")
            
            policy[s] = best_action
            print()
        
        print_policy_step(iteration, V, policy)
        
        if policy_stable:
            print("POLICY CONVERGED!")
            print("No more policy changes - optimal policy found!")
            print()
            print("Key insights:")
            print("- State = (J1_ops_done, J2_ops_done) captures progression")
            print("- Environment handles timing constraints internally")
            print("- Value function represents expected cumulative reward")
            print("- Policy improvement selects actions with highest Q-values")
            return policy, V

def plot_gantt_chart(schedule, title="Gantt Chart"):
    fig, gnt = plt.subplots(figsize=(12, 6))
    
    machines = sorted(schedule.keys())
    machine_indices = {m: i for i, m in enumerate(machines)}
    
    gnt.set_ylim(0, len(machines) * 10)
    gnt.set_xlim(0, max(max(task['end'] for task in machine_schedule) for machine_schedule in schedule.values()))

    gnt.set_xlabel('Time')
    gnt.set_ylabel('Machines')
    
    gnt.set_yticks([i * 10 + 5 for i in range(len(machines))])
    gnt.set_yticklabels(machines)
    gnt.grid(True)
    
    # Create color map for jobs
    num_jobs = max(task['job_id'] for machine_schedule in schedule.values() for task in machine_schedule) + 1
    job_colors = plt.cm.get_cmap('Paired', num_jobs)
    
    for machine, tasks in schedule.items():
        for task in tasks:
            machine_idx = machine_indices[machine]
            gnt.broken_barh([(task['start'], task['end'] - task['start'])], 
                            (machine_idx * 10, 9), 
                            facecolors=(job_colors(task['job_id'])))
            gnt.text(task['start'] + (task['end'] - task['start']) / 2, 
                     machine_idx * 10 + 4.5, 
                     f'J{task["job_id"]}-Op{task["op_idx"]+1}', 
                     ha='center', va='center', color='white', fontweight='bold')
    
    plt.title(title)
    plt.show()

def select_problem_instance():
    """Select the large problem instance for testing"""
    jobs = LARGE_JOBS_DATA
    machines = LARGE_MACHINES
    arrivals = LARGE_ARRIVALS
    
    print("📊 Using LARGE instance (7 jobs from test3.py - all arrive at t=0)")
    print(f"   Jobs: {len(jobs)}")
    print(f"   Machines: {machines}")
    print(f"   All jobs arrive at t=0 (static scenario)")
    
    # State space analysis
    state_space_size = 1
    for job_id in jobs:
        state_space_size *= (len(jobs[job_id]) + 1)
    
    print(f"   State space size: {state_space_size:,}")
    if state_space_size > 100000:
        print("   ⚠️  Very large state space - Policy Iteration will be slow!")
    else:
        print("   ✅ Manageable state space size")
    
    return jobs, machines, arrivals


if __name__ == "__main__":
    print("  - 'original': 2 jobs, 2 ops each (original small instance)")
    print("  - 'medium': 4 jobs from test3.py (all arrive at t=0)")
    print("  - 'large': 7 jobs from test3.py (all arrive at t=0)")
    print("  - 'xlarge': 10 jobs (all arrive at t=0)")
    
    # You can change this to test different instances
    INSTANCE_TYPE = "large"  # Change to "medium", "large", or "xlarge" to test larger instances
    
    selected_jobs, selected_machines, selected_arrivals = select_problem_instance(INSTANCE_TYPE)
    
    
    print(f"\n--- PROBLEM INSTANCE DETAILS ---")
    for job_id, ops in selected_jobs.items():
        print(f"Job {job_id}: {len(ops)} operations (arrives at t={selected_arrivals[job_id]})")
        for op_idx, op in enumerate(ops):
            print(f"  Op {op_idx+1}: {op['proc_times']}")
    
    print(f"Machines: {selected_machines}")
    print("All jobs arrive at t=0 (no dynamic arrivals)")

    # Demonstrate both reward modes
    for reward_mode in ["processing_time", "makespan_increment"]:
        print("\n" + "=" * 80)
        print(f"RUNNING POLICY ITERATION WITH REWARD MODE: {reward_mode.upper()}")
        print("=" * 80)
        
        env = SimplifiedFJSPEnv(selected_jobs, selected_machines, reward_mode, selected_arrivals)
        optimal_policy, optimal_values = policy_iteration(env, reward_mode=reward_mode)
        
        print(f"\n--- FINAL OPTIMAL POLICY ({reward_mode}) ---")
        print("Optimal Policy:")
        terminal_state = tuple(len(env.jobs[job_id]) for job_id in env.job_ids)
        for state, action in sorted(optimal_policy.items()):
            if state != terminal_state and action:
                print(f"  π({state}): Action J{action[0]}-Op{action[1]+1} on {action[2]}")
        
        print(f"\n--- SIMULATING OPTIMAL SCHEDULE ({reward_mode}) ---")
        final_schedule_env = SimplifiedFJSPEnv(selected_jobs, selected_machines, reward_mode, selected_arrivals)
        state = final_schedule_env.reset()
        step = 0
        
        print("Executing optimal policy:")
        while state != terminal_state:
            step += 1
            action = optimal_policy[state]
            
            # Store old makespan for detailed output
            old_makespan = final_schedule_env.makespan
            print(f"Step {step}: State {state} → Action J{action[0]}-Op{action[1]+1} on {action[2]}")
            
            state, reward = final_schedule_env.transition(action)
            new_makespan = final_schedule_env.makespan
            
            if reward_mode == "makespan_increment":
                print(f"         E(t) = {old_makespan:.1f}, E(t+1) = {new_makespan:.1f}")
                print(f"         Reward = E(t) - E(t+1) = {old_makespan:.1f} - {new_makespan:.1f} = {reward:.3f}")
            else:
                print(f"         Reward = -proc_time = {reward:.2f}")
            print(f"         New state: {state}, Current makespan: {new_makespan:.2f}")
            print()
        
        makespan = final_schedule_env.makespan
        print(f"Final Makespan: {makespan}")
        
        # Print detailed schedule
        print("\nDetailed Schedule:")
        for machine in selected_machines:
            print(f"{machine}:")
            for task in final_schedule_env.schedule_log[machine]:
                print(f"  J{task['job_id']}-Op{task['op_idx']+1}: [{task['start']:.1f}, {task['end']:.1f}]")
        
        plot_gantt_chart(final_schedule_env.schedule_log, 
                        title=f"Optimal Schedule - {reward_mode} (Makespan: {makespan})")
        print("\n")    # Event-based demo disabled for static analysis (all jobs arrive at t=0)
    # if selected_arrivals and any(t > 0 for t in selected_arrivals.values()):
    #     print("=" * 80)
    #     print("DYNAMIC ARRIVAL DEMONSTRATION - EVENT-BASED GREEDY") 
    #     print("=" * 80)
    #     event_makespan, event_schedule = event_based_greedy_demo(selected_jobs, selected_machines, selected_arrivals)
    #     plot_gantt_chart(event_schedule, 
    #                     title=f"Event-based Greedy Schedule (Makespan: {event_makespan:.2f})",
    #                     arrival_times=selected_arrivals)

def run_instance(instance_type, verbose=True):
    """Run a specific instance individually with MILP benchmark"""
    print(f"🎯 RUNNING {instance_type.upper()} INSTANCE WITH MILP BENCHMARK")
    print("=" * 70)
    
    selected_jobs, selected_machines, selected_arrivals = select_problem_instance()
    
    if selected_jobs is None:
        print(f"❌ {instance_type} instance is too large for Policy Iteration")
        return
    
    results = {}
    
    # First, get MILP optimal solution as benchmark
    print(f"\n🔢 RUNNING MILP SOLVER FOR OPTIMAL BENCHMARK")
    print("-" * 50)
    
    try:
        optimal_makespan, optimal_schedule = milp_scheduler(selected_jobs, selected_machines, selected_arrivals)
        results['milp_optimal'] = {
            'makespan': optimal_makespan,
            'schedule': optimal_schedule
        }
        print(f"✅ MILP Optimal Makespan: {optimal_makespan:.2f}")
        
        # Plot MILP optimal schedule
        if optimal_makespan != float('inf'):
            plot_gantt_chart(optimal_schedule, 
                            title=f"{instance_type.title()} - MILP Optimal (Makespan: {optimal_makespan:.2f})")
    
    except Exception as e:
        print(f"❌ MILP solver error: {str(e)}")
        results['milp_optimal'] = {'makespan': float('inf'), 'schedule': {}}
    
    # Test both PI reward modes for comparison
    print(f"\n🧠 RUNNING POLICY ITERATION METHODS")
    print("-" * 50)
    
    for reward_mode in ["processing_time", "makespan_increment"]:
        print(f"\n--- Testing PI with {reward_mode} reward mode ---")
        
        try:
            # Run policy iteration
            env = SimplifiedFJSPEnv(selected_jobs, selected_machines, reward_mode, selected_arrivals)
            
            # For large instances, reduce verbosity
            if len(selected_jobs) > 4 and not verbose:
                print("Running Policy Iteration (output suppressed for large instance)...")
                # Temporarily redirect output for large instances
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    optimal_policy, optimal_values = policy_iteration(env, reward_mode=reward_mode)
                finally:
                    sys.stdout = old_stdout
                print("Policy Iteration completed!")
            else:
                optimal_policy, optimal_values = policy_iteration(env, reward_mode=reward_mode)
            
            # Execute policy
            final_env = SimplifiedFJSPEnv(selected_jobs, selected_machines, reward_mode, selected_arrivals)
            state = final_env.reset()
            terminal_state = tuple(len(final_env.jobs[job_id]) for job_id in final_env.job_ids)
            
            step = 0
            if verbose or len(selected_jobs) <= 4:
                print(f"\nExecuting optimal policy for {reward_mode}:")
                while state != terminal_state:
                    step += 1
                    action = optimal_policy[state]
                    old_makespan = final_env.makespan
                    state, reward = final_env.transition(action)
                    print(f"Step {step}: J{action[0]}-Op{action[1]+1} on {action[2]} → Makespan: {final_env.makespan:.2f}")
            else:
                # Silent execution for large instances
                while state != terminal_state:
                    step += 1
                    action = optimal_policy[state]
                    state, reward = final_env.transition(action)
                print(f"Executed {step} steps silently")
            
            makespan = final_env.makespan
            results[f'pi_{reward_mode}'] = {
                'makespan': makespan,
                'schedule': dict(final_env.schedule_log),
                'steps': step
            }
            
            print(f"✅ PI ({reward_mode}) Makespan: {makespan:.2f} (in {step} steps)")
            plot_gantt_chart(final_env.schedule_log, 
                            title=f"{instance_type.title()} - PI {reward_mode} (Makespan: {makespan:.2f})")
            
        except Exception as e:
            print(f"❌ Error with PI {reward_mode}: {str(e)}")
            results[f'pi_{reward_mode}'] = {'makespan': float('inf'), 'schedule': {}, 'steps': 0}
    
    # Compare results with MILP benchmark
    print(f"\n📊 COMPARISON RESULTS for {instance_type.upper()} INSTANCE")
    print("=" * 60)
    
    milp_makespan = results.get('milp_optimal', {}).get('makespan', float('inf'))
    
    for method, result in results.items():
        makespan = result['makespan']
        if method == 'milp_optimal':
            print(f"MILP Optimal:           {makespan:.2f}")
        else:
            gap = ((makespan - milp_makespan) / milp_makespan * 100) if milp_makespan != float('inf') and milp_makespan > 0 else float('inf')
            method_name = method.replace('pi_', 'PI ').replace('_', ' ').title()
            steps = result.get('steps', 0)
            if gap != float('inf'):
                print(f"{method_name:20}: {makespan:.2f} (gap: {gap:+.1f}%, {steps} steps)")
            else:
                print(f"{method_name:20}: {makespan:.2f} ({steps} steps)")
    
    return results

def test_large_instance():
    """Test large instance with reduced output for performance"""
    print("🔥 TESTING LARGE INSTANCE (7 jobs)")
    print("=" * 50)
    print("This may take several minutes due to large state space...")
    
    import time
    start_time = time.time()
    
    results = run_instance("large", verbose=False)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n⏱️  Large instance completed in {elapsed:.1f} seconds")
    return results

def test_xlarge_instance():
    """Test extra large instance - WARNING: Very slow!"""
    print("🚨 TESTING EXTRA LARGE INSTANCE (10 jobs)")
    print("=" * 50)
    print("WARNING: This will take a very long time (potentially hours)!")
    print("State space is over 1 million states!")
    
    # Check state space first
    selected_jobs, selected_machines, selected_arrivals = select_problem_instance("xlarge")
    if selected_jobs is None:
        return None
    
    response = input("Are you sure you want to continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled XLarge instance test")
        return None
    
    import time
    start_time = time.time()
    
    results = run_instance("xlarge", verbose=False)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n⏱️  XLarge instance completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    return results

def event_based_greedy_demo(jobs_data, machine_list, arrival_times):
    """
    Simple event-based greedy scheduler for comparison
    Shows how scheduling adapts when jobs arrive dynamically
    """
    print(f"\n🔄 EVENT-BASED GREEDY SCHEDULER DEMO")
    print("=" * 50)
    
    env = SimplifiedFJSPEnv(jobs_data, machine_list, "processing_time", arrival_times)
    
    # Sort arrival events
    arrival_events = sorted([(time, job_id) for job_id, time in arrival_times.items() if time > 0])
    
    current_time = 0.0
    next_arrival_idx = 0
    step = 0
    
    print(f"Initial jobs: {[job_id for job_id, t in arrival_times.items() if t <= 0]}")
    print(f"Future arrivals: {arrival_events}")
    print()
    
    while not env.is_done():
        # Check for new arrivals
        while (next_arrival_idx < len(arrival_events) and 
               arrival_events[next_arrival_idx][0] <= current_time):
            arrival_time, job_id = arrival_events[next_arrival_idx]
            print(f"⚡ Job {job_id} arrives at time {arrival_time:.1f}")
            next_arrival_idx += 1
        
        # Get valid actions at current time
        valid_actions = env.get_valid_actions(env.state, current_time)
        
        if not valid_actions:
            # Advance to next arrival
            if next_arrival_idx < len(arrival_events):
                next_time = arrival_events[next_arrival_idx][0]
                current_time = next_time
                print(f"⏭ Advancing time to {current_time:.1f}")
                continue
            else:
                break
        
        # Greedy: shortest processing time
        action = min(valid_actions, key=lambda x: x[1])[0]
        
        step += 1
        old_makespan = env.makespan
        state, reward = env.transition(action)
        current_time = env.makespan
        
        print(f"Step {step}: J{action[0]}-Op{action[1]+1} on {action[2]} → Time: {current_time:.1f}")
    
    print(f"\nEvent-based makespan: {env.makespan:.2f}")
    return env.makespan, dict(env.schedule_log)

if __name__ == "__main__":
    print("🔧 POLICY ITERATION vs MILP OPTIMAL BENCHMARK")
    print("=" * 60)
    print("Testing Policy Iteration against MILP optimal solver")
    print("All jobs arrive at t=0 (no dynamic arrivals)")
    print("Each instance includes:")
    print("  📊 MILP optimal solution (benchmark)")
    print("  🧠 Policy Iteration with processing_time reward")
    print("  🧠 Policy Iteration with makespan_increment reward")
    print()
    
    print("🎯 Testing large dataset (7 jobs, 3-4 ops each, state space: ~25,600)")
    print()
    
    all_results = {}
    total_start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"TESTING LARGE INSTANCE")
    print(f"{'='*80}")
    
    import time
    start_time = time.time()
    
    try:
        # Run both reward modes on large dataset
        results = run_instance("large", verbose=False)
        all_results["large"] = results
        
        elapsed = time.time() - start_time
        print(f"✅ LARGE instance completed in {elapsed:.1f} seconds")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error in large instance after {elapsed:.1f} seconds: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    
    total_elapsed = time.time() - total_start_time
    
    # Summary of all results
    print(f"\n🏁 SUMMARY OF LARGE DATASET TEST ({total_elapsed:.1f} seconds total)")
    print("=" * 50)
    for instance_type, results in all_results.items():
        print(f"\n{instance_type.upper()} Instance:")
        if results:
            for mode, result in results.items():
                print(f"  {mode}: Makespan = {result['makespan']:.2f} (Steps: {result['steps']})")
        else:
            print("  Failed to complete")
    
    print(f"\n⏱️  Large dataset test completed in {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")