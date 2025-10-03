# Clean Policy Iteration for FJSP with Proper State Space Exploration
import itertools
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpBinary

# Problem definition - LARGE dataset for testing scalability
LARGE_JOBS_DATA = {
    1: [
        {'proc_times': {'M1': 3, 'M2': 2, 'M4': 4}},
        {'proc_times': {'M1': 2, 'M3': 3, 'M5': 2}},
        {'proc_times': {'M2': 4, 'M4': 3, 'M5': 3}}
    ],
    2: [
        {'proc_times': {'M2': 3, 'M3': 2, 'M5': 4}},
        {'proc_times': {'M1': 3, 'M4': 3, 'M5': 2}},
        {'proc_times': {'M3': 2, 'M4': 2, 'M5': 2}}
    ],
    3: [
        {'proc_times': {'M1': 3, 'M3': 3, 'M4': 2}},
        {'proc_times': {'M2': 3, 'M4': 2, 'M5': 3}}
    ],
    4: [
        {'proc_times': {'M2': 2, 'M4': 4, 'M5': 3}},
        {'proc_times': {'M1': 2, 'M3': 2, 'M4': 3}}
    ]
}
LARGE_MACHINES = ['M1', 'M2', 'M3', 'M4', 'M5']
LARGE_ARRIVALS = {1: 0, 2: 0, 3: 0, 4: 0}

# Use LARGE dataset
JOBS_DATA = LARGE_JOBS_DATA
MACHINES = LARGE_MACHINES
ARRIVALS = LARGE_ARRIVALS

job_ids = list(JOBS_DATA.keys())
machines = MACHINES[:]
total_operations = sum(len(ops) for ops in JOBS_DATA.values())

print(f"Problem: {len(job_ids)} jobs, {len(machines)} machines, {total_operations} operations")

def create_compact_state(next_ops, machine_next_free, job_ready_time, current_time):
    """Create compact state representation focusing on current decision information."""
    ready_jobs = []
    machine_idle = []
    proc_times = []
    
    # Which jobs have ready operations now?
    for j_idx in range(len(job_ids)):
        job_id = job_ids[j_idx]
        op_idx = next_ops[j_idx]
        is_ready = (op_idx < len(JOBS_DATA[job_id]) and job_ready_time[j_idx] <= current_time)
        ready_jobs.append(1 if is_ready else 0)
    
    # Which machines are idle now?
    for m_idx in range(len(machines)):
        is_idle = machine_next_free[m_idx] <= current_time
        machine_idle.append(1 if is_idle else 0)
    
    # Processing times for ready operations on each machine
    for j_idx in range(len(job_ids)):
        job_id = job_ids[j_idx]
        op_idx = next_ops[j_idx]
        
        if ready_jobs[j_idx]:  # Job has ready operation
            operation = JOBS_DATA[job_id][op_idx]
            for machine in machines:
                proc_times.append(operation['proc_times'].get(machine, 0))
        else:
            # Job not ready: zeros for all machines
            for machine in machines:
                proc_times.append(0)
    
    return (tuple(ready_jobs), tuple(machine_idle), tuple(proc_times))

def compute_makespan(machine_next_free, job_ready_time):
    vals = []
    if machine_next_free:
        vals.append(max(machine_next_free))
    if job_ready_time:
        vals.append(max(job_ready_time))
    return max(vals) if vals else 0

# Build MDP with complete state space exploration
print("Building MDP via exhaustive BFS...")
state_to_id = {}
id_to_state = []
internal_states = {}  # state_id -> internal representation
transitions = defaultdict(list)

def add_state(compact_state, internal_state):
    if compact_state not in state_to_id:
        sid = len(id_to_state)
        state_to_id[compact_state] = sid
        id_to_state.append(compact_state)
        internal_states[sid] = internal_state
    return state_to_id[compact_state]

# Start BFS from initial state
init_next_ops = [0] * len(job_ids)
init_machine_next_free = [0] * len(machines)
init_job_ready = [ARRIVALS[job_ids[i]] for i in range(len(job_ids))]
init_time = 0

init_compact = create_compact_state(init_next_ops, init_machine_next_free, init_job_ready, init_time)
init_id = add_state(init_compact, (init_next_ops[:], init_machine_next_free[:], init_job_ready[:], init_time))

# BFS queue with internal states
queue = deque()
queue.append((init_next_ops[:], init_machine_next_free[:], init_job_ready[:], init_time))
visited = set()

MAX_STATES = 50000
states_processed = 0

while queue and len(id_to_state) < MAX_STATES:
    next_ops, machine_next_free, job_ready_time, current_time = queue.popleft()
    
    # Create unique key for this internal state
    internal_key = (tuple(next_ops), tuple(machine_next_free), tuple(job_ready_time), current_time)
    if internal_key in visited:
        continue
    visited.add(internal_key)
    
    states_processed += 1
    if states_processed % 1000 == 0:
        print(f"  Processed {states_processed} states, generated {len(id_to_state)} compact states")
    
    # Get compact state and its ID
    compact_state = create_compact_state(next_ops, machine_next_free, job_ready_time, current_time)
    sid = add_state(compact_state, (next_ops[:], machine_next_free[:], job_ready_time[:], current_time))
    
    # Check if terminal (all operations completed)
    finished_ops = sum(next_ops)
    if finished_ops >= total_operations:
        continue
    
    # Find legal immediate scheduling actions
    legal_actions = []
    for j_idx in range(len(job_ids)):
        job_id = job_ids[j_idx]
        op_idx = next_ops[j_idx]
        
        if op_idx >= len(JOBS_DATA[job_id]):
            continue  # Job completed
            
        if job_ready_time[j_idx] <= current_time:  # Job ready
            operation = JOBS_DATA[job_id][op_idx]
            for m_idx in range(len(machines)):
                machine = machines[m_idx]
                if (machine_next_free[m_idx] <= current_time and  # Machine idle
                    machine in operation['proc_times']):  # Machine can process
                    legal_actions.append((j_idx, op_idx, m_idx))
    
    current_makespan = compute_makespan(machine_next_free, job_ready_time)
    
    if legal_actions:
        # Add scheduling transitions
        for j_idx, op_idx, m_idx in legal_actions:
            job_id = job_ids[j_idx]
            machine = machines[m_idx]
            proc_time = JOBS_DATA[job_id][op_idx]['proc_times'][machine]
            
            # Create next state
            new_next_ops = next_ops[:]
            new_next_ops[j_idx] += 1
            new_machine_next_free = machine_next_free[:]
            new_machine_next_free[m_idx] = current_time + proc_time
            new_job_ready = job_ready_time[:]
            new_job_ready[j_idx] = current_time + proc_time
            
            new_compact = create_compact_state(new_next_ops, new_machine_next_free, new_job_ready, current_time)
            new_sid = add_state(new_compact, (new_next_ops, new_machine_next_free, new_job_ready, current_time))
            
            # Reward: negative makespan increase
            new_makespan = compute_makespan(new_machine_next_free, new_job_ready)
            reward = -(new_makespan - current_makespan)
            
            # Add transition (avoid duplicates)
            action = (j_idx, op_idx, m_idx)
            if not any(t[0] == action and t[1] == new_sid for t in transitions[sid]):
                transitions[sid].append((action, new_sid, reward))
            
            # Add to queue for further exploration
            new_internal_key = (tuple(new_next_ops), tuple(new_machine_next_free), tuple(new_job_ready), current_time)
            if new_internal_key not in visited:
                queue.append((new_next_ops, new_machine_next_free, new_job_ready, current_time))
    
    else:
        # No immediate actions - WAIT transition to next event
        candidates = []
        for t in machine_next_free:
            if t > current_time:
                candidates.append(t)
        for t in job_ready_time:
            if t > current_time:
                candidates.append(t)
        
        if candidates:
            next_time = min(candidates)
            new_compact = create_compact_state(next_ops, machine_next_free, job_ready_time, next_time)
            new_sid = add_state(new_compact, (next_ops[:], machine_next_free[:], job_ready_time[:], next_time))
            
            # WAIT reward (no makespan change but small penalty)
            reward = -0.01
            
            # Add WAIT transition
            if not any(t[0] == 'WAIT' and t[1] == new_sid for t in transitions[sid]):
                transitions[sid].append(('WAIT', new_sid, reward))
            
            # Add to queue
            new_internal_key = (tuple(next_ops), tuple(machine_next_free), tuple(job_ready_time), next_time)
            if new_internal_key not in visited:
                queue.append((next_ops, machine_next_free, job_ready_time, next_time))

num_states = len(id_to_state)
num_transitions = sum(len(v) for v in transitions.values())
print(f"MDP complete: {num_states} states, {num_transitions} transitions")

if num_states >= MAX_STATES:
    print("WARNING: Hit state limit - MDP may be incomplete")

# Run Policy Iteration with better convergence
print("\nRunning Policy Iteration...")
gamma = 0.99  # Slight discount to ensure convergence

# Initialize random policy
policy = {}
for sid in range(num_states):
    acts = transitions.get(sid, [])
    if acts:
        # Choose first available action
        policy[sid] = acts[0][0]
    else:
        policy[sid] = None

V = [0.0] * num_states

def policy_evaluation(tol=1e-6, max_iter=500):
    for iteration in range(max_iter):
        delta = 0.0
        for sid in range(num_states):
            if policy[sid] is None:
                continue
            
            # Find transition for current policy
            transition = None
            for action, nsid, reward in transitions[sid]:
                if action == policy[sid]:
                    transition = (nsid, reward)
                    break
            
            if transition:
                nsid, reward = transition
                new_v = reward + gamma * V[nsid]
                delta = max(delta, abs(new_v - V[sid]))
                V[sid] = new_v
        
        if iteration % 50 == 0:
            print(f"  Evaluation iter {iteration}: delta = {delta:.6e}")
            
        if delta < tol:
            print(f"  Policy evaluation converged in {iteration + 1} iterations")
            return True
    
    print(f"  Policy evaluation did not converge (delta = {delta:.6e})")
    return False

def policy_improvement():
    stable = True
    changes = 0
    
    for sid in range(num_states):
        if not transitions[sid]:
            continue
            
        old_action = policy[sid]
        best_action = None
        best_value = float('-inf')
        
        for action, nsid, reward in transitions[sid]:
            value = reward + gamma * V[nsid]
            if value > best_value:
                best_value = value
                best_action = action
        
        if old_action != best_action:
            policy[sid] = best_action
            stable = False
            changes += 1
    
    print(f"  Policy improvement: {changes} changes")
    return stable

# Policy iteration loop
iteration = 0
max_iterations = 20

while iteration < max_iterations:
    iteration += 1
    print(f"\nPolicy Iteration {iteration}:")
    
    converged = policy_evaluation()
    if not converged:
        print("Policy evaluation failed to converge - stopping")
        break
        
    stable = policy_improvement()
    if stable:
        print("Policy converged!")
        break

print(f"\nFinal: V(init) = {V[init_id]:.3f}")

# Execute policy with proper state tracking
print("\nExecuting policy...")
current_internal = internal_states[init_id]
next_ops, machine_next_free, job_ready_time, current_time = [x[:] if isinstance(x, list) else x for x in current_internal]

schedule = {m: [] for m in machines}
step = 0
total_reward = 0

while step < 50:  # Safety limit
    finished_ops = sum(next_ops)
    if finished_ops >= total_operations:
        break
    
    compact_state = create_compact_state(next_ops, machine_next_free, job_ready_time, current_time)
    
    if compact_state not in state_to_id:
        print(f"STOP: Reached unexplored state at step {step}")
        break
    
    sid = state_to_id[compact_state]
    action = policy[sid]
    
    if action is None:
        print(f"STOP: No action available at step {step}")
        break
    
    step += 1
    
    if action == 'WAIT':
        # Advance time
        candidates = [t for t in machine_next_free if t > current_time] + [t for t in job_ready_time if t > current_time]
        if candidates:
            current_time = min(candidates)
        else:
            break
    else:
        # Execute scheduling action
        j_idx, op_idx, m_idx = action
        job_id = job_ids[j_idx]
        machine = machines[m_idx]
        proc_time = JOBS_DATA[job_id][op_idx]['proc_times'][machine]
        
        start_time = current_time
        end_time = start_time + proc_time
        
        schedule[machine].append({
            'job': job_id,
            'op': op_idx + 1,
            'start': start_time,
            'end': end_time
        })
        
        next_ops[j_idx] += 1
        machine_next_free[m_idx] = end_time
        job_ready_time[j_idx] = end_time

final_makespan = max(max(machine_next_free), max(job_ready_time))
print(f"Policy execution: {step} steps, makespan = {final_makespan}")

# MILP for comparison
def solve_milp():
    print("\nSolving MILP...")
    prob = LpProblem("FJSP", LpMinimize)
    
    ops = [(j, i) for j in job_ids for i in range(len(JOBS_DATA[j]))]
    
    x, s, c = {}, {}, {}
    for j, i in ops:
        s[j, i] = LpVariable(f"s_{j}_{i}", lowBound=0)
        c[j, i] = LpVariable(f"c_{j}_{i}", lowBound=0)
        for m in machines:
            if m in JOBS_DATA[j][i]['proc_times']:
                x[j, i, m] = LpVariable(f"x_{j}_{i}_{m}", cat=LpBinary)
    
    Cmax = LpVariable("Cmax", lowBound=0)
    prob += Cmax
    
    # Constraints
    for j, i in ops:
        eligible = [m for m in machines if m in JOBS_DATA[j][i]['proc_times']]
        prob += lpSum(x[j, i, m] for m in eligible) == 1
        prob += c[j, i] == s[j, i] + lpSum(x[j, i, m] * JOBS_DATA[j][i]['proc_times'][m] for m in eligible)
        if i > 0:
            prob += s[j, i] >= c[j, i-1]
        prob += Cmax >= c[j, i]
    
    # Disjunctive constraints
    for m in machines:
        ops_on_m = [(j, i) for j, i in ops if m in JOBS_DATA[j][i]['proc_times']]
        for idx1 in range(len(ops_on_m)):
            for idx2 in range(idx1 + 1, len(ops_on_m)):
                op1, op2 = ops_on_m[idx1], ops_on_m[idx2]
                y = LpVariable(f"y_{op1[0]}_{op1[1]}_{op2[0]}_{op2[1]}_{m}", cat=LpBinary)
                M = 1000
                prob += s[op1] >= c[op2] - M * (1 - y) - M * (2 - x[op1 + (m,)] - x[op2 + (m,)])
                prob += s[op2] >= c[op1] - M * y - M * (2 - x[op1 + (m,)] - x[op2 + (m,)])
    
    prob.solve(PULP_CBC_CMD(msg=False))
    
    milp_schedule = {m: [] for m in machines}
    for j, i in ops:
        for m in machines:
            if (j, i, m) in x and x[j, i, m].value() and x[j, i, m].value() > 0.5:
                milp_schedule[m].append({
                    'job': j, 'op': i+1,
                    'start': s[j, i].value(),
                    'end': c[j, i].value()
                })
    
    for m in machines:
        milp_schedule[m].sort(key=lambda x: x['start'])
    
    return Cmax.value(), milp_schedule

milp_makespan, milp_schedule = solve_milp()

print(f"\nRESULTS:")
print(f"Policy Iteration makespan: {final_makespan}")
print(f"MILP optimal makespan: {milp_makespan}")
print(f"Gap: {final_makespan - milp_makespan}")
print(f"Operations scheduled by PI: {sum(len(schedule[m]) for m in machines)}/{total_operations}")
print(f"Operations scheduled by MILP: {sum(len(milp_schedule[m]) for m in machines)}/{total_operations}")
