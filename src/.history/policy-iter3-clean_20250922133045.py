# Compact Event-Driven No-Queue MDP for FJSP Policy Iteration
# Observation: (ready_jobs, machine_idle, processing_times_ready_ops)

import itertools
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpBinary

# Problem definition
JOBS_DATA = {
    1: [
        {'proc_times': {'M1': 2, 'M2': 4, 'M3': 3}},
        {'proc_times': {'M1': 3, 'M2': 2, 'M3': 4}}
    ],
    2: [
        {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2}},
        {'proc_times': {'M1': 2, 'M2': 3, 'M3': 4}}
    ]
}
MACHINES = ['M1', 'M2', 'M3']
ARRIVALS = {1: 0, 2: 0}

job_ids = list(JOBS_DATA.keys())
machines = MACHINES[:]
total_operations = sum(len(ops) for ops in JOBS_DATA.values())

def compact_observation(next_ops, machine_next_free, job_ready_time, current_time):
    """
    Create compact observation: (ready_jobs, machine_idle, proc_times_for_ready_ops)
    - ready_jobs: binary vector indicating which jobs have ready operations now
    - machine_idle: binary vector indicating which machines are idle now  
    - proc_times_for_ready_ops: processing times for ready operations on each machine
    """
    ready_jobs = []
    proc_times = []
    
    for j_idx, job_id in enumerate(job_ids):
        op_idx = next_ops[j_idx]
        job_has_ready_op = (op_idx < len(JOBS_DATA[job_id]) and 
                           job_ready_time[j_idx] <= current_time)
        ready_jobs.append(1 if job_has_ready_op else 0)
        
        if job_has_ready_op:
            operation = JOBS_DATA[job_id][op_idx]
            for machine in machines:
                proc_times.append(operation['proc_times'].get(machine, 0))
        else:
            for machine in machines:
                proc_times.append(0)
    
    machine_idle = [1 if machine_next_free[m_idx] <= current_time else 0 
                    for m_idx in range(len(machines))]
    
    return (tuple(ready_jobs), tuple(machine_idle), tuple(proc_times))

def is_terminal(obs):
    """Check if observation represents terminal state (no ready jobs)"""
    ready_jobs, _, _ = obs
    return sum(ready_jobs) == 0

def compute_makespan(machine_next_free, job_ready_time):
    vals = []
    if machine_next_free:
        vals.append(max(machine_next_free))
    if job_ready_time:
        vals.append(max(job_ready_time))
    return max(vals) if vals else 0

# Build MDP via BFS enumeration
print("Building compact event-driven no-queue MDP...")
state_to_id = {}
id_to_state = []
internal_states = {}  # state_id -> representative internal state
transitions = defaultdict(list)

def add_state(obs, internal_state):
    if obs not in state_to_id:
        sid = len(id_to_state)
        state_to_id[obs] = sid
        id_to_state.append(obs)
        internal_states[sid] = internal_state
    return state_to_id[obs]

# BFS from initial state
init_next_ops = [0, 0]
init_machine_next_free = [0, 0, 0]  
init_job_ready = [0, 0]
init_time = 0

init_obs = compact_observation(init_next_ops, init_machine_next_free, init_job_ready, init_time)
init_id = add_state(init_obs, (init_next_ops, init_machine_next_free, init_job_ready, init_time))

queue = deque([(init_next_ops, init_machine_next_free, init_job_ready, init_time)])
visited_internal = set()

while queue:
    next_ops, machine_next_free, job_ready_time, current_time = queue.popleft()
    internal_key = (tuple(next_ops), tuple(machine_next_free), tuple(job_ready_time), current_time)
    
    if internal_key in visited_internal:
        continue
    visited_internal.add(internal_key)
    
    obs = compact_observation(next_ops, machine_next_free, job_ready_time, current_time)
    sid = add_state(obs, (next_ops[:], machine_next_free[:], job_ready_time[:], current_time))
    
    finished_ops = sum(next_ops)
    if finished_ops >= total_operations:
        continue  # Terminal state
    
    # Find legal immediate scheduling actions
    legal_actions = []
    for j_idx, job_id in enumerate(job_ids):
        op_idx = next_ops[j_idx]
        if op_idx >= len(JOBS_DATA[job_id]):
            continue
        if job_ready_time[j_idx] <= current_time:  # job ready
            operation = JOBS_DATA[job_id][op_idx]
            for m_idx, machine in enumerate(machines):
                if (machine_next_free[m_idx] <= current_time and  # machine idle
                    machine in operation['proc_times']):  # machine can process
                    legal_actions.append((j_idx, op_idx, m_idx))
    
    current_makespan = compute_makespan(machine_next_free, job_ready_time)
    
    if legal_actions:
        # Add scheduling transitions
        for j_idx, op_idx, m_idx in legal_actions:
            job_id = job_ids[j_idx]
            machine = machines[m_idx]
            proc_time = JOBS_DATA[job_id][op_idx]['proc_times'][machine]
            
            # Execute action
            new_next_ops = next_ops[:]
            new_next_ops[j_idx] += 1
            new_machine_next_free = machine_next_free[:]
            new_machine_next_free[m_idx] = current_time + proc_time
            new_job_ready = job_ready_time[:]
            new_job_ready[j_idx] = current_time + proc_time
            
            new_obs = compact_observation(new_next_ops, new_machine_next_free, new_job_ready, current_time)
            new_sid = add_state(new_obs, (new_next_ops, new_machine_next_free, new_job_ready, current_time))
            
            # Reward: -(makespan_increase + small_step_penalty)
            new_makespan = compute_makespan(new_machine_next_free, new_job_ready)
            reward = -(new_makespan - current_makespan) - 0.01
            
            transitions[sid].append(((j_idx, op_idx, m_idx), new_sid, reward))
            
            # Add to queue if not seen
            new_internal_key = (tuple(new_next_ops), tuple(new_machine_next_free), tuple(new_job_ready), current_time)
            if new_internal_key not in visited_internal:
                queue.append((new_next_ops, new_machine_next_free, new_job_ready, current_time))
    else:
        # No immediate actions possible - add WAIT transition to next event
        candidates = []
        for t in machine_next_free:
            if t > current_time:
                candidates.append(t)
        for t in job_ready_time:
            if t > current_time:
                candidates.append(t)
        
        if candidates:
            next_time = min(candidates)
            new_obs = compact_observation(next_ops, machine_next_free, job_ready_time, next_time)
            new_sid = add_state(new_obs, (next_ops[:], machine_next_free[:], job_ready_time[:], next_time))
            
            new_makespan = compute_makespan(machine_next_free, job_ready_time)
            reward = -(new_makespan - current_makespan) - 0.01
            
            transitions[sid].append(('WAIT', new_sid, reward))
            
            # Add to queue
            new_internal_key = (tuple(next_ops), tuple(machine_next_free), tuple(job_ready_time), next_time)
            if new_internal_key not in visited_internal:
                queue.append((next_ops, machine_next_free, job_ready_time, next_time))

num_states = len(id_to_state)
num_transitions = sum(len(v) for v in transitions.values())
print(f"MDP built: {num_states} states, {num_transitions} transitions")

# Show state breakdown
print("\nState breakdown by ready_jobs:")
ready_counts = {}
for obs in id_to_state:
    ready_jobs = obs[0]
    ready_counts[ready_jobs] = ready_counts.get(ready_jobs, 0) + 1
for rj, count in sorted(ready_counts.items()):
    print(f"  ready_jobs={rj}: {count} states")

# Policy Iteration
print("\nRunning Policy Iteration...")
gamma = 1.0

# Initialize policy
policy = {}
for sid in range(num_states):
    acts = transitions.get(sid, [])
    if acts:
        # Choose first non-WAIT action if available
        policy[sid] = next((a for a, _, _ in acts if a != 'WAIT'), acts[0][0])
    else:
        policy[sid] = None

V = [0.0] * num_states

def policy_evaluation(V, tol=1e-9, max_iter=1000):
    for iteration in range(max_iter):
        delta = 0.0
        for sid in range(num_states):
            action = policy[sid]
            if action is None:
                continue
            # Find transition for this action
            transition = next(((nsid, r) for a, nsid, r in transitions[sid] if a == action), None)
            if transition:
                nsid, reward = transition
                new_v = reward + gamma * V[nsid]
                delta = max(delta, abs(new_v - V[sid]))
                V[sid] = new_v
        
        if iteration % 100 == 0:
            print(f"  Evaluation iter {iteration}: max_delta = {delta:.6e}")
        
        if delta < tol:
            print(f"  Policy evaluation converged in {iteration + 1} iterations")
            return V
    print(f"  Policy evaluation stopped at max iterations")
    return V

def policy_improvement(V):
    stable = True
    changes = 0
    for sid in range(num_states):
        acts = transitions.get(sid, [])
        if not acts:
            continue
        
        best_action = None
        best_value = float('-inf')
        for action, nsid, reward in acts:
            value = reward + gamma * V[nsid]
            if value > best_value:
                best_value = value
                best_action = action
        
        if policy[sid] != best_action:
            stable = False
            changes += 1
            policy[sid] = best_action
    
    print(f"  Policy improvement: {changes} changes")
    return stable

# Run Policy Iteration
iteration = 0
while iteration < 50:
    iteration += 1
    print(f"\nPolicy Iteration {iteration}:")
    V = policy_evaluation(V)
    stable = policy_improvement(V)
    if stable:
        print("Policy converged!")
        break

print(f"\nOptimal value: V(init) = {V[init_id]:.3f}")
print(f"Predicted optimal makespan: {-V[init_id]:.3f}")

# Execute optimal policy
print("\nExecuting optimal policy...")
cur_state = internal_states[init_id]
next_ops, machine_next_free, job_ready_time, current_time = [x[:] if isinstance(x, list) else x for x in cur_state]
schedule = {m: [] for m in machines}
step = 0

while step < 20:  # Safety limit
    obs = compact_observation(next_ops, machine_next_free, job_ready_time, current_time)
    sid = state_to_id[obs]
    action = policy[sid]
    
    if action is None or sum(next_ops) >= total_operations:
        break
    
    step += 1
    
    if action == 'WAIT':
        # Advance time to next event
        candidates = [t for t in machine_next_free if t > current_time] + [t for t in job_ready_time if t > current_time]
        if candidates:
            current_time = min(candidates)
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

actual_makespan = max(max(machine_next_free), max(job_ready_time))
print(f"Policy execution completed in {step} steps")
print(f"Actual makespan: {actual_makespan}")

# MILP comparison
def solve_milp():
    print("\nSolving MILP for comparison...")
    prob = LpProblem("FJSP", LpMinimize)
    
    ops = [(j, i) for j in job_ids for i in range(len(JOBS_DATA[j]))]
    
    x = {}  # x[op, machine] = 1 if op assigned to machine
    s = {}  # s[op] = start time
    c = {}  # c[op] = completion time
    
    for op in ops:
        j, i = op
        s[op] = LpVariable(f"s_{j}_{i}", lowBound=0)
        c[op] = LpVariable(f"c_{j}_{i}", lowBound=0)
        for m in machines:
            if m in JOBS_DATA[j][i]['proc_times']:
                x[op, m] = LpVariable(f"x_{j}_{i}_{m}", cat=LpBinary)
    
    Cmax = LpVariable("Cmax", lowBound=0)
    prob += Cmax
    
    # Each operation assigned to exactly one machine
    for j, i in ops:
        eligible = [m for m in machines if m in JOBS_DATA[j][i]['proc_times']]
        prob += lpSum(x[j, i, m] for m in eligible) == 1
    
    # Completion time constraints
    for j, i in ops:
        eligible = [m for m in machines if m in JOBS_DATA[j][i]['proc_times']]
        prob += c[j, i] == s[j, i] + lpSum(x[j, i, m] * JOBS_DATA[j][i]['proc_times'][m] for m in eligible)
    
    # Precedence constraints
    for j in job_ids:
        for i in range(1, len(JOBS_DATA[j])):
            prob += s[j, i] >= c[j, i-1]
    
    # Makespan constraints
    for op in ops:
        prob += Cmax >= c[op]
    
    # Disjunctive constraints (no two operations on same machine at same time)
    for m in machines:
        ops_on_m = [(j, i) for j, i in ops if m in JOBS_DATA[j][i]['proc_times']]
        for idx1 in range(len(ops_on_m)):
            for idx2 in range(idx1 + 1, len(ops_on_m)):
                op1, op2 = ops_on_m[idx1], ops_on_m[idx2]
                y = LpVariable(f"y_{op1[0]}_{op1[1]}_{op2[0]}_{op2[1]}_{m}", cat=LpBinary)
                M = 1000
                prob += s[op1] >= c[op2] - M * (1 - y) - M * (2 - x[op1, m] - x[op2, m])
                prob += s[op2] >= c[op1] - M * y - M * (2 - x[op1, m] - x[op2, m])
    
    prob.solve(PULP_CBC_CMD(msg=False))
    
    milp_schedule = {m: [] for m in machines}
    for j, i in ops:
        for m in machines:
            if (j, i, m) in x and x[j, i, m].value() and x[j, i, m].value() > 0.5:
                start = s[j, i].value()
                end = c[j, i].value()
                milp_schedule[m].append({'job': j, 'op': i+1, 'start': start, 'end': end})
    
    for m in machines:
        milp_schedule[m].sort(key=lambda x: x['start'])
    
    return Cmax.value(), milp_schedule

milp_makespan, milp_schedule = solve_milp()
print(f"MILP optimal makespan: {milp_makespan}")

# Print schedules
print(f"\nPolicy-derived schedule (makespan {actual_makespan}):")
for m in machines:
    for task in schedule[m]:
        print(f"  {m}: J{task['job']}-Op{task['op']} [{task['start']:.1f}, {task['end']:.1f}]")

print(f"\nMILP optimal schedule (makespan {milp_makespan}):")  
for m in machines:
    for task in milp_schedule[m]:
        print(f"  {m}: J{task['job']}-Op{task['op']} [{task['start']:.1f}, {task['end']:.1f}]")

print(f"\nSUMMARY:")
print(f"Compact MDP states: {num_states}")
print(f"Policy iteration makespan: {actual_makespan}")
print(f"MILP optimal makespan: {milp_makespan}")
print(f"Gap: {actual_makespan - milp_makespan}")
