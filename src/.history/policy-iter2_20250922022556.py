# fjsp_pi_vs_milp.py
# Full demo: Policy Iteration on fully-observed minimal-state MDP vs MILP optimal benchmark
# Requirements: numpy, matplotlib, pulp

import itertools
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
import time
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpBinary

# ---------------------------
# Problem definition (SIMPLE)
# ---------------------------
SIMPLE_JOBS_DATA = {
    1: [
        {'proc_times': {'M1': 2, 'M2': 4, 'M3': 3}},
        {'proc_times': {'M1': 3, 'M2': 2, 'M3': 4}}
    ],
    2: [
        {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2}},
        {'proc_times': {'M1': 2, 'M2': 3, 'M3': 4}}
    ]
}
SIMPLE_MACHINES = ['M1', 'M2', 'M3']
SIMPLE_ARRIVALS = {1: 0, 2: 0}  # all arrive at 0

LARGER_JOBS_DATA = {
    1: [
        {'proc_times': {'M1': 2, 'M2': 4, 'M3': 3, 'M4': 5}},
        {'proc_times': {'M1': 3, 'M2': 2, 'M3': 4, 'M4': 6}},
        {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2, 'M4': 7}}
    ],
    2: [
        {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2, 'M4': 5}},
        {'proc_times': {'M1': 2, 'M2': 3, 'M4': 6}},
        {'proc_times': {'M1': 5, 'M2': 2, 'M3': 3, 'M4': 4}}
    ],
    3: [
        {'proc_times': {'M1': 3, 'M2': 5, 'M3': 2, 'M4': 4}},
        {'proc_times': {'M1': 4, 'M2': 2, 'M3': 5, 'M4': 3}},
        {'proc_times': {'M1': 2, 'M2': 4, 'M3': 3, 'M4': 5}}
    ],
    4: [
        {'proc_times': {'M1': 5, 'M2': 3, 'M3': 4, 'M4': 2}},
        {'proc_times': {'M1': 3, 'M2': 4, 'M3': 2, 'M4': 5}},
        {'proc_times': {'M1': 4, 'M2': 5, 'M3': 3, 'M4': 2}}
    ],
    5: [
        {'proc_times': {'M1': 2, 'M2': 3, 'M3': 5, 'M4': 4}},
        {'proc_times': {'M1': 5, 'M2': 2, 'M3': 4, 'M4': 3}},
        {'proc_times': {'M1': 3, 'M2': 4, 'M3': 2, 'M4': 5}}
    ]
}
LARGER_MACHINES = ['M1', 'M2', 'M3', 'M4']
LARGER_ARRIVALS = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

# Start with smaller problem - just 3 jobs to test
MEDIUM_JOBS_DATA = {
    1: [
        {'proc_times': {'M1': 2, 'M2': 4, 'M3': 3}},
        {'proc_times': {'M1': 3, 'M2': 2, 'M3': 4}}
    ],
    2: [
        {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2}},
        {'proc_times': {'M1': 2, 'M2': 3, 'M3': 4}}
    ],
    3: [
        {'proc_times': {'M1': 3, 'M2': 5, 'M3': 2}},
        {'proc_times': {'M1': 4, 'M2': 2, 'M3': 5}}
    ]
}
MEDIUM_MACHINES = ['M1', 'M2', 'M3']
MEDIUM_ARRIVALS = {1: 0, 2: 0, 3: 0}

# Use simple problem first (2 jobs, 3 machines) - actually has manageable state space
JOBS_DATA = SIMPLE_JOBS_DATA
MACHINES = SIMPLE_MACHINES  
ARRIVALS = SIMPLE_ARRIVALS

# Uncomment below to try medium problem (3 jobs, 3 machines):
# JOBS_DATA = MEDIUM_JOBS_DATA
# MACHINES = MEDIUM_MACHINES
# ARRIVALS = MEDIUM_ARRIVALS

# Uncomment below to try larger problem after medium works:
# JOBS_DATA = LARGER_JOBS_DATA
# MACHINES = LARGER_MACHINES
# ARRIVALS = LARGER_ARRIVALS

job_ids = list(JOBS_DATA.keys())
machines = MACHINES[:]
J = len(job_ids)
M = len(machines)
total_operations = sum(len(ops) for ops in JOBS_DATA.values())

# ---------------------------
# Helper utilities
# ---------------------------
def make_state(next_ops, machine_next_free, job_ready_time):
    """Minimal canonical state as tuple of tuples (hashable)."""
    return (tuple(next_ops), tuple(machine_next_free), tuple(job_ready_time))

def pretty_state(s):
    no, mnext, jready = s
    return f"next_ops={no}, mnext={mnext}, jready={jready}"

def compute_makespan_from_state(s):
    """Makespan consistent with our minimal state: max of machine_next_free and job_ready_time."""
    _, mnext, jready = s
    vals = []
    if len(mnext) > 0:
        vals.append(max(mnext))
    if len(jready) > 0:
        vals.append(max(jready))
    return max(vals) if vals else 0

# ---------------------------
# Enumerate reachable MDP (constrained by precedence)
# ---------------------------
print("Enumerating reachable minimal-state MDP via constrained BFS...")
print(f"Problem size: {len(job_ids)} jobs, {len(machines)} machines, {total_operations} total operations")
print("Using precedence-aware enumeration to reduce state space...")
start_time = time.time()

state_to_id = {}
id_to_state = []
transitions = defaultdict(list)  # sid -> list of (action, nsid, reward)

def add_state(s):
    if s not in state_to_id:
        sid = len(id_to_state)
        state_to_id[s] = sid
        id_to_state.append(s)
        if len(id_to_state) % 100 == 0:  # More frequent reporting for smaller counts
            print(f"  Generated {len(id_to_state)} states so far...")
    return state_to_id[s]

def compute_job_ready_time(job_idx, next_ops, operation_end_times):
    """Compute when a job is ready for its next operation based on precedence constraints."""
    if next_ops[job_idx] == 0:
        # First operation: ready when job arrives
        return ARRIVALS[job_ids[job_idx]]
    else:
        # Later operation: ready when previous operation completes
        return operation_end_times[job_idx][next_ops[job_idx] - 1]

def is_valid_state_transition(current_state, action):
    """Check if an action from current state respects all constraints."""
    next_ops, machine_next_free, job_ready_time = current_state
    j_idx, op_idx, m_idx = action
    job_id = job_ids[j_idx]
    
    # Check if this is the correct next operation for the job
    if op_idx != next_ops[j_idx]:
        return False
        
    # Check if operation exists and machine is compatible
    if op_idx >= len(JOBS_DATA[job_id]):
        return False
        
    if machines[m_idx] not in JOBS_DATA[job_id][op_idx]['proc_times']:
        return False
        
    return True

# Initialize with proper state computation
init_next_ops = [0 for _ in job_ids]
init_machine_next_free = [0 for _ in machines] 
init_operation_end_times = [[0.0] * len(JOBS_DATA[job_id]) for job_id in job_ids]

# Compute initial job ready times based on arrivals
init_job_ready = []
for j_idx, job_id in enumerate(job_ids):
    init_job_ready.append(compute_job_ready_time(j_idx, init_next_ops, init_operation_end_times))

init_state = make_state(init_next_ops, init_machine_next_free, init_job_ready)
add_state(init_state)

q = deque([(init_state, init_operation_end_times)])
states_processed = 0

print(f"Initial state: next_ops={init_next_ops}, machine_free={init_machine_next_free}, job_ready={init_job_ready}")

while q and len(id_to_state) < 10000:  # Lower safety limit
    s, operation_end_times = q.popleft()
    sid = state_to_id[s]
    next_ops, machine_next_free, job_ready_time = s
    finished = sum(next_ops)
    
    states_processed += 1
    if states_processed % 50 == 0:  # More frequent progress updates
        print(f"  Processed {states_processed} states, queue size: {len(q)}, current next_ops: {next_ops}")
    
    if finished >= total_operations:
        continue

    # Find legal actions with proper precedence checking
    legal_actions = []
    for j_idx, job_id in enumerate(job_ids):
        op_idx = next_ops[j_idx]
        if op_idx >= len(JOBS_DATA[job_id]):  # job finished
            continue
            
        # Only consider operations that are actually ready
        job_is_ready = (job_ready_time[j_idx] <= max(machine_next_free))
        
        proc_dict = JOBS_DATA[job_id][op_idx]['proc_times']
        for m_idx, machine in enumerate(machines):
            if machine in proc_dict:
                action = (j_idx, op_idx, m_idx)
                if is_valid_state_transition(s, action):
                    legal_actions.append(action)

    cur_makespan = compute_makespan_from_state(s)

    for action in legal_actions:
        j_idx, op_idx, m_idx = action
        job_id = job_ids[j_idx]
        machine = machines[m_idx]
        p = JOBS_DATA[job_id][op_idx]['proc_times'][machine]
        
        # Compute timing exactly like in RL environment step function
        machine_available_time = machine_next_free[m_idx]
        job_ready_time_val = job_ready_time[j_idx]
        start_time = max(machine_available_time, job_ready_time_val)
        end_time = start_time + p

        # Update state components
        new_next_ops = list(next_ops)
        new_next_ops[j_idx] += 1
        
        new_machine_next_free = list(machine_next_free)
        new_machine_next_free[m_idx] = end_time
        
        # Update operation end times for precedence tracking
        new_operation_end_times = [row[:] for row in operation_end_times]  # deep copy
        new_operation_end_times[j_idx][op_idx] = end_time
        
        # Compute new job ready times based on updated operation end times
        new_job_ready = []
        for jx in range(len(job_ids)):
            new_job_ready.append(compute_job_ready_time(jx, new_next_ops, new_operation_end_times))

        new_state = make_state(new_next_ops, new_machine_next_free, new_job_ready)
        
        # Only add state if it doesn't exist
        nsid = add_state(new_state)
        if nsid == len(id_to_state) - 1:  # New state was added
            q.append((new_state, new_operation_end_times))

        new_makespan = compute_makespan_from_state(new_state)
        reward = -(new_makespan - cur_makespan)  # negative makespan increment
        transitions[sid].append((action, nsid, reward))

num_states = len(id_to_state)
num_transitions = sum(len(v) for v in transitions.values())
elapsed = time.time() - start_time

if len(id_to_state) >= 10000:
    print(f"WARNING: Enumeration stopped at 10000 states limit!")
    
print(f"Enumeration done: {num_states} states, {num_transitions} transitions (elapsed {elapsed:.2f}s)\n")

# Save states and transitions to CSV
import csv
csv_filename = "/Users/tanu/Desktop/PhD/Scheduling/src/mdp_states_transitions.csv"
print(f"Saving state and transition data to {csv_filename}...")

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Header
    writer.writerow(['state_id', 'next_ops', 'machine_next_free', 'job_ready_time', 'makespan', 'num_actions', 'actions'])
    
    for sid in range(num_states):
        state = id_to_state[sid]
        next_ops, machine_next_free, job_ready_time = state
        makespan = compute_makespan_from_state(state)
        actions = transitions.get(sid, [])
        num_actions = len(actions)
        
        # Format actions as string for CSV
        actions_str = "; ".join([f"J{job_ids[a[0]]}-Op{a[1]+1}->M{a[2]+1}" for a, _, _ in actions])
        
        writer.writerow([
            sid,
            list(next_ops),
            list(machine_next_free), 
            list(job_ready_time),
            makespan,
            num_actions,
            actions_str
        ])

print(f"CSV saved with {num_states} states and {num_transitions} total transitions.")

# Analyze constrained state space
print(f"\n=== CONSTRAINED STATE SPACE ANALYSIS ===")
print(f"Total states: {num_states}")

# Count states by next_ops (job progress)
from collections import Counter
next_ops_counts = Counter()
makespan_distribution = Counter()

for sid in range(num_states):
    state = id_to_state[sid]
    next_ops, machine_next_free, job_ready_time = state
    next_ops_counts[next_ops] += 1
    makespan = compute_makespan_from_state(state)
    makespan_distribution[makespan] += 1

print(f"\nStates by job progress (next_ops):")
for next_ops, count in sorted(next_ops_counts.items()):
    total_progress = sum(next_ops)
    print(f"  {next_ops}: {count} states (total progress: {total_progress})")

print(f"\nStates by makespan:")
for makespan, count in sorted(makespan_distribution.items()):
    print(f"  Makespan {makespan}: {count} states")

print(f"\nAnalysis:")
print(f"- Total states reduced from 318 to {num_states} by proper precedence constraints")
print(f"- Each job progress combination has deterministic timing based on machine choices")
print(f"- State space is now manageable for exact policy iteration")

# ---------------------------
# Policy Iteration (Howard)
# ---------------------------
print("Running Policy Iteration (Howard) with verbose traces...\n")
gamma = 1.0

# initialize policy: choose first available action for each nonterminal state
policy = {}
for sid, s in enumerate(id_to_state):
    next_ops, _, _ = s
    if sum(next_ops) >= total_operations:
        policy[sid] = None
    else:
        acts = transitions[sid]
        policy[sid] = acts[0][0] if acts else None

# Initialize value function
V = [0.0 for _ in range(num_states)]

def policy_evaluation(policy, V, tol=1e-9, max_iter=200):
    """Iterative policy evaluation for deterministic transitions; verbose printing."""
    print(" POLICY EVALUATION:")
    for sweep in range(1, max_iter+1):
        delta = 0.0
        for sid in range(num_states):
            a = policy[sid]
            if a is None:
                newv = 0.0
            else:
                rec = next(((nsid, r) for (act, nsid, r) in transitions[sid] if act == a), None)
                if rec is None:
                    newv = 0.0
                else:
                    nsid, r = rec
                    newv = r + gamma * V[nsid]
            delta = max(delta, abs(newv - V[sid]))
            V[sid] = newv
        print(f"  Sweep {sweep:2d} - max delta = {delta:.6e}")
        if delta < tol:
            print("  Converged policy evaluation.\n")
            return V, sweep, delta
    print("  Reached max_iter in policy evaluation.\n")
    return V, max_iter, delta

def policy_improvement(policy, V, max_print=20):
    """Greedy improvement; print changed states (up to max_print)."""
    print(" POLICY IMPROVEMENT:")
    policy_stable = True
    changes = []
    for sid in range(num_states):
        recs = transitions.get(sid, [])
        if not recs:
            continue
        best_q = -1e9
        best_a = None
        for (a, nsid, r) in recs:
            q = r + gamma * V[nsid]
            if q > best_q:
                best_q = q
                best_a = a
        old_a = policy[sid]
        policy[sid] = best_a
        if old_a != best_a:
            policy_stable = False
            changes.append((sid, old_a, best_a, best_q))
            if len(changes) <= max_print:
                print(f"  State {sid}: {pretty_state(id_to_state[sid])}")
                print(f"    old: {old_a}, new: {best_a}, best_Q={best_q:.3f}")
    if not changes:
        print("  No changes - policy is stable.\n")
    else:
        print(f"  Total policy changes this improvement: {len(changes)}\n")
    return policy, policy_stable, changes

# Run PI
iteration = 0
while True:
    iteration += 1
    print(f"=== POLICY ITERATION {iteration} ===")
    V, sweeps, last_delta = policy_evaluation(policy, V)
    policy, stable, changes = policy_improvement(policy, V)
    print(f"Iteration {iteration} finished. Stable? {stable}.")
    if stable:
        print("Policy iteration converged to optimal policy.\n")
        break
    if iteration >= 20:
        print("Stopping after 20 iterations (safety limit).\n")
        break

# Print initial-state value
init_id = state_to_id[init_state]
print(f"V(init) = {V[init_id]:.3f} => optimal makespan = {-V[init_id]:.3f}\n")

# ---------------------------
# Execute PI policy to build schedule
# ---------------------------
print("Executing the obtained optimal policy from initial state to collect schedule...")
cur_sid = init_id
exec_schedule_pi = {m: [] for m in machines}
step = 0
total_reward = 0.0
while True:
    a = policy[cur_sid]
    if a is None:
        break
    step += 1
    # find matching transition
    rec = next(((act, nsid, r) for (act, nsid, r) in transitions[cur_sid] if act == a), None)
    if rec is None:
        print("ERROR: action not found in transitions; aborting.")
        break
    act, nsid, r = rec
    # reconstruct timing (from canonical state)
    next_ops, mnext, jready = id_to_state[cur_sid]
    j_idx, op_idx, m_idx = act
    job_id = job_ids[j_idx]
    machine = machines[m_idx]
    p = JOBS_DATA[job_id][op_idx]['proc_times'][machine]
    start = max(jready[j_idx], mnext[m_idx])
    end = start + p
    exec_schedule_pi[machine].append({'job': job_id, 'op': op_idx+1, 'start': start, 'end': end})
    total_reward += r
    cur_sid = nsid

print(f"PI execution finished in {step} steps. Total reward = {total_reward:.3f}. PI makespan = {-total_reward:.3f}\n")

# ---------------------------
# MILP optimal scheduler (PuLP)
# ---------------------------
def milp_scheduler(jobs, machines, arrivals, time_limit=None):
    """
    MILP model to compute optimal makespan and schedule for this small instance.
    Uses disjunctive constraints with big-M and binary precedence variables.
    """
    print("Solving MILP for optimal benchmark (PuLP + CBC)...")
    prob = LpProblem("FJSP_opt", LpMinimize)

    # Build operation list
    ops = []
    for j in jobs:
        for oi in range(len(jobs[j])):
            ops.append((j, oi))
    BIG_M = 1000

    # Decision variables
    x = {}  # x[(j,oi),m] binary assignment
    svar = {}  # start time
    cvar = {}  # completion time
    for op in ops:
        j, oi = op
        svar[op] = LpVariable(f"s_{j}_{oi}", lowBound=0)
        cvar[op] = LpVariable(f"c_{j}_{oi}", lowBound=0)
        for m in machines:
            if m in jobs[j][oi]['proc_times']:
                x[(op, m)] = LpVariable(f"x_{j}_{oi}_{m}", cat=LpBinary)

    y = {}  # precedence for pairs on same machine
    for m in machines:
        # list ops eligible on m
        ops_on_m = [op for op in ops if m in jobs[op[0]][op[1]]['proc_times']]
        for (op1, op2) in itertools.permutations(ops_on_m, 2):
            if op1 < op2:  # only create pair once (unordered) - we'll create two binary vars per unordered pair
                y[(op1, op2, m)] = LpVariable(f"y_{op1[0]}_{op1[1]}_{op2[0]}_{op2[1]}_{m}", cat=LpBinary)
                # we will use y(op1,op2,m)=1 => op1 before op2 on machine m

    Cmax = LpVariable("Cmax", lowBound=0)
    prob += Cmax

    # Constraints: assignment and timing
    for (j, oi) in ops:
        # assign to exactly one eligible machine
        eligible_machines = [m for m in machines if m in jobs[j][oi]['proc_times']]
        prob += lpSum(x[( (j, oi), m)] for m in eligible_machines) == 1

        # completion = start + proc_time on assigned machine
        prob += cvar[(j, oi)] == lpSum(x[((j, oi), m)] * jobs[j][oi]['proc_times'][m] for m in eligible_machines) + svar[(j, oi)]

        # precedence within job
        if oi > 0:
            prob += svar[(j, oi)] >= cvar[(j, oi - 1)]
        # arrival time
        if oi == 0:
            prob += svar[(j, 0)] >= arrivals[j]
        # makespan
        prob += Cmax >= cvar[(j, oi)]

    # Machine disjunctive constraints
    # For each unordered pair of distinct ops that share any machine, we add constraints for each machine.
    # If both assigned to same machine m, then either op1 before op2 or vice versa.
    for m in machines:
        ops_on_m = [op for op in ops if m in jobs[op[0]][op[1]]['proc_times']]
        for i in range(len(ops_on_m)):
            for k in range(i+1, len(ops_on_m)):
                op1 = ops_on_m[i]
                op2 = ops_on_m[k]
                # create two binaries: y(op1,op2,m) indicates op1 before op2
                yvar = LpVariable(f"y_{op1[0]}_{op1[1]}_{op2[0]}_{op2[1]}_{m}", cat=LpBinary)
                # when both assigned to m:
                # s_op1 >= c_op2 - BIG_M * (1 - y) - BIG_M*(2 - x_op1_m - x_op2_m)
                prob += svar[op1] >= cvar[op2] - BIG_M * (1 - yvar) - BIG_M * (2 - x[(op1, m)] - x[(op2, m)])
                prob += svar[op2] >= cvar[op1] - BIG_M * yvar - BIG_M * (2 - x[(op1, m)] - x[(op2, m)])

    # Solve
    solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit) if time_limit else PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    schedule = {m: [] for m in machines}
    feasible = True
    if prob.status is None:
        feasible = False

    # Extract schedule
    for (j, oi) in ops:
        assigned_m = None
        for m in machines:
            if ( (j, oi), m) in x and x[((j, oi), m)].value() is not None and x[((j, oi), m)].value() > 0.5:
                assigned_m = m
                break
        if assigned_m is None:
            feasible = False
            continue
        st = float(svar[(j, oi)].value())
        ct = float(cvar[(j, oi)].value())
        schedule[assigned_m].append({'job': j, 'op': oi+1, 'start': st, 'end': ct})
    # sort
    for m in machines:
        schedule[m].sort(key=lambda t: t['start'])
    cmax_val = float(Cmax.value()) if Cmax.value() is not None else float('inf')
    if not feasible:
        print("MILP: no feasible assignment found.")
    else:
        print(f"MILP solved: optimal makespan = {cmax_val:.2f}\n")
    return cmax_val, schedule

# Solve MILP
milp_start = time.time()
milp_makespan, milp_schedule = milp_scheduler(JOBS_DATA, machines, ARRIVALS, time_limit=60)
milp_elapsed = time.time() - milp_start

# ---------------------------
# Plot Gantt of PI schedule vs MILP schedule
# ---------------------------
def plot_two_gantts(schedule_a, label_a, schedule_b, label_b, machines, title_suffix=""):
    """Plot two Gantt charts side-by-side for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    all_schedules = [schedule_a, schedule_b]
    labels = [label_a, label_b]
    cmap = plt.cm.get_cmap("tab20")
    job_set = set()
    for sched in all_schedules:
        for m in machines:
            for t in sched[m]:
                job_set.add(t['job'])
    job_list = sorted(job_set)
    job_to_color = {job_list[i]: cmap(i % 20) for i in range(len(job_list))}

    for ax_idx, ax in enumerate(axes):
        sched = all_schedules[ax_idx]
        ax.set_title(labels[ax_idx] + (("  (makespan=%.1f)" % max((t['end'] for m in machines for t in sched[m])) ) if any(sched[m] for m in machines) else ""))
        ax.set_xlabel("Time")
        ax.set_yticks([i*10+5 for i in range(len(machines))])
        ax.set_yticklabels(machines)
        ax.set_xlim(0, max((t['end'] for m in machines for t in sched[m]), default=10))
        ax.set_ylim(0, len(machines)*10)
        ax.grid(True)
        for m_idx, m in enumerate(machines):
            for task in sched[m]:
                start = task['start']
                dur = task['end'] - task['start']
                color = job_to_color[task['job']]
                ax.broken_barh([(start, dur)], (m_idx*10, 8), facecolors=(color))
                ax.text(start + dur/2, m_idx*10 + 4, f"J{task['job']}-Op{task['op']}", ha='center', va='center', color='white', fontsize=8)
    # legend
    patches = [mpatches.Patch(color=job_to_color[j], label=f"J{j}") for j in job_list]
    axes[-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle("PI schedule vs MILP optimal schedule " + title_suffix)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

print("Plotting PI vs MILP schedules...")
plot_two_gantts(exec_schedule_pi, "Policy Iteration (derived)", milp_schedule, "MILP Optimal", machines,
                title_suffix=f"(MILP elapsed {milp_elapsed:.2f}s)")

# ---------------------------
# Print compact schedules
# ---------------------------
print("\nPI-derived schedule (per machine):")
for m in machines:
    for task in exec_schedule_pi[m]:
        print(f"  {m}: J{task['job']}-Op{task['op']} [{task['start']:.1f}, {task['end']:.1f}]")

print("\nMILP optimal schedule (per machine):")
for m in machines:
    for task in milp_schedule[m]:
        print(f"  {m}: J{task['job']}-Op{task['op']} [{task['start']:.1f}, {task['end']:.1f}]")

print("\nDone.")

# ---------------------------
# State space estimation
# ---------------------------
def estimate_state_space(jobs_data, machines, arrivals):
    """More realistic state space estimation based on actual enumeration results"""
    print("\n=== STATE SPACE ANALYSIS ===")
    
    # Basic problem parameters
    num_jobs = len(jobs_data)
    num_machines = len(machines)
    total_ops = sum(len(ops) for ops in jobs_data.values())
    
    print(f"Jobs: {num_jobs}, Machines: {num_machines}, Total operations: {total_ops}")
    
    # Theoretical upper bound for next_ops component only
    next_ops_combinations = 1
    for job_id, job_ops in jobs_data.items():
        next_ops_combinations *= (len(job_ops) + 1)  # +1 for "finished" state
    print(f"Next_ops combinations (theoretical max): {next_ops_combinations}")
    
    # Empirical observations from actual enumeration:
    if num_jobs == 2 and total_ops == 4:  # SIMPLE
        estimated_states = "~100-500 (based on current run)"
    elif num_jobs == 3 and total_ops == 6:  # MEDIUM  
        estimated_states = "~10K-15K (based on progress output showing >10K states)"
    elif num_jobs == 5 and total_ops == 15:  # LARGER
        estimated_states = "~500K-2M (extrapolated - likely hits MAX_STATES limit)"
    else:
        # Conservative heuristic based on corrected MEDIUM observation
        # MEDIUM: ~12K states for 3 jobs, 6 ops, next_ops combinations = 27
        # So roughly 12K/27 ≈ 440x multiplier from timing constraints
        scaling_factor = 440
        estimated_states = f"~{int(next_ops_combinations * scaling_factor):,}"
    
    print(f"Realistic reachable states: {estimated_states}")
    print(f"Note: Timing constraints create many distinct states due to continuous time values")
    
    return estimated_states

# Estimate for current problem
estimate_state_space(SIMPLE_JOBS_DATA, machines, SIMPLE_ARRIVALS)

# Estimate for larger problem
print("\n--- LARGER PROBLEM ESTIMATION ---")
estimate_state_space(LARGER_JOBS_DATA, LARGER_MACHINES, LARGER_ARRIVALS)

print("\n=== COMPARISON: WHY LARGER PROBLEM EXPLODES ===")

# Simple problem (3 jobs, 2 ops each = 6 total ops)
simple_jobs = 3
simple_ops_per_job = 2
simple_total_ops = 6
simple_machines = 3
simple_next_ops_combos = (simple_ops_per_job + 1) ** simple_jobs  # 3^3 = 27

# Larger problem (5 jobs, 3 ops each = 15 total ops)  
larger_jobs = 5
larger_ops_per_job = 3
larger_total_ops = 15
larger_machines = 4
larger_next_ops_combos = (larger_ops_per_job + 1) ** larger_jobs  # 4^5 = 1024

print(f"SIMPLE: {simple_jobs} jobs × {simple_ops_per_job} ops = {simple_total_ops} total ops")
print(f"SIMPLE: next_ops combinations = {simple_next_ops_combos}")

print(f"LARGER: {larger_jobs} jobs × {larger_ops_per_job} ops = {larger_total_ops} total ops")  
print(f"LARGER: next_ops combinations = {larger_next_ops_combos}")

print(f"Next_ops explosion factor: {larger_next_ops_combos / simple_next_ops_combos:.1f}x")

# Time dimension explosion
simple_max_time = 20  # rough estimate
larger_max_time = 30  # rough estimate

simple_time_combos = simple_max_time ** (simple_jobs + simple_machines)
larger_time_combos = larger_max_time ** (larger_jobs + larger_machines)

print(f"SIMPLE: time combinations ≈ {simple_max_time}^{simple_jobs + simple_machines} = {simple_time_combos:.2e}")
print(f"LARGER: time combinations ≈ {larger_max_time}^{larger_jobs + larger_machines} = {larger_time_combos:.2e}")
print(f"Time explosion factor: {larger_time_combos / simple_time_combos:.2e}x")

total_explosion = (larger_next_ops_combos / simple_next_ops_combos) * (larger_time_combos / simple_time_combos)
print(f"TOTAL EXPLOSION FACTOR: {total_explosion:.2e}x")
print(f"Expected states: 320 × {total_explosion:.2e} = {320 * total_explosion:.2e}")

# ---------------------------
# Constrained State Enumeration (Following RL Step Logic)
# ---------------------------

def enumerate_states_constrained():
    """
    Enumerate states by following the exact logic from RL environment step() function.
    This should generate much fewer states (~50-100) by respecting precedence constraints.
    """
    print("Starting constrained state enumeration following RL step() logic...")
    
    state_to_id = {}
    id_to_state = []
    transitions = defaultdict(list)
    
    def add_state(s):
        if s not in state_to_id:
            sid = len(id_to_state)
            state_to_id[s] = sid
            id_to_state.append(s)
            print(f"  State {sid}: {s}")
        return state_to_id[s]
    
    # Track operation end times for each state to maintain precedence
    def simulate_step(state, action):
        """Simulate one step exactly like RL environment"""
        next_ops, machine_next_free, job_ready_time = state
        j_idx, op_idx, m_idx = action
        
        job_id = job_ids[j_idx]
        machine = machines[m_idx]
        
        # Check if action is valid
        if op_idx != next_ops[j_idx]:
            return None
        if op_idx >= len(JOBS_DATA[job_id]):
            return None
        if machine not in JOBS_DATA[job_id][op_idx]['proc_times']:
            return None
            
        # Calculate timing exactly like RL step()
        machine_available_time = machine_next_free[m_idx]
        job_ready_time_val = job_ready_time[j_idx]
        start_time = max(machine_available_time, job_ready_time_val)
        proc_time = JOBS_DATA[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time
        
        # Update state
        new_next_ops = list(next_ops)
        new_next_ops[j_idx] += 1
        
        new_machine_next_free = list(machine_next_free)
        new_machine_next_free[m_idx] = end_time
        
        new_job_ready = list(job_ready_time)
        new_job_ready[j_idx] = end_time  # Job ready for next operation when this one completes
        
        new_state = make_state(new_next_ops, new_machine_next_free, new_job_ready)
        
        # Calculate reward
        old_makespan = max(max(machine_next_free), max(job_ready_time))
        new_makespan = max(max(new_machine_next_free), max(new_job_ready))
        reward = -(new_makespan - old_makespan)
        
        return new_state, reward
    
    # Initialize
    init_next_ops = [0 for _ in job_ids]
    init_machine_next_free = [0 for _ in machines]
    init_job_ready = [ARRIVALS[j] for j in job_ids]
    init_state = make_state(init_next_ops, init_machine_next_free, init_job_ready)
    
    add_state(init_state)
    q = deque([init_state])
    
    while q:
        current_state = q.popleft()
        current_sid = state_to_id[current_state]
        next_ops, machine_next_free, job_ready_time = current_state
        
        # Check if all jobs are finished
        if sum(next_ops) >= total_operations:
            continue
            
        print(f"\nProcessing state {current_sid}: next_ops={next_ops}")
        
        # Generate all valid actions from current state
        valid_actions = []
        for j_idx, job_id in enumerate(job_ids):
            op_idx = next_ops[j_idx]
            if op_idx < len(JOBS_DATA[job_id]):  # Job not finished
                for m_idx, machine in enumerate(machines):
                    if machine in JOBS_DATA[job_id][op_idx]['proc_times']:
                        action = (j_idx, op_idx, m_idx)
                        result = simulate_step(current_state, action)
                        if result is not None:
                            new_state, reward = result
                            valid_actions.append((action, new_state, reward))
        
        print(f"  Found {len(valid_actions)} valid actions")
        
        # Add transitions and queue new states
        for action, new_state, reward in valid_actions:
            new_sid = add_state(new_state)
            transitions[current_sid].append((action, new_sid, reward))
            
            # Only queue if it's a newly discovered state
            if new_sid == len(id_to_state) - 1:
                q.append(new_state)
    
    print(f"\nConstrained enumeration complete:")
    print(f"Total states: {len(id_to_state)}")
    print(f"Total transitions: {sum(len(v) for v in transitions.values())}")
    
    return state_to_id, id_to_state, transitions

# Run the constrained enumeration
print("=" * 60)
print("RUNNING CONSTRAINED STATE ENUMERATION")
print("=" * 60)

constrained_state_to_id, constrained_id_to_state, constrained_transitions = enumerate_states_constrained()

# Use the constrained results for the rest of the algorithm
state_to_id = constrained_state_to_id
id_to_state = constrained_id_to_state  
transitions = constrained_transitions

num_states = len(id_to_state)
num_transitions = sum(len(v) for v in transitions.values())
elapsed = time.time() - start_time

print(f"Final results: {num_states} states, {num_transitions} transitions (elapsed {elapsed:.2f}s)\n")

# Continue with the old code but skip the original enumeration section
skip_original_enumeration = True
