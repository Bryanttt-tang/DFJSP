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
        {'proc_times': {'M1': 2, 'M2': 3, 'M3': 4, 'M4': 6}},
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

SIMPLE_JOBS_DATA = LARGER_JOBS_DATA
SIMPLE_MACHINES = LARGER_MACHINES
SIMPLE_ARRIVALS = LARGER_ARRIVALS

job_ids = list(SIMPLE_JOBS_DATA.keys())
machines = SIMPLE_MACHINES[:]
J = len(job_ids)
M = len(machines)
total_operations = sum(len(ops) for ops in SIMPLE_JOBS_DATA.values())

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
# Enumerate reachable MDP (minimal-state)
# ---------------------------
print("Enumerating reachable minimal-state MDP via BFS...")
print(f"Problem size: {len(job_ids)} jobs, {len(machines)} machines, {total_operations} total operations")
start_time = time.time()

state_to_id = {}
id_to_state = []
transitions = defaultdict(list)  # sid -> list of (action, nsid, reward)
# action representation: (job_idx, op_idx, machine_idx)

def add_state(s):
    if s not in state_to_id:
        sid = len(id_to_state)
        state_to_id[s] = sid
        id_to_state.append(s)
        # Progress reporting
        if len(id_to_state) % 10000 == 0:
            print(f"  Generated {len(id_to_state)} states so far...")
    return state_to_id[s]

# initial state
init_next_ops = [0 for _ in job_ids]
init_machine_next_free = [0 for _ in machines]
init_job_ready = [SIMPLE_ARRIVALS[j] for j in job_ids]  # job_ready for first op is arrival time
init_state = make_state(init_next_ops, init_machine_next_free, init_job_ready)
add_state(init_state)

q = deque([init_state])
states_processed = 0
while q:
    s = q.popleft()
    sid = state_to_id[s]
    next_ops, machine_next_free, job_ready_time = s
    finished = sum(next_ops)
    
    states_processed += 1
    if states_processed % 10000 == 0:
        print(f"  Processed {states_processed} states, queue size: {len(q)}")
    
    if finished >= total_operations:
        continue

    # legal actions: for each job whose next_op exists, all eligible machines for that op
    legal_actions = []
    for j_idx, job_id in enumerate(job_ids):
        op_idx = next_ops[j_idx]
        if op_idx >= len(SIMPLE_JOBS_DATA[job_id]):  # job finished
            continue
        proc_dict = SIMPLE_JOBS_DATA[job_id][op_idx]['proc_times']
        for m_idx, machine in enumerate(machines):
            if machine in proc_dict:
                legal_actions.append((j_idx, op_idx, m_idx))

    cur_makespan = compute_makespan_from_state(s)

    for action in legal_actions:
        j_idx, op_idx, m_idx = action
        job_id = job_ids[j_idx]
        p = SIMPLE_JOBS_DATA[job_id][op_idx]['proc_times'][machines[m_idx]]
        j_ready = job_ready_time[j_idx]
        m_free = machine_next_free[m_idx]
        start = max(j_ready, m_free)
        end = start + p

        # Use integer time values to keep state space discrete
        start = int(start)
        end = int(end)

        new_next_ops = list(next_ops)
        new_next_ops[j_idx] += 1
        new_machine_next_free = list(machine_next_free)
        new_machine_next_free[m_idx] = end
        new_job_ready = list(job_ready_time)
        new_job_ready[j_idx] = end  # when next op becomes ready

        new_state = make_state(new_next_ops, new_machine_next_free, new_job_ready)
        nsid = add_state(new_state)
        if nsid == len(id_to_state) - 1:
            q.append(new_state)

        new_makespan = compute_makespan_from_state(new_state)
        reward = -(new_makespan - cur_makespan)  # negative makespan increment
        transitions[sid].append((action, nsid, reward))

num_states = len(id_to_state)
num_transitions = sum(len(v) for v in transitions.values())
elapsed = time.time() - start_time
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
    p = SIMPLE_JOBS_DATA[job_id][op_idx]['proc_times'][machine]
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
milp_makespan, milp_schedule = milp_scheduler(SIMPLE_JOBS_DATA, machines, SIMPLE_ARRIVALS, time_limit=60)
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
