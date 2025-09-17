# fjsp_pi_noqueue_wait_vs_milp.py
# Policy Iteration on fully-observed minimal-state MDP with NO-QUEUE semantics + WAIT action
# Compares PI schedule with MILP optimal schedule (PuLP/CBC)
# Requirements: numpy, matplotlib, pulp

import itertools
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import time
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpBinary

# ---------------------------
# Problem definition (SIMPLE)
# ---------------------------
JOBS = {
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
ARRIVALS = {1: 0, 2: 0}  # all arrive at t=0

job_ids = list(JOBS.keys())
machines = MACHINES[:]
J = len(job_ids)
M = len(machines)
TOTAL_OPS = sum(len(ops) for ops in JOBS.values())

# ---------------------------
# Minimal fully-observed state for no-queue semantics:
# (next_ops_tuple, machine_next_free_tuple, job_ready_time_tuple, current_time)
# - next_ops: how many ops completed (or scheduled) per job
# - machine_next_free: absolute times machines become free (ints)
# - job_ready_time: absolute times when the next op for each job becomes eligible (previous op end or arrival)
# - current_time: the global clock; actions that start an op must have start==current_time
# ---------------------------

def make_state(next_ops, machine_next_free, job_ready_time, current_time):
    return (tuple(next_ops), tuple(machine_next_free), tuple(job_ready_time), int(current_time))

def pretty_state(s):
    no, mnext, jready, ct = s
    return f"next_ops={no}, mnext={mnext}, jready={jready}, t={ct}"

def makespan_of_state(s):
    # makespan is the maximum completion time scheduled so far (machine_next_free and job_ready_time hold end times)
    _, mnext, jready, ct = s
    vals = []
    if len(mnext)>0: vals.append(max(mnext))
    if len(jready)>0: vals.append(max(jready))
    vals.append(ct)
    return max(vals)

# ---------------------------
# BFS enumerate reachable states under NO-QUEUE semantics + WAIT action
# - Legal scheduling action (j_idx, op_idx, m_idx) only if:
#     job_ready_time[j_idx] <= current_time AND machine_next_free[m_idx] <= current_time
# - If no scheduling actions available, add WAIT action that advances current_time to next event:
#     next_time = min(t for t in machine_next_free if t > current_time)
#   (we can also include future job_ready_time if arrivals > 0; here arrivals are 0)
# - When scheduling, we still record the operation's end-time into machine_next_free and job_ready_time
#   and increment next_ops immediately (this aligns with the earlier deterministic-transition representation).
# ---------------------------

print("Enumerating minimal-state NO-QUEUE MDP (with WAIT action)...")
t0 = time.time()

state_to_id = {}
id_to_state = []
transitions = defaultdict(list)  # sid -> list of (action, nsid, reward)

def add_state(s):
    if s not in state_to_id:
        sid = len(id_to_state)
        state_to_id[s] = sid
        id_to_state.append(s)
    return state_to_id[s]

# initial state
init_next_ops = [0 for _ in job_ids]
init_machine_next_free = [0 for _ in machines]
init_job_ready = [ARRIVALS[j] for j in job_ids]
init_current_time = 0
init_state = make_state(init_next_ops, init_machine_next_free, init_job_ready, init_current_time)
add_state(init_state)

q = deque([init_state])
while q:
    s = q.popleft()
    sid = state_to_id[s]
    next_ops, machine_next_free, job_ready_time, current_time = s
    finished = sum(next_ops)
    if finished >= TOTAL_OPS:
        continue

    # build legal schedule-now actions (no-queue)
    legal_actions = []
    for j_idx, job_id in enumerate(job_ids):
        op_idx = next_ops[j_idx]
        if op_idx >= len(JOBS[job_id]):  # job finished
            continue
        # job must be ready by current_time
        if job_ready_time[j_idx] > current_time:
            continue
        proc_dict = JOBS[job_id][op_idx]['proc_times']
        for m_idx, mach in enumerate(machines):
            if mach in proc_dict:
                # machine must be idle at current_time (no queuing)
                if machine_next_free[m_idx] <= current_time:
                    legal_actions.append((j_idx, op_idx, m_idx))
    # If no legal schedule-now action, we allow WAIT
    allow_wait = (len(legal_actions) == 0)

    cur_makespan = makespan_of_state(s)

    # Scheduling actions
    for action in legal_actions:
        j_idx, op_idx, m_idx = action
        job_id = job_ids[j_idx]
        p = JOBS[job_id][op_idx]['proc_times'][machines[m_idx]]
        start = current_time  # by construction
        end = start + p

        new_next_ops = list(next_ops)
        new_next_ops[j_idx] += 1
        new_mnext = list(machine_next_free)
        new_mnext[m_idx] = end
        new_jready = list(job_ready_time)
        new_jready[j_idx] = end
        # Note: current_time remains the same because scheduling at same time may allow more starts on other idle machines
        new_current_time = current_time

        new_state = make_state(new_next_ops, new_mnext, new_jready, new_current_time)
        nsid = add_state(new_state)
        if nsid == len(id_to_state) - 1:
            q.append(new_state)
        new_makespan = makespan_of_state(new_state)
        reward = -(new_makespan - cur_makespan)
        transitions[sid].append((('S', action), nsid, reward))  # prefix 'S' for scheduling action

    # WAIT action if allowed (or optionally always include WAIT)
    if allow_wait:
        # find next event time (next machine free > current_time or next job_ready > current_time)
        cand_times = []
        for t in machine_next_free:
            if t > current_time:
                cand_times.append(t)
        for t in job_ready_time:
            if t > current_time:
                cand_times.append(t)
        # Also consider arrival times if they are > current_time (not in this toy)
        arr_times = [ARRIVALS[jj] for jj in job_ids if ARRIVALS[jj] > current_time]
        cand_times.extend(arr_times)
        if not cand_times:
            # No future events -> dead-end; but if unfinished ops remain, this shouldn't happen for correct models
            continue
        next_time = min(cand_times)
        new_state = make_state(list(next_ops), list(machine_next_free), list(job_ready_time), next_time)
        nsid = add_state(new_state)
        if nsid == len(id_to_state) - 1:
            q.append(new_state)
        new_makespan = makespan_of_state(new_state)
        reward = -(new_makespan - cur_makespan)  # often 0 for WAIT
        transitions[sid].append((('W', None), nsid, reward))

num_states = len(id_to_state)
num_transitions = sum(len(v) for v in transitions.values())
t_elapsed = time.time() - t0
print(f"Enumeration done: {num_states} states, {num_transitions} transitions (elapsed {t_elapsed:.2f}s)\n")

# ---------------------------
# Policy Iteration
# ---------------------------
print("Running Policy Iteration (no-queue + WAIT) with verbose traces...\n")
gamma = 1.0

# initialize policy: choose first available transition for each nonterminal state
policy = {}
for sid, s in enumerate(id_to_state):
    next_ops, _, _, _ = s
    if sum(next_ops) >= TOTAL_OPS:
        policy[sid] = None
    else:
        acts = transitions[sid]
        policy[sid] = acts[0][0] if acts else None

V = [0.0 for _ in range(num_states)]

def policy_evaluation(policy, V, tol=1e-9, max_iter=500):
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
        print(f"  Sweep {sweep:3d} - max delta = {delta:.6e}")
        if delta < tol:
            print("  Converged policy evaluation.\n")
            return V, sweep, delta
    print("  Reached max_iter in policy evaluation.\n")
    return V, max_iter, delta

def policy_improvement(policy, V, max_print=40):
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
    if iteration >= 40:
        print("Stopping after 40 iterations (safety limit).\n")
        break

init_id = state_to_id[init_state]
print(f"V(init) = {V[init_id]:.3f} => optimal makespan = {-V[init_id]:.3f}\n")

# ---------------------------
# Execute PI policy to collect schedule (respecting WAIT)
# ---------------------------
print("Executing the optimal policy from initial state to collect PI schedule (no-queue semantics)...")
cur_sid = init_id
exec_schedule_pi = {m: [] for m in machines}
step = 0
total_reward = 0.0
visited_states = set()
while True:
    a = policy[cur_sid]
    if a is None:
        break
    # Avoid infinite loop (shouldn't happen)
    if (cur_sid, a) in visited_states:
        print("Warning: loop detected in execution; breaking.")
        break
    visited_states.add((cur_sid, a))

    # find transition record
    rec = next(((act, nsid, r) for (act, nsid, r) in transitions[cur_sid] if act == a), None)
    if rec is None:
        print("ERROR: action not found in transitions; aborting execution.")
        break
    act, nsid, r = rec
    tag, payload = act
    next_ops, mnext, jready, ct = id_to_state[cur_sid]

    if tag == 'W':
        # WAIT: just advance current_time (no schedule entry)
        print(f"Step {step+1}: WAIT at t={ct} -> advance to t={id_to_state[nsid][3]}")
    else:
        # Scheduling
        (j_idx, op_idx, m_idx) = payload
        job_id = job_ids[j_idx]
        machine = machines[m_idx]
        p = JOBS[job_id][op_idx]['proc_times'][machine]
        start = ct  # start must equal current_time for no-queue action
        end = start + p
        exec_schedule_pi[machine].append({'job': job_id, 'op': op_idx+1, 'start': start, 'end': end})
        print(f"Step {step+1}: Schedule J{job_id}-Op{op_idx+1} on {machine} @ [{start},{end}] (reward {r:.3f})")
    total_reward += r
    cur_sid = nsid
    step += 1

print(f"\nPI execution finished in {step} steps. Total reward = {total_reward:.3f}. PI makespan = {-total_reward:.3f}\n")

# ---------------------------
# MILP optimal scheduler for benchmark (same code pattern as earlier)
# ---------------------------
def milp_scheduler(jobs, machines, arrivals, time_limit=None):
    print("Solving MILP (PuLP CBC) for optimal benchmark...")
    prob = LpProblem("FJSP_opt", LpMinimize)
    ops = []
    for j in jobs:
        for oi in range(len(jobs[j])):
            ops.append((j, oi))
    BIG_M = 1000

    # decision variables
    x = {}
    s = {}
    c = {}
    for op in ops:
        s[op] = LpVariable(f"s_{op[0]}_{op[1]}", lowBound=0)
        c[op] = LpVariable(f"c_{op[0]}_{op[1]}", lowBound=0)
        j, oi = op
        for m in machines:
            if m in jobs[j][oi]['proc_times']:
                x[(op, m)] = LpVariable(f"x_{j}_{oi}_{m}", cat=LpBinary)

    y = {}
    for m in machines:
        ops_on_m = [op for op in ops if m in jobs[op[0]][op[1]]['proc_times']]
        for i in range(len(ops_on_m)):
            for k in range(i+1, len(ops_on_m)):
                op1 = ops_on_m[i]; op2 = ops_on_m[k]
                y[(op1, op2, m)] = LpVariable(f"y_{op1[0]}_{op1[1]}_{op2[0]}_{op2[1]}_{m}", cat=LpBinary)

    Cmax = LpVariable("Cmax", lowBound=0)
    prob += Cmax

    # constraints
    for op in ops:
        j, oi = op
        eligible = [m for m in machines if m in jobs[j][oi]['proc_times']]
        prob += lpSum(x[(op, m)] for m in eligible) == 1
        prob += c[op] == s[op] + lpSum(x[(op, m)] * jobs[j][oi]['proc_times'][m] for m in eligible)
        if oi > 0:
            prob += s[(j, oi)] >= c[(j, oi - 1)]
        if oi == 0:
            prob += s[(j, 0)] >= arrivals[j]
        prob += Cmax >= c[op]

    # disjunctive machine constraints using y variables
    for m in machines:
        ops_on_m = [op for op in ops if m in jobs[op[0]][op[1]]['proc_times']]
        for i in range(len(ops_on_m)):
            for k in range(i+1, len(ops_on_m)):
                op1 = ops_on_m[i]; op2 = ops_on_m[k]
                # if both assigned to m then either op1 before op2 or op2 before op1
                prob += s[op1] >= c[op2] - BIG_M * (1 - y[(op1, op2, m)]) - BIG_M * (2 - x[(op1, m)] - x[(op2, m)])
                prob += s[op2] >= c[op1] - BIG_M * (y[(op1, op2, m)]) - BIG_M * (2 - x[(op1, m)] - x[(op2, m)])

    solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit) if time_limit else PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    schedule = {m: [] for m in machines}
    feasible = True
    for op in ops:
        j, oi = op
        assigned = None
        for m in machines:
            if (op, m) in x and x[(op, m)].value() is not None and x[(op, m)].value() > 0.5:
                assigned = m
                break
        if assigned is None:
            feasible = False
            continue
        st = float(s[op].value())
        ct = float(c[op].value())
        schedule[assigned].append({'job': j, 'op': oi+1, 'start': st, 'end': ct})
    for m in machines:
        schedule[m].sort(key=lambda t: t['start'])
    Cval = float(Cmax.value()) if Cmax.value() is not None else float('inf')
    if feasible:
        print(f"MILP solved: optimal makespan = {Cval:.2f}")
    else:
        print("MILP failed (infeasible or no assignment).")
    return Cval, schedule

milp_start = time.time()
milp_makespan, milp_schedule = milp_scheduler(JOBS, machines, ARRIVALS, time_limit=60)
milp_elapsed = time.time() - milp_start

# ---------------------------
# Plotting routines
# ---------------------------
def plot_two_gantts(schedule_a, label_a, schedule_b, label_b, machines):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    all_scheds = [schedule_a, schedule_b]
    labels = [label_a, label_b]
    cmap = plt.cm.get_cmap("tab20")
    # gather job ids
    job_set = set()
    for sched in all_scheds:
        for m in machines:
            for t in sched[m]:
                job_set.add(t['job'])
    job_list = sorted(job_set)
    job_to_color = {job_list[i]: cmap(i % 20) for i in range(len(job_list))}

    for axi, ax in enumerate(axes):
        sched = all_scheds[axi]
        ax.set_title(labels[axi])
        ax.set_xlabel("Time")
        ax.set_yticks([i*10+5 for i in range(len(machines))])
        ax.set_yticklabels(machines)
        ax.set_xlim(0, max((t['end'] for m in machines for t in sched[m]), default=10))
        ax.set_ylim(0, len(machines)*10)
        ax.grid(True)
        for m_idx, m in enumerate(machines):
            for task in sched[m]:
                start = task['start']; dur = task['end'] - task['start']
                color = job_to_color[task['job']]
                ax.broken_barh([(start, dur)], (m_idx*10, 8), facecolors=(color))
                ax.text(start + dur/2, m_idx*10 + 4, f"J{task['job']}-Op{task['op']}", ha='center', va='center', color='white', fontsize=8)
    patches = [mpatches.Patch(color=job_to_color[j], label=f"J{j}") for j in job_list]
    axes[-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle("PI (no-queue + WAIT) vs MILP optimal")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

print("\nPlotting PI vs MILP schedules...")
plot_two_gantts(exec_schedule_pi, "Policy Iteration (no-queue + WAIT)", milp_schedule, "MILP Optimal", machines)

print("\nPI-derived schedule (per machine):")
for m in machines:
    for t in exec_schedule_pi[m]:
        print(f"  {m}: J{t['job']}-Op{t['op']}  [{t['start']:.1f}, {t['end']:.1f}]")
print("\nMILP optimal schedule (per machine):")
for m in machines:
    for t in milp_schedule[m]:
        print(f"  {m}: J{t['job']}-Op{t['op']}  [{t['start']:.1f}, {t['end']:.1f}]")

print("\nDone.")
