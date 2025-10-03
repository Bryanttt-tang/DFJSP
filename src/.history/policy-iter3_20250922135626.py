# fjsp_pi_vs_milp_compact.py
# Policy Iteration on event-driven no-queue minimal-state MDP vs MILP optimal benchmark
# Requirements: numpy, matplotlib, pulp

import itertools
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
import csv
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

LARGE_JOBS_DATA = {
    1: [
        {'proc_times': {'M1': 3, 'M2': 5, 'M3': 4, 'M4': 2}},
        {'proc_times': {'M1': 2, 'M2': 3, 'M3': 6, 'M4': 4}},
        {'proc_times': {'M2': 4, 'M3': 2, 'M4': 5}}
    ],
    2: [
        {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2, 'M4': 6}},
        {'proc_times': {'M1': 5, 'M2': 2, 'M4': 3}},
        {'proc_times': {'M1': 3, 'M3': 4, 'M4': 2}}
    ],
    3: [
        {'proc_times': {'M1': 2, 'M2': 6, 'M3': 3}},
        {'proc_times': {'M2': 3, 'M3': 5, 'M4': 4}},
        {'proc_times': {'M1': 4, 'M2': 2, 'M3': 6, 'M4': 3}}
    ],
    4: [
        {'proc_times': {'M1': 5, 'M3': 4, 'M4': 3}},
        {'proc_times': {'M1': 3, 'M2': 4, 'M3': 2, 'M4': 5}}
    ]
}
LARGE_MACHINES = ['M1', 'M2', 'M3', 'M4']
LARGE_ARRIVALS = {1: 0, 2: 0, 3: 0, 4: 1}

XLARGE_JOBS_DATA = {
    1: [
        {'proc_times': {'M1': 3, 'M2': 5, 'M3': 4, 'M4': 2, 'M5': 6}},
        {'proc_times': {'M1': 2, 'M2': 3, 'M3': 6, 'M4': 4}},
        {'proc_times': {'M2': 4, 'M3': 2, 'M4': 5, 'M5': 3}}
    ],
    2: [
        {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2, 'M4': 6}},
        {'proc_times': {'M1': 5, 'M2': 2, 'M4': 3, 'M5': 4}},
        {'proc_times': {'M1': 3, 'M3': 4, 'M4': 2}}
    ],
    3: [
        {'proc_times': {'M1': 2, 'M2': 6, 'M3': 3, 'M5': 5}},
        {'proc_times': {'M2': 3, 'M3': 5, 'M4': 4}},
        {'proc_times': {'M1': 4, 'M2': 2, 'M3': 6, 'M4': 3}}
    ],
    4: [
        {'proc_times': {'M1': 5, 'M3': 4, 'M4': 3}},
        {'proc_times': {'M1': 3, 'M2': 4, 'M3': 2, 'M4': 5, 'M5': 6}},
        {'proc_times': {'M2': 5, 'M3': 3, 'M5': 4}}
    ],
    5: [
        {'proc_times': {'M1': 6, 'M2': 2, 'M4': 4, 'M5': 3}},
        {'proc_times': {'M1': 4, 'M3': 5, 'M4': 2}}
    ]
}
XLARGE_MACHINES = ['M1', 'M2', 'M3', 'M4', 'M5']
XLARGE_ARRIVALS = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2}

# Select problem instance here:
JOBS_DATA = SIMPLE_JOBS_DATA
MACHINES = SIMPLE_MACHINES
ARRIVALS = SIMPLE_ARRIVALS

# Select problem instance here:
# JOBS_DATA = SIMPLE_JOBS_DATA
# MACHINES = SIMPLE_MACHINES
# ARRIVALS = SIMPLE_ARRIVALS

# To test larger datasets:
JOBS_DATA = LARGE_JOBS_DATA
MACHINES = LARGE_MACHINES
ARRIVALS = LARGE_ARRIVALS

job_ids = list(JOBS_DATA.keys())
machines = MACHINES[:]
J = len(job_ids)
M = len(machines)
total_operations = sum(len(ops) for ops in JOBS_DATA.values())

# ---------------------------
# Utilities (canonicalization helpers)
# ---------------------------
def canonicalize_state(next_ops, machine_next_free, job_ready_time, current_time, time_unit=1, cap=None):
    """
    Convert internal state to compact canonical observation format:
    (a) which jobs' next ops are ready now
    (b) processing times for those ready ops on each machine  
    (c) which machines are idle now
    
    Returns hashable tuple for minimal Markov state representation.
    """
    # (a) Ready job indicators (binary: 1 if job has ready operation available now)
    ready_jobs = []
    for j_idx in range(len(next_ops)):
        job_id = job_ids[j_idx]
        op_idx = next_ops[j_idx]
        
        # Job has remaining operations and is ready now?
        if (op_idx < len(JOBS_DATA[job_id]) and 
            job_ready_time[j_idx] <= current_time):
            ready_jobs.append(1)
        else:
            ready_jobs.append(0)
    
    # (c) Machine idle status (binary: 1 if idle now, 0 if busy)
    machine_idle = []
    for m_idx in range(len(machines)):
        is_idle = machine_next_free[m_idx] <= current_time
        machine_idle.append(1 if is_idle else 0)
    
    # (b) Processing times for ready operations on each machine
    proc_times = []
    for j_idx in range(len(next_ops)):
        job_id = job_ids[j_idx]
        op_idx = next_ops[j_idx]
        
        if (op_idx < len(JOBS_DATA[job_id]) and 
            job_ready_time[j_idx] <= current_time):
            # Job has ready operation - include processing times
            operation = JOBS_DATA[job_id][op_idx]
            for machine in machines:
                if machine in operation['proc_times']:
                    proc_time = operation['proc_times'][machine]
                    proc_times.append(proc_time)  # Keep actual processing times
                else:
                    proc_times.append(0)  # Machine cannot process this operation
        else:
            # Job not ready: zeros for all machines
            for machine in machines:
                proc_times.append(0)
    
    # Compact 3-tuple observation
    observation = (tuple(ready_jobs), tuple(machine_idle), tuple(proc_times))
    
    return observation

def pretty_state(s):
    ready_jobs, machine_idle, proc_times = s
    return f"ready_jobs={ready_jobs}, machine_idle={machine_idle}, proc_times_sample={proc_times[:6]}..."

def compute_makespan_internal(machine_next_free, job_ready_time):
    vals = []
    if machine_next_free:
        vals.append(max(machine_next_free))
    if job_ready_time:
        vals.append(max(job_ready_time))
    return max(vals) if vals else 0

# ---------------------------
# Enumerate MDP (event-driven, no-queue, canonicalized states)
# ---------------------------
print("Enumerating reachable canonicalized event-driven no-queue MDP via BFS...")
print(f"Problem size: {len(job_ids)} jobs, {len(machines)} machines, {total_operations} total operations")
start_time = time.time()

state_to_id = {}
id_to_state = []           # canonical states (what agent would observe)
internal_repr = {}         # maps state_id -> one canonical representative internal tuple for debugging
transitions = defaultdict(list)  # sid -> list of (action, nsid, reward)
# action representation: (job_idx, op_idx, machine_idx)  (no WAIT action index used; WAIT is encoded as action ('WAIT',))

MAX_STATES = 200000

def add_canonical_state(canon_state, repr_internal):
    """Add canonical state and keep an internal representative copy for reconstruction."""
    if canon_state not in state_to_id:
        sid = len(id_to_state)
        state_to_id[canon_state] = sid
        id_to_state.append(canon_state)
        internal_repr[sid] = repr_internal
        # progress prints on big enumerations
        if len(id_to_state) % 2000 == 0:
            print(f"  Generated {len(id_to_state)} canonical states...")
    return state_to_id[canon_state]

# initial internal state (absolute times)
init_next_ops = [0 for _ in job_ids]
init_machine_next_free = [0 for _ in machines]
init_job_ready = [ARRIVALS[j] for j in job_ids]  # first op ready time = arrival
init_time = 0

# BFS queue stores full internal state tuples: (next_ops, machine_next_free, job_ready_time, current_time)
q = deque()
q.append((tuple(init_next_ops), tuple(init_machine_next_free), tuple(init_job_ready), init_time))

# add canonical initial state
init_canon = canonicalize_state(init_next_ops, init_machine_next_free, init_job_ready, init_time, time_unit=1)
init_id = add_canonical_state(init_canon, (init_next_ops, init_machine_next_free, init_job_ready, init_time))

states_processed = 0
while q and len(id_to_state) < MAX_STATES:
    next_ops, machine_next_free, job_ready_time, current_time = q.popleft()
    states_processed += 1
    if states_processed % 5000 == 0:
        print(f"  Processed {states_processed} internal states; canonical states so far: {len(id_to_state)}; queue size: {len(q)}")

    finished_ops = sum(next_ops)
    # canonical id for current internal state
    canon = canonicalize_state(next_ops, machine_next_free, job_ready_time, current_time, time_unit=1)
    sid = add_canonical_state(canon, (next_ops, machine_next_free, job_ready_time, current_time))

    if finished_ops >= total_operations:
        # terminal: no outgoing transitions
        continue

    # compute legal scheduling actions under no-queue semantics:
    legal_actions = []
    for j_idx, job_id in enumerate(job_ids):
        op_idx = next_ops[j_idx]
        if op_idx >= len(JOBS_DATA[job_id]):
            continue
        # job ready now?
        if job_ready_time[j_idx] <= current_time:
            proc_dict = JOBS_DATA[job_id][op_idx]['proc_times']
            for m_idx, machine in enumerate(machines):
                # machine idle now?
                if machine_next_free[m_idx] <= current_time and machine in proc_dict:
                    legal_actions.append((j_idx, op_idx, m_idx))

    if not legal_actions:
        # If no schedule-now action possible, add a WAIT transition: advance time to next event deterministically
        # Next event candidates: next machine free time > current_time, next job_ready > current_time (arrivals or prev op completion)
        candidates = []
        for t in machine_next_free:
            if t > current_time:
                candidates.append(t)
        for t in job_ready_time:
            if t > current_time:
                candidates.append(t)
        if not candidates:
            # Shouldn't happen except terminal; treat as terminal
            continue
        next_time = min(candidates)
        # WAIT transition: times don't change except current_time advances
        new_next_ops = tuple(next_ops)
        new_mnext = tuple(machine_next_free)
        new_jready = tuple(job_ready_time)
        new_time = next_time

        # canonicalize and add
        new_canon = canonicalize_state(new_next_ops, new_mnext, new_jready, new_time, time_unit=1)
        nsid = add_canonical_state(new_canon, (new_next_ops, new_mnext, new_jready, new_time))
        # reward: negative makespan increment (makespan = max(max(machine_next_free), max(job_ready_time)))
        old_mk = compute_makespan_internal(list(machine_next_free), list(job_ready_time))
        new_mk = compute_makespan_internal(list(new_mnext), list(new_jready))
        reward = -(new_mk - old_mk)
        # avoid duplicate transitions to same target
        existing = next((t for t in transitions[sid] if t[0] == ('WAIT',) and t[1] == nsid), None)
        if existing is None:
            transitions[sid].append((('WAIT',), nsid, reward))
        # enqueue internal representation for BFS (we use representative full state)
        q.append((new_next_ops, new_mnext, new_jready, new_time))
        continue

    # For each legal scheduling action, compute deterministic next internal state exactly
    cur_makespan = compute_makespan_internal(list(machine_next_free), list(job_ready_time))
    for action in legal_actions:
        j_idx, op_idx, m_idx = action
        job_id = job_ids[j_idx]
        p = JOBS_DATA[job_id][op_idx]['proc_times'][machines[m_idx]]
        # start at current_time (no queue allowed)
        start = current_time
        end = start + p

        # update internal arrays
        new_next_ops = list(next_ops)
        new_next_ops[j_idx] += 1
        new_machine_next_free = list(machine_next_free)
        new_machine_next_free[m_idx] = end
        new_job_ready = list(job_ready_time)
        new_job_ready[j_idx] = end  # next op will be ready at 'end'

        # After scheduling, environment may still have other machines idle at same current_time.
        # For our event-driven decision model we will treat the next decision epoch as still current_time
        # if there are still idle machines and ready jobs; otherwise we will rely on WAIT transitions to advance.
        # For BFS we enqueue the new internal state with same current_time (we stay at the epoch to allow
        # scheduling multiple simultaneous assignments).
        new_time = current_time

        # canonicalize and add
        new_canon = canonicalize_state(new_next_ops, new_machine_next_free, new_job_ready, new_time, time_unit=1)
        nsid = add_canonical_state(new_canon, (tuple(new_next_ops), tuple(new_machine_next_free), tuple(new_job_ready), new_time))
        if nsid >= len(id_to_state) - 1:  # if newly created canonical state, add to BFS queue internal repr
            q.append((tuple(new_next_ops), tuple(new_machine_next_free), tuple(new_job_ready), new_time))

        # reward: negative makespan increment (no step penalty to avoid bias)
        new_mk = compute_makespan_internal(new_machine_next_free, new_job_ready)
        reward = -(new_mk - cur_makespan)

        # avoid duplicate transitions to same target
        existing = next((t for t in transitions[sid] if t[0] == action and t[1] == nsid), None)
        if existing is None:
            transitions[sid].append((action, nsid, reward))

num_states = len(id_to_state)
num_transitions = sum(len(v) for v in transitions.values())
elapsed = time.time() - start_time

incomplete_mdp = len(id_to_state) >= MAX_STATES

print(f"Enumeration done: canonical states = {num_states}, transitions = {num_transitions} (elapsed {elapsed:.2f}s)")
if incomplete_mdp:
    print("WARNING: reached MAX_STATES cap — enumeration truncated (increase MAX_STATES).")

# Show breakdown by ready_jobs (first component of canonical state)
print("\n=== STATE SPACE BY ready_jobs (canonical) ===")
ready_jobs_counts = {}
for sid, canon in enumerate(id_to_state):
    ready_jobs = canon[0]  # First component is ready job indicators
    ready_jobs_counts[ready_jobs] = ready_jobs_counts.get(ready_jobs, 0) + 1
for rj, cnt in sorted(ready_jobs_counts.items()):
    print(f"  ready_jobs={rj}: {cnt} states")

# Save states & transitions to local CSV
csv_filename = "mdp_states_transitions.csv"
print(f"\nSaving canonical states and transitions to {csv_filename} ...")
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['state_id', 'ready_jobs', 'machine_idle', 'proc_times_sample', 'num_actions', 'actions'])
    for sid, canon in enumerate(id_to_state):
        ready_jobs, machine_idle, proc_times = canon
        acts = transitions.get(sid, [])
        actions_str = "; ".join([ (f"{a}->sid{nsid}" if isinstance(a, tuple) and len(a)==1 and a[0]=='WAIT' else f"J{job_ids[a[0]]}-Op{a[1]+1}-M{a[2]+1}->sid{nsid}") for (a, nsid, r) in acts ])
        writer.writerow([sid, list(ready_jobs), list(machine_idle), list(proc_times[:6]), len(acts), actions_str])
print("CSV saved.")

# ---------------------------
# Policy Iteration (Howard) on canonical MDP
# ---------------------------
if incomplete_mdp:
    print("\nSKIPPING Policy Iteration due to incomplete MDP (truncated enumeration).")
    policy = {}
    V = []
    exec_schedule_pi = {m: [] for m in machines}
else:
    print("\nRunning Policy Iteration (Howard) on canonical MDP...\n")
    gamma = 1.0

    # init policy: choose greedy action based on shortest processing time
    policy = {}
    for sid in range(num_states):
        acts = transitions.get(sid, [])
        if not acts:
            policy[sid] = None
        else:
            # prefer scheduling actions over WAIT, and among scheduling actions prefer shortest processing time
            scheduling_acts = [a for a in acts if not (isinstance(a[0], tuple) and a[0][0]=='WAIT')]
            if scheduling_acts:
                # find action with shortest processing time
                best_act = None
                best_proc_time = float('inf')
                for (action, nsid, reward) in scheduling_acts:
                    if isinstance(action, tuple) and len(action) == 3:
                        j_idx, op_idx, m_idx = action
                        job_id = job_ids[j_idx]
                        proc_time = JOBS_DATA[job_id][op_idx]['proc_times'][machines[m_idx]]
                        if proc_time < best_proc_time:
                            best_proc_time = proc_time
                            best_act = action
                policy[sid] = best_act
            else:
                policy[sid] = acts[0][0]  # WAIT action

    # initialize V
    V = [0.0 for _ in range(num_states)]

    def policy_evaluation(policy, V, tol=1e-9, max_iter=1000):
        print(" POLICY EVALUATION:")
        for sweep in range(1, max_iter+1):
            delta = 0.0
            for sid in range(num_states):
                a = policy[sid]
                if a is None:
                    newv = 0.0
                else:
                    # find transition for this action
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
                print("  Policy evaluation converged.\n")
                return V, sweep, delta
        print("  Policy evaluation ended (max iterations reached).\n")
        return V, max_iter, delta

    def policy_improvement(policy, V, max_print=30):
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
            if old_a != best_a:
                policy_stable = False
                changes.append((sid, old_a, best_a, best_q))
                if len(changes) <= max_print:
                    print(f"  State {sid}: {pretty_state(id_to_state[sid])}")
                    print(f"    old: {old_a}, new: {best_a}, best_Q={best_q:.3f}")
            policy[sid] = best_a
        if policy_stable:
            print("  No changes - policy is stable.\n")
        else:
            print(f"  Total policy changes this improvement: {len(changes)}\n")
        return policy, policy_stable, changes

    # Run Howard's Policy Iteration loop
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
        if iteration >= 50:
            print("Stopping after 50 iterations (safety limit).\n")
            break

    # initial state id (canonical)
    init_canon = canonicalize_state(init_next_ops, init_machine_next_free, init_job_ready, init_time, time_unit=1)
    init_id = state_to_id[init_canon]
    print(f"V(init) = {V[init_id]:.3f} => optimal makespan = {-V[init_id]:.3f}\n")

    # ---------------------------
    # Execute PI policy to build actual schedule (simulate with internal dynamics)
    # ---------------------------
    print("Executing the obtained optimal policy from canonical initial state to collect schedule...")
    # reconstruct a representative internal state from internal_repr[init_id]
    rep = internal_repr[init_id]
    cur_next_ops, cur_mnext, cur_jready, cur_time = rep
    cur_next_ops = list(cur_next_ops)
    cur_mnext = list(cur_mnext)
    cur_jready = list(cur_jready)

    exec_schedule_pi = {m: [] for m in machines}
    step = 0
    total_reward = 0.0

    while True:
        cur_canon = canonicalize_state(cur_next_ops, cur_mnext, cur_jready, cur_time, time_unit=1)
        if cur_canon not in state_to_id:
            print(f"Warning: execution reached unexplored state {cur_canon}")
            break
        cur_sid = state_to_id[cur_canon]
        a = policy[cur_sid]
        finished_ops = sum(cur_next_ops)
        if a is None:
            break
        if finished_ops >= total_operations:
            break
        step += 1
        if isinstance(a, tuple) and a[0] == 'WAIT':
            # advance to next event
            candidates = [t for t in cur_mnext if t > cur_time] + [t for t in cur_jready if t > cur_time]
            if not candidates:
                break
            new_time = min(candidates)
            old_mk = compute_makespan_internal(cur_mnext, cur_jready)
            cur_time = new_time
            new_mk = compute_makespan_internal(cur_mnext, cur_jready)
            reward = -(new_mk - old_mk)
            total_reward += reward
            continue
        j_idx, op_idx, m_idx = a
        job_id = job_ids[j_idx]
        machine = machines[m_idx]
        p = JOBS_DATA[job_id][op_idx]['proc_times'][machine]
        start = cur_time
        end = start + p
        # update internal
        cur_next_ops[j_idx] += 1
        cur_mnext[m_idx] = end
        cur_jready[j_idx] = end
        # append to exec schedule
        exec_schedule_pi[machine].append({'job': job_id, 'op': op_idx+1, 'start': start, 'end': end})
        # reward: negative makespan increment (consistent with MDP enumeration)
        old_mk = compute_makespan_internal([cur_mnext[i] if i != m_idx else start for i in range(len(cur_mnext))], 
                                         [cur_jready[i] if i != j_idx else start for i in range(len(cur_jready))])
        new_mk = compute_makespan_internal(cur_mnext, cur_jready)
        reward = -(new_mk - old_mk)
        total_reward += reward

    print(f"PI execution finished in {step} steps. Total reward = {total_reward:.3f}. PI makespan = {-total_reward:.3f}\n")

# ---------------------------
# MILP optimal scheduler (PuLP)
# ---------------------------
def milp_scheduler(jobs, machines, arrivals, time_limit=None):
    print("Solving MILP for optimal benchmark (PuLP + CBC)...")
    prob = LpProblem("FJSP_opt", LpMinimize)

    ops = []
    for j in jobs:
        for oi in range(len(jobs[j])):
            ops.append((j, oi))
    BIG_M = 1000

    # Decision variables
    x = {}
    s = {}
    c = {}
    for op in ops:
        s[op] = LpVariable(f"s_{op[0]}_{op[1]}", lowBound=0)
        c[op] = LpVariable(f"c_{op[0]}_{op[1]}", lowBound=0)
        for m in machines:
            if m in jobs[op[0]][op[1]]['proc_times']:
                x[(op, m)] = LpVariable(f"x_{op[0]}_{op[1]}_{m}", cat=LpBinary)

    Cmax = LpVariable("Cmax", lowBound=0)
    prob += Cmax

    # constraints
    for (j, oi) in ops:
        eligible_machines = [m for m in machines if m in jobs[j][oi]['proc_times']]
        prob += lpSum(x[( (j, oi), m)] for m in eligible_machines) == 1
        prob += c[(j, oi)] == s[(j, oi)] + lpSum(x[((j, oi), m)] * jobs[j][oi]['proc_times'][m] for m in eligible_machines)
        if oi > 0:
            prob += s[(j, oi)] >= c[(j, oi - 1)]
        if oi == 0:
            prob += s[(j, 0)] >= arrivals[j]
        prob += Cmax >= c[(j, oi)]

    # disjunctive constraints
    for m in machines:
        ops_on_m = [op for op in ops if m in jobs[op[0]][op[1]]['proc_times']]
        for i in range(len(ops_on_m)):
            for k in range(i+1, len(ops_on_m)):
                op1 = ops_on_m[i]
                op2 = ops_on_m[k]
                yvar = LpVariable(f"y_{op1[0]}_{op1[1]}_{op2[0]}_{op2[1]}_{m}", cat=LpBinary)
                prob += s[op1] >= c[op2] - BIG_M * (1 - yvar) - BIG_M * (2 - x[(op1, m)] - x[(op2, m)])
                prob += s[op2] >= c[op1] - BIG_M * yvar - BIG_M * (2 - x[(op1, m)] - x[(op2, m)])

    solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit) if time_limit else PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    schedule = {m: [] for m in machines}
    feasible = True
    if prob.status is None:
        feasible = False

    for (j, oi) in ops:
        assigned_m = None
        for m in machines:
            if ((j, oi), m) in x and x[((j, oi), m)].value() is not None and x[((j, oi), m)].value() > 0.5:
                assigned_m = m
                break
        if assigned_m is None:
            feasible = False
            continue
        st = float(s[(j, oi)].value())
        ct = float(c[(j, oi)].value())
        schedule[assigned_m].append({'job': j, 'op': oi+1, 'start': st, 'end': ct})
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
milp_makespan, milp_schedule = milp_scheduler(JOBS_DATA, machines, ARRIVALS, time_limit=30)
milp_elapsed = time.time() - milp_start

# ---------------------------
# Plot Gantt of PI schedule vs MILP schedule
# ---------------------------
def plot_two_gantts(schedule_a, label_a, schedule_b, label_b, machines, title_suffix=""):
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
        max_end = max((t['end'] for m in machines for t in sched[m]), default=10)
        ax.set_title(labels[ax_idx] + (("  (makespan=%.1f)" % max_end ) if any(sched[m] for m in machines) else ""))
        ax.set_xlabel("Time")
        ax.set_yticks([i*10+5 for i in range(len(machines))])
        ax.set_yticklabels(machines)
        ax.set_xlim(0, max_end + 1)
        ax.set_ylim(0, len(machines)*10)
        ax.grid(True)
        for m_idx, m in enumerate(machines):
            for task in sched[m]:
                start = task['start']
                dur = task['end'] - task['start']
                color = job_to_color[task['job']]
                ax.broken_barh([(start, dur)], (m_idx*10, 8), facecolors=(color))
                ax.text(start + dur/2, m_idx*10 + 4, f"J{task['job']}-Op{task['op']}", ha='center', va='center', color='white', fontsize=8)
    patches = [mpatches.Patch(color=job_to_color[j], label=f"J{j}") for j in job_list]
    axes[-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle("PI schedule vs MILP optimal schedule " + title_suffix)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

print("Plotting PI vs MILP schedules...")
# if PI didn't run (incomplete), exec_schedule_pi may not exist - guard:
if 'exec_schedule_pi' not in globals():
    exec_schedule_pi = {m: [] for m in machines}
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
# Simple state space estimation helper
# ---------------------------
def estimate_state_space(jobs_data, machines, arrivals):
    print("\n=== STATE SPACE ESTIMATION (rough) ===")
    num_jobs = len(jobs_data)
    total_ops = sum(len(ops) for ops in jobs_data.values())
    next_ops_combinations = 1
    for job_id, job_ops in jobs_data.items():
        next_ops_combinations *= (len(job_ops) + 1)
    print(f"Next_ops combinations: {next_ops_combinations}")
    if num_jobs == 2 and total_ops == 4:
        estimate = f"canonical states observed: {num_states}"
    else:
        estimate = f"canonical states observed: {num_states}"
    print(estimate)
    return estimate

estimate_state_space(JOBS_DATA, machines, ARRIVALS)
