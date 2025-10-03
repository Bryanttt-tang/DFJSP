# fjsp_pi_event_noqueue_compactobs.py
# Policy Iteration for event-driven NO-QUEUE FJSP, with compact observation for agent
# PI enumerates canonicalized/quantized internal states for exact DP.

import collections
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict

# --------------------------
# Problem (SIMPLE)
# --------------------------
JOBS = collections.OrderedDict({
    1: [
        {'proc_times': {'M1': 2, 'M2': 4, 'M3': 3}},
        {'proc_times': {'M1': 3, 'M2': 2, 'M3': 4}}
    ],
    2: [
        {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2}},
        {'proc_times': {'M1': 2, 'M2': 3, 'M3': 4}}
    ]
})
MACHINES = ['M1', 'M2', 'M3']
ARRIVALS = {1: 0, 2: 0}
JOB_IDS = list(JOBS.keys())
TOTAL_OPS = sum(len(ops) for ops in JOBS.values())

# --------------------------
# Utilities: canonicalize full internal state for hashing
# --------------------------
def canonicalize_state(next_ops, machine_next_free, job_ready_time, current_time,
                       time_unit=1, cap_horizon=None, shift_normalize=True):
    """
    Convert full absolute-time state into canonical (quantized, shift-normalized) tuple:
      (tuple(next_ops), tuple(machine_can), tuple(job_can))
    This removes uniform time shifts and quantizes to keep a finite state space.
    """
    # remaining times
    machine_rem = [max(0.0, mf - current_time) for mf in machine_next_free]
    job_rem = [max(0.0, jr - current_time) for jr in job_ready_time]

    # cap
    if cap_horizon is not None:
        machine_rem = [min(r, cap_horizon) for r in machine_rem]
        job_rem = [min(r, cap_horizon) for r in job_rem]

    # quantize
    if time_unit and time_unit > 0:
        def q(x): return int(round(x / time_unit))
        machine_q = [q(x) for x in machine_rem]
        job_q = [q(x) for x in job_rem]
    else:
        machine_q = [int(round(x)) for x in machine_rem]
        job_q = [int(round(x)) for x in job_rem]

    # shift-normalize (subtract min so earliest event 0)
    if shift_normalize:
        all_q = machine_q + job_q
        delta = min(all_q) if len(all_q) > 0 else 0
        machine_can = tuple(max(0, mq - delta) for mq in machine_q)
        job_can = tuple(max(0, jq - delta) for jq in job_q)
    else:
        machine_can = tuple(machine_q)
        job_can = tuple(job_q)

    return (tuple(next_ops), machine_can, job_can)


# --------------------------
# Simulator: event-driven, no-queue + WAIT
# internal state kept as absolute times; observations returned to policy are compact
# --------------------------
class EventNoQueueEnv:
    def __init__(self, jobs, machines, arrivals):
        self.jobs = jobs
        self.machines = machines
        self.job_ids = list(jobs.keys())
        self.arrivals = arrivals.copy()
        self.max_ops_per_job = max(len(ops) for ops in jobs.values())
        self.total_operations = sum(len(ops) for ops in jobs.values())
        self.reset()

    def reset(self):
        # internal absolute-time state
        self.machine_next_free = {m: 0 for m in self.machines}
        self.operation_end_times = {j: [0]*len(self.jobs[j]) for j in self.job_ids}
        self.next_operation = {j: 0 for j in self.job_ids}
        self.current_makespan = 0
        self.arrived_jobs = {j for j,a in self.arrivals.items() if a <= 0}
        self.operations_scheduled = 0
        self.schedule = {m: [] for m in self.machines}
        # initial canonicalized state id will be computed externally
        return self._get_observation_compact()

    def is_job_ready_now(self, j, current_time=None):
        if current_time is None:
            current_time = self.current_makespan
        next_op = self.next_operation[j]
        if next_op >= len(self.jobs[j]):
            return False
        # job ready if arrival <= now and previous op end <= now
        if self.arrivals.get(j, 0) > current_time:
            return False
        if next_op == 0:
            return True
        return self.operation_end_times[j][next_op-1] <= current_time

    def _get_observation_compact(self, time_scale=None):
        """
        Compact agent observation at an event time (sensible normalization).
        - machine idle flags (1/0)
        - job_ready flags (1/0)
        - proc times of next op for each job on each machine (0 if not ready/invalid), normalized
        - job progress fraction
        - fraction ops done
        """
        if time_scale is None:
            # heuristic time scale
            all_min = []
            for ops in self.jobs.values():
                for op in ops:
                    all_min.append(min(op['proc_times'].values()))
            time_scale = max(sum(all_min), 1.0)

        t = self.current_makespan
        obs = []

        # machine idle flags
        for m in self.machines:
            idle = 1.0 if self.machine_next_free[m] <= t + 1e-9 else 0.0
            obs.append(idle)

        # job ready flags
        job_ready_flags = {}
        for j in self.job_ids:
            ready = 1.0 if self.is_job_ready_now(j, t) else 0.0
            job_ready_flags[j] = ready
            obs.append(ready)

        # per-job next-op proc times (only if ready, else zeros)
        for j in self.job_ids:
            next_op = self.next_operation[j]
            if next_op < len(self.jobs[j]) and job_ready_flags[j] == 1.0:
                proc_map = self.jobs[j][next_op]['proc_times']
                for m in self.machines:
                    if m in proc_map:
                        obs.append(min(1.0, proc_map[m] / time_scale))
                    else:
                        obs.append(0.0)
            else:
                obs.extend([0.0]*len(self.machines))

        # job progress
        for j in self.job_ids:
            total = len(self.jobs[j])
            obs.append(float(self.next_operation[j]) / total)

        # fraction ops done
        obs.append(self.operations_scheduled / max(1, self.total_operations))

        return np.array(obs, dtype=np.float32)

    def _next_event_time(self):
        t = self.current_makespan
        candidates = []
        # next machine free > t
        for mf in self.machine_next_free.values():
            if mf > t:
                candidates.append(mf)
        # next arrival > t
        for j,a in self.arrivals.items():
            if a > t:
                candidates.append(a)
        return min(candidates) if candidates else None

    def action_mask(self):
        """
        Mask for valid schedule-now actions and WAIT:
         - schedule action (j,op,m) valid if job ready now AND machine idle now and machine compatible
         - WAIT valid only if no schedule-now action (you can change to always-allow WAIT)
        We'll flatten action index as job_idx * (max_ops*|M|) + op_idx*|M| + machine_idx
        """
        base_actions = len(self.job_ids) * self.max_ops_per_job * len(self.machines)
        mask = np.zeros(base_actions + 1, dtype=bool)  # last index is WAIT
        t = self.current_makespan
        valid_count = 0
        for j_idx, j in enumerate(self.job_ids):
            next_op = self.next_operation[j]
            if next_op >= len(self.jobs[j]): 
                continue
            if self.arrivals.get(j, 0) > t:
                continue
            # previous op must be finished
            if next_op > 0 and self.operation_end_times[j][next_op-1] > t + 1e-9:
                continue
            # for each machine compatibility
            for m_idx, m in enumerate(self.machines):
                if m in self.jobs[j][next_op]['proc_times'] and self.machine_next_free[m] <= t + 1e-9:
                    idx = j_idx * (self.max_ops_per_job * len(self.machines)) + next_op*len(self.machines) + m_idx
                    mask[idx] = True
                    valid_count += 1
        if valid_count == 0:
            mask[-1] = True   # allow WAIT
        return mask

    def _decode_action(self, action):
        base_actions = len(self.job_ids) * self.max_ops_per_job * len(self.machines)
        if int(action) == base_actions:
            return ("WAIT", None, None, None)
        action = int(action) % base_actions
        num_machines = len(self.machines)
        ops_per_job = self.max_ops_per_job
        job_idx = action // (ops_per_job * num_machines)
        op_idx = (action % (ops_per_job * num_machines)) // num_machines
        machine_idx = action % num_machines
        return ("ACT", job_idx, op_idx, machine_idx)

    def step(self, action):
        """
        Apply action at event time:
         - If WAIT: advance to next event time deterministically
         - If ACT: must be schedule-now (no-queue): job_ready && machine idle
           start = current_time, end = start + proc_time
        Returns compact obs, reward (makespan-increment negative), done, info
        """
        act_type, j_idx, op_idx, m_idx = self._decode_action(action)
        t = self.current_makespan
        prev_makespan = t

        if act_type == "WAIT":
            nxt = self._next_event_time()
            if nxt is None:
                # nothing else will happen -> terminal
                done = True
                reward = 0.0
                return self._get_observation_compact(), reward, done, {}
            # advance time and update arrived jobs
            self.current_makespan = nxt
            for j,a in self.arrivals.items():
                if a <= self.current_makespan:
                    self.arrived_jobs.add(j)
            done = False
            reward = -(self.current_makespan - prev_makespan)  # usually zero if jump to arrival only
            return self._get_observation_compact(), reward, done, {}

        # ACT
        job = self.job_ids[j_idx]
        machine = self.machines[m_idx]
        # validation (should be checked by mask)
        if not (machine in self.jobs[job][op_idx]['proc_times']):
            return self._get_observation_compact(), -50.0, False, {"err":"incompatible"}
        if not self.is_job_ready_now(job, t):
            return self._get_observation_compact(), -50.0, False, {"err":"job not ready"}
        if self.machine_next_free[machine] > t + 1e-9:
            return self._get_observation_compact(), -50.0, False, {"err":"machine busy"}

        p = self.jobs[job][op_idx]['proc_times'][machine]
        start = t
        end = start + p
        # update
        self.machine_next_free[machine] = end
        self.operation_end_times[job][op_idx] = end
        self.next_operation[job] += 1
        self.operations_scheduled += 1
        self.schedule[machine].append({'job':job, 'op':op_idx+1, 'start':start, 'end':end})
        # current makespan possibly updated but we remain at same event time
        self.current_makespan = max(self.current_makespan, end)
        # Advance automatically to next event if no further schedule-now actions exist (common policy)
        # But for event-driven decision model we return after single action and caller may choose to continue
        done = (self.operations_scheduled >= self.total_operations)
        reward = -(self.current_makespan - prev_makespan)
        return self._get_observation_compact(), reward, done, {}

    def execute_action_from_full_state(self, state_tuple, action):
        """
        Deterministic transition from a *full* absolute state (used by enumerator).
        state_tuple: (next_ops, machine_next_free, job_ready_time, current_time)
        action: either ('WAIT',) or ('ACT', j_idx, op_idx, m_idx)
        returns next_full_state_tuple and reward
        """
        next_ops, machine_next_free, job_ready_time, current_time = state_tuple
        next_ops = list(next_ops)
        machine_next_free = list(machine_next_free)
        job_ready_time = list(job_ready_time)
        prev_makespan = current_time

        if action[0] == 'WAIT':
            # advance time to earliest event > current_time
            cand = []
            for t in machine_next_free:
                if t > current_time:
                    cand.append(t)
            for a in self.arrivals.values():
                if a > current_time:
                    cand.append(a)
            if not cand:
                # terminal still (no future events)
                return (tuple(next_ops), tuple(machine_next_free), tuple(job_ready_time), current_time), 0.0
            new_time = min(cand)
            # update arrivals as job_ready (we store job_ready_time explicit)
            for i,j in enumerate(self.job_ids):
                if self.arrivals[j] <= new_time:
                    job_ready_time[i] = min(job_ready_time[i], self.arrivals[j])  # already
            return (tuple(next_ops), tuple(machine_next_free), tuple(job_ready_time), new_time), -(new_time - prev_makespan)

        # ACT case
        _, j_idx, op_idx, m_idx = action
        j = self.job_ids[j_idx]
        m = self.machines[m_idx]
        # compute job index in job_ready_time list
        job_index = j_idx
        # check start criteria
        start = max(machine_next_free[m_idx], job_ready_time[job_index], current_time)
        p = self.jobs[j][op_idx]['proc_times'][m]
        end = start + p
        # update
        machine_next_free[m_idx] = end
        job_ready_time[job_index] = end
        next_ops[j_idx] += 1
        new_makespan = max(prev_makespan, end)
        # note: current_time remains the same (we are still at event epoch, can schedule further)
        return (tuple(next_ops), tuple(machine_next_free), tuple(job_ready_time), current_time), -(new_makespan - prev_makespan)



# --------------------------
# Enumerate reachable full states (canonicalized) and transitions for PI
# --------------------------
def enumerate_state_graph(env, time_unit=1, cap=None, shift_normalize=True):
    """
    BFS over full internal states starting from reset.
    Use canonicalize_state to keep states finite.
    Returns:
      - id_to_state: list of canonical state tuples (next_ops, machine_can, job_can)
      - transitions: dict state_id -> list of (action_label, next_state_id, reward)
    Actions are labeled either ('WAIT',) or ('ACT', j_idx, op_idx, m_idx)
    """
    # initial full state from env reset
    env.reset()
    init_next_ops = tuple(env.next_operation[j] for j in env.job_ids)
    init_machine_next_free = tuple(env.machine_next_free[m] for m in env.machines)
    init_job_ready = tuple(
        env.operation_end_times[j][env.next_operation[j]-1] if env.next_operation[j] > 0 else env.arrivals[j]
        for j in env.job_ids
    )
    init_current_time = env.current_makespan

    def canon_from_full(nops, mnext, jready, cur):
        return canonicalize_state(nops, mnext, jready, cur, time_unit=time_unit, cap_horizon=cap, shift_normalize=shift_normalize)

    id_map = {}
    id_to_full = {}
    id_to_state = []
    transitions = defaultdict(list)

    def add_state(full_state):
        nops, mnext, jready, cur = full_state
        canon = canon_from_full(nops, mnext, jready, cur)
        if canon not in id_map:
            sid = len(id_to_state)
            id_map[canon] = sid
            id_to_state.append(canon)
            id_to_full[sid] = full_state
            return sid, True
        return id_map[canon], False

    init_full = (init_next_ops, init_machine_next_free, init_job_ready, init_current_time)
    init_id, _ = add_state(init_full)
    q = deque([init_full])
    while q:
        full = q.popleft()
        nops, mnext, jready, cur = full
        sid = id_map[canonicalize_state(nops, mnext, jready, cur, time_unit=time_unit, cap_horizon=cap, shift_normalize=shift_normalize)]
        # Build legal actions at this full state (no-queue semantics)
        legal = []
        # schedule-now actions
        for j_idx, j in enumerate(env.job_ids):
            op_idx = nops[j_idx]
            if op_idx >= len(env.jobs[j]):
                continue
            # check arrival and readiness based on jready (which stores job_ready_time)
            if jready[j_idx] > cur + 1e-9:
                continue
            for m_idx, m in enumerate(env.machines):
                if m in env.jobs[j][op_idx]['proc_times'] and mnext[m_idx] <= cur + 1e-9:
                    legal.append(('ACT', j_idx, op_idx, m_idx))
        if not legal:
            legal = [('WAIT',)]
        # For each action compute deterministic next full state and reward (using env.execute_action_from_full_state)
        for a in legal:
            next_full, reward = env.execute_action_from_full_state((list(nops), list(mnext), list(jready), cur), a)
            # next_full returns tuples; add to id map
            nsid, added = add_state(next_full)
            transitions[sid].append((a, nsid, reward))
            if added:
                q.append(next_full)

    return id_to_state, id_to_full, transitions


# --------------------------
# Policy Iteration (tabular) over the enumerated state graph
# --------------------------
def policy_iteration_on_graph(id_to_state, id_to_full, transitions, gamma=1.0, theta=1e-9):
    N = len(id_to_state)
    # initialize policy: choose first available action per state
    policy = {s: (transitions[s][0][0] if len(transitions[s])>0 else None) for s in range(N)}
    V = np.zeros(N, dtype=float)

    def action_to_str(a):
        if a is None: return "None"
        if a[0]=='WAIT': return "WAIT"
        return f"J{a[1]+1}-Op{a[2]+1}-on-{MACHINES[a[3]]}"

    iteration = 0
    while True:
        iteration += 1
        # Policy Evaluation (sweep until converge)
        print(f"\n--- Policy Iteration {iteration}: POLICY EVALUATION ---")
        sweep = 0
        while True:
            sweep += 1
            delta = 0.0
            V_old = V.copy()
            for s in range(N):
                a = policy[s]
                if a is None:
                    continue
                # find transition for this action
                rec = next(((ns, r) for (act, ns, r) in transitions[s] if act==a), None)
                if rec is None:
                    newv = 0.0
                else:
                    ns, r = rec
                    newv = r + gamma * V_old[ns]
                delta = max(delta, abs(newv - V[s]))
                V[s] = newv
            print(f"  Sweep {sweep} - max delta = {delta:.3e}")
            if delta < theta:
                print(f"  Policy evaluation converged after {sweep} sweeps.")
                break

        # Policy Improvement
        print("\n--- POLICY IMPROVEMENT ---")
        policy_stable = True
        changes = 0
        for s in range(N):
            opts = transitions[s]
            if not opts:
                continue
            best_q = -1e9
            best_a = None
            for (a, ns, r) in opts:
                q = r + gamma * V[ns]
                if q > best_q:
                    best_q = q
                    best_a = a
            if best_a != policy[s]:
                policy_stable = False
                changes += 1
            policy[s] = best_a
        print(f"Policy improvement: {changes} changes")
        # print few example policy entries
        sample_print = 0
        for s in range(min(10, N)):
            print(f"  state {s}: {id_to_state[s]} -> {action_to_str(policy[s])}")
            sample_print += 1
        if policy_stable:
            print("\nPolicy Iteration converged to optimal policy.")
            break
        if iteration >= 50:
            print("\nReached safe iteration limit.")
            break
    return policy, V


# --------------------------
# Run enumeration, PI, and simulate optimal policy
# --------------------------
if __name__ == "__main__":
    env = EventNoQueueEnv(JOBS, MACHINES, ARRIVALS)

    print("Enumerating canonicalized state graph (no-queue + WAIT)...")
    id_to_state, id_to_full, transitions = enumerate_state_graph(env, time_unit=1, cap=None, shift_normalize=True)
    print(f"Enumerated {len(id_to_state)} canonical states and {sum(len(v) for v in transitions.values())} transitions.\n")

    print("Running Policy Iteration on the enumerated graph...")
    pi_policy, V = policy_iteration_on_graph(id_to_state, id_to_full, transitions, gamma=1.0, theta=1e-6)

    # Execute the learned policy in the simulator from initial reset for demonstration
    print("\nExecuting optimal policy in simulator (show schedule):")
    env.reset()
    step = 0
    total_reward = 0.0
    # start from canonicalized initial state id
    init_full = (tuple(env.next_operation[j] for j in env.job_ids),
                 tuple(env.machine_next_free[m] for m in env.machines),
                 tuple( (env.operation_end_times[j][env.next_operation[j]-1] if env.next_operation[j]>0 else env.arrivals[j])
                        for j in env.job_ids),
                 env.current_makespan)
    init_canon = canonicalize_state(*init_full, time_unit=1, cap_horizon=None, shift_normalize=True)
    init_id = id_to_state.index(init_canon)
    cur_id = init_id
    cur_full = id_to_full[cur_id]
    # we'll step by looking up pi_policy until terminal (no outgoing transitions)
    while True:
        a = pi_policy[cur_id]
        if a is None:
            break
        step += 1
        print(f"\nStep {step}: State {cur_id} -> action {a}")
        # apply action using env.execute_action_from_full_state to get next full state and reward
        next_full, reward = env.execute_action_from_full_state(cur_full, a)
        total_reward += reward
        # update env internal state to match next_full for display/schedule (replay)
        # convert next_full to env internal
        nops, mnext, jready, curtime = next_full
        # write into env
        env.next_operation = {j: nops[i] for i,j in enumerate(env.job_ids)}
        env.machine_next_free = {m: mnext[i] for i,m in enumerate(env.machines)}
        # reconstruct operation_end_times approximately from jready (only used for readiness checks)
        for i,j in enumerate(env.job_ids):
            # if next_op>0, previous op end is jready (we stored job_ready_time accordingly in enumerator)
            if env.next_operation[j] > 0:
                env.operation_end_times[j][env.next_operation[j]-1] = jready[i]
        env.current_makespan = curtime
        # compute canonical of next
        cur_full = next_full
        cur_id = id_to_state.index(canonicalize_state(*cur_full, time_unit=1, cap_horizon=None, shift_normalize=True))
        print(f"  -> next state id {cur_id}, reward {reward:.3f}, makespan {env.current_makespan}")
        # termination if no outgoing transitions
        if len(transitions[cur_id]) == 0:
            print("Reached terminal-like state (no outgoing transitions).")
            break
        if step > 200:
            print("Safety break.")
            break

    print(f"\nExecution finished: steps={step}, total_reward={total_reward:.3f}")
    # Print the schedule recorded in env.schedule
    print("\nFinal schedule (per machine):")
    for m in env.machines:
        for t in env.schedule[m]:
            print(f"  {m}: J{t['job']}-Op{t['op']} [{t['start']},{t['end']}]")

    # Optional: simple gantt plot
    fig, ax = plt.subplots(figsize=(8,4))
    machine_idx = {m:i for i,m in enumerate(env.machines)}
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
    for m in env.machines:
        for task in env.schedule[m]:
            s = task['start']; e = task['end']; dur = e-s
            ax.broken_barh([(s,dur)], (machine_idx[m]*10, 9), facecolors=(colors[(task['job']-1) % len(colors)]))
            ax.text(s+dur/2, machine_idx[m]*10+4.5, f"J{task['job']}-O{task['op']}", ha='center', va='center', color='white')
    ax.set_yticks([i*10+4.5 for i in range(len(env.machines))])
    ax.set_yticklabels(env.machines)
    ax.set_xlabel("Time")
    ax.set_title("PI-derived schedule (no-queue event-driven)")
    plt.tight_layout()
    plt.show()
