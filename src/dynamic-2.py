import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import heapq
import itertools
import pulp

# ──────────── Problem Data ────────────

machines = ["M1", "M2", "M3"]

all_jobs = {
    1: {'proc_times': {'M1': 10, 'M2': 12, 'M3': 15}, 'arrival': 0},
    2: {'proc_times': {'M1': 8,  'M2': 14, 'M3': 11}, 'arrival': 0},
    3: {'proc_times': {'M1': 9,  'M2': 10, 'M3': 13}, 'arrival': 0},
    4: {'proc_times': {'M1': 11, 'M2': 9,  'M3': 14}, 'arrival': 0},
    5: {'proc_times': {'M1': 12, 'M2': 13, 'M3': 10}, 'arrival': 0},
    6: {'proc_times': {'M1': 14, 'M2': 10, 'M3': 12}, 'arrival': 4},
    7: {'proc_times': {'M1': 9,  'M2': 11, 'M3': 13}, 'arrival': 8},
    8: {'proc_times': {'M1': 13, 'M2': 12, 'M3': 14}, 'arrival': 12}
}


# ──────────── 1) Greedy Dynamic Scheduler (SPT dispatch) ────────────

def greedy_dynamic_scheduler(jobs, machines):
    """
    At each time t, assign any idle machine to the available job with smallest
    processing time on that machine.  Jobs arrive dynamically at their 'arrival'.
    """
    time = 0
    schedule = {m: [] for m in machines}      # store (job, start, end) per machine
    busy = {m: 0 for m in machines}           # next free time of each machine
    completed = set()                         # which jobs have been scheduled

    # Min‐heap of (arrival_time, job_id)
    arrivals = [(v['arrival'], j) for j, v in jobs.items()]
    heapq.heapify(arrivals)

    unscheduled = {}  # jobs that have arrived but not yet scheduled
    arrival_events = []  # just to record arrival lines for plotting

    while len(completed) < len(jobs):
        # 1) Move any job whose arrival ≤ time from arrivals→unscheduled
        while arrivals and arrivals[0][0] <= time:
            at, jid = heapq.heappop(arrivals)
            unscheduled[jid] = jobs[jid]
            arrival_events.append((jid, at))

        # 2) Find all currently idle machines
        idle = [m for m in machines if busy[m] <= time]

        # 3) Find all jobs that have arrived and not scheduled
        avail = [j for j in unscheduled if unscheduled[j]['arrival'] <= time]

        # 4) Assign each idle machine greedily (SPT on that machine)
        for m in idle:
            if not avail: 
                break
            # pick job with minimum processing time on machine m
            best_j = min(avail, key=lambda j: jobs[j]['proc_times'][m])
            p = jobs[best_j]['proc_times'][m]
            schedule[m].append((best_j, time, time + p))
            busy[m] = time + p
            completed.add(best_j)
            del unscheduled[best_j]
            avail.remove(best_j)

        # 5) Advance time: either next arrival or next machine‐free time
        next_arrival = arrivals[0][0] if arrivals else float('inf')
        next_free   = min(busy[m] for m in machines if busy[m] > time) \
                      if any(busy[m] > time for m in machines) else float('inf')

        if idle and not avail:
            time = min(next_arrival, next_free)
        elif idle and avail:
            continue
        else:
            time = min(next_arrival, next_free)

    return schedule, arrival_events


# ──────────── 2) “Exact” by Simple Assignment + Arrival‐order Sequencing ────────────

def exact_schedule_all(jobs, machines):
    """
    For each possible assignment of jobs→machines (compatible sets),
    schedule on each machine in the order of increasing arrival time.
    Then pick the assignment that yields the lowest makespan.
    """
    best_makespan = float('inf')
    best_schedule = None

    job_ids = sorted(jobs.keys())
    machine_options = [list(jobs[j]['proc_times'].keys()) for j in job_ids]

    for assign_tuple in itertools.product(*machine_options):
        # build a dict job→assigned_machine
        assign_map = {job_ids[i]: assign_tuple[i] for i in range(len(job_ids))}
        sch = {m: [] for m in machines}
        completion = {}

        # sort jobs by arrival time, then place them FIFO on their assigned machine
        for j in sorted(job_ids, key=lambda jj: jobs[jj]['arrival']):
            m = assign_map[j]
            r = jobs[j]['arrival']
            last_end = sch[m][-1][2] if sch[m] else 0
            start_time = max(last_end, r)
            end_time = start_time + jobs[j]['proc_times'][m]
            sch[m].append((j, start_time, end_time))
            completion[j] = end_time

        mk = max(completion.values())
        if mk < best_makespan:
            best_makespan = mk
            best_schedule = sch

    return best_schedule


# ──────────── 3) “MIP‐Equivalent” Full Enumeration ────────────

def full_exact_scheduler(jobs, machines):
    """
    FULL enumeration of:
      1. all assignments job→machine,
      2. all possible permutations on each machine,
      3. schedule respecting release times + sequence,
      4. pick global min makespan.
    This replicates exactly what a MIP would do (assignment + ordering binaries).
    """
    job_ids = sorted(jobs.keys())
    best_makespan = float('inf')
    best_schedule = None

    # Precompute each job's feasible machines:
    machine_options = { j: list(jobs[j]['proc_times'].keys()) for j in job_ids }

    # 1) Enumerate every possible assignment of jobs→machines
    #    (each product shows one combination of machines for all jobs)
    for assign_tuple in itertools.product(*[machine_options[j] for j in job_ids]):
        assign_map = { job_ids[i]: assign_tuple[i] for i in range(len(job_ids)) }

        # 2) For this assignment, group jobs by their assigned machine
        jobs_on_machine = { m: [] for m in machines }
        for j in job_ids:
            m = assign_map[j]
            jobs_on_machine[m].append(j)

        # 3) For each machine, list ALL permutations of jobs_on_machine[m]
        all_orders_on_m = []
        for m in machines:
            all_orders_on_m.append(list(itertools.permutations(jobs_on_machine[m])))

        # 4) Take Cartesian product of these per-machine permutations
        #    so ‘orders’ is a tuple of length len(machines), where
        #    orders[i] is a permutation of the jobs assigned to machine i.
        for orders in itertools.product(*all_orders_on_m):
            # Build a tentative schedule under these per-machine sequences:
            current_schedule = { m: [] for m in machines }
            completion = {}

            # On each machine m (index mach_idx), schedule its permutation in order:
            for mach_idx, m in enumerate(machines):
                last_finish = 0
                for job_id in orders[mach_idx]:
                    r = jobs[job_id]['arrival']
                    p = jobs[job_id]['proc_times'][m]
                    start_time = max(last_finish, r)
                    finish_time = start_time + p
                    current_schedule[m].append((job_id, start_time, finish_time))
                    completion[job_id] = finish_time
                    last_finish = finish_time

            makespan = max(completion.values())
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = current_schedule

    return best_schedule

# ──────────── Big‐M Parameter ────────────
M_big = 1000  # large enough to deactivate constraints when needed

job_ids = sorted(all_jobs.keys())

# ──────────── Build MIP Model ────────────

model = pulp.LpProblem("Dynamic_FJSP_MIP", pulp.LpMinimize)

# 1) x[i,k] = 1 if job i is assigned to machine k
x = {}
for i in job_ids:
    for k in all_jobs[i]['proc_times']:
        x[(i, k)] = pulp.LpVariable(f"x_{i}_{k}", cat="Binary")

# 2) y[i,j,k] = 1 if job i precedes job j on machine k  (only for i<j and k in common)
y = {}
for i in job_ids:
    for j in job_ids:
        if i < j:
            common = set(all_jobs[i]['proc_times']).intersection(all_jobs[j]['proc_times'])
            for k in common:
                y[(i, j, k)] = pulp.LpVariable(f"y_{i}_{j}_{k}", cat="Binary")

# 3) s[i], c[i] = start and completion times of job i (continuous)
s = {i: pulp.LpVariable(f"s_{i}", lowBound=0) for i in job_ids}
c = {i: pulp.LpVariable(f"c_{i}", lowBound=0) for i in job_ids}

# 4) Makespan
Cmax = pulp.LpVariable("Cmax", lowBound=0)

# Objective: minimize makespan
model += Cmax

# ──────────── Constraints ────────────

# A) Each job must go to exactly one machine
for i in job_ids:
    model += pulp.lpSum(x[(i, k)] for k in all_jobs[i]['proc_times']) == 1

# B) Completion time definition: c[i] = s[i] + sum_k(p[i,k] * x[i,k])
for i in job_ids:
    model += c[i] == s[i] + pulp.lpSum(all_jobs[i]['proc_times'][k] * x[(i, k)]
                                       for k in all_jobs[i]['proc_times'])

# C) Release time constraint: s[i] ≥ arrival[i]
for i in job_ids:
    model += s[i] >= all_jobs[i]['arrival']

# D) No‐overlap constraints for any two jobs i < j on any common machine k
#    If both assigned to k (x[i,k] = x[j,k] = 1), then either s[i] ≥ c[j] (i after j)
#    or s[j] ≥ c[i] (j after i), controlled by y[i,j,k].
for i in job_ids:
    for j in job_ids:
        if i < j:
            common = set(all_jobs[i]['proc_times']).intersection(all_jobs[j]['proc_times'])
            for k in common:
                # If x[i,k] = x[j,k] = 1 and y[i,j,k] = 1 ⇒ s[i] ≥ c[j]
                model += s[i] >= c[j] - M_big * (1 - y[(i, j, k)]) - M_big * (2 - x[(i, k)] - x[(j, k)])
                # If x[i,k] = x[j,k] = 1 and y[i,j,k] = 0 ⇒ s[j] ≥ c[i]
                model += s[j] >= c[i] - M_big * (y[(i, j, k)])     - M_big * (2 - x[(i, k)] - x[(j, k)])

# E) Makespan definition: Cmax ≥ c[i] for all i
for i in job_ids:
    model += Cmax >= c[i]

# ──────────── Solve MILP ────────────

# Use CBC (comes with PuLP) silently
model.solve(pulp.PULP_CBC_CMD(msg=False))

# ──────────── Extract MIP Schedule ────────────

schedule = {m: [] for m in machines}
for i in job_ids:
    chosen_machine = None
    for k in all_jobs[i]['proc_times']:
        if pulp.value(x[(i, k)]) > 0.5:
            chosen_machine = k
            break
    start_time = pulp.value(s[i])
    end_time = pulp.value(c[i])
    schedule[chosen_machine].append((i, start_time, end_time))

# Sort each machine’s schedule by start time
for m in schedule:
    schedule[m].sort(key=lambda trip: trip[1])

# Build arrival_events for plotting (red arrows for J6, J7, J8)
arrival_events = [(all_jobs[i]['arrival'], i) for i in job_ids]
arrival_events.sort()


# ──────────── 4) Plotting Utility ────────────

def make_timeline_plot(schedule, arrival_events, title, filename):
    """
    Draws a 4‐panel timeline:
      1) Initial view – jobs 1–5 only
      2) After J6 (t=4)
      3) After J7 (t=8)
      4) After J8 (t=12)
    Vertical red arrows mark the arrival times of J6, J7, J8.
    """
    snapshots = [
        ("Initial (J1–J5)", {1,2,3,4,5}),
        ("After J6 (t=4)",    {1,2,3,4,5,6}),
        ("After J7 (t=8)",    {1,2,3,4,5,6,7}),
        ("After J8 (t=12)",   {1,2,3,4,5,6,7,8})
    ]

    fig, axs = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    colors = plt.cm.tab20.colors

    for ax, (label, visible_jobs) in zip(axs, snapshots):
        for i, m in enumerate(machines):
            for (j, start, end) in schedule[m]:
                if j in visible_jobs:
                    ax.broken_barh(
                        [(start, end - start)],
                        (i*10, 9),
                        facecolors=colors[j % 20],
                        edgecolor='black'
                    )
                    ax.text(start + 0.2, i*10 + 2, f"J{j}", color='white')

        # Draw vertical arrival lines (only for J6, J7, J8)
        for (jid, at) in arrival_events:
            if (jid in visible_jobs) and (label != "Initial (J1–J5)"):
                ax.axvline(at, color='red', linestyle='--')
                ymin, ymax = ax.get_ylim()
                ax.annotate(
                    '',
                    xy=(at, ymax),
                    xytext=(at, ymin),
                    arrowprops=dict(arrowstyle='->', color='red')
                )

        ax.set_yticks([i*10 + 4.5 for i in range(len(machines))])
        ax.set_yticklabels(machines)

    axs[-1].set_xlabel("Time")
    plt.tight_layout(rect=[0,0,1,0.96])
    fig.suptitle(title, y=1.02)
    fig.savefig(filename)
    plt.show()
    plt.close(fig)


# ──────────── 5) Run All Three Schedulers and Plot ────────────

# 5.1 Greedy dynamic (SPT dispatch)
greedy_sched, arrivals = greedy_dynamic_scheduler(all_jobs, machines)
make_timeline_plot(
    greedy_sched, arrivals,
    title="Greedy (SPT) Dynamic Timeline",
    filename="greedy_timeline.png"
)

# 5.2 “Exact” by Simple Assignment + Arrival‐order
exact_sched = exact_schedule_all(all_jobs, machines)
make_timeline_plot(
    exact_sched, arrivals,
    title="Exact (FIFO on Assigned Machine) Timeline",
    filename="exact_timeline.png"
)

# 5.3 “MIP‐Equivalent” Full Enumeration
mip_equiv_sched = full_exact_scheduler(all_jobs, machines)
make_timeline_plot(
    mip_equiv_sched, arrivals,
    title="MIP‐Equivalent Full Enumeration Timeline",
    filename="mip_enum_timeline.png"
)
# MIP formulation
make_timeline_plot(
    schedule, arrival_events,
    title="MIP‐Scheduler Timeline",
    filename="mip_scheduler_timeline.png"
)

print("MIP Schedule (job, start, end) per machine:")
for m in machines:
    print(f"  {m}: {schedule[m]}")

# ──────────── 6) Print‐out for Comparison ────────────

print("Exact (arrival‐order) Schedule:")
for m in sorted(exact_sched):
    sorted_by_start = sorted(exact_sched[m], key=lambda x: x[1])
    print(f"  {m}: {sorted_by_start}")

print("\nMIP‐Equivalent (full enumeration) Schedule:")
for m in sorted(mip_equiv_sched):
    sorted_by_start = sorted(mip_equiv_sched[m], key=lambda x: x[1])
    print(f"  {m}: {sorted_by_start}")
