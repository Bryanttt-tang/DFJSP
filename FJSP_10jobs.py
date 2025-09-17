import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# --- Problem Data Generation ---
NUM_JOBS = 10
NUM_MACHINES = 6
MAX_RELEASE = 20
MAX_OPS_PER_JOB = 5
MAX_PROC_TIME = 10
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

class Operation:
    def __init__(self, job_id, op_id, machine_options):
        self.job_id = job_id
        self.op_id = op_id
        self.machine_options = machine_options
        self.assigned_machine = None
        self.start_time = None
        self.end_time = None

class Job:
    def __init__(self, job_id, release_time, operations):
        self.job_id = job_id
        self.release_time = release_time
        self.operations = deque(operations)

# Create 10 dynamic jobs with random releases and random machine options
dynamic_jobs_template = []
for j in range(NUM_JOBS):
    r = random.randint(0, MAX_RELEASE)
    num_ops = random.randint(1, MAX_OPS_PER_JOB)
    ops = []
    for o in range(num_ops):
        # Each operation can go on 2-4 random machines
        machines = random.sample(range(NUM_MACHINES), random.randint(2, 4))
        machine_options = [(m, random.randint(1, MAX_PROC_TIME)) for m in machines]
        ops.append(Operation(j, o, machine_options))
    dynamic_jobs_template.append(Job(j, r, ops))

# --- Scheduler Interface ---
class Scheduler:
    def schedule(self, jobs, machines, current_time):
        raise NotImplementedError

# Weighted-Sum GA Dispatcher at each epoch
class WeightedGAScheduler(Scheduler):
    def __init__(self, weight, pop_size=20, gens=30):
        self.w = weight
        self.pop = pop_size
        self.gens = gens

    def schedule(self, jobs, machines, current_time):
        # collect available operations
        avail = [job.operations[0] for job in jobs 
                 if job.release_time <= current_time and job.operations]
        if not avail:
            return []

        # GA population: assignment dicts mapping op->(machine,pt)
        best_assign, best_score = None, float('inf')
        for _ in range(self.gens):
            popu = []
            for _ in range(self.pop):
                assign = {op: random.choice(op.machine_options) for op in avail}
                # simulate to compute objectives
                sim_m = machines.copy()
                comp_times = []
                flow_sum = 0
                for op in avail:
                    m, pt = assign[op]
                    start = max(sim_m[m], current_time)
                    end = start + pt
                    sim_m[m] = end
                    comp_times.append(end)
                    flow_sum += (end - op.job_id)  # release->end approx
                makespan = max(comp_times)
                avg_flow = flow_sum / len(avail)
                score = self.w * makespan + (1 - self.w) * avg_flow
                popu.append((assign, score, makespan, avg_flow))
            # pick best in generation
            gen_best = min(popu, key=lambda x: x[1])
            if gen_best[1] < best_score:
                best_assign, best_score = gen_best[0], gen_best[1]

        # apply best assignment
        scheduled = []
        for op, (m, pt) in best_assign.items():
            op.assigned_machine = m
            op.start_time = current_time
            op.end_time = current_time + pt
            machines[m] = op.end_time
            # remove op from its job
            for job in jobs:
                if job.job_id == op.job_id:
                    job.operations.popleft()
                    break
            scheduled.append(op)
        return scheduled

# Simulation engine
def simulate(jobs_template, scheduler):
    # deep copy template
    jobs = [Job(j.job_id, j.release_time, [Operation(op.job_id, op.op_id, list(op.machine_options))
             for op in j.operations]) for j in jobs_template]
    time = 0
    machines = {m:0 for m in range(NUM_MACHINES)}
    completed_ops = []

    pending = []
    release_list = sorted(jobs, key=lambda j:j.release_time)
    while release_list or any(job.operations for job in pending):
        # add newly released
        while release_list and release_list[0].release_time <= time:
            pending.append(release_list.pop(0))
        # dispatch
        ops_run = scheduler.schedule(pending, machines, time)
        completed_ops += ops_run
        # advance to next event
        next_times = [machines[m] for m in machines]
        if release_list:
            next_times.append(release_list[0].release_time)
        time = min(t for t in next_times if t > time)
    # compute objectives
    job_ends = {}
    for op in completed_ops:
        job_ends.setdefault(op.job_id, []).append(op.end_time)
    makespan = max(max(ts) for ts in job_ends.values())
    avg_flow = np.mean([max(ts)-j.release_time for ts,j in zip(job_ends.values(), jobs_template)])
    return makespan, avg_flow, completed_ops

# --- Sweep Weights & Collect Pareto Data ---
weights = np.linspace(0,1,11)
pareto_data = []
for w in weights:
    sched = WeightedGAScheduler(w)
    Cmax, AF, ops = simulate(dynamic_jobs_template, sched)
    pareto_data.append((Cmax, AF, ops))

# Extract for plotting
cmax_vals = [d[0] for d in pareto_data]
af_vals   = [d[1] for d in pareto_data]

# Pareto front: find non-dominated
pareto_idx = []
for i,(c,a,_) in enumerate(pareto_data):
    dominated = any((c2<=c and a2<=a) and (c2<c or a2<a) for j,(c2,a2,_) in enumerate(pareto_data) if j!=i)
    if not dominated:
        pareto_idx.append(i)

# --- Visualization ---

# 1) Pareto Front Scatter
plt.figure(figsize=(6,4))
plt.scatter(cmax_vals, af_vals, label='All solutions')
plt.scatter([cmax_vals[i] for i in pareto_idx],
            [af_vals[i] for i in pareto_idx],
            color='red', label='Pareto front')
plt.xlabel('Makespan')
plt.ylabel('Avg Flow Time')
plt.title('Pareto Front: Makespan vs. Avg Flow Time')
plt.legend()
plt.tight_layout()
plt.show()

# 2) Gantt Chart for one Pareto-optimal point
idx = pareto_idx[0]
_,_,best_ops = pareto_data[idx]
fig, ax = plt.subplots(figsize=(8,4))
cols = plt.cm.tab10.colors
for op in best_ops:
    ax.barh(f"Machine {op.assigned_machine}",
            op.end_time-op.start_time,
            left=op.start_time,
            color=cols[op.job_id % len(cols)], edgecolor='black')
    ax.text(op.start_time + 0.5*(op.end_time-op.start_time),
            op.assigned_machine,
            f"J{op.job_id}O{op.op_id}",
            ha='center', va='center', color='white')
ax.set_xlabel('Time')
ax.set_title(f'Gantt Chart (w={weights[idx]:.1f})')
plt.tight_layout()
plt.show()
