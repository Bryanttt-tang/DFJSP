import random
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

# --- Dynamic FJSP Framework ---

class Operation:
    def __init__(self, job_id, op_id, machine_options):
        self.job_id = job_id
        self.op_id = op_id
        self.machine_options = machine_options  # list of (machine_id, proc_time)
        self.assigned_machine = None
        self.start_time = None
        self.end_time = None

class Job:
    def __init__(self, job_id, release_time, operations):
        self.job_id = job_id
        self.release_time = release_time
        self.operations = deque(operations)

class Scheduler:
    def schedule(self, jobs, machines, current_time):
        raise NotImplementedError

class DispatchRuleScheduler(Scheduler):
    # Greedy: pick shortest processing time among available ops on idle machines
    def schedule(self, jobs, machines, current_time):
        schedule = []
        # find available operations
        available_ops = [(job.operations[0], job) for job in jobs
                         if job.release_time <= current_time and job.operations]
        # sort by shortest processing time across machine options
        available_ops.sort(key=lambda xo: min(pt for mid, pt in xo[0].machine_options))
        
        for op, job in available_ops:
            # find earliest available machine
            best_mid, best_pt = min(op.machine_options, key=lambda x: x[1])
            if machines[best_mid] <= current_time:
                op.assigned_machine = best_mid
                op.start_time = current_time
                op.end_time = current_time + best_pt
                machines[best_mid] = op.end_time
                schedule.append(op)
                job.operations.popleft()
        return schedule

class GAScheduler(Scheduler):
    # Simple GA at each dispatching point
    def schedule(self, jobs, machines, current_time, pop_size=10, gens=20):
        ops = [job.operations[0] for job in jobs
               if job.release_time <= current_time and job.operations]
        if not ops:
            return []
        
        best_schedule = None
        best_obj = float('inf')
        
        for _ in range(gens):
            pop = []
            for _ in range(pop_size):
                assignment = {op: random.choice(op.machine_options) for op in ops}
                sim_machines = machines.copy()
                makespan = current_time
                for op in ops:
                    mid, pt = assignment[op]
                    start = max(sim_machines[mid], current_time)
                    makespan = max(makespan, start + pt)
                    sim_machines[mid] = start + pt
                pop.append((assignment, makespan))
            gen_best = min(pop, key=lambda x: x[1])
            if gen_best[1] < best_obj:
                best_obj = gen_best[1]
                best_schedule = gen_best[0]
        
        scheduled = []
        for op, (mid, pt) in best_schedule.items():
            op.assigned_machine = mid
            op.start_time = current_time
            op.end_time = current_time + pt
            machines[mid] = op.end_time
            # remove operation from the corresponding job
            for job in jobs:
                if job.job_id == op.job_id:
                    job.operations.popleft()
                    break
            scheduled.append(op)
        return scheduled

# --- Simulation and Visualization ---

def simulate(dynamic_jobs, scheduler, num_machines=2):
    current_time = 0
    machines = {m: 0 for m in range(num_machines)}
    all_ops = []
    pending_jobs = []
    release_times = sorted(dynamic_jobs, key=lambda j: j.release_time)
    
    while release_times or any(job.operations for job in pending_jobs):
        # add newly released jobs
        while release_times and release_times[0].release_time <= current_time:
            pending_jobs.append(release_times.pop(0))
        # schedule
        ops = scheduler.schedule(pending_jobs, machines, current_time)
        if ops:
            all_ops.extend(ops)
        # advance time
        next_times = [machines[m] for m in machines]
        if release_times:
            next_times.append(release_times[0].release_time)
        current_time = min(t for t in next_times if t > current_time)
    return all_ops

def plot_gantt(ops, num_machines=2):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10.colors
    for op in ops:
        ax.barh(f"Machine {op.assigned_machine}", op.end_time - op.start_time, left=op.start_time,
                color=colors[op.job_id % len(colors)], edgecolor='black')
        ax.text(op.start_time + (op.end_time - op.start_time)/2, op.assigned_machine,
                f"J{op.job_id}O{op.op_id}", va='center', ha='center', color='white')
    ax.set_xlabel("Time")
    ax.set_title("Dynamic FJSP Gantt Chart")
    plt.tight_layout()
    plt.show()

def plot_obj(ops, title="Completion Times by Job"):
    end_times = {op.job_id: [] for op in ops}
    for op in ops:
        end_times[op.job_id].append(op.end_time)
    jobs = sorted(end_times.keys())
    comp_times = [max(end_times[j]) for j in jobs]
    plt.figure(figsize=(6, 4))
    plt.plot(jobs, comp_times, marker='o')
    plt.xlabel("Job ID")
    plt.ylabel("Completion Time")
    plt.title(title)
    plt.show()

# --- Demo Execution ---
# Job is a class def __init__(self, job_id, release_time, operations)
# Operation is a class def __init__(self, job_id, op_id, machine_options), # machine_options= list of (machine_id, proc_time)
dynamic_jobs = [
    Job(0, 0, [Operation(0, 0, [(0,3),(1,4)]), Operation(0,1, [(0,2),(1,1)])]),
    Job(1, 2, [Operation(1, 0, [(0,2),(1,2)]), Operation(1,1, [(0,1),(1,3)])]),
    Job(2, 4, [Operation(2, 0, [(0,4),(1,3)]), Operation(2,1, [(0,3),(1,2)])]),
]

ops_dr = simulate(dynamic_jobs, DispatchRuleScheduler())
plot_gantt(ops_dr)
plot_obj(ops_dr, title="Dispatch Rule Completion Times")

# Re-create jobs for fair comparison
dynamic_jobs = [
    Job(0, 0, [Operation(0, 0, [(0,3),(1,4)]), Operation(0,1, [(0,2),(1,1)])]),
    Job(1, 2, [Operation(1, 0, [(0,2),(1,2)]), Operation(1,1, [(0,1),(1,3)])]),
    Job(2, 4, [Operation(2, 0, [(0,4),(1,3)]), Operation(2,1, [(0,3),(1,2)])]),
]

ops_ga = simulate(dynamic_jobs, GAScheduler())
plot_gantt(ops_ga)
plot_obj(ops_ga, title="GA Scheduler Completion Times")

# Compare average completion times
avg_dr = np.mean([max(end_times) for end_times in 
                  [[op.end_time for op in ops_dr if op.job_id == j] for j in set(op.job_id for op in ops_dr)]])
avg_ga = np.mean([max(end_times) for end_times in 
                  [[op.end_time for op in ops_ga if op.job_id == j] for j in set(op.job_id for op in ops_ga)]])
print(f"Avg Completion Time - Dispatch Rule: {avg_dr:.2f}, GA: {avg_ga:.2f}")
