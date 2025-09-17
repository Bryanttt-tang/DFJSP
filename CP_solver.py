from docplex.cp.model import CpoModel
from docplex.cp.constraints import alternative, noOverlap

# Create model
mdl = CpoModel()

# Define machines
machines = list(range(6))  # M0..M5

# Job definitions: (job_id, release_time, [(op1_machines, op1_durs), (op2_machines, op2_durs)])
jobs = []
import random
random.seed(0)

# Ten initial jobs (release_time=0)
for j in range(10):
    ops = []
    for _ in range(2):
        # Randomly pick 2 eligible machines and durations
        elig = random.sample(machines, 2)
        durs = [random.randint(5, 15) for _ in elig]
        ops.append((elig, durs))
    jobs.append((f"J{j}", 0, ops))

# Five dynamic jobs (release_time=20)
for j in range(10, 15):
    ops = []
    for _ in range(2):
        elig = random.sample(machines, 2)
        durs = [random.randint(5, 15) for _ in elig]
        ops.append((elig, durs))
    jobs.append((f"J{j}", 20, ops))

# Create interval variables and optional for dynamic jobs
job_ops = {}
for jid, rel, ops in jobs:
    ivars = []
    for k, (machs, durs) in enumerate(ops):
        # Create alternatives for each machine option
        alts = []
        for m, dur in zip(machs, durs):
            iv = mdl.interval_var(optional=(rel>0),
                                  size=dur,
                                  start=(rel if k==0 else None),
                                  name=f"{jid}_op{k}_M{m}")
            alts.append(iv)
        # Operation is one of its alternatives
        ivars.append(mdl.alternative_var(alts, name=f"{jid}_op{k}"))
    # Precedence within job
    mdl.add(mdl.end_before_start(ivars[0], ivars[1]))
    job_ops[jid] = ivars

# Machine occupancy: for each machine, collect intervals on that machine
for m in machines:
    mdl.add(noOverlap([iv for jid, rel, ops in jobs
                       for iv in job_ops[jid]
                       if iv.get_name().endswith(f"_M{m}")]),
            name=f"noOverlap_M{m}")

# Objective: minimize makespan
mdl.add(mdl.minimize(mdl.max([mdl.end_of(iv) for jid in job_ops for iv in job_ops[jid]])))

# Solve
sol = mdl.solve(TimeLimit=10)
print(sol)

# Extract schedule for visualization
schedule = []
for m in machines:
    for jid, rel, ops in jobs:
        for iv in job_ops[jid]:
            if iv.is_present() and sol.is_present(iv):
                st = sol.get_var_solution(iv).get_start()
                et = sol.get_var_solution(iv).get_end()
                schedule.append((m, iv.get_name().split("_")[0], st, et))
