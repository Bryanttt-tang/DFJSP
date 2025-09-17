import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

# Define machines and jobs
domain_machines = ["M1", "M2", "M3"]
initial_jobs = {
    1: [(3, ["M1","M2"]), (2, ["M2","M3"])],
    2: [(2, ["M2"]), (4, ["M1","M3"])],
    3: [(4, ["M3"])],
    4: [(1, ["M1","M2","M3"]), (3, ["M1"]), (2, ["M2","M3"])],
    5: [(5, ["M2","M3"])],
    6: [(3, ["M1","M3"]), (2, ["M2"])]
}
arrivals = {j: m for j, m in zip(initial_jobs, [0,1,2,3,5,6])}

def heuristic_schedule(jobs, arrivals):
    schedule = {m:[] for m in domain_machines}
    completion_time = {}
    for j in sorted(jobs, key=lambda x: arrivals[x]):
        t_ready = arrivals[j]
        for k, (proc, m_allowed) in enumerate(jobs[j]):
            if k > 0:
                t_ready = completion_time[(j,k-1)]
            best_m, best_start = None, float('inf')
            for m in m_allowed:
                last_end = schedule[m][-1][3] if schedule[m] else t_ready
                start = max(last_end, t_ready)
                if start < best_start:
                    best_start, best_m = start, m
            end = best_start + proc
            schedule[best_m].append((j, k, best_start, end))
            completion_time[(j,k)] = end
    makespan = max(end for ops in schedule.values() for *_, end in ops)
    return schedule, makespan

def exact_makespan(jobs, arrivals):
    prob = LpProblem("FJSP", LpMinimize)
    Cmax = LpVariable('Cmax', lowBound=0)
    S = {}
    for j in jobs:
        for k, (p, m_allowed) in enumerate(jobs[j]):
            for m in m_allowed:
                S[(j,k,m)] = LpVariable(f"S_{j}_{k}_{m}", lowBound=0)
    X = {}
    for j in jobs:
        for k, (p, m_allowed) in enumerate(jobs[j]):
            for m in m_allowed:
                X[(j,k,m)] = LpVariable(f"X_{j}_{k}_{m}", cat='Binary')
    for j in jobs:
        for k, (_, m_allowed) in enumerate(jobs[j]):
            prob += lpSum(X[(j,k,m)] for m in m_allowed) == 1
    for j in jobs:
        for k, (p, m_allowed) in enumerate(jobs[j]):
            prob += lpSum(S[(j,k,m)] + p*X[(j,k,m)] for m in m_allowed) >= arrivals[j]
            prob += lpSum(S[(j,k,m)] + p*X[(j,k,m)] for m in m_allowed) <= Cmax
            if k > 0:
                p_prev = jobs[j][k-1][0]
                prob += (lpSum(S[(j,k-1,mp)] + p_prev*X[(j,k-1,mp)] for mp in jobs[j][k-1][1])
                         <= lpSum(S[(j,k,m)] for m in m_allowed))
    BIG = 1000
    for m in domain_machines:
        ops = [(j,k) for j in jobs for k,(p,ms) in enumerate(jobs[j]) if m in ms]
        for (j1,k1) in ops:
            for (j2,k2) in ops:
                if (j1,k1) < (j2,k2):
                    p1 = jobs[j1][k1][0]
                    p2 = jobs[j2][k2][0]
                    Y = LpVariable(f"Y_{j1}_{k1}_{j2}_{k2}_{m}", cat='Binary')
                    prob += S[(j1,k1,m)] + p1 <= S[(j2,k2,m)] + BIG*(1-Y)
                    prob += S[(j2,k2,m)] + p2 <= S[(j1,k1,m)] + BIG*Y
    prob += Cmax
    prob.solve()
    return prob.status, Cmax.varValue

def plot_gantt(schedule, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4))
    machine_y = {m:i*10 for i,m in enumerate(domain_machines)}
    colors = plt.cm.tab20.colors
    for mi,m in enumerate(domain_machines):
        for (j,k,start,end) in schedule[m]:
            ax.broken_barh([(start, end-start)], (machine_y[m], 9), facecolors=colors[j%20], label=f"Job {j}" if k==0 else "")
    ax.set_yticks([machine_y[m]+5 for m in domain_machines])
    ax.set_yticklabels(domain_machines)
    ax.set_xlabel('Time')
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Initial scheduling
sched_h, makespan_h = heuristic_schedule(initial_jobs, arrivals)
status, makespan_opt = exact_makespan(initial_jobs, arrivals)
print("Heuristic makespan:", makespan_h)
print("Optimal makespan:", makespan_opt)
print("Optimality gap: {:.2f}%".format((makespan_h - makespan_opt)/makespan_opt*100))
plot_gantt(sched_h, f"Initial Schedule (makespan {makespan_h}) vs optimal {makespan_opt}")

# Unexpected new job arrives
new_job = {7: [(4,["M1","M3"]),(3,["M2","M3"])]}
initial_jobs.update(new_job)
arrivals[7] = 4
sched_h2, makespan_h2 = heuristic_schedule(initial_jobs, arrivals)
status2, makespan_opt2 = exact_makespan(initial_jobs, arrivals)
print("\nAfter arrival Heuristic makespan:", makespan_h2)
print("After arrival Optimal makespan:", makespan_opt2)
print("Optimality gap: {:.2f}%".format((makespan_h2 - makespan_opt2)/makespan_opt2*100))
plot_gantt(sched_h2, f"After New Arrival (makespan {makespan_h2}) vs optimal {makespan_opt2}")
