import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import itertools

# ====================
# Problem Definition
# ====================
domain_machines = ["M1", "M2", "M3"]
jobs_initial = {
    1: {'proc_times': {'M1': 10, 'M2': 12, 'M3': 15}, 'arrival': 0},
    2: {'proc_times': {'M1': 8,  'M2': 14, 'M3': 11}, 'arrival': 2},
    3: {'proc_times': {'M1': 9,  'M2': 10, 'M3': 13}, 'arrival': 3},
    4: {'proc_times': {'M1': 11, 'M2': 9,  'M3': 14}, 'arrival': 5},
    5: {'proc_times': {'M1': 12, 'M2': 13, 'M3': 10}, 'arrival': 6}
}

# ====================
# Visualization: Initial Scenario
# ====================
def plot_initial(jobs):
    df = pd.DataFrame({jid: data['proc_times'] for jid, data in jobs.items()}).T
    df['arrival'] = [data['arrival'] for data in jobs.values()]
    print("Initial Job Data:")
    print(df.sort_values('arrival'), "\n")

    avg_times = [sum(d['proc_times'].values())/len(d['proc_times']) for d in jobs.values()]
    plt.figure(figsize=(8,4))
    plt.scatter([d['arrival'] for d in jobs.values()], list(jobs.keys()), s=[t*10 for t in avg_times])
    for jid, d in jobs.items():
        plt.text(d['arrival'] + 0.1, jid + 0.1, f"J{jid}")
    plt.xlabel('Arrival Time')
    plt.ylabel('Job ID')
    plt.title('Initial Job Arrivals')
    plt.grid(True)
    plt.show()

# ====================
# Heuristic Scheduler
# ====================
def heuristic_schedule(jobs, machines):
    start_time = time.time()
    schedule = {m: [] for m in machines}
    completion = {}
    for jid, data in sorted(jobs.items(), key=lambda x: x[1]['arrival']):
        ready = data['arrival']
        best_m, best_s = None, float('inf')
        for m, p in data['proc_times'].items():
            last = schedule[m][-1][2] if schedule[m] else 0
            s = max(last, ready)
            if s < best_s:
                best_s, best_m = s, m
        e = best_s + data['proc_times'][best_m]
        schedule[best_m].append((jid, best_s, e))
        completion[jid] = e
    makespan = max(completion.values())
    return schedule, makespan, time.time() - start_time

# ====================
# Genetic Algorithm Scheduler
# ====================
def ga_schedule(jobs, machines, pop=30, gens=50, mr=0.2):
    def eval_seq(seq):
        sch, comp = {m: [] for m in machines}, {}
        for jid in seq:
            data = jobs[jid]; ready = data['arrival']
            best_m, best_s = None, float('inf')
            for m, p in data['proc_times'].items():
                last = sch[m][-1][2] if sch[m] else 0
                s = max(last, ready)
                if s < best_s:
                    best_s, best_m = s, m
            e = best_s + data['proc_times'][best_m]
            sch[best_m].append((jid, best_s, e)); comp[jid] = e
        return sch, max(comp.values())

    t0 = time.time()
    popu = [random.sample(list(jobs), len(jobs)) for _ in range(pop)]
    for _ in range(gens):
        scored = sorted((eval_seq(ind)[1], ind) for ind in popu)
        popu = [ind for _, ind in scored[:pop//2]]
        kids = []
        while len(kids) < pop - len(popu):
            p1, p2 = random.sample(popu, 2); c = []
            cut = random.randint(1, len(jobs)-2)
            head = p1[:cut]; tail = [j for j in p2 if j not in head]
            kids.append(head + tail)
        popu += kids
        for ind in popu[1:]:
            if random.random() < mr:
                i,j = random.sample(range(len(jobs)),2); ind[i],ind[j] = ind[j],ind[i]
    best = min(popu, key=lambda ind: eval_seq(ind)[1])
    sched, mks = eval_seq(best)
    return sched, mks, time.time() - t0

# ====================
# Exact Exhaustive Scheduler
# ====================
def exact_schedule(jobs, machines):
    t0 = time.time()
    best_mk = float('inf'); best_sch = None
    # For each assignment of job->machine
    for assignment in itertools.product(*[list(jobs[j]['proc_times'].keys()) for j in jobs]):
        sch = {m: [] for m in machines}; comp = {}
        # build assignment map
        amap = {j: assignment[i] for i,j in enumerate(jobs.keys())}
        # schedule jobs in arrival order
        for j in sorted(jobs, key=lambda x: jobs[x]['arrival']):
            ready = jobs[j]['arrival']; m = amap[j]
            p = jobs[j]['proc_times'][m]
            last = sch[m][-1][2] if sch[m] else 0
            s = max(last, ready); e = s + p
            sch[m].append((j, s, e)); comp[j] = e
        mk = max(comp.values())
        if mk < best_mk:
            best_mk = mk; best_sch = sch
    return best_sch, best_mk, time.time() - t0

# ====================
# Gantt Plot with Legends
# ====================
def plot_three(s1,s2,s3,m1,m2,m3,title):
    fig, axs = plt.subplots(1,3,figsize=(15,5), sharey=True)
    scheds = [s1, s2, s3]; ms = [m1, m2, m3]
    colors = plt.cm.tab20.colors
    methods = ['Heuristic','GA','Exact']
    for ax, sch, mk, name in zip(axs, scheds, ms, methods):
        for i,m in enumerate(domain_machines):
            for (j,s,e) in sch[m]:
                ax.broken_barh([(s,e-s)], (i*10,8), facecolors=colors[j%20], label=f"J{j}")
        ax.set_title(f"{name} (M={mk})"); ax.set_yticks([i*10+4 for i in range(len(domain_machines))])
        ax.set_yticklabels(domain_machines); ax.legend(loc='upper right')
        ax.set_xlabel('Time')
    axs[0].set_ylabel('Machine'); fig.suptitle(title); plt.tight_layout(); plt.show()

# ====================
# Main
# ====================
if __name__ == '__main__':
    plot_initial(jobs_initial)
    s_h,m_h,t_h = heuristic_schedule(jobs_initial, domain_machines)
    s_ga,m_ga,t_ga = ga_schedule(jobs_initial, domain_machines)
    s_ex,m_ex,t_ex = exact_schedule(jobs_initial, domain_machines)
    print(f"Heuristic: M={m_h}, t={t_h:.3f}s")
    print(f"GA:        M={m_ga}, t={t_ga:.3f}s")
    print(f"Exact:     M={m_ex}, t={t_ex:.3f}s")
    plot_three(s_h,s_ga,s_ex,m_h,m_ga,m_ex,"Initial Comparison")

    # Dynamic timeline with 3 arrivals
    jobs1 = dict(jobs_initial)
    jobs2 = dict(jobs1); jobs2[6] = {'proc_times':{'M1':14,'M2':10,'M3':12},'arrival':4}
    jobs3 = dict(jobs2); jobs3[7] = {'proc_times':{'M1':9,'M2':11,'M3':13},'arrival':8}
    jobs4 = dict(jobs3); jobs4[8] = {'proc_times':{'M1':13,'M2':12,'M3':14},'arrival':12}
    arrivals = [('Init',jobs1),('+J6',jobs2),('+J7',jobs3),('+J8',jobs4)]
        # plot each stage for heuristic only, with arrival indicators
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    for ax_idx, (lbl, jset) in enumerate(arrivals):
        sch, _, _ = heuristic_schedule(jset, domain_machines)
        # Plot Gantt bars
        for i, m in enumerate(domain_machines):
            for j, s, e in sch[m]:
                ax = axs[ax_idx]
                ax.broken_barh([(s, e-s)], (i*10, 8), facecolors=plt.cm.tab20.colors[j % 20], label=f"J{j}")
        # Mark arrival time of the new job if not initial
        if ax_idx > 0:
            # arrivals list holds ('+J6', jobs2) etc; extract arrival of the last job
            new_job_id = list(arrivals[ax_idx][1].keys())[-1]
            arr_time = arrivals[ax_idx][1][new_job_id]['arrival']
            axs[ax_idx].axvline(arr_time, color='red', linestyle='--', label=f"Arrival J{new_job_id}")
        axs[ax_idx].set_ylabel(lbl)
        axs[ax_idx].set_yticks([i*10+4 for i in range(len(domain_machines))])
        axs[ax_idx].set_yticklabels(domain_machines)
        axs[ax_idx].legend(loc='upper right')
    axs[-1].set_xlabel('Time')
    fig.suptitle('Dynamic Timeline (Heuristic)')
    plt.tight_layout()
    plt.show()


    fig2, axs2 = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    for ax_idx, (lbl, jset) in enumerate(arrivals):
        sch, _, _ = exact_schedule(jset, domain_machines)
        # Plot Gantt bars
        for i, m in enumerate(domain_machines):
            for j, s, e in sch[m]:
                ax = axs2[ax_idx]
                ax.broken_barh([(s, e-s)], (i*10, 8), facecolors=plt.cm.tab20.colors[j % 20], label=f"J{j}")
        # Mark arrival time of the new job if not initial
        if ax_idx > 0:
            # arrivals list holds ('+J6', jobs2) etc; extract arrival of the last job
            new_job_id = list(arrivals[ax_idx][1].keys())[-1]
            arr_time = arrivals[ax_idx][1][new_job_id]['arrival']
            axs2[ax_idx].axvline(arr_time, color='red', linestyle='--', label=f"Arrival J{new_job_id}")
        axs2[ax_idx].set_ylabel(lbl)
        axs2[ax_idx].set_yticks([i*10+4 for i in range(len(domain_machines))])
        axs2[ax_idx].set_yticklabels(domain_machines)
        axs2[ax_idx].legend(loc='upper right')
    axs2[-1].set_xlabel('Time')
    fig2.suptitle('Dynamic Timeline (Exact)')
    plt.tight_layout()
    plt.show()