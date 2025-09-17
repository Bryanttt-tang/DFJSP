import heapq
from collections import deque, defaultdict
from typing import List, Tuple

# --- Problem definition ---
# jobs_data[j] = list of (machine_id, processing_time) for job j
jobs_data: List[List[Tuple[int,int]]] = [
    [(0, 3), (1, 2), (2, 2)],  # Job 0
    [(0, 2), (2, 1), (1, 4)],  # Job 1
    [(1, 4), (2, 3)],          # Job 2, release at t=3
]
release_times = [0, 0, 3]
num_machines = 3

# --- Dispatching rule: Shortest Processing Time (SPT) ---
def select_spt(ready_ops):
    # ready_ops: list of (processing_time, job, task_idx)
    return min(ready_ops, key=lambda x: x[0])

# --- Simulation state ---
time = 0
# Next events: (event_time, machine_id)
machine_events = [(0, m) for m in range(num_machines)]
heapq.heapify(machine_events)

# Track each job's next operation index
next_op = [0]*len(jobs_data)
# Queues of ready operations per machine
ready_queue = defaultdict(list)

# Record start/end times
start_times = {}
end_times = {}

# --- Main simulation loop ---
while machine_events:
    cur_time, m_id = heapq.heappop(machine_events)
    time = cur_time

    # 1) Add newly released jobs' first ops
    for j, r in enumerate(release_times):
        if r == time and next_op[j] == 0:
            p = jobs_data[j][0][1]
            ready_queue[jobs_data[j][0][0]].append((p, j, 0))

    # 2) When a task finishes, its successor (if any) becomes ready
    for (j,k), e in list(end_times.items()):
        if e == time and next_op[j] == k+1 < len(jobs_data[j]):
            m_next, p_next = jobs_data[j][k+1]
            ready_queue[m_next].append((p_next, j, k+1))

    # 3) Dispatch on machine m_id if any ready ops
    if ready_queue[m_id]:
        p, j, k = select_spt(ready_queue[m_id])
        ready_queue[m_id].remove((p,j,k))

        # Schedule it
        s = time
        e = s + p
        start_times[(j,k)] = s
        end_times[(j,k)] = e
        next_op[j] = k+1

        # Enqueue machine free event
        heapq.heappush(machine_events, (e, m_id))

# --- Output results ---
makespan = max(end_times.values())
print(f"Schedule complete. Makespan = {makespan}\n")
for (j,k), s in sorted(start_times.items()):
    m, p = jobs_data[j][k]
    print(f"Job {j}, Task {k} on Machine {m}: start at {s}, end at {end_times[(j,k)]}")
