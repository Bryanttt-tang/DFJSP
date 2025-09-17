import collections
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD

def heuristic_spt_scheduler(jobs_data, machine_list, job_arrival_times):
    """
    Schedules jobs based on the Shortest Processing Time (SPT) heuristic,
    considering dynamic job arrivals.
    """
    print("\n--- Running SPT Heuristic Scheduler ---")
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    
    schedule = {m: [] for m in machine_list}
    operations_scheduled_count = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    
    arrived_jobs = {job_id for job_id, arrival in job_arrival_times.items() if arrival <= 0}
    
    current_time = 0
    while operations_scheduled_count < total_operations:
        candidate_operations = []
        
        # Update arrived jobs based on current time
        arrived_jobs.update({j_id for j_id, arrival in job_arrival_times.items() if arrival <= current_time})

        for job_id in arrived_jobs:
            op_idx = next_operation_for_job[job_id]
            if op_idx < len(jobs_data[job_id]):
                op_data = jobs_data[job_id][op_idx]
                job_ready_time = operation_end_times[job_id][op_idx - 1] if op_idx > 0 else job_arrival_times[job_id]
                
                for machine_name, proc_time in op_data['proc_times'].items():
                    earliest_start_time = max(machine_next_free[machine_name], job_ready_time)
                    if earliest_start_time <= current_time:
                        candidate_operations.append((
                            proc_time,
                            earliest_start_time,
                            job_id, 
                            op_idx, 
                            machine_name
                        ))
        
        if not candidate_operations:
            # If no operations can be scheduled at current_time, advance time
            min_next_free = min(machine_next_free.values())
            upcoming_arrivals = [arr for arr in job_arrival_times.values() if arr > current_time]
            next_event_time = min(upcoming_arrivals) if upcoming_arrivals else float('inf')
            current_time = max(min_next_free, next_event_time) if next_event_time != float('inf') else min_next_free
            continue

        # Select the operation with the shortest processing time
        selected_op = min(candidate_operations, key=lambda x: x[0])
        proc_time, start_time, job_id, op_idx, machine_name = selected_op
        
        end_time = start_time + proc_time

        machine_next_free[machine_name] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled_count += 1
        
        schedule[machine_name].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"SPT Heuristic Makespan: {makespan:.2f}")
    return makespan, schedule

def milp_scheduler(jobs, machines, arrival_times):
    """MILP approach for optimal dynamic scheduling."""
    print("\n--- Running MILP Optimal Scheduler ---")
    prob = LpProblem("DynamicFJSP_Optimal", LpMinimize)
    
    ops = [(j, oi) for j in jobs for oi in range(len(jobs[j]))]
    BIG_M = 1000 

    x = LpVariable.dicts("x", (ops, machines), cat="Binary")
    s = LpVariable.dicts("s", ops, lowBound=0)
    c = LpVariable.dicts("c", ops, lowBound=0)
    y = LpVariable.dicts("y", (ops, ops, machines), cat="Binary")
    Cmax = LpVariable("Cmax", lowBound=0)

    prob += Cmax

    for j, oi in ops:
        valid_machines = [m for m in machines if m in jobs[j][oi]['proc_times']]
        # Assignment constraint
        prob += lpSum(x[j, oi][m] for m in valid_machines) == 1
        # Completion time
        prob += c[j, oi] == s[j, oi] + lpSum(x[j, oi][m] * jobs[j][oi]['proc_times'][m] for m in valid_machines)
        # Precedence within a job
        if oi > 0:
            prob += s[j, oi] >= c[j, oi - 1]
        # Arrival time constraint
        else:
            prob += s[j, oi] >= arrival_times[j]
        # Makespan definition
        prob += Cmax >= c[j, oi]

    for m in machines:
        ops_on_m = [op for op in ops if m in jobs[op[0]][op[1]]['proc_times']]
        for i in range(len(ops_on_m)):
            for k in range(i + 1, len(ops_on_m)):
                op1, op2 = ops_on_m[i], ops_on_m[k]
                # Disjunctive constraints
                prob += s[op1] >= c[op2] - BIG_M * (1 - y[op1][op2][m]) - BIG_M * (2 - x[op1][m] - x[op2][m])
                prob += s[op2] >= c[op1] - BIG_M * y[op1][op2][m] - BIG_M * (2 - x[op1][m] - x[op2][m])

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=120)) # 2-minute time limit

    schedule = {m: [] for m in machines}
    if Cmax.varValue is not None:
        for (j, oi), m in ((op, m) for op in ops for m in jobs[op[0]][op[1]]['proc_times']):
            if x[j, oi][m].varValue > 0.5:
                schedule[m].append((f"J{j}-O{oi+1}", s[j, oi].varValue, c[j, oi].varValue))
        print(f"MILP (optimal) Makespan: {Cmax.varValue:.2f}")
        return Cmax.varValue, schedule
    print("MILP solver failed to find an optimal solution within the time limit.")
    return float('inf'), schedule
