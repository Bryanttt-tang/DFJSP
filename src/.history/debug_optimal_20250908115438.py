#!/usr/bin/env python3

import collections

# Simple problem from dynamic-tutorial.py
SIMPLE_JOBS_DATA = collections.OrderedDict({
    1: [
        {'proc_times': {'M1': 2, 'M2': 4, 'M3': 3}},  # J1-O1
        {'proc_times': {'M1': 3, 'M2': 2, 'M3': 4}}   # J1-O2
    ],
    2: [
        {'proc_times': {'M1': 4, 'M2': 3, 'M3': 2}},  # J2-O1
        {'proc_times': {'M1': 2, 'M2': 3, 'M3': 4}}   # J2-O2
    ]
})
SIMPLE_MACHINES = ['M1', 'M2', 'M3']

def verify_optimal_schedule():
    """
    Manually verify the optimal schedule with makespan 4
    """
    print("=== VERIFYING OPTIMAL SCHEDULE ===")
    print("Problem:")
    for job_id, operations in SIMPLE_JOBS_DATA.items():
        print(f"  Job {job_id}:")
        for op_idx, op in enumerate(operations):
            proc_times_str = ", ".join([f"{m}({t})" for m, t in op['proc_times'].items()])
            print(f"    Operation {op_idx+1}: {proc_times_str}")
    print()
    
    # Optimal schedule: makespan = 4
    # J1-O1 on M1: 0->2
    # J2-O1 on M3: 0->2  
    # J1-O2 on M2: 2->4
    # J2-O2 on M1: 2->4
    
    print("Optimal Schedule (should give makespan = 4):")
    print("  J1-O1 on M1: 0 -> 2 (proc_time=2)")
    print("  J2-O1 on M3: 0 -> 2 (proc_time=2)")
    print("  J1-O2 on M2: 2 -> 4 (proc_time=2)")
    print("  J2-O2 on M1: 2 -> 4 (proc_time=2)")
    print("  Final makespan: 4")
    print()
    
    # Verify this is feasible
    machine_schedule = {
        'M1': [(1, 1, 0, 2), (2, 2, 2, 4)],  # (job, op, start, end)
        'M2': [(1, 2, 2, 4)],
        'M3': [(2, 1, 0, 2)]
    }
    
    print("Machine schedules:")
    for machine, ops in machine_schedule.items():
        print(f"  {machine}: {ops}")
    
    # Check processing times
    print("\nVerifying processing times:")
    for machine, ops in machine_schedule.items():
        for job, op_num, start, end in ops:
            expected_time = SIMPLE_JOBS_DATA[job][op_num-1]['proc_times'][machine]
            actual_time = end - start
            print(f"  J{job}-O{op_num} on {machine}: expected={expected_time}, actual={actual_time}, ✓" if expected_time == actual_time else f"  J{job}-O{op_num} on {machine}: expected={expected_time}, actual={actual_time}, ✗")
    
    # Check precedence constraints
    print("\nVerifying precedence constraints:")
    job_completion_times = {}
    for machine, ops in machine_schedule.items():
        for job, op_num, start, end in ops:
            if job not in job_completion_times:
                job_completion_times[job] = {}
            job_completion_times[job][op_num] = end
    
    for job in [1, 2]:
        if 1 in job_completion_times[job] and 2 in job_completion_times[job]:
            j1_end = job_completion_times[job][1]
            j2_start = None
            # Find when J*-O2 starts
            for machine, ops in machine_schedule.items():
                for j, op_num, start, end in ops:
                    if j == job and op_num == 2:
                        j2_start = start
                        break
            if j2_start is not None and j2_start >= j1_end:
                print(f"  Job {job}: O1 ends at {j1_end}, O2 starts at {j2_start}, ✓")
            else:
                print(f"  Job {job}: O1 ends at {j1_end}, O2 starts at {j2_start}, ✗")
    
    print(f"\nFinal makespan: {max(end for ops in machine_schedule.values() for _, _, _, end in ops)}")

if __name__ == "__main__":
    verify_optimal_schedule()
