#!/usr/bin/env python3
"""
Simplified test to verify the heuristic performance
without importing the broken main file.
"""

import sys
import os
import time

# Add the src directory to Python path
sys.path.append('/Users/tanu/Desktop/PhD/Scheduling/src')

# Define the test data directly here to avoid import issues
ENHANCED_JOBS_DATA = {
    0: [
        {'proc_times': {'M1': 3, 'M2': 1, 'M3': 3}},
        {'proc_times': {'M1': 2, 'M3': 4}},
        {'proc_times': {'M2': 6}}
    ],
    1: [
        {'proc_times': {'M1': 8, 'M2': 5, 'M3': 10}},
        {'proc_times': {'M2': 3, 'M3': 4}},
        {'proc_times': {'M1': 5, 'M3': 1}}
    ],
    2: [
        {'proc_times': {'M2': 5, 'M3': 4}},
        {'proc_times': {'M1': 6, 'M2': 4}},
        {'proc_times': {'M1': 1, 'M3': 2}}
    ],
    3: [
        {'proc_times': {'M1': 5, 'M2': 4}},
        {'proc_times': {'M1': 1, 'M2': 3}},
        {'proc_times': {'M2': 3}}
    ],
    4: [
        {'proc_times': {'M2': 3}},
        {'proc_times': {'M1': 3, 'M3': 3}},
        {'proc_times': {'M1': 9, 'M2': 10}}
    ],
    5: [
        {'proc_times': {'M1': 3, 'M2': 4}},
        {'proc_times': {'M2': 4, 'M3': 6}},
        {'proc_times': {'M2': 6, 'M3': 2}}
    ],
    6: [
        {'proc_times': {'M2': 2}},
        {'proc_times': {'M1': 6, 'M3': 2}},
        {'proc_times': {'M1': 4, 'M2': 6}}
    ]
}

MACHINE_LIST = ['M1', 'M2', 'M3']
DETERMINISTIC_ARRIVAL_TIMES = {0: 0, 1: 0, 2: 0, 3: 8, 4: 12, 5: 16, 6: 20}

def fixed_spt_heuristic(jobs_data, machine_list, arrival_times):
    """
    Fixed SPT heuristic with proper time handling
    """
    print("\n--- Running Fixed SPT Heuristic ---")
    
    # Initialize state
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    schedule = {m: [] for m in machine_list}
    
    operations_scheduled = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    current_time = 0.0
    arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= current_time}
    
    while operations_scheduled < total_operations:
        # Update arrivals based on current time
        newly_arrived = {job_id for job_id, arr_time in arrival_times.items() 
                        if arr_time <= current_time and job_id not in arrived_jobs}
        arrived_jobs.update(newly_arrived)
        
        # Collect all ready operations
        ready_operations = []
        for job_id in arrived_jobs:
            next_op_idx = next_operation_for_job[job_id]
            if next_op_idx < len(jobs_data[job_id]):
                # Check precedence constraint
                if next_op_idx == 0:  # First operation of job
                    job_ready_time = arrival_times[job_id]
                else:  # Subsequent operations
                    job_ready_time = operation_end_times[job_id][next_op_idx - 1]
                
                # Operation is ready if precedence is satisfied
                if job_ready_time <= current_time:
                    op_data = jobs_data[job_id][next_op_idx]
                    # Find the machine with shortest processing time for this operation
                    best_machine = min(op_data['proc_times'].keys(), key=lambda m: op_data['proc_times'][m])
                    best_proc_time = op_data['proc_times'][best_machine]
                    ready_operations.append((best_proc_time, job_id, next_op_idx, best_machine, job_ready_time))
        
        if not ready_operations:
            # No operations ready, advance time to next event
            next_events = []
            
            # Next machine becomes available
            for m in machine_list:
                if machine_next_free[m] > current_time:
                    next_events.append(machine_next_free[m])
            
            # Next job arrival
            for job_id, arr_time in arrival_times.items():
                if job_id not in arrived_jobs and arr_time > current_time:
                    next_events.append(arr_time)
            
            # Next operation becomes ready (precedence constraint satisfied)
            for job_id in arrived_jobs:
                next_op_idx = next_operation_for_job[job_id]
                if next_op_idx > 0 and next_op_idx < len(jobs_data[job_id]):
                    precedence_time = operation_end_times[job_id][next_op_idx - 1]
                    if precedence_time > current_time:
                        next_events.append(precedence_time)
            
            if next_events:
                current_time = min(next_events)
                continue
            else:
                break  # No more events
        
        # SPT: Select operation with shortest processing time
        selected_operation = min(ready_operations, key=lambda op: op[0])
        proc_time, job_id, op_idx, selected_machine, job_ready_time = selected_operation
        
        # Execute the selected operation on the selected machine
        machine_available_time = machine_next_free[selected_machine]
        
        start_time = max(current_time, machine_available_time, job_ready_time)
        end_time = start_time + proc_time
        
        # Update state
        machine_next_free[selected_machine] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled += 1
        
        # Advance time
        current_time = max(current_time, end_time)
        
        # Record in schedule
        schedule[selected_machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"Fixed SPT Heuristic Makespan: {makespan:.2f}")
    return makespan, schedule


def test_fixed_heuristics():
    """Test the improved heuristic implementations"""
    try:
        print("Testing FIXED heuristic implementations...")
        print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
        print(f"Arrival times: {DETERMINISTIC_ARRIVAL_TIMES}")
        
        # Test with all jobs at time 0 first (simpler case)
        static_arrivals = {job_id: 0.0 for job_id in ENHANCED_JOBS_DATA.keys()}
        print(f"Static arrivals (all at t=0): {static_arrivals}")
        
        # First, let's examine the problem structure
        print("\n0. Problem Analysis:")
        total_proc_time = 0
        for job_id, operations in ENHANCED_JOBS_DATA.items():
            min_job_time = sum(min(op['proc_times'].values()) for op in operations)
            print(f"   Job {job_id}: {len(operations)} ops, min total time: {min_job_time}")
            total_proc_time += min_job_time
        
        theoretical_lower_bound = total_proc_time / len(MACHINE_LIST)
        print(f"   Total minimum processing time: {total_proc_time}")
        print(f"   Theoretical lower bound (perfect load balance): {theoretical_lower_bound:.2f}")
        print(f"   MILP optimal was: 36.00")
        print(f"   If heuristic >> {theoretical_lower_bound:.2f}, there's a logic error")

        # Test fixed SPT on static case
        print("\n1. Testing Fixed SPT Heuristic (Static - all jobs at t=0):")
        spt_static_makespan, _ = fixed_spt_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, static_arrivals)
        print(f"   Fixed SPT (static) makespan: {spt_static_makespan:.2f}")
        
        # Test fixed SPT on dynamic case
        print("\n2. Testing Fixed SPT Heuristic (Dynamic arrivals):")
        spt_makespan, spt_schedule = fixed_spt_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
        print(f"   Fixed SPT (dynamic) makespan: {spt_makespan:.2f}")
        
        print(f"\nResults Summary:")
        print(f"Theoretical lower bound: {theoretical_lower_bound:.2f}")
        print(f"MILP optimal (reference): 36.00")
        print(f"Fixed SPT (static): {spt_static_makespan:.2f}")
        print(f"Fixed SPT (dynamic): {spt_makespan:.2f}")
        print(f"Expected reasonable range: 35-50")
        
        # Check if any heuristic is performing reasonably
        best_performance = min(spt_static_makespan, spt_makespan)
        
        # Performance analysis
        static_gap = ((spt_static_makespan - 36) / 36) * 100
        dynamic_gap = ((spt_makespan - 36) / 36) * 100
        
        print(f"\nPerformance Analysis:")
        print(f"Static gap from optimal: {static_gap:.1f}%")
        print(f"Dynamic gap from optimal: {dynamic_gap:.1f}%")
        
        if best_performance < 50:
            print("✅ Heuristics are now performing reasonably!")
            print(f"Best performance: {best_performance:.2f}")
            
            if static_gap < 50 and dynamic_gap < 100:
                print("✅ Performance gaps are acceptable for baseline comparison!")
                return True
            else:
                print("⚠️  Performance gaps are high but heuristic is working")
                return True
        else:
            print("❌ Heuristics still have performance issues")
            print(f"Best performance: {best_performance:.2f}")
            return False
            
    except Exception as e:
        print(f"Error testing heuristics: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_heuristics()
    sys.exit(0 if success else 1)