#!/usr/bin/env python3
"""
Debug test for heuristic - start with static arrivals to isolate the issue
"""

import sys
import os
sys.path.append('/Users/tanu/Desktop/PhD/Scheduling/src')

def debug_test():
    """Debug the heuristic with static arrivals first"""
    from clean_dynamic_vs_static_comparison import (
        ENHANCED_JOBS_DATA, MACHINE_LIST, simple_spt_heuristic
    )
    
    # Test with all jobs arriving at t=0 (static case)
    static_arrivals = {job_id: 0.0 for job_id in ENHANCED_JOBS_DATA.keys()}
    
    print("Debug Test: Static Arrivals (all jobs at t=0)")
    print(f"Jobs: {list(ENHANCED_JOBS_DATA.keys())}")
    print(f"Machines: {MACHINE_LIST}")
    print(f"Total operations: {sum(len(ops) for ops in ENHANCED_JOBS_DATA.values())}")
    
    # Show job structure
    for job_id, operations in ENHANCED_JOBS_DATA.items():
        print(f"Job {job_id}: {len(operations)} operations")
        for i, op in enumerate(operations):
            proc_times = list(op['proc_times'].values())
            print(f"  Op {i+1}: min_time={min(proc_times)}, machines={list(op['proc_times'].keys())}")
    
    # Test SPT heuristic
    makespan, schedule = simple_spt_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, static_arrivals)
    
    print(f"\nResult: Makespan = {makespan:.2f}")
    print("Schedule summary:")
    for machine, ops in schedule.items():
        if ops:
            print(f"  {machine}: {len(ops)} operations, spans {ops[0][1]:.1f} to {ops[-1][2]:.1f}")
        else:
            print(f"  {machine}: 0 operations")
    
    # Check if all operations were scheduled
    total_scheduled = sum(len(ops) for ops in schedule.values())
    total_expected = sum(len(ops) for ops in ENHANCED_JOBS_DATA.values())
    print(f"Operations scheduled: {total_scheduled}/{total_expected}")
    
    return makespan < 50  # Should be much better than 50 with static arrivals

if __name__ == "__main__":
    success = debug_test()
    print(f"{'✅ SUCCESS' if success else '❌ STILL ISSUES'}")