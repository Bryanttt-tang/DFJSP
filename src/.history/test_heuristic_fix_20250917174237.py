#!/usr/bin/env python3
"""
Quick test to verify the improved heuristic implementation
"""

import sys
import os

# Add the src directory to Python path
sys.path.append('/Users/tanu/Desktop/PhD/Scheduling/src')

def test_heuristics():
    """Test the improved heuristic implementations"""
    try:
        from clean_dynamic_vs_static_comparison import (
            ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES,
            simple_spt_heuristic, basic_greedy_scheduler
        )
        
        print("Testing improved heuristic implementations...")
        print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
        print(f"Arrival times: {DETERMINISTIC_ARRIVAL_TIMES}")
        
        # Test simple SPT
        print("\n1. Testing Simple SPT Heuristic:")
        spt_makespan, spt_schedule = simple_spt_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
        print(f"   Simple SPT makespan: {spt_makespan:.2f}")
        
        # Test basic greedy
        print("\n2. Testing Basic Greedy Scheduler:")
        greedy_makespan, greedy_schedule = basic_greedy_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
        print(f"   Basic Greedy makespan: {greedy_makespan:.2f}")
        
        # Use the better of the two as our baseline
        if greedy_makespan < spt_makespan:
            best_makespan = greedy_makespan
            best_schedule = greedy_schedule
            best_name = "Basic Greedy"
        else:
            best_makespan = spt_makespan
            best_schedule = spt_schedule
            best_name = "Simple SPT"
        
        print(f"\nResults Summary:")
        print(f"Simple SPT: {spt_makespan:.2f}")
        print(f"Basic Greedy: {greedy_makespan:.2f}")
        print(f"Best ({best_name}): {best_makespan:.2f}")
        print(f"Expected optimal range: 35-45")
        
        # Debug: Check if simple SPT schedule looks reasonable
        print(f"\nDebug: Simple SPT Schedule Summary:")
        for machine, ops in spt_schedule.items():
            if ops:
                total_time = sum(op[2] - op[1] for op in ops)
                print(f"  {machine}: {len(ops)} operations, total time: {total_time:.1f}")
        
        if spt_makespan < 50 and best_makespan < 60:
            print("✅ Heuristics are now performing reasonably!")
            return True
        else:
            print("❌ Heuristics still have performance issues")
            print("   Issue might be in time advancement or precedence handling")
            return False
            
    except Exception as e:
        print(f"Error testing heuristics: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_heuristics()
    sys.exit(0 if success else 1)