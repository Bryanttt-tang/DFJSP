#!/usr/bin/env python3
"""
Quick test to verify the improved heuristic implementation
"""

import sys
import os

# Add the src directory to Python path
sys.path.append('/Users/tanu/Desktop/PhD/Scheduling/src')

from clean_dynamic_vs_static_comparison import (
    ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES,
    simple_spt_heuristic, run_heuristic_comparison, improved_dispatching_heuristic
)

def test_heuristics():
    """Test the improved heuristic implementations"""
    print("Testing improved heuristic implementations...")
    print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
    print(f"Arrival times: {DETERMINISTIC_ARRIVAL_TIMES}")
    
    # Test simple SPT
    print("\n1. Testing Simple SPT Heuristic:")
    spt_makespan, spt_schedule = simple_spt_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
    print(f"   Simple SPT makespan: {spt_makespan:.2f}")
    
    # Test individual two-stage heuristics
    print("\n2. Testing Two-Stage Heuristics:")
    
    # SPT + LWR
    spt_lwr_makespan, _ = improved_dispatching_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES, 'SPT', 'LWR')
    print(f"   SPT + LWR: {spt_lwr_makespan:.2f}")
    
    # SPT + EAM
    spt_eam_makespan, _ = improved_dispatching_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES, 'SPT', 'EAM')
    print(f"   SPT + EAM: {spt_eam_makespan:.2f}")
    
    # FIFO + LWR
    fifo_lwr_makespan, _ = improved_dispatching_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES, 'FIFO', 'LWR')
    print(f"   FIFO + LWR: {fifo_lwr_makespan:.2f}")
    
    # Test the comparison function
    print("\n3. Testing Heuristic Comparison Function:")
    best_makespan, best_schedule = run_heuristic_comparison(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
    
    print(f"\nResults Summary:")
    print(f"Simple SPT: {spt_makespan:.2f}")
    print(f"Best Two-Stage: {best_makespan:.2f}")
    print(f"Expected optimal range: 35-45")
    
    if spt_makespan < 50 and best_makespan < 50:
        print("✅ Heuristics are now performing reasonably!")
        return True
    else:
        print("❌ Heuristics still have performance issues")
        return False

if __name__ == "__main__":
    success = test_heuristics()
    sys.exit(0 if success else 1)