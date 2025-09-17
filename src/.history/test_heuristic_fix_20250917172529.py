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
            simple_spt_heuristic, run_heuristic_comparison, improved_dispatching_heuristic
        )
        
        print("Testing improved heuristic implementations...")
        print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
        print(f"Arrival times: {DETERMINISTIC_ARRIVAL_TIMES}")
        
        # Test with all jobs at time 0 first (simpler case)
        static_arrivals = {job_id: 0.0 for job_id in ENHANCED_JOBS_DATA.keys()}
        print(f"Static arrivals (all at t=0): {static_arrivals}")
        
        # Test simple SPT on static case
        print("\n1. Testing Simple SPT Heuristic (Static - all jobs at t=0):")
        spt_static_makespan, _ = simple_spt_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, static_arrivals)
        print(f"   Simple SPT (static) makespan: {spt_static_makespan:.2f}")
        
        # Test simple SPT on dynamic case
        print("\n2. Testing Simple SPT Heuristic (Dynamic arrivals):")
        spt_makespan, spt_schedule = simple_spt_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
        print(f"   Simple SPT (dynamic) makespan: {spt_makespan:.2f}")
        
        # Test two-stage on static case
        print("\n3. Testing Two-Stage Heuristics (Static - all jobs at t=0):")
        spt_lwr_static, _ = improved_dispatching_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, static_arrivals, 'SPT', 'LWR')
        print(f"   SPT + LWR (static): {spt_lwr_static:.2f}")
        
        # Test individual two-stage heuristics on dynamic case
        print("\n4. Testing Two-Stage Heuristics (Dynamic arrivals):")
        
        # SPT + LWR
        spt_lwr_makespan, _ = improved_dispatching_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES, 'SPT', 'LWR')
        print(f"   SPT + LWR (dynamic): {spt_lwr_makespan:.2f}")
        
        # SPT + EAM
        spt_eam_makespan, _ = improved_dispatching_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES, 'SPT', 'EAM')
        print(f"   SPT + EAM (dynamic): {spt_eam_makespan:.2f}")
        
        print(f"\nResults Summary:")
        print(f"Simple SPT (static): {spt_static_makespan:.2f}")
        print(f"Simple SPT (dynamic): {spt_makespan:.2f}")
        print(f"SPT+LWR (static): {spt_lwr_static:.2f}")
        print(f"SPT+LWR (dynamic): {spt_lwr_makespan:.2f}")
        print(f"Expected optimal range: 35-45")
        
        if spt_makespan < 60 and best_makespan < 60:
            print("✅ Heuristics are now performing reasonably!")
            return True
        else:
            print("❌ Heuristics still have performance issues")
            return False
            
    except Exception as e:
        print(f"Error testing heuristics: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_heuristics()
    sys.exit(0 if success else 1)