#!/usr/bin/env python3
"""
Quick test of the optimized heuristics
"""

import sys
import os

# Add the src directory to Python path
sys.path.append('/Users/tanu/Desktop/PhD/Scheduling/src')

def test_optimized_heuristics():
    """Test the new optimized heuristic implementations"""
    try:
        from clean_dynamic_vs_static_comparison import (
            ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES,
            simple_spt_heuristic, basic_greedy_scheduler,
            optimized_spt_scheduler, earliest_completion_scheduler
        )
        
        print("Testing all heuristic implementations...")
        print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
        print(f"Arrival times: {DETERMINISTIC_ARRIVAL_TIMES}")
        
        # Test all heuristics
        print("\n=== HEURISTIC PERFORMANCE COMPARISON ===")
        
        results = []
        
        # 1. Simple SPT
        spt_makespan, _ = simple_spt_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
        results.append(("Simple SPT", spt_makespan))
        
        # 2. Basic Greedy
        greedy_makespan, _ = basic_greedy_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
        results.append(("Basic Greedy", greedy_makespan))
        
        # 3. Optimized SPT
        opt_spt_makespan, _ = optimized_spt_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
        results.append(("Optimized SPT", opt_spt_makespan))
        
        # 4. Earliest Completion
        ec_makespan, _ = earliest_completion_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
        results.append(("Earliest Completion", ec_makespan))
        
        # Sort by performance
        results.sort(key=lambda x: x[1])
        
        print(f"\n=== FINAL RESULTS (Ranked by Performance) ===")
        print(f"Expected optimal range: ~35-45")
        print("-" * 50)
        
        for i, (name, makespan) in enumerate(results, 1):
            status = "🎯 EXCELLENT" if makespan < 40 else "✅ GOOD" if makespan < 50 else "⚠️ OK" if makespan < 60 else "❌ POOR"
            print(f"{i}. {name:20s}: {makespan:6.2f} {status}")
        
        best_makespan = results[0][1]
        best_name = results[0][0]
        
        print(f"\n🏆 BEST HEURISTIC: {best_name} (makespan: {best_makespan:.2f})")
        
        # Success criteria
        if best_makespan < 45:
            print("✅ SUCCESS: Best heuristic is within competitive range!")
            return True
        elif best_makespan < 55:
            print("🟡 MODERATE: Best heuristic is acceptable but could be better")
            return True
        else:
            print("❌ ISSUE: All heuristics are performing poorly")
            return False
            
    except Exception as e:
        print(f"Error testing heuristics: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimized_heuristics()
    sys.exit(0 if success else 1)