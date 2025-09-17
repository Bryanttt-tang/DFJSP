#!/usr/bin/env python3
"""
Simple verification script for the cleaned up dynamic vs static comparison
"""

def test_heuristics():
    """Test the enhanced heuristics in the main file"""
    try:
        print("Testing enhanced heuristics in clean_dynamic_vs_static_comparison.py...")
        
        from clean_dynamic_vs_static_comparison import (
            ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES,
            simple_spt_heuristic, basic_greedy_scheduler,
            optimized_spt_scheduler, earliest_completion_scheduler,
            spt_heuristic_poisson
        )
        
        print("✅ All imports successful!")
        print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
        
        # Test the enhanced heuristics
        print("\nTesting enhanced heuristics:")
        
        # Test simple SPT
        spt_makespan, _ = simple_spt_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
        print(f"  Simple SPT: {spt_makespan:.2f}")
        
        # Test the main comparison function
        best_makespan, _ = spt_heuristic_poisson(ENHANCED_JOBS_DATA, MACHINE_LIST, DETERMINISTIC_ARRIVAL_TIMES)
        print(f"  Best heuristic: {best_makespan:.2f}")
        
        if best_makespan < 50:
            print("✅ SUCCESS: Enhanced heuristics are working properly!")
            print("✅ Main file is ready for full dynamic vs static comparison!")
            return True
        else:
            print("⚠️  Heuristics working but performance could be better")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_comparison():
    """Run the full dynamic vs static comparison"""
    try:
        print("\n" + "="*60)
        print("RUNNING FULL DYNAMIC vs STATIC RL COMPARISON")
        print("="*60)
        
        from clean_dynamic_vs_static_comparison import main
        main()
        
        print("\n✅ Full comparison completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Full comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Verifying cleaned up dynamic vs static comparison...")
    print("="*60)
    
    # First test just the heuristics
    if test_heuristics():
        # If heuristics work, try the full comparison
        user_input = input("\nRun full comparison with RL training? (y/n): ").lower().strip()
        if user_input == 'y':
            run_full_comparison()
        else:
            print("Skipped full comparison. Heuristics verified successfully!")
    else:
        print("Heuristic verification failed. Please check the main file.")