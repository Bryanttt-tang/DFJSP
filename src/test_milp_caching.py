"""
Test script to verify MILP caching functionality
"""
import collections
import time

# Import from backup_no_wait.py
from backup_no_wait import (
    ENHANCED_JOBS_DATA, 
    MACHINE_LIST, 
    DETERMINISTIC_ARRIVAL_TIMES,
    milp_optimal_scheduler,
    validate_schedule_makespan
)

def test_milp_caching():
    """Test that MILP caching works correctly."""
    
    print("=" * 80)
    print("TESTING MILP CACHING FUNCTIONALITY")
    print("=" * 80)
    
    # Test scenario 1: First run should compute and cache
    print("\n📝 TEST 1: First run with scenario A (should compute)")
    print("-" * 80)
    arrival_times_A = {0: 0, 1: 0, 2: 0, 3: 8, 4: 12, 5: 16, 6: 20}
    
    start_time = time.time()
    makespan1, schedule1 = milp_optimal_scheduler(
        ENHANCED_JOBS_DATA, 
        MACHINE_LIST, 
        arrival_times_A, 
        time_limit=60,
        verbose=True
    )
    time1 = time.time() - start_time
    
    print(f"\n✅ Result: Makespan = {makespan1:.2f}, Time = {time1:.2f}s")
    
    # Test scenario 2: Second run with SAME inputs should use cache
    print("\n" + "=" * 80)
    print("📝 TEST 2: Second run with scenario A (should use cache)")
    print("-" * 80)
    
    start_time = time.time()
    makespan2, schedule2 = milp_optimal_scheduler(
        ENHANCED_JOBS_DATA, 
        MACHINE_LIST, 
        arrival_times_A,  # SAME as before
        time_limit=60,
        verbose=True
    )
    time2 = time.time() - start_time
    
    print(f"\n✅ Result: Makespan = {makespan2:.2f}, Time = {time2:.2f}s")
    
    # Verify results match
    assert makespan1 == makespan2, f"Makespans don't match! {makespan1} vs {makespan2}"
    assert time2 < time1 / 10, f"Cache not faster! time1={time1:.2f}s, time2={time2:.2f}s"
    
    print(f"\n🎉 Cache speedup: {time1/time2:.1f}x faster!")
    
    # Test scenario 3: Different inputs should compute fresh
    print("\n" + "=" * 80)
    print("📝 TEST 3: Different scenario B (should compute fresh)")
    print("-" * 80)
    arrival_times_B = {0: 0, 1: 2, 2: 4, 3: 8, 4: 12, 5: 16, 6: 20}  # DIFFERENT
    
    start_time = time.time()
    makespan3, schedule3 = milp_optimal_scheduler(
        ENHANCED_JOBS_DATA, 
        MACHINE_LIST, 
        arrival_times_B,  # DIFFERENT from A
        time_limit=60,
        verbose=True
    )
    time3 = time.time() - start_time
    
    print(f"\n✅ Result: Makespan = {makespan3:.2f}, Time = {time3:.2f}s")
    
    # Test scenario 4: Use scenario B cache
    print("\n" + "=" * 80)
    print("📝 TEST 4: Second run with scenario B (should use cache)")
    print("-" * 80)
    
    start_time = time.time()
    makespan4, schedule4 = milp_optimal_scheduler(
        ENHANCED_JOBS_DATA, 
        MACHINE_LIST, 
        arrival_times_B,  # SAME as scenario B
        time_limit=60,
        verbose=True
    )
    time4 = time.time() - start_time
    
    print(f"\n✅ Result: Makespan = {makespan4:.2f}, Time = {time4:.2f}s")
    
    assert makespan3 == makespan4, f"Makespans don't match! {makespan3} vs {makespan4}"
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 CACHING TEST SUMMARY")
    print("=" * 80)
    print(f"Scenario A - First run:  {time1:.2f}s (compute + cache)")
    print(f"Scenario A - Second run: {time2:.2f}s (cached) → {time1/time2:.1f}x speedup ✅")
    print(f"Scenario B - First run:  {time3:.2f}s (compute + cache)")
    print(f"Scenario B - Second run: {time4:.2f}s (cached) → {time3/time4:.1f}x speedup ✅")
    print(f"\nAll tests passed! 🎉")
    print("=" * 80)


if __name__ == "__main__":
    test_milp_caching()
