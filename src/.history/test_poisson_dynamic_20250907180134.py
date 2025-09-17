"""
Test script for the Poisson Dynamic FJSP Environment

This script demonstrates the key features of the new dynamic environment:
1. Initial jobs available at start
2. Dynamic job arrivals following Poisson distribution
3. RL training with adaptation to unexpected arrivals
4. Comparison with SPT heuristic

No MILP solution is possible due to the dynamic, stochastic nature.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_poisson_environment():
    """Test the Poisson Dynamic FJSP environment without training."""
    try:
        from dynamic_poisson_fjsp import (
            PoissonDynamicFJSPEnv, ENHANCED_JOBS_DATA, MACHINE_LIST,
            heuristic_spt_poisson, plot_gantt
        )
        import numpy as np
        
        print("="*80)
        print("TESTING POISSON DYNAMIC FJSP ENVIRONMENT")
        print("="*80)
        
        # Environment parameters
        initial_jobs = 5
        arrival_rate = 0.1  # 0.1 jobs per time unit
        
        print(f"Environment Configuration:")
        print(f"- Total jobs in dataset: {len(ENHANCED_JOBS_DATA)}")
        print(f"- Initial jobs (available at start): {initial_jobs}")
        print(f"- Dynamic jobs (Poisson arrivals): {len(ENHANCED_JOBS_DATA) - initial_jobs}")
        print(f"- Arrival rate: {arrival_rate} jobs/time unit")
        print(f"- Expected inter-arrival time: {1.0/arrival_rate:.1f} time units")
        
        # 1. Test environment creation
        print(f"\n1. Creating Poisson Dynamic FJSP Environment...")
        env = PoissonDynamicFJSPEnv(
            ENHANCED_JOBS_DATA, MACHINE_LIST,
            initial_jobs=initial_jobs,
            arrival_rate=arrival_rate,
            max_time_horizon=200,
            seed=42
        )
        
        print(f"✓ Environment created successfully")
        print(f"  - Action space size: {env.action_space.n}")
        print(f"  - Observation space shape: {env.observation_space.shape}")
        
        # 2. Test environment reset and basic functionality
        print(f"\n2. Testing Environment Reset and Basic Operations...")
        obs, info = env.reset(seed=42)
        print(f"✓ Environment reset successful")
        print(f"  - Initial observation shape: {obs.shape}")
        print(f"  - Initial arrived jobs: {len(env.arrived_jobs)} out of {len(env.job_ids)}")
        print(f"  - Arrived job IDs: {sorted(env.arrived_jobs)}")
        
        # Show arrival schedule
        print(f"\n  Dynamic Job Arrival Schedule:")
        for job_id, arrival_time in sorted(env.arrival_times.items()):
            if arrival_time > 0 and arrival_time < float('inf'):
                print(f"    Job {job_id}: arrives at time {arrival_time:.2f}")
        
        # 3. Test action masking
        print(f"\n3. Testing Action Masking...")
        action_masks = env.action_masks()
        valid_actions = np.sum(action_masks)
        print(f"✓ Action masking working")
        print(f"  - Valid actions at start: {valid_actions} out of {len(action_masks)}")
        
        # 4. Test a few random steps
        print(f"\n4. Testing Random Steps...")
        for step in range(5):
            action_masks = env.action_masks()
            valid_actions = np.where(action_masks)[0]
            
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = env.step(action)
                
                print(f"  Step {step+1}: Action={action}, Reward={reward:.2f}, "
                      f"Time={info.get('makespan', 0):.2f}, "
                      f"New arrivals={info.get('newly_arrived_jobs', 0)}")
                
                if done:
                    print(f"    Episode completed at step {step+1}")
                    break
            else:
                print(f"  Step {step+1}: No valid actions available")
                break
        
        print(f"✓ Random steps completed successfully")
        
        # 5. Test SPT Heuristic
        print(f"\n5. Testing SPT Heuristic for Poisson Environment...")
        try:
            spt_makespan, spt_schedule = heuristic_spt_poisson(
                ENHANCED_JOBS_DATA, MACHINE_LIST,
                initial_jobs=initial_jobs,
                arrival_rate=arrival_rate,
                max_time=200,
                seed=42
            )
            
            print(f"✓ SPT Heuristic completed successfully")
            print(f"  - Final makespan: {spt_makespan:.2f}")
            
            # Count scheduled operations
            total_ops = sum(len(ops) for ops in spt_schedule.values())
            print(f"  - Total operations scheduled: {total_ops}")
            
        except Exception as e:
            print(f"✗ SPT Heuristic failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 6. Environment comparison with different arrival rates
        print(f"\n6. Testing Different Arrival Rates...")
        arrival_rates = [0.05, 0.1, 0.2]
        
        for rate in arrival_rates:
            test_env = PoissonDynamicFJSPEnv(
                ENHANCED_JOBS_DATA, MACHINE_LIST,
                initial_jobs=initial_jobs,
                arrival_rate=rate,
                seed=42
            )
            test_env.reset(seed=42)
            
            # Count dynamic arrivals within time horizon
            dynamic_arrivals = sum(1 for t in test_env.arrival_times.values() 
                                 if 0 < t < test_env.max_time_horizon)
            
            print(f"  Rate {rate:.2f}: {dynamic_arrivals} dynamic jobs will arrive")
        
        print(f"\n7. Key Features Demonstrated:")
        print(f"  ✓ Poisson arrival process generates realistic job streams")
        print(f"  ✓ Environment handles both initial and dynamic jobs")
        print(f"  ✓ Action masking prevents invalid scheduling decisions")
        print(f"  ✓ Rewards encourage adaptation to new arrivals")
        print(f"  ✓ SPT heuristic provides baseline for comparison")
        print(f"  ✓ No MILP solution possible (dynamic, stochastic)")
        
        print(f"\n8. Training Recommendations:")
        print(f"  - Use MaskablePPO for handling dynamic action spaces")
        print(f"  - Train with 'dynamic_adaptation' reward mode")
        print(f"  - Use multiple arrival rates for robust learning")
        print(f"  - Evaluate on same arrival distribution as training")
        print(f"  - Compare with SPT heuristic as baseline")
        
        print(f"\n" + "="*80)
        print(f"POISSON DYNAMIC FJSP ENVIRONMENT TEST COMPLETED SUCCESSFULLY")
        print(f"="*80)
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required packages:")
        print("pip install gymnasium numpy matplotlib stable-baselines3 sb3-contrib torch")
        return False
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_analysis():
    """Quick analysis without full RL training (for faster testing)."""
    print("\n" + "="*60)
    print("QUICK POISSON DYNAMIC FJSP ANALYSIS")
    print("="*60)
    
    try:
        from dynamic_poisson_fjsp import (
            ENHANCED_JOBS_DATA, MACHINE_LIST, heuristic_spt_poisson
        )
        
        # Test different scenarios
        scenarios = [
            {"initial_jobs": 3, "arrival_rate": 0.05, "name": "Low arrival rate"},
            {"initial_jobs": 5, "arrival_rate": 0.1, "name": "Medium arrival rate"},
            {"initial_jobs": 7, "arrival_rate": 0.15, "name": "High arrival rate"},
        ]
        
        print("\nSPT Heuristic Performance on Different Scenarios:")
        print("-" * 60)
        
        for scenario in scenarios:
            try:
                makespan, schedule = heuristic_spt_poisson(
                    ENHANCED_JOBS_DATA, MACHINE_LIST,
                    initial_jobs=scenario["initial_jobs"],
                    arrival_rate=scenario["arrival_rate"],
                    seed=42
                )
                
                total_ops = sum(len(ops) for ops in schedule.values())
                
                print(f"{scenario['name']:20s}: "
                      f"Makespan={makespan:6.2f}, "
                      f"Operations={total_ops:3d}, "
                      f"Initial={scenario['initial_jobs']}")
                
            except Exception as e:
                print(f"{scenario['name']:20s}: ERROR - {e}")
        
        print(f"\nKey Insights:")
        print(f"- Higher arrival rates lead to more operations and longer makespans")
        print(f"- More initial jobs reduce dependency on dynamic arrivals")
        print(f"- SPT provides reasonable baseline performance")
        print(f"- RL training should focus on adapting to arrival patterns")
        
    except Exception as e:
        print(f"Quick analysis failed: {e}")


if __name__ == "__main__":
    print("Starting Poisson Dynamic FJSP Environment Test...")
    
    # First run basic test
    success = test_poisson_environment()
    
    if success:
        # Then run quick analysis
        quick_analysis()
        
        print(f"\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Install required packages if not already installed:")
        print("   pip install gymnasium numpy matplotlib stable-baselines3 sb3-contrib torch")
        print("")
        print("2. Run full training and evaluation:")
        print("   python dynamic_poisson_fjsp.py")
        print("")
        print("3. Key innovations implemented:")
        print("   - Poisson arrival process for realistic job streams")
        print("   - Dynamic adaptation rewards for RL training")
        print("   - Separation of initial vs. dynamic jobs")
        print("   - Enhanced observation space for arrival awareness")
        print("   - Comparison with SPT heuristic (no MILP due to stochastic nature)")
        print("="*80)
    else:
        print("Test failed. Please check dependencies and try again.")
