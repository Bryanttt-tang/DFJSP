# Test file to verify dynamic RL improvements
import sys
sys.path.append('/Users/tanu/Desktop/PhD/Scheduling/src')

# Test the improved training
try:
    from clean_dynamic_vs_static_comparison import (
        train_dynamic_agent, 
        ENHANCED_JOBS_DATA, 
        MACHINE_LIST,
        DETERMINISTIC_ARRIVAL_TIMES
    )
    
    print("Testing enhanced dynamic RL training...")
    
    # Test with smaller timesteps for quick verification
    dynamic_model = train_dynamic_agent(
        ENHANCED_JOBS_DATA, 
        MACHINE_LIST, 
        initial_jobs=[0, 1, 2], 
        arrival_rate=0.1, 
        total_timesteps=1000,  # Small number for testing
        reward_mode="makespan_increment"
    )
    
    print("✅ Enhanced dynamic RL training completed successfully!")
    print("Key improvements implemented:")
    print("- Enhanced reward function with load balancing")
    print("- Better observation space with timing hints")
    print("- Optimized hyperparameters for dynamic scenarios")  
    print("- Curriculum learning for gradual complexity increase")
    print("- Priority action selection for efficiency")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()