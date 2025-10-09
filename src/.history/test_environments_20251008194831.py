#!/usr/bin/env python3
"""
Quick test script to verify the fixed environments work correctly.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import clean_dynamic_vs_static_comparison as cdsc
    import numpy as np
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_perfect_knowledge_env():
    """Test PerfectKnowledgeFJSPEnv"""
    print("\n🔬 Testing PerfectKnowledgeFJSPEnv...")
    
    jobs_data = cdsc.ENHANCED_JOBS_DATA
    machine_list = cdsc.MACHINE_LIST
    arrival_times = cdsc.DETERMINISTIC_ARRIVAL_TIMES
    
    try:
        env = cdsc.PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times)
        obs, _ = env.reset()
        print(f"   ✓ Environment created and reset")
        print(f"   ✓ Observation shape: {obs.shape}")
        print(f"   ✓ Action space: {env.action_space}")
        
        # Test few steps
        total_reward = 0
        steps_taken = 0
        for i in range(10):
            action_mask = env.action_masks()
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = valid_actions[0]
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps_taken += 1
                if done:
                    print(f"   ✓ Episode completed in {steps_taken} steps")
                    print(f"   ✓ Final makespan: {info.get('makespan', 0):.2f}")
                    break
            else:
                print(f"   ⚠️ No valid actions at step {i+1}")
                break
        
        if not done:
            print(f"   ⚠️ Episode didn't complete in 10 steps")
        
        print(f"   ✓ Total reward: {total_reward:.2f}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in PerfectKnowledgeFJSPEnv: {e}")
        return False

def test_poisson_dynamic_env():
    """Test PoissonDynamicFJSPEnv"""
    print("\n🔬 Testing PoissonDynamicFJSPEnv...")
    
    jobs_data = cdsc.ENHANCED_JOBS_DATA
    machine_list = cdsc.MACHINE_LIST
    
    try:
        env = cdsc.PoissonDynamicFJSPEnv(jobs_data, machine_list, initial_jobs=[0,1,2], arrival_rate=0.1)
        obs, _ = env.reset()
        print(f"   ✓ Environment created and reset")
        print(f"   ✓ Observation shape: {obs.shape}")
        print(f"   ✓ Action space: {env.action_space}")
        print(f"   ✓ Max episode steps: {env.max_episode_steps}")
        
        # Test few steps
        total_reward = 0
        steps_taken = 0
        wait_actions = 0
        schedule_actions = 0
        
        for i in range(20):  # Test more steps for dynamic env
            action_mask = env.action_masks()
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                # Prefer scheduling actions over WAIT when both available
                scheduling_actions = valid_actions[valid_actions < env.WAIT_ACTION]
                if len(scheduling_actions) > 0:
                    action = scheduling_actions[0]
                else:
                    action = valid_actions[0]  # WAIT action
                
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps_taken += 1
                
                action_type = info.get('action_type', 'UNKNOWN')
                if action_type == 'WAIT':
                    wait_actions += 1
                elif action_type == 'SCHEDULE':
                    schedule_actions += 1
                
                if done:
                    print(f"   ✓ Episode completed in {steps_taken} steps")
                    print(f"   ✓ Final makespan: {info.get('makespan', 0):.2f}")
                    print(f"   ✓ Operations scheduled: {env.operations_scheduled}/{env.total_operations}")
                    break
                    
                if reward <= -100:  # Check for major penalties
                    print(f"   ⚠️ Large penalty at step {steps_taken}: {reward:.2f}")
                    break
                    
            else:
                print(f"   ⚠️ No valid actions at step {i+1}")
                break
        
        if not done:
            print(f"   ⚠️ Episode didn't complete in 20 steps")
            print(f"   ⚠️ Current operations: {env.operations_scheduled}/{env.total_operations}")
        
        print(f"   ✓ Total reward: {total_reward:.2f}")
        print(f"   ✓ Actions taken - Schedule: {schedule_actions}, Wait: {wait_actions}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in PoissonDynamicFJSPEnv: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Fixed Environments")
    print("="*50)
    
    success1 = test_perfect_knowledge_env()
    success2 = test_poisson_dynamic_env()
    
    print("\n📊 Test Results:")
    print(f"   PerfectKnowledgeFJSPEnv: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   PoissonDynamicFJSPEnv:   {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Environments are working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        sys.exit(1)