"""
Clean Dynamic vs Static RL Comparison for Poisson FJSP
=====================================================

This script compares:
1. Dynamic RL (trained on Poisson job arrivals) vs Static RL (trained on all jobs at t=0) 
2. Dynamic RL vs SPT Heuristic

All evaluations are done on the same Poisson job arrival test cases.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import gymnasium as gym
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- Job Data ---
ENHANCED_JOBS_DATA = collections.OrderedDict({
    0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}, {'proc_times': {'M0': 2}}],
    1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M1': 3, 'M2': 4}}],
    2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M0': 3, 'M1': 4}}, {'proc_times': {'M2': 7}}],
    3: [{'proc_times': {'M1': 8}}, {'proc_times': {'M2': 2}}, {'proc_times': {'M0': 5, 'M1': 6}}],
    4: [{'proc_times': {'M0': 6, 'M1': 9}}, {'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M2': 6}}],
    5: [{'proc_times': {'M1': 5, 'M2': 8}}, {'proc_times': {'M0': 6}}, {'proc_times': {'M1': 4, 'M2': 3}}],
    6: [{'proc_times': {'M0': 7, 'M2': 4}}, {'proc_times': {'M0': 5, 'M1': 6}}, {'proc_times': {'M1': 3}}, {'proc_times': {'M0': 2, 'M2': 5}}]
})

MACHINE_LIST = ['M0', 'M1', 'M2']

def mask_fn(env):
    """Mask function for ActionMasker wrapper"""
    return env.action_masks()

# --- Import Environment Classes ---
# Import the already working environment classes
exec(open('dynamic_poisson_fjsp.py').read())

def train_static_agent(jobs_data, machine_list, total_timesteps=150000):
    """Train a static RL agent where all jobs are available at t=0."""
    print(f"\n--- Training Static RL Agent ---")
    print(f"Timesteps: {total_timesteps}")
    print("Training scenario: All jobs available at time 0")
    
    def make_static_env():
        env = StaticFJSPEnv(jobs_data, machine_list, reward_mode="makespan_increment")
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_static_env])
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        )
    )
    
    # Show a training episode every 50,000 timesteps
    for phase in range(3):
        phase_timesteps = total_timesteps // 3
        print(f"\nTraining Phase {phase + 1}/3 ({phase_timesteps} timesteps)")
        
        if phase == 0:
            print("Training episodes sample:")
            # Test current model performance
            test_env = StaticFJSPEnv(jobs_data, machine_list, reward_mode="makespan_increment")
            obs, _ = test_env.reset()
            total_reward = 0
            step_count = 0
            
            while step_count < 100:  # Quick test
                action_masks = test_env.action_masks()
                if not any(action_masks):
                    break
                    
                # Random action for initial demonstration
                valid_actions = [i for i, valid in enumerate(action_masks) if valid]
                if valid_actions:
                    action = np.random.choice(valid_actions)
                    obs, reward, done, truncated, info = test_env.step(action)
                    total_reward += reward
                    step_count += 1
                    
                    if step_count <= 5:  # Show first 5 steps
                        print(f"  Step {step_count}: Action {action}, Reward {reward:.2f}, Time {test_env.current_time:.2f}")
                    
                    if done or truncated:
                        break
            
            print(f"  Sample episode: {step_count} steps, Total reward: {total_reward:.2f}, Final makespan: {test_env.current_time:.2f}")
        
        model.learn(total_timesteps=phase_timesteps)
    
    print("Static RL agent training completed!")
    return model

def train_dynamic_agent(jobs_data, machine_list, initial_jobs=3, arrival_rate=0.08, total_timesteps=150000):
    """Train a dynamic RL agent on Poisson job arrivals."""
    print(f"\n--- Training Dynamic RL Agent ---")
    print(f"Initial jobs: {initial_jobs}, Arrival rate: {arrival_rate}, Timesteps: {total_timesteps}")
    
    def make_dynamic_env():
        env = PoissonDynamicFJSPEnv(
            jobs_data, machine_list, 
            initial_jobs=initial_jobs,
            arrival_rate=arrival_rate,
            reward_mode="dynamic_adaptation"
        )
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_dynamic_env])
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.1,
        ent_coef=0.005,
        policy_kwargs=dict(
            net_arch=[512, 512, 256, 128],
            activation_fn=torch.nn.ReLU
        )
    )
    
    model.learn(total_timesteps=total_timesteps)
    print("Dynamic RL agent training completed!")
    return model

def generate_test_scenarios(jobs_data, initial_jobs=[0, 1, 2], arrival_rate=0.08, num_scenarios=5):
    """Generate test scenarios with fixed Poisson arrival times."""
    print(f"\n--- Generating {num_scenarios} Test Scenarios ---")
    print(f"Initial jobs: {initial_jobs}, Arrival rate: {arrival_rate}")
    
    scenarios = []
    for i in range(num_scenarios):
        np.random.seed(100 + i)  # Fixed seeds for reproducibility
        arrival_times = {}
        
        # Initial jobs arrive at t=0
        for job_id in initial_jobs:
            arrival_times[job_id] = 0.0
        
        # Generate Poisson arrivals for remaining jobs
        remaining_jobs = [j for j in jobs_data.keys() if j not in initial_jobs]
        current_time = 0.0
        
        for job_id in remaining_jobs:
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival_time
            if current_time <= 200:  # Max simulation time
                arrival_times[job_id] = current_time
            else:
                arrival_times[job_id] = float('inf')  # Won't arrive
        
        scenarios.append(arrival_times)
        print(f"Scenario {i+1}: {arrival_times}")
    
    return scenarios

def evaluate_static_on_dynamic(static_model, jobs_data, machine_list, arrival_times):
    """Evaluate static model on dynamic scenario with observation space mapping."""
    
    # Create static environment to get expected observation size
    static_env = StaticFJSPEnv(jobs_data, machine_list)
    static_obs_size = static_env.observation_space.shape[0]
    
    # Create dynamic environment for testing
    test_env = PoissonDynamicFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=[k for k, v in arrival_times.items() if v == 0],
        arrival_rate=0.1,
        reward_mode="makespan_increment"
    )
    
    # Override with fixed arrival times
    test_env.arrival_times = arrival_times.copy()
    test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
    test_env.next_arrival_events.sort(key=lambda x: x[0])
    
    obs, _ = test_env.reset()
    
    # Re-override after reset
    test_env.arrival_times = arrival_times.copy()
    test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
    test_env.next_arrival_events.sort(key=lambda x: x[0])
    test_env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    
    def map_dynamic_to_static_obs(dynamic_obs):
        """Map dynamic observation to static observation space."""
        static_obs = []
        
        # Machine availability (first 3 elements)
        static_obs.extend(dynamic_obs[:len(machine_list)])
        
        # Operation completion status
        num_jobs = len(jobs_data)
        max_ops = max(len(ops) for ops in jobs_data.values())
        start_idx = len(machine_list)
        end_idx = start_idx + num_jobs * max_ops
        static_obs.extend(dynamic_obs[start_idx:end_idx])
        
        # Job progress ratios
        start_idx = end_idx
        end_idx = start_idx + num_jobs
        static_obs.extend(dynamic_obs[start_idx:end_idx])
        
        # Machine workloads (skip job arrival info in dynamic obs)
        machine_workloads_start = len(machine_list) + num_jobs * max_ops + num_jobs + num_jobs
        static_obs.extend(dynamic_obs[machine_workloads_start:machine_workloads_start + len(machine_list)])
        
        # Current time
        time_idx = machine_workloads_start + len(machine_list)
        static_obs.append(dynamic_obs[time_idx])
        
        # Number of completed jobs
        completed_jobs_idx = time_idx + 2  # Skip arrived jobs count
        static_obs.append(dynamic_obs[completed_jobs_idx])
        
        # Pad or truncate to match static observation space
        if len(static_obs) < static_obs_size:
            static_obs.extend([0.0] * (static_obs_size - len(static_obs)))
        elif len(static_obs) > static_obs_size:
            static_obs = static_obs[:static_obs_size]
        
        return np.array(static_obs, dtype=np.float32)
    
    step_count = 0
    invalid_actions = 0
    
    while step_count < 1000:
        action_masks = test_env.action_masks()
        
        if not any(action_masks):
            break
        
        # Map observation and predict action
        static_obs = map_dynamic_to_static_obs(obs)
        action, _ = static_model.predict(static_obs, action_masks=action_masks, deterministic=True)
        
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if info.get("error") == "Invalid action":
            invalid_actions += 1
        
        if done or truncated:
            break
    
    makespan = test_env.current_time
    return makespan, test_env.schedule, invalid_actions

def evaluate_dynamic_on_dynamic(dynamic_model, jobs_data, machine_list, arrival_times):
    """Evaluate dynamic model on dynamic scenario."""
    
    test_env = PoissonDynamicFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=[k for k, v in arrival_times.items() if v == 0],
        arrival_rate=0.1,
        reward_mode="dynamic_adaptation"
    )
    
    # Override with fixed arrival times
    test_env.arrival_times = arrival_times.copy()
    test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
    test_env.next_arrival_events.sort(key=lambda x: x[0])
    
    obs, _ = test_env.reset()
    
    # Re-override after reset
    test_env.arrival_times = arrival_times.copy()
    test_env.next_arrival_events = [(time, job_id) for job_id, time in arrival_times.items() if time > 0]
    test_env.next_arrival_events.sort(key=lambda x: x[0])
    test_env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    
    step_count = 0
    invalid_actions = 0
    
    while step_count < 1000:
        action_masks = test_env.action_masks()
        
        if not any(action_masks):
            break
        
        action, _ = dynamic_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if info.get("error") == "Invalid action":
            invalid_actions += 1
        
        if done or truncated:
            break
    
    makespan = test_env.current_time
    return makespan, test_env.schedule, invalid_actions

def spt_heuristic_poisson(jobs_data, machine_list, arrival_times):
    """SPT (Shortest Processing Time) heuristic for dynamic scheduling."""
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    schedule = {m: [] for m in machine_list}
    
    arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0}
    operations_scheduled = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    sim_time = 0.0
    
    while operations_scheduled < total_operations:
        # Update arrivals
        new_arrivals = []
        for job_id, arr_time in arrival_times.items():
            if job_id not in arrived_jobs and arr_time <= sim_time:
                arrived_jobs.add(job_id)
                new_arrivals.append(job_id)
        
        # Collect available operations
        available_ops = []
        for job_id in arrived_jobs:
            next_op = next_operation_for_job[job_id]
            if next_op < len(jobs_data[job_id]):
                # Check if job is ready (previous operation completed)
                if next_op == 0 or operation_end_times[job_id][next_op - 1] <= sim_time:
                    op_data = jobs_data[job_id][next_op]
                    for machine, proc_time in op_data['proc_times'].items():
                        available_ops.append((job_id, next_op, machine, proc_time))
        
        if not available_ops:
            # No operations available, advance time to next event
            next_events = []
            
            # Next machine available time
            for m, next_free in machine_next_free.items():
                if next_free > sim_time:
                    next_events.append(next_free)
            
            # Next job arrival
            for job_id, arr_time in arrival_times.items():
                if job_id not in arrived_jobs and arr_time > sim_time:
                    next_events.append(arr_time)
            
            # Next operation ready time
            for job_id in arrived_jobs:
                next_op = next_operation_for_job[job_id]
                if next_op > 0 and next_op < len(jobs_data[job_id]):
                    ready_time = operation_end_times[job_id][next_op - 1]
                    if ready_time > sim_time:
                        next_events.append(ready_time)
            
            if next_events:
                sim_time = min(next_events)
            else:
                break
            continue
        
        # Apply SPT: select operation with shortest processing time
        available_ops.sort(key=lambda x: x[3])  # Sort by processing time
        job_id, op_idx, machine, proc_time = available_ops[0]
        
        # Calculate timing
        machine_available_time = machine_next_free[machine]
        job_ready_time = operation_end_times[job_id][op_idx - 1] if op_idx > 0 else arrival_times[job_id]
        
        start_time = max(machine_available_time, job_ready_time, sim_time)
        end_time = start_time + proc_time
        
        # Update state
        machine_next_free[machine] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled += 1
        sim_time = max(sim_time, end_time)
        
        # Record in schedule
        schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    return makespan, schedule

def plot_gantt(schedule, machines, title="Schedule", save_path=None):
    """Plot Gantt chart for the schedule."""
    if not schedule or all(len(ops) == 0 for ops in schedule.values()):
        print("No schedule to plot - schedule is empty")
        return

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(14, len(machines) * 0.8 + 1))

    for idx, m in enumerate(machines):
        machine_ops = schedule.get(m, [])
        machine_ops.sort(key=lambda x: x[1])  # Sort by start time

        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op, start_time, end_time = op_data[:3]
                duration = end_time - start_time
                
                # Extract job number for coloring
                job_num = 0
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                    except:
                        job_num = 0
                
                color = colors[job_num % len(colors)]
                
                ax.barh(idx, duration, left=start_time, height=0.6, 
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add operation label
                if duration > 1:  # Only add text if bar is wide enough
                    ax.text(start_time + duration/2, idx, job_op, 
                           ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gantt chart saved to {save_path}")
    
    plt.show()

def main():
    """Main comparison function."""
    print("=" * 80)
    print("DYNAMIC vs STATIC RL COMPARISON FOR POISSON FJSP")
    print("=" * 80)
    print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
    print("Research Question: Does Dynamic RL outperform Static RL on Poisson arrivals?")
    print("=" * 80)
    
    # Step 1: Train both agents
    print("\n1. TRAINING PHASE")
    print("-" * 40)
    
    # Train static RL agent (all jobs available at t=0)
    static_model = train_static_agent(ENHANCED_JOBS_DATA, MACHINE_LIST, total_timesteps=150000)
    
    # Train dynamic RL agent (Poisson job arrivals)
    dynamic_model = train_dynamic_agent(ENHANCED_JOBS_DATA, MACHINE_LIST, 
                                       initial_jobs=3, arrival_rate=0.08, total_timesteps=150000)
    
    # Step 2: Generate test scenarios
    print("\n2. TEST SCENARIO GENERATION")
    print("-" * 40)
    test_scenarios = generate_test_scenarios(ENHANCED_JOBS_DATA, 
                                           initial_jobs=[0, 1, 2], 
                                           arrival_rate=0.08, 
                                           num_scenarios=5)
    
    # Step 3: Evaluate all methods on the same test scenarios
    print("\n3. EVALUATION PHASE")
    print("-" * 40)
    
    static_results = []
    dynamic_results = []
    spt_results = []
    
    for i, arrival_times in enumerate(test_scenarios):
        print(f"\nEvaluating Scenario {i+1}...")
        
        # Static RL on dynamic scenario
        static_makespan, static_schedule, static_invalid = evaluate_static_on_dynamic(
            static_model, ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_times)
        static_results.append(static_makespan)
        print(f"  Static RL: {static_makespan:.2f} (invalid: {static_invalid})")
        
        # Dynamic RL on dynamic scenario
        dynamic_makespan, dynamic_schedule, dynamic_invalid = evaluate_dynamic_on_dynamic(
            dynamic_model, ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_times)
        dynamic_results.append(dynamic_makespan)
        print(f"  Dynamic RL: {dynamic_makespan:.2f} (invalid: {dynamic_invalid})")
        
        # SPT Heuristic on dynamic scenario
        spt_makespan, spt_schedule = spt_heuristic_poisson(
            ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_times)
        spt_results.append(spt_makespan)
        print(f"  SPT Heuristic: {spt_makespan:.2f}")
    
    # Step 4: Results Analysis
    print("\n4. RESULTS ANALYSIS")
    print("=" * 50)
    
    # Calculate statistics
    static_avg = np.mean(static_results)
    static_std = np.std(static_results)
    static_min = np.min(static_results)
    
    dynamic_avg = np.mean(dynamic_results)
    dynamic_std = np.std(dynamic_results)
    dynamic_min = np.min(dynamic_results)
    
    spt_avg = np.mean(spt_results)
    spt_std = np.std(spt_results)
    spt_min = np.min(spt_results)
    
    print(f"Static RL    - Avg: {static_avg:.2f} ± {static_std:.2f}, Best: {static_min:.2f}")
    print(f"Dynamic RL   - Avg: {dynamic_avg:.2f} ± {dynamic_std:.2f}, Best: {dynamic_min:.2f}")
    print(f"SPT Heuristic- Avg: {spt_avg:.2f} ± {spt_std:.2f}, Best: {spt_min:.2f}")
    
    # Performance comparisons
    print("\n5. PERFORMANCE COMPARISON")
    print("-" * 40)
    
    # Dynamic RL vs Static RL
    if dynamic_avg < static_avg:
        improvement = ((static_avg - dynamic_avg) / static_avg) * 100
        print(f"✓ Dynamic RL outperforms Static RL by {improvement:.1f}%")
    else:
        gap = ((dynamic_avg - static_avg) / static_avg) * 100
        print(f"✗ Dynamic RL underperforms Static RL by {gap:.1f}%")
    
    # Dynamic RL vs SPT
    if dynamic_avg < spt_avg:
        improvement = ((spt_avg - dynamic_avg) / spt_avg) * 100
        print(f"✓ Dynamic RL outperforms SPT Heuristic by {improvement:.1f}%")
    else:
        gap = ((dynamic_avg - spt_avg) / spt_avg) * 100
        print(f"✗ Dynamic RL underperforms SPT Heuristic by {gap:.1f}%")
    
    # Plot comparison results
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(test_scenarios))
    width = 0.25
    
    plt.bar(x - width, static_results, width, label='Static RL', alpha=0.8, color='red')
    plt.bar(x, dynamic_results, width, label='Dynamic RL', alpha=0.8, color='blue')
    plt.bar(x + width, spt_results, width, label='SPT Heuristic', alpha=0.8, color='green')
    
    plt.xlabel('Test Scenario')
    plt.ylabel('Makespan')
    plt.title('Comparison: Dynamic RL vs Static RL vs SPT Heuristic\n(Lower is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dynamic_vs_static_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate Gantt charts for best scenario
    best_scenario_idx = np.argmin(dynamic_results)
    best_arrival_times = test_scenarios[best_scenario_idx]
    
    print(f"\n6. GANTT CHARTS FOR BEST SCENARIO (Scenario {best_scenario_idx + 1})")
    print("-" * 60)
    
    # Re-evaluate best scenario to get schedules
    static_makespan, static_schedule, _ = evaluate_static_on_dynamic(
        static_model, ENHANCED_JOBS_DATA, MACHINE_LIST, best_arrival_times)
    
    dynamic_makespan, dynamic_schedule, _ = evaluate_dynamic_on_dynamic(
        dynamic_model, ENHANCED_JOBS_DATA, MACHINE_LIST, best_arrival_times)
    
    spt_makespan, spt_schedule = spt_heuristic_poisson(
        ENHANCED_JOBS_DATA, MACHINE_LIST, best_arrival_times)
    
    # Plot Gantt charts
    plot_gantt(static_schedule, MACHINE_LIST, 
              f"Static RL - Makespan: {static_makespan:.2f}", 
              "static_rl_gantt.png")
    
    plot_gantt(dynamic_schedule, MACHINE_LIST, 
              f"Dynamic RL - Makespan: {dynamic_makespan:.2f}", 
              "dynamic_rl_gantt.png")
    
    plot_gantt(spt_schedule, MACHINE_LIST, 
              f"SPT Heuristic - Makespan: {spt_makespan:.2f}", 
              "spt_heuristic_gantt.png")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED!")
    print("Generated files:")
    print("- dynamic_vs_static_comparison.png: Performance comparison chart")
    print("- static_rl_gantt.png: Static RL Gantt chart")
    print("- dynamic_rl_gantt.png: Dynamic RL Gantt chart") 
    print("- spt_heuristic_gantt.png: SPT Heuristic Gantt chart")
    print("=" * 80)

if __name__ == "__main__":
    main()
