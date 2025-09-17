"""
Comprehensive comparison between Dynamic RL and Static RL for FJSP

This script addresses potential issues with dynamic RL performance by:
1. Training both methods with identical architectures and hyperparameters
2. Testing both on the same dynamic test scenarios  
3. Analyzing different arrival rates to find optimal ranges
4. Using proper reward shaping for dynamic environments
5. Implementing curriculum learning for dynamic RL

Expected: Dynamic RL should outperform Static RL when job arrival rate is suitable
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from dynamic_poisson_fjsp import PoissonDynamicFJSPEnv, ENHANCED_JOBS_DATA, MACHINE_LIST
from dynamic_fjsp_env import FJSPEnv
import collections
import time

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def mask_fn(env):
    """Action masking function for invalid actions."""
    return env.action_masks()

class StaticFJSPWrapper(FJSPEnv):
    """Wrapper that treats all jobs as available from the start (static scheduling)"""
    
    def __init__(self, jobs_data, machines):
        # Convert to expected format and pass to parent
        super().__init__(
            jobs_data=jobs_data, 
            machines=machines,
            reward_mode="optimized"
        )
    
    def _generate_poisson_arrivals(self):
        """Override to prevent any dynamic arrivals."""
        self.arrival_times = {job_id: 0.0 for job_id in self.job_ids}
        self.next_arrival_events = []

def train_static_rl_agent(jobs_data, machines, timesteps=200000, verbose=1):
    """Train a static RL agent (all jobs available at t=0)."""
    print(f"Training Static RL Agent for {timesteps:,} timesteps...")
    
    def make_static_env():
        env = StaticFJSPWrapper(jobs_data, machines)
        return ActionMasker(env, mask_fn)
    
    vec_env = DummyVecEnv([make_static_env])
    
    model = MaskablePPO(
        "MlpPolicy", 
        vec_env, 
        verbose=verbose,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        ),
        seed=SEED
    )
    
    model.learn(total_timesteps=timesteps)
    return model

def train_dynamic_rl_agent_curriculum(jobs_data, machines, arrival_rate=0.5, 
                                    timesteps=200000, verbose=1):
    """Train dynamic RL agent with curriculum learning."""
    print(f"Training Dynamic RL Agent with curriculum learning...")
    print(f"Arrival rate: {arrival_rate}, Total timesteps: {timesteps:,}")
    
    def make_dynamic_env(phase_arrival_rate):
        initial_jobs = [0, 1, 2] if phase_arrival_rate > 0 else list(jobs_data.keys())
        env = PoissonDynamicFJSPEnv(
            jobs_data=jobs_data,
            machines=machines,
            initial_jobs=initial_jobs,
            arrival_rate=phase_arrival_rate,
            max_time_horizon=100,
            reward_mode="makespan_increment"
        )
        return ActionMasker(env, mask_fn)
    
    # Phase 1: Start with static environment (easier)
    print("  Phase 1/3: Static environment (warm-up)...")
    vec_env = DummyVecEnv([lambda: make_dynamic_env(0.0)])
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=verbose,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        ),
        seed=SEED
    )
    
    model.learn(total_timesteps=timesteps // 3)
    
    # Phase 2: Moderate arrival rate
    print("  Phase 2/3: Moderate arrival rate...")
    moderate_rate = arrival_rate * 0.5
    vec_env = DummyVecEnv([lambda: make_dynamic_env(moderate_rate)])
    model.set_env(vec_env)
    model.learn(total_timesteps=timesteps // 3)
    
    # Phase 3: Target arrival rate
    print("  Phase 3/3: Target arrival rate...")
    vec_env = DummyVecEnv([lambda: make_dynamic_env(arrival_rate)])
    model.set_env(vec_env)
    model.learn(total_timesteps=timesteps // 3)
    
    return model

def train_dynamic_rl_agent_direct(jobs_data, machines, arrival_rate=0.5, 
                                timesteps=200000, verbose=1):
    """Train dynamic RL agent directly on target arrival rate."""
    print(f"Training Dynamic RL Agent (direct) for {timesteps:,} timesteps...")
    print(f"Arrival rate: {arrival_rate}")
    
    def make_dynamic_env():
        env = PoissonDynamicFJSPEnv(
            jobs_data=jobs_data,
            machines=machines,
            initial_jobs=[0, 1, 2],  # Only first 3 jobs initially
            arrival_rate=arrival_rate,
            max_time_horizon=100,
            reward_mode="makespan_increment"
        )
        return ActionMasker(env, mask_fn)
    
    vec_env = DummyVecEnv([make_dynamic_env])
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=verbose,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        ),
        seed=SEED
    )
    
    model.learn(total_timesteps=timesteps)
    return model

def evaluate_agent_on_dynamic_scenarios(model, jobs_data, machines, arrival_rate, 
                                       num_scenarios=10, method_name="Agent"):
    """Evaluate agent on multiple dynamic scenarios."""
    print(f"Evaluating {method_name} on {num_scenarios} dynamic scenarios (λ={arrival_rate})...")
    
    makespans = []
    
    for scenario in range(num_scenarios):
        # Create test environment
        env = PoissonDynamicFJSPEnv(
            jobs_data=jobs_data,
            machines=machines,
            initial_jobs=[0, 1, 2],
            arrival_rate=arrival_rate,
            max_time_horizon=100,
            reward_mode="makespan_increment"
        )
        
        # Seed for this scenario
        env.reset(seed=SEED + scenario)
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            # Get action masks
            action_masks = env.action_masks()
            
            # Get action from model
            if hasattr(model, 'predict'):
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                # Fallback for other model types
                action = model.predict(obs)[0]
            
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
        
        makespan = env.current_time
        makespans.append(makespan)
    
    avg_makespan = np.mean(makespans)
    std_makespan = np.std(makespans)
    
    print(f"  {method_name} avg makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
    return avg_makespan, std_makespan, makespans

def spt_heuristic_dynamic(jobs_data, machines, arrival_rate, num_scenarios=10):
    """SPT heuristic baseline for dynamic scenarios."""
    print(f"Running SPT heuristic on {num_scenarios} dynamic scenarios (λ={arrival_rate})...")
    
    makespans = []
    
    for scenario in range(num_scenarios):
        np.random.seed(SEED + scenario)
        
        # Generate arrival times
        current_time = 0.0
        arrival_times = {0: 0.0, 1: 0.0, 2: 0.0}  # Initial jobs
        
        for job_id in [3, 4, 5, 6]:
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival_time
            if current_time <= 100:  # Within time horizon
                arrival_times[job_id] = current_time
            else:
                arrival_times[job_id] = float('inf')
        
        # SPT scheduling
        machine_available = {m: 0.0 for m in machines}
        schedule = {m: [] for m in machines}
        
        # Create operation list with arrival constraints
        operations = []
        for job_id, job_ops in jobs_data.items():
            for op_idx, op_data in enumerate(job_ops):
                operations.append({
                    'job_id': job_id,
                    'op_idx': op_idx,
                    'arrival_time': arrival_times.get(job_id, 0.0),
                    'proc_times': op_data['proc_times'],
                    'min_proc_time': min(op_data['proc_times'].values())
                })
        
        # Sort by arrival time, then by processing time (SPT within available jobs)
        operations.sort(key=lambda x: (x['arrival_time'], x['min_proc_time']))
        
        # Schedule operations
        completed_jobs = set()
        while len(completed_jobs) < len(jobs_data) and operations:
            # Find next available operation
            scheduled = False
            for i, op in enumerate(operations):
                job_id = op['job_id']
                op_idx = op['op_idx']
                arrival_time = op['arrival_time']
                
                # Check if job has arrived and all previous operations completed
                current_sim_time = min(machine_available.values())
                if (arrival_time <= current_sim_time and 
                    all(schedule[m] and max(end for _, _, end in schedule[m]) >= arrival_time 
                        for m in machines if schedule[m]) and
                    op_idx == 0):  # First operation of job
                    
                    # Find best machine (shortest processing time)
                    best_machine = min(op['proc_times'].keys(), 
                                     key=lambda m: op['proc_times'][m])
                    proc_time = op['proc_times'][best_machine]
                    
                    # Schedule operation
                    start_time = max(machine_available[best_machine], arrival_time)
                    end_time = start_time + proc_time
                    
                    schedule[best_machine].append((f"J{job_id}-O{op_idx}", start_time, end_time))
                    machine_available[best_machine] = end_time
                    
                    operations.pop(i)
                    if op_idx == len(jobs_data[job_id]) - 1:  # Last operation
                        completed_jobs.add(job_id)
                    
                    scheduled = True
                    break
            
            if not scheduled:
                break
        
        # Calculate makespan
        makespan = max(machine_available.values()) if machine_available else 0
        makespans.append(makespan)
    
    avg_makespan = np.mean(makespans)
    std_makespan = np.std(makespans)
    
    print(f"  SPT heuristic avg makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
    return avg_makespan, std_makespan, makespans

def run_arrival_rate_analysis():
    """Test different arrival rates to find optimal range for dynamic RL."""
    print("\n" + "="*80)
    print("ARRIVAL RATE ANALYSIS")
    print("="*80)
    
    arrival_rates = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    results = {
        'arrival_rates': arrival_rates,
        'spt_makespans': [],
        'dynamic_rl_makespans': [],
        'static_rl_makespans': []
    }
    
    # Train static RL once (it doesn't depend on arrival rate)
    static_model = train_static_rl_agent(ENHANCED_JOBS_DATA, MACHINE_LIST, timesteps=100000, verbose=0)
    
    for arrival_rate in arrival_rates:
        print(f"\n--- Testing arrival rate λ = {arrival_rate} ---")
        
        # SPT baseline
        spt_avg, _, _ = spt_heuristic_dynamic(ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_rate, num_scenarios=5)
        results['spt_makespans'].append(spt_avg)
        
        # Train dynamic RL for this arrival rate
        dynamic_model = train_dynamic_rl_agent_curriculum(
            ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_rate, timesteps=100000, verbose=0
        )
        
        # Evaluate both models
        dynamic_avg, _, _ = evaluate_agent_on_dynamic_scenarios(
            dynamic_model, ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_rate, num_scenarios=5, method_name="Dynamic RL"
        )
        results['dynamic_rl_makespans'].append(dynamic_avg)
        
        static_avg, _, _ = evaluate_agent_on_dynamic_scenarios(
            static_model, ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_rate, num_scenarios=5, method_name="Static RL"
        )
        results['static_rl_makespans'].append(static_avg)
        
        # Show improvement
        improvement = ((static_avg - dynamic_avg) / static_avg * 100) if static_avg > 0 else 0
        print(f"  Dynamic RL improvement over Static RL: {improvement:.1f}%")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(arrival_rates, results['spt_makespans'], 'o-', label='SPT Heuristic', linewidth=2)
    plt.plot(arrival_rates, results['dynamic_rl_makespans'], 's-', label='Dynamic RL', linewidth=2)
    plt.plot(arrival_rates, results['static_rl_makespans'], '^-', label='Static RL', linewidth=2)
    
    plt.xlabel('Job Arrival Rate (λ)')
    plt.ylabel('Average Makespan')
    plt.title('Performance vs Job Arrival Rate\n(Lower is better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('arrival_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def run_comprehensive_comparison(arrival_rate=0.5):
    """Run comprehensive comparison at optimal arrival rate."""
    print("\n" + "="*80)
    print(f"COMPREHENSIVE COMPARISON (λ = {arrival_rate})")
    print("="*80)
    
    num_test_scenarios = 20
    training_timesteps = 200000
    
    # 1. Train Static RL
    print("\n1. Training Static RL Agent...")
    static_model = train_static_rl_agent(ENHANCED_JOBS_DATA, MACHINE_LIST, training_timesteps)
    
    # 2. Train Dynamic RL with curriculum
    print("\n2. Training Dynamic RL Agent (Curriculum)...")
    dynamic_curriculum_model = train_dynamic_rl_agent_curriculum(
        ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_rate, training_timesteps
    )
    
    # 3. Train Dynamic RL direct
    print("\n3. Training Dynamic RL Agent (Direct)...")
    dynamic_direct_model = train_dynamic_rl_agent_direct(
        ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_rate, training_timesteps
    )
    
    # 4. Evaluate all methods
    print(f"\n4. Evaluating all methods on {num_test_scenarios} test scenarios...")
    
    # SPT Baseline
    spt_avg, spt_std, _ = spt_heuristic_dynamic(
        ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_rate, num_test_scenarios
    )
    
    # Static RL
    static_avg, static_std, _ = evaluate_agent_on_dynamic_scenarios(
        static_model, ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_rate, 
        num_test_scenarios, "Static RL"
    )
    
    # Dynamic RL (Curriculum)
    dynamic_curr_avg, dynamic_curr_std, _ = evaluate_agent_on_dynamic_scenarios(
        dynamic_curriculum_model, ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_rate, 
        num_test_scenarios, "Dynamic RL (Curriculum)"
    )
    
    # Dynamic RL (Direct)
    dynamic_direct_avg, dynamic_direct_std, _ = evaluate_agent_on_dynamic_scenarios(
        dynamic_direct_model, ENHANCED_JOBS_DATA, MACHINE_LIST, arrival_rate, 
        num_test_scenarios, "Dynamic RL (Direct)"
    )
    
    # 5. Results summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    results = [
        ("SPT Heuristic", spt_avg, spt_std),
        ("Static RL", static_avg, static_std),
        ("Dynamic RL (Curriculum)", dynamic_curr_avg, dynamic_curr_std),
        ("Dynamic RL (Direct)", dynamic_direct_avg, dynamic_direct_std)
    ]
    
    # Sort by performance
    results.sort(key=lambda x: x[1])
    
    print("Performance Ranking (Average ± Std Dev):")
    for i, (method, avg, std) in enumerate(results):
        if i == 0:
            print(f"{i+1}. {method}: {avg:.2f} ± {std:.2f} (Best)")
        else:
            gap = ((avg - results[0][1]) / results[0][1] * 100)
            print(f"{i+1}. {method}: {avg:.2f} ± {std:.2f} (+{gap:.1f}%)")
    
    # Specific comparisons
    print(f"\nKey Comparisons:")
    
    # Dynamic vs Static
    if dynamic_curr_avg < static_avg:
        improvement = ((static_avg - dynamic_curr_avg) / static_avg * 100)
        print(f"✓ Dynamic RL (Curriculum) beats Static RL by {improvement:.1f}%")
    else:
        gap = ((dynamic_curr_avg - static_avg) / static_avg * 100)
        print(f"✗ Dynamic RL (Curriculum) loses to Static RL by {gap:.1f}%")
    
    # Curriculum vs Direct training
    if dynamic_curr_avg < dynamic_direct_avg:
        improvement = ((dynamic_direct_avg - dynamic_curr_avg) / dynamic_direct_avg * 100)
        print(f"✓ Curriculum learning improves Dynamic RL by {improvement:.1f}%")
    else:
        gap = ((dynamic_curr_avg - dynamic_direct_avg) / dynamic_direct_avg * 100)
        print(f"✗ Curriculum learning hurts Dynamic RL by {gap:.1f}%")
    
    # Bar plot comparison
    plt.figure(figsize=(10, 6))
    methods = [r[0] for r in results]
    avgs = [r[1] for r in results]
    stds = [r[2] for r in results]
    
    bars = plt.bar(methods, avgs, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel('Average Makespan')
    plt.title(f'Method Comparison (λ = {arrival_rate})\nLower is Better')
    plt.xticks(rotation=45, ha='right')
    
    # Color bars by performance
    colors = ['green', 'orange', 'red', 'red']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(f'method_comparison_lambda_{arrival_rate}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    print("Dynamic vs Static RL Comprehensive Comparison")
    print("="*80)
    print(f"Jobs: {len(ENHANCED_JOBS_DATA)}, Machines: {len(MACHINE_LIST)}")
    print(f"Initial jobs: 0,1,2 | Dynamic jobs: 3,4,5,6")
    print("="*80)
    
    # Run arrival rate analysis first
    rate_results = run_arrival_rate_analysis()
    
    # Find optimal arrival rate (where dynamic RL has largest advantage)
    improvements = []
    for i, rate in enumerate(rate_results['arrival_rates']):
        static_makespan = rate_results['static_rl_makespans'][i]
        dynamic_makespan = rate_results['dynamic_rl_makespans'][i]
        improvement = ((static_makespan - dynamic_makespan) / static_makespan * 100) if static_makespan > 0 else 0
        improvements.append(improvement)
    
    best_rate_idx = np.argmax(improvements)
    optimal_arrival_rate = rate_results['arrival_rates'][best_rate_idx]
    max_improvement = improvements[best_rate_idx]
    
    print(f"\nOptimal arrival rate found: λ = {optimal_arrival_rate}")
    print(f"Maximum Dynamic RL improvement: {max_improvement:.1f}%")
    
    # Run detailed comparison at optimal rate
    final_results = run_comprehensive_comparison(optimal_arrival_rate)
    
    print(f"\nAnalysis complete! Check generated plots for visualization.")
