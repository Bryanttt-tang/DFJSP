"""
Evaluation Script for Trained RL Models on Dynamic FJSP
=========================================================

This script loads pre-trained RL models and performs comprehensive evaluation:
- Loads saved models: reactive_rl, proactive_rl, rule_based_rl, static_rl, perfect_knowledge_rl
- Evaluates on test scenarios
- Compares with MILP optimal and heuristics
- Generates Gantt charts and performance analysis
- No retraining required - pure evaluation mode

Usage:
    python eval.py
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import gymnasium as gym
import torch
import time
import json
import os
from sb3_contrib import MaskablePPO

# Import from main file
from proactive_sche import (
    # Environment classes
    PoissonDynamicFJSPEnv,
    ProactiveDynamicFJSPEnv,
    DispatchingRuleFJSPEnv,
    PerfectKnowledgeFJSPEnv,
    # Dataset and configuration
    ENHANCED_JOBS_DATA,
    MACHINE_LIST,
    INITIAL_JOB_IDS,
    GLOBAL_SEED,
    # Evaluation functions
    evaluate_static_on_dynamic,
    evaluate_static_on_static,
    evaluate_dynamic_on_dynamic,
    evaluate_rule_based_on_dynamic,
    evaluate_proactive_on_dynamic,
    evaluate_perfect_knowledge_on_scenario,
    # Utility functions
    generate_test_scenarios,
    verify_schedule_correctness,
    milp_optimal_scheduler,
    ortools_cp_sat_scheduler,
    best_heuristic,
    calculate_regret_analysis,
)

# Set seed for reproducibility
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


def load_trained_models():
    """
    Load all pre-trained RL models from disk.
    
    Returns:
        dict: Dictionary containing all loaded models
    """
    print("\n" + "="*80)
    print("LOADING PRE-TRAINED RL MODELS")
    print("="*80)
    
    models = {}
    model_paths = {
        'reactive_rl': 'reactive_rl_model.zip',
        'proactive_rl': 'proactive_rl_model.zip',
        'rule_based_rl': 'rule_based_rl_model.zip',
        'static_rl': 'static_rl_model.zip',
        'perfect_knowledge_rl': 'perfect_knowledge_rl_model.zip'
    }
    
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            try:
                models[model_name] = MaskablePPO.load(model_path)
                print(f"✅ Loaded {model_name}: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                models[model_name] = None
        else:
            print(f"⚠️  Model not found: {model_path}")
            models[model_name] = None
    
    # Check which models are available
    available_models = [name for name, model in models.items() if model is not None]
    missing_models = [name for name, model in models.items() if model is None]
    
    print(f"\n✅ Available models: {', '.join(available_models)}")
    if missing_models:
        print(f"⚠️  Missing models: {', '.join(missing_models)}")
        print(f"   Run training first to generate these models.")
    
    return models


def evaluate_all_methods(models, test_scenarios, arrival_rate):
    """
    Evaluate all methods on test scenarios.
    
    Args:
        models: Dictionary of loaded RL models
        test_scenarios: List of test scenario dictionaries
        arrival_rate: Arrival rate used for testing
        
    Returns:
        dict: Results for all methods across all scenarios
    """
    print("\n" + "="*80)
    print("EVALUATION PHASE - MULTIPLE SCENARIOS")
    print("="*80)
    print(f"Evaluating on {len(test_scenarios)} test scenarios...")
    
    # Initialize results storage
    all_results = {
        'Perfect Knowledge RL': [],
        'Proactive RL': [],
        'Reactive RL': [],
        'Rule-Based RL': [],
        'Static RL (dynamic)': [],
        'Static RL (static)': [],
        'Best Heuristic': [],
        'MILP Optimal': []
    }
    
    # Storage for Gantt chart data
    gantt_scenarios_data = []
    
    for i, scenario in enumerate(test_scenarios):
        print("\n" + "-"*60)
        print(f"SCENARIO {i+1}/{len(test_scenarios)}")
        scenario_arrivals = scenario['arrival_times']
        print(f"Arrival times: {scenario_arrivals}")
        
        # STEP 1: Compute Optimal Solution (MILP or OR-Tools)
        print(f"  Computing Optimal Solution for scenario {i+1}...")
        
        # Try OR-Tools CP-SAT first
        ortools_available = True
        try:
            print(f"    Attempting OR-Tools CP-SAT solver...")
            ortools_start_time = time.time()
            optimal_makespan, optimal_schedule = ortools_cp_sat_scheduler(
                ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals, time_limit=300
            )
            ortools_end_time = time.time()
            ortools_solve_time = ortools_end_time - ortools_start_time
            
            if optimal_makespan != float('inf'):
                print(f"    ✅ OR-Tools CP-SAT: {optimal_makespan:.2f} (solved in {ortools_solve_time:.1f}s)")
                milp_makespan = optimal_makespan
                milp_schedule = optimal_schedule
            else:
                ortools_available = False
        except:
            ortools_available = False
        
        if not ortools_available:
            print(f"    OR-Tools not available, using MILP solver...")
            milp_start_time = time.time()
            milp_makespan, milp_schedule = milp_optimal_scheduler(
                ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals, solver='gurobi'
            )
            milp_end_time = time.time()
        
        milp_is_valid = milp_makespan != float('inf')
        all_results['MILP Optimal'].append(milp_makespan if milp_is_valid else None)
        
        # STEP 2: Evaluate Perfect Knowledge RL (if available)
        if models.get('perfect_knowledge_rl'):
            print(f"  Evaluating Perfect Knowledge RL...")
            perfect_makespan, perfect_schedule = evaluate_perfect_knowledge_on_scenario(
                models['perfect_knowledge_rl'], ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals
            )
            all_results['Perfect Knowledge RL'].append(perfect_makespan)
            print(f"    Perfect RL: {perfect_makespan:.2f}")
        else:
            perfect_makespan = float('inf')
            perfect_schedule = None
            all_results['Perfect Knowledge RL'].append(None)
        
        # STEP 3: Evaluate Reactive RL
        if models.get('reactive_rl'):
            print(f"  Evaluating Reactive RL...")
            dynamic_makespan, dynamic_schedule = evaluate_dynamic_on_dynamic(
                models['reactive_rl'], ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals
            )
            all_results['Reactive RL'].append(dynamic_makespan)
            print(f"    Reactive RL: {dynamic_makespan:.2f}")
        else:
            dynamic_makespan = float('inf')
            dynamic_schedule = None
            all_results['Reactive RL'].append(None)
        
        # STEP 4: Evaluate Proactive RL
        if models.get('proactive_rl'):
            print(f"  Evaluating Proactive RL...")
            proactive_makespan, proactive_schedule = evaluate_proactive_on_dynamic(
                models['proactive_rl'], ENHANCED_JOBS_DATA, MACHINE_LIST, 
                scenario_arrivals, predictor_mode='map'
            )
            all_results['Proactive RL'].append(proactive_makespan)
            print(f"    Proactive RL: {proactive_makespan:.2f}")
        else:
            proactive_makespan = float('inf')
            proactive_schedule = None
            all_results['Proactive RL'].append(None)
        
        # STEP 5: Evaluate Rule-Based RL
        if models.get('rule_based_rl'):
            print(f"  Evaluating Rule-Based RL...")
            rule_based_makespan, rule_based_schedule = evaluate_rule_based_on_dynamic(
                models['rule_based_rl'], ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals
            )
            all_results['Rule-Based RL'].append(rule_based_makespan)
            print(f"    Rule-Based RL: {rule_based_makespan:.2f}")
        else:
            rule_based_makespan = float('inf')
            rule_based_schedule = None
            all_results['Rule-Based RL'].append(None)
        
        # STEP 6: Evaluate Static RL (on dynamic scenario)
        if models.get('static_rl'):
            print(f"  Evaluating Static RL...")
            static_dynamic_makespan, static_dynamic_schedule = evaluate_static_on_dynamic(
                models['static_rl'], ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals
            )
            all_results['Static RL (dynamic)'].append(static_dynamic_makespan)
            print(f"    Static RL (dynamic): {static_dynamic_makespan:.2f}")
            
            # Evaluate on static scenario (only once)
            if i == 0:
                static_static_makespan, static_static_schedule = evaluate_static_on_static(
                    models['static_rl'], ENHANCED_JOBS_DATA, MACHINE_LIST
                )
            all_results['Static RL (static)'].append(static_static_makespan)
        else:
            static_dynamic_makespan = float('inf')
            static_dynamic_schedule = None
            static_static_makespan = float('inf')
            static_static_schedule = None
            all_results['Static RL (dynamic)'].append(None)
            all_results['Static RL (static)'].append(None)
        
        # STEP 7: Evaluate Best Heuristic
        print(f"  Evaluating Best Heuristic...")
        spt_makespan, spt_schedule = best_heuristic(
            ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals
        )
        all_results['Best Heuristic'].append(spt_makespan)
        print(f"    Best Heuristic: {spt_makespan:.2f}")
        
        # Store schedules for Gantt plotting
        schedules_dict = {
            'Perfect Knowledge RL': (perfect_makespan, perfect_schedule),
            'Proactive RL': (proactive_makespan, proactive_schedule),
            'Reactive RL': (dynamic_makespan, dynamic_schedule),
            'Rule-Based RL': (rule_based_makespan, rule_based_schedule),
            'Static RL (dynamic)': (static_dynamic_makespan, static_dynamic_schedule),
            'Static RL (static)': (static_static_makespan, static_static_schedule),
            'Best Heuristic': (spt_makespan, spt_schedule)
        }
        
        if milp_is_valid:
            schedules_dict['MILP Optimal'] = (milp_makespan, milp_schedule)
        
        gantt_scenarios_data.append({
            'scenario_id': i,
            'arrival_times': scenario_arrivals,
            'schedules': schedules_dict
        })
        
        # Print summary
        print(f"\n  Results Summary:")
        if milp_is_valid:
            print(f"    MILP: {milp_makespan:.2f}, Perfect: {perfect_makespan:.2f}, " +
                  f"Proactive: {proactive_makespan:.2f}, Reactive: {dynamic_makespan:.2f}, " +
                  f"Rule-Based: {rule_based_makespan:.2f}, Static: {static_dynamic_makespan:.2f}, " +
                  f"Heuristic: {spt_makespan:.2f}")
        else:
            print(f"    Perfect: {perfect_makespan:.2f}, Proactive: {proactive_makespan:.2f}, " +
                  f"Reactive: {dynamic_makespan:.2f}, Rule-Based: {rule_based_makespan:.2f}, " +
                  f"Static: {static_dynamic_makespan:.2f}, Heuristic: {spt_makespan:.2f}")
    
    return all_results, gantt_scenarios_data


def print_results_analysis(all_results):
    """Print comprehensive results analysis."""
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    # Calculate average results
    avg_results = {}
    std_results = {}
    for method, results in all_results.items():
        valid_results = [r for r in results if r is not None and r != float('inf')]
        if valid_results:
            avg_results[method] = np.mean(valid_results)
            std_results[method] = np.std(valid_results)
        else:
            avg_results[method] = float('inf')
            std_results[method] = 0
    
    print("\nAVERAGE RESULTS ACROSS TEST SCENARIOS:")
    for method in ['MILP Optimal', 'Perfect Knowledge RL', 'Proactive RL', 'Reactive RL', 
                   'Rule-Based RL', 'Static RL (dynamic)', 'Static RL (static)', 'Best Heuristic']:
        if method in avg_results:
            if avg_results[method] != float('inf'):
                valid_count = len([r for r in all_results[method] if r is not None and r != float('inf')])
                print(f"{method:25s}: {avg_results[method]:.2f} ± {std_results[method]:.2f} ({valid_count} valid)")
            else:
                print(f"{method:25s}: No valid results")
    
    # Performance ranking
    print("\nPERFORMANCE RANKING:")
    avg_results_list = [(method, makespan) for method, makespan in avg_results.items()]
    avg_results_list.sort(key=lambda x: x[1])
    for i, (method, makespan) in enumerate(avg_results_list, 1):
        if makespan != float('inf'):
            print(f"{i}. {method}: {makespan:.2f}")
    
    return avg_results, std_results


def generate_gantt_charts(gantt_scenarios_data, arrival_rate, folder_name=None):
    """
    Generate Gantt charts for all test scenarios.
    
    Args:
        gantt_scenarios_data: List of scenario dictionaries with schedules
        arrival_rate: Arrival rate for folder naming
        folder_name: Optional custom folder name
    """
    print("\n" + "="*80)
    print("GENERATING GANTT CHARTS")
    print("="*80)
    
    # Create folder for Gantt charts
    if folder_name is None:
        folder_name = f"eval_results_{arrival_rate}_rate"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    
    colors = plt.cm.tab20.colors
    
    for scenario_idx in range(len(gantt_scenarios_data)):
        scenario = gantt_scenarios_data[scenario_idx]
        scenario_id = scenario['scenario_id']
        arrival_times = scenario['arrival_times']
        schedules = scenario['schedules']
        
        print(f"\nGenerating Gantt chart for Scenario {scenario_id + 1}...")
        
        # Determine number of methods available (ensure makespan is numeric and valid)
        available_methods = []
        for name, data in schedules.items():
            if data and len(data) >= 2:
                makespan, schedule = data[0], data[1]
                # Convert makespan to float if it's not already numeric
                if makespan is not None and makespan != float('inf'):
                    try:
                        makespan = float(makespan) if not isinstance(makespan, (int, float)) else makespan
                        if schedule is not None:
                            available_methods.append((name, (makespan, schedule)))
                    except (ValueError, TypeError):
                        continue
        
        num_methods = len(available_methods)
        if num_methods == 0:
            print(f"  ⚠️  No valid schedules for scenario {scenario_id + 1}, skipping...")
            continue
        
        fig, axes = plt.subplots(num_methods, 1, figsize=(16, num_methods * 3))
        if num_methods == 1:
            axes = [axes]
        
        fig.suptitle(f'Test Scenario {scenario_id + 1} - {num_methods} Method Comparison\n' + 
                     f'Arrival Times: {arrival_times}', 
                     fontsize=14, fontweight='bold')
        
        # Calculate consistent x-axis limits (ensure makespans are numeric)
        max_makespan_scenario = max([float(makespan) if not isinstance(makespan, (int, float)) else makespan 
                                     for makespan, _ in available_methods])
        x_limit_scenario = max_makespan_scenario * 1.15 if max_makespan_scenario > 0 else 100
        
        for plot_idx, (method_name, (makespan, schedule)) in enumerate(available_methods):
            ax = axes[plot_idx]
            
            # Plot operations for each machine
            for idx, machine in enumerate(MACHINE_LIST):
                machine_ops = schedule.get(machine, [])
                machine_ops.sort(key=lambda x: x[1])
                
                for op_data in machine_ops:
                    if len(op_data) >= 3:
                        job_op, start_time, end_time = op_data[:3]
                        duration = end_time - start_time
                        
                        job_num = 0
                        if 'J' in job_op:
                            try:
                                job_num = int(job_op.split('J')[1].split('-')[0])
                            except:
                                job_num = 0
                        
                        color = colors[job_num % len(colors)]
                        
                        ax.barh(idx, duration, left=start_time, height=0.6, 
                               color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                        
                        if duration > 1:
                            ax.text(start_time + duration/2, idx, job_op, 
                                   ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Add arrival arrows
            arrow_y_position = len(MACHINE_LIST) + 0.2
            for job_id, arrival_time in arrival_times.items():
                if arrival_time > 0 and arrival_time < x_limit_scenario:
                    ax.axvline(x=arrival_time, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                    ax.annotate(f'J{job_id}', 
                               xy=(arrival_time, arrow_y_position), 
                               xytext=(arrival_time, arrow_y_position + 0.3),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                               ha='center', va='bottom', color='red', fontweight='bold', fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='red', alpha=0.8))
            
            # Formatting
            ax.set_yticks(range(len(MACHINE_LIST)))
            ax.set_yticklabels(MACHINE_LIST)
            ax.set_xlabel("Time" if plot_idx == len(available_methods)-1 else "")
            ax.set_ylabel("Machines")
            ax.set_title(f"{method_name} (Makespan: {makespan:.2f})", fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, x_limit_scenario)
            ax.set_ylim(-0.5, len(MACHINE_LIST) + 1.5)
        
        # Add legend
        legend_elements = []
        for i in range(len(ENHANCED_JOBS_DATA)):
            color = colors[i % len(colors)]
            initial_or_dynamic = ' (Initial)' if i < len(INITIAL_JOB_IDS) else ' (Dynamic)'
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                            alpha=0.8, label=f'Job {i}{initial_or_dynamic}'))
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
                ncol=len(ENHANCED_JOBS_DATA), fontsize=9)
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        
        scenario_filename = os.path.join(folder_name, f'scenario_{scenario_id + 1}_gantt.png')
        plt.savefig(scenario_filename, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {scenario_filename}")
        plt.close()
    
    print(f"\n✅ All Gantt charts saved in {folder_name}/")


def generate_focused_comparison_chart(gantt_scenarios_data, arrival_rate):
    """
    Generate focused 5-method comparison chart for first scenario.
    
    Methods: MILP, Perfect RL, Proactive RL, Rule-Based RL, Best Heuristic
    """
    print("\n" + "="*80)
    print("GENERATING FOCUSED 5-METHOD COMPARISON CHART")
    print("="*80)
    
    if not gantt_scenarios_data:
        print("⚠️  No scenario data available")
        return
    
    first_scenario = gantt_scenarios_data[0]
    schedules = first_scenario['schedules']
    arrival_times = first_scenario['arrival_times']
    
    # Select 5 key methods
    key_methods = ['MILP Optimal', 'Perfect Knowledge RL', 'Proactive RL', 
                   'Rule-Based RL', 'Best Heuristic']
    
    available_methods = [(name, schedules[name]) for name in key_methods 
                        if name in schedules and schedules[name][0] != float('inf')]
    
    if len(available_methods) < 3:
        print(f"⚠️  Not enough methods available ({len(available_methods)} < 3), skipping...")
        return
    
    num_methods = len(available_methods)
    fig, axes = plt.subplots(num_methods, 1, figsize=(18, num_methods * 3.5))
    if num_methods == 1:
        axes = [axes]
    
    fig.suptitle('Focused 5-Method Comparison: First Test Scenario\n' + 
                 f'Comparing: MILP Optimal, Perfect RL, Proactive RL, Rule-Based RL, Best Heuristic', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab20.colors
    
    # Calculate consistent x-axis (ensure makespans are numeric)
    max_makespan = max([float(makespan) if not isinstance(makespan, (int, float)) else makespan 
                       for makespan, _ in available_methods])
    x_limit = max_makespan * 1.1
    
    for plot_idx, (method_name, (makespan, schedule)) in enumerate(available_methods):
        ax = axes[plot_idx]
        
        for idx, machine in enumerate(MACHINE_LIST):
            machine_ops = schedule.get(machine, [])
            machine_ops.sort(key=lambda x: x[1])
            
            for op_data in machine_ops:
                if len(op_data) >= 3:
                    job_op, start_time, end_time = op_data[:3]
                    duration = end_time - start_time
                    
                    job_num = 0
                    if 'J' in job_op:
                        try:
                            job_num = int(job_op.split('J')[1].split('-')[0])
                        except:
                            job_num = 0
                    
                    color = colors[job_num % len(colors)]
                    ax.barh(idx, duration, left=start_time, height=0.6, 
                           color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                    
                    if duration > 1:
                        ax.text(start_time + duration/2, idx, job_op, 
                               ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Add arrival arrows
        arrow_y_position = len(MACHINE_LIST) + 0.3
        for job_id, arrival_time in arrival_times.items():
            if arrival_time > 0 and arrival_time < x_limit:
                ax.axvline(x=arrival_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax.annotate(f'Job {job_id} arrives', 
                           xy=(arrival_time, arrow_y_position), 
                           xytext=(arrival_time, arrow_y_position + 0.5),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           ha='center', va='bottom', color='red', fontweight='bold', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))
        
        ax.set_yticks(range(len(MACHINE_LIST)))
        ax.set_yticklabels(MACHINE_LIST)
        ax.set_xlabel("Time" if plot_idx == len(available_methods)-1 else "")
        ax.set_ylabel("Machines")
        ax.set_title(f"{method_name} (Makespan: {makespan:.2f})", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x_limit)
        ax.set_ylim(-0.5, len(MACHINE_LIST) + 2.0)
    
    # Add legend
    legend_elements = []
    for i in range(len(ENHANCED_JOBS_DATA)):
        color = colors[i % len(colors)]
        initial_or_poisson = ' (Initial)' if i < len(INITIAL_JOB_IDS) else ' (Poisson)'
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                          alpha=0.8, label=f'Job {i}{initial_or_poisson}'))
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(ENHANCED_JOBS_DATA), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    filename = 'focused_5method_gantt_comparison_eval.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


def main():
    """Main evaluation workflow."""
    print("\n" + "="*80)
    print("RL MODEL EVALUATION SCRIPT")
    print("="*80)
    print(f"Dataset: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
    print(f"Seed: {GLOBAL_SEED}")
    print("="*80)
    
    # Configuration
    arrival_rate = 0.1
    num_scenarios = 1  # Adjust as needed
    
    # Step 1: Load trained models
    models = load_trained_models()
    
    if not any(models.values()):
        print("\n❌ No trained models found!")
        print("Please run training first: python proactive_sche.py")
        return
    
    # Step 2: Generate test scenarios
    print("\n" + "="*80)
    print("GENERATING TEST SCENARIOS")
    print("="*80)
    test_scenarios = generate_test_scenarios(
        ENHANCED_JOBS_DATA,
        initial_jobs=INITIAL_JOB_IDS,
        arrival_rate=arrival_rate,
        num_scenarios=num_scenarios
    )
    
    # Step 3: Evaluate all methods
    all_results, gantt_scenarios_data = evaluate_all_methods(
        models, test_scenarios, arrival_rate
    )
    
    # Step 4: Print results analysis
    avg_results, std_results = print_results_analysis(all_results)
    
    # Step 5: Calculate regret analysis
    print("\n" + "="*80)
    print("REGRET ANALYSIS")
    print("="*80)
    
    valid_milp = [m for m in all_results['MILP Optimal'] if m is not None and m != float('inf')]
    if valid_milp:
        benchmark = np.mean(valid_milp)
        benchmark_name = "MILP Optimal"
    else:
        benchmark = avg_results.get('Perfect Knowledge RL', float('inf'))
        benchmark_name = "Perfect RL"
    
    methods_for_regret = {
        k: v for k, v in avg_results.items()
        if k not in ['MILP Optimal'] and v != float('inf')
    }
    
    if benchmark != float('inf'):
        calculate_regret_analysis(benchmark, methods_for_regret, benchmark_name)
    
    # Step 6: Generate Gantt charts
    generate_gantt_charts(gantt_scenarios_data, arrival_rate)
    
    # Step 7: Generate focused comparison chart
    generate_focused_comparison_chart(gantt_scenarios_data, arrival_rate)
    
    # Step 8: Save results to JSON
    results_filename = f'evaluation_results_rate_{arrival_rate}.json'
    with open(results_filename, 'w') as f:
        json.dump({
            'average_results': {k: float(v) if v != float('inf') else None 
                              for k, v in avg_results.items()},
            'std_results': {k: float(v) for k, v in std_results.items()},
            'all_results': {k: [float(r) if r != float('inf') else None for r in v] 
                          for k, v in all_results.items()},
            'arrival_rate': arrival_rate,
            'num_scenarios': num_scenarios
        }, f, indent=2)
    print(f"\n💾 Evaluation results saved to: {results_filename}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {results_filename}: Numerical results in JSON format")
    print(f"  - eval_results_{arrival_rate}_rate/: Folder with all Gantt charts")
    print(f"  - focused_5method_gantt_comparison_eval.png: Key methods comparison")


if __name__ == "__main__":
    main()
