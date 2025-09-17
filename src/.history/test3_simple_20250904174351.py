import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import os
import collections
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import math
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- Extended 7-Job Instance Data ---
EXTENDED_JOBS_DATA = collections.OrderedDict({
    0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}, {'proc_times': {'M0': 2}}],
    1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M1': 3, 'M2': 4}}],
    2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M0': 3, 'M1': 4}}, {'proc_times': {'M2': 7}}],
    3: [{'proc_times': {'M1': 8}}, {'proc_times': {'M2': 2}}, {'proc_times': {'M0': 5, 'M1': 6}}],
    4: [{'proc_times': {'M0': 6, 'M1': 9}}, {'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M2': 6}}],
    5: [{'proc_times': {'M1': 5, 'M2': 8}}, {'proc_times': {'M0': 6}}, {'proc_times': {'M1': 4, 'M2': 3}}],
    6: [{'proc_times': {'M0': 7, 'M2': 4}}, {'proc_times': {'M0': 5, 'M1': 6}}, {'proc_times': {'M1': 3}}, {'proc_times': {'M0': 2, 'M2': 5}}]
})
EXTENDED_MACHINE_LIST = ['M0', 'M1', 'M2']
EXTENDED_ARRIVAL_TIMES = {0: 0, 1: 0, 2: 0, 3: 10, 4: 15, 5: 25, 6: 35}

def plot_job_structure_table(jobs_data, machine_list, arrival_times, save_path=None):
    """Create a table figure showing the job data structure."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 12,
        'axes.titlesize': 20,
        'figure.titlesize': 22
    })
    
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('white')
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Job', 'Arrival Time', 'Operation', 'Available Machines', 'Processing Times']
    
    job_colors = ['#FFFFFF', '#F5F5F5', '#E8E8E8', '#DDDDDD', '#D0D0D0', '#C8C8C8']
    
    for job_id, operations in jobs_data.items():
        for op_idx, op_data in enumerate(operations):
            machines = list(op_data['proc_times'].keys())
            times = [str(op_data['proc_times'][m]) for m in machines]
            
            table_data.append([
                f"J{job_id}" if op_idx == 0 else "",
                f"{arrival_times[job_id]}" if op_idx == 0 else "",
                f"O{op_idx+1}",
                ", ".join(machines),
                ", ".join(f"{m}:{t}" for m, t in zip(machines, times))
            ])
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2.5)
    
    header_color = '#E3F2FD'
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(weight='bold', color='black')
        table[(0, i)].set_height(0.15)
    
    current_job = None
    color_idx = 0
    for i, row in enumerate(table_data):
        if row[0]:  # New job
            current_job = row[0]
            color_idx = (color_idx + 1) % len(job_colors)
        
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(job_colors[color_idx])
            table[(i + 1, j)].set_height(0.12)
    
    ax.set_title('Dynamic FJSP Instance - Job Structure and Arrival Times', 
                pad=30, fontsize=22, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Job structure table saved to {save_path}")
    
    plt.show()

def heuristic_spt_scheduler(jobs_data, machine_list, job_arrival_times):
    """SPT Heuristic Scheduler"""
    print("\n--- Running SPT Heuristic Scheduler ---")
    
    machine_next_free = {m: 0.0 for m in machine_list}
    operation_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation_for_job = {job_id: 0 for job_id in jobs_data}
    
    schedule = {m: [] for m in machine_list}
    operations_scheduled_count = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    
    arrived_jobs = {job_id for job_id, arrival in job_arrival_times.items() if arrival <= 0}
    
    while operations_scheduled_count < total_operations:
        candidate_operations = []
        
        if not any(next_operation_for_job[job_id] < len(jobs_data[job_id]) for job_id in arrived_jobs):
            upcoming_arrivals = [arr for arr in job_arrival_times.values() if arr > min(machine_next_free.values())]
            if not upcoming_arrivals: 
                break
            
            next_arrival_time = min(upcoming_arrivals)
            for m in machine_list:
                if machine_next_free[m] < next_arrival_time:
                    machine_next_free[m] = next_arrival_time
            
            arrived_jobs.update({job_id for job_id, arrival in job_arrival_times.items() if arrival <= next_arrival_time})

        for job_id in arrived_jobs:
            op_idx = next_operation_for_job[job_id]
            if op_idx < len(jobs_data[job_id]):
                op_data = jobs_data[job_id][op_idx]
                job_ready_time = operation_end_times[job_id][op_idx - 1] if op_idx > 0 else job_arrival_times[job_id]
                
                for machine_name, proc_time in op_data['proc_times'].items():
                    earliest_start_time = max(machine_next_free[machine_name], job_ready_time)
                    candidate_operations.append((
                        proc_time, earliest_start_time, job_id, op_idx, machine_name
                    ))
        
        if not candidate_operations:
            break

        selected_op = min(candidate_operations, key=lambda x: x[0])
        proc_time, start_time, job_id, op_idx, machine_name = selected_op
        
        end_time = start_time + proc_time

        machine_next_free[machine_name] = end_time
        operation_end_times[job_id][op_idx] = end_time
        next_operation_for_job[job_id] += 1
        operations_scheduled_count += 1
        
        schedule[machine_name].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        
        current_time = min(t for t in machine_next_free.values() if t > 0)
        arrived_jobs.update({j_id for j_id, arrival in job_arrival_times.items() if arrival <= current_time})

    makespan = max(machine_next_free.values()) if machine_next_free else 0
    print(f"SPT Heuristic Makespan: {makespan:.2f}")
    return makespan, schedule

def milp_scheduler(jobs, machines, arrival_times):
    """MILP approach for optimal dynamic scheduling."""
    print("\n--- Running MILP Optimal Scheduler ---")
    prob = LpProblem("DynamicFJSP_Optimal", LpMinimize)
    
    ops = [(j, oi) for j in jobs for oi in range(len(jobs[j]))]
    BIG_M = 1000 

    x = LpVariable.dicts("x", (ops, machines), cat="Binary")
    s = LpVariable.dicts("s", ops, lowBound=0)
    c = LpVariable.dicts("c", ops, lowBound=0)
    y = LpVariable.dicts("y", (ops, ops, machines), cat="Binary")
    Cmax = LpVariable("Cmax", lowBound=0)

    prob += Cmax

    for j, oi in ops:
        prob += lpSum(x[j, oi][m] for m in jobs[j][oi]['proc_times']) == 1
        prob += c[j, oi] == s[j, oi] + lpSum(x[j, oi][m] * jobs[j][oi]['proc_times'][m] for m in jobs[j][oi]['proc_times'])
        if oi > 0:
            prob += s[j, oi] >= c[j, oi - 1]
        else:
            prob += s[j, oi] >= arrival_times[j]
        prob += Cmax >= c[j, oi]

    for m in machines:
        ops_on_m = [op for op in ops if m in jobs[op[0]][op[1]]['proc_times']]
        for i in range(len(ops_on_m)):
            for k in range(i + 1, len(ops_on_m)):
                op1, op2 = ops_on_m[i], ops_on_m[k]
                prob += s[op1] >= c[op2] - BIG_M * (1 - y[op1][op2][m]) - BIG_M * (2 - x[op1][m] - x[op2][m])
                prob += s[op2] >= c[op1] - BIG_M * y[op1][op2][m] - BIG_M * (2 - x[op1][m] - x[op2][m])

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=120))

    schedule = {m: [] for m in machines}
    if prob.status == 1 and Cmax.varValue is not None:
        for (j, oi), m in ((op, m) for op in ops for m in jobs[op[0]][op[1]]['proc_times']):
            if x[j, oi][m].varValue > 0.5:
                schedule[m].append((f"J{j}-O{oi+1}", s[j, oi].varValue, c[j, oi].varValue))
        
        for m in machines:
            schedule[m].sort(key=lambda x: x[1])
        
        print(f"MILP (optimal) Makespan: {Cmax.varValue:.2f}")
        return Cmax.varValue, schedule
    else:
        print("MILP solver failed to find optimal solution")
        return float('inf'), schedule

def plot_gantt_charts(figure_num, schedules, makespans, titles, machine_list, arrival_times, save_path=None):
    """Plot multiple Gantt charts with arrival indicators."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 16
    })
    
    num_charts = len(schedules)
    fig = plt.figure(figure_num, figsize=(18, num_charts * 3.5))
    
    colors = plt.cm.Set3.colors
    
    max_time = 0
    for schedule in schedules:
        for machine_ops in schedule.values():
            for op_data in machine_ops:
                if len(op_data) >= 3:
                    max_time = max(max_time, op_data[2])
    
    max_time = max_time * 1.05

    for i, (schedule, makespan, title) in enumerate(zip(schedules, makespans, titles)):
        ax = fig.add_subplot(num_charts, 1, i + 1)
        for idx, m in enumerate(machine_list):
            for op_data in schedule.get(m, []):
                if len(op_data) == 3:
                    job_id_str, start, end = op_data
                    try:
                        j = int(job_id_str.split('-')[0][1:])
                    except (ValueError, IndexError):
                        j = hash(job_id_str) % len(colors)
                    
                    ax.broken_barh(
                        [(start, end - start)],
                        (idx * 10, 8),
                        facecolors=colors[j % len(colors)],
                        edgecolor='black',
                        alpha=0.8
                    )
                    label = job_id_str
                    ax.text(start + (end - start) / 2, idx * 10 + 4,
                           label, color='white', fontsize=10,
                           ha='center', va='center', weight='bold')

        current_arrival_times = arrival_times[i] if isinstance(arrival_times, list) else arrival_times
        for job_id, arrival in current_arrival_times.items():
            if arrival > 0:
                ax.axvline(x=arrival, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax.text(arrival, len(machine_list) * 10 - 5, f'J{job_id}↓', 
                       ha='center', va='bottom', color='red', fontweight='bold')

        ax.set_yticks([k * 10 + 4 for k in range(len(machine_list))])
        ax.set_yticklabels(machine_list)
        ax.set_ylabel("Machines", fontsize=12, fontfamily='serif')
        ax.set_title(f"{title}\nMakespan: {makespan:.2f}", fontsize=14, pad=20, 
                    fontfamily='serif', weight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, max_time)
        
        if i < num_charts - 1:
            ax.set_xticklabels([])
    
    plt.xlabel("Time", fontsize=14, fontfamily='serif', weight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.98], h_pad=2.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()

def simple_visualize_action_mask(jobs_data, machine_list, save_path=None):
    """Simple action mask visualization without gymnasium environment"""
    print("\n--- Creating Action Mask Visualization ---")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    # Create a simplified action space visualization
    action_data = []
    action_labels = []
    
    action_idx = 0
    for job_id, operations in jobs_data.items():
        for op_idx, op_data in enumerate(operations):
            for machine in op_data['proc_times'].keys():
                action_data.append((job_id, op_idx, machine, True))  # All actions valid at start
                action_labels.append(f"J{job_id}-O{op_idx+1}-{machine}")
                action_idx += 1
    
    # Create matrix visualization
    jobs = list(jobs_data.keys())
    max_ops = max(len(ops) for ops in jobs_data.values())
    
    matrix = np.zeros((len(jobs) * max_ops, len(machine_list)))
    row_labels = []
    
    for job_id in jobs:
        for op_idx in range(max_ops):
            row_idx = job_id * max_ops + op_idx
            if op_idx < len(jobs_data[job_id]):
                row_labels.append(f"J{job_id}-O{op_idx+1}")
                for m_idx, machine in enumerate(machine_list):
                    if machine in jobs_data[job_id][op_idx]['proc_times']:
                        matrix[row_idx, m_idx] = 1
            else:
                row_labels.append("")
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', alpha=0.8)
    
    ax.set_xticks(range(len(machine_list)))
    ax.set_xticklabels(machine_list)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    
    ax.set_xlabel('Machines', fontsize=12, fontweight='bold')
    ax.set_ylabel('Job Operations', fontsize=12, fontweight='bold')
    ax.set_title('Action Mask: Valid Job-Operation-Machine Combinations\n(Green = Valid Action)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1:
                ax.text(j, i, '✓', ha='center', va='center', color='black', fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Action Validity')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Action mask visualization saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    """
    Simplified Dynamic FJSP visualization without ML dependencies
    """
    print("Dynamic FJSP - Basic Algorithm Comparison and Visualization")
    print("="*80)
    
    random.seed(42)
    np.random.seed(42)
    
    jobs_data = EXTENDED_JOBS_DATA
    machine_list = EXTENDED_MACHINE_LIST
    arrival_times = EXTENDED_ARRIVAL_TIMES
    
    print(f"\nProblem Instance: {len(jobs_data)} jobs, {len(machine_list)} machines")
    print(f"Dynamic arrival times: {arrival_times}")
    print(f"Total operations: {sum(len(ops) for ops in jobs_data.values())}")
    
    # 1. Plot Job Structure Table
    print("\n1. PLOTTING JOB STRUCTURE...")
    print("-" * 50)
    plot_job_structure_table(jobs_data, machine_list, arrival_times, 
                           save_path="job_structure_table.png")
    
    # 2. Simple Action Mask Visualization
    print("\n2. VISUALIZING ACTION MASK...")
    print("-" * 50)
    simple_visualize_action_mask(jobs_data, machine_list, 
                                save_path="action_mask_visualization.png")
    
    # 3. Get schedules from basic algorithms
    print("\n3. RUNNING SCHEDULING ALGORITHMS...")
    print("-" * 50)
    
    schedules = []
    makespans = []
    titles = []
    
    # 3.1. MILP Optimal Solution
    print("Running MILP optimal scheduler...")
    try:
        milp_makespan, milp_schedule = milp_scheduler(jobs_data, machine_list, arrival_times)
        schedules.append(milp_schedule)
        makespans.append(milp_makespan)
        titles.append("MILP Optimal")
        print(f"MILP Makespan: {milp_makespan:.2f}")
    except Exception as e:
        print(f"MILP solver error: {e}")
        schedules.append({m: [] for m in machine_list})
        makespans.append(float('inf'))
        titles.append("MILP Optimal (Failed)")
    
    # 3.2. SPT Heuristic
    print("Running SPT heuristic...")
    spt_makespan, spt_schedule = heuristic_spt_scheduler(jobs_data, machine_list, arrival_times)
    schedules.append(spt_schedule)
    makespans.append(spt_makespan)
    titles.append("SPT Heuristic")
    print(f"SPT Makespan: {spt_makespan:.2f}")
    
    # 3.3 & 3.4. Placeholder for RL methods (will show "Method Not Available")
    schedules.extend([{m: [] for m in machine_list}, {m: [] for m in machine_list}])
    makespans.extend([float('inf'), float('inf')])
    titles.extend(["Static RL (Not Available)", "Dynamic RL (Not Available)"])
    
    # 4. Create Gantt chart comparison
    print("\n4. CREATING GANTT CHART COMPARISON...")
    print("-" * 50)
    
    arrival_times_list = [arrival_times] * len(schedules)
    
    # Only plot the working methods
    working_schedules = schedules[:2]  # MILP and SPT
    working_makespans = makespans[:2]
    working_titles = titles[:2]
    working_arrival_times = arrival_times_list[:2]
    
    plot_gantt_charts(
        figure_num=1, 
        schedules=working_schedules, 
        makespans=working_makespans, 
        titles=working_titles, 
        machine_list=machine_list, 
        arrival_times=working_arrival_times,
        save_path="figure1_basic_comparison.png"
    )
    
    # 5. Results Summary
    print("\n5. RESULTS SUMMARY")
    print("=" * 80)
    
    print("Algorithm Performance Comparison:")
    valid_results = [(title, makespan) for title, makespan in zip(titles[:2], makespans[:2]) 
                    if makespan != float('inf')]
    
    if valid_results:
        sorted_results = sorted(valid_results, key=lambda x: x[1])
        best_makespan = sorted_results[0][1]
        
        for i, (method, makespan) in enumerate(sorted_results):
            gap = makespan - best_makespan
            gap_percent = (gap / best_makespan * 100) if best_makespan > 0 else 0
            print(f"{i+1:2d}. {method:25s}: {makespan:7.2f} (+{gap:5.2f}, +{gap_percent:4.1f}%)")
        
        print(f"\nBest performing method: {sorted_results[0][0]}")
        
        print("\nNOTE: RL methods require PyTorch installation to be fixed.")
        print("Once PyTorch is working, run the full version for complete comparison.")
    
    print("\n" + "="*80)
    print("BASIC DYNAMIC FJSP ANALYSIS COMPLETED")
    print("="*80)
