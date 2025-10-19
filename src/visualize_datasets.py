"""
Visualization script for generated FJSP datasets.
Helps understand dataset characteristics through plots and statistics.
"""

from utils import generate_fjsp_dataset, generate_arrival_times
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def visualize_dataset_properties(jobs_data, machine_list, arrival_times=None):
    """
    Create visualizations showing dataset characteristics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FJSP Dataset Characteristics', fontsize=16, fontweight='bold')
    
    # 1. Operations per job (Bar chart)
    ax = axes[0, 0]
    job_ids = list(jobs_data.keys())
    ops_per_job = [len(ops) for ops in jobs_data.values()]
    
    ax.bar(job_ids, ops_per_job, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Job ID', fontsize=12)
    ax.set_ylabel('Number of Operations', fontsize=12)
    ax.set_title('Operations per Job', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(job_ids)
    
    # 2. Machines available per operation (Histogram)
    ax = axes[0, 1]
    machines_per_op = []
    for job_ops in jobs_data.values():
        for op in job_ops:
            machines_per_op.append(len(op['proc_times']))
    
    ax.hist(machines_per_op, bins=range(1, len(machine_list)+2), 
            color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Machines Available', fontsize=12)
    ax.set_ylabel('Number of Operations', fontsize=12)
    ax.set_title('Machine Availability Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Processing time distribution (Histogram)
    ax = axes[0, 2]
    all_proc_times = []
    for job_ops in jobs_data.values():
        for op in job_ops:
            all_proc_times.extend(op['proc_times'].values())
    
    ax.hist(all_proc_times, bins=range(1, 12), color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Processing Time (units)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Processing Time Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Machine workload (operations assigned to each machine)
    ax = axes[1, 0]
    machine_workload = defaultdict(int)
    for job_ops in jobs_data.values():
        for op in job_ops:
            for machine in op['proc_times'].keys():
                machine_workload[machine] += 1
    
    machines = sorted(machine_workload.keys())
    workloads = [machine_workload[m] for m in machines]
    
    ax.bar(machines, workloads, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Machine', fontsize=12)
    ax.set_ylabel('Number of Compatible Operations', fontsize=12)
    ax.set_title('Machine Workload Potential', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Arrival times (if provided)
    ax = axes[1, 1]
    if arrival_times:
        arrivals = sorted(arrival_times.items())
        job_ids_arr = [j for j, _ in arrivals]
        times = [t for _, t in arrivals]
        
        colors = ['blue' if t == 0 else 'orange' for t in times]
        ax.bar(job_ids_arr, times, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Job ID', fontsize=12)
        ax.set_ylabel('Arrival Time', fontsize=12)
        ax.set_title('Job Arrival Schedule', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(job_ids_arr)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Initial (t=0)'),
            Patch(facecolor='orange', alpha=0.7, label='Future')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
    else:
        ax.text(0.5, 0.5, 'No arrival times provided', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    # 6. Dataset statistics (Text summary)
    ax = axes[1, 2]
    ax.axis('off')
    
    total_jobs = len(jobs_data)
    total_ops = sum(len(ops) for ops in jobs_data.values())
    avg_ops = total_ops / total_jobs
    min_ops = min(len(ops) for ops in jobs_data.values())
    max_ops = max(len(ops) for ops in jobs_data.values())
    
    avg_machines_per_op = np.mean(machines_per_op)
    avg_proc_time = np.mean(all_proc_times)
    
    stats_text = f"""
    DATASET STATISTICS
    
    Jobs: {total_jobs}
    Machines: {len(machine_list)}
    Total Operations: {total_ops}
    
    Operations per Job:
      Mean: {avg_ops:.2f}
      Min: {min_ops}
      Max: {max_ops}
    
    Machines per Operation:
      Mean: {avg_machines_per_op:.2f}
      Min: {min(machines_per_op)}
      Max: {max(machines_per_op)}
    
    Processing Time:
      Mean: {avg_proc_time:.2f}
      Min: {min(all_proc_times)}
      Max: {max(all_proc_times)}
    """
    
    if arrival_times:
        initial_jobs = sum(1 for t in arrival_times.values() if t == 0)
        future_jobs = total_jobs - initial_jobs
        max_arrival = max(arrival_times.values())
        stats_text += f"""
    Arrivals:
      Initial: {initial_jobs}
      Future: {future_jobs}
      Latest: {max_arrival:.2f}
        """
    
    ax.text(0.1, 0.9, stats_text, fontsize=11, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def compare_arrival_patterns(num_initial, num_future, num_machines, seed=42):
    """
    Compare deterministic vs Poisson arrivals for the same job structure.
    """
    # Generate job structure
    jobs_data, machine_list = generate_fjsp_dataset(
        num_initial_jobs=num_initial,
        num_future_jobs=num_future,
        total_num_machines=num_machines,
        seed=seed
    )
    
    # Generate both arrival patterns
    arrivals_det = generate_arrival_times(
        num_initial_jobs=num_initial,
        num_future_jobs=num_future,
        arrival_mode='deterministic',
        seed=seed
    )
    
    arrivals_poisson = generate_arrival_times(
        num_initial_jobs=num_initial,
        num_future_jobs=num_future,
        arrival_mode='poisson',
        arrival_rate=0.08,
        seed=seed
    )
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Arrival Pattern Comparison', fontsize=14, fontweight='bold')
    
    # Deterministic
    ax = axes[0]
    arrivals = sorted(arrivals_det.items())
    job_ids = [j for j, _ in arrivals]
    times = [t for _, t in arrivals]
    colors = ['blue' if t == 0 else 'orange' for t in times]
    
    ax.bar(job_ids, times, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Job ID', fontsize=11)
    ax.set_ylabel('Arrival Time', fontsize=11)
    ax.set_title('Deterministic Arrivals\n(Evenly Spaced)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(job_ids)
    
    # Poisson
    ax = axes[1]
    arrivals = sorted(arrivals_poisson.items())
    job_ids = [j for j, _ in arrivals]
    times = [t for _, t in arrivals]
    colors = ['blue' if t == 0 else 'orange' for t in times]
    
    ax.bar(job_ids, times, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Job ID', fontsize=11)
    ax.set_ylabel('Arrival Time', fontsize=11)
    ax.set_title('Poisson Arrivals\n(Stochastic)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(job_ids)
    
    plt.tight_layout()
    return fig


def main():
    """
    Generate and visualize example datasets.
    """
    print("="*80)
    print("FJSP DATASET VISUALIZATION")
    print("="*80)
    
    # Example 1: Small dataset with deterministic arrivals
    print("\nGenerating and visualizing small dataset...")
    jobs_data, machine_list = generate_fjsp_dataset(
        num_initial_jobs=3,
        num_future_jobs=4,
        total_num_machines=3,
        seed=42
    )
    
    arrival_times = generate_arrival_times(
        num_initial_jobs=3,
        num_future_jobs=4,
        arrival_mode='deterministic',
        seed=42
    )
    
    fig1 = visualize_dataset_properties(jobs_data, machine_list, arrival_times)
    plt.savefig('dataset_visualization_small.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: dataset_visualization_small.png")
    
    # Example 2: Larger dataset
    print("\nGenerating and visualizing larger dataset...")
    jobs_data2, machine_list2 = generate_fjsp_dataset(
        num_initial_jobs=10,
        num_future_jobs=10,
        total_num_machines=5,
        seed=123
    )
    
    arrival_times2 = generate_arrival_times(
        num_initial_jobs=10,
        num_future_jobs=10,
        arrival_mode='poisson',
        arrival_rate=0.1,
        seed=123
    )
    
    fig2 = visualize_dataset_properties(jobs_data2, machine_list2, arrival_times2)
    plt.savefig('dataset_visualization_large.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: dataset_visualization_large.png")
    
    # Example 3: Compare arrival patterns
    print("\nComparing arrival patterns...")
    fig3 = compare_arrival_patterns(
        num_initial=5,
        num_future=5,
        num_machines=3,
        seed=456
    )
    plt.savefig('arrival_pattern_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: arrival_pattern_comparison.png")
    
    print("\n" + "="*80)
    print("Visualization complete! Check the generated PNG files.")
    print("="*80)
    
    plt.show()


if __name__ == "__main__":
    main()
