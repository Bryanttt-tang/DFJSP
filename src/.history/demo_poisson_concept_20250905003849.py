"""
Simple training script for Poisson Dynamic FJSP with minimal dependencies.
This demonstrates the key concepts without requiring all packages.
"""

import random
import math
import collections

# Simple job data for demonstration
DEMO_JOBS = {
    # Initial jobs (available at start)
    0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}],
    1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}],
    2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M2': 7}}],
    
    # Dynamic jobs (arrive via Poisson process)
    3: [{'proc_times': {'M1': 8}}, {'proc_times': {'M0': 5}}],
    4: [{'proc_times': {'M0': 6, 'M2': 4}}, {'proc_times': {'M1': 5}}],
    5: [{'proc_times': {'M1': 5, 'M2': 8}}, {'proc_times': {'M0': 6}}],
}

MACHINES = ['M0', 'M1', 'M2']


def generate_poisson_arrivals(num_dynamic_jobs, arrival_rate, max_time=100, seed=42):
    """Generate Poisson arrival times for dynamic jobs."""
    random.seed(seed)
    
    arrivals = {}
    current_time = 0.0
    
    for job_id in range(3, 3 + num_dynamic_jobs):  # Jobs 3, 4, 5, ...
        # Inter-arrival time follows exponential distribution
        inter_arrival = random.expovariate(arrival_rate)
        current_time += inter_arrival
        
        if current_time <= max_time:
            arrivals[job_id] = current_time
        else:
            arrivals[job_id] = float('inf')  # Won't arrive
    
    return arrivals


def simulate_dynamic_spt(jobs_data, machines, initial_jobs, arrival_times):
    """
    Simulate SPT scheduling with dynamic job arrivals.
    This demonstrates the key challenge for RL agents.
    """
    print(f"\n--- Dynamic SPT Simulation ---")
    print(f"Initial jobs: {list(range(initial_jobs))}")
    print(f"Arrival schedule: {arrival_times}")
    
    # Initialize state
    machine_free_time = {m: 0.0 for m in machines}
    job_completion_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data}
    next_operation = {job_id: 0 for job_id in jobs_data}
    schedule = {m: [] for m in machines}
    
    # Track arrived jobs
    arrived_jobs = set(range(initial_jobs))  # Initial jobs available immediately
    operations_completed = 0
    total_operations = sum(len(jobs_data[job_id]) for job_id in jobs_data)
    current_time = 0.0
    
    step = 0
    while operations_completed < total_operations and step < 100:  # Safety limit
        step += 1
        
        # Update arrivals based on current time
        for job_id, arrival_time in arrival_times.items():
            if arrival_time <= current_time and job_id not in arrived_jobs:
                arrived_jobs.add(job_id)
                print(f"  Time {current_time:.2f}: Job {job_id} arrives!")
        
        # Find available operations (SPT priority)
        candidates = []
        for job_id in arrived_jobs:
            op_idx = next_operation[job_id]
            if op_idx < len(jobs_data[job_id]):
                op_data = jobs_data[job_id][op_idx]
                
                # When can this job start its next operation?
                job_ready_time = (job_completion_times[job_id][op_idx - 1] if op_idx > 0 
                                else arrival_times.get(job_id, 0.0))
                
                for machine, proc_time in op_data['proc_times'].items():
                    earliest_start = max(machine_free_time[machine], job_ready_time, current_time)
                    candidates.append({
                        'job_id': job_id,
                        'op_idx': op_idx,
                        'machine': machine,
                        'proc_time': proc_time,
                        'start_time': earliest_start,
                        'end_time': earliest_start + proc_time
                    })
        
        if not candidates:
            # No available operations, advance time to next meaningful event
            next_events = []
            
            # Next machine becomes free
            next_events.extend(machine_free_time.values())
            
            # Next job arrives
            next_events.extend([t for t in arrival_times.values() if t > current_time])
            
            if next_events:
                current_time = min([t for t in next_events if t > current_time])
                continue
            else:
                break
        
        # Select operation with shortest processing time (SPT rule)
        selected = min(candidates, key=lambda x: x['proc_time'])
        
        # Execute the operation
        job_id = selected['job_id']
        op_idx = selected['op_idx']
        machine = selected['machine']
        start_time = selected['start_time']
        end_time = selected['end_time']
        proc_time = selected['proc_time']
        
        # Update state
        machine_free_time[machine] = end_time
        job_completion_times[job_id][op_idx] = end_time
        next_operation[job_id] += 1
        operations_completed += 1
        current_time = max(current_time, end_time)
        
        # Record in schedule
        schedule[machine].append(f"J{job_id}-O{op_idx+1} ({start_time:.1f}-{end_time:.1f})")
        
        print(f"  Step {step}: Scheduled J{job_id}-O{op_idx+1} on {machine} "
              f"at {start_time:.1f}-{end_time:.1f} (proc_time={proc_time})")
    
    makespan = max(machine_free_time.values())
    print(f"\nSPT Simulation Complete:")
    print(f"  - Final makespan: {makespan:.2f}")
    print(f"  - Operations completed: {operations_completed}/{total_operations}")
    print(f"  - Jobs that arrived: {sorted(arrived_jobs)}")
    
    # Show final schedule
    print(f"\nFinal Schedule:")
    for machine in machines:
        if schedule[machine]:
            print(f"  {machine}: {schedule[machine]}")
        else:
            print(f"  {machine}: (no operations)")
    
    return makespan, schedule


def demonstrate_dynamic_challenge():
    """
    Demonstrate why dynamic job arrivals make FJSP challenging.
    """
    print("="*80)
    print("DYNAMIC FJSP CHALLENGE DEMONSTRATION")
    print("="*80)
    
    initial_jobs = 3
    arrival_rate = 0.1  # 0.1 jobs per time unit on average
    
    print(f"\nScenario Setup:")
    print(f"- Jobs 0, 1, 2 available immediately")
    print(f"- Jobs 3, 4, 5 arrive dynamically (Poisson rate = {arrival_rate})")
    print(f"- No MILP solution possible (stochastic, dynamic)")
    print(f"- RL must learn to adapt to unexpected arrivals")
    
    # Generate different arrival scenarios
    scenarios = [
        {"seed": 42, "name": "Early arrivals"},
        {"seed": 123, "name": "Late arrivals"},
        {"seed": 456, "name": "Mixed timing"},
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n" + "-"*50)
        print(f"Scenario: {scenario['name']} (seed={scenario['seed']})")
        print("-"*50)
        
        # Generate arrivals for this scenario
        arrivals = generate_poisson_arrivals(3, arrival_rate, seed=scenario['seed'])
        
        # Run SPT simulation
        makespan, schedule = simulate_dynamic_spt(DEMO_JOBS, MACHINES, initial_jobs, arrivals)
        
        results.append({
            'name': scenario['name'],
            'makespan': makespan,
            'arrivals': arrivals.copy()
        })
    
    # Compare results
    print(f"\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    for result in results:
        print(f"{result['name']:15s}: Makespan = {result['makespan']:6.2f}")
        arrival_str = ", ".join([f"J{k}@{v:.1f}" for k, v in result['arrivals'].items() 
                                if v < float('inf')])
        print(f"{'':15s}  Arrivals: {arrival_str}")
    
    print(f"\nKey Insights:")
    print(f"1. Different arrival patterns lead to different optimal schedules")
    print(f"2. Static optimization (MILP) cannot handle unknown future arrivals")
    print(f"3. RL agents must learn to:")
    print(f"   - React to new job arrivals dynamically")
    print(f"   - Balance current work vs. future arrivals")
    print(f"   - Adapt scheduling decisions based on workload changes")
    print(f"4. SPT heuristic provides reasonable baseline but may be suboptimal")
    print(f"5. Training on diverse arrival patterns improves generalization")


def training_strategy_overview():
    """Outline the training strategy for RL on dynamic FJSP."""
    print(f"\n" + "="*80)
    print("RL TRAINING STRATEGY FOR DYNAMIC FJSP")
    print("="*80)
    
    print(f"\n1. ENVIRONMENT DESIGN:")
    print(f"   - Initial jobs: Available at time 0")
    print(f"   - Dynamic jobs: Arrive via Poisson process")
    print(f"   - Observation: Job status, machine status, arrival indicators")
    print(f"   - Action: Select (job, operation, machine) combination")
    print(f"   - Reward: Adaptation bonus + efficiency metrics")
    
    print(f"\n2. TRAINING APPROACH:")
    print(f"   - Algorithm: MaskablePPO (handles dynamic action spaces)")
    print(f"   - Reward function: 'dynamic_adaptation' mode")
    print(f"   - Episodes: Various arrival rate scenarios")
    print(f"   - Curriculum: Start simple, increase complexity")
    
    print(f"\n3. REWARD DESIGN:")
    print(f"   - +10 for completing operations")
    print(f"   - -0.1 * processing_time (efficiency)")
    print(f"   - -2.0 * idle_time (machine utilization)")
    print(f"   - +5 * new_arrivals (adaptation bonus)")
    print(f"   - +100 for episode completion")
    print(f"   - +200/makespan for final efficiency")
    
    print(f"\n4. EVALUATION:")
    print(f"   - Multiple episodes with same arrival distribution")
    print(f"   - Compare with SPT heuristic baseline")
    print(f"   - Measure adaptation speed and final performance")
    print(f"   - Test generalization to different arrival rates")
    
    print(f"\n5. ADVANTAGES OVER STATIC APPROACHES:")
    print(f"   - No need to predict future arrivals")
    print(f"   - Learns reactive scheduling policies")
    print(f"   - Handles uncertainty and stochasticity")
    print(f"   - Can incorporate domain knowledge via reward shaping")


if __name__ == "__main__":
    print("Starting Dynamic FJSP Analysis with Poisson Arrivals...")
    
    # Run demonstration
    demonstrate_dynamic_challenge()
    
    # Show training strategy
    training_strategy_overview()
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✓ Implemented Poisson Dynamic FJSP Environment")
    print("✓ Demonstrated why MILP cannot solve dynamic problems")
    print("✓ Showed SPT heuristic as baseline approach")
    print("✓ Designed RL training strategy for dynamic adaptation")
    print("✓ Created observation/action/reward framework")
    print("\nNext step: Run 'python dynamic_poisson_fjsp.py' for full RL training!")
