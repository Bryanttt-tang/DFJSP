"""
Utility functions for Flexible Job Shop Scheduling Problem (FJSP)
"""
import collections
import random
import numpy as np


def generate_fjsp_dataset(num_initial_jobs, num_future_jobs, total_num_machines, seed=None):
    """
    Generate a Flexible Job Shop Problem dataset with random job configurations.
    
    Parameters:
    -----------
    num_initial_jobs : int
        Number of jobs available at time 0
    num_future_jobs : int
        Number of jobs that arrive dynamically
    total_num_machines : int
        Total number of machines available (M)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    jobs_data : collections.OrderedDict
        Dictionary mapping job_id to list of operations
        Each operation has 'proc_times' dict mapping machine to processing time
    machine_list : list
        List of machine names ['M0', 'M1', ..., 'M{total_num_machines-1}']
        
    Generation Rules:
    -----------------
    - Number of operations per job: Uniform[1, 5]
    - Number of available machines per operation: Uniform[1, M]
    - Processing time per operation-machine pair: Uniform[1, 50]
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    total_jobs = num_initial_jobs + num_future_jobs
    machine_list = [f'M{i}' for i in range(total_num_machines)]
    
    jobs_data = collections.OrderedDict()
    
    for job_id in range(total_jobs):
        # Number of operations for this job: Uniform[1, 5]
        num_operations = random.randint(1, 5)
        
        operations = []
        for op_idx in range(num_operations):
            # Number of available machines for this operation: Uniform[1, M]
            num_available_machines = random.randint(1, total_num_machines)
            
            # Randomly select which machines are available
            available_machines = random.sample(machine_list, num_available_machines)
            
            # Generate processing times for each available machine: Uniform[1, 10]
            proc_times = {}
            for machine in available_machines:
                proc_time = random.randint(1, 10)
                proc_times[machine] = proc_time
            
            operations.append({'proc_times': proc_times})
        
        jobs_data[job_id] = operations
    
    return jobs_data, machine_list


def generate_simplified_fjsp_dataset(num_initial_jobs, num_future_jobs, total_num_machines,
                                     machine_speed_variance=0.5, proc_time_variance_range=(1, 50),
                                     seed=None):
    """
    Generate a SIMPLIFIED Flexible Job Shop Problem dataset with:
    1. NO job classification - all jobs are homogeneous in structure
    2. Machine heterogeneity (fast/medium/slow machines) - KEY SOURCE OF STRATEGIC DECISIONS
    3. High variance in processing times across jobs - creates wait opportunities
    
    DESIGN PHILOSOPHY:
    ------------------
    Without loss of generality, we assume a specific job arrival sequence (e.g., J0→J1→J2→...)
    since jobs are homogeneous. The strategic depth comes from:
    
    1. **Machine Heterogeneity**: Fast machines process ALL jobs faster than slow machines
       - Fast machine: proc_time * 0.6-0.8
       - Slow machine: proc_time * 1.2-1.5
       - Gap: Up to 2.5x difference!
    
    2. **Processing Time Variance**: Even without job types, individual operations vary widely
       - Operation A: base_time = 10
       - Operation B: base_time = 45
       - This creates waiting opportunities: worth waiting for fast machine on long operations
    
    3. **Poisson Arrivals**: Jobs arrive randomly, forcing agent to decide:
       - Schedule now on available slow machine?
       - Wait for fast machine (but when will it be free)?
       - Wait for next job arrival (but when will it arrive)?
    
    WAIT ACTION BECOMES CRITICAL:
    -----------------------------
    Example: Operation with proc_time=40
    - On slow machine (speed=1.5): 40 * 1.5 = 60 time units
    - On fast machine (speed=0.7): 40 * 0.7 = 28 time units
    - Savings: 32 time units!
    - Decision: Worth waiting if fast machine free soon OR next job is fast
    
    Parameters:
    -----------
    num_initial_jobs : int
        Number of jobs available at time 0
    num_future_jobs : int
        Number of jobs that arrive dynamically via Poisson process
    total_num_machines : int
        Total number of machines (will be categorized as fast/medium/slow)
    machine_speed_variance : float
        How much machines differ in speed (0=identical, 1=very different)
        Recommended: 0.5-0.7 for meaningful wait decisions
    proc_time_variance_range : tuple
        Range for base processing times (min, max)
        Larger range = more variance = more strategic wait opportunities
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    jobs_data : collections.OrderedDict
        Dictionary mapping job_id to list of operations (NO metadata)
        Format: {job_id: [{'proc_times': {machine: time, ...}}, ...]}
    machine_list : list
        List of machine names ['M0', 'M1', ..., 'M{total_num_machines-1}']
    machine_metadata : dict
        Machine information: {machine_name: {'speed_factor': float, 'category': str}}
        
    Generation Rules:
    -----------------
    - Number of operations per job: Uniform[2, 4] (consistent across jobs)
    - Number of available machines per operation: Uniform[max(2, M//2), M]
    - Base processing time: Uniform[proc_time_variance_range]
    - Actual processing time: base_time * machine_speed_factor
    
    Machine Categories:
    -------------------
    - FAST machines: speed_factor = 0.6-0.8 (20-40% faster)
    - MEDIUM machines: speed_factor = 0.9-1.1 (baseline)
    - SLOW machines: speed_factor = 1.2-1.5 (20-50% slower)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    total_jobs = num_initial_jobs + num_future_jobs
    machine_list = [f'M{i}' for i in range(total_num_machines)]
    
    # Create machine heterogeneity (KEY STRATEGIC ELEMENT)
    machine_metadata = _create_machine_categories(machine_list, machine_speed_variance)
    
    jobs_data = collections.OrderedDict()
    
    for job_id in range(total_jobs):
        # All jobs have similar structure (homogeneous)
        num_operations = random.randint(2, 4)
        
        operations = []
        for op_idx in range(num_operations):
            # Sample base processing time with HIGH VARIANCE
            base_proc_time = random.randint(*proc_time_variance_range)
            
            # Determine which machines can process this operation
            num_compatible_machines = random.randint(
                max(2, total_num_machines // 2),  # At least half machines
                total_num_machines
            )
            compatible_machines = random.sample(machine_list, num_compatible_machines)
            
            # Calculate actual processing times based on machine speed
            # THIS IS WHERE STRATEGIC WAITING DECISIONS EMERGE
            proc_times = {}
            for machine in compatible_machines:
                speed_factor = machine_metadata[machine]['speed_factor']
                # Actual time = base_time * speed_factor
                actual_proc_time = int(base_proc_time * speed_factor)
                actual_proc_time = max(1, actual_proc_time)
                proc_times[machine] = actual_proc_time
            
            operations.append({'proc_times': proc_times})
        
        # NO METADATA - jobs are homogeneous
        jobs_data[job_id] = operations
    
    return jobs_data, machine_list, machine_metadata


def generate_realistic_fjsp_dataset(num_initial_jobs, num_future_jobs, total_num_machines, 
                                    job_type_distribution=None, machine_speed_variance=0.5,
                                    seed=None):
    """
    Generate a REALISTIC Flexible Job Shop Problem dataset with:
    1. Job categorization (short/moderate/long jobs)
    2. Machine heterogeneity (fast/medium/slow machines)
    3. Structured processing time relationships
    
    This creates strategic depth: waiting for fast machines vs scheduling on slow machines.
    
    Parameters:
    -----------
    num_initial_jobs : int
        Number of jobs available at time 0
    num_future_jobs : int
        Number of jobs that arrive dynamically
    total_num_machines : int
        Total number of machines (will be categorized as fast/medium/slow)
    job_type_distribution : dict, optional
        Distribution of job types: {'short': 0.5, 'moderate': 0.3, 'long': 0.2}
        If None, uses default distribution
    machine_speed_variance : float
        How much machines differ in speed (0=identical, 1=very different)
        Controls the gap between fast and slow machines
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    jobs_data : collections.OrderedDict
        Dictionary mapping job_id to list of operations
        Each job has metadata: {'type': 'short'|'moderate'|'long', 'operations': [...]}
    machine_list : list
        List of machine names
    machine_metadata : dict
        Machine information: {machine_name: {'speed_factor': float, 'category': str}}
        
    Job Categories:
    ---------------
    - SHORT jobs: 1-2 operations, base processing time 5-15 per operation
    - MODERATE jobs: 2-4 operations, base processing time 15-30 per operation
    - LONG jobs: 3-5 operations, base processing time 30-50 per operation
    
    Machine Categories:
    -------------------
    - FAST machines: speed_factor = 0.6-0.8 (40% faster than baseline)
    - MEDIUM machines: speed_factor = 0.9-1.1 (baseline speed)
    - SLOW machines: speed_factor = 1.2-1.5 (50% slower than baseline)
    
    Key Design Principle:
    ---------------------
    Processing time = base_time * machine_speed_factor
    
    This creates meaningful wait decisions:
    - Long job on slow machine: 50 * 1.5 = 75 time units
    - Long job on fast machine: 50 * 0.7 = 35 time units
    - Difference: 40 time units! (Worth waiting for fast machine!)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Default job type distribution
    if job_type_distribution is None:
        job_type_distribution = {
            'short': 0.5,      # 50% short jobs
            'moderate': 0.3,   # 30% moderate jobs
            'long': 0.2        # 20% long jobs
        }
    
    total_jobs = num_initial_jobs + num_future_jobs
    
    # Step 1: Create machine categorization (fast/medium/slow)
    machine_list = [f'M{i}' for i in range(total_num_machines)]
    machine_metadata = _create_machine_categories(machine_list, machine_speed_variance)
    
    # Step 2: Determine job types for all jobs
    job_types = _assign_job_types(total_jobs, job_type_distribution)
    
    # Step 3: Generate jobs with realistic processing times
    jobs_data = collections.OrderedDict()
    
    for job_id in range(total_jobs):
        job_type = job_types[job_id]
        
        # Determine number of operations based on job type
        if job_type == 'short':
            num_operations = random.randint(1, 2)
            base_proc_time_range = (5, 15)
        elif job_type == 'moderate':
            num_operations = random.randint(2, 4)
            base_proc_time_range = (15, 30)
        else:  # long
            num_operations = random.randint(3, 5)
            base_proc_time_range = (30, 50)
        
        operations = []
        for op_idx in range(num_operations):
            # Sample base processing time for this operation
            base_proc_time = random.randint(*base_proc_time_range)
            
            # Determine which machines can process this operation
            # Not all machines can process all operations (adds realism)
            num_compatible_machines = random.randint(
                max(2, total_num_machines // 2),  # At least half machines, minimum 2
                total_num_machines
            )
            compatible_machines = random.sample(machine_list, num_compatible_machines)
            
            # Calculate actual processing times based on machine speed factors
            proc_times = {}
            for machine in compatible_machines:
                speed_factor = machine_metadata[machine]['speed_factor']
                # Actual time = base_time * speed_factor
                # Add small random noise (±10%) for realism
                noise = random.uniform(0.9, 1.1)
                actual_proc_time = int(base_proc_time * speed_factor * noise)
                actual_proc_time = max(1, actual_proc_time)  # Ensure at least 1
                proc_times[machine] = actual_proc_time
            
            operations.append({'proc_times': proc_times})
        
        # Store job with metadata
        jobs_data[job_id] = {
            'type': job_type,
            'operations': operations
        }
    
    return jobs_data, machine_list, machine_metadata


def _create_machine_categories(machine_list, speed_variance):
    """
    Categorize machines into fast/medium/slow with different speed factors.
    
    Speed variance controls how different the machines are:
    - 0.0: All machines identical (no heterogeneity)
    - 0.5: Moderate differences (realistic)
    - 1.0: Large differences (high heterogeneity)
    """
    num_machines = len(machine_list)
    machine_metadata = {}
    
    # Distribute machines across categories
    num_fast = max(1, num_machines // 4)      # 25% fast
    num_slow = max(1, num_machines // 4)      # 25% slow
    num_medium = num_machines - num_fast - num_slow  # 50% medium
    
    # Shuffle to randomize which machines are fast/slow
    shuffled_machines = machine_list.copy()
    random.shuffle(shuffled_machines)
    
    machine_idx = 0
    
    # Assign FAST machines
    for i in range(num_fast):
        machine = shuffled_machines[machine_idx]
        # Fast machines: 0.6 to 0.8 (20-40% faster)
        speed_factor = random.uniform(
            1.0 - 0.4 * speed_variance,  # At variance=0.5: 0.8
            1.0 - 0.2 * speed_variance   # At variance=0.5: 0.9
        )
        machine_metadata[machine] = {
            'speed_factor': speed_factor,
            'category': 'fast'
        }
        machine_idx += 1
    
    # Assign MEDIUM machines
    for i in range(num_medium):
        machine = shuffled_machines[machine_idx]
        # Medium machines: 0.9 to 1.1 (baseline)
        speed_factor = random.uniform(
            1.0 - 0.1 * speed_variance,
            1.0 + 0.1 * speed_variance
        )
        machine_metadata[machine] = {
            'speed_factor': speed_factor,
            'category': 'medium'
        }
        machine_idx += 1
    
    # Assign SLOW machines
    for i in range(num_slow):
        machine = shuffled_machines[machine_idx]
        # Slow machines: 1.2 to 1.5 (20-50% slower)
        speed_factor = random.uniform(
            1.0 + 0.2 * speed_variance,  # At variance=0.5: 1.1
            1.0 + 0.5 * speed_variance   # At variance=0.5: 1.25
        )
        machine_metadata[machine] = {
            'speed_factor': speed_factor,
            'category': 'slow'
        }
        machine_idx += 1
    
    return machine_metadata


def _assign_job_types(total_jobs, distribution):
    """
    Assign job types to all jobs based on distribution.
    
    Returns list of job types in order: ['short', 'short', 'long', 'moderate', ...]
    """
    job_types = []
    
    # Calculate counts
    num_short = int(total_jobs * distribution['short'])
    num_moderate = int(total_jobs * distribution['moderate'])
    num_long = total_jobs - num_short - num_moderate  # Remainder
    
    # Create list with all types
    job_types.extend(['short'] * num_short)
    job_types.extend(['moderate'] * num_moderate)
    job_types.extend(['long'] * num_long)
    
    # Shuffle to randomize order (but we'll use patterns during arrival generation)
    random.shuffle(job_types)
    
    return job_types


def generate_arrival_times(num_initial_jobs, num_future_jobs, arrival_mode='deterministic', 
                           arrival_rate=0.08, seed=None):
    """
    Generate job arrival times for dynamic scheduling.
    
    Parameters:
    -----------
    num_initial_jobs : int
        Number of jobs available at time 0
    num_future_jobs : int
        Number of jobs arriving dynamically
    arrival_mode : str
        'deterministic' - evenly spaced arrivals
        'poisson' - Poisson distributed arrivals
        'custom' - provide custom times
    arrival_rate : float
        Rate parameter for Poisson arrivals (used only if arrival_mode='poisson')
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    arrival_times : dict
        Dictionary mapping job_id to arrival time
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    total_jobs = num_initial_jobs + num_future_jobs
    arrival_times = {}
    
    # Initial jobs arrive at time 0
    for job_id in range(num_initial_jobs):
        arrival_times[job_id] = 0.0
    
    if arrival_mode == 'deterministic':
        # Evenly spaced arrivals
        if num_future_jobs > 0:
            spacing = 4  # Default spacing of 4 time units
            for i, job_id in enumerate(range(num_initial_jobs, total_jobs)):
                arrival_times[job_id] = (i + 1) * spacing
                
    elif arrival_mode == 'poisson':
        # Poisson distributed arrivals
        current_time = 0.0
        for job_id in range(num_initial_jobs, total_jobs):
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival_time
            arrival_times[job_id] = round(current_time)
    
    return arrival_times


def generate_realistic_arrival_sequence(jobs_data, num_initial_jobs, arrival_rate=0.08, 
                                        pattern_strength=0.3, seed=None):
    """
    Generate REALISTIC arrival sequence with:
    1. Poisson-distributed arrival TIMES
    2. UNCERTAIN job sequence (which job arrives is unknown)
    3. SOFT PROBABILISTIC PATTERNS (realistic industrial behavior)
    
    Key Innovation: Job arrival SEQUENCE is stochastic, not predetermined!
    
    Parameters:
    -----------
    jobs_data : OrderedDict
        Job data with metadata: {job_id: {'type': 'short'|'moderate'|'long', 'operations': [...]}}
    num_initial_jobs : int
        Number of jobs available at time 0
    arrival_rate : float
        Poisson arrival rate (mean arrivals per time unit)
    pattern_strength : float (0.0 to 1.0)
        How strong the arrival patterns are:
        - 0.0: Pure random (no pattern)
        - 0.5: Realistic patterns (RECOMMENDED)
        - 1.0: Strong deterministic patterns
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    arrival_times : dict
        Dictionary mapping job_id to arrival time
    arrival_sequence : list
        Ordered list of (time, job_id, job_type) tuples showing arrival order
        
    Pattern Rules (when pattern_strength > 0):
    -------------------------------------------
    1. Base distribution: 50% short, 30% moderate, 20% long
    2. After 4+ short jobs: Increase long job probability (workload balancing)
    3. After 1+ long job: Increase short job probability (recovery period)
    4. Moderate jobs: Bridge between clusters
    
    Example Arrival Sequence:
    -------------------------
    t=0:    J0 (short), J1 (short), J2 (moderate)  [initial jobs]
    t=8:    J5 (short)     [random from future jobs]
    t=15:   J7 (short)     [pattern: short after short]
    t=22:   J3 (moderate)  [pattern: moderate emerging]
    t=31:   J9 (long)      [pattern: long after cluster of shorts]
    t=45:   J4 (short)     [pattern: short after long]
    
    This creates STRATEGIC WAITING scenarios:
    - "Many short jobs arrived, should I wait for a long job?"
    - "Long job just finished, should I schedule aggressively?"
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    total_jobs = len(jobs_data)
    
    # Separate initial and future jobs
    initial_job_ids = list(range(num_initial_jobs))
    future_job_ids = list(range(num_initial_jobs, total_jobs))
    
    # Create job type mapping
    job_type_map = {job_id: jobs_data[job_id]['type'] for job_id in jobs_data.keys()}
    
    # Group future jobs by type for pattern-based sampling
    future_jobs_by_type = {
        'short': [j for j in future_job_ids if job_type_map[j] == 'short'],
        'moderate': [j for j in future_job_ids if job_type_map[j] == 'moderate'],
        'long': [j for j in future_job_ids if job_type_map[j] == 'long']
    }
    
    # Initialize outputs
    arrival_times = {}
    arrival_sequence = []
    
    # Initial jobs arrive at t=0
    for job_id in initial_job_ids:
        arrival_times[job_id] = 0.0
        arrival_sequence.append((0.0, job_id, job_type_map[job_id]))
    
    # Generate arrival sequence for future jobs
    current_time = 0.0
    recent_history = []  # Track last 10 arrivals for pattern detection
    
    # Track which jobs have been assigned
    remaining_jobs = {
        'short': future_jobs_by_type['short'].copy(),
        'moderate': future_jobs_by_type['moderate'].copy(),
        'long': future_jobs_by_type['long'].copy()
    }
    
    while any(len(jobs) > 0 for jobs in remaining_jobs.values()):
        # Sample inter-arrival time (Poisson process)
        inter_arrival = np.random.exponential(1.0 / arrival_rate)
        current_time += inter_arrival
        
        # Determine job TYPE based on patterns
        job_type = _sample_next_job_type(
            recent_history, 
            remaining_jobs,
            pattern_strength
        )
        
        # If no jobs of this type remain, pick another type
        if len(remaining_jobs[job_type]) == 0:
            # Find available types
            available_types = [t for t in ['short', 'moderate', 'long'] 
                             if len(remaining_jobs[t]) > 0]
            if len(available_types) == 0:
                break  # No more jobs
            job_type = random.choice(available_types)
        
        # Randomly select specific job from this type
        job_id = random.choice(remaining_jobs[job_type])
        remaining_jobs[job_type].remove(job_id)
        
        # Record arrival
        arrival_times[job_id] = round(current_time, 2)
        arrival_sequence.append((round(current_time, 2), job_id, job_type))
        
        # Update history
        recent_history.append(job_type)
        if len(recent_history) > 10:
            recent_history.pop(0)
    
    return arrival_times, arrival_sequence


def _sample_next_job_type(recent_history, remaining_jobs, pattern_strength):
    """
    Sample next job type based on recent arrival history and patterns.
    
    Pattern Logic:
    --------------
    - After many SHORT jobs → Higher probability of LONG job (workload balancing)
    - After LONG job → Higher probability of SHORT jobs (recovery)
    - MODERATE jobs → Bridge transitions
    
    Pattern strength controls interpolation between:
    - Base distribution (no pattern)
    - Pattern-driven distribution
    """
    # Base distribution (no pattern)
    base_probs = {
        'short': 0.5,
        'moderate': 0.3,
        'long': 0.2
    }
    
    if len(recent_history) == 0 or pattern_strength == 0.0:
        # No history or no pattern → use base distribution
        pattern_probs = base_probs
    else:
        # Detect patterns in recent history
        recent_5 = recent_history[-5:] if len(recent_history) >= 5 else recent_history
        short_count = sum(1 for t in recent_5 if t == 'short')
        long_count = sum(1 for t in recent_5 if t == 'long')
        
        # Pattern-driven probabilities
        if short_count >= 4:
            # Cluster of short jobs → Increase long job probability
            pattern_probs = {
                'short': 0.2,
                'moderate': 0.3,
                'long': 0.5
            }
        elif long_count >= 1 and recent_history[-1] == 'long':
            # Just had long job → Increase short job probability
            pattern_probs = {
                'short': 0.7,
                'moderate': 0.2,
                'long': 0.1
            }
        elif recent_history[-1] == 'moderate':
            # After moderate → Balanced distribution
            pattern_probs = {
                'short': 0.4,
                'moderate': 0.3,
                'long': 0.3
            }
        else:
            # Default: Use base distribution
            pattern_probs = base_probs
    
    # Interpolate between base and pattern probabilities
    final_probs = {}
    for job_type in ['short', 'moderate', 'long']:
        final_probs[job_type] = (
            (1 - pattern_strength) * base_probs[job_type] +
            pattern_strength * pattern_probs[job_type]
        )
    
    # Adjust probabilities if some job types are exhausted
    for job_type in ['short', 'moderate', 'long']:
        if len(remaining_jobs[job_type]) == 0:
            final_probs[job_type] = 0.0
    
    # Normalize probabilities
    total_prob = sum(final_probs.values())
    if total_prob > 0:
        final_probs = {k: v/total_prob for k, v in final_probs.items()}
    else:
        # All types exhausted (shouldn't happen)
        return 'short'
    
    # Sample job type
    job_types = list(final_probs.keys())
    probs = list(final_probs.values())
    
    return np.random.choice(job_types, p=probs)


def print_dataset_info(jobs_data, machine_list, arrival_times=None, machine_metadata=None):
    """
    Print detailed information about the generated dataset.
    
    Parameters:
    -----------
    jobs_data : collections.OrderedDict
        Job data structure (either simple or with metadata)
    machine_list : list
        List of machine names
    arrival_times : dict, optional
        Job arrival times
    machine_metadata : dict, optional
        Machine speed factors and categories
    """
    print("\n" + "="*80)
    print("FLEXIBLE JOB SHOP PROBLEM DATASET")
    print("="*80)
    
    print(f"\nMachines: {len(machine_list)}")
    print(f"Machine List: {machine_list}")
    
    # Print machine metadata if available
    if machine_metadata:
        print("\nMachine Categories:")
        for machine in machine_list:
            meta = machine_metadata[machine]
            print(f"  {machine}: {meta['category']:8s} (speed factor: {meta['speed_factor']:.2f})")
    
    # Handle both simple and metadata job structures
    is_metadata_format = isinstance(list(jobs_data.values())[0], dict) and 'type' in list(jobs_data.values())[0]
    
    if is_metadata_format:
        total_operations = sum(len(job_data['operations']) for job_data in jobs_data.values())
        job_type_counts = {'short': 0, 'moderate': 0, 'long': 0}
        for job_data in jobs_data.values():
            job_type_counts[job_data['type']] += 1
    else:
        total_operations = sum(len(ops) for ops in jobs_data.values())
    
    print(f"\nJobs: {len(jobs_data)}")
    print(f"Total Operations: {total_operations}")
    
    if is_metadata_format:
        print(f"\nJob Type Distribution:")
        print(f"  SHORT jobs: {job_type_counts['short']} ({job_type_counts['short']/len(jobs_data)*100:.1f}%)")
        print(f"  MODERATE jobs: {job_type_counts['moderate']} ({job_type_counts['moderate']/len(jobs_data)*100:.1f}%)")
        print(f"  LONG jobs: {job_type_counts['long']} ({job_type_counts['long']/len(jobs_data)*100:.1f}%)")
    
    if arrival_times:
        initial_jobs = sum(1 for t in arrival_times.values() if t == 0)
        future_jobs = len(arrival_times) - initial_jobs
        print(f"\nInitial Jobs (t=0): {initial_jobs}")
        print(f"Future Jobs: {future_jobs}")
    
    print("\n" + "-"*80)
    print("Job Details:")
    print("-"*80)
    
    for job_id, job_info in jobs_data.items():
        # Handle both formats
        if is_metadata_format:
            job_type = job_info['type']
            operations = job_info['operations']
            type_str = f" [{job_type.upper()}]"
        else:
            operations = job_info
            type_str = ""
        
        arrival_str = f" (arrives at t={arrival_times[job_id]})" if arrival_times else ""
        print(f"\nJob {job_id}{type_str}{arrival_str}:")
        print(f"  Operations: {len(operations)}")
        
        for op_idx, operation in enumerate(operations):
            proc_times = operation['proc_times']
            machines = sorted(proc_times.keys())
            print(f"    Op {op_idx}: {len(machines)} machines available")
            for machine in machines:
                proc_time = proc_times[machine]
                # Show machine category if available
                if machine_metadata:
                    cat = machine_metadata[machine]['category']
                    print(f"      {machine} ({cat:6s}): {proc_time:3d} time units")
                else:
                    print(f"      {machine}: {proc_time} time units")
    
    print("\n" + "="*80)

def print_dataset_table(jobs_data, machine_list, machine_metadata=None):
    """Print generated dataset in a readable table format."""
    print("\n" + "="*120)
    print("GENERATED FJSP DATASET STRUCTURE")
    print("="*120)
    print(f"Total Jobs: {len(jobs_data)} | Machines: {machine_list}")
    
    # Print machine info
    if machine_metadata:
        print("\nMachine Categories:")
        fast_machines = [m for m in machine_list if machine_metadata[m]['category'] == 'fast']
        medium_machines = [m for m in machine_list if machine_metadata[m]['category'] == 'medium']
        slow_machines = [m for m in machine_list if machine_metadata[m]['category'] == 'slow']
        print(f"  FAST machines ({len(fast_machines)}): {fast_machines}")
        print(f"  MEDIUM machines ({len(medium_machines)}): {medium_machines}")
        print(f"  SLOW machines ({len(slow_machines)}): {slow_machines}")
    
    print("="*120)
    
    # Handle both simple and metadata job structures
    is_metadata_format = isinstance(list(jobs_data.values())[0], dict) and 'type' in list(jobs_data.values())[0]
    
    if is_metadata_format:
        max_ops = max(len(job_data['operations']) for job_data in jobs_data.values())
    else:
        max_ops = max(len(ops) for ops in jobs_data.values())
    
    # Print header
    header = f"{'Job':^10} | {'Type':^8} | {'#Ops':^5} | "
    for i in range(max_ops):
        header += f"Operation {i} {' '*20}| "
    print(header)
    print("-"*120)
    
    # Print each job
    for job_id, job_info in jobs_data.items():
        # Handle both formats
        if is_metadata_format:
            job_type = job_info['type']
            operations = job_info['operations']
        else:
            job_type = "N/A"
            operations = job_info
        
        row = f"J{job_id:2d} {' '*5} | {job_type.upper():^8} | {len(operations):^5} | "
        
        for op_idx, operation in enumerate(operations):
            proc_times = operation['proc_times']
            # Format: M0:4, M1:6
            op_str = ", ".join([f"{m}:{t}" for m, t in sorted(proc_times.items())])
            row += f"{op_str:^28} | "
        
        # Fill empty columns
        for _ in range(max_ops - len(operations)):
            row += f"{'-':^28} | "
        
        print(row)
    
    print("="*120)
    
    # Print statistics
    if is_metadata_format:
        total_ops = sum(len(job_data['operations']) for job_data in jobs_data.values())
        avg_ops = total_ops / len(jobs_data)
        
        all_machines_per_op = []
        all_proc_times = []
        for job_data in jobs_data.values():
            for op in job_data['operations']:
                all_machines_per_op.append(len(op['proc_times']))
                all_proc_times.extend(op['proc_times'].values())
    else:
        total_ops = sum(len(ops) for ops in jobs_data.values())
        avg_ops = total_ops / len(jobs_data)
        
        all_machines_per_op = []
        all_proc_times = []
        for job_ops in jobs_data.values():
            for op in job_ops:
                all_machines_per_op.append(len(op['proc_times']))
                all_proc_times.extend(op['proc_times'].values())
    
    print(f"\nDataset Statistics:")
    print(f"  Total Operations: {total_ops}")
    print(f"  Avg Operations/Job: {avg_ops:.2f}")
    print(f"  Avg Machines/Operation: {np.mean(all_machines_per_op):.2f}")
    print(f"  Avg Processing Time: {np.mean(all_proc_times):.2f}")
    print(f"  Processing Time Range: [{min(all_proc_times)}, {max(all_proc_times)}]")
    print("="*120 + "\n")


if __name__ == "__main__":
    print("="*80)
    print(" EXAMPLE 1: REALISTIC DATASET WITH MACHINE HETEROGENEITY")
    print("="*80)
    
    # Generate realistic dataset
    jobs_data, machine_list, machine_metadata = generate_realistic_fjsp_dataset(
        num_initial_jobs=5,
        num_future_jobs=15,
        total_num_machines=6,
        job_type_distribution={'short': 0.5, 'moderate': 0.3, 'long': 0.2},
        machine_speed_variance=0.5,  # Moderate machine differences
        seed=42
    )
    
    # Generate realistic arrival sequence with patterns
    arrival_times, arrival_sequence = generate_realistic_arrival_sequence(
        jobs_data=jobs_data,
        num_initial_jobs=5,
        arrival_rate=0.08,
        pattern_strength=0.5,  # Realistic patterns
        seed=42
    )
    
    # Print comprehensive dataset info
    print_dataset_info(jobs_data, machine_list, arrival_times, machine_metadata)
    print_dataset_table(jobs_data, machine_list, machine_metadata)
    
    # Print arrival sequence
    print("\n" + "="*80)
    print("ARRIVAL SEQUENCE (showing pattern)")
    print("="*80)
    print(f"{'Time':>6} | {'Job':^6} | {'Type':^10} | Pattern Context")
    print("-"*80)
    
    recent_types = []
    for time, job_id, job_type in arrival_sequence:
        # Show pattern context
        if len(recent_types) >= 5:
            context_str = f"Recent: {recent_types[-5:]}"
        elif len(recent_types) > 0:
            context_str = f"Recent: {recent_types}"
        else:
            context_str = "Initial arrivals"
        
        print(f"{time:6.1f} | J{job_id:2d}    | {job_type.upper():^10} | {context_str}")
        recent_types.append(job_type[0].upper())  # S, M, L
    
    print("="*80)
    
    # Analyze the arrival pattern
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    # Count transitions
    transitions = {'S→S': 0, 'S→M': 0, 'S→L': 0, 
                  'M→S': 0, 'M→M': 0, 'M→L': 0,
                  'L→S': 0, 'L→M': 0, 'L→L': 0}
    
    for i in range(len(arrival_sequence) - 1):
        current_type = arrival_sequence[i][2][0].upper()
        next_type = arrival_sequence[i+1][2][0].upper()
        key = f"{current_type}→{next_type}"
        transitions[key] += 1
    
    print("\nJob Type Transitions:")
    print(f"  After SHORT:    {transitions['S→S']} SHORT, {transitions['S→M']} MODERATE, {transitions['S→L']} LONG")
    print(f"  After MODERATE: {transitions['M→S']} SHORT, {transitions['M→M']} MODERATE, {transitions['M→L']} LONG")
    print(f"  After LONG:     {transitions['L→S']} SHORT, {transitions['L→M']} MODERATE, {transitions['L→L']} LONG")
    
    print("\n" + "="*80)
    print("\n\n")
    
    # ===== EXAMPLE 2: TRADITIONAL SIMPLE DATASET =====
    print("="*80)
    print(" EXAMPLE 2: TRADITIONAL SIMPLE DATASET (backward compatibility)")
    print("="*80)
    
    jobs_data_simple, machine_list_simple = generate_fjsp_dataset(
        num_initial_jobs=3,
        num_future_jobs=4,
        total_num_machines=3,
        seed=123
    )
    
    arrival_times_simple = generate_arrival_times(
        num_initial_jobs=3,
        num_future_jobs=4,
        arrival_mode='poisson',
        arrival_rate=0.08,
        seed=123
    )
    
    print_dataset_info(jobs_data_simple, machine_list_simple, arrival_times_simple)
    print("="*80)

