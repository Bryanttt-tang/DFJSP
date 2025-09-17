"""
Reproduce DFJSP instance generation from:
'Deep Reinforcement Learning for Dynamic Flexible Job Shop Scheduling with Random Job Arrival'
(Processes 2022). See Table 2 and eq. (8) for parameterization. 
Citations: the paper describes Eave (mean interarrival), hi ~ U[1,30], tij~U[0,100], fi~U[0.5,2], weights etc.
"""

import numpy as np
import json
import os
from typing import List, Dict, Any
import collections

def generate_instance(m: int = 10,
                      nini: int = 10,
                      nadd: int = 20,
                      Eave: float = 30.0,
                      op_min: int = 1,
                      op_max: int = 30,
                      proc_low: float = 0.0,
                      proc_high: float = 100.0,
                      fi_low: float = 0.5,
                      fi_high: float = 2.0,
                      we_range=(1.0, 1.5),
                      wt_range=(1.0, 2.0),
                      max_eligible: int = 5,
                      seed: int = None) -> Dict[str, Any]:
    """
    Generate a single DFJSP instance.
    - m: number of machines
    - nini: initial jobs present at time 0
    - nadd: number of newly arriving jobs (to be generated with Exp interarrival times)
    - Eave: mean interarrival time (scale parameter for Exp)
    - max_eligible: maximum eligible machines per operation (cap; set to m to allow full flexibility)
    - returns a dict with job list and meta-info
    """
    rng = np.random.default_rng(seed)

    # rate lambda is 1/Eave, but numpy.exponential takes scale=Eave directly
    scale = float(Eave)

    jobs = []

    # 1) initial jobs at time 0
    for i in range(nini):
        Ai = 0.0
        hi = int(rng.integers(low=op_min, high=op_max + 1))
        fi = float(rng.uniform(fi_low, fi_high))
        we = float(rng.uniform(*we_range))
        wt = float(rng.uniform(*wt_range))

        ops = []
        for j in range(hi):
            # choose eligible machines for this operation
            max_choice = min(max_eligible, m)
            eligible_count = int(rng.integers(1, max_choice + 1))
            # sample unique machine indices 0..m-1
            eligible_machines = list(rng.choice(np.arange(m), size=eligible_count, replace=False))
            # per-machine processing times
            t_ijk = {str(k): float(rng.uniform(proc_low, proc_high)) for k in eligible_machines}
            ops.append({"op_index": j, "eligible_machines": eligible_machines, "t_ijk": t_ijk})

        # compute mean processing per operation, then total
        mean_tj = [np.mean(list(op["t_ijk"].values())) for op in ops]
        total_mean = float(np.sum(mean_tj))
        Di = float(Ai + fi * total_mean)  # Eq (8) in paper.

        jobs.append({
            "job_id": len(jobs),
            "arrival": float(Ai),
            "h": hi,
            "fi": fi,
            "we": we,
            "wt": wt,
            "operations": ops,
            "D": Di
        })

    # 2) newly arriving jobs: sample interarrival times exponential(scale=Eave)
    taus = rng.exponential(scale=scale, size=nadd)
    arrivals = np.cumsum(taus)

    for k in range(nadd):
        Ai = float(arrivals[k])
        hi = int(rng.integers(low=op_min, high=op_max + 1))
        fi = float(rng.uniform(fi_low, fi_high))
        we = float(rng.uniform(*we_range))
        wt = float(rng.uniform(*wt_range))

        ops = []
        for j in range(hi):
            max_choice = min(max_eligible, m)
            eligible_count = int(rng.integers(1, max_choice + 1))
            eligible_machines = list(rng.choice(np.arange(m), size=eligible_count, replace=False))
            t_ijk = {str(km): float(rng.uniform(proc_low, proc_high)) for km in eligible_machines}
            ops.append({"op_index": j, "eligible_machines": eligible_machines, "t_ijk": t_ijk})

        mean_tj = [np.mean(list(op["t_ijk"].values())) for op in ops]
        total_mean = float(np.sum(mean_tj))
        Di = float(Ai + fi * total_mean)

        jobs.append({
            "job_id": len(jobs),
            "arrival": Ai,
            "h": hi,
            "fi": fi,
            "we": we,
            "wt": wt,
            "operations": ops,
            "D": Di
        })

    instance = {
        "meta": {
            "m": m,
            "nini": nini,
            "nadd": nadd,
            "Eave": Eave,
            "seed": int(seed) if seed is not None else None
        },
        "jobs": jobs
    }
    return instance


def convert_to_test2_format(instance):
    """
    Convert the generated instance to the format expected by test2.py
    Returns: (jobs_data, machine_list, job_arrival_times)
    """
    jobs = instance["jobs"]
    m = instance["meta"]["m"]
    
    # Create machine list (M0, M1, ..., Mm-1)
    machine_list = [f"M{i}" for i in range(m)]
    
    # Convert jobs to test2.py format
    jobs_data = collections.OrderedDict()
    job_arrival_times = {}
    
    for job in jobs:
        job_id = job["job_id"]
        arrival_time = job["arrival"]
        operations = job["operations"]
        
        # Convert operations to test2.py format
        converted_ops = []
        for op in operations:
            proc_times = {}
            for machine_idx_str, time in op["t_ijk"].items():
                machine_name = f"M{machine_idx_str}"
                proc_times[machine_name] = round(time, 2)  # Round to 2 decimal places
            converted_ops.append({"proc_times": proc_times})
        
        jobs_data[job_id] = converted_ops
        job_arrival_times[job_id] = round(arrival_time, 2)
    
    return jobs_data, machine_list, job_arrival_times


def save_for_test2(jobs_data, machine_list, job_arrival_times, filename):
    """
    Save the data in a format that can be easily imported by test2.py
    """
    with open(filename, 'w') as f:
        f.write("# Generated DFJSP instance for test2.py\n")
        f.write("import collections\n\n")
        
        # Write jobs_data
        f.write("jobs_data = collections.OrderedDict({\n")
        for job_id, operations in jobs_data.items():
            f.write(f"    {job_id}: [\n")
            for op in operations:
                proc_times_str = ", ".join([f"'{m}': {t}" for m, t in op["proc_times"].items()])
                f.write(f"        {{'proc_times': {{{proc_times_str}}}}},\n")
            f.write("    ],\n")
        f.write("})\n\n")
        
        # Write machine_list
        machine_list_str = ", ".join([f"'{m}'" for m in machine_list])
        f.write(f"machine_list = [{machine_list_str}]\n\n")
        
        # Write job_arrival_times
        f.write("job_arrival_times = {\n")
        for job_id, arrival_time in job_arrival_times.items():
            f.write(f"    {job_id}: {arrival_time},\n")
        f.write("}\n")


def generate_txt_instance(output_file: str, **kwargs):
    """
    Generate a single instance and save it as a .txt file for test2.py
    """
    # Generate the instance
    instance = generate_instance(**kwargs)
    
    # Convert to test2.py format
    jobs_data, machine_list, job_arrival_times = convert_to_test2_format(instance)
    
    # Save as .txt file
    save_for_test2(jobs_data, machine_list, job_arrival_times, output_file)
    
    print(f"Generated instance saved to {output_file}")
    print(f"Jobs: {len(jobs_data)}, Machines: {len(machine_list)}")
    print(f"Arrival times range: {min(job_arrival_times.values()):.2f} - {max(job_arrival_times.values()):.2f}")
    
    return jobs_data, machine_list, job_arrival_times


def generate_dataset(outdir: str,
                     configs: List[Dict[str, Any]],
                     per_config_instances: int = 30,
                     seed_base: int = 0):
    """
    Generate a dataset folder with instances for a list of configs.
    Each config is a dict that will be passed to generate_instance().
    Files are saved as both JSON and TXT formats.
    """
    os.makedirs(outdir, exist_ok=True)
    idx = 0
    for cidx, cfg in enumerate(configs):
        for rep in range(per_config_instances):
            seed = seed_base + idx
            inst = generate_instance(seed=seed, **cfg)
            
            # Save JSON format (original)
            json_fname = f"instance_cfg{cidx:02d}_rep{rep:02d}.json"
            with open(os.path.join(outdir, json_fname), "w") as fh:
                json.dump(inst, fh, indent=2)
            
            # Save TXT format for test2.py
            txt_fname = f"instance_cfg{cidx:02d}_rep{rep:02d}.txt"
            jobs_data, machine_list, job_arrival_times = convert_to_test2_format(inst)
            save_for_test2(jobs_data, machine_list, job_arrival_times, 
                          os.path.join(outdir, txt_fname))
            
            idx += 1
    print(f"Saved {idx} instances (JSON + TXT) to {outdir}")


# -------------------------
# Example usage
if __name__ == "__main__":
    # Small example configuration for testing
    cfg_small = {
        "m": 3,           # 3 machines
        "nini": 2,        # 2 initial jobs
        "nadd": 2,        # 2 additional jobs
        "Eave": 10.0,     # Mean interarrival time
        "op_min": 2,      # Min operations per job
        "op_max": 4,      # Max operations per job
        "proc_low": 1.0,  # Min processing time
        "proc_high": 10.0, # Max processing time
        "fi_low": 0.5,
        "fi_high": 2.0,
        "we_range": (1.0, 1.5),
        "wt_range": (1.0, 2.0),
        "max_eligible": 3  # All machines eligible for each operation
    }
    
    # Generate a single instance for testing
    jobs_data, machine_list, job_arrival_times = generate_txt_instance(
        "test_instance.txt", 
        seed=42, 
        **cfg_small
    )
    
    # Also demonstrate the original paper configuration
    cfg_paper = {
        "m": 10,
        "nini": 10,
        "nadd": 20,
        "Eave": 30.0,
        "op_min": 1,
        "op_max": 30,
        "proc_low": 0.0,
        "proc_high": 100.0,
        "fi_low": 0.5,
        "fi_high": 2.0,
        "we_range": (1.0, 1.5),
        "wt_range": (1.0, 2.0),
        "max_eligible": 5
    }
    
    # Generate a larger instance
    generate_txt_instance("paper_instance.txt", seed=12345, **cfg_paper)
    
    print("\nExample of generated test2 data structure:")
    print("jobs_data keys:", list(jobs_data.keys()))
    print("machine_list:", machine_list)
    print("job_arrival_times:", job_arrival_times)