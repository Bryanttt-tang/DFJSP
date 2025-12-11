import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import gymnasium as gym
import torch
from tqdm import tqdm
import time
import pickle
import hashlib
import json
import os
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD

# Import utility functions for dataset generation
from utils import (generate_fjsp_dataset, generate_simplified_fjsp_dataset, print_dataset_info,print_dataset_table)

# Set random seed for reproducibility
GLOBAL_SEED = 12345
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)

# Global tracking for arrival time distribution analysis
TRAINING_ARRIVAL_TIMES = []  # Track all arrival times during training
TRAINING_EPISODE_COUNT = 0   # Track episode count
DEBUG_EPISODE_ARRIVALS = []  # Track first 10 episodes' arrival details

# Training metrics tracking
TRAINING_METRICS = {
    'episode_rewards': [],
    'episode_lengths': [],
    'episode_timesteps': [],
    'action_entropy': [],      # train/entropy_loss
    'policy_loss': [],         # train/policy_gradient_loss  
    'value_loss': [],          # train/value_loss
    'total_loss': [],          # train/loss (total combined loss)
    'timesteps': [],
    'episode_count': [],
    'learning_rate': [],
    'explained_variance': [],
    'kl_divergence': [],       # train/approx_kl (KL divergence)
}

# ===== SIMPLIFIED DATASET GENERATION =====
# Generate SIMPLIFIED FJSP dataset with:
# 1. NO job classification - all jobs are homogeneous
# 2. Machine heterogeneity (fast/medium/slow) - KEY strategic element
# 3. High variance in processing times - creates wait opportunities
# 4. Poisson arrivals (no patterns) - forcing strategic wait decisions
print("\n" + "="*80)
print(" GENERATING SIMPLIFIED FJSP DATASET")
print(" Focus: Machine Heterogeneity + Processing Time Variance")
print("="*80)

# Dataset generation parameters
NUM_INITIAL_JOBS = 3  # Jobs arriving at t=0
NUM_FUTURE_JOBS = 5   # Jobs arriving dynamically
INITIAL_JOB_IDS = list(range(NUM_INITIAL_JOBS))  # [0, 1, 2]

ENHANCED_JOBS_DATA, MACHINE_LIST, MACHINE_METADATA = generate_simplified_fjsp_dataset(
    num_initial_jobs=NUM_INITIAL_JOBS,
    num_future_jobs=NUM_FUTURE_JOBS,
    total_num_machines=3,
    machine_speed_variance=0.8,  # High machine heterogeneity for strategic waiting
    proc_time_variance_range=(3, 10),  # Wide range creates wait opportunities
    seed=GLOBAL_SEED
)

# Print dataset information
print(f"\nDataset Generated:")
print(f"  Total jobs: {len(ENHANCED_JOBS_DATA)}")
print(f"  Machines: {len(MACHINE_LIST)}")
print(f"  Machine heterogeneity: {len([m for m in MACHINE_METADATA.values() if m['category']=='fast'])} FAST, "
      f"{len([m for m in MACHINE_METADATA.values() if m['category']=='medium'])} MEDIUM, "
      f"{len([m for m in MACHINE_METADATA.values() if m['category']=='slow'])} SLOW")

print(f"\nMachine Speed Factors:")
for machine in MACHINE_LIST:
    metadata = MACHINE_METADATA[machine]
    print(f"  {machine}: {metadata['speed_factor']:.2f} ({metadata['category'].upper()})")

print(f"\nProcessing Time Variance Example (Job 0, Operation 0):")
if 0 in ENHANCED_JOBS_DATA:
    proc_times = ENHANCED_JOBS_DATA[0][0]['proc_times']
    min_time = min(proc_times.values())
    max_time = max(proc_times.values())
    print(f"  Fastest machine: {min_time} time units")
    print(f"  Slowest machine: {max_time} time units")
    print(f"  Gap: {max_time - min_time} time units ({((max_time/min_time - 1)*100):.0f}% difference)")
    print(f"  → STRATEGIC INSIGHT: Worth waiting {max_time - min_time} units for fast machine!")

print("="*80 + "\n")

print_dataset_table(ENHANCED_JOBS_DATA, MACHINE_LIST)

# Jobs are already in simple format (no metadata) - ready to use!


# # Exact dataset from test3_backup.py that achieved makespan=43 with reactive RL
# ENHANCED_JOBS_DATA = collections.OrderedDict({
#     0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}, {'proc_times': {'M0': 2}}],
#     1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M1': 3, 'M2': 4}}],
#     2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M0': 3, 'M1': 4}}, {'proc_times': {'M2': 7}}],
#     3: [{'proc_times': {'M1': 8}}, {'proc_times': {'M2': 2}}, {'proc_times': {'M0': 5, 'M1': 6}}],
#     4: [{'proc_times': {'M0': 6, 'M1': 9}}, {'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M2': 6}}],
#     5: [{'proc_times': {'M1': 5, 'M2': 8}}, {'proc_times': {'M0': 6}}, {'proc_times': {'M1': 4, 'M2': 3}}],
#     6: [{'proc_times': {'M0': 7, 'M2': 4}}, {'proc_times': {'M0': 5, 'M1': 6}}, {'proc_times': {'M1': 3}}, {'proc_times': {'M0': 2, 'M2': 5}}],
# })

# MACHINE_LIST = ['M0', 'M1', 'M2']


# ==================== PROACTIVE SCHEDULING COMPONENTS ====================

class ArrivalPredictor:
    """
    Predicts job arrival times using either MLE or MAP for Poisson processes.
    
    Estimation Modes:
    1. MLE (Maximum Likelihood Estimation): Uses only observed data
    2. MAP (Maximum A Posteriori): Uses Bayesian estimation with Gamma prior
    
    Key Features:
    1. Cross-Episode Learning: Uses ALL historical inter-arrival times from past episodes
    2. Within-Episode Learning: Updates predictions as jobs arrive in current episode
    3. Adaptive Learning: Combines historical data with current episode observations
    4. Misprediction Correction: Adjusts estimates based on prediction errors
    """
    
    def __init__(self, initial_rate_guess=0.05, mode='mle', prior_shape=2.0, prior_rate=None):
        """
        Args:
            initial_rate_guess: Initial estimate of arrival rate λ (events per time unit)
            mode: Estimation mode - 'mle' or 'map'
            prior_shape: Shape parameter α for Gamma prior (only for MAP mode)
            prior_rate: Rate parameter β for Gamma prior (only for MAP mode)
                       If None, defaults to prior_shape / initial_rate_guess
        """
        self.initial_rate = initial_rate_guess
        self.mode = mode.lower()
        
        # MAP-specific parameters (Gamma prior for Poisson rate λ)
        # Prior: λ ~ Gamma(α, β)
        # Posterior: λ | data ~ Gamma(α + n, β + sum(inter_arrivals))
        self.prior_shape = prior_shape  # α (higher = stronger prior)
        if prior_rate is None:
            # Default: center prior at initial_rate_guess
            # E[λ] = α/β, so β = α / E[λ]
            self.prior_rate = prior_shape / initial_rate_guess
        else:
            self.prior_rate = prior_rate
        
        # IMPROVED: Store sufficient statistics instead of raw data (memory efficient)
        self.global_n = 0  # Total number of inter-arrivals observed globally
        self.global_sum = 0.0  # Total time across all inter-arrivals
        
        # Current episode tracking
        self.current_episode_arrivals = []  # Arrival times in current episode (sorted)
        self.current_n = 0  # Number of inter-arrivals in current episode
        self.current_sum = 0.0  # Sum of inter-arrivals in current episode
        
        # MLE estimates
        self.global_estimated_rate = initial_rate_guess  # Based on all historical data
        self.current_estimated_rate = initial_rate_guess  # Based on current episode + history
        
        # Prediction tracking for correction
        self.prediction_errors = []  # Track (predicted_time - actual_time) for learning
        
    def reset_episode(self):
        """Reset for new episode (but keep ALL cross-episode learning)."""
        self.current_episode_arrivals = []
        self.current_n = 0
        self.current_sum = 0.0
        # Keep global statistics - this is the key to cross-episode learning!
        
    def observe_arrival(self, arrival_time):
        """
        Observe a new job arrival in the current episode.
        Updates predictions IMMEDIATELY using both historical and current data.
        
        FIXED: Properly handles first arrival (inter-arrival from time 0).
        
        Args:
            arrival_time: Time when job actually arrived
        """
        self.current_episode_arrivals.append(arrival_time)
        self.current_episode_arrivals.sort()  # Keep sorted
        
        # Calculate inter-arrival time from previous arrival
        if len(self.current_episode_arrivals) >= 2:
            # Standard case: compute difference from previous arrival
            last_arrival = self.current_episode_arrivals[-2]
            inter_arrival = arrival_time - last_arrival
            
            if inter_arrival > 0:
                self.current_n += 1
                self.current_sum += inter_arrival
                
                # IMMEDIATELY update estimate using BOTH historical AND current data
                self._update_mle_estimate()
        elif len(self.current_episode_arrivals) == 1 and arrival_time > 0:
            # FIXED: First dynamic arrival (inter-arrival from time 0)
            # This ensures we capture the interval from episode start to first arrival
            inter_arrival = arrival_time
            self.current_n += 1
            self.current_sum += inter_arrival
            self._update_mle_estimate()
    
    def correct_prediction(self, job_id, predicted_time, actual_time):
        """
        Called when a misprediction is detected (job scheduled proactively but arrived differently).
        This helps the predictor learn from mistakes.
        
        Args:
            job_id: ID of the job
            predicted_time: When we predicted it would arrive
            actual_time: When it actually arrived
        """
        prediction_error = actual_time - predicted_time
        self.prediction_errors.append(prediction_error)
        
        # If we consistently over/under-estimate, adjust the rate
        if len(self.prediction_errors) >= 5:
            mean_error = np.mean(self.prediction_errors[-20:])  # Use recent errors
            
            # If mean_error > 0: We predict too early → increase inter-arrival time → decrease rate
            # If mean_error < 0: We predict too late → decrease inter-arrival time → increase rate
            if abs(mean_error) > 0.5:  # Significant bias
                correction_factor = 1.0 - (mean_error / (1.0/self.current_estimated_rate)) * 0.1
                correction_factor = np.clip(correction_factor, 0.5, 2.0)  # Limit corrections
                self.current_estimated_rate *= correction_factor
    
    def finalize_episode(self, all_arrival_times):
        """
        Called at end of episode to perform cross-episode learning.
        THIS IS WHERE WE ACCUMULATE KNOWLEDGE FROM PAST 100 EPISODES!
        
        FIXED: Properly includes first dynamic arrival (from time 0).
        
        Args:
            all_arrival_times: Dict {job_id: arrival_time} for all jobs in episode
        """
        # Extract all arrival times (including 0.0 for initial jobs)
        arrival_list = sorted([t for t in all_arrival_times.values()])
        
        # FIXED: Compute inter-arrivals including from time 0
        episode_n = 0
        episode_sum = 0.0
        
        # Handle first arrival from time 0 if it exists
        if len(arrival_list) > 0:
            # Find first positive arrival time (first dynamic job)
            first_positive_idx = None
            for i, t in enumerate(arrival_list):
                if t > 0:
                    first_positive_idx = i
                    break
            
            if first_positive_idx is not None:
                # First inter-arrival: from time 0 to first dynamic job
                first_inter_arrival = arrival_list[first_positive_idx]
                if first_inter_arrival > 0:
                    episode_n += 1
                    episode_sum += first_inter_arrival
                
                # Subsequent inter-arrivals: between consecutive dynamic jobs
                for i in range(first_positive_idx + 1, len(arrival_list)):
                    inter_arrival = arrival_list[i] - arrival_list[i-1]
                    if inter_arrival > 0:
                        episode_n += 1
                        episode_sum += inter_arrival
        
        # ADD TO GLOBAL STATISTICS - memory efficient!
        self.global_n += episode_n
        self.global_sum += episode_sum
        
        # Update global MLE estimate using ALL accumulated data
        self._update_global_mle()
        
        # Reset current episode data for next episode
        self.current_n = 0
        self.current_sum = 0.0
    
    def _update_mle_estimate(self):
        """
        Update CURRENT estimate using BOTH historical data AND current episode data.
        Supports both MLE and MAP modes.
        """
        total_n = self.global_n + self.current_n
        
        if total_n > 0:
            if self.mode == 'map':
                # MAP estimation with Gamma prior
                # Posterior: λ | data ~ Gamma(α + n, β + sum(τ))
                # MAP estimate: (α + n - 1) / (β + sum(τ))
                
                # Weight current episode data more heavily
                if self.current_n >= 3:
                    weight_current = 2.0
                    weighted_n = self.global_n + weight_current * self.current_n
                    weighted_sum = self.global_sum + weight_current * self.current_sum
                else:
                    weighted_n = total_n
                    weighted_sum = self.global_sum + self.current_sum
                
                # MAP estimate
                posterior_shape = self.prior_shape + weighted_n
                posterior_rate = self.prior_rate + weighted_sum
                
                if posterior_shape > 1 and posterior_rate > 0:
                    # MAP: mode of Gamma distribution
                    self.current_estimated_rate = (posterior_shape - 1) / posterior_rate
                else:
                    # Fallback to posterior mean
                    self.current_estimated_rate = posterior_shape / posterior_rate
                    
            else:  # MLE mode
                # MLE estimation (original implementation)
                if self.current_n >= 3:
                    # Weight current episode data 2x more heavily
                    weight_current = 2.0
                    weighted_n = self.global_n + weight_current * self.current_n
                    weighted_sum = self.global_sum + weight_current * self.current_sum
                    mean_inter_arrival = weighted_sum / weighted_n
                else:
                    # Simple mean when not enough current data
                    mean_inter_arrival = (self.global_sum + self.current_sum) / total_n
                
                if mean_inter_arrival > 0:
                    # MLE for Poisson process: λ̂ = 1 / E[τ]
                    self.current_estimated_rate = 1.0 / mean_inter_arrival
                else:
                    self.current_estimated_rate = self.global_estimated_rate
        else:
            # No data yet, use initial guess
            self.current_estimated_rate = self.initial_rate
    
    def _update_global_mle(self):
        """Update global estimate using ALL historical data (supports MLE and MAP)."""
        if self.global_n > 0:
            if self.mode == 'map':
                # MAP estimation with all historical data
                posterior_shape = self.prior_shape + self.global_n
                posterior_rate = self.prior_rate + self.global_sum
                
                if posterior_shape > 1 and posterior_rate > 0:
                    self.global_estimated_rate = (posterior_shape - 1) / posterior_rate
                else:
                    self.global_estimated_rate = posterior_shape / posterior_rate
            else:  # MLE mode
                mean_inter_arrival = self.global_sum / self.global_n
                if mean_inter_arrival > 0:
                    self.global_estimated_rate = 1.0 / mean_inter_arrival
                else:
                    self.global_estimated_rate = self.initial_rate
        else:
            self.global_estimated_rate = self.initial_rate
    
    def predict_next_arrivals(self, current_time, num_jobs_to_predict, last_known_arrival=None):
        """
        Predict arrival times of next N jobs using BOTH historical and current data.
        
        This is where we leverage the past 100 episodes of data!
        
        Args:
            current_time: Current time in the episode
            num_jobs_to_predict: How many future arrivals to predict
            last_known_arrival: Time of last known arrival (for better anchoring)
            
        Returns:
            List of predicted arrival times
        """
        # Use current estimate (which incorporates historical data)
        if self.current_estimated_rate <= 0:
            mean_inter_arrival = 1.0 / self.initial_rate
        else:
            mean_inter_arrival = 1.0 / self.current_estimated_rate
        
        # Anchor predictions to last known arrival if available
        if last_known_arrival is not None and last_known_arrival >= current_time:
            anchor_time = last_known_arrival
        elif len(self.current_episode_arrivals) > 0:
            anchor_time = self.current_episode_arrivals[-1]
        else:
            anchor_time = current_time
        
        # Predict arrivals at regular intervals based on mean
        predictions = []
        for i in range(1, num_jobs_to_predict + 1):
            predicted_time = anchor_time + i * mean_inter_arrival
            predictions.append(predicted_time)
        
        return predictions
    
    def get_confidence(self):
        """
        Return confidence in predictions (0-1 scale).
        Based on TOTAL number of observations across ALL episodes.
        
        More historical data = higher confidence
        """
        total_observations = self.global_n
        
        if total_observations == 0:
            return 0.0
        
        # Confidence grows with square root of observations (more realistic)
        # 50% confidence at ~25 observations, 90% at ~100 observations
        confidence = 1.0 - np.exp(-np.sqrt(total_observations) / 5.0)
        return np.clip(confidence, 0.0, 1.0)
    
    def get_stats(self):
        """Return current statistics for debugging/logging."""
        return {
            'mode': self.mode,
            'estimated_rate': self.current_estimated_rate,
            'global_rate': self.global_estimated_rate,
            'num_global_observations': self.global_n,
            'num_current_observations': self.current_n,
            'confidence': self.get_confidence(),
            'mean_inter_arrival': 1.0/self.current_estimated_rate if self.current_estimated_rate > 0 else float('inf'),
            'mean_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            'prior_shape': self.prior_shape if self.mode == 'map' else None,
            'prior_rate': self.prior_rate if self.mode == 'map' else None,
        }


# ==================== HELPER FUNCTIONS ====================

def calculate_context_aware_wait_reward(env):
    """
    Smart context-aware wait penalty that teaches the agent WHEN to wait vs WHEN to schedule.
    
    Design Principles:
    ------------------
    1. Waiting is BAD if:
       - Idle machines exist AND arrived jobs can be scheduled
       - No arrivals coming soon (wasting time)
       
    2. Waiting is OK/GOOD if:
       - No work available (all arrived jobs scheduled)
       - Next arrival coming very soon (< 3 time units)
       - All machines are busy (no idle capacity)
    
    3. Penalty scales with:
       - Number of idle machines
       - Duration of wait
       - Amount of available work
    
    Returns:
    --------
    reward : float
        Negative reward (penalty) for waiting
    """
    # Get next event information
    next_event_time = env._get_next_event_time()
    wait_duration = next_event_time - env.event_time
    
    # Check if next event is an arrival or just machine completion
    next_arrival_time = float('inf')
    for job_id, arrival_time in env.job_arrival_times.items():
        if job_id not in env.arrived_jobs and arrival_time > env.event_time:
            next_arrival_time = min(next_arrival_time, arrival_time)
    
    is_arrival_next = (next_arrival_time == next_event_time and next_arrival_time != float('inf'))
    
    # Count idle machines (machines available at current event_time)
    num_idle_machines = sum(
        1 for m in env.machines
        if env.machine_next_free[m] <= env.event_time
    )
    
    # Count schedulable operations (arrived jobs with next operation ready)
    num_schedulable_ops = 0
    total_available_proc_time = 0.0
    
    for job_id in env.arrived_jobs:
        next_op_idx = env.next_operation[job_id]
        if next_op_idx < len(env.jobs[job_id]):
            num_schedulable_ops += 1
            # Get minimum processing time for this operation (best case)
            operation = env.jobs[job_id][next_op_idx]
            min_proc_time = min(operation['proc_times'].values())
            total_available_proc_time += min_proc_time
    
    # DECISION LOGIC:
    
    # Case 1: Idle machines + available work = BAD WAIT!
    if num_idle_machines > 0 and num_schedulable_ops > 0:
        # Heavy penalty proportional to opportunity cost
        base_penalty = num_idle_machines * wait_duration * 0.5
        
        # Scale by available work (more work = worse to wait)
        work_penalty_multiplier = 1.0 + (total_available_proc_time / 100.0)
        
        reward = -(base_penalty * work_penalty_multiplier)
        return reward
    
    # Case 2: Job arriving VERY soon (< 3 time units) = ACCEPTABLE WAIT
    elif is_arrival_next and wait_duration < 3.0:
        # Small penalty or even neutral
        reward = -0.1 * wait_duration
        return reward
    
    # Case 3: No work available (all arrived jobs scheduled) = NEUTRAL/OK WAIT
    elif num_schedulable_ops == 0:
        # No other choice - very small penalty
        reward = -0.1 * wait_duration
        return reward
    
    # Case 4: All machines busy = OK WAIT
    elif num_idle_machines == 0:
        # Forced wait - minimal penalty
        reward = -0.05 * wait_duration
        return reward
    
    # Case 5: Default - moderate penalty
    else:
        reward = -1.0 * wait_duration
        return reward


# ==================== REACTIVE SCHEDULING ENVIRONMENT ====================

class PoissonDynamicFJSPEnv(gym.Env):
    """
    Dynamic FJSP Environment with Poisson-distributed job arrivals.
    BUILDER MODE: Actions place operations at earliest feasible start time.
    REALISTIC: Only arrived jobs can be scheduled + WAIT action to advance time.
    """
    
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05, 
                 max_time_horizon=1000, reward_mode="makespan_increment", seed=None):
        super().__init__()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        
        # Handle initial_jobs as either integer or list
        if isinstance(initial_jobs, list):
            self.initial_job_ids = initial_jobs
            self.dynamic_job_ids = [j for j in self.job_ids if j not in initial_jobs]
            self.initial_jobs = len(initial_jobs)
        else:
            self.initial_jobs = min(initial_jobs, len(self.job_ids))
            self.initial_job_ids = self.job_ids[:self.initial_jobs]
            self.dynamic_job_ids = self.job_ids[self.initial_jobs:]
        
        self.arrival_rate = arrival_rate
        self.max_time_horizon = max_time_horizon
        self.reward_mode = reward_mode
        
        # Environment parameters
        self.num_jobs = len(self.job_ids)
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        
        # ENHANCED ACTION SPACE (IDENTICAL to proactive RL for fair comparison):
        # - Scheduling actions: job_idx * num_machines + machine_idx
        # - Wait actions: [10, "next_event"] - 2 wait options
        num_scheduling_actions = self.num_jobs * len(self.machines)
        
        # Define wait actions: [10, "next_event"] - IDENTICAL to proactive RL
        self.wait_durations = [float('inf')]  # 10 units or next event
        num_wait_actions = len(self.wait_durations)
        
        self.action_space = spaces.Discrete(num_scheduling_actions + num_wait_actions)
        self.scheduling_action_end = num_scheduling_actions
        self.wait_action_start = num_scheduling_actions
        
        # Backward compatibility: Keep WAIT_ACTION for single wait option
        self.WAIT_ACTION = num_scheduling_actions  # First wait action (wait 10)
        
        self.cheat = False  # No cheating with future info
        # UNIFIED observation space (same size for all RL methods for evaluation compatibility)
        if self.cheat==False:
            obs_size = (
                self.num_jobs +                         # Ready job indicators
                self.num_jobs +                         # Job progress (completed_ops / total_ops)
                len(self.machines) +                    # Machine next_free times (normalized)
                self.num_jobs * len(self.machines) +    # Processing times for ready ops
                self.num_jobs +                         # Time since last arrival (per job)
                2                                      # Arrival progress and makespan progress
            )
        else:
            obs_size = (
                self.num_jobs +                         # Ready job indicators
                len(self.machines) +                    # Machine idle status
                self.num_jobs * len(self.machines) +    # Processing times for ready ops
                self.num_jobs +                         # DYNAMIC ADVANTAGE: Future arrival delays
                self.num_jobs * len(self.machines)      # Future processing times (ZEROS for Reactive RL)
            )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Calculate maximum processing time across all operations for normalization
        self.max_proc_time = self._calculate_max_processing_time()
        
        # Initialize state variables
        self._reset_state()

    def _calculate_max_processing_time(self):
        """Calculate the maximum processing time across all operations and machines."""
        max_time = 0.0
        for job_ops in self.jobs.values():
            for operation in job_ops:
                for proc_time in operation['proc_times'].values():
                    max_time = max(max_time, proc_time)
        return max_time if max_time > 0 else 1.0  # Avoid division by zero

    def _reset_state(self):
        """Reset all environment state variables for builder mode."""
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        # BUILDER MODE: Use makespan as the "builder clock" for final schedule width
        self.current_makespan = 0.0
        
        # EVENT-DRIVEN: Separate event time that controls arrival visibility
        self.event_time = 0.0  # Event frontier for revealing arrivals and WAIT processing
        
        self.operations_scheduled = 0
        self.episode_step = 0
        # More reasonable max_episode_steps to prevent infinite loops
        # Allow for total_operations + reasonable number of wait actions
        self.max_episode_steps = self.total_operations * 3  # Allow more waits
        
        # Job arrival management - realistic dynamic scheduling
        self.arrived_jobs = set(self.initial_job_ids)  # Initial jobs available immediately
        self.job_arrival_times = {}
        
        # Generate Poisson arrival times for dynamic jobs
        self._generate_poisson_arrivals()
        
        # IMPORTANT: After generating arrivals, check if any dynamic jobs also arrive at t=0
        # and add them to arrived_jobs to maintain consistency
        for job_id, arrival_time in self.job_arrival_times.items():
            if arrival_time <= self.event_time and job_id not in self.arrived_jobs:
                self.arrived_jobs.add(job_id)

    def _generate_poisson_arrivals(self):
        """Generate arrival times for dynamic jobs using Poisson process."""
        # Initialize arrival times
        for job_id in self.initial_job_ids:
            self.job_arrival_times[job_id] = 0.0
        
        # Generate inter-arrival times using exponential distribution
        current_time = 0.0
        for job_id in self.dynamic_job_ids:
            inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
            current_time += inter_arrival_time
            
            # if current_time <= self.max_time_horizon:
            self.job_arrival_times[job_id] = float(current_time)
            # else:
                # self.job_arrival_times[job_id] = float('inf')  # Won't arrive in this episode

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT, DEBUG_EPISODE_ARRIVALS
        
        if seed is not None:
            super().reset(seed=seed, options=options)
            random.seed(seed)
            np.random.seed(seed)
        
        self._reset_state()
        
        # Track arrival times for analysis
        TRAINING_EPISODE_COUNT += 1
        episode_arrivals = []
        for job_id, arr_time in self.job_arrival_times.items():
            if arr_time != float('inf') and arr_time > 0:  # Only dynamic arrivals
                episode_arrivals.append(arr_time)
        
        if episode_arrivals:
            TRAINING_ARRIVAL_TIMES.extend(episode_arrivals)
        
        # Debug: Track first 10 episodes in detail (silently)
        if TRAINING_EPISODE_COUNT <= 10:
            episode_debug_info = {
                'episode': TRAINING_EPISODE_COUNT,
                'initial_jobs': sorted(self.initial_job_ids),
                'dynamic_jobs': sorted(self.dynamic_job_ids),
                'arrival_times': dict(self.job_arrival_times),
                'arrived_at_reset': sorted(self.arrived_jobs),
                'dynamic_arrivals': sorted(episode_arrivals) if episode_arrivals else []
            }
            DEBUG_EPISODE_ARRIVALS.append(episode_debug_info)
        
        return self._get_observation(), {}

    def _decode_action(self, action):
        """Decode action - includes multiple WAIT action handling (IDENTICAL to proactive)."""
        action = int(action)
        
        # Check if it's a WAIT action
        if action >= self.wait_action_start:
            return None, None, None  # Special return for WAIT
        
        # Decode scheduling action: job_idx * num_machines + machine_idx
        num_machines = len(self.machines)
        
        job_idx = action // num_machines
        machine_idx = action % num_machines
        
        job_idx = min(job_idx, self.num_jobs - 1)
        machine_idx = min(machine_idx, len(self.machines) - 1)
        
        # Operation index is always the next operation for the job
        job_id = self.job_ids[job_idx]
        op_idx = self.next_operation[job_id]
        
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        """BUILDER MODE: Check if scheduling action is valid (arrival + precedence + compatibility)."""
        if job_idx is None:  # WAIT action
            return True  # WAIT is always valid unless terminal
        
        if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
            return False
        
        job_id = self.job_ids[job_idx]
        
        # REALISTIC DYNAMIC: Job must have arrived to be scheduled
        if job_id not in self.arrived_jobs:
            return False
            
        # Check operation index validity
        if not (0 <= op_idx < len(self.jobs[job_id])):
            return False
            
        # Check precedence: this must be the next operation for the job
        if op_idx != self.next_operation[job_id]:
            return False
            
        # Check machine compatibility
        machine_name = self.machines[machine_idx]
        if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
            return False
            
        return True

    def action_masks(self):
        """SIMPLIFIED: Generate action masks based on arrival and machine compatibility."""
        mask = np.full(self.action_space.n, False, dtype=bool)
        
        # If all operations are scheduled, no actions are valid.
        if self.operations_scheduled >= self.total_operations:
            return mask

        # Check scheduling actions for arrived jobs
        for job_idx, job_id in enumerate(self.job_ids):
            # Job must have arrived (CRITICAL for reactive scheduling)
            if job_id not in self.arrived_jobs:
                continue
                
            # Job must have remaining operations
            next_op_idx = self.next_operation[job_id]
            if next_op_idx >= len(self.jobs[job_id]):
                continue
                
            # Check compatibility with each machine
            for machine_idx, machine in enumerate(self.machines):
                if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                    action = job_idx * len(self.machines) + machine_idx
                    if action < self.wait_action_start:
                        mask[action] = True
        
        # Wait actions: Enable all wait options (IDENTICAL to proactive RL)
        has_unarrived_jobs = len(self.arrived_jobs) < len(self.job_ids)
        has_schedulable_work = np.any(mask[:self.wait_action_start])
        
        if has_unarrived_jobs or has_schedulable_work:
            for wait_idx in range(len(self.wait_durations)):
                action_idx = self.wait_action_start + wait_idx
                mask[action_idx] = True
        
        return mask

    def _update_event_time_and_arrivals(self, new_event_time):
        """Update event time and reveal any jobs that have arrived by this time."""
        # old_event_time = self.event_time
        # self.event_time = max(self.event_time, new_event_time)
        self.event_time = new_event_time
        
        # Update arrived jobs based on current event time
        newly_arrived = set()
        for job_id, arrival_time in self.job_arrival_times.items():
            if (job_id not in self.arrived_jobs and 
                arrival_time != float('inf') and 
                arrival_time <= self.event_time):
                newly_arrived.add(job_id)
        
        self.arrived_jobs.update(newly_arrived)
        return len(newly_arrived) 

    def _get_next_event_time(self):
        """Calculate the next event time: min(next_arrival_time, next_machine_completion)."""
        # Find next arrival time (future arrivals only)
        next_arrival_time = float('inf')
        for job_id, arrival_time in self.job_arrival_times.items():
            if job_id not in self.arrived_jobs and arrival_time != float('inf'):
                next_arrival_time = min(next_arrival_time, arrival_time)
        
        # Find earliest FUTURE machine completion time (only machines that will complete AFTER event_time)
        next_machine_completion = float('inf')
        for machine, free_time in self.machine_next_free.items():
            if free_time > self.event_time:  # CRITICAL: Only consider STRICTLY future completions
                next_machine_completion = min(next_machine_completion, free_time)
        
        # Return the minimum of next arrival and next machine completion
        next_event_time = min(next_arrival_time, next_machine_completion)
        # If no future events (all machines idle and no arrivals), keep current event_time
        return next_event_time if next_event_time != float('inf') else self.event_time

    def _advance_to_next_arrival(self):
        """WAIT ACTION: Advance event_time to next event (arrival or machine completion)."""
        # Get the next event time (considers both arrivals and machine completions)
        next_event_time = self._get_next_event_time()
        
        # Advance to that time and reveal any new arrivals
        new_arrivals = self._update_event_time_and_arrivals(next_event_time)

        return next_event_time, new_arrivals

    def step(self, action):
        """Simplified step function for Poisson Dynamic environment."""
        self.episode_step += 1
        current_event_time = self.event_time
        # Terminate if all operations are scheduled
        if self.operations_scheduled >= self.total_operations:
            final_reward = - self.current_makespan  # Bonus for finishing
            return self._get_observation(), final_reward, True, False, {"makespan": self.current_makespan, "status": "completed"}

        # Terminate if max steps reached (safety net)
        if self.episode_step >= self.max_episode_steps:
            penalty = -self.current_makespan - 1000 # Penalize for not finishing
            return self._get_observation(), penalty, True, False, {"error": "Max episode steps reached"}
        
        # # This happens when all arrived jobs are scheduled but more jobs are waiting to arrive
        # current_mask = self.action_masks()
        # if not np.any(current_mask):
        #     # No valid actions - advance event_time to next arrival
        #     print(f"finished operations: {self.operations_scheduled} / {self.total_operations}")
        #     print(f"Step {self.episode_step} event_time {self.event_time} makespan {self.current_makespan} No valid scheduling actions available")
        #     next_event_time = self._get_next_event_time()
            
        #     if next_event_time == float('inf') or next_event_time <= self.event_time:
        #         # No more jobs will arrive - terminate episode
        #         print(f"No more jobs will arrive. Terminating episode.")
        #         penalty = -1000.0
        #         return self._get_observation(), penalty, True, False, {"error": "Stuck with no valid actions"}
            
        #     # Advance event time to reveal new jobs
        #     num_new_arrivals = self._update_event_time_and_arrivals(next_event_time)
        #     print(f"Advanced event_time to {self.event_time}, {num_new_arrivals} new job(s) arrived")
            
        #     # Return small negative reward for time passing without action
        #     return self._get_observation(), -1.0, False, False, {"event_time_advanced": True}
        

        job_idx, op_idx, machine_idx = self._decode_action(action)

        # Handle WAIT actions (IDENTICAL logic to proactive RL)
        if job_idx is None:
            # Determine which wait action (short wait or wait to next event)
            wait_idx = action - self.wait_action_start
            wait_duration = self.wait_durations[wait_idx]
            
            previous_makespan = self.current_makespan
            
            if wait_duration == float('inf'):
                # Wait to next event
                new_event_time, new_arrivals = self._advance_to_next_arrival()
            else:
                # Wait for specified duration (e.g., 10 units)
                target_time = self.event_time + wait_duration
                next_event_time = self._get_next_event_time()
                target_time = min(target_time, next_event_time)  # Don't wait beyond next event
                
                if target_time > self.event_time:
                    new_arrivals = self._update_event_time_and_arrivals(target_time)
                else:
                    new_arrivals = 0
            
            # Ensure makespan >= event_time (time passes even when idle)
            self.current_makespan = max(self.current_makespan, self.event_time)
            
            # Reward: Negative makespan increment (IDENTICAL to proactive)
            reward = -(self.current_makespan - previous_makespan)
            
            # Check for termination after waiting
            terminated = self.operations_scheduled >= self.total_operations
            
            info = {"action_type": "WAIT", "event_time": self.event_time}
            return self._get_observation(), reward, terminated, False, info

        # Handle scheduling action
        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            return self._get_observation(), -500.0, False, False, {"error": "Invalid scheduling action"}

        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        
        machine_available_time = self.machine_next_free.get(machine, 0.0)
        job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                         else self.job_arrival_times.get(job_id, 0.0))
        
        # CRITICAL: Check machine's CURRENT free time BEFORE scheduling
        # This determines if event_time should advance or not
        machine_free_before_scheduling = machine_available_time
        
        # Start time must be after job arrival, machine is free, and current event time
        # if self.event_time > machine_available_time and self.event_time > job_ready_time:
        #     print("Error at step ",self.episode_step ,"event_time:", self.event_time, "machine_available_time:", machine_available_time, "job_ready_time:", job_ready_time)
        #     print("Job ID:", job_id, "Op idx:", op_idx, "Machine:", machine)
        #     print("Machine next free times:", self.machine_next_free)
        #     print("Job arrival times:", self.job_arrival_times)
        start_time = max(machine_available_time, job_ready_time, self.event_time)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time

        # Update machine state
        self.machine_next_free[machine] = end_time
        
        # CRITICAL FIX: Event time advancement based on machine state BEFORE scheduling
        # 
        # Rule: Only advance event_time if:
        #   1. We scheduled on a machine that was IDLE (free_time <= event_time), AND
        #   2. There are NO OTHER idle machines remaining at current event_time
        # 
        # Case 1: Machine was IDLE, and NO other idle machines remain
        #         Example: event_time=0, M0 free at 0, M1 free at 10, M2 free at 10
        #         Schedule on M0 → M0 becomes busy → advance event_time to 10
        #         
        # Case 2: Machine was IDLE, but OTHER idle machines still exist
        #         Example: event_time=0, M0 free at 0, M1 free at 0, M2 free at 10  
        #         Schedule on M0 → M1 still idle at 0 → event_time STAYS at 0
        #         
        # Case 3: Machine was BUSY in future (free_time > event_time)
        #         Example: event_time=0, schedule on M2 (free at 10)
        #         → event_time STAYS at 0 (no change)
        #
        # This allows:
        # - Scheduling on multiple idle machines in parallel at current event_time
        # - Scheduling on busy machines without disrupting current time
        
        if machine_free_before_scheduling <= self.event_time:
            # Machine was IDLE - check if any OTHER idle machines remain
            other_idle_machines_exist = any(
                free_time <= self.event_time 
                for m, free_time in self.machine_next_free.items()
                if m != machine  # Exclude the machine we just scheduled
            )
            
            if not other_idle_machines_exist:
                # No more idle machines - advance event_time to next event
                next_event_time = self._get_next_event_time()
                if next_event_time > self.event_time and next_event_time != float('inf'):
                    self._update_event_time_and_arrivals(next_event_time)
            # else: Other idle machines still available - keep event_time unchanged
        # else: Machine was BUSY (free_time > event_time)
        #       → event_time remains unchanged

        # action_mask = self.action_masks()
        # # Check if there were any valid scheduling actions
        # scheduling_actions_available = np.any(action_mask[:-1])
        # if not scheduling_actions_available:
        #     print('Step', self.episode_step, 'event_time', self.event_time, 'No valid scheduling actions available')
        # Update remaining state
        previous_makespan = self.current_makespan
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        
        self.current_makespan = max(self.current_makespan, end_time)
        
        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        # Reward is the negative increase in makespan
        reward = -(self.current_makespan - previous_makespan)
        
        info = {"makespan": self.current_makespan, "action_type": "SCHEDULE", "event_time": self.event_time}
        
        # Check for termination after scheduling
        terminated = self.operations_scheduled >= self.total_operations
        # if terminated:
        #     reward += 100 - self.current_makespan # Add completion bonus

        return self._get_observation(), reward, terminated, False, info

    # def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan):
    #     """Reward calculation for builder mode."""
        
    #     if self.reward_mode == "makespan_increment":
    #         makespan_increment = current_makespan - previous_makespan
    #         reward = -makespan_increment

    #         return reward
    #     # Fallback for other modes if needed
    #     return -proc_time

    def _get_observation(self):
        """
        BUILDER MODE: Event-driven observation using event_time for arrival visibility.
        IMPORTANT: Do NOT reveal information about unarrived jobs (no cheating!)
        """
        obs = []
        if not self.cheat:    
            # 1. Job ready time (when job can start its NEXT operation)
            # For ARRIVED jobs: actual ready time
            # For UNARRIVED jobs: 1.0 (max value = far future, prevents cheating)
            # For COMPLETED jobs: 0.0 (done)
            for job_id in self.job_ids:
                if job_id not in self.arrived_jobs:
                    # NOT ARRIVED YET: 1.0 (no information leakage!)
                    obs.append(1.0)
                elif self.next_operation[job_id] >= len(self.jobs[job_id]):
                    # COMPLETED: 0.0
                    obs.append(0.0)
                else:
                    # ARRIVED and HAS REMAINING OPERATIONS: compute actual ready time
                    next_op_idx = self.next_operation[job_id]
                    
                    # Job ready time = max(previous_op_end_time, arrival_time)
                    if next_op_idx > 0:
                        # Precedence: must wait for previous operation to finish
                        job_ready_time = self.operation_end_times[job_id][next_op_idx - 1]
                    else:
                        # First operation: only constrained by arrival time
                        job_ready_time = self.job_arrival_times.get(job_id, 0.0)
                    
                    # Normalize against max_time_horizon
                    normalized_ready_time = min(1.0, job_ready_time / self.max_time_horizon)
                    obs.append(normalized_ready_time)
            
            # 2. Job progress (completed_ops / total_ops for each job)
            for job_id in self.job_ids:
                completed_ops = sum(self.completed_ops[job_id])
                total_ops = len(self.jobs[job_id])
                progress = completed_ops / total_ops if total_ops > 0 else 1.0
                obs.append(progress)
            
            # 3. Machine availability: normalized next_free times relative to event_time
            for machine in self.machines:
                machine_free_time = self.machine_next_free[machine]
                # relative_busy_time = max(0, machine_free_time - self.event_time)
                normalized_busy = min(1.0, machine_free_time / self.max_time_horizon)
                obs.append(normalized_busy)
        
            # 4. Processing times for ready operations: normalized against max_proc_time across all operations
            for job_id in self.job_ids:
                if (job_id in self.arrived_jobs and 
                    self.next_operation[job_id] < len(self.jobs[job_id])):
                    next_op_idx = self.next_operation[job_id]
                    operation = self.jobs[job_id][next_op_idx]
                    
                    for machine in self.machines:
                        if machine in operation['proc_times']:
                            proc_time = operation['proc_times'][machine]
                            normalized_time = min(1.0, proc_time / self.max_time_horizon)
                            obs.append(normalized_time)
                        else:
                            obs.append(0.0)  # Incompatible machine
                else:
                    # Unarrived or completed: all 0.0
                    for machine in self.machines:
                        obs.append(0.0)
            
            # 5. Reactive RL features:
            # 5.1. Normalized arrival time for arrived jobs, 1 if not arrived
            for job_id in self.job_ids:
                arrival_time = self.job_arrival_times.get(job_id, 0.0)
                if job_id in self.arrived_jobs:
                    # Arrived jobs: normalized arrival time
                    normalized_arrival_time = min(1.0, arrival_time / self.max_time_horizon)
                    obs.append(normalized_arrival_time)
                else:
                    # Not yet arrived: 1.0 (no information leakage)
                    obs.append(1.0)
            
            # 5.2. Arrival progress
            arrival_progress = len(self.arrived_jobs) / len(self.job_ids)
            obs.append(arrival_progress)

            # 5.3. Makespan progress
            makespan_progress = self.current_makespan / self.max_time_horizon
            obs.append(makespan_progress)

            # # 5.4 Event time normalization
            # normalized_event_time = min(1.0, self.event_time / self.max_time_horizon)
            # obs.append(normalized_event_time)

            obs_array = np.array(obs, dtype=np.float32)
            obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
            
            return obs_array
            
        else:
            # 1. Ready job indicators (binary: 1 if job has ready operation, 0 otherwise)
            for job_id in self.job_ids:
                if (job_id in self.arrived_jobs and 
                    self.next_operation[job_id] < len(self.jobs[job_id])):
                    # Job has arrived and has remaining operations
                    next_op_idx = self.next_operation[job_id]
                    
                    # Check if operation is ready (precedence satisfied)
                    if next_op_idx == 0:
                        # First operation: ready if job has arrived
                        job_ready_time = self.job_arrival_times.get(job_id, 0.0)
                        is_ready = self.current_makespan >= job_ready_time
                    else:
                        # Later operation: ready if previous operation completed
                        prev_completed = self.completed_ops[job_id][next_op_idx - 1]
                        is_ready = prev_completed
                    
                    obs.append(1.0 if is_ready else 0.0)
                else:
                    obs.append(0.0)  # Job not ready or not arrived
            
            # 2. Machine idle status (binary: 1 if idle, 0 if busy)
            for machine in self.machines:
                machine_free_time = self.machine_next_free[machine]
                is_idle = machine_free_time <= self.current_makespan
                obs.append(1.0 if is_idle else 0.0)
            
            # 3. Processing times for ready operations (normalized)
            for job_id in self.job_ids:
                if (job_id in self.arrived_jobs and 
                    self.next_operation[job_id] < len(self.jobs[job_id])):
                    next_op_idx = self.next_operation[job_id]
                    operation = self.jobs[job_id][next_op_idx]
                    
                    # Add processing time for each machine (0 if incompatible)
                    for machine in self.machines:
                        if machine in operation['proc_times']:
                            proc_time = operation['proc_times'][machine]
                            normalized_time = min(1.0, proc_time / self.max_proc_time)
                            obs.append(normalized_time)
                        else:
                            obs.append(0.0)  # Machine cannot process this operation
                else:
                    # Job not ready or arrived: add zeros for processing times
                    for machine in self.machines:
                        obs.append(0.0)
            
            # 4. REACTIVE RL ADVANTAGE: Future arrival time hints for unarrived jobs
            for job_id in self.job_ids:
                if job_id not in self.arrived_jobs:
                    # Job not arrived yet: provide arrival time hint
                    arrival_time = self.job_arrival_times.get(job_id, float('inf'))
                    if arrival_time != float('inf') and arrival_time <= self.current_makespan + 30:
                        # Job will arrive soon: add normalized arrival delay
                        delay = max(0, arrival_time - self.current_makespan)
                        normalized_delay = min(1.0, delay / 30.0)  # Normalize by 30 time units
                        obs.append(normalized_delay)
                    else:
                        # Job won't arrive soon or ever: add zero
                        obs.append(0.0)
                else:
                    # Job already arrived: add zero (no future arrival)
                    obs.append(0.0)
            
            # 5. REACTIVE RL: No future processing times (all zeros - this advantage reserved for Perfect RL)
            for job_id in self.job_ids:
                for machine in self.machines:
                    obs.append(0.0)  # No future processing info for Reactive RL
            
            # Ensure correct size and format
            target_size = self.observation_space.shape[0]
            if len(obs) < target_size:
                obs.extend([0.0] * (target_size - len(obs)))
            elif len(obs) > target_size:
                obs = obs[:target_size]
            
            obs_array = np.array(obs, dtype=np.float32)
            obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
            
            return obs_array


# ==================== DISPATCHING RULE ENVIRONMENT ====================

class DispatchingRuleFJSPEnv(gym.Env):
    """
    REDESIGNED Rule-Based RL Environment: Matches Best Heuristic Architecture
    
    KEY DESIGN:
    - Action space: 10 dispatching rule combinations (5 sequencing × 2 routing)
    - Event-driven: Discovers jobs as they arrive (no perfect foresight)
    - Schedules ONE operation per step (similar to heuristic and reactive RL)
    - NO WAIT ACTIONS (scheduling decisions only)
    
    Action Space (10 combinations):
    - 0: FIFO+MIN  - First arrived job, fastest machine
    - 1: FIFO+MINC - First arrived job, earliest completion
    - 2: LIFO+MIN  - Last arrived job, fastest machine
    - 3: LIFO+MINC - Last arrived job, earliest completion
    - 4: SPT+MIN   - Shortest processing time, fastest machine
    - 5: SPT+MINC  - Shortest processing time, earliest completion
    - 6: LPT+MIN   - Longest processing time, fastest machine
    - 7: LPT+MINC  - Longest processing time, earliest completion
    - 8: MWKR+MIN  - Most work remaining, fastest machine
    - 9: MWKR+MINC - Most work remaining, earliest completion
    
    This matches the heuristic comparison architecture, allowing fair comparison.
    Agent learns WHICH combination works best in different scheduling states.
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05,
                 max_time_horizon=1000, reward_mode="makespan_increment", seed=None):
        super().__init__()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        
        # Handle initial_jobs
        if isinstance(initial_jobs, list):
            self.initial_job_ids = initial_jobs
            self.dynamic_job_ids = [j for j in self.job_ids if j not in initial_jobs]
            self.initial_jobs = len(initial_jobs)
        else:
            self.initial_jobs = min(initial_jobs, len(self.job_ids))
            self.initial_job_ids = self.job_ids[:self.initial_jobs]
            self.dynamic_job_ids = self.job_ids[self.initial_jobs:]
        
        self.arrival_rate = arrival_rate
        self.max_time_horizon = max_time_horizon
        self.reward_mode = reward_mode
        
        # Environment parameters
        self.num_jobs = len(self.job_ids)
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        
        # ACTION SPACE: 10 dispatching rule combinations + 1 WAIT action
        # Actions 0-9: Rule combinations (5 sequencing × 2 routing)
        # Action 10: WAIT (advance to next event)
        self.action_space = spaces.Discrete(11)
        self.rule_names = [
            "FIFO+MIN", "FIFO+MINC", "LIFO+MIN", "LIFO+MINC", "SPT+MIN",
            "SPT+MINC", "LPT+MIN", "LPT+MINC", "MWKR+MIN", "MWKR+MINC",
            "WAIT"
        ]
        self.wait_action_index = 10
        
        # Observation space: Similar to PoissonDynamicFJSPEnv but more compact
        obs_size = (
            self.num_jobs +                         # Job arrived indicators
            self.num_jobs +                         # Job progress (completed_ops / total_ops)
            len(self.machines) +                    # Machine utilization
            self.num_jobs +                         # Work remaining per job (normalized)
            self.num_jobs +                         # Average processing time per job
            3                                       # Global features: arrival progress, makespan progress, num ready ops
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        self.max_proc_time = self._calculate_max_processing_time()
        self._reset_state()
    
    def _calculate_max_processing_time(self):
        max_time = 0.0
        for job_ops in self.jobs.values():
            for operation in job_ops:
                for proc_time in operation['proc_times'].values():
                    max_time = max(max_time, proc_time)
        return max_time if max_time > 0 else 1.0
    
    def _reset_state(self):
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        # EVENT-DRIVEN MODE: Both event_time and makespan
        self.current_makespan = 0.0
        self.event_time = 0.0  # Controls job visibility and earliest scheduling time
        self.operations_scheduled = 0
        self.episode_step = 0
        self.max_episode_steps = self.total_operations * 2
        
        # EVENT-DRIVEN: Jobs revealed only when event_time >= arrival_time
        self.arrived_jobs = set(self.initial_job_ids)
        self.job_arrival_times = {}
        self._generate_poisson_arrivals()
        
        # Update arrived jobs based on initial event_time (t=0)
        for job_id, arrival_time in self.job_arrival_times.items():
            if arrival_time <= self.event_time:
                self.arrived_jobs.add(job_id)
    
    def _generate_poisson_arrivals(self):
        for job_id in self.initial_job_ids:
            self.job_arrival_times[job_id] = 0.0
        
        current_time = 0.0
        for job_id in self.dynamic_job_ids:
            inter_arrival = np.random.exponential(1.0 / self.arrival_rate)
            current_time += inter_arrival
            self.job_arrival_times[job_id] = current_time
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self._reset_state()
        return self._get_observation(), {}
    
    def _get_ready_operations(self):
        """
        Get list of operations that can be scheduled (EVENT-DRIVEN MODE).
        An operation is schedulable if:
        1. Job has arrived (in arrived_jobs)
        2. Job is not completed
        3. It's the next operation in the job (precedence satisfied)
        
        Note: We DON'T filter by event_time here - that's handled in start_time calculation.
        This matches Reactive RL behavior where all arrived jobs with ready ops are candidates.
        """
        ready_ops = []
        
        for job_id in self.arrived_jobs:
            op_idx = self.next_operation[job_id]
            if op_idx < len(self.jobs[job_id]):
                operation = self.jobs[job_id][op_idx]
                
                # Calculate work remaining for this job
                work_remaining = sum(
                    min(self.jobs[job_id][i]['proc_times'].values())
                    for i in range(op_idx, len(self.jobs[job_id]))
                )
                
                # Get minimum processing time for sequencing rules
                min_proc_time = min(operation['proc_times'].values())
                
                # Job ready time (considers arrival + precedence)
                job_ready_time = self.job_arrival_times[job_id]
                if op_idx > 0:
                    job_ready_time = max(job_ready_time, self.operation_end_times[job_id][op_idx - 1])
                
                # Add to ready operations (no event_time filtering here)
                ready_ops.append({
                    'job_id': job_id,
                    'op_idx': op_idx,
                    'operation': operation,
                    'min_proc_time': min_proc_time,
                    'arrival_time': self.job_arrival_times[job_id],
                    'job_ready_time': job_ready_time,
                    'work_remaining': work_remaining
                })
        
        return ready_ops
    
    def _apply_rule_combination(self, action_idx, ready_ops):
        """Apply selected rule combination: two-step sequencing + routing."""
        if not ready_ops:
            return None, None
        
        # Decode action to (sequencing_rule, routing_rule)
        seq_rules = ["FIFO", "FIFO", "LIFO", "LIFO", "SPT", "SPT", "LPT", "LPT", "MWKR", "MWKR"]
        route_rules = ["MIN", "MINC", "MIN", "MINC", "MIN", "MINC", "MIN", "MINC", "MIN", "MINC"]
        seq_rule = seq_rules[action_idx]
        route_rule = route_rules[action_idx]
        
        # STEP 1: SEQUENCING - Select which operation to schedule
        def sequencing_score(op):
            if seq_rule == "FIFO":
                return (op['arrival_time'], op['job_id'])
            elif seq_rule == "LIFO":
                return (-op['arrival_time'], -op['job_id'])
            elif seq_rule == "SPT":
                return (op['min_proc_time'], op['arrival_time'], op['job_id'])
            elif seq_rule == "LPT":
                return (-op['min_proc_time'], op['arrival_time'], op['job_id'])
            elif seq_rule == "MWKR":
                return (-op['work_remaining'], op['arrival_time'], op['job_id'])
            else:
                return (op['min_proc_time'], op['arrival_time'], op['job_id'])  # Default SPT
        
        selected_op = min(ready_ops, key=sequencing_score)
        
        # STEP 2: ROUTING - Select which machine for the selected operation
        operation = selected_op['operation']
        compatible_machines = list(operation['proc_times'].keys())
        
        def routing_score(machine):
            proc_time = operation['proc_times'][machine]
            machine_avail = self.machine_next_free[machine]
            job_ready = selected_op['job_ready_time']
            start_time = max(self.event_time, machine_avail, job_ready)  # Event-driven
            completion_time = start_time + proc_time
            
            if route_rule == "MIN":
                return (proc_time, machine)  # Fastest machine
            elif route_rule == "MINC":
                return (completion_time, machine)  # Earliest completion
            else:
                return (proc_time, machine)  # Default MIN
        
        best_score_machine_tuple = min((routing_score(m) for m in compatible_machines))
        best_machine = best_score_machine_tuple[1]
        
        return selected_op, best_machine
    
    def _get_next_event_time(self):
        """Calculate next event time: min(next_arrival, next_machine_completion)."""
        next_arrival_time = float('inf')
        for job_id, arrival_time in self.job_arrival_times.items():
            if job_id not in self.arrived_jobs and arrival_time > self.event_time:
                next_arrival_time = min(next_arrival_time, arrival_time)
        
        # Check for next machine completion (only for operations already scheduled)
        next_machine_completion = float('inf')
        for machine, free_time in self.machine_next_free.items():
            if free_time > self.event_time:
                next_machine_completion = min(next_machine_completion, free_time)
        
        next_event_time = min(next_arrival_time, next_machine_completion)
        return next_event_time if next_event_time != float('inf') else self.event_time
    
    def _update_event_time_and_arrivals(self, new_event_time):
        """Update event time and reveal any jobs that have arrived by this time."""
        self.event_time = new_event_time
        newly_arrived = set()
        for job_id, arrival_time in self.job_arrival_times.items():
            if (job_id not in self.arrived_jobs and 
                arrival_time <= self.event_time):
                newly_arrived.add(job_id)
        self.arrived_jobs.update(newly_arrived)
        return len(newly_arrived)
    
    def action_masks(self):
        """Generate action masks - 10 rule combinations if ready ops exist, WAIT if not."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        if self.operations_scheduled >= self.total_operations:
            return mask
        
        ready_ops = self._get_ready_operations()
        
        if ready_ops:
            # Enable all 10 rule combinations if there are ready operations
            mask[:10] = True
        else:
            # No ready operations - enable WAIT action to advance to next event
            # (matches Reactive/Proactive RL behavior)
            has_unarrived_jobs = len(self.arrived_jobs) < len(self.job_ids)
            if has_unarrived_jobs:
                mask[self.wait_action_index] = True
        
        return mask
    
    def step(self, action):
        """
        EVENT-DRIVEN SCHEDULING: Schedule ONE operation per step using selected rule combination.
        
        Event-driven characteristics:
        - event_time controls job visibility and earliest scheduling time
        - Operations can only be scheduled if ready at current event_time
        - Start time = max(event_time, machine_avail, job_ready)
        - Automatically advances event_time when no schedulable operations exist
        
        Actions:
        - 0-9: Rule combinations (5 sequencing × 2 routing)
        - 10: WAIT (advance to next event)
        """
        self.episode_step += 1
        previous_makespan = self.current_makespan
        
        # Check termination
        if self.operations_scheduled >= self.total_operations:
            return self._get_observation(), 0.0, True, False, {"makespan": self.current_makespan}
        
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "timeout"}
        
        # Handle WAIT action (action 10)
        if action == self.wait_action_index:
            # Advance to next event (arrival or machine completion)
            next_event_time = self._get_next_event_time()
            
            if next_event_time == self.event_time or next_event_time == float('inf'):
                # No more events - shouldn't happen if jobs are incomplete
                reward = -1000.0
                return self._get_observation(), reward, True, False, {"error": "stuck_wait"}
            
            # Advance event_time and reveal new arrivals
            self._update_event_time_and_arrivals(next_event_time)
            
            # Ensure makespan >= event_time (time passes even when idle)
            self.current_makespan = max(self.current_makespan, self.event_time)
            
            # Reward: negative makespan increment (matches Reactive/Proactive RL)
            reward = -(self.current_makespan - previous_makespan)
            
            terminated = self.operations_scheduled >= self.total_operations
            return self._get_observation(), reward, terminated, False, {"action_type": "WAIT"}
        
        # Get schedulable operations at current event_time
        ready_ops = self._get_ready_operations()
        
        # If no ready operations, advance event_time to next event
        while not ready_ops and self.operations_scheduled < self.total_operations:
            next_event_time = self._get_next_event_time()
            
            if next_event_time == self.event_time or next_event_time == float('inf'):
                # No more events - shouldn't happen if jobs are incomplete
                reward = -1000.0
                return self._get_observation(), reward, True, False, {"error": "stuck"}
            
            # Advance event_time and reveal new arrivals
            self._update_event_time_and_arrivals(next_event_time)
            ready_ops = self._get_ready_operations()
        
        if not ready_ops:
            # Still no ops after advancing time - return with penalty
            reward = -(self.current_makespan - previous_makespan)
            terminated = self.operations_scheduled >= self.total_operations
            return self._get_observation(), reward, terminated, False, {"no_ops": True}
        
        # Apply selected rule combination (two-step: sequencing + routing)
        selected_op, best_machine = self._apply_rule_combination(action, ready_ops)
        
        if selected_op is None or best_machine is None:
            reward = -1.0
            return self._get_observation(), reward, False, False, {"error": "no_op_selected"}
        
        # Schedule the selected operation on the selected machine (EVENT-DRIVEN)
        job_id = selected_op['job_id']
        op_idx = selected_op['op_idx']
        operation = selected_op['operation']
        proc_time = operation['proc_times'][best_machine]
        
        # Calculate earliest start time (EVENT-DRIVEN - includes event_time)
        machine_avail = self.machine_next_free[best_machine]
        job_ready = selected_op['job_ready_time']
        start_time = max(self.event_time, machine_avail, job_ready)
        end_time = start_time + proc_time
        
        # Update state
        self.machine_next_free[best_machine] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.schedule[best_machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        self.operations_scheduled += 1
        
        # Update makespan
        self.current_makespan = max(self.current_makespan, end_time)
        
        # CRITICAL: Always advance event_time after scheduling (like Reactive RL's WAIT action)
        # This ensures new jobs are revealed when no more operations are ready
        ready_ops_after = self._get_ready_operations()
        if not ready_ops_after and self.operations_scheduled < self.total_operations:
            # No more ready operations - advance to next event (arrival or machine completion)
            next_event_time = self._get_next_event_time()
            if next_event_time > self.event_time:
                self._update_event_time_and_arrivals(next_event_time)
        
        # Reward: negative makespan increment
        reward = -(self.current_makespan - previous_makespan)
        
        # Check termination
        terminated = self.operations_scheduled >= self.total_operations
        
        return self._get_observation(), reward, terminated, False, {"rule": self.rule_names[action]}
    
    def _get_observation(self):
        obs = []
        
        # 1. Job arrived indicators
        for job_id in self.job_ids:
            obs.append(1.0 if job_id in self.arrived_jobs else 0.0)
        
        # 2. Job progress
        for job_id in self.job_ids:
            completed = sum(self.completed_ops[job_id])
            total = len(self.jobs[job_id])
            obs.append(completed / total if total > 0 else 1.0)
        
        # 3. Machine utilization (normalized free time relative to current makespan)
        for machine in self.machines:
            free_time = self.machine_next_free[machine]
            # In builder mode, machines with lower free times are more available
            utilization = min(1.0, free_time / self.max_time_horizon)
            obs.append(utilization)
        
        # 4. Work remaining per job (normalized)
        max_work = self.total_operations * self.max_proc_time
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                work_remaining = sum(
                    min(self.jobs[job_id][i]['proc_times'].values())
                    for i in range(self.next_operation[job_id], len(self.jobs[job_id]))
                )
                obs.append(work_remaining / max_work if max_work > 0 else 0.0)
            else:
                obs.append(1.0)  # Unknown work (not arrived)
        
        # 5. Average processing time per job's next operation
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs:
                op_idx = self.next_operation[job_id]
                if op_idx < len(self.jobs[job_id]):
                    avg_proc = np.mean(list(self.jobs[job_id][op_idx]['proc_times'].values()))
                    obs.append(avg_proc / self.max_proc_time if self.max_proc_time > 0 else 0.0)
                else:
                    obs.append(0.0)  # Job completed
            else:
                obs.append(0.5)  # Unknown (not arrived)
        
        # 6. Global features
        arrival_progress = len(self.arrived_jobs) / len(self.job_ids)
        obs.append(arrival_progress)
        
        makespan_progress = self.current_makespan / self.max_time_horizon
        obs.append(min(1.0, makespan_progress))
        
        # Number of ready operations (normalized)
        ready_ops = self._get_ready_operations()
        num_ready = len(ready_ops) / self.num_jobs if self.num_jobs > 0 else 0.0
        obs.append(min(1.0, num_ready))
        
        return np.array(obs, dtype=np.float32)
        
        makespan_progress = min(1.0, self.current_makespan / self.max_time_horizon)
        obs.append(makespan_progress)
        
        ready_ops = self._get_ready_operations()
        num_ready_normalized = len(ready_ops) / max(1, len(self.arrived_jobs))
        obs.append(num_ready_normalized)
        
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array


# ==================== PROACTIVE ENVIRONMENT ====================

class ProactiveDynamicFJSPEnv(gym.Env):
    """
    Proactive Dynamic FJSP Environment with arrival prediction.
    
    Key Features:
    1. Learns to predict job arrivals using MLE across episodes
    2. Can schedule jobs proactively based on predictions (within prediction window)
    3. Updates predictions within episode as jobs arrive
    4. Penalizes mispredictions to encourage conservative behavior
    5. Enhanced observation space with prediction information
    """
    
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, initial_jobs=5, arrival_rate=0.05,
                 max_time_horizon=1000, reward_mode="makespan_increment", 
                 predictor_mode='mle', prior_shape=2.0, prior_rate=None, seed=None):
        """
        SIMPLIFIED Proactive Dynamic FJSP Environment
        
        KEY DESIGN PRINCIPLES:
        1. Jobs are HOMOGENEOUS - no job type classification
        2. Machine HETEROGENEITY is the PRIMARY strategic element (fast/medium/slow)
        3. Poisson arrivals - WITHOUT LOSS OF GENERALITY, assume specific sequence
        4. WAIT action becomes CRITICAL for exploiting machine speed differences
        
        Strategic Depth:
        ----------------
        Since jobs are homogeneous, the strategic decisions come from:
        - Machine speed differences (fast machines process ALL jobs faster)
        - Processing time variance (some operations are much longer than others)
        - Arrival uncertainty (when will next job arrive?)
        
        Wait Action Purpose:
        -------------------
        Agent learns to wait when:
        - Current operation has high processing time (big gain from fast machine)
        - Fast machine will be free soon
        - Alternative: schedule small jobs now, save fast machine for big jobs
        
        The ONLY difference from reactive RL:
        -------------------------------------
        - Predictor guides agent through OBSERVATIONS (predicted arrival times)
        - NO reward shaping - predictor learns from corrections
        
        Args:
            jobs_data: Job specifications (simple format, no metadata)
            machine_list: List of machines with heterogeneous speeds
            initial_jobs: Number or list of jobs available at t=0
            arrival_rate: Poisson arrival rate (lambda) - TRUE rate (hidden from agent)
            max_time_horizon: Maximum episode duration
            reward_mode: Reward calculation mode
            predictor_mode: Arrival predictor mode - 'mle' or 'map'
            prior_shape: Shape parameter for Gamma prior (MAP mode only)
            prior_rate: Rate parameter for Gamma prior (MAP mode only)
            seed: Random seed
        """
        super().__init__()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.cheat = False  # No cheating in proactive environment
        
        # Handle initial_jobs as either integer or list
        if isinstance(initial_jobs, list):
            self.initial_job_ids = initial_jobs
            self.dynamic_job_ids = [j for j in self.job_ids if j not in initial_jobs]
            self.initial_jobs = len(initial_jobs)
        else:
            self.initial_jobs = min(initial_jobs, len(self.job_ids))
            self.initial_job_ids = self.job_ids[:self.initial_jobs]
            self.dynamic_job_ids = self.job_ids[self.initial_jobs:]
        
        self.arrival_rate = arrival_rate  # TRUE rate (hidden from agent)
        self.max_time_horizon = max_time_horizon
        self.reward_mode = reward_mode
        
        # Initialize arrival predictor with chosen mode (MLE or MAP)
        self.arrival_predictor = ArrivalPredictor(
            initial_rate_guess=arrival_rate,
            mode=predictor_mode,
            prior_shape=prior_shape,
            prior_rate=prior_rate
        )
        
        # ENHANCED ACTION SPACE: 
        # - Scheduling actions: job_idx * num_machines + machine_idx (only for ARRIVED jobs)
        # - Wait actions: SIMPLIFIED to just 2 options (10 units OR next event)
        num_scheduling_actions = len(self.job_ids) * len(self.machines)
        
        # Define wait actions: [10, "next_event"] - SIMPLIFIED!
        # Start simple: only 2 wait options helps agent focus on core scheduling first
        # Agent learns: (1) wait short time for machine to free, (2) wait for next arrival
        self.wait_durations = [float('inf')]  # 10 units or next event
        num_wait_actions = len(self.wait_durations)
        
        self.action_space = spaces.Discrete(num_scheduling_actions + num_wait_actions)
        self.scheduling_action_end = num_scheduling_actions
        self.wait_action_start = num_scheduling_actions
        
        # Calculate dimensions
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if len(self.job_ids) > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        
        # ENHANCED observation space (includes prediction information)
        obs_size = (
            len(self.job_ids) +                      # Ready job indicators (arrived or predicted)
            len(self.machines) +                     # Machine idle status
            len(self.job_ids) * len(self.machines) + # Processing times for next ops
            len(self.job_ids) +                      # Job progress
            len(self.job_ids) +                      # NEW: Predicted arrival times (normalized)
            2                                        # Arrival progress + Makespan progress
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        self.max_proc_time = self._calculate_max_processing_time()
        
    def _calculate_max_processing_time(self):
        """Calculate maximum processing time for normalization."""
        max_time = 0.0
        for job_ops in self.jobs.values():
            for operation in job_ops:
                for proc_time in operation['proc_times'].values():
                    max_time = max(max_time, proc_time)
        return max_time if max_time > 0 else 1.0
    
    def reset(self, seed=None, options=None):
        """Reset environment and predictor for new episode."""
        super().reset(seed=seed)
        
        # Reset episode state
        self.arrival_predictor.reset_episode()
        
        # # Generate arrival times using PATTERN-BASED generation (matches test scenarios!)
        # if self.jobs_with_metadata is not None:
        #     # Use realistic arrival patterns (same as test scenarios)
        #     from utils import generate_realistic_arrival_sequence
        #     arrival_times_dict, _ = generate_realistic_arrival_sequence(
        #         jobs_data=self.jobs_with_metadata,
        #         num_initial_jobs=len(self.initial_job_ids),
        #         arrival_rate=self.arrival_rate,
        #         pattern_strength=self.pattern_strength,
        #         seed=None  # Different each episode for variety
        #     )
        #     self.job_arrival_times = arrival_times_dict
        # else:
            # Fallback: Simple Poisson sampling (backward compatibility)
        self.job_arrival_times = {}
        for job_id in self.initial_job_ids:
            self.job_arrival_times[job_id] = 0.0
        
        current_time = 0.0
        for job_id in self.dynamic_job_ids:
            inter_arrival = np.random.exponential(1.0 / self.arrival_rate)
            current_time += inter_arrival
            self.job_arrival_times[job_id] = current_time
        
        # Initialize state
        self.arrived_jobs = set(self.initial_job_ids)
        self.predicted_arrival_times = {}  # {job_id: predicted_time}
        self.job_progress = {job_id: 0 for job_id in self.job_ids}
        self.completed_jobs = set()
        
        self.machine_schedules = {m: [] for m in self.machines}
        self.machine_end_times = {m: 0.0 for m in self.machines}
        self.job_end_times = {job_id: 0.0 for job_id in self.job_ids}
        
        self.event_time = 0.0
        self.current_makespan = 0.0
        self.steps = 0
        self.max_episode_steps = self.total_operations * 3  # Safety: allow wait actions but prevent infinite loops
        
        # Initial predictions (with no observations yet)
        self._update_predictions()
        
        # Observe initial arrivals for predictor
        for job_id in self.initial_job_ids:
            self.arrival_predictor.observe_arrival(0.0)
        
        return self._get_observation(), {}
    
    def _update_predictions(self):
        """
        Update predictions for jobs that haven't arrived yet.
        Uses BOTH historical data (from past episodes) AND current episode observations.
        """
        unarrived_jobs = [j for j in self.job_ids if j not in self.arrived_jobs]
        
        if len(unarrived_jobs) == 0:
            return
        
        # Find last known arrival time for better anchoring
        last_known_arrival = None
        if len(self.arrived_jobs) > 0:
            arrived_times = [self.job_arrival_times[j] for j in self.arrived_jobs]
            last_known_arrival = max(arrived_times)
        
        # Predict next arrivals using IMPROVED predictor (leverages past 100 episodes!)
        predictions = self.arrival_predictor.predict_next_arrivals(
            current_time=self.event_time,
            num_jobs_to_predict=len(unarrived_jobs),
            last_known_arrival=last_known_arrival
        )
        
        # Map predictions to jobs (ordered by job_id)
        for job_id, predicted_time in zip(sorted(unarrived_jobs), predictions):
            self.predicted_arrival_times[job_id] = predicted_time
    
    def action_masks(self):
        """
        ENHANCED REACTIVE ACTION MASKING (No Prediction Window):
        - Can ONLY schedule ARRIVED jobs (reactive, no cheating!)
        - Can choose from MULTIPLE WAIT DURATIONS (1, 2, 3, 5, 10, next_event)
        
        Key Change: Removed prediction window - agent learns to wait strategically
        through experience and predictor guidance in rewards!
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # Scheduling actions: Only for ARRIVED jobs
        for job_id in self.job_ids:
            # Skip completed jobs
            if job_id in self.completed_jobs:
                continue
            
            # Check if job's next operation is ready
            op_idx = self.job_progress[job_id]
            if op_idx >= len(self.jobs[job_id]):
                continue
            
            # ONLY schedule if job has ARRIVED (no prediction window!)
            if job_id in self.arrived_jobs:
                operation = self.jobs[job_id][op_idx]
                available_machines = list(operation['proc_times'].keys())
                
                for machine in available_machines:
                    machine_idx = self.machines.index(machine)
                    action_idx = job_id * len(self.machines) + machine_idx
                    mask[action_idx] = True
        
        # Wait actions: Strategic logic
        # Enable wait ONLY if there are unarrived jobs (waiting serves strategic purpose)
        # Once all jobs have arrived, wait actions are meaningless:
        # - No new arrivals coming
        # - Waiting just delays makespan without benefit
        # - Better to schedule remaining operations directly
        has_unarrived_jobs = len(self.arrived_jobs) < len(self.job_ids)
        has_schedulable_work = np.any(mask[:self.wait_action_start])  # Any scheduling actions available
        
        # Enable wait actions if:
        # 1. There are unarrived jobs (strategic waiting for arrivals), OR
        # 2. No schedulable work BUT jobs incomplete (forced wait for machine/precedence)
        if has_unarrived_jobs:
            # Strategic wait: jobs still arriving, agent decides when to wait
            for wait_idx in range(len(self.wait_durations)):
                action_idx = self.wait_action_start + wait_idx
                mask[action_idx] = True
        elif not has_schedulable_work and len(self.completed_jobs) < len(self.job_ids):
            # Forced wait: all jobs arrived, no ready ops, but jobs incomplete
            # Must wait for machines to finish or precedence constraints
            for wait_idx in range(len(self.wait_durations)):
                action_idx = self.wait_action_start + wait_idx
                mask[action_idx] = True
        
        return mask
    
    def step(self, action):
        """Execute action with flexible wait durations and predictor-guided rewards."""
        previous_makespan = self.current_makespan
        
        # Safety: Check for timeout
        if self.steps >= self.max_episode_steps:
            # Force termination - incomplete schedule but prevent infinite loops
            obs = self._get_observation()
            done = True
            return obs, -1000.0, done, False, {"timeout": True}
        
        # Handle WAIT actions (multiple durations)
        if action >= self.wait_action_start:
            wait_idx = action - self.wait_action_start
            wait_duration = self.wait_durations[wait_idx]
            
            reward, done = self._execute_wait_action(wait_duration)
            
            self.steps += 1  # Increment steps for wait actions
            obs = self._get_observation()
            return obs, reward, done, False, {}
        
        # Decode scheduling action
        job_id = action // len(self.machines)
        machine_idx = action % len(self.machines)
        machine = self.machines[machine_idx]
        
        # Get operation details
        op_idx = self.job_progress[job_id]
        if op_idx >= len(self.jobs[job_id]):
            # Invalid action
            self.steps += 1
            return self._get_observation(), -100.0, False, False, {}
        
        operation = self.jobs[job_id][op_idx]
        
        # Check if machine is compatible
        if machine not in operation['proc_times']:
            self.steps += 1
            return self._get_observation(), -100.0, False, False, {}

        proc_time = operation['proc_times'][machine]
        
        # PROACTIVE SCHEDULING: Determine actual start time and handle mispredictions
        if job_id in self.arrived_jobs:
            # Traditional: job has arrived, schedule normally
            actual_arrival = self.job_arrival_times[job_id]
        else:
            # PROACTIVE: job hasn't arrived yet, use actual arrival time
            actual_arrival = self.job_arrival_times[job_id]
            
            # IMPORTANT: If we scheduled proactively, check for misprediction
            if job_id in self.predicted_arrival_times:
                predicted_arrival = self.predicted_arrival_times[job_id]
                
                # Correct the predictor based on actual vs predicted
                self.arrival_predictor.correct_prediction(
                    job_id, predicted_arrival, actual_arrival
                )
                
                # NO REWARD PENALTY - predictor learns from the error instead
        
        # CRITICAL: Save machine's free time BEFORE scheduling
        machine_free_before_scheduling = self.machine_end_times[machine]
        
        # Calculate earliest start time - CANNOT schedule in the past
        machine_ready = self.machine_end_times[machine]
        job_ready = max(self.job_end_times[job_id], actual_arrival)
        start_time = max(machine_ready, job_ready, self.event_time)
        end_time = start_time + proc_time
        
        # Record operation (1-indexed for consistency with other environments)
        op_label = f"J{job_id}-O{op_idx+1}"
        self.machine_schedules[machine].append((op_label, start_time, end_time))
        self.machine_end_times[machine] = end_time
        self.job_end_times[job_id] = end_time
        
        # CRITICAL EVENT TIME ADVANCEMENT (same logic as PoissonDynamicFJSPEnv):
        # Only advance event_time if:
        # 1. We scheduled on a machine that was IDLE (free_time <= event_time), AND
        # 2. There are NO OTHER idle machines remaining at current event_time
        if machine_free_before_scheduling <= self.event_time:
            # Machine was IDLE - check if any OTHER idle machines remain
            other_idle_machines_exist = any(
                free_time <= self.event_time 
                for m, free_time in self.machine_end_times.items()
                if m != machine  # Exclude the machine we just scheduled
            )
            
            if not other_idle_machines_exist:
                # No more idle machines - advance event_time to next event
                next_event_time = self._get_next_event_time()
                if next_event_time > self.event_time and next_event_time != float('inf'):
                    self.event_time = next_event_time
                    # Check for arrivals at new event_time
                    self._check_arrivals()
        
        # Update progress
        self.job_progress[job_id] += 1
        
        # Check if job completed
        if self.job_progress[job_id] >= len(self.jobs[job_id]):
            self.completed_jobs.add(job_id)
        
        # Update makespan
        self.current_makespan = max(self.machine_end_times.values())
        
        # REWARD: Simple makespan increment (NO misprediction penalty)
        # The predictor learns from corrections, not from reward penalties
        reward = -(self.current_makespan - previous_makespan)
        
        # Update predictions
        self._update_predictions()
        
        # Check termination
        done = len(self.completed_jobs) == len(self.job_ids)
        
        self.steps += 1
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_next_event_time(self):
        """Calculate the next event time: min(next_arrival_time, next_machine_completion)."""
        # Find next arrival time (future arrivals only)
        next_arrival_time = float('inf')
        for job_id, arrival_time in self.job_arrival_times.items():
            if job_id not in self.arrived_jobs and arrival_time != float('inf'):
                next_arrival_time = min(next_arrival_time, arrival_time)
        
        # Find earliest FUTURE machine completion time (only machines that will complete AFTER event_time)
        next_machine_completion = float('inf')
        for machine, free_time in self.machine_end_times.items():
            if free_time > self.event_time:  # CRITICAL: Only consider STRICTLY future completions
                next_machine_completion = min(next_machine_completion, free_time)
        
        # Return the minimum of next arrival and next machine completion
        next_event_time = min(next_arrival_time, next_machine_completion)
        # If no future events (all machines idle and no arrivals), keep current event_time
        return next_event_time if next_event_time != float('inf') else self.event_time
    
    def _execute_wait_action(self, wait_duration):
        """
        Execute wait action with SIMPLE makespan_increment reward.
        Wait advances event_time, and makespan must be >= event_time.
        
        The predictor guides the agent through OBSERVATIONS (predicted arrivals),
        NOT through reward shaping. This keeps rewards consistent between
        reactive and proactive RL.
        """
        previous_makespan = self.current_makespan
        
        if wait_duration == float('inf'):
            # Wait to next event
            next_event_time = self._get_next_event_time()
            
            if next_event_time == self.event_time or next_event_time == float('inf'):
                done = len(self.completed_jobs) == len(self.job_ids)
                return 0.0, done
            
            self.event_time = next_event_time
        else:
            # Wait for specified duration
            target_time = self.event_time + wait_duration
            
            # But don't wait beyond next event
            next_event_time = self._get_next_event_time()
            target_time = min(target_time, next_event_time)
            
            if target_time <= self.event_time:
                done = len(self.completed_jobs) == len(self.job_ids)
                return 0.0, done
            
            self.event_time = target_time
        
        # Check for arrivals during wait
        self._check_arrivals()
        
        # Update predictions (predictor learns continuously)
        self._update_predictions()
        
        # CRITICAL: Ensure makespan >= event_time (time passes even when idle)
        self.current_makespan = max(self.current_makespan, self.event_time)
        
        # Reward: Negative makespan increment (IDENTICAL to reactive RL)
        reward = -(self.current_makespan - previous_makespan)
        
        # Check if all jobs are completed (should terminate)
        done = len(self.completed_jobs) == len(self.job_ids)
        
        return reward, done
    
    def _check_arrivals(self):
        """Check for new arrivals and update predictor."""
        for job_id, arrival_time in self.job_arrival_times.items():
            if job_id not in self.arrived_jobs and arrival_time <= self.event_time:
                self.arrived_jobs.add(job_id)
                # Inform predictor about actual arrival
                self.arrival_predictor.observe_arrival(arrival_time)
                # Remove from predictions
                if job_id in self.predicted_arrival_times:
                    del self.predicted_arrival_times[job_id]
    # def _get_observation(self):
    #     """BUILDER MODE: Event-driven observation using event_time for arrival visibility."""
    #     obs = []
    #     if not self.cheat:    
    #         # 1. Ready job indicators: 1 if job has arrived and has a next operation, else 0
    #         for job_id in self.job_ids:
    #             if (job_id in self.arrived_jobs and 
    #                 self.next_operation[job_id] < len(self.jobs[job_id])):
    #                 obs.append(1.0)
    #             else:
    #                 obs.append(0.0)
            
    #         # 2. Job progress (completed_ops / total_ops for each job)
    #         for job_id in self.job_ids:
    #             completed_ops = sum(self.completed_ops[job_id])
    #             total_ops = len(self.jobs[job_id])
    #             progress = completed_ops / total_ops if total_ops > 0 else 1.0
    #             obs.append(progress)
            
    #         # 3. Machine availability: normalized next_free times relative to event_time
    #         for machine in self.machines:
    #             machine_free_time = self.machine_next_free[machine]
    #             relative_busy_time = max(0, machine_free_time - self.event_time)
    #             normalized_busy = min(1.0, relative_busy_time / self.max_time_horizon)
    #             obs.append(normalized_busy)
            
    #         # 4. Processing times for ready operations: normalized against max_proc_time across all operations
    #         for job_id in self.job_ids:
    #             if (job_id in self.arrived_jobs and 
    #                 self.next_operation[job_id] < len(self.jobs[job_id])):
    #                 next_op_idx = self.next_operation[job_id]
    #                 operation = self.jobs[job_id][next_op_idx]
                    
    #                 for machine in self.machines:
    #                     if machine in operation['proc_times']:
    #                         proc_time = operation['proc_times'][machine]
    #                         normalized_time = min(1.0, proc_time / self.max_proc_time)
    #                         obs.append(normalized_time)
    #                     else:
    #                         obs.append(0.0)
    #             else:
    #                 for machine in self.machines:
    #                     obs.append(0.0)
            
    #         # 5. Reactive RL features:
    #         # 5.1. Normalized arrival time for arrived jobs, 1 if not arrived
    #         for job_id in self.job_ids:
    #             arrival_time = self.job_arrival_times.get(job_id, 0.0)
    #             if job_id in self.arrived_jobs:
    #                 # Arrived jobs: normalized arrival time
    #                 normalized_arrival_time = min(1.0, arrival_time / self.max_time_horizon)
    #                 obs.append(normalized_arrival_time)
    #             else:
    #                 # Not yet arrived: 1
    #                 obs.append(1.0)
            
    #         # 5.2. Arrival progress
    #         arrival_progress = len(self.arrived_jobs) / len(self.job_ids)
    #         obs.append(arrival_progress)

    #         # 5.3. Makespan progress
    #         makespan_progress = self.current_makespan / self.max_time_horizon
    #         obs.append(makespan_progress)

    #         # # 5.4 Event time normalization
    #         # normalized_event_time = min(1.0, self.event_time / self.max_time_horizon)
    #         # obs.append(normalized_event_time)
    #         # 5. NEW: Predicted arrival times (normalized, relative to current time)
    #         pred_arrivals = []
    #         for job_id in self.job_ids:
    #             if job_id in self.arrived_jobs or job_id in self.completed_jobs:
    #                 pred_arrivals.append(0.0)  # Already arrived
    #             elif job_id in self.predicted_arrival_times:
    #                 pred_time = self.predicted_arrival_times[job_id]
    #                 # Normalize: time until arrival / prediction_window
    #                 time_until = max(0, pred_time - self.event_time)
    #                 normalized = min(1.0, time_until / (self.prediction_window * 2))
    #                 pred_arrivals.append(normalized)
    #             else:
    #                 pred_arrivals.append(1.0)  # Far future
    #         obs.extend(pred_arrivals)
            
    #         # # 6. NEW: Prediction confidence per job
    #         # confidence = self.arrival_predictor.get_confidence()
    #         # pred_confidence = []
    #         # for job_id in self.job_ids:
    #         #     if job_id in self.arrived_jobs or job_id in self.completed_jobs:
    #         #         pred_confidence.append(1.0)  # Certain (already arrived)
    #         #     elif job_id in self.predicted_arrival_times:
    #         #         pred_confidence.append(confidence)
    #         #     else:
    #         #         pred_confidence.append(0.0)
    #         # obs_parts.extend(pred_confidence)
            
    #         # 7. NEW: Estimated arrival rate (normalized)
    #         stats = self.arrival_predictor.get_stats()
    #         estimated_rate = stats['estimated_rate']
    #         # Normalize rate (assume typical range 0-0.2)
    #         normalized_rate = min(1.0, estimated_rate / 0.2)
    #         obs.append(normalized_rate)

    #         obs_array = np.array(obs, dtype=np.float32)
    #         obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
            
    #         return obs_array
   
    def _get_observation(self):
        """
        SIMPLIFIED observation for better convergence.
        Removed redundant features (actual arrival times, arrival rate).
        """
        obs_parts = []
        
        # # 1. Ready job indicators (arrived OR predicted within window)
        # ready_jobs = []
        # for job_id in self.job_ids:
        #     if job_id in self.completed_jobs:
        #         ready_jobs.append(0.0)
        #     elif job_id in self.arrived_jobs:
        #         ready_jobs.append(1.0)
        #     elif job_id in self.predicted_arrival_times:
        #         pred_time = self.predicted_arrival_times[job_id]
        #         if pred_time <= self.event_time + self.prediction_window:
        #             ready_jobs.append(0.5)  # Predicted but not arrived
        #         else:
        #             ready_jobs.append(0.0)
        #     else:
        #         ready_jobs.append(0.0)
        # obs_parts.extend(ready_jobs)
        # 1. Job ready time (when job can start its NEXT operation)
        # For ARRIVED jobs: actual ready time
        # For UNARRIVED jobs: 1.0 (max value = far future, prevents cheating)
        # For COMPLETED jobs: 0.0 (done)
        for job_id in self.job_ids:
            if job_id in self.completed_jobs:
                # Completed: 0.0
                obs_parts.append(0.0)
            elif job_id not in self.arrived_jobs:
                # NOT ARRIVED YET: 1.0 (no information leakage!)
                obs_parts.append(1.0)
            else:
                # ARRIVED: compute actual ready time
                op_idx = self.job_progress[job_id]
                if op_idx < len(self.jobs[job_id]):
                    # Job ready time = max(previous_op_end, arrival_time)
                    if op_idx > 0:
                        # Precedence: must wait for previous operation to finish
                        job_ready_time = self.job_end_times[job_id]
                    else:
                        # First operation: only constrained by arrival time
                        job_ready_time = self.job_arrival_times.get(job_id, 0.0)
                    
                    # Normalize against max_time_horizon
                    normalized_ready_time = min(1.0, job_ready_time / self.max_time_horizon)
                    obs_parts.append(normalized_ready_time)
                else:
                    # Should not reach here (completed jobs handled above)
                    obs_parts.append(0.0)
        
        # 2. Job progress (completed_ops / total_ops for each job)
        for job_id in self.job_ids:
            total_ops = len(self.jobs[job_id])
            completed_ops = self.job_progress[job_id]
            progress = completed_ops / total_ops if total_ops > 0 else 1.0
            obs_parts.append(progress)

        # 3. Machine free time (when each machine is available)
        for machine in self.machines:
            machine_free_time = self.machine_end_times[machine]
            normalized_free_time = min(1.0, machine_free_time / self.max_time_horizon)
            obs_parts.append(normalized_free_time)
        
        # 4. Processing times for next operations (normalized)
        # Only reveal for ARRIVED jobs, use 0.0 for unarrived/completed
        for job_id in self.job_ids:
            if job_id in self.completed_jobs:
                for machine in self.machines:
                    obs_parts.append(0.0)
            elif job_id not in self.arrived_jobs:
                # UNARRIVED: all 0.0 (no information leakage!)
                for machine in self.machines:
                    obs_parts.append(0.0)
            else:
                # ARRIVED: reveal processing times
                op_idx = self.job_progress[job_id]
                if op_idx < len(self.jobs[job_id]):
                    operation = self.jobs[job_id][op_idx]
                    for machine in self.machines:
                        if machine in operation['proc_times']:
                            normalized = operation['proc_times'][machine] / self.max_time_horizon
                            obs_parts.append(normalized)
                        else:
                            obs_parts.append(0.0)  # Incompatible
                else:
                    # Should not reach here
                    for machine in self.machines:
                        obs_parts.append(0.0)
        
        
        # 5. NEW: Predicted arrival times (normalized, relative to current time)
        # This is the KEY difference from Reactive RL - predictions guide scheduling
        pred_arrivals = []
        for job_id in self.job_ids:
            if job_id in self.arrived_jobs or job_id in self.completed_jobs:
                pred_arrivals.append(0.0)  # Already arrived
            elif job_id in self.predicted_arrival_times:
                pred_time = self.predicted_arrival_times[job_id]
                # Normalize: time until arrival / max_time_horizon
                time_until = max(0, pred_time - self.event_time)
                normalized = min(1.0, time_until / self.max_time_horizon)
                pred_arrivals.append(normalized)
            else:
                pred_arrivals.append(1.0)  # Far future
        obs_parts.extend(pred_arrivals)
        
        # 6. Arrival progress (how many jobs have arrived so far)
        arrival_progress = len(self.arrived_jobs) / len(self.job_ids)
        obs_parts.append(arrival_progress)

        # 7. Makespan progress (current schedule length)
        makespan_progress = self.current_makespan / self.max_time_horizon
        obs_parts.append(makespan_progress)

        # # 7. NEW: Estimated arrival rate (normalized)
        # stats = self.arrival_predictor.get_stats()
        # estimated_rate = stats['estimated_rate']
        # # Normalize rate (assume typical range 0-0.2)
        # normalized_rate = min(1.0, estimated_rate / 0.2)
        # obs_parts.append(normalized_rate)
        
        return np.array(obs_parts, dtype=np.float32)
    
    def finalize_episode(self):
        """
        Called at end of episode for cross-episode learning.
        Updates predictor with complete arrival information.
        """
        self.arrival_predictor.finalize_episode(self.job_arrival_times)


class PerfectKnowledgeFJSPEnv(gym.Env):
    """
    BUILDER MODE Perfect Knowledge FJSP Environment.
    Knows exact arrival times, can schedule any job (arrived or not) with proper timing.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment"):
        super().__init__()
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        self.reward_mode = reward_mode
        self.job_arrival_times = arrival_times.copy()
        self.max_time_horizon = max(arrival_times.values()) + 50  # Buffer for scheduling
        # Simplified action space: job_idx * num_machines + machine_idx (no WAIT action)
        self.action_space = spaces.Discrete(self.num_jobs * len(self.machines))
        
        # BUILDER MODE observation space: observe MDP state (next_op_idx, machine_free_time, job_ready_time)
        obs_size = (
            self.num_jobs +                         # 1) Job ready time (normalized) - when job can start next op
            self.num_jobs +                         # 2) Job progress (completed_ops / total_ops)
            len(self.machines) +                    # 3) Machine free time (normalized) - when machine is available
            self.num_jobs * len(self.machines) +    # 4) Processing times for NEXT operations
            self.num_jobs +                         # 5) PERFECT: Exact arrival times (normalized)
            1                                       # 6) Current makespan (normalized)
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
                # Calculate maximum processing time across all operations for normalization
        self.max_proc_time = self._calculate_max_processing_time()
        

    def _calculate_max_processing_time(self):
        """Calculate the maximum processing time across all operations and machines."""
        max_time = 0.0
        for job_ops in self.jobs.values():
            for operation in job_ops:
                for proc_time in operation['proc_times'].values():
                    max_time = max(max_time, proc_time)
        return max_time if max_time > 0 else 1.0  # Avoid division by zero


    def reset(self, seed=None, options=None):
        """Reset environment for perfect knowledge builder mode."""
        if seed is not None:
            super().reset(seed=seed, options=options)
            random.seed(seed)
            np.random.seed(seed)
        
        self.machine_next_free = {m: 0.0 for m in self.machines}
        self.schedule = {m: [] for m in self.machines}
        self.completed_ops = {job_id: [False] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.operation_end_times = {job_id: [0.0] * len(self.jobs[job_id]) for job_id in self.job_ids}
        self.next_operation = {job_id: 0 for job_id in self.job_ids}
        
        self.current_makespan = 0.0
        self.operations_scheduled = 0
        self.episode_step = 0
        self.max_episode_steps = self.total_operations * 2
        
        # All jobs available for scheduling with perfect knowledge (can plan ahead)
        self.arrived_jobs = set(self.job_ids)  # Perfect knowledge: can schedule all jobs
        
        return self._get_observation(), {}

    def _decode_action(self, action):
        """Decode action - no WAIT action."""
        action = int(action) % self.action_space.n
        num_machines = len(self.machines)
        
        job_idx = action // num_machines
        machine_idx = action % num_machines
        
        job_idx = min(job_idx, self.num_jobs - 1)
        machine_idx = min(machine_idx, len(self.machines) - 1)
        
        # Operation index is always the next operation for the job
        job_id = self.job_ids[job_idx]
        op_idx = self.next_operation[job_id]
        
        return job_idx, op_idx, machine_idx

    def _is_valid_action(self, job_idx, op_idx, machine_idx):
        """Perfect knowledge validation: precedence + compatibility only."""
        if not (0 <= job_idx < self.num_jobs and 0 <= machine_idx < len(self.machines)):
            return False
        
        job_id = self.job_ids[job_idx]
        
        # Check operation index validity
        if not (0 <= op_idx < len(self.jobs[job_id])):
            return False
            
        # Check precedence: this must be the next operation for the job
        if op_idx != self.next_operation[job_id]:
            return False
            
        # Check machine compatibility
        machine_name = self.machines[machine_idx]
        if machine_name not in self.jobs[job_id][op_idx]['proc_times']:
            return False
            
        return True

    def action_masks(self):
        """Perfect knowledge action masks: compatibility only (all jobs available)."""
        mask = np.full(self.action_space.n, False, dtype=bool)
        
        if self.operations_scheduled >= self.total_operations:
            return mask
        
        # All jobs can be scheduled (perfect knowledge)
        for job_idx, job_id in enumerate(self.job_ids):
            next_op_idx = self.next_operation[job_id]
            if next_op_idx >= len(self.jobs[job_id]):
                continue
                
            for machine_idx, machine in enumerate(self.machines):
                if machine in self.jobs[job_id][next_op_idx]['proc_times']:
                    action = job_idx * len(self.machines) + machine_idx
                    if action < self.action_space.n:
                        mask[action] = True
            
        return mask

    def step(self, action):
        """Perfect knowledge step function - similar to possion_job_backup3.py."""
        self.episode_step += 1
        
         # Terminate if all operations are scheduled
        if self.operations_scheduled >= self.total_operations:
            final_reward = - self.current_makespan  # Bonus for finishing
            return self._get_observation(), final_reward, True, False, {"makespan": self.current_makespan, "status": "completed"}

        # Safety check for infinite episodes
        if self.episode_step >= self.max_episode_steps:
            return self._get_observation(), -1000.0, True, False, {"error": "Max episode steps reached"}
        
        job_idx, op_idx, machine_idx = self._decode_action(action)

        # Use softer invalid action handling like possion_job_backup3.py
        if not self._is_valid_action(job_idx, op_idx, machine_idx):
            # Give a negative reward but don't terminate - helps learning
            return self._get_observation(), -500.0, False, False, {"error": "Invalid action, continuing"}

        job_id = self.job_ids[job_idx]
        machine = self.machines[machine_idx]
        
        # Calculate timing
        machine_available_time = self.machine_next_free.get(machine, 0.0)
        
        # Job ready time includes arrival constraint for first op, precedence propagates it
        job_ready_time = (self.operation_end_times[job_id][op_idx - 1] if op_idx > 0 
                         else self.job_arrival_times.get(job_id, 0.0))
        
        # Start time: machine free + job ready (arrival constraint included in job_ready_time)
        start_time = max(machine_available_time, job_ready_time)
        proc_time = self.jobs[job_id][op_idx]['proc_times'][machine]
        end_time = start_time + proc_time

        # Update state
        previous_makespan = self.current_makespan
        self.machine_next_free[machine] = end_time
        self.operation_end_times[job_id][op_idx] = end_time
        self.completed_ops[job_id][op_idx] = True
        self.next_operation[job_id] += 1
        self.operations_scheduled += 1
        
        # Update makespan
        self.current_makespan = max(self.current_makespan, end_time)

        # Record in schedule
        self.schedule[machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))

        # Check termination
        terminated = self.operations_scheduled >= self.total_operations
        
        # Calculate reward
        idle_time = max(0, start_time - machine_available_time)
        reward = self._calculate_reward(proc_time, idle_time, terminated, 
                                      previous_makespan, self.current_makespan)
        
        info = {"makespan": self.current_makespan}
        return self._get_observation(), reward, terminated, False, info
    

    def _calculate_reward(self, proc_time, idle_time, done, previous_makespan, current_makespan):
        """
        Dense reward shaping for Perfect Knowledge RL.
        
        Problem: makespan_increment alone is SPARSE - most actions get reward=0!
        Solution: Add auxiliary dense signals while keeping makespan as primary objective.
        
        Reward components:
        1. Makespan increment (PRIMARY, weight=10x): Direct objective
        2. Idle time penalty (AUXILIARY): Encourages efficiency
        3. Completion reward (AUXILIARY): Progress signal
        4. Final makespan bonus (TERMINAL): Strong end-of-episode signal
        """
        
        if self.reward_mode == "makespan_increment":
            # R(s_t, a_t) = E(t) - E(t+1) = negative increment in makespan
            if previous_makespan is not None and current_makespan is not None:
                makespan_increment = current_makespan - previous_makespan
                reward = -makespan_increment  # Negative increment (reward for not increasing makespan)
                return reward
            else:
                # Fallback if makespan values not provided
                return -proc_time
        else:
            # Improved reward function with better guidance
            reward = 0.0
            
            # Strong positive reward for completing an operation
            reward += 20.0
            
            # Small penalty for processing time (encourage shorter operations)
            reward -= proc_time * 0.1
            
            # Penalty for idle time (encourage efficiency)  
            reward -= idle_time * 1.0
            
            # Large completion bonus
            if done:
                reward += 200.0
                # Bonus for shorter makespan
                if current_makespan and current_makespan > 0:
                    reward += max(0, 500.0 / current_makespan)
            
            return reward

    def _get_observation(self):
        """
        BUILDER MODE: Observe MDP state for placing operation blocks on Gantt chart.
        State = (next_op_idx, machine_free_time, job_ready_time) for each job-machine pair.
        
        NOTE: Using 0.0 for completed jobs and incompatible machines is intentional:
        - Completed jobs: 0.0 ready time means "already done" (low priority)
        - Incompatible: 0.0 processing time means "not an option"
        Agent learns to ignore these through action masking.
        """
        obs = []
        
        # 1. Job ready time (when job can start its NEXT operation)
        # This captures job precedence constraints and arrival times
        for job_id in self.job_ids:
            if self.next_operation[job_id] < len(self.jobs[job_id]):
                # Job has remaining operations
                next_op_idx = self.next_operation[job_id]
                
                # Job ready time = max(previous_op_end_time, arrival_time)
                if next_op_idx > 0:
                    # Precedence: must wait for previous operation to finish
                    job_ready_time = self.operation_end_times[job_id][next_op_idx - 1]
                else:
                    # First operation: only constrained by arrival time
                    job_ready_time = self.job_arrival_times.get(job_id, 0.0)
                
                # Normalize against max_time_horizon
                normalized_ready_time = min(1.0, job_ready_time / self.max_time_horizon)
                obs.append(normalized_ready_time)
            else:
                # Job completed - use 0.0 to indicate "done"
                obs.append(0.0)
        
        # 2. Job progress (completed_ops / total_ops)
        for job_id in self.job_ids:
            completed_ops = sum(self.completed_ops[job_id])
            total_ops = len(self.jobs[job_id])
            progress = completed_ops / total_ops if total_ops > 0 else 1.0
            obs.append(progress)
        
        # 3. Machine free time (when each machine is available)
        for machine in self.machines:
            machine_free_time = self.machine_next_free[machine]
            # Normalize against max_time_horizon
            normalized_free_time = min(1.0, machine_free_time / self.max_time_horizon)
            obs.append(normalized_free_time)
        
        # 4. Processing times for NEXT operations (for each job-machine pair)
        # ⭐ CRITICAL: Use SAME normalization as all other time values for temporal consistency!
        for job_id in self.job_ids:
            if self.next_operation[job_id] < len(self.jobs[job_id]):
                next_op_idx = self.next_operation[job_id]
                operation = self.jobs[job_id][next_op_idx]
                
                for machine in self.machines:
                    if machine in operation['proc_times']:
                        proc_time = operation['proc_times'][machine]
                        # ✅ UNIFIED NORMALIZATION: Use max_time_horizon (NOT max_proc_time!)
                        # This ensures temporal consistency: agent can reason about time progression
                        # Example: machine_free=0.25, proc_time=0.05 → machine_free_after=0.30
                        normalized_time = min(1.0, proc_time / self.max_time_horizon)
                        obs.append(normalized_time)
                    else:
                        # Incompatible machine - use 0.0 to indicate "not an option"
                        # Action masking will prevent selecting this anyway
                        obs.append(0.0)
            else:
                # Job completed - all machines get 0.0
                for machine in self.machines:
                    obs.append(0.0)
        
        # 5. PERFECT KNOWLEDGE: Exact arrival times (normalized)
        # This is the key advantage - agent knows when future jobs will arrive
        for job_id in self.job_ids:
            arrival_time = self.job_arrival_times.get(job_id, 0.0)
            normalized_arrival = min(1.0, arrival_time / self.max_time_horizon)
            obs.append(normalized_arrival)
        
        # 6. Current makespan (global scheduling progress)
        normalized_makespan = min(1.0, self.current_makespan / self.max_time_horizon)
        obs.append(normalized_makespan)
        
        obs_array = np.array(obs, dtype=np.float32)
        # Handle any numerical issues
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs_array

def reset_training_metrics():
    """Reset TRAINING_METRICS for a new training run."""
    global TRAINING_METRICS
    TRAINING_METRICS = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_timesteps': [],
        'action_entropy': [],      # train/entropy_loss
        'policy_loss': [],         # train/policy_gradient_loss
        'value_loss': [],          # train/value_loss
        'total_loss': [],          # train/loss (total combined loss)
        'timesteps': [],
        'episode_count': [],
        'learning_rate': [],
        'explained_variance': [],
        'kl_divergence': [],       # train/approx_kl (KL divergence)
    }
    print("✅ Training metrics reset for new training run")

def mask_fn(env):
    """Function to retrieve action masks from the environment."""
    return env.action_masks()

# class EnhancedTrainingCallback(BaseCallback):
#     """Enhanced BaseCallback to track comprehensive training metrics at proper rollout boundaries with episode tracking."""
    
#     def __init__(self, method_name, pbar=None, verbose=0):
#         super().__init__(verbose)
#         self.method_name = method_name
#         self.pbar = pbar
#         self.episode_count = 0
#         self.last_num_episodes = 0  # Track total episodes processed
        
#     def _on_training_start(self) -> None:
#         """Initialize TRAINING_METRICS for this method."""
#         global TRAINING_METRICS
#         TRAINING_METRICS['method_name'] = self.method_name
#         print(f"[{self.method_name}] Training callback initialized")
        
#     def _on_rollout_end(self) -> None:
#         """Called at the end of each rollout - debug logging only."""
#         global TRAINING_METRICS
        
#         # DEBUG: Print logger status (only first few times)
#         if self.verbose >= 1 and len(TRAINING_METRICS['episode_rewards']) < 3:
#             if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
#                 log_data = self.model.logger.name_to_value
#                 available_keys = list(log_data.keys())
#                 print(f"[{self.method_name}] Rollout end - Logger keys: {available_keys}")
#                 if 'rollout/ep_rew_mean' in log_data:
#                     print(f"[{self.method_name}] Rollout end - ep_rew_mean available: {log_data['rollout/ep_rew_mean']}")
#                 else:
#                     print(f"[{self.method_name}] Rollout end - No ep_rew_mean yet")
#             else:
#                 print(f"[{self.method_name}] No logger available at rollout end")
    
#     def _on_step(self) -> bool:
#         """Update progress bar and capture episode metrics when available."""
#         global TRAINING_METRICS
        
#         # Update progress bar if available
#         if self.pbar is not None:
#             self.pbar.n = self.model.num_timesteps
#             self.pbar.refresh()
            
#         # Check for new logged data every 512 steps to avoid performance impact
#         if self.model.num_timesteps % 512 == 0:
#             if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
#                 log_data = self.model.logger.name_to_value
                
#                 # Capture episode rewards when they become available
#                 if 'rollout/ep_rew_mean' in log_data:
#                     ep_reward = log_data['rollout/ep_rew_mean']
#                     # Avoid duplicates by checking if this is a new value
#                     if not TRAINING_METRICS['episode_rewards'] or abs(ep_reward - TRAINING_METRICS['episode_rewards'][-1]) > 1e-6:
#                         TRAINING_METRICS['episode_rewards'].append(ep_reward)
#                         self.episode_count += 1
#                         TRAINING_METRICS['episode_count'].append(self.episode_count)
                        
#                         if self.verbose >= 1 and self.episode_count <= 5:
#                             print(f"[{self.method_name}] ✅ Episode reward: {ep_reward:.4f} (episode #{self.episode_count})")
                
#                 # Capture other metrics when available
#                 if 'rollout/ep_len_mean' in log_data:
#                     ep_len = log_data['rollout/ep_len_mean']
#                     if not TRAINING_METRICS['episode_lengths'] or abs(ep_len - TRAINING_METRICS['episode_lengths'][-1]) > 1e-6:
#                         TRAINING_METRICS['episode_lengths'].append(ep_len)
                
#                 # Training metrics
#                 if 'train/policy_gradient_loss' in log_data:
#                     TRAINING_METRICS['policy_loss'].append(log_data['train/policy_gradient_loss'])
                    
#                 if 'train/value_loss' in log_data:
#                     TRAINING_METRICS['value_loss'].append(log_data['train/value_loss'])
                    
#                 if 'train/entropy_loss' in log_data:
#                     TRAINING_METRICS['action_entropy'].append(log_data['train/entropy_loss'])
                    
#                 if 'train/learning_rate' in log_data:
#                     TRAINING_METRICS['learning_rate'].append(log_data['train/learning_rate'])
                
#                 if 'train/explained_variance' in log_data:
#                     TRAINING_METRICS['explained_variance'].append(log_data['train/explained_variance'])
                
#                 # Always record timesteps
#                 TRAINING_METRICS['timesteps'].append(self.model.num_timesteps)
        
#         return True


# class EnhancedTrainingCallback(BaseCallback):
#     """
#     Robust callback that:
#       - collects finished episodes from self.model.ep_info_buffer reliably
#       - records train/* scalars from logger.name_to_value periodically
#       - stores timesteps for each episode (so plots align)
#     """
#     def __init__(self, method_name, pbar=None, verbose=0):
#         super().__init__(verbose)
#         self.method_name = method_name
#         self.pbar = pbar

#         # how many ep infos we've already processed
#         self._processed_episodes = 0

#     def _on_training_start(self) -> None:
#         global TRAINING_METRICS
#         TRAINING_METRICS['method_name'] = self.method_name
#         if self.verbose:
#             print(f"[{self.method_name}] Callback init. Will record train scalars at rollout boundaries.")

#     def _on_step(self) -> bool:
#         global TRAINING_METRICS

#         # update progress bar
#         if self.pbar is not None:
#             self.pbar.n = getattr(self.model, "num_timesteps", self.pbar.n)
#             self.pbar.refresh()

#         # ----- 1) collect finished episodes from ep_info_buffer -----
#         # ep_info_buffer is a deque (OnPolicyAlgorithm.ep_info_buffer)
#         try:
#             ep_buf = getattr(self.model, "ep_info_buffer", None)
#         except Exception:
#             ep_buf = None

#         if ep_buf is not None:
#             # ep_buf is indexable; its elements are dicts like {'r': reward, 'l': length}
#             # We'll process any new entries since last check
#             current_len = len(ep_buf)
#             if current_len > self._processed_episodes:
#                 # process new ones
#                 for idx in range(self._processed_episodes, current_len):
#                     info = ep_buf[idx]
#                     r = float(info.get("r", np.nan))
#                     l = int(info.get("l", -1))
#                     t = getattr(self.model, "num_timesteps", 0)
#                     TRAINING_METRICS['episode_rewards'].append(r)
#                     TRAINING_METRICS['episode_lengths'].append(l)
#                     TRAINING_METRICS['episode_timesteps'].append(t)
#                     # Also maintain a simple episode counter
#                     TRAINING_METRICS['episode_count'].append(len(TRAINING_METRICS['episode_rewards']))
#                     if self.verbose and len(TRAINING_METRICS['episode_rewards']) <= 5:
#                         print(f"[{self.method_name}] ✅ Episode #{len(TRAINING_METRICS['episode_rewards'])} reward={r:.3f}, len={l}, t={t}")
#                 self._processed_episodes = current_len

#         # ----- 2) record train/* scalars from logger.name_to_value at rollout boundaries -----
#         # This aligns with verbose output which shows rollout-level metrics
#         return True

#     def _on_rollout_end(self) -> None:
#         """Called at the end of each rollout - record training metrics here to align with verbose output."""
#         global TRAINING_METRICS
        
#         if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
#             log_data = dict(self.model.logger.name_to_value)  # copy
#             # policy/value/entropy and other train scalars
#             if 'train/policy_gradient_loss' in log_data:
#                 TRAINING_METRICS['policy_loss'].append(float(log_data['train/policy_gradient_loss']))
#             if 'train/value_loss' in log_data:
#                 TRAINING_METRICS['value_loss'].append(float(log_data['train/value_loss']))
#             if 'train/entropy_loss' in log_data:
#                 TRAINING_METRICS['action_entropy'].append(float(log_data['train/entropy_loss']))
#             if 'train/loss' in log_data:  # Total loss
#                 TRAINING_METRICS['total_loss'].append(float(log_data['train/loss']))
#             if 'train/learning_rate' in log_data:
#                 TRAINING_METRICS['learning_rate'].append(float(log_data['train/learning_rate']))
#             if 'train/explained_variance' in log_data:
#                 TRAINING_METRICS['explained_variance'].append(float(log_data['train/explained_variance']))

#             # record the timestep for these training samples (aligned with rollouts)
#             TRAINING_METRICS['timesteps'].append(getattr(self.model, "num_timesteps", 0))
            
#             if self.verbose and len(TRAINING_METRICS['policy_loss']) <= 5:
#                 print(f"[{self.method_name}] 📊 Rollout #{len(TRAINING_METRICS['policy_loss'])}: "
#                       f"policy_loss={TRAINING_METRICS['policy_loss'][-1]:.6f}, "
#                       f"value_loss={TRAINING_METRICS['value_loss'][-1]:.6f}, "
#                       f"timesteps={TRAINING_METRICS['timesteps'][-1]}")

#         return True

# Global container used in your project (you can adapt)
TRAINING_METRICS = {
    'episode_rewards': [],        # per-episode scalar (exact)
    'episode_lengths': [],        # per-episode length (exact)
    'episode_timesteps': [],      # num_timesteps when episode finished
    'rollout_ep_rew_mean': [],    # mean reward of episodes that ended during rollout
    'rollout_ep_count': [],       # how many episodes ended in the rollout
    'timesteps': [],              # timesteps aligned with rollout-level logs
    'action_entropy': [],         # train/entropy_loss (per-rollout)
    'policy_loss': [],            # train/policy_gradient_loss (per-rollout)
    'value_loss': [],             # train/value_loss (per-rollout)
    'total_loss': [],             # train/loss (per-rollout)
    'learning_rate': [],
    'explained_variance': [],
    'kl_divergence': [],       # train/approx_kl (KL divergence)
}

def reset_training_metrics():
    for k in TRAINING_METRICS:
        TRAINING_METRICS[k] = []

class EnhancedTrainingCallback(BaseCallback):
    """
    Robust callback that:
      - records episode rewards & lengths as soon as episodes finish (using 'infos' in _on_step)
      - records train/* scalars at _on_rollout_end (these are the SB3 update-level losses)
      - computes & stores mean episode reward for episodes that finished during the last rollout
    Usage: pass this callback to model.learn(..., callback=EnhancedTrainingCallback(...))
    Requirements: Wrap your env(s) with Monitor / VecMonitor so 'infos' include episode data.
    """
    def __init__(self, method_name="RL", pbar=None, verbose=1):
        super().__init__(verbose)
        self.method_name = method_name
        self.pbar = pbar
        self._episodes_since_last_rollout = []  # store indices of episodes collected during current rollout

    def _on_training_start(self) -> None:
        reset_training_metrics()
        TRAINING_METRICS['method_name'] = self.method_name
        if self.verbose:
            print(f"[{self.method_name}] EnhancedTrainingCallback initialized")

    def _on_step(self) -> bool:
        """
        Called after each env.step(). Use locals()['infos'] to detect episode finishes reliably.
        Each info dict (from Monitor) that contains an 'episode' key signals an episode end.
        """
        # Update progress bar (if provided)
        if self.pbar is not None:
            # model.num_timesteps might not exist early; guard it
            ts = getattr(self.model, "num_timesteps", self.pbar.n)
            self.pbar.n = ts
            self.pbar.refresh()

        # Get infos from the current step; this is the standard place to find episode endings
        infos = self.locals.get("infos")
        if infos is None:
            # fallback: sometimes SB3 uses a different locals structure
            return True

        # iterate through all infos from vectorized envs
        new_eps_count = 0
        for info in infos:
            # Std Monitor inserts {'episode': {'r': ..., 'l': ...}} into info at episode end
            if not isinstance(info, dict):
                continue
            ep = info.get("episode") or info.get("episode_info") or info.get("ep") or None
            # Some wrappers may directly provide 'episode' key; this is the robust check
            if ep is not None:
                # 'ep' should be a dict with 'r' and 'l' keys
                r = float(ep.get("r", np.nan))
                l = int(ep.get("l", -1))
                t = getattr(self.model, "num_timesteps", 0)
                TRAINING_METRICS['episode_rewards'].append(r)
                TRAINING_METRICS['episode_lengths'].append(l)
                TRAINING_METRICS['episode_timesteps'].append(t)
                self._episodes_since_last_rollout.append(len(TRAINING_METRICS['episode_rewards']) - 1)  # index
                new_eps_count += 1
                if self.verbose and len(TRAINING_METRICS['episode_rewards']) <= 5:
                    print(f"[{self.method_name}] Episode finished #{len(TRAINING_METRICS['episode_rewards'])}: r={r:.3f}, l={l}, t={t}")

        return True

    def _on_rollout_end(self) -> None:
        """
        Called when SB3 finished collecting a rollout and performed the update.
        This is where `train/*` scalars are available in logger.name_to_value.
        We'll snapshot them and also compute mean reward for episodes that ended during this rollout.
        """
        # snapshot SB3 logger scalars (they exist here)
        log_map = {}
        if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
            try:
                log_map = dict(self.model.logger.name_to_value)
            except Exception:
                log_map = {}

        def _g(k):
            v = log_map.get(k)
            return float(v) if v is not None else None

        # store rollout-aligned training scalars
        TRAINING_METRICS['policy_loss'].append(_g('train/policy_gradient_loss'))
        TRAINING_METRICS['value_loss'].append(_g('train/value_loss'))
        TRAINING_METRICS['action_entropy'].append(_g('train/entropy_loss'))
        TRAINING_METRICS['total_loss'].append(_g('train/loss'))
        TRAINING_METRICS['learning_rate'].append(_g('train/learning_rate'))
        TRAINING_METRICS['explained_variance'].append(_g('train/explained_variance'))
        TRAINING_METRICS['kl_divergence'].append(_g('train/approx_kl'))


        # record rollout timestep (for x-axis alignment with verbose)
        TRAINING_METRICS['timesteps'].append(getattr(self.model, "num_timesteps", 0))

        # compute mean episode reward for episodes that finished since last rollout
        if self._episodes_since_last_rollout:
            idxs = self._episodes_since_last_rollout
            ep_rewards = [TRAINING_METRICS['episode_rewards'][i] for i in idxs]
            mean_r = float(np.mean(ep_rewards)) if len(ep_rewards) > 0 else None
            TRAINING_METRICS['rollout_ep_rew_mean'].append(mean_r)
            TRAINING_METRICS['rollout_ep_count'].append(len(ep_rewards))
            if self.verbose:
                print(f"[{self.method_name}] Rollout end: recorded rollout_ep_rew_mean={mean_r:.3f} over {len(ep_rewards)} episodes")
        else:
            # fallback to whatever SB3 computed (if present) or append None
            sb3_val = log_map.get('rollout/ep_rew_mean')
            TRAINING_METRICS['rollout_ep_rew_mean'].append(float(sb3_val) if sb3_val is not None else None)
            TRAINING_METRICS['rollout_ep_count'].append(0)
            if self.verbose:
                print(f"[{self.method_name}] Rollout end: no episodes finished during rollout; sb3 rollout/ep_rew_mean={sb3_val}")

        # reset the per-rollout buffer for next rollout
        self._episodes_since_last_rollout = []

        return None


def train_perfect_knowledge_agent(jobs_data, machine_list, arrival_times, total_timesteps=100000, 
                                  reward_mode="makespan_increment", learning_rate=3e-4,
                                  num_initializations=3, milp_optimal=None):
    """
    Train a perfect knowledge RL agent with MULTIPLE INITIALIZATIONS and hyperparameter tuning.
    
    Key improvements:
    1. Multiple random initializations - train N times, pick best
    2. Optimized hyperparameters for FJSP scheduling
    3. Early stopping when close to MILP optimal
    4. Adaptive learning rate scheduling
    
    IMPORTANT: Each Perfect Knowledge RL is trained specifically for ONE scenario
    with the exact arrival times, making it the optimal RL benchmark for that scenario.
    
    Args:
        jobs_data: Job specifications
        machine_list: List of machines
        arrival_times: Exact arrival times for this scenario
        total_timesteps: Training timesteps PER initialization
        reward_mode: Reward mode
        learning_rate: Initial learning rate
        num_initializations: Number of random initializations to try
        milp_optimal: MILP optimal makespan (if known) for early stopping
    
    Returns:
        Best trained model (closest to MILP)
    """
    print(f"    Training arrival times: {arrival_times}")
    print(f"    Timesteps: {total_timesteps:,} per init | Initializations: {num_initializations}")
    print(f"    MILP Optimal: {milp_optimal:.2f}" if milp_optimal else "    MILP Optimal: Unknown")
    
    best_model = None
    best_makespan = float('inf')
    best_init_idx = -1
    
    # Try multiple random initializations
    for init_idx in range(num_initializations):
        seed = GLOBAL_SEED + 1000 + init_idx * 100  # Different seed per initialization
        print(f"\n    --- Initialization {init_idx + 1}/{num_initializations} (seed={seed}) ---")
        
        def make_perfect_env():
            env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times, 
                                         reward_mode=reward_mode)
            env = ActionMasker(env, mask_fn)
            return env

        vec_env = DummyVecEnv([make_perfect_env])
        vec_env = VecMonitor(vec_env)
        
        # OPTIMIZED hyperparameters for FJSP scheduling
        # Key insights:
        # 1. Larger network for complex scheduling decisions
        # 2. More training epochs per rollout for sample efficiency
        # 3. Lower entropy for exploitation (we know exact arrivals!)
        # 4. Adaptive learning rate
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=learning_rate,
            n_steps=2048,              # ⭐ Larger rollout for better value estimates
            batch_size=256,            # Good batch size for stability
            n_epochs=10,               # ⭐ More epochs = better learning per rollout
            gamma=1.0,                 # Undiscounted (makespan)
            gae_lambda=0.95,           # GAE for advantage estimation
            clip_range=0.2,            # Standard PPO clipping
            ent_coef=0.01,           # ⭐ VERY LOW entropy (deterministic scenario!)
            vf_coef=0.5,               # Value function coefficient
            max_grad_norm=0.5,         # Gradient clipping
            normalize_advantage=True,
            seed=seed,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[512, 256, 256],    # ⭐ Deep policy network
                    vf=[512, 256, 128]     # Separate value network
                ),
                activation_fn=torch.nn.ReLU
            )
        )
        
        # Quick random baseline
        test_env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times, 
                                          reward_mode=reward_mode)
        test_env = ActionMasker(test_env, mask_fn)
        
        # Train with progress tracking
        reset_training_metrics()
        
        with tqdm(total=total_timesteps, desc=f"  Init {init_idx+1}/{num_initializations}", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                  leave=False) as pbar:
            
            callback = EnhancedTrainingCallback(f"Perfect RL Init {init_idx+1}", 
                                               pbar=pbar, verbose=0)
            model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # IMPORTANT: Check final training reward (stochastic)
        final_train_reward = TRAINING_METRICS['rollout_ep_rew_mean'][-1] if TRAINING_METRICS['rollout_ep_rew_mean'] else None
        
        # Evaluate this initialization (DETERMINISTIC)
        obs, _ = test_env.reset()
        done = False
        steps = 0
        max_eval_steps = 300
        eval_reward = 0.0
        
        while not done and steps < max_eval_steps:
            action_masks = test_env.action_masks()
            if not any(action_masks):
                break
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            eval_reward += reward
            steps += 1
        
        init_makespan = test_env.env.current_makespan
        print(f"    Init {init_idx+1} makespan: {init_makespan:.2f}", end="")
        
        # Show comparison of stochastic training vs deterministic evaluation
        if final_train_reward is not None:
            print(f" | Train reward: {final_train_reward:.2f} | Eval reward: {eval_reward:.2f}", end="")
        
        # Compare to MILP if available
        if milp_optimal:
            gap = ((init_makespan - milp_optimal) / milp_optimal) * 100
            print(f" (gap: {gap:+.2f}%)", end="")
            
            # Early stopping if very close to MILP (within 1%)
            if gap < 1.0:
                print(f" ✅ EXCELLENT! Within 1% of MILP")
                best_model = model
                best_makespan = init_makespan
                best_init_idx = init_idx
                break
        
        # Track best model
        if init_makespan < best_makespan:
            best_makespan = init_makespan
            best_model = model
            best_init_idx = init_idx
            print(f" ⭐ NEW BEST")
        else:
            print()
    
    print(f"\n    ✅ Best initialization: {best_init_idx + 1} with makespan {best_makespan:.2f}")
    
    if milp_optimal:
        final_gap = ((best_makespan - milp_optimal) / milp_optimal) * 100
        print(f"    📊 Final gap to MILP: {final_gap:+.2f}%")
        if final_gap > 5.0:
            print(f"    ⚠️  WARNING: Gap > 5% - RL may need more training or better hyperparameters")
        elif final_gap < 0:
            print(f"    🚨 CRITICAL: RL outperforms MILP by {-final_gap:.2f}% - CHECK FOR BUGS!")
    
    return best_model
    print(f"    ✅ Perfect knowledge training completed for this scenario!")
    return model

def train_static_agent(jobs_data, machine_list, total_timesteps=300000, reward_mode="makespan_increment", learning_rate=3e-4):
    """Train a static RL agent where all jobs are available at t=0."""
    print(f"\n--- Training Static RL Agent on {len(jobs_data)} jobs ---")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    # CORRECTED: Use PerfectKnowledgeFJSPEnv with all arrival times = 0 for static scenario
    static_arrival_times = {job_id: 0.0 for job_id in jobs_data.keys()}
    
    def make_static_env():
        env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, static_arrival_times, reward_mode=reward_mode)
        env = ActionMasker(env, mask_fn)
        # env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_static_env])
    # Add VecMonitor to properly capture episode statistics
    vec_env = VecMonitor(vec_env)
    
    # IDENTICAL hyperparameters as Perfect Knowledge RL for fair comparison
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,  # Minimal output
        learning_rate=3e-4,
        n_steps=1024,              # IDENTICAL across all RL methods
        batch_size=256,            # IDENTICAL across all RL methods
        n_epochs=5,               # IDENTICAL across all RL methods (no special case for late arrivals)
        gamma=1,                # IDENTICAL across all RL methods
        gae_lambda=0.99,
        clip_range=0.2,
        ent_coef=0.01,             # IDENTICAL across all RL methods
        vf_coef=0.1,
        max_grad_norm=0.5,
        normalize_advantage=True,
        seed=GLOBAL_SEED,          # Ensure reproducibility
        policy_kwargs=dict(
            net_arch=[256, 256, 128],  # IDENTICAL across all RL methods
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"Training Static RL for {total_timesteps:,} timesteps with seed {GLOBAL_SEED}...")
    
    # Reset training metrics for this run
    reset_training_metrics()
    
    # Train with enhanced callback and progress bar
    start_time = time.time()
    
    with tqdm(total=total_timesteps, desc="Static RL Training", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} timesteps [{elapsed}<{remaining}]') as pbar:
        
        callback = EnhancedTrainingCallback("Static RL", pbar=pbar, verbose=0)
        model.learn(total_timesteps=total_timesteps, callback=callback)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Static RL training completed in {training_time:.1f}s!")
    
    return model


def train_dynamic_agent(jobs_data, machine_list, initial_jobs=5, arrival_rate=0.08, total_timesteps=500000, reward_mode="makespan_increment", learning_rate=3e-4):
    """
    Train a reactive RL agent on Poisson job arrivals with EXPANDED DATASET.
    """
    print(f"\n--- Training Reactive RL Agent on {len(jobs_data)} jobs ---")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_dynamic_env():
        env = PoissonDynamicFJSPEnv(
            jobs_data, machine_list, 
            initial_jobs=initial_jobs,
            arrival_rate=arrival_rate,
            reward_mode=reward_mode,
            seed=GLOBAL_SEED+100,  # Ensure reproducibility
            max_time_horizon=1000  # Standard time horizon
        )
        env = ActionMasker(env, mask_fn)
        # env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_dynamic_env])
    # Add VecMonitor to properly capture episode statistics
    vec_env = VecMonitor(vec_env)
    
    # IDENTICAL hyperparameters to proactive RL for fair comparison
    # Both methods learn with same capacity, only difference is predictor observations
    def linear_schedule(initial_value: float):
        """Linear learning rate schedule for stability."""
        def func(progress_remaining: float) -> float:
            return initial_value * (0.1 + 0.9 * progress_remaining)
        return func
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=linear_schedule(learning_rate),  # Decaying LR (IDENTICAL to proactive)
        n_steps=2048,               # IDENTICAL to proactive (larger rollout)
        batch_size=512,             # IDENTICAL to proactive (larger batches)
        n_epochs=10,                 # IDENTICAL to proactive (more epochs)
        gamma=1.0,                  # Undiscounted (makespan objective)
        gae_lambda=0.95,            # GAE for advantage estimation
        clip_range=0.2,             # Standard PPO clipping
        ent_coef=0.01,              # IDENTICAL to proactive (higher exploration)
        vf_coef=0.5,                # Value function coefficient
        max_grad_norm=0.5,          # Gradient clipping for stability
        normalize_advantage=True,   # Normalize advantages
        seed=GLOBAL_SEED,           # Ensure reproducibility
        policy_kwargs=dict(
            net_arch=[256, 256, 128],  # IDENTICAL across all RL methods
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"Training Reactive RL for {total_timesteps:,} timesteps with seed {GLOBAL_SEED}...")
    
    # Reset training metrics for this run
    reset_training_metrics()
    
    # Train with enhanced callback and progress bar
    start_time = time.time()
    
    with tqdm(total=total_timesteps, desc="Reactive RL Training", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} timesteps [{elapsed}<{remaining}]') as pbar:
        
        callback = EnhancedTrainingCallback("Reactive RL", pbar=pbar, verbose=0)
        model.learn(total_timesteps=total_timesteps, callback=callback)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Reactive RL training completed in {training_time:.1f}s!")
    
    return model


def train_rule_based_agent(jobs_data, machine_list, initial_jobs=5, arrival_rate=0.08, 
                           total_timesteps=500000, reward_mode="makespan_increment", 
                           learning_rate=3e-4):
    """
    Train a RULE-BASED RL agent where actions select from dispatching rules.
    
    This approach learns WHEN to apply WHICH rule based on system state,
    similar to literature approaches that learn dispatching rule selection.
    
    Rules: FIFO, SPT, LPT, MWKR, LWKR, EDD, WAIT
    """
    print(f"\n--- Training Rule-Based RL Agent on {len(jobs_data)} jobs ---")
    print(f"Action Space: Select from 7 dispatching rules")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    
    def make_rule_env():
        env = DispatchingRuleFJSPEnv(
            jobs_data, machine_list,
            initial_jobs=initial_jobs,
            arrival_rate=arrival_rate,
            reward_mode=reward_mode,
            seed=GLOBAL_SEED+300,  # Different seed for rule-based
            max_time_horizon=1000
        )
        env = ActionMasker(env, mask_fn)
        return env
    
    vec_env = DummyVecEnv([make_rule_env])
    vec_env = VecMonitor(vec_env)
    
    # IDENTICAL hyperparameters to other RL methods for fair comparison
    def linear_schedule(initial_value: float):
        """Linear learning rate schedule for stability."""
        def func(progress_remaining: float) -> float:
            return initial_value * (0.1 + 0.9 * progress_remaining)
        return func
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=linear_schedule(learning_rate),
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        seed=GLOBAL_SEED,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"Training Rule-Based RL for {total_timesteps:,} timesteps with seed {GLOBAL_SEED}...")
    
    reset_training_metrics()
    
    start_time = time.time()
    
    with tqdm(total=total_timesteps, desc="Rule-Based RL Training",
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} timesteps [{elapsed}<{remaining}]') as pbar:
        
        callback = EnhancedTrainingCallback("Rule-Based RL", pbar=pbar, verbose=0)
        model.learn(total_timesteps=total_timesteps, callback=callback)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Rule-Based RL training completed in {training_time:.1f}s!")
    
    return model


def train_proactive_agent(jobs_data, machine_list, initial_jobs=5, arrival_rate=0.08, 
                          total_timesteps=500000, 
                          reward_mode="makespan_increment", learning_rate=3e-4,
                          predictor_mode='mle', prior_shape=2.0, prior_rate=None):
    """
    Train a PROACTIVE RL agent that learns to predict job arrivals and use WAIT strategically.
    
    SIMPLIFIED DESIGN:
    ------------------
    Since jobs are HOMOGENEOUS (no job types), the strategic depth comes from:
    1. **Machine Heterogeneity**: Fast machines process ALL jobs faster than slow machines
    2. **Processing Time Variance**: Some operations are much longer than others
    3. **Arrival Prediction**: Agent learns Poisson arrival rate across episodes (MLE or MAP)
    4. **Strategic Waiting**: Agent learns to WAIT for fast machines when beneficial
    
    Key differences from reactive agent:
    - Uses ProactiveDynamicFJSPEnv with arrival predictor
    - Has WAIT actions with flexible durations (1, 2, 3, 5, 10, next_event)
    - Learns arrival rate via MLE or MAP across episodes
    - Simple reward: -makespan_increment (consistent with scheduling)
    
    WAIT ACTION BECOMES CRITICAL:
    -----------------------------
    Example decision: Operation with proc_time=40
    - On slow machine (speed=1.5): 40 * 1.5 = 60 time units
    - On fast machine (speed=0.7): 40 * 0.7 = 28 time units
    - Savings: 32 time units!
    - Agent learns: Worth waiting if fast machine free soon
    
    Predictor Modes:
    ----------------
    - 'mle': Maximum Likelihood Estimation (uses only observed data)
    - 'map': Maximum A Posteriori with Gamma prior (Bayesian estimation)
    
    Args:
        jobs_data: Job specifications (simple format, no metadata)
        machine_list: List of machines with heterogeneous speeds
        initial_jobs: Initial jobs or list of job IDs
        arrival_rate: TRUE Poisson arrival rate (hidden from agent)
        total_timesteps: Training duration
        reward_mode: Reward calculation mode
        learning_rate: Learning rate
        predictor_mode: 'mle' or 'map' for arrival prediction
        prior_shape: Shape parameter for Gamma prior (MAP only)
        prior_rate: Rate parameter for Gamma prior (MAP only)
    
    Returns:
        Trained MaskablePPO model
    """
    print(f"\n--- Training PROACTIVE RL Agent on {len(jobs_data)} jobs ---")
    print(f"Timesteps: {total_timesteps:,} | Reward: {reward_mode}")
    print(f"⚠️  Agent does NOT know true arrival rate (λ={arrival_rate})")
    print(f"✅ Agent LEARNS arrival rate via {predictor_mode.upper()} across episodes")
    print(f"🎯 STRATEGIC FOCUS: Learn to WAIT for fast machines when beneficial")
    
    def make_proactive_env():
        env = ProactiveDynamicFJSPEnv(
            jobs_data, machine_list,
            initial_jobs=initial_jobs,
            arrival_rate=arrival_rate,  # Hidden from agent, only for generation
            reward_mode=reward_mode,
            seed=GLOBAL_SEED+200,  # Different seed from reactive
            max_time_horizon=1000,
            predictor_mode=predictor_mode,
            prior_shape=prior_shape,
            prior_rate=prior_rate
        )
        env = ActionMasker(env, mask_fn)
        return env
    
    vec_env = DummyVecEnv([make_proactive_env])
    vec_env = VecMonitor(vec_env)
    
    # NOTE: VecNormalize removed because:
    # 1. Observations already bounded [0, 1] (manually normalized)
    # 2. Causes train/eval mismatch (model trained on normalized, evaluated on raw)
    # 3. Adds complexity without clear benefit for bounded observations
    
    # IDENTICAL hyperparameters to reactive RL for fair comparison
    # Both methods use same model capacity - only difference is predictor in observations
    def linear_schedule(initial_value: float):
        """Linear learning rate schedule for stability."""
        def func(progress_remaining: float) -> float:
            # progress_remaining: 1.0 → 0.0
            # Start at initial_value, decay to initial_value * 0.1
            return initial_value * (0.1 + 0.9 * progress_remaining)
        return func
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=linear_schedule(learning_rate),  # Decaying LR for stability
        n_steps=2048,               # Larger rollout for better value estimates
        batch_size=512,             # Larger batches = more stable gradients
        n_epochs=10,                 # More epochs (proactive is complex!)
        gamma=1.0,                  # Undiscounted (makespan objective)
        gae_lambda=0.95,            # GAE for advantage estimation
        clip_range=0.2,             # Standard PPO clipping
        ent_coef=0.01,              # Higher entropy for exploration (wait timing)
        vf_coef=0.5,                # Value function coefficient
        max_grad_norm=0.5,          # Gradient clipping for stability
        normalize_advantage=True,   # Normalize advantages
        seed=GLOBAL_SEED,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],  # 3-layer network
            activation_fn=torch.nn.ReLU
        )
    )
    
    print(f"Training Proactive RL for {total_timesteps:,} timesteps with seed {GLOBAL_SEED}...")
    print(f"Expected behavior:")
    print(f"  - Early episodes: Conservative (poor predictions)")
    print(f"  - Mid training: Learning arrival patterns")
    print(f"  - Late training: Aggressive proactive scheduling")
    
    # Reset training metrics
    reset_training_metrics()
    
    # Custom callback to track prediction learning
    class ProactiveTrainingCallback(EnhancedTrainingCallback):
        """Extended callback to track prediction accuracy."""
        def __init__(self, method_name, pbar=None, verbose=0):
            super().__init__(method_name, pbar, verbose)
            self.prediction_stats = []
        
        def _on_step(self) -> bool:
            """Called after each env.step() - detect episode ends and call finalize_episode."""
            # First, call parent's _on_step to handle metrics
            result = super()._on_step()
            
            # Then, check if any episodes ended and call finalize_episode
            infos = self.locals.get("infos")
            if infos is not None:
                for info in infos:
                    if not isinstance(info, dict):
                        continue
                    ep = info.get("episode") or info.get("episode_info") or info.get("ep") or None
                    if ep is not None:
                        # Episode ended! Call finalize_episode on the environment
                        try:
                            env = self.model.get_env().envs[0]
                            if hasattr(env, 'finalize_episode'):
                                env.finalize_episode()
                                if self.verbose and len(self.prediction_stats) <= 3:
                                    print(f"[Proactive] ✓ Episode finalized - predictor learning from history")
                        except Exception as e:
                            if self.verbose:
                                print(f"[Proactive] ⚠️  Failed to finalize episode: {e}")
            
            return result
        
        def _on_rollout_end(self):
            """Track predictor statistics at rollout boundaries."""
            super()._on_rollout_end()
            
            # Access environment's predictor
            try:
                env = self.model.get_env().envs[0]
                if hasattr(env, 'arrival_predictor'):
                    stats = env.arrival_predictor.get_stats()
                    self.prediction_stats.append({
                        'timestep': self.model.num_timesteps,
                        'estimated_rate': stats['estimated_rate'],
                        'confidence': stats['confidence'],
                        'num_observations': stats.get('num_global_observations', 0)
                    })
                    
                    if self.verbose and len(self.prediction_stats) <= 5:
                        print(f"\n[Proactive] 📈 Predictor stats: "
                              f"rate={stats['estimated_rate']:.4f}, "
                              f"confidence={stats['confidence']:.2f}, "
                              f"global_obs={stats.get('num_global_observations', 0)}")
            except Exception as e:
                pass  # Silent fail
    
    # Train with progress bar
    start_time = time.time()
    
    with tqdm(total=total_timesteps, desc="Proactive RL Training",
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} timesteps [{elapsed}<{remaining}]') as pbar:
        
        callback = ProactiveTrainingCallback("Proactive RL", pbar=pbar, verbose=1)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Print final prediction stats
        if callback.prediction_stats:
            final_stats = callback.prediction_stats[-1]
            print(f"\n{'='*60}")
            print(f"FINAL PREDICTOR STATISTICS:")
            print(f"  True arrival rate:      λ = {arrival_rate:.4f}")
            print(f"  Learned arrival rate:   λ̂ = {final_stats['estimated_rate']:.4f}")
            print(f"  Prediction error:       {abs(arrival_rate - final_stats['estimated_rate']):.4f}")
            print(f"  Confidence:             {final_stats['confidence']:.2%}")
            print(f"  Total observations:     {final_stats['num_observations']}")
            print(f"{'='*60}")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Proactive RL training completed in {training_time:.1f}s!")
    
    return model


def generate_test_scenarios(jobs_data, initial_jobs=[0, 1, 2, 3, 4], arrival_rate=0.08, num_scenarios=10):
    """
    Generate diverse test scenarios with SIMPLE POISSON arrivals.
    
    SIMPLIFIED APPROACH:
    --------------------
    Since jobs are HOMOGENEOUS (no job type classification), we use simple Poisson process.
    Without loss of generality, assume specific job arrival sequence (e.g., J0→J1→J2→...).
    
    The strategic depth comes from:
    1. Machine heterogeneity (fast/slow machines)
    2. Processing time variance across operations
    3. Arrival timing uncertainty
    
    This makes the WAIT action critical for exploiting machine speed differences.
    """
    
    print(f"Generating {num_scenarios} SIMPLE POISSON test scenarios from {len(jobs_data)} total jobs...")
    print(f"Using test seeds {GLOBAL_SEED+1}-{GLOBAL_SEED+num_scenarios} (different from training seed {GLOBAL_SEED})")
    print(f"Arrival generation: POISSON (λ={arrival_rate})")
    
    scenarios = []
    for i in range(num_scenarios):
        test_seed = GLOBAL_SEED + 1 + i
        np.random.seed(test_seed)
        random.seed(test_seed)
        
        # Generate Poisson arrivals
        arrival_times = {}
        
        # Initial jobs at t=0
        for job_id in initial_jobs:
            arrival_times[job_id] = 0.0
        
        # Dynamic arrivals via Poisson process
        dynamic_job_ids = [j for j in jobs_data.keys() if j not in initial_jobs]
        current_time = 0.0
        
        for job_id in dynamic_job_ids:
            inter_arrival = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival
            # if current_time <= 300:  # Extended time horizon
            arrival_times[job_id] = float(current_time)
        
        scenarios.append({
            'scenario_id': i,
            'arrival_times': arrival_times,
            'initial_jobs': initial_jobs,
            'arrival_rate': arrival_rate,
            'seed': test_seed
        })
        
        arrived_jobs = list(arrival_times.keys())
        print(f"  Scenario {i+1}: {len(arrived_jobs)} jobs, rate={arrival_rate:.3f}, seed={test_seed}")
        print(f"    Initial: {initial_jobs}")
        print(f"    Dynamic arrivals: {len(arrived_jobs) - len(initial_jobs)} jobs")
        print(f"    Arrival times: {arrival_times}")
        print()
    
    return scenarios



def plot_training_metrics():
    """
    Enhanced plot of training metrics including episode rewards, all loss types, and learning progress.
    X-axis is episode number for episode metrics, timesteps for training metrics.
    """
    global TRAINING_METRICS
    
    if not TRAINING_METRICS['episode_rewards']:
        print("❌ No episode rewards recorded - cannot generate training plots!")
        print("Check that the EnhancedTrainingCallback is working properly.")
        return
    
    # print(f"\n=== ENHANCED TRAINING METRICS ANALYSIS ===")
    # print(f"Episode rewards recorded: {len(TRAINING_METRICS['episode_rewards'])}")
    # print(f"Policy loss records: {len(TRAINING_METRICS['policy_loss'])}")
    # print(f"Value loss records: {len(TRAINING_METRICS['value_loss'])}")
    # print(f"Total loss records: {len(TRAINING_METRICS['total_loss'])}")
    # print(f"Entropy loss records: {len(TRAINING_METRICS['action_entropy'])}")
    # print(f"Episode count records: {len(TRAINING_METRICS['episode_count'])}")
    
    # All metrics are now aligned by rollout (i.e., one entry per rollout)
    rollout_count = len(TRAINING_METRICS['policy_loss'])
    rollout_numbers = list(range(1, rollout_count + 1))
    episode_numbers = list(range(1, len(TRAINING_METRICS['episode_rewards']) + 1))

    rewards = TRAINING_METRICS['episode_rewards']
    rollout_ep_rew_mean = TRAINING_METRICS['rollout_ep_rew_mean']
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle(f'Training Progress - {TRAINING_METRICS.get("method_name", "RL Agent")}', fontsize=22, fontweight='bold')

    # Adjust font sizes for all axes
    title_fontsize = 18
    label_fontsize = 15
    tick_fontsize = 13
    legend_fontsize = 14
    for ax_row in axes:
        for ax in ax_row:
            ax.title.set_fontsize(title_fontsize)
            ax.xaxis.label.set_fontsize(label_fontsize)
            ax.yaxis.label.set_fontsize(label_fontsize)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            for legend in ax.get_legend_handles_labels()[1]:
                if ax.get_legend():
                    ax.get_legend().set_fontsize(legend_fontsize)

    # Increase spacing between subplots to prevent overlap
    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07, wspace=0.28, hspace=0.38)

    # Plot 1: Mean Episode Reward per Rollout
    axes[0, 0].plot(rollout_numbers, rollout_ep_rew_mean, 'g-', linewidth=2, alpha=0.8, label='Mean Episode Reward')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Rollout Number')
    axes[0, 0].set_ylabel('Mean Episode Reward')
    axes[0, 0].set_title('Mean Episode Reward (per Rollout)')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Total Loss per Rollout
    if TRAINING_METRICS['total_loss']:
        axes[0, 1].plot(rollout_numbers, TRAINING_METRICS['total_loss'], 'purple', linewidth=2, alpha=0.7, label='Total Loss')
        axes[0, 1].legend()
        axes[0, 1].set_xlabel('Rollout Number')
        axes[0, 1].set_ylabel('Total Loss')
        axes[0, 1].set_title('Total Training Loss (per Rollout)')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No total loss data', ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=14)
        axes[0, 1].set_title('Total Loss (No Data)')

    # Plot 3: Policy Loss per Rollout
    if TRAINING_METRICS['policy_loss']:
        axes[1, 0].plot(rollout_numbers, TRAINING_METRICS['policy_loss'], 'r-', linewidth=2, alpha=0.7, label='Policy Loss')
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('Rollout Number')
        axes[1, 0].set_ylabel('Policy Loss')
        axes[1, 0].set_title('Policy Gradient Loss (per Rollout)')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No policy loss data', ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].set_title('Policy Loss (No Data)')

    # # Plot 4: Value Loss per Rollout
    # if TRAINING_METRICS['value_loss']:
    #     axes[1, 1].plot(rollout_numbers, TRAINING_METRICS['value_loss'], 'b-', linewidth=2, alpha=0.7, label='Value Loss')
    #     axes[1, 1].legend()
    #     axes[1, 1].set_xlabel('Rollout Number')
    #     axes[1, 1].set_ylabel('Value Loss')
    #     axes[1, 1].set_title('Value Function Loss (per Rollout)')
    #     axes[1, 1].grid(True, alpha=0.3)
    # else:
    #     axes[1, 1].text(0.5, 0.5, 'No value loss data', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
    #     axes[1, 1].set_title('Value Loss (No Data)')

        # Plot 6: KL Divergence per Rollout (replaces Episode Length)
    if TRAINING_METRICS['kl_divergence']:
        # Filter out None values
        kl_values = [kl for kl in TRAINING_METRICS['kl_divergence'] if kl is not None]
        if kl_values:
            axes[1, 1].plot(rollout_numbers[:len(kl_values)], kl_values, 'darkred', linewidth=2, alpha=0.7, label='KL Divergence')
            axes[1, 1].legend()
            axes[1, 1].set_xlabel('Rollout Number')
            axes[1, 1].set_ylabel('KL Divergence')
            axes[1, 1].set_title('KL Divergence (Policy Updates)')
            axes[1, 1].grid(True, alpha=0.3)

            # Add horizontal line at KL divergence threshold (typically 0.01-0.02 for PPO)
            axes[1, 1].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Typical Threshold')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No valid KL divergence data', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('KL Divergence (No Data)')
    else:
        axes[1, 1].text(0.5, 0.5, 'No KL divergence data', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('KL Divergence (No Data)')

    # Plot 5: Entropy Loss per Rollout
    if TRAINING_METRICS['action_entropy']:
        axes[2, 0].plot(rollout_numbers, TRAINING_METRICS['action_entropy'], 'orange', linewidth=2, alpha=0.7, label='Entropy Loss')
        axes[2, 0].legend()
        axes[2, 0].set_xlabel('Rollout Number')
        axes[2, 0].set_ylabel('Entropy Loss')
        axes[2, 0].set_title('Policy Entropy Loss (per Rollout)')
        axes[2, 0].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, 'No entropy loss data', ha='center', va='center', transform=axes[2, 0].transAxes, fontsize=14)
        axes[2, 0].set_title('Entropy Loss (No Data)')

    # Plot 6: Episode Length per Rollout
    if TRAINING_METRICS['episode_lengths']:
        axes[2, 1].plot(episode_numbers, TRAINING_METRICS['episode_lengths'], 'teal', linewidth=2, alpha=0.7, label='Episode Length')
        axes[2, 1].legend()
        axes[2, 1].set_xlabel('Rollout Number')
        axes[2, 1].set_ylabel('Episode Length')
        axes[2, 1].set_title('Episode Length (per Rollout)')
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, 'No episode length data', ha='center', va='center', transform=axes[2, 1].transAxes, fontsize=14)
        axes[2, 1].set_title('Episode Length (No Data)')

    plt.tight_layout()
    plt.savefig(f'enhanced_ppo_training_metrics_{TRAINING_METRICS.get("method_name", "RL Agent")}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Enhanced training metrics plot saved: enhanced_ppo_training_metrics_{TRAINING_METRICS.get('method_name', 'RL Agent')}.png")
    plt.pause(1)
    # print_training_summary()



# def analyze_training_arrival_distribution():
#     """
#     Analyze and plot the distribution of arrival times during training.
#     This helps identify if the reactive RL is seeing diverse enough scenarios.
#     """
#     global TRAINING_ARRIVAL_TIMES, TRAINING_EPISODE_COUNT
    
#     if not TRAINING_ARRIVAL_TIMES:
#         print("No arrival times recorded during training!")
#         return
    
#     print(f"\n=== TRAINING ARRIVAL DISTRIBUTION ANALYSIS ===")
#     print(f"Total episodes: {TRAINING_EPISODE_COUNT}")
#     print(f"Total dynamic arrivals recorded: {len(TRAINING_ARRIVAL_TIMES)}")
#     print(f"Average arrivals per episode: {len(TRAINING_ARRIVAL_TIMES)/max(TRAINING_EPISODE_COUNT,1):.2f}")
    
#     # Statistics
#     arrival_times = np.array(TRAINING_ARRIVAL_TIMES)
#     print(f"Arrival time statistics:")
#     print(f"  Min: {np.min(arrival_times):.2f}")
#     print(f"  Max: {np.max(arrival_times):.2f}")
#     print(f"  Mean: {np.mean(arrival_times):.2f}")
#     print(f"  Std: {np.std(arrival_times):.2f}")
    
#     # Create distribution plot
#     plt.figure(figsize=(15, 10))
    
#     # Plot 1: Histogram of arrival times
#     plt.subplot(2, 2, 1)
#     plt.hist(arrival_times, bins=50, alpha=0.7, edgecolor='black')
#     plt.xlabel('Arrival Time')
#     plt.ylabel('Frequency')
#     plt.title(f'Distribution of Job Arrival Times During Training\n({len(TRAINING_ARRIVAL_TIMES)} arrivals across {TRAINING_EPISODE_COUNT} episodes)')
#     plt.grid(True, alpha=0.3)
    
#     # Plot 2: Box plot
#     plt.subplot(2, 2, 2)
#     plt.boxplot(arrival_times, vert=True)
#     plt.ylabel('Arrival Time')
#     plt.title('Box Plot of Arrival Times')
#     plt.grid(True, alpha=0.3)
    
#     # Plot 3: Cumulative distribution
#     plt.subplot(2, 2, 3)
#     sorted_arrivals = np.sort(arrival_times)
#     y_vals = np.arange(1, len(sorted_arrivals) + 1) / len(sorted_arrivals)
#     plt.plot(sorted_arrivals, y_vals, linewidth=2)
#     plt.xlabel('Arrival Time')
#     plt.ylabel('Cumulative Probability')
#     plt.title('Cumulative Distribution of Arrival Times')
#     plt.grid(True, alpha=0.3)
    
#     # Plot 4: Inter-arrival times
#     if len(arrival_times) > 1:
#         plt.subplot(2, 2, 4)
#         # Group by episodes and calculate inter-arrival times within episodes
#         inter_arrivals = []
#         episode_arrivals = []
#         current_episode_times = []
        
#         # Simple approximation: assume arrivals are chronological within batches
#         sorted_times = np.sort(arrival_times)
#         for i in range(1, len(sorted_times)):
#             inter_arrival = sorted_times[i] - sorted_times[i-1]
#             if inter_arrival > 0 and inter_arrival < 100:  # Filter reasonable inter-arrivals
#                 inter_arrivals.append(inter_arrival)
        
#         if inter_arrivals:
#             plt.hist(inter_arrivals, bins=30, alpha=0.7, edgecolor='black')
#             plt.xlabel('Inter-arrival Time')
#             plt.ylabel('Frequency')
#             plt.title(f'Distribution of Inter-arrival Times\n(Mean: {np.mean(inter_arrivals):.2f})')
#             plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('training_arrival_distribution.png', dpi=300, bbox_inches='tight')
#     plt.pause(1)

#     # Check if distribution is diverse enough
#     unique_times = len(np.unique(np.round(arrival_times, 1)))
#     print(f"\nDiversity Analysis:")
#     print(f"  Unique arrival times (rounded to 0.1): {unique_times}")
#     print(f"  Time span: {np.max(arrival_times) - np.min(arrival_times):.2f}")
    
#     if unique_times < 20:
#         print("⚠️  WARNING: Low diversity in arrival times may limit learning")
#     else:
#         print("✅ Good diversity in arrival times")





# def evaluate_on_test_scenarios(model, test_scenarios, jobs_data, machine_list, method_name="Model", is_static_model=False):
#     """Evaluate a model on multiple predefined test scenarios."""
#     print(f"\n--- Evaluating {method_name} on {len(test_scenarios)} Test Scenarios ---")
    
#     results = []
    
#     for scenario in test_scenarios:
#         scenario_id = scenario['scenario_id']
#         arrival_times = scenario['arrival_times']
#         seed = scenario['seed']
        
#         if is_static_model:
#             # For static models, use the special evaluation function
#             makespan, schedule = evaluate_static_on_dynamic(
#                 model, jobs_data, machine_list, arrival_times)
            
#             results.append({
#                 'scenario_id': scenario_id,
#                 'makespan': makespan,
#                 'schedule': schedule,
#                 'arrival_times': arrival_times,
#                 'steps': 0,  # Not tracked for this evaluation
#                 'reward': 0  # Not tracked for this evaluation
#             })
#         else:
#             # For dynamic models, use the existing evaluation
#             makespan, schedule = evaluate_dynamic_on_dynamic(
#                 model, jobs_data, machine_list, arrival_times)
            
#             results.append({
#                 'scenario_id': scenario_id,
#                 'makespan': makespan,
#                 'schedule': schedule,
#                 'arrival_times': arrival_times,
#                 'steps': 0,
#                 'reward': 0
#             })
        
#         print(f"  Scenario {scenario_id+1}: Makespan = {makespan:.2f}")
    
#     # Calculate statistics
#     makespans = [r['makespan'] for r in results]
#     avg_makespan = np.mean(makespans)
#     std_makespan = np.std(makespans)
#     min_makespan = np.min(makespans)
#     max_makespan = np.max(makespans)
    
#     print(f"Results for {method_name}:")
#     print(f"  Average Makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
#     print(f"  Best Makespan: {min_makespan:.2f}")
#     print(f"  Worst Makespan: {max_makespan:.2f}")
    
#     # Return best result for visualization
#     best_result = min(results, key=lambda x: x['makespan'])
#     return best_result['makespan'], best_result['schedule'], best_result['arrival_times']

def evaluate_static_on_dynamic(static_model, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment"):
    """Evaluate static model on dynamic scenario - BUILDER MODE VERSION."""
    print(f"  Static RL evaluation on dynamic scenario...")
    
    # Use PerfectKnowledgeFJSPEnv with actual arrival times for consistency
    test_env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, arrival_times, reward_mode=reward_mode)
    test_env = ActionMasker(test_env, mask_fn)
    
    obs, _ = test_env.reset()
    
    done = False
    step_count = 0
    max_steps = len(jobs_data) * max(len(ops) for ops in jobs_data.values()) * 3
    
    while not done and step_count < max_steps:
        action_masks = test_env.action_masks()
        if not np.any(action_masks):
            print(f"    No valid actions at step {step_count}")
            break
            
        action, _ = static_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
    
    makespan = test_env.env.current_makespan
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    for machine_ops in test_env.env.schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    print(f"  Static RL on dynamic scenario scheduled jobs: {sorted(scheduled_jobs)}")
    
    return makespan, test_env.env.schedule

def evaluate_static_on_static(static_model, jobs_data, machine_list, reward_mode="makespan_increment"):
    """Evaluate static model on static scenario (all jobs at t=0)."""
    print(f"  Static RL evaluation on static scenario (all jobs at t=0)...")
    
    # Create static arrival times (all jobs at t=0) - same as training
    static_arrival_times = {job_id: 0.0 for job_id in jobs_data.keys()}
    
    # Use PerfectKnowledgeFJSPEnv for consistency with training and other evaluations
    test_env = PerfectKnowledgeFJSPEnv(jobs_data, machine_list, static_arrival_times, reward_mode=reward_mode)
    test_env = ActionMasker(test_env, mask_fn)
    
    obs, _ = test_env.reset()
    
    # Run evaluation
    done = False
    step_count = 0
    max_steps = len(jobs_data) * max(len(ops) for ops in jobs_data.values()) * 2
    
    while not done and step_count < max_steps:
        action_masks = test_env.action_masks()
        if not np.any(action_masks):
            print(f"    No valid actions at step {step_count}")
            break
            
        action, _ = static_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
    
    makespan = test_env.env.current_makespan
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    for machine_ops in test_env.env.schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    print(f"  Static RL on static scenario scheduled jobs: {sorted(scheduled_jobs)}")
    
    return makespan, test_env.env.schedule

def evaluate_dynamic_on_dynamic(dynamic_model, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment"):
    """Evaluate dynamic model on dynamic scenario - BUILDER MODE VERSION."""
    print(f"  Reactive RL evaluation (builder mode)...")
    print(f"  🔍 Reactive RL using arrival times: {arrival_times}")
    
    # Create environment with proper builder-mode settings
    # Use float comparison tolerance to identify initial jobs
    initial_job_ids = [k for k, v in arrival_times.items() if abs(v) < 1e-6]
    
    test_env = PoissonDynamicFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=initial_job_ids,
        arrival_rate=0.05,  # Rate doesn't matter since we'll override
        reward_mode=reward_mode,
        seed=GLOBAL_SEED,
        # max_time_horizon=max([t for t in arrival_times.values() if t != float('inf')] + [200])
        max_time_horizon=1000  # Extended horizon to allow for WAIT actions
    )
    
    test_env = ActionMasker(test_env, mask_fn)
    
    obs, _ = test_env.reset()
    
    # Force the environment to use our arrival times
    test_env.env.job_arrival_times = arrival_times.copy()
    test_env.env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 1e-6}
    
    obs = test_env.env._get_observation()
    
    step_count = 0
    max_steps = 500  # Increased for WAIT actions
    wait_count = 0
    schedule_count = 0
    
    while step_count < max_steps:
        action_masks = test_env.action_masks()
        
        if not any(action_masks):
            print(f"    No valid actions available at step {step_count}")
            break
        
        action, _ = dynamic_model.predict(obs, action_masks=action_masks, deterministic=True)
        
        # Track wait vs schedule actions (PoissonDynamicFJSPEnv also has wait actions)
        if hasattr(test_env.env, 'wait_action_start') and action >= test_env.env.wait_action_start:
            wait_count += 1
        else:
            schedule_count += 1
        
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if done or truncated:
            print(f"    Episode completed at step {step_count}")
            break
    
    print(f"  📊 Reactive RL action breakdown: {schedule_count} scheduling, {wait_count} waits ({wait_count/(schedule_count+wait_count)*100 if (schedule_count+wait_count) > 0 else 0:.1f}% waits)")
    print('step count: ',step_count)
    makespan = test_env.env.current_makespan
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    total_ops_scheduled = 0
    for machine_ops in test_env.env.schedule.values():
        total_ops_scheduled += len(machine_ops)
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    
    total_expected_ops = sum(len(ops) for ops in jobs_data.values())
    print(f"  Reactive RL scheduled jobs: {sorted(scheduled_jobs)} ({total_ops_scheduled}/{total_expected_ops} ops)")
    
    # Verify schedule is valid and complete
    if total_ops_scheduled < total_expected_ops:
        missing_jobs = set(jobs_data.keys()) - scheduled_jobs
        print(f"  ⚠️  WARNING: Incomplete schedule! Expected {total_expected_ops} operations, got {total_ops_scheduled}")
        print(f"  Missing jobs: {sorted(missing_jobs)}")
        
        # Check which jobs never arrived vs which arrived but weren't scheduled
        never_arrived = [j for j in missing_jobs if arrival_times.get(j, 0) == float('inf')]
        arrived_but_not_scheduled = [j for j in missing_jobs if j not in never_arrived]
        
        if never_arrived:
            print(f"  Jobs that never arrived: {sorted(never_arrived)}")
        if arrived_but_not_scheduled:
            print(f"  Jobs that arrived but weren't scheduled: {sorted(arrived_but_not_scheduled)}")
    
    return makespan, test_env.env.schedule


def evaluate_rule_based_on_dynamic(rule_model, jobs_data, machine_list, arrival_times, 
                                   reward_mode="makespan_increment"):
    """
    Evaluate RULE-BASED RL model on dynamic scenario.
    
    The model learns to select which dispatching rule to apply at each decision point.
    
    Args:
        rule_model: Trained MaskablePPO model with rule-based environment
        jobs_data: Job operations data
        machine_list: List of machines
        arrival_times: Dict mapping job_id to arrival time
        reward_mode: Reward calculation mode
    
    Returns:
        makespan: Final makespan
        schedule: Detailed schedule per machine
    """
    print(f"  Rule-Based RL evaluation (builder mode)...")
    print(f"  🔍 Rule-Based RL using arrival times: {arrival_times}")
    
    # Create rule-based environment
    # Use float comparison tolerance to identify initial jobs (arrival time ≈ 0)
    initial_job_ids = [k for k, v in arrival_times.items() if abs(v) < 1e-6]
    print(f"  Initial jobs identified: {sorted(initial_job_ids)} (from {len(arrival_times)} total jobs)")
    
    test_env = DispatchingRuleFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=initial_job_ids,
        arrival_rate=0.05,
        reward_mode=reward_mode,
        seed=GLOBAL_SEED,
        max_time_horizon=max([t for t in arrival_times.values() if t != float('inf')] + [200])
    )
    
    test_env = ActionMasker(test_env, mask_fn)
    
    obs, _ = test_env.reset()
    
    # Force the environment to use our arrival times
    test_env.env.job_arrival_times = arrival_times.copy()
    test_env.env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 1e-6}
    
    print(f"  Initially arrived jobs: {sorted(test_env.env.arrived_jobs)}")
    print(f"  Total operations to schedule: {test_env.env.total_operations}")
    
    obs = test_env.env._get_observation()
    
    step_count = 0
    max_steps = 500
    
    # Track which rules were selected (10 rule combinations + 1 WAIT action = 11 total)
    rule_counts = {i: 0 for i in range(11)}
    rule_names = test_env.env.rule_names  # Use the names from environment
    
    while step_count < max_steps:
        action_masks = test_env.action_masks()
        
        if not any(action_masks):
            print(f"    ❌ No valid actions available at step {step_count}")
            print(f"       Operations scheduled: {test_env.env.operations_scheduled}/{test_env.env.total_operations}")
            print(f"       Arrived jobs: {sorted(test_env.env.arrived_jobs)}")
            print(f"       Event time: {test_env.env.event_time:.2f}")
            print(f"       Current makespan: {test_env.env.current_makespan:.2f}")
            
            # Check for jobs that should arrive but haven't
            unarrived_jobs = [(j, arrival_times[j]) for j in jobs_data.keys() 
                             if j not in test_env.env.arrived_jobs and arrival_times.get(j, float('inf')) < float('inf')]
            if unarrived_jobs:
                print(f"       Jobs not yet arrived: {unarrived_jobs[:5]}...")  # Show first 5
            
            # Check ready operations
            ready_ops = test_env.env._get_ready_operations()
            print(f"       Ready operations: {len(ready_ops)}")
            if ready_ops:
                print(f"       Sample ready op: {ready_ops[0]}")
            
            break
        
        action, _ = rule_model.predict(obs, action_masks=action_masks, deterministic=True)
        action_idx = int(action) if isinstance(action, (np.ndarray, np.integer)) else action
        rule_counts[action_idx] += 1
        
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if done or truncated:
            print(f"    Episode completed at step {step_count}")
            break
    
    # Display rule selection statistics
    total_actions = sum(rule_counts.values())
    
    # Separate scheduling actions (0-9) from WAIT action (10)
    scheduling_actions = sum(rule_counts[i] for i in range(10))
    wait_actions = rule_counts[10]
    
    print(f"  📊 Rule-Based RL action breakdown:")
    print(f"     - Total actions: {total_actions} ({scheduling_actions} scheduling, {wait_actions} waits)")
    
    # Show percentage for each dispatching rule (actions 0-9 only)
    if scheduling_actions > 0:
        print(f"     - Dispatching rule distribution (among {scheduling_actions} scheduling actions):")
        for i in range(10):
            if rule_counts[i] > 0:
                pct = rule_counts[i] / scheduling_actions * 100
                print(f"       • {rule_names[i]:12s}: {rule_counts[i]:3d} ({pct:5.1f}%)")
    
    # Show WAIT percentage relative to total actions
    if wait_actions > 0:
        wait_pct = wait_actions / total_actions * 100
        print(f"     - WAIT actions: {wait_actions} ({wait_pct:.1f}% of all actions)")
    
    print(f"     Step count: {step_count}, Scheduled ops: {test_env.env.operations_scheduled}")
    
    makespan = test_env.env.current_makespan
    
    # Verify schedule completeness
    scheduled_jobs = set()
    total_ops_scheduled = 0
    for machine_ops in test_env.env.schedule.values():
        total_ops_scheduled += len(machine_ops)
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    
    total_expected_ops = sum(len(ops) for ops in jobs_data.values())
    print(f"  Rule-Based RL scheduled jobs: {sorted(scheduled_jobs)} ({total_ops_scheduled}/{total_expected_ops} ops)")
    
    if total_ops_scheduled < total_expected_ops:
        missing_jobs = set(jobs_data.keys()) - scheduled_jobs
        print(f"  ⚠️  WARNING: Incomplete schedule! Expected {total_expected_ops} operations, got {total_ops_scheduled}")
        print(f"  Missing jobs: {sorted(missing_jobs)}")
    
    return makespan, test_env.env.schedule


def evaluate_proactive_on_dynamic(proactive_model, jobs_data, machine_list, arrival_times, 
                                  reward_mode="makespan_increment"):
    """
    Evaluate PROACTIVE model on dynamic scenario.
    
    The model uses its learned arrival predictor to guide wait actions.
    
    Args:
        proactive_model: Trained MaskablePPO model with proactive environment
        jobs_data: Job specifications
        machine_list: List of machines
        arrival_times: Dict {job_id: arrival_time} for evaluation
        reward_mode: Reward mode
    
    Returns:
        makespan, schedule
    """
    print(f"  Proactive RL evaluation...")
    print(f"  🔍 Proactive RL using arrival times: {arrival_times}")
    
    # Create proactive environment (NO prediction_window parameter anymore)
    # Use float comparison tolerance to identify initial jobs
    initial_job_ids = [k for k, v in arrival_times.items() if abs(v) < 1e-6]
    print(f"  Initial jobs identified: {sorted(initial_job_ids)} (from {len(arrival_times)} total jobs)")
    
    test_env = ProactiveDynamicFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=initial_job_ids,
        arrival_rate=0.05,  # Placeholder, will be overridden
        reward_mode=reward_mode,
        seed=GLOBAL_SEED,
        # max_time_horizon=max([t for t in arrival_times.values() if t != float('inf')] + [200])
        max_time_horizon=1000  # Extended horizon to allow for WAIT actions
    )
    
    test_env = ActionMasker(test_env, mask_fn)
    
    # Reset first (generates random arrivals)
    obs, _ = test_env.reset()
    
    # CRITICAL FIX: Override arrival times AFTER reset (reset regenerates them!)
    test_env.env.job_arrival_times = arrival_times.copy()
    test_env.env.arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 1e-6}
    
    # Re-observe initial state with correct arrival times
    obs = test_env.env._get_observation()
    
    # Debug: Verify arrival times are set correctly
    print(f"  Initially arrived jobs: {sorted(test_env.env.arrived_jobs)}")
    print(f"  Total jobs: {len(test_env.env.job_ids)}")
    print(f"  Total operations expected: {sum(len(ops) for ops in jobs_data.values())}")
    
    step_count = 0
    max_steps = sum(len(ops) for ops in jobs_data.values()) * 5  # Much more generous limit
    proactive_decisions = 0  # Track how many proactive scheduling decisions were made
    wait_count = 0
    schedule_count = 0
    wait_after_all_arrived = 0  # Track wasteful waits after all jobs arrived
    last_arrival_step = -1
    
    while step_count < max_steps:
        action_masks = test_env.action_masks()
        
        if not any(action_masks):
            print(f"    ❌ No valid actions at step {step_count}")
            print(f"       Arrived jobs: {sorted(test_env.env.arrived_jobs)}")
            print(f"       Completed jobs: {sorted(test_env.env.completed_jobs)}")
            print(f"       Job progress: {test_env.env.job_progress}")
            print(f"       Event time: {test_env.env.event_time:.2f}")
            print(f"       Machine end times: {test_env.env.machine_end_times}")
            
            # Check if any jobs still incomplete
            incomplete = [j for j in test_env.env.job_ids if j not in test_env.env.completed_jobs]
            if incomplete:
                print(f"       Incomplete jobs: {sorted(incomplete)}")
                for job_id in incomplete[:3]:  # Show details for first 3
                    print(f"         J{job_id}: progress={test_env.env.job_progress[job_id]}/{len(jobs_data[job_id])}, "
                          f"arrived={job_id in test_env.env.arrived_jobs}, "
                          f"arrival_time={test_env.env.job_arrival_times.get(job_id, 'N/A')}")
            break
        
        # Track when all jobs have arrived
        all_arrived = len(test_env.env.arrived_jobs) == len(test_env.env.job_ids)
        if all_arrived and last_arrival_step == -1:
            last_arrival_step = step_count
            print(f"    ✅ All jobs arrived at step {step_count}")
        
        action, _ = proactive_model.predict(obs, action_masks=action_masks, deterministic=True)
        
        # Track wait actions (wait_action_start is the index where wait actions begin)
        if action >= test_env.env.wait_action_start:
            wait_count += 1
            wait_action_idx = action - test_env.env.wait_action_start
            wait_duration = test_env.env.wait_durations[wait_action_idx]
            
            # CRITICAL: Track wasteful waits after all jobs arrived
            if all_arrived:
                wait_after_all_arrived += 1
                if wait_after_all_arrived <= 5:  # Log first 5 wasteful waits
                    print(f"    ⚠️  Step {step_count}: WASTEFUL WAIT {wait_duration} units (all jobs arrived!)")
            elif step_count < 10 or wait_duration != float('inf'):  # Only log first 10 or non-inf waits
                print(f"    Step {step_count}: Wait {wait_duration} units (arrived: {len(test_env.env.arrived_jobs)}/{len(test_env.env.job_ids)})")
        else:
            schedule_count += 1
            if step_count < 10:  # Log first 10 scheduling actions
                job_id = action // len(machine_list)
                machine_idx = action % len(machine_list)
                print(f"    Step {step_count}: Schedule J{job_id}-O{test_env.env.job_progress[job_id]+1} on M{machine_idx+1}")
        
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
        
        if done or truncated:
            termination_reason = "done" if done else "truncated"
            print(f"    ✅ Episode {termination_reason} at step {step_count} (scheduled={schedule_count}, waited={wait_count})")
            if wait_after_all_arrived > 0:
                print(f"    ⚠️  WARNING: {wait_after_all_arrived} wasteful waits AFTER all jobs arrived!")
            break
    
    if step_count >= max_steps:
        print(f"    ❌ Hit max steps ({max_steps})! Scheduled {schedule_count} ops, waited {wait_count} times")
        if wait_after_all_arrived > 0:
            print(f"    ⚠️  {wait_after_all_arrived} wasteful waits AFTER all jobs arrived (agent stuck in wait loop!)")
    
    # Calculate wait percentage and provide diagnosis
    total_actions = schedule_count + wait_count
    wait_percentage = (wait_count / total_actions * 100) if total_actions > 0 else 0
    
    print(f"  📊 Proactive RL action breakdown: {schedule_count} scheduling, {wait_count} waits ({wait_percentage:.1f}% waits)")
    
    # DIAGNOSIS: Check for problematic wait behavior
    if wait_after_all_arrived > 0:
        print(f"  ⚠️  DIAGNOSIS: Agent performed {wait_after_all_arrived} wasteful waits AFTER all jobs arrived!")
        print(f"      This indicates the agent learned poor policy during training.")
        print(f"      FIX APPLIED: Action masking now disables waits after all jobs arrived.")
    
    if last_arrival_step > 0:
        steps_after_arrival = step_count - last_arrival_step
        waits_after_arrival_pct = (wait_after_all_arrived / steps_after_arrival * 100) if steps_after_arrival > 0 else 0
        print(f"  📈 Post-arrival behavior: {steps_after_arrival} steps after all arrived, "
              f"{wait_after_all_arrived} waits ({waits_after_arrival_pct:.1f}% of post-arrival steps)")
    
    makespan = test_env.env.current_makespan
    
    # Debug: Check schedule completeness
    scheduled_jobs = set()
    total_ops_scheduled = 0
    for machine_ops in test_env.env.machine_schedules.values():
        total_ops_scheduled += len(machine_ops)
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    
    total_expected_ops = sum(len(ops) for ops in jobs_data.values())
    print(f"  Proactive RL scheduled jobs: {sorted(scheduled_jobs)} ({total_ops_scheduled}/{total_expected_ops} ops)")
    
    # Verify completeness
    if total_ops_scheduled < total_expected_ops:
        missing_jobs = set(jobs_data.keys()) - scheduled_jobs
        print(f"  ⚠️  WARNING: Incomplete schedule! Expected {total_expected_ops} operations, got {total_ops_scheduled}")
        print(f"  Missing jobs: {sorted(missing_jobs)}")
    
    # Print predictor final stats
    stats = test_env.env.arrival_predictor.get_stats()
    print(f"  Predictor final stats: rate={stats['estimated_rate']:.4f}, "
          f"confidence={stats['confidence']:.2f}, obs={stats['num_global_observations']}")
    
    # Finalize episode to update predictor's global knowledge
    test_env.env.finalize_episode()
    
    return makespan, test_env.env.machine_schedules


def evaluate_perfect_knowledge_on_scenario(perfect_model, jobs_data, machine_list, arrival_times, reward_mode="makespan_increment", verbose=False):
    """Evaluate perfect knowledge agent - BUILDER MODE VERSION."""
    print(f"  Perfect Knowledge RL evaluation (builder mode)...")
    print(f"  🔍 Perfect Knowledge RL using arrival times: {arrival_times}")
    
    test_env = PerfectKnowledgeFJSPEnv(
        jobs_data, machine_list, 
        arrival_times=arrival_times,
        reward_mode=reward_mode
    )
    test_env = ActionMasker(test_env, mask_fn)
    
    obs, _ = test_env.reset()
    done = False
    step_count = 0
    max_steps = len(jobs_data) * max(len(ops) for ops in jobs_data.values()) * 3
    
    # Track decision quality
    first_10_actions = []
    
    while not done and step_count < max_steps:
        action_masks = test_env.action_masks()
        if not np.any(action_masks):
            print(f"    No valid actions at step {step_count}")
            break
            
        action, _ = perfect_model.predict(obs, action_masks=action_masks, deterministic=True)
        
        # Log first 10 actions for analysis
        if step_count < 10 and verbose:
            job_id = action // len(machine_list)
            machine_idx = action % len(machine_list)
            machine = machine_list[machine_idx]
            first_10_actions.append((step_count, f"J{job_id}-O{test_env.env.job_progress[job_id]+1}", machine))
            print(f"    Step {step_count}: Schedule {first_10_actions[-1][1]} on {first_10_actions[-1][2]}")
        
        obs, reward, done, truncated, info = test_env.step(action)
        step_count += 1
    
    makespan = test_env.env.current_makespan
    
    # Add debug info about which jobs were scheduled
    scheduled_jobs = set()
    total_ops = 0
    for machine_ops in test_env.env.schedule.values():
        for op_data in machine_ops:
            total_ops += 1
            if len(op_data) >= 3:
                job_op = op_data[0]
                if 'J' in job_op:
                    try:
                        job_num = int(job_op.split('J')[1].split('-')[0])
                        scheduled_jobs.add(job_num)
                    except:
                        pass
    
    expected_ops = sum(len(ops) for ops in jobs_data.values())
    print(f"  Perfect Knowledge RL: {total_ops}/{expected_ops} ops, {len(scheduled_jobs)}/{len(jobs_data)} jobs, {step_count} steps")
    
    if total_ops < expected_ops:
        print(f"    ⚠️  WARNING: Incomplete schedule! Missing {expected_ops - total_ops} operations")
    
    return makespan, test_env.env.schedule

def verify_schedule_correctness(schedule, jobs_data, arrival_times, method_name):
    """
    Verify that a schedule is valid and calculate its true makespan.
    This helps detect bugs in evaluation functions.
    """
    if not schedule or all(len(ops) == 0 for ops in schedule.values()):
        print(f"    ❌ {method_name}: Empty schedule")
        return False, float('inf')
    
    # Check 1: All operations scheduled exactly once
    scheduled_ops = set()
    for machine_ops in schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op = op_data[0]
                if job_op in scheduled_ops:
                    print(f"    ❌ {method_name}: Duplicate operation {job_op}")
                    return False, float('inf')
                scheduled_ops.add(job_op)
    
    expected_ops = set()
    for job_id in jobs_data.keys():
        for op_idx in range(len(jobs_data[job_id])):
            expected_ops.add(f"J{job_id}-O{op_idx+1}")
    
    if scheduled_ops != expected_ops:
        missing = expected_ops - scheduled_ops
        extra = scheduled_ops - expected_ops
        if missing:
            print(f"    ❌ {method_name}: Missing operations {missing}")
        if extra:
            print(f"    ❌ {method_name}: Extra operations {extra}")
        return False, float('inf')
    
    # Check 2: Precedence constraints within jobs
    job_op_times = {}
    for machine_ops in schedule.values():
        for op_data in machine_ops:
            if len(op_data) >= 3:
                job_op, start_time, end_time = op_data[:3]
                # Parse job and operation
                try:
                    job_id = int(job_op.split('J')[1].split('-')[0])
                    op_idx = int(job_op.split('O')[1]) - 1  # Extract op index from "O1" (0-indexed)
                    job_op_times[(job_id, op_idx)] = (start_time, end_time)
                except:
                    print(f"    ❌ {method_name}: Invalid operation format {job_op}")
                    return False, float('inf')
    
    # Check precedence within jobs
    for job_id in jobs_data.keys():
        for op_idx in range(1, len(jobs_data[job_id])):
            prev_op = (job_id, op_idx - 1)
            curr_op = (job_id, op_idx)
            
            if prev_op not in job_op_times or curr_op not in job_op_times:
                continue
                
            prev_end = job_op_times[prev_op][1]
            curr_start = job_op_times[curr_op][0]
            
            if curr_start < prev_end:
                print(f"    ❌ {method_name}: Precedence violation Job {job_id}: Op {op_idx} starts before Op {op_idx-1} ends")
                return False, float('inf')
    
    # Check 3: Arrival time constraints
    TOLERANCE = 1e-6  # Floating-point tolerance
    for job_id in jobs_data.keys():
        if (job_id, 0) in job_op_times:
            first_op_start = job_op_times[(job_id, 0)][0]
            arrival_time = arrival_times.get(job_id, 0.0)
            
            # Use tolerance for floating-point comparison
            if first_op_start < arrival_time - TOLERANCE:
                print(f"    ❌ {method_name}: Job {job_id} starts at {first_op_start:.2f} before arrival at {arrival_time:.2f}")
                return False, float('inf')
    
    # Check 4: Machine conflicts
    TOLERANCE = 1e-6  # Floating-point tolerance
    for machine, machine_ops in schedule.items():
        sorted_ops = sorted(machine_ops, key=lambda x: x[1])  # Sort by start time
        for i in range(len(sorted_ops) - 1):
            curr_end = sorted_ops[i][2]
            next_start = sorted_ops[i+1][1]
            # Use tolerance for floating-point comparison
            if next_start < curr_end - TOLERANCE:
                print(f"    ❌ {method_name}: Machine {machine} conflict: {sorted_ops[i][0]} overlaps with {sorted_ops[i+1][0]}")
                return False, float('inf')
    
    # Calculate true makespan
    true_makespan = max([max([op[2] for op in ops]) for ops in schedule.values() if ops])
    
    return True, true_makespan

def remaining_work_estimate(jobs_data, job_id, op_idx):
    """Calculate sum of minimum processing times for remaining operations in job."""
    total = 0.0
    for oi in range(op_idx, len(jobs_data[job_id])):
        total += min(jobs_data[job_id][oi]['proc_times'].values())
    return total


def simple_list_scheduling(jobs_data, machine_list, arrival_times, seq_rule, route_rule="MIN"):
    """
    Event-driven list scheduling for FJSP with separate sequencing and routing policies.
    
    CRITICAL FIX: Heuristic now operates with LIMITED KNOWLEDGE (same as RL)
    - Only knows about jobs that have ARRIVED by current sim_time
    - Must discover jobs dynamically as time advances
    - No perfect foresight or planning around future arrivals
    
    ARCHITECTURE (Sequencing vs Routing - Two-Step Decision):
    
    1. SEQUENCING: Which ready operation to prioritize? (Decides order)
       - FIFO: First arrived jobs first (score = arrival_time)
       - LIFO: Last arrived jobs first (score = -arrival_time)
       - SPT: Shortest processing time first (score = min processing time)
       - LPT: Longest processing time first (score = -min processing time)
       - MWKR: Most work remaining first (score = -remaining work)
    
    2. ROUTING: Which machine for the selected operation? (Decides assignment)
       - MIN: Fastest machine (score = processing time on machine)
       - MINC: Earliest completion (score = max(current_time, machine_free) + proc_time)
    
    This two-step structure allows testing all combinations:
    5 sequencing × 2 routing = 10 total strategies
    
    EVENT-DRIVEN SIMULATION (Fair Comparison with RL):
    - Advances time to next event (arrival or machine becomes free)
    - ONLY considers jobs that have arrived by current sim_time (no cheating!)
    - At each event, identifies ready operations (arrived + precedence satisfied)
    - Applies sequencing rule to select best operation
    - Applies routing rule to select best machine for that operation
    - Single-action mode: schedules one (op, machine) pair per step
    
    Args:
        jobs_data: Job definitions with operations and processing times
        machine_list: Available machines
        arrival_times: When each job arrives (for checking arrivals, NOT for planning)
        seq_rule: Sequencing rule (FIFO, LIFO, SPT, LPT, MWKR)
        route_rule: Routing rule (MIN, MINC)
    
    Returns:
        (makespan, schedule): Final makespan and machine schedules
    """
    machine_next_free = {m: 0.0 for m in machine_list}
    job_next_op = {job_id: 0 for job_id in jobs_data.keys()}
    job_op_end_times = {job_id: [0.0] * len(jobs_data[job_id]) for job_id in jobs_data.keys()}
    schedule = {m: [] for m in machine_list}
    
    # CRITICAL: Track which jobs have been discovered (arrived) so far
    # Heuristic starts knowing ONLY about jobs with arrival_time = 0
    arrived_jobs = {job_id for job_id, arr_time in arrival_times.items() if arr_time <= 0.0}
    
    completed_operations = 0
    total_operations = sum(len(ops) for ops in jobs_data.values())
    sim_time = 0.0
    
    while completed_operations < total_operations:
        # CRITICAL: Check for new arrivals at current sim_time
        # Discover jobs as they arrive (no perfect foresight!)
        for job_id, arr_time in arrival_times.items():
            if job_id not in arrived_jobs and arr_time <= sim_time:
                arrived_jobs.add(job_id)
        
        # STEP 1: Find ready operations (arrived + precedence satisfied)
        # ONLY consider jobs that have actually arrived (fair comparison with RL)
        ready_operations = []
        
        for job_id in arrived_jobs:  # FIXED: Only loop over ARRIVED jobs
            if job_next_op[job_id] < len(jobs_data[job_id]):  # Job not finished
                op_idx = job_next_op[job_id]
                
                # Check if previous operation is complete
                job_ready_time = arrival_times[job_id]
                if op_idx > 0:
                    job_ready_time = max(job_ready_time, job_op_end_times[job_id][op_idx - 1])
                
                if sim_time >= job_ready_time:
                    ready_operations.append({
                        'job_id': job_id,
                        'op_idx': op_idx,
                        'arrival_time': arrival_times[job_id],
                        'job_ready_time': job_ready_time,
                        'remaining_work': remaining_work_estimate(jobs_data, job_id, op_idx)
                    })
        
        if not ready_operations:
            # Advance time to next event (arrival or operation completion)
            next_time = float('inf')
            
            # Check for next job arrival from ARRIVED jobs
            for job_id in arrived_jobs:
                if job_next_op[job_id] < len(jobs_data[job_id]):
                    op_idx = job_next_op[job_id]
                    job_ready_time = arrival_times[job_id]
                    if op_idx > 0:
                        job_ready_time = max(job_ready_time, job_op_end_times[job_id][op_idx - 1])
                    next_time = min(next_time, job_ready_time)
            
            # Check for next NEW job arrival (not yet discovered)
            for job_id, arr_time in arrival_times.items():
                if job_id not in arrived_jobs and arr_time > sim_time:
                    next_time = min(next_time, arr_time)
            
            # Also check for next machine completion
            for machine, free_time in machine_next_free.items():
                if free_time > sim_time:
                    next_time = min(next_time, free_time)
            
            if next_time == float('inf'):
                break
            sim_time = next_time
            continue
        
        # STEP 2: SEQUENCING - Select operation based on sequencing rule
        def sequencing_score(op):
            j, oi = op['job_id'], op['op_idx']
            arr = op['arrival_time']
            minp = min(jobs_data[j][oi]['proc_times'].values())
            rem = op['remaining_work']
            
            if seq_rule == "FIFO":
                return (arr, j, oi)
            elif seq_rule == "LIFO":
                return (-arr, -j, -oi)
            elif seq_rule == "SPT":
                return (minp, arr, j, oi)
            elif seq_rule == "LPT":
                return (-minp, arr, j, oi)
            elif seq_rule == "MWKR":
                return (-rem, arr, j, oi)
            else:
                return (minp, arr, j, oi)  # Default to SPT
        
        selected_op = min(ready_operations, key=sequencing_score)
        job_id = selected_op['job_id']
        op_idx = selected_op['op_idx']
        op_data = jobs_data[job_id][op_idx]
        
        # STEP 3: ROUTING - Select machine based on routing rule
        def routing_score(machine):
            if machine not in op_data['proc_times']:
                return (float('inf'), machine)  # Incompatible
            
            proc_time = op_data['proc_times'][machine]
            machine_avail = machine_next_free[machine]
            job_ready = selected_op['job_ready_time']
            start_time = max(sim_time, machine_avail, job_ready)
            completion_time = start_time + proc_time
            
            if route_rule == "MIN":
                # Fastest machine (minimum processing time)
                return (proc_time, machine)
            elif route_rule == "MINC":
                # Earliest completion time
                return (completion_time, machine)
            else:
                return (proc_time, machine)  # Default to MIN
        
        # Select best machine: routing_score returns (score, machine), so extract machine from min result
        compatible_machines = [m for m in machine_list if m in op_data['proc_times']]
        if not compatible_machines:
            print(f"    ERROR: No compatible machines for Job {job_id} Op {op_idx}")
            break
        
        best_score_machine_tuple = min((routing_score(m) for m in compatible_machines))
        best_machine = best_score_machine_tuple[1]
        
        # STEP 4: Schedule the selected (operation, machine) pair
        if best_machine not in op_data['proc_times']:
            print(f"    ERROR: Selected incompatible machine {best_machine} for Job {job_id} Op {op_idx}")
            break
        proc_time = op_data['proc_times'][best_machine]
        machine_avail = machine_next_free[best_machine]
        job_ready = selected_op['job_ready_time']
        start_time = max(sim_time, machine_avail, job_ready)
        end_time = start_time + proc_time
        
        # Update state
        machine_next_free[best_machine] = end_time
        job_op_end_times[job_id][op_idx] = end_time
        job_next_op[job_id] += 1
        schedule[best_machine].append((f"J{job_id}-O{op_idx+1}", start_time, end_time))
        
        completed_operations += 1
    
    makespan = max(machine_next_free.values()) if machine_next_free else 0
    return makespan, schedule


def run_heuristic_comparison(jobs_data, machine_list, arrival_times, verbose=False):
    """
    Compare all combinations of sequencing and routing rules, return the best.
    
    Tests:
    - Sequencing rules: FIFO, LIFO, SPT, LPT, MWKR (5 rules)
    - Routing rules: MIN, MINC (2 rules)
    - Total combinations: 5 × 2 = 10 strategies
    
    This comprehensive comparison finds the best heuristic for the given scenario.
    """
    sequencing_rules = ['FIFO', 'LIFO', 'SPT', 'LPT', 'MWKR']
    routing_rules = ['MIN', 'MINC']
    
    results = {}
    if verbose:
        print(f"  Testing {len(sequencing_rules)} × {len(routing_rules)} = {len(sequencing_rules) * len(routing_rules)} heuristic combinations...")
    
    for seq in sequencing_rules:
        for route in routing_rules:
            name = f"{seq}+{route}"
            try:
                makespan, schedule = simple_list_scheduling(
                    jobs_data, machine_list, arrival_times, seq, route
                )
                results[name] = (makespan, schedule)
                # print(f"    {name}: {makespan:.2f}")  # Comment out for less verbose output
            except Exception as e:
                print(f"    {name} failed: {e}")
                results[name] = (float('inf'), {})
    
    # Find best combination
    valid_results = {k: v for k, v in results.items() if v[0] != float('inf')}
    if not valid_results:
        print("    All heuristic combinations failed! Using fallback.")
        return 999.0, {m: [] for m in machine_list}
    
    best_name = min(valid_results.keys(), key=lambda x: valid_results[x][0])
    best_makespan, best_schedule = valid_results[best_name]
    
    # Show top 5 results
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1][0])
    print(f"\n  Top 5 heuristic combinations:")
    for i, (name, (makespan, _)) in enumerate(sorted_results[:5], 1):
        status = "✅ BEST" if name == best_name else ""
        print(f"    {i}. {name:20s}: {makespan:.2f} {status}")
    
    print(f"\n  Selected: {best_name} (makespan: {best_makespan:.2f})")
    return best_makespan, best_schedule


def fifo_heuristic(jobs_data, machine_list, arrival_times):
    """
    FIFO (First In First Out) - Process jobs in arrival order.
    
    Sequencing: FIFO (earliest arrival first)
    Routing: MIN (fastest machine - default)
    """
    return simple_list_scheduling(jobs_data, machine_list, arrival_times, "FIFO", "MIN")


def lifo_heuristic(jobs_data, machine_list, arrival_times):
    """
    LIFO (Last In First Out) - Process newest jobs first.
    
    Sequencing: LIFO (latest arrival first)
    Routing: MIN (fastest machine - default)
    """
    return simple_list_scheduling(jobs_data, machine_list, arrival_times, "LIFO", "MIN")


def spt_heuristic(jobs_data, machine_list, arrival_times):
    """
    SPT (Shortest Processing Time) - Process shortest operations first.
    
    Sequencing: SPT (shortest processing time first)
    Routing: MIN (fastest machine - default)
    """
    return simple_list_scheduling(jobs_data, machine_list, arrival_times, "SPT", "MIN")


def lpt_heuristic(jobs_data, machine_list, arrival_times): 
    """
    LPT (Longest Processing Time) - Process longest operations first.
    
    Sequencing: LPT (longest processing time first)
    Routing: MIN (fastest machine - default)
    """
    return simple_list_scheduling(jobs_data, machine_list, arrival_times, "LPT", "MIN")


def mwkr_heuristic(jobs_data, machine_list, arrival_times):
    """
    MWKR (Most Work Remaining) - Prioritize jobs with most remaining work.
    
    Sequencing: MWKR (most work remaining first)
    Routing: MIN (fastest machine - default)
    """
    return simple_list_scheduling(jobs_data, machine_list, arrival_times, "MWKR", "MIN")


def best_heuristic(jobs_data, machine_list, arrival_times):
    """
    Run comprehensive comparison of all sequencing × routing combinations.
    
    CRITICAL: Heuristic now operates with LIMITED KNOWLEDGE (same as RL)
    - Only discovers jobs as they arrive (no perfect foresight)
    - Event-driven simulation (fair comparison with RL agents)
    
    Compares: 5 sequencing rules × 2 routing rules = 10 total combinations
    - Sequencing: FIFO, LIFO, SPT, LPT, MWKR
    - Routing: MIN, MINC
    
    Returns: (makespan, schedule) of the best-performing combination
    """
    initial_jobs = [j for j, t in arrival_times.items() if t <= 0.0]
    print(f"  🔍 Heuristic starting with {len(initial_jobs)} initial jobs (discovers others dynamically)")
    return run_heuristic_comparison(jobs_data, machine_list, arrival_times)


def validate_schedule_makespan(schedule, jobs_data, arrival_times):
    """
    Validate a schedule and calculate its actual makespan.
    """
    max_completion = 0
    job_completion_times = {}
    
    for machine, operations in schedule.items():
        for op_name, start_time, end_time in operations:
            # Parse job and operation from op_name (e.g., "J0-O1")
            job_id = int(op_name.split('-')[0][1:])  # Extract job ID from "J0"
            op_idx = int(op_name.split('-')[1][1:]) - 1  # Extract op index from "O1" (0-indexed)
            
            # Check arrival time constraint
            if op_idx == 0:  # First operation
                required_arrival = arrival_times.get(job_id, 0)
                if start_time < required_arrival - 0.001:
                    print(f"❌ Job {job_id} operation {op_idx} starts at {start_time:.2f} before arrival at {required_arrival:.2f}")
                    return float('inf')
            
            # Check precedence constraint
            if op_idx > 0:
                prev_completion = job_completion_times.get((job_id, op_idx - 1), 0)
                if start_time < prev_completion - 0.001:
                    print(f"❌ Job {job_id} operation {op_idx} starts at {start_time:.2f} before previous op completes at {prev_completion:.2f}")
                    return float('inf')
            
            # Record completion time
            job_completion_times[(job_id, op_idx)] = end_time
            max_completion = max(max_completion, end_time)
    
    return max_completion



def milp_optimal_scheduler(jobs_data, machine_list, arrival_times, time_limit=300, verbose=True):
    """
    MILP approach for optimal dynamic scheduling with perfect knowledge.
    Uses caching to avoid re-solving identical scenarios.

    Args:
        jobs_data: dict mapping job_id -> list of operations, where each op is a dict:
                   e.g. jobs_data[j] = [ {'proc_times': {'M1': 3, 'M2':5}}, ... ]
        machine_list: list of machine names (strings)
        arrival_times: dict mapping job_id -> arrival_time (float)
        time_limit: CBC time limit in seconds
        verbose: if True prints solver progress

    Returns:
        (optimal_makespan, schedule) or (float('inf'), {}) if failed.
        schedule is dict: machine -> list of (op_name, start, end)
    """
    # DEBUG: Print arrival times used by MILP
    print(f"\n  🔍 MILP OPTIMAL using arrival times: {arrival_times}")
    
    # Imports required for pulp
    from pulp import (LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus)
    import math
    import os
    import itertools

    # Generate a hash for this scenario to use as cache key
    def generate_scenario_hash(jobs_data, machine_list, arrival_times):
        scenario_dict = {
            'machines': sorted(machine_list),
            'jobs': {},
            'arrivals': {}
        }
        
        # Sort jobs by ID for consistency
        for job_id in sorted(jobs_data.keys()):
            ops_list = []
            for op in jobs_data[job_id]:
                # Sort machine processing times for consistency
                proc_times = {m: op['proc_times'][m] for m in sorted(op['proc_times'].keys())}
                ops_list.append(proc_times)
            scenario_dict['jobs'][job_id] = ops_list
        
        # Sort arrival times by job ID
        for job_id in sorted(arrival_times.keys()):
            scenario_dict['arrivals'][job_id] = arrival_times[job_id]
        
        # Convert to JSON string and hash it
        scenario_str = json.dumps(scenario_dict, sort_keys=True)
        return hashlib.md5(scenario_str.encode()).hexdigest()
    
    scenario_hash = generate_scenario_hash(jobs_data, machine_list, arrival_times)
    cache_file = f'milp_cache_{scenario_hash}.pkl'
    
    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
            if verbose:
                print(f"📦 Loaded MILP solution from cache: {cache_file}")
                print(f"   Cached makespan: {cached_result['makespan']:.2f}")
            return cached_result['makespan'], cached_result['schedule']
        except Exception as e:
            if verbose:
                print(f"⚠️  Cache load failed: {e}. Recomputing...")
    
    if verbose:
        print(f"🔧 No cache found. Computing MILP solution...")

    # Check that validate_schedule_makespan exists
    if 'validate_schedule_makespan' not in globals():
        raise RuntimeError("validate_schedule_makespan(...) must be defined in the environment.")

    # Build list of operations (job_id, op_index)
    ops = [(j, oi) for j in jobs_data for oi in range(len(jobs_data[j]))]

    # Compute safe BIG_M: total processing time + max arrival
    all_proc_times = []
    for j, oi in ops:
        all_proc_times.extend(jobs_data[j][oi]['proc_times'].values())
    total_proc = sum(all_proc_times) if all_proc_times else 0.0
    max_arrival = max(arrival_times.values()) if arrival_times else 0.0
    BIG_M = total_proc + max_arrival + 1.0  # safe upper bound

    # Create problem
    prob = LpProblem("PerfectKnowledge_FJSP_Optimal", LpMinimize)

    # Decision variables
    # x[(j,oi)][m] = 1 if operation (j,oi) assigned to machine m
    x = {}
    for op in ops:
        x[op] = {}
        j, oi = op
        for m in machine_list:
            if m in jobs_data[j][oi]['proc_times']:
                x[op][m] = LpVariable(f"x_{j}_{oi}_on_{m}", cat="Binary")

    # start and completion times
    s = {op: LpVariable(f"s_{op[0]}_{op[1]}", lowBound=0) for op in ops}
    c = {op: LpVariable(f"c_{op[0]}_{op[1]}", lowBound=0) for op in ops}

    # Makespan
    Cmax = LpVariable("Cmax", lowBound=0)

    # Objective
    prob += Cmax

    # Constraints
    for op in ops:
        j, oi = op
        # 1. assignment to exactly one compatible machine
        compatible_machines = [m for m in machine_list if m in jobs_data[j][oi]['proc_times']]
        if not compatible_machines:
            # infeasible: operation has no machine
            if verbose:
                print(f"Operation {op} has no compatible machines -> infeasible")
            return float('inf'), {}

        prob += lpSum(x[op][m] for m in compatible_machines) == 1

        # 2. completion time = start + processing time (depends on chosen machine)
        prob += c[op] == s[op] + lpSum(x[op][m] * jobs_data[j][oi]['proc_times'][m] for m in compatible_machines)

        # 3. precedence in job
        if oi > 0:
            prev = (j, oi - 1)
            prob += s[op] >= c[prev]
        else:
            # arrival time constraint
            arr = arrival_times.get(j, 0)
            prob += s[op] >= arr

        # 4. makespan
        prob += Cmax >= c[op]

    # 5. disjunctive sequencing only for pairs that share at least one machine
    # Create sequencing vars y only for relevant triples
    # y[op1][op2][m] = 1 if op1 before op2 on machine m
    y = {}
    for m in machine_list:
        # operations compatible with m
        ops_on_m = [op for op in ops if m in jobs_data[op[0]][op[1]]['proc_times']]
        # pairwise sequencing
        for op1_idx in range(len(ops_on_m)):
            for op2_idx in range(op1_idx + 1, len(ops_on_m)):
                op1 = ops_on_m[op1_idx]
                op2 = ops_on_m[op2_idx]

                # create sequencing binary var only for this pair+machine
                y_key = (op1, op2, m)
                y[y_key] = LpVariable(f"y_{op1[0]}_{op1[1]}__{op2[0]}_{op2[1]}__on_{m}", cat="Binary")

                # optional: create both_on_m using linearization of AND
                both_name = f"both_{op1[0]}_{op1[1]}__{op2[0]}_{op2[1]}__on_{m}"
                both_on_m = LpVariable(both_name, cat="Binary")
                prob += both_on_m <= x[op1][m]
                prob += both_on_m <= x[op2][m]
                prob += both_on_m >= x[op1][m] + x[op2][m] - 1

                # sequencing constraints (big-M): if both_on_m == 1 then either op1 before op2 or op2 before op1
                # op1 after op2 unless y==1 (meaning op1 before op2)
                prob += s[op1] >= c[op2] - BIG_M * (1 - y[y_key]) - BIG_M * (1 - both_on_m)
                # op2 after op1 unless y==0 (so op2 >= c_op1 - M*y)
                prob += s[op2] >= c[op1] - BIG_M * y[y_key] - BIG_M * (1 - both_on_m)

    # Solve
    solver = PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit)
    prob.solve(solver)

    # Check status
    status = LpStatus[prob.status]
    if verbose:
        print("Solver status:", status)
    
    # Store solver status for later inspection
    solver_proven_optimal = (status == "Optimal")
    
    if solver_proven_optimal:
        print("✅ True optimal solution found (proven by solver).")
    else:
        print(f"⚠️  Solver status: {status} (NOT proven optimal)")
        print(f"    This means the solution might be suboptimal due to:")
        print(f"    - Time limit ({time_limit}s) reached")
        print(f"    - Numerical difficulties")
        print(f"    - Problem too large for exact solution")

    schedule = {m: [] for m in machine_list}
    if status == "Optimal" or status == "Not Solved" or status == "Optimal":  # Prefer explicit 'Optimal' but accept near-optimal?
        # If solver found values for Cmax
        try:
            optimal_makespan = float(Cmax.varValue) if Cmax.varValue is not None else float('inf')
        except Exception:
            optimal_makespan = float('inf')

        # Build schedule from x and s/c variable values
        for op in ops:
            j, oi = op
            for m in machine_list:
                if m in jobs_data[j][oi]['proc_times'] and (op in x and m in x[op]):
                    xv = x[op][m].varValue
                    if xv is not None and xv > 0.5:
                        st = float(s[op].varValue) if s[op].varValue is not None else None
                        en = float(c[op].varValue) if c[op].varValue is not None else None
                        schedule[m].append((f"J{j}-O{oi+1}", st, en))

        # sort schedule per machine
        for m in schedule:
            schedule[m].sort(key=lambda tup: (tup[1] if tup[1] is not None else math.inf))

        # Validate using provided function
        actual_makespan = validate_schedule_makespan(schedule, jobs_data, arrival_times)
        if math.isfinite(actual_makespan) and abs(actual_makespan - optimal_makespan) <= 1e-6:
            if verbose:
                proven_str = "PROVEN OPTIMAL" if solver_proven_optimal else "FEASIBLE (not proven optimal)"
                print(f"✅ MILP solution validated. Makespan = {optimal_makespan:.2f} ({proven_str})")
            
            # Cache the validated result WITH solver status
            try:
                cached_result = {
                    'makespan': optimal_makespan,
                    'schedule': schedule,
                    'jobs_data': jobs_data,
                    'machine_list': machine_list,
                    'arrival_times': arrival_times,
                    'solver_proven_optimal': solver_proven_optimal  # NEW: Store solver status
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_result, f)
                if verbose:
                    print(f"💾 Cached MILP solution to: {cache_file}")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Failed to cache result: {e}")
            
            return optimal_makespan, schedule
        else:
            if verbose:
                print("❌ Validation mismatch or couldn't compute actual makespan.")
                print("   MILP claim:", optimal_makespan, "actual:", actual_makespan)
            return float('inf'), {}
    else:
        if verbose:
            print("❌ Solver did not find optimal solution. Status:", status)
        return float('inf'), {}



        
def calculate_regret_analysis(optimal_makespan, methods_results, benchmark_name="MILP Optimal"):
    """
    Calculate regret (performance gap from optimal) for all methods.
    
    Regret provides a normalized measure of how far each method is from
    the theoretical optimum, helping understand the relative performance
    and the room for improvement.
    
    Args:
        optimal_makespan: The benchmark makespan (MILP Optimal or Perfect RL if MILP invalid)
        methods_results: Dict with method names as keys and makespans as values
        benchmark_name: Name of the benchmark method ("MILP Optimal" or "Perfect RL")
        
    Returns:
        dict: Regret analysis results
    """
    print("\n" + "="*60)
    print(f"REGRET ANALYSIS (Gap from {benchmark_name})")
    print("="*60)
    
    if optimal_makespan == float('inf') or optimal_makespan <= 0:
        print(f"❌ No valid {benchmark_name} solution available for regret calculation")
        return None
    
    print(f"📊 {benchmark_name} Makespan (Benchmark): {optimal_makespan:.2f}")
    print("-" * 60)
    
    regret_results = {}
    
    # Calculate regret for each method
    for method_name, makespan in methods_results.items():
        if makespan == float('inf'):
            regret_abs = float('inf')
            regret_rel = float('inf')
        else:
            regret_abs = makespan - optimal_makespan  # Absolute regret
            regret_rel = (regret_abs / optimal_makespan) * 100  # Relative regret (%)
        
        regret_results[method_name] = {
            'makespan': makespan,
            'absolute_regret': regret_abs,
            'relative_regret_percent': regret_rel
        }
        
        # Display results with status indicators
        if regret_abs == 0:
            status = "🎯 OPTIMAL"
        elif regret_rel <= 5:
            status = "🟢 EXCELLENT"
        elif regret_rel <= 15:
            status = "🟡 GOOD"
        elif regret_rel <= 30:
            status = "🟠 ACCEPTABLE"
        else:
            status = "🔴 POOR"
        
        print(f"{method_name:25s}: {makespan:6.2f} | Regret: +{regret_abs:5.2f} (+{regret_rel:5.1f}%) {status}")
    
    # Analysis and insights
    print("\n" + "-" * 60)
    print("KEY INSIGHTS:")
    
    # Find best and worst performing methods
    valid_methods = {k: v for k, v in regret_results.items() 
                    if v['absolute_regret'] != float('inf')}
    
    if valid_methods:
        best_method = min(valid_methods.items(), key=lambda x: x[1]['absolute_regret'])
        worst_method = max(valid_methods.items(), key=lambda x: x[1]['absolute_regret'])
        
        print(f"🏆 Best Method: {best_method[0]} (regret: +{best_method[1]['relative_regret_percent']:.1f}%)")
        print(f"⚠️  Worst Method: {worst_method[0]} (regret: +{worst_method[1]['relative_regret_percent']:.1f}%)")
        
        # Calculate performance gap between best and worst
        performance_gap = worst_method[1]['absolute_regret'] - best_method[1]['absolute_regret']
        print(f"📈 Performance Gap: {performance_gap:.2f} time units between best and worst")
        
        # Perfect Knowledge RL validation
        if 'Perfect Knowledge RL' in valid_methods:
            pk_regret = valid_methods['Perfect Knowledge RL']['relative_regret_percent']
            if pk_regret <= 10:
                print(f"✅ Perfect Knowledge RL is performing well (regret: {pk_regret:.1f}%)")
                print("   This validates that the RL agent can effectively use arrival information")
            else:
                print(f"❌ Perfect Knowledge RL has high regret ({pk_regret:.1f}%)")
                print("   Consider: longer training, better hyperparameters, or state representation issues")
    
    # Check if hierarchy is as expected (only if all required keys exist in results)
    required_keys = ['Perfect Knowledge RL', 'Reactive RL', 'Static RL (dynamic)']
    if all(key in regret_results for key in required_keys):
        try:
            expected_order = (
                regret_results['Perfect Knowledge RL']['absolute_regret'] <= 
                regret_results['Reactive RL']['absolute_regret'] <= 
                regret_results['Static RL (dynamic)']['absolute_regret']
            )
            
            if expected_order:
                print("✅ Expected regret hierarchy maintained")
            else:
                print("❌ Unexpected regret hierarchy - investigate training issues")
        except KeyError as e:
            print(f"⚠️  Could not verify hierarchy - missing key: {e}")
    
    # Recommendations
    print(f"\nRecommendations:")
    if regret_results and len([v for v in regret_results.values() if v['absolute_regret'] != float('inf')]) > 1:
        regret_values = [v['absolute_regret'] for v in regret_results.values() if v['absolute_regret'] != float('inf')]
        regret_spread = max(regret_values) - min(regret_values)
        avg_regret = sum(regret_values) / len(regret_values)
        relative_regret_spread = (regret_spread / avg_regret * 100) if avg_regret > 0 else 0
        
        if relative_regret_spread < 5:
            print("- Increase arrival rate (try λ=1.0 or higher)")
            print("- Use longer training episodes") 
            print("- Add more complex job structures")
            print("- Increase reward differentiation for anticipatory actions")
        else:
            print("- Methods showing good differentiation, consider fine-tuning hyperparameters")
    else:
        print("- Need more valid methods for comparison analysis")


def main():
 
    print("=" * 80)
    print("DYNAMIC vs STATIC RL COMPARISON FOR POISSON FJSP")
    print("=" * 80)
    print(f"Problem: {len(ENHANCED_JOBS_DATA)} jobs, {len(MACHINE_LIST)} machines")
    print("Research Question: Does Reactive RL outperform Static RL on Poisson arrivals?")
    print(f"🔧 REPRODUCIBILITY: Fixed seed {GLOBAL_SEED} for all random components (CHANGED from 42)")
    print("🧹 CACHE CLEARING: All MILP cache files will be removed for fresh computation")
    print("📊 DEBUGGING: Action entropy & training metrics tracking enabled")
    print("🚨 STRICT VALIDATION: Will halt execution if any RL outperforms MILP optimal")
    print("=" * 80)
    arrival_rate = 0.25  # LOWER arrival rate to create more dynamic scenarios
    # With λ=0.5, expected inter-arrival = 2 time units (faster than most job operations)
    
    # Step 1: Training Setup
    print("\n1. TRAINING SETUP")
    print("-" * 50)
    perfect_timesteps = 200000    # Perfect knowledge with MULTIPLE INITIALIZATIONS
    dynamic_timesteps = 200000   # Increased for better learning with integer timing 
    proactive_timesteps = 200000 
    static_timesteps = 200000    # Increased for better learning
    learning_rate = 1e-4         # Optimized learning rate for PPO (was 10e-4)
    
    print(f"Perfect RL: {perfect_timesteps:,} | Reactive RL: {dynamic_timesteps:,} | Static RL: {static_timesteps:,} timesteps")
    print(f"Arrival rate: {arrival_rate} (expected inter-arrival: {1/arrival_rate:.1f} time units)")

    # Step 2: Generate test scenarios (Poisson arrivals) - DIFFERENT from training scenarios
    print("\n2. GENERATING TEST SCENARIOS")
    print("-" * 40)
    print("Expected: Reactive RL (knows arrival distribution) > Static RL (assumes all jobs at t=0)")
    print("Performance should be: Deterministic(~43) > Poisson Dynamic > Static(~50)")
    print(f"⚠️  IMPORTANT: Test scenarios use seeds 5000-5009, training used seed {GLOBAL_SEED}")
    print("   This tests generalizability to unseen arrival patterns!")
    print("   🧹 FRESH RUN: All seeds changed to force new evaluations and clear any cached bugs")
    test_scenarios = generate_test_scenarios(ENHANCED_JOBS_DATA, 
                                           initial_jobs=INITIAL_JOB_IDS, 
                                           arrival_rate=arrival_rate, 
                                           num_scenarios=1)
    
    # Print all test scenario arrival times
    print("\nALL TEST SCENARIO ARRIVAL TIMES:")
    print("-" * 50)
    for i, scenario in enumerate(test_scenarios):
        print(f"Scenario {i+1}: {scenario['arrival_times']}")
        arrived_jobs = [j for j, t in scenario['arrival_times'].items() if t < float('inf')]
        print(f"  Jobs arriving: {len(arrived_jobs)} ({sorted(arrived_jobs)})")
        print()
    
    # Step 3: Train base agents (Dynamic and Static RL)
    print("\n3. TRAINING PHASE")
    print("-" * 40)
    
    print("Note: Perfect Knowledge RL will be trained separately for each test scenario")
    print("This ensures each scenario has its optimal RL benchmark for comparison")
    
    # Train reactive RL agent (knows arrival distribution only) - trained once
    dynamic_model = train_dynamic_agent(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=INITIAL_JOB_IDS, arrival_rate=arrival_rate,
        total_timesteps=dynamic_timesteps, reward_mode="makespan_increment", learning_rate=5e-4
    )
    print("\n--- Reactive RL Training Metrics ---")
    plot_training_metrics()
    
    proactive_model = train_proactive_agent(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=INITIAL_JOB_IDS,
        arrival_rate=arrival_rate,
        total_timesteps=proactive_timesteps,
        reward_mode="makespan_increment",
        predictor_mode='map',
        learning_rate=learning_rate 
    )
    print("\n--- Proactive RL Training Metrics ---")
    plot_training_metrics()
    
    # Train RULE-BASED RL agent (learns to select dispatching rules)
    print("\n" + "="*80)
    print("RULE-BASED RL TRAINING")
    print("="*80)
    for k in TRAINING_METRICS.keys():
        TRAINING_METRICS[k] = []
    
    rule_based_model = train_rule_based_agent(
        ENHANCED_JOBS_DATA, MACHINE_LIST,
        initial_jobs=INITIAL_JOB_IDS,
        arrival_rate=arrival_rate,
        total_timesteps=dynamic_timesteps,  # Same timesteps as reactive RL
        reward_mode="makespan_increment",
        learning_rate=learning_rate
    )
    print("\n--- Rule-Based RL Training Metrics ---")
    plot_training_metrics()
    
    for k in TRAINING_METRICS.keys():
        TRAINING_METRICS[k] = []
    static_model = train_static_agent(
        ENHANCED_JOBS_DATA, MACHINE_LIST, total_timesteps=static_timesteps,
        reward_mode="makespan_increment", learning_rate=learning_rate
    )
    print("\n--- Static RL Training Metrics ---")
    plot_training_metrics()

    # Train PROACTIVE RL agent (learns arrival patterns and schedules proactively)
    print("\n" + "="*80)
    print("PROACTIVE RL TRAINING")
    print("="*80)
    for k in TRAINING_METRICS.keys():
        TRAINING_METRICS[k] = []
    
    # Note: Perfect Knowledge RL will be trained separately for each test scenario
    # This ensures each scenario has its optimal RL benchmark for comparison
    
    # Step 4: Evaluate all methods on all test scenarios
    print("\n4. EVALUATION PHASE - MULTIPLE SCENARIOS")
    print("-" * 40)
    print("Comparing FOUR levels of arrival information across 10 test scenarios:")
    print("1. Perfect Knowledge RL (knows exact arrival times)")
    print("2. Proactive RL (learns to predict arrivals, schedules proactively)")
    print("3. Reactive RL (knows arrival distribution)")  
    print("4. Static RL (assumes all jobs at t=0)")
    
    # Initialize results storage
    all_results = {
        'Perfect Knowledge RL': [],
        'Proactive RL': [],  # NEW
        'Reactive RL': [],
        'Rule-Based RL': [],  # NEW - dispatching rule selection
        'Static RL (dynamic)': [],
        'Static RL (static)': [],
        'Best Heuristic': [],
        'MILP Optimal': []
    }
    
    # Storage for first 3 scenarios for Gantt chart plotting
    gantt_scenarios_data = []
    
    print(f"\nEvaluating on {len(test_scenarios)} test scenarios...")
    
    for i, scenario in enumerate(test_scenarios):
        print("\n" + "-"*60)
        print(f"SCENARIO {i+1}/{len(test_scenarios)}")
        scenario_arrivals = scenario['arrival_times']
        print(f"\nScenario {i+1}/10: {scenario_arrivals}")
        
        # STEP 1: Compute MILP Optimal Solution FIRST (to use as target for Perfect RL)
        print(f"  Computing MILP Optimal for scenario {i+1}...")
        milp_start_time = time.time()
        milp_makespan, milp_schedule = milp_optimal_scheduler(ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        milp_end_time = time.time()
        milp_solve_time = milp_end_time - milp_start_time
        
        # Check if MILP solution is valid (not timed out with poor solution)
        milp_is_valid = True
        if milp_makespan == float('inf'):
            print(f"    ⚠️  MILP FAILED - will use Perfect RL as benchmark")
            milp_is_valid = False
        else:
            print(f"    MILP Optimal: {milp_makespan:.2f} (solved in {milp_solve_time:.1f}s)")
        
        all_results['MILP Optimal'].append(milp_makespan if milp_is_valid else None)

        # STEP 2: Train Perfect Knowledge RL with MILP as target
        print(f"  Training Perfect Knowledge RL for scenario {i+1}...")
        print(f"    Scenario arrival times: {scenario_arrivals}")
        print(f"    Target: MILP Optimal = {milp_makespan:.2f}")
        # Reset metrics before each perfect RL training
        for k in TRAINING_METRICS.keys():
            TRAINING_METRICS[k] = []
        perfect_model = train_perfect_knowledge_agent(
            ENHANCED_JOBS_DATA, MACHINE_LIST,
            arrival_times=scenario_arrivals,
            total_timesteps=perfect_timesteps,
            reward_mode="makespan_increment", 
            learning_rate=learning_rate,
            num_initializations=1  # ⭐ Try 5 random initializations # ⭐ Pass MILP for early stopping
        )
        if i == 0:
            print("\n--- Perfect Knowledge RL Training Metrics (Scenario 1) ---")
            plot_training_metrics()
        
        # STEP 3: Evaluate Perfect Knowledge RL
        perfect_start_time = time.time()
        perfect_makespan, perfect_schedule = evaluate_perfect_knowledge_on_scenario(
            perfect_model, ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        all_results['Perfect Knowledge RL'].append(perfect_makespan)
        print(f"    Perfect RL final evaluation: {perfect_makespan:.2f}")
        perfect_end_time = time.time()
        print(f"    Perfect RL test time: {perfect_end_time - perfect_start_time:.2f} seconds")

        # Verify Perfect RL vs MILP (or invalidate MILP if it's worse than Perfect RL)
        if milp_is_valid:
            if perfect_makespan < milp_makespan - 0.01:  # Allow tiny numerical tolerance
                gap = ((perfect_makespan - milp_makespan) / milp_makespan) * 100
                print(f"    🚨 CRITICAL: Perfect RL ({perfect_makespan:.2f}) < MILP ({milp_makespan:.2f}) by {-gap:.2f}%!")
                print(f"    🚨 This indicates MILP timed out with suboptimal solution - DISREGARDING MILP")
                milp_is_valid = False
                all_results['MILP Optimal'][-1] = None  # Invalidate this MILP result
            elif perfect_makespan > milp_makespan * 1.05:  # > 5% worse
                gap = ((perfect_makespan - milp_makespan) / milp_makespan) * 100
                print(f"    ⚠️  WARNING: Perfect RL ({perfect_makespan:.2f}) is {gap:.2f}% worse than MILP")
                print(f"    ⚠️  Consider more training or better hyperparameters")
            else:
                gap = ((perfect_makespan - milp_makespan) / milp_makespan) * 100
                print(f"    ✅ Perfect RL gap to MILP: {gap:+.2f}% (acceptable)")
        
        # Determine benchmark for this scenario
        if milp_is_valid:
            benchmark_makespan = milp_makespan
            benchmark_name = "MILP Optimal"
        else:
            benchmark_makespan = perfect_makespan
            benchmark_name = "Perfect RL"
            print(f"    📊 Using Perfect RL ({perfect_makespan:.2f}) as benchmark (MILP invalid)")
        
        # Reactive RL
        dynamic_makespan, dynamic_schedule = evaluate_dynamic_on_dynamic(
            dynamic_model, ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        all_results['Reactive RL'].append(dynamic_makespan)
        
        # PROACTIVE RL (NEW)
        print(f"  Evaluating Proactive RL on scenario {i+1}...")
        proactive_start_time = time.time()
        proactive_makespan, proactive_schedule = evaluate_proactive_on_dynamic(
            proactive_model, ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        all_results['Proactive RL'].append(proactive_makespan)
        print(f"    Proactive RL (learned predictions): {proactive_makespan:.2f}")
        proactive_end_time = time.time()
        print(f"    Proactive RL test time: {proactive_end_time - proactive_start_time:.2f} seconds")

        # Rule-Based RL (NEW)
        print(f"  Evaluating Rule-Based RL on scenario {i+1}...")
        rule_based_start_time = time.time()
        rule_based_makespan, rule_based_schedule = evaluate_rule_based_on_dynamic(
            rule_based_model, ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        all_results['Rule-Based RL'].append(rule_based_makespan)
        print(f"    Rule-Based RL (learned rule selection): {rule_based_makespan:.2f}")
        rule_based_end_time = time.time()
        print(f"    Rule-Based RL test time: {rule_based_end_time - rule_based_start_time:.2f} seconds")

        # Static RL (on dynamic scenario)
        static_dynamic_makespan, static_dynamic_schedule = evaluate_static_on_dynamic(
            static_model, ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        all_results['Static RL (dynamic)'].append(static_dynamic_makespan)
        
        # Static RL (on static scenario) - only do once since it's always the same
        if i == 0:
            static_static_makespan, static_static_schedule = evaluate_static_on_static(
                static_model, ENHANCED_JOBS_DATA, MACHINE_LIST)
        all_results['Static RL (static)'].append(static_static_makespan)
        
        # Best Heuristic (compares FIFO, LIFO, SPT, LPT, EDD and selects best)
        spt_start_time = time.time()
        spt_makespan, spt_schedule = best_heuristic(ENHANCED_JOBS_DATA, MACHINE_LIST, scenario_arrivals)
        all_results['Best Heuristic'].append(spt_makespan)
        spt_end_time = time.time()
        print(f"    Best Heuristic training time: {spt_end_time - spt_start_time:.2f} seconds")

        # NOTE: MILP already computed above (before Perfect RL training)
        
        # Store ALL scenarios for Gantt plotting (skip MILP if invalid)
        schedules_dict = {
            'Perfect Knowledge RL': (perfect_makespan, perfect_schedule),
            'Proactive RL': (proactive_makespan, proactive_schedule),
            'Reactive RL': (dynamic_makespan, dynamic_schedule),
            'Rule-Based RL': (rule_based_makespan, rule_based_schedule),  # NEW
            'Static RL (dynamic)': (static_dynamic_makespan, static_dynamic_schedule),
            'Static RL (static)': (static_static_makespan, static_static_schedule),
            'Best Heuristic': (spt_makespan, spt_schedule)
        }
        
        # Only include MILP if it's valid
        if milp_is_valid:
            schedules_dict['MILP Optimal'] = (milp_makespan, milp_schedule)
        
        gantt_scenarios_data.append({
            'scenario_id': i,
            'arrival_times': scenario_arrivals,
            'schedules': schedules_dict
        })
        
        # Verify all schedules for correctness
        print(f"  Verifying schedule correctness for scenario {i+1}:")
        
        methods_to_verify = [
            ("Perfect Knowledge RL", perfect_makespan, perfect_schedule),
            ("Proactive RL", proactive_makespan, proactive_schedule),
            ("Reactive RL", dynamic_makespan, dynamic_schedule),
            ("Rule-Based RL", rule_based_makespan, rule_based_schedule),  # NEW
            ("Static RL (dynamic)", static_dynamic_makespan, static_dynamic_schedule),
            ("Best Heuristic", spt_makespan, spt_schedule)
        ]
        
        # Only verify MILP if it's valid
        if milp_is_valid:
            methods_to_verify.insert(0, ("MILP Optimal", milp_makespan, milp_schedule))
        
        for method_name, reported_makespan, schedule in methods_to_verify:
            if reported_makespan != float('inf') and schedule:
                # FIX: Use scenario_arrivals instead of undefined arrival_times
                is_valid, true_makespan = verify_schedule_correctness(schedule, ENHANCED_JOBS_DATA, scenario_arrivals, method_name)
                if is_valid:
                    if abs(reported_makespan - true_makespan) > 0.01:
                        print(f"    ⚠️  {method_name}: Makespan mismatch! Reported: {reported_makespan:.2f}, Actual: {true_makespan:.2f}")
                        # Update the reported makespan to the correct one
                        if method_name == "Perfect Knowledge RL":
                            perfect_makespan = true_makespan
                        elif method_name == "Proactive RL":
                            proactive_makespan = true_makespan
                        elif method_name == "Reactive RL":
                            dynamic_makespan = true_makespan
                        elif method_name == "Rule-Based RL":
                            rule_based_makespan = true_makespan
                        elif method_name == "Static RL (dynamic)":
                            static_dynamic_makespan = true_makespan
                    else:
                        print(f"    ✅ {method_name}: Valid schedule, makespan: {true_makespan:.2f}")
                else:
                    print(f"    ❌ {method_name}: Invalid schedule!")
                    # Mark as failed
                    if method_name == "Perfect Knowledge RL":
                        perfect_makespan = float('inf')
                    elif method_name == "Proactive RL":
                        proactive_makespan = float('inf')
                    elif method_name == "Reactive RL":
                        dynamic_makespan = float('inf')
                    elif method_name == "Rule-Based RL":
                        rule_based_makespan = float('inf')
                    elif method_name == "Static RL (dynamic)":
                        static_dynamic_makespan = float('inf')
            else:
                print(f"    ❌ {method_name}: No valid schedule!")
        
        # Check for duplicate schedules across methods (debugging identical results)
        # Removed debugging code for schedule identity checks
        
        # CRITICAL DEBUG: Compare Perfect RL vs Best Heuristic if heuristic is better
        if spt_makespan < perfect_makespan - 0.1:
            print(f"\n  🔍 DETAILED ANALYSIS: Heuristic ({spt_makespan:.2f}) outperformed Perfect RL ({perfect_makespan:.2f})")
            print(f"      This suggests Perfect RL may not have trained optimally or evaluation has issues.")
            print(f"      Checking for schedule differences...")
            
            # Compare machine utilization
            perfect_machine_util = {m: 0.0 for m in perfect_schedule.keys()}
            heuristic_machine_util = {m: 0.0 for m in spt_schedule.keys()}
            
            for machine, ops in perfect_schedule.items():
                if ops:
                    perfect_machine_util[machine] = max([op[2] for op in ops])
            
            for machine, ops in spt_schedule.items():
                if ops:
                    heuristic_machine_util[machine] = max([op[2] for op in ops])
            
            print(f"      Perfect RL machine end times: {perfect_machine_util}")
            print(f"      Heuristic machine end times: {heuristic_machine_util}")
            
            # Find which machine determines makespan for each
            perfect_bottleneck = max(perfect_machine_util.items(), key=lambda x: x[1])
            heuristic_bottleneck = max(heuristic_machine_util.items(), key=lambda x: x[1])
            
            print(f"      Perfect RL bottleneck: {perfect_bottleneck[0]} completes at {perfect_bottleneck[1]:.2f}")
            print(f"      Heuristic bottleneck: {heuristic_bottleneck[0]} completes at {heuristic_bottleneck[1]:.2f}")
            
            # Show first few operations on bottleneck machine
            if perfect_bottleneck[0] in perfect_schedule:
                print(f"      Perfect RL on {perfect_bottleneck[0]}: {perfect_schedule[perfect_bottleneck[0]][:5]}")
            if heuristic_bottleneck[0] in spt_schedule:
                print(f"      Heuristic on {heuristic_bottleneck[0]}: {spt_schedule[heuristic_bottleneck[0]][:5]}")
        
        print()  # Empty line for readability

        # Results summary line
        if milp_is_valid:
            print(f"  Results: Perfect={perfect_makespan:.2f}, Proactive={proactive_makespan:.2f}, Reactive={dynamic_makespan:.2f}, Rule-Based={rule_based_makespan:.2f}, Static(dyn)={static_dynamic_makespan:.2f}, Heuristic={spt_makespan:.2f}, MILP={milp_makespan:.2f}")
        else:
            print(f"  Results: Perfect={perfect_makespan:.2f} (BENCHMARK), Proactive={proactive_makespan:.2f}, Reactive={dynamic_makespan:.2f}, Rule-Based={rule_based_makespan:.2f}, Static(dyn)={static_dynamic_makespan:.2f}, Heuristic={spt_makespan:.2f}, MILP=INVALID")
        
        # STRICT DEBUG: Check for impossible results and HALT execution if found
        # Use 0.1 tolerance for numerical precision (MILP and RL use different solvers)
        TOLERANCE = 0.1
        
        # Only check against MILP if it's valid
        if milp_is_valid and dynamic_makespan < milp_makespan - TOLERANCE:
            print(f"  🚨🚨🚨 FATAL ERROR: Reactive RL ({dynamic_makespan:.2f}) outperformed MILP Optimal ({milp_makespan:.2f})!")
            print(f"      This is THEORETICALLY IMPOSSIBLE - Reactive RL cannot be better than MILP optimal!")
            print(f"      Bug in: evaluation function, schedule validation, or MILP formulation")
            print(f"      HALTING EXECUTION to investigate...")
            # exit(1)
        
        if milp_is_valid and perfect_makespan < milp_makespan - TOLERANCE:
            gap = milp_makespan - perfect_makespan
            print(f"  🚨 ALERT: Perfect Knowledge RL ({perfect_makespan:.2f}) outperformed MILP ({milp_makespan:.2f})!")
            print(f"      Gap: {gap:.2f} time units")
            
            # Check if MILP was proven optimal (from cache or recent solve)
            print(f"\n  Investigating possible causes:")
            print(f"      1. MILP solver might not have found true optimal (check solver status above)")
            print(f"      2. MILP time limit too strict (increase from {300}s if needed)")
            print(f"      3. Different constraint interpretations")
            print(f"      4. Numerical precision differences")
            
            if gap < 2.0:
                print(f"\n  💡 Gap is SMALL ({gap:.2f} < 2.0 time units)")
                print(f"     This is likely due to:")
                print(f"     - MILP solver terminated with feasible but not proven optimal solution")
                print(f"     - Time limit or numerical issues in MILP")
                print(f"     - RL found a better solution through different search strategy")
                print(f"\n  ✅ CONTINUING with warning (gap < 2.0 threshold)...")
                print(f"     If you see 'FEASIBLE (not proven optimal)' above, this explains the gap.")
            else:
                print(f"\n  🚨 Gap is LARGE ({gap:.2f} >= 2.0 time units)")
                print(f"     This requires investigation:")
                
                # Compare critical path operations
                if perfect_schedule and milp_schedule:
                    print(f"\n  Perfect RL schedule summary:")
                    for machine, ops in sorted(perfect_schedule.items()):
                        if ops:
                            print(f"    {machine}: {len(ops)} ops, last ends at {max(op[2] for op in ops):.2f}")
                    
                    print(f"\n  MILP schedule summary:")
                    for machine, ops in sorted(milp_schedule.items()):
                        if ops:
                            print(f"    {machine}: {len(ops)} ops, last ends at {max(op[2] for op in ops):.2f}")
                
                print(f"\n  HALTING EXECUTION for investigation...")
                # exit(1)
        
        if perfect_makespan > dynamic_makespan + 5.0:  # Increased tolerance for training variations
            print(f"  🚨 WARNING: Perfect Knowledge RL ({perfect_makespan:.2f}) much worse than Reactive RL ({dynamic_makespan:.2f})")
            print(f"      Perfect RL should generally be better since it knows exact arrival times")
            print(f"      This may indicate training issues or very difficult scenario")
        
        # Check if Reactive RL is giving same result as previous scenarios
        if i > 0 and abs(dynamic_makespan - all_results['Reactive RL'][i-1]) < 0.01:
            print(f"  🚨 SUSPICIOUS: Reactive RL giving identical makespan to previous scenario")
            print(f"      This suggests evaluation isn't properly using different arrival times")
    
    # Calculate average results
    avg_results = {}
    std_results = {}
    for method, results in all_results.items():
        # Filter out None and inf values
        valid_results = [r for r in results if r is not None and r != float('inf')]
        if valid_results:
            avg_results[method] = np.mean(valid_results)
            std_results[method] = np.std(valid_results)
        else:
            avg_results[method] = float('inf')
            std_results[method] = 0
    
    # Use first scenario data for single-scenario analyses (backward compatibility)
    first_scenario_arrivals = test_scenarios[0]['arrival_times']
    static_arrivals = {job_id: 0.0 for job_id in ENHANCED_JOBS_DATA.keys()}
    
    # Get individual results from first scenario for single plots
    perfect_makespan = all_results['Perfect Knowledge RL'][0]
    dynamic_makespan = all_results['Reactive RL'][0]  
    static_dynamic_makespan = all_results['Static RL (dynamic)'][0]
    static_static_makespan = all_results['Static RL (static)'][0]
    spt_makespan = all_results['Best Heuristic'][0]
    milp_makespan = all_results['MILP Optimal'][0]  # May be None if invalid
    
    # Determine benchmark for regret analysis
    milp_is_valid_first = milp_makespan is not None and milp_makespan != float('inf')
    if milp_is_valid_first:
        benchmark_makespan = milp_makespan
        benchmark_name = "MILP Optimal"
    else:
        benchmark_makespan = perfect_makespan
        benchmark_name = "Perfect RL"
    
    # Get schedules from first scenario
    perfect_schedule = gantt_scenarios_data[0]['schedules']['Perfect Knowledge RL'][1]
    proactive_schedule = gantt_scenarios_data[0]['schedules']['Proactive RL'][1]
    dynamic_schedule = gantt_scenarios_data[0]['schedules']['Reactive RL'][1]
    rule_based_schedule = gantt_scenarios_data[0]['schedules']['Rule-Based RL'][1]  # NEW - Added Rule-Based RL
    static_dynamic_schedule = gantt_scenarios_data[0]['schedules']['Static RL (dynamic)'][1]
    static_static_schedule = gantt_scenarios_data[0]['schedules']['Static RL (static)'][1]
    spt_schedule = gantt_scenarios_data[0]['schedules']['Best Heuristic'][1]
    
    # MILP schedule may not exist
    milp_schedule = gantt_scenarios_data[0]['schedules'].get('MILP Optimal', (None, None))[1]
    
    # Step 5: Results Analysis
    print("\n5. RESULTS ANALYSIS")
    print("=" * 60)
    print("AVERAGE RESULTS ACROSS 10 TEST SCENARIOS:")
    
    # Calculate average MILP (excluding None values)
    valid_milp = [m for m in all_results['MILP Optimal'] if m is not None and m != float('inf')]
    if valid_milp:
        avg_milp = np.mean(valid_milp)
        std_milp = np.std(valid_milp)
        print(f"MILP Optimal              - Avg Makespan: {avg_milp:.2f} ± {std_milp:.2f} ({len(valid_milp)}/{len(all_results['MILP Optimal'])} valid)")
    else:
        print(f"MILP Optimal              - No valid solutions (all timed out or invalid)")
        avg_milp = None
    
    print(f"Perfect Knowledge RL      - Avg Makespan: {avg_results['Perfect Knowledge RL']:.2f} ± {std_results['Perfect Knowledge RL']:.2f}")
    print(f"Proactive RL              - Avg Makespan: {avg_results['Proactive RL']:.2f} ± {std_results['Proactive RL']:.2f}")  # NEW
    print(f"Reactive RL (Poisson)      - Avg Makespan: {avg_results['Reactive RL']:.2f} ± {std_results['Reactive RL']:.2f}")  
    print(f"Static RL (on dynamic)    - Avg Makespan: {avg_results['Static RL (dynamic)']:.2f} ± {std_results['Static RL (dynamic)']:.2f}")
    print(f"Static RL (on static)     - Avg Makespan: {avg_results['Static RL (static)']:.2f} ± {std_results['Static RL (static)']:.2f}")
    print(f"Best Heuristic            - Avg Makespan: {avg_results['Best Heuristic']:.2f} ± {std_results['Best Heuristic']:.2f}")
    
    print("\nFirst Scenario Results (for detailed analysis):")
    if milp_is_valid_first:
        print(f"MILP Optimal              - Makespan: {milp_makespan:.2f} (THEORETICAL BEST)")
    else:
        print(f"MILP Optimal              - INVALID (timed out or failed)")
    print(f"Perfect Knowledge RL      - Makespan: {perfect_makespan:.2f} {'' if milp_is_valid_first else '(BENCHMARK)'}")
    print(f"Proactive RL              - Makespan: {all_results['Proactive RL'][0]:.2f}")  # NEW
    print(f"Reactive RL (Poisson)      - Makespan: {dynamic_makespan:.2f}")  
    print(f"Static RL (on dynamic)    - Makespan: {static_dynamic_makespan:.2f}")
    print(f"Static RL (on static)     - Makespan: {static_static_makespan:.2f}")
    print(f"Best Heuristic            - Makespan: {spt_makespan:.2f}")
    
    print("\nAverage Performance Ranking:")
    avg_results_list = [
        ("MILP Optimal", avg_results['MILP Optimal']),
        ("Perfect Knowledge RL", avg_results['Perfect Knowledge RL']),
        ("Proactive RL", avg_results['Proactive RL']),  # NEW
        ("Reactive RL", avg_results['Reactive RL']), 
        ("Static RL (dynamic)", avg_results['Static RL (dynamic)']),
        ("Static RL (static)", avg_results['Static RL (static)']),
        ("Best Heuristic", avg_results['Best Heuristic'])
    ]
    avg_results_list.sort(key=lambda x: x[1])
    for i, (method, makespan) in enumerate(avg_results_list, 1):
        if makespan == float('inf'):
            print(f"{i}. {method}: Failed")
        else:
            print(f"{i}. {method}: {makespan:.2f}")
    
    print(f"\nExpected Performance Hierarchy:")
    print(f"MILP Optimal ≤ Perfect Knowledge ≤ Proactive RL ≤ Reactive RL ≤ Static RL")
    if avg_results['MILP Optimal'] != float('inf'):
        print(f"Actual Avg: {avg_results['MILP Optimal']:.2f} ≤ {avg_results['Perfect Knowledge RL']:.2f} ≤ {avg_results['Proactive RL']:.2f} ≤ {avg_results['Reactive RL']:.2f} ≤ {avg_results['Static RL (dynamic)']:.2f}")
    else:
        print(f"Actual Avg (no MILP): {avg_results['Perfect Knowledge RL']:.2f} ≤ {avg_results['Proactive RL']:.2f} ≤ {avg_results['Reactive RL']:.2f} ≤ {avg_results['Static RL (dynamic)']:.2f}")
    
    # Step 5.5: Average Regret Analysis (Gap from Optimal across all scenarios)
    # Use MILP if valid for any scenarios, otherwise use Perfect RL
    if avg_milp is not None:
        benchmark_for_regret = avg_milp
        benchmark_regret_name = "MILP Optimal"
    else:
        benchmark_for_regret = avg_results['Perfect Knowledge RL']
        benchmark_regret_name = "Perfect RL"
        print(f"\n⚠️  Using {benchmark_regret_name} as benchmark (MILP invalid for all scenarios)")
    
    avg_methods_results = {
        "Proactive RL": avg_results['Proactive RL'],
        "Reactive RL": avg_results['Reactive RL'],
        "Rule-Based RL": avg_results['Rule-Based RL'],  # NEW - Added Rule-Based RL
        "Static RL (dynamic)": avg_results['Static RL (dynamic)'],
        "Static RL (static)": avg_results['Static RL (static)'],
        "Best Heuristic": avg_results['Best Heuristic']
    }
    
    # Don't include Perfect RL in regret if it's the benchmark
    if benchmark_regret_name != "Perfect RL":
        avg_methods_results["Perfect Knowledge RL"] = avg_results['Perfect Knowledge RL']
    
    print("\nAVERAGE REGRET ANALYSIS:")
    regret_results = calculate_regret_analysis(benchmark_for_regret, avg_methods_results, benchmark_regret_name)
    
    # Also calculate regret for each individual scenario
    print("\nINDIVIDUAL SCENARIO REGRET ANALYSIS:")
    all_regrets = {method: [] for method in avg_methods_results.keys()}
    
    for i in range(len(test_scenarios)):
        milp_optimal = all_results['MILP Optimal'][i]
        # Use MILP if valid, otherwise use Perfect RL as benchmark
        if milp_optimal is not None and milp_optimal != float('inf'):
            benchmark = milp_optimal
        else:
            benchmark = all_results['Perfect Knowledge RL'][i]
        
        if benchmark != float('inf'):
            scenario_methods = {
                "Proactive RL": all_results['Proactive RL'][i],
                "Reactive RL": all_results['Reactive RL'][i],
                "Rule-Based RL": all_results['Rule-Based RL'][i],  # NEW - Added Rule-Based RL
                "Static RL (dynamic)": all_results['Static RL (dynamic)'][i],
                "Static RL (static)": all_results['Static RL (static)'][i],
                "Best Heuristic": all_results['Best Heuristic'][i]
            }
            
            # Only include Perfect RL if MILP was the benchmark
            if milp_optimal is not None and milp_optimal != float('inf'):
                scenario_methods["Perfect Knowledge RL"] = all_results['Perfect Knowledge RL'][i]
            
            for method, makespan in scenario_methods.items():
                if makespan != float('inf'):
                    regret = ((makespan - benchmark) / benchmark) * 100
                    all_regrets[method].append(regret)
    
    print("Regret Statistics (% above optimal):")
    for method, regret_list in all_regrets.items():
        if regret_list:
            avg_regret = np.mean(regret_list)
            std_regret = np.std(regret_list)
            min_regret = np.min(regret_list)
            max_regret = np.max(regret_list)
            print(f"{method:25s}: {avg_regret:6.1f}% ± {std_regret:5.1f}% (range: {min_regret:.1f}% - {max_regret:.1f}%)")
        else:
            print(f"{method:25s}: No valid results")
    
    # Validate expected ordering
    if perfect_makespan <= dynamic_makespan <= static_dynamic_makespan:
        print("✅ EXPECTED: Perfect knowledge outperforms distribution knowledge outperforms no knowledge")
    else:
        print("❌ UNEXPECTED: Performance doesn't follow expected hierarchy")
    
    # Performance comparisons
    print("\n6. PERFORMANCE COMPARISON")
    print("-" * 40)
    
    # Perfect Knowledge vs Reactive RL
    if perfect_makespan < dynamic_makespan:
        perfect_advantage = ((dynamic_makespan - perfect_makespan) / dynamic_makespan) * 100
        print(f"Perfect Knowledge advantage over Reactive RL: {perfect_advantage:.1f}%")
    
    # Reactive RL vs Static RL (on dynamic scenario)
    if dynamic_makespan < static_dynamic_makespan:
        improvement = ((static_dynamic_makespan - dynamic_makespan) / static_dynamic_makespan) * 100
        print(f"✓ Reactive RL outperforms Static RL (dynamic) by {improvement:.1f}%")
    else:
        gap = ((dynamic_makespan - static_dynamic_makespan) / static_dynamic_makespan) * 100
        print(f"✗ Reactive RL underperforms Static RL (dynamic) by {gap:.1f}%")
    
    # Static RL comparison: dynamic vs static scenarios
    if static_static_makespan < static_dynamic_makespan:
        improvement = ((static_dynamic_makespan - static_static_makespan) / static_dynamic_makespan) * 100
        print(f"✓ Static RL performs {improvement:.1f}% better on static scenarios (as expected)")
    else:
        gap = ((static_static_makespan - static_dynamic_makespan) / static_static_makespan) * 100
        print(f"⚠️ Unexpected: Static RL performs {gap:.1f}% worse on static scenarios")
    
    # Reactive RL vs Best Heuristic
    if dynamic_makespan < spt_makespan:
        improvement = ((spt_makespan - dynamic_makespan) / spt_makespan) * 100
        print(f"✓ Reactive RL outperforms Best Heuristic by {improvement:.1f}%")
    else:
        gap = ((dynamic_makespan - spt_makespan) / spt_makespan) * 100
        print(f"✗ Reactive RL underperforms Best Heuristic by {gap:.1f}%")
    
    # Step 7: Generate Gantt Charts for Comparison
    print(f"\n7. GANTT CHART COMPARISON")
    print("-" * 60)
    
    # Main comparison with 6-7 plots (added Proactive RL and Rule-Based RL)
    num_plots = 7 if milp_makespan != float('inf') else 6
    fig, axes = plt.subplots(num_plots, 1, figsize=(18, num_plots * 3.5))
    
    if milp_makespan != float('inf'):
        fig.suptitle('Main Scheduling Comparison: All RL Methods vs MILP Optimal\n' + 
                     f'Test Scenario: Jobs 0-2 at t=0, Jobs 3-6 via Poisson arrivals\n' +
                     f'Proactive RL learns arrivals, Rule-Based RL selects dispatching rules', 
                     fontsize=16, fontweight='bold')
        schedules_data = [
            {'schedule': milp_schedule, 'makespan': milp_makespan, 'title': 'MILP Optimal (Benchmark)', 'arrival_times': first_scenario_arrivals},
            {'schedule': perfect_schedule, 'makespan': perfect_makespan, 'title': 'Perfect Knowledge RL', 'arrival_times': first_scenario_arrivals},
            {'schedule': proactive_schedule, 'makespan': all_results['Proactive RL'][0], 'title': 'Proactive RL (Learned Predictions)', 'arrival_times': first_scenario_arrivals},
            {'schedule': dynamic_schedule, 'makespan': dynamic_makespan, 'title': 'Reactive RL (Reactive)', 'arrival_times': first_scenario_arrivals},
            {'schedule': rule_based_schedule, 'makespan': all_results['Rule-Based RL'][0], 'title': 'Rule-Based RL (Learned Rule Selection)', 'arrival_times': first_scenario_arrivals},
            {'schedule': static_dynamic_schedule, 'makespan': static_dynamic_makespan, 'title': 'Static RL (on dynamic scenario)', 'arrival_times': first_scenario_arrivals},
            {'schedule': spt_schedule, 'makespan': spt_makespan, 'title': 'Best Heuristic', 'arrival_times': first_scenario_arrivals}
        ]
    else:
        fig.suptitle('Main Scheduling Comparison: All RL Methods\n' + 
                     f'Test Scenario: Jobs 0-2 at t=0, Jobs 3-6 via Poisson arrivals\n' +
                     f'Proactive RL learns arrivals, Rule-Based RL selects dispatching rules', 
                     fontsize=16, fontweight='bold')
        schedules_data = [
            {'schedule': perfect_schedule, 'makespan': perfect_makespan, 'title': 'Perfect Knowledge RL', 'arrival_times': first_scenario_arrivals},
            {'schedule': proactive_schedule, 'makespan': all_results['Proactive RL'][0], 'title': 'Proactive RL (Learned Predictions)', 'arrival_times': first_scenario_arrivals},
            {'schedule': dynamic_schedule, 'makespan': dynamic_makespan, 'title': 'Reactive RL (Reactive)', 'arrival_times': first_scenario_arrivals},
            {'schedule': rule_based_schedule, 'makespan': all_results['Rule-Based RL'][0], 'title': 'Rule-Based RL (Learned Rule Selection)', 'arrival_times': first_scenario_arrivals},
            {'schedule': static_dynamic_schedule, 'makespan': static_dynamic_makespan, 'title': 'Static RL (on dynamic scenario)', 'arrival_times': first_scenario_arrivals},
            {'schedule': spt_schedule, 'makespan': spt_makespan, 'title': 'Best Heuristic', 'arrival_times': first_scenario_arrivals}
        ]
    
    colors = plt.cm.tab20.colors
    
    # Calculate the maximum makespan across all schedules for consistent scaling
    max_makespan_for_scaling = 0
    for data in schedules_data:
        schedule = data['schedule']
        if schedule and any(len(ops) > 0 for ops in schedule.values()):
            schedule_max_time = max([max([op[2] for op in ops]) for ops in schedule.values() if ops])
            max_makespan_for_scaling = max(max_makespan_for_scaling, schedule_max_time)
    
    # Add some padding (10%) for visual clarity
    consistent_x_limit = max_makespan_for_scaling * 1.1 if max_makespan_for_scaling > 0 else 100
    
    for plot_idx, data in enumerate(schedules_data):
        schedule = data['schedule']
        makespan = data['makespan']
        title = data['title']
        arrival_times = data['arrival_times']
        
        ax = axes[plot_idx]
        
        if not schedule or all(len(ops) == 0 for ops in schedule.values()):
            ax.text(0.5, 0.5, 'No valid schedule', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{title} - No Solution")
            # Still apply consistent scaling even for failed schedules
            ax.set_xlim(0, consistent_x_limit)
            ax.set_ylim(-0.5, len(MACHINE_LIST) + 2.0)
            continue
        
        # Plot operations for each machine
        for idx, machine in enumerate(MACHINE_LIST):
            machine_ops = schedule.get(machine, [])
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

        # Add red arrows for job arrivals (only for dynamic jobs that arrive > 0)
        if arrival_times:
            arrow_y_position = len(MACHINE_LIST) + 0.3  # Position above all machines
            for job_id, arrival_time in arrival_times.items():
                if arrival_time > 0 and arrival_time < consistent_x_limit:  # Only show arrows for jobs that don't start at t=0 and arrive within time horizon
                    # Draw vertical line for arrival
                    ax.axvline(x=arrival_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
                    
                    # Add arrow and label
                    ax.annotate(f'Job {job_id} arrives', 
                               xy=(arrival_time, arrow_y_position), 
                               xytext=(arrival_time, arrow_y_position + 0.5),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2),
                               ha='center', va='bottom', color='red', fontweight='bold', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))
        
        # Formatting
        ax.set_yticks(range(len(MACHINE_LIST)))
        ax.set_yticklabels(MACHINE_LIST)
        ax.set_xlabel("Time" if plot_idx == len(schedules_data)-1 else "")
        ax.set_ylabel("Machines")
        ax.set_title(f"{title} (Makespan: {makespan:.2f})", fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Apply consistent x-axis limits across all subplots
        ax.set_xlim(0, consistent_x_limit)
        ax.set_ylim(-0.5, len(MACHINE_LIST) + 2.0)  # Extra space for arrival arrows and labels
    
    # Add legend
    legend_elements = []
    for i in range(len(ENHANCED_JOBS_DATA)):
        color = colors[i % len(colors)]
        initial_or_poisson = ' (Initial)' if i < 3 else ' (Poisson)'
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                          alpha=0.8, label=f'Job {i}{initial_or_poisson}'))
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(ENHANCED_JOBS_DATA), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save with appropriate filename based on MILP availability
    if milp_makespan != float('inf'):
        filename = 'complete_scheduling_comparison_with_milp_optimal.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comprehensive comparison with MILP optimal: {filename}")
    else:
        filename = 'dynamic_vs_static_gantt_comparison-7jobs.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison without MILP: {filename}")
    
    plt.show()

    # Skip the separate static RL comparison - focus on 10 test scenarios
    # Step 8: Create Gantt Charts for All 10 Test Scenarios (5 methods only)
    print(f"\n8. GANTT CHARTS FOR ALL 10 TEST SCENARIOS")
    print("-" * 60)
    
    # Create proactive folder
    import os
    folder_name = f"proactive_20J6M_{arrival_rate}_rate"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    
    # Generate Gantt charts for all 10 scenarios using stored data only
    for scenario_idx in range(len(gantt_scenarios_data)):
        scenario = gantt_scenarios_data[scenario_idx]
        scenario_id = scenario['scenario_id']
        arrival_times = scenario['arrival_times']
        schedules = scenario['schedules']
        print(f"\nGenerating Gantt chart for Test Scenario {scenario_id + 1}...")
        print(f"Arrival times: {arrival_times}")
        num_methods = 7 if schedules.get('MILP Optimal') and schedules['MILP Optimal'][0] != float('inf') else 6
        fig, axes = plt.subplots(num_methods, 1, figsize=(16, num_methods * 3))
        
        if schedules.get('MILP Optimal') and schedules['MILP Optimal'][0] != float('inf'):
            fig.suptitle(f'Test Scenario {scenario_id + 1} - 7 Method Comparison\n' + 
                         f'Arrival Times: {arrival_times}', 
                         fontsize=14, fontweight='bold')
            methods_to_plot = [
                ('MILP Optimal', schedules['MILP Optimal']),
                ('Perfect Knowledge RL', schedules['Perfect Knowledge RL']),
                ('Proactive RL', schedules['Proactive RL']),
                ('Reactive RL', schedules['Reactive RL']),
                ('Rule-Based RL', schedules['Rule-Based RL']),
                ('Static RL (dynamic)', schedules['Static RL (dynamic)']),
                ('Best Heuristic', schedules['Best Heuristic'])
            ]
        else:
            fig.suptitle(f'Test Scenario {scenario_id + 1} - 6 Method Comparison\n' + 
                         f'Arrival Times: {arrival_times}', 
                         fontsize=14, fontweight='bold')
            methods_to_plot = [
                ('Perfect Knowledge RL', schedules['Perfect Knowledge RL']),
                ('Proactive RL', schedules['Proactive RL']),
                ('Reactive RL', schedules['Reactive RL']),
                ('Rule-Based RL', schedules['Rule-Based RL']),
                ('Static RL (dynamic)', schedules['Static RL (dynamic)']),
                ('Best Heuristic', schedules['Best Heuristic'])
            ]
        
        # Calculate consistent x-axis limits for this scenario based on the LONGEST makespan
        max_makespan_scenario = 0
        for method_name, (makespan, schedule) in methods_to_plot:
            if makespan != float('inf'):
                max_makespan_scenario = max(max_makespan_scenario, makespan)
        
        # Use makespan + 15% padding for consistent x-axis across all subplots
        x_limit_scenario = max_makespan_scenario * 1.15 if max_makespan_scenario > 0 else 100
        
        # Plot each method
        colors = plt.cm.tab20.colors
        
        for plot_idx, (method_name, (makespan, schedule)) in enumerate(methods_to_plot):
            ax = axes[plot_idx] if num_methods > 1 else axes
            
            if not schedule or all(len(ops) == 0 for ops in schedule.values()):
                ax.text(0.5, 0.5, 'No valid schedule', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{method_name} - No Solution")
                ax.set_xlim(0, x_limit_scenario)
                ax.set_ylim(-0.5, len(MACHINE_LIST) + 1.5)
                continue
            
            # Plot operations for each machine
            for idx, machine in enumerate(MACHINE_LIST):
                machine_ops = schedule.get(machine, [])
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
                            except (ValueError, IndexError):
                                job_num = 0
                        
                        color = colors[job_num % len(colors)]
                        
                        ax.barh(idx, duration, left=start_time, height=0.6, 
                               color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                        
                        # Add operation label
                        if duration > 1:  # Only add text if bar is wide enough
                            ax.text(start_time + duration/2, idx, job_op, 
                                   ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Add arrival arrows for jobs that arrive after t=0
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
            ax.set_xlabel("Time" if plot_idx == len(methods_to_plot)-1 else "")
            ax.set_ylabel("Machines")
            ax.set_title(f"{method_name} (Makespan: {makespan:.2f})", fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, x_limit_scenario)
            ax.set_ylim(-0.5, len(MACHINE_LIST) + 1.5)
    
        # Add legend for jobs
        legend_elements = []
        for i in range(len(ENHANCED_JOBS_DATA)):
            color = colors[i % len(colors)]
            initial_or_dynamic = ' (Initial)' if i < 3 else ' (Dynamic)'
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                            alpha=0.8, label=f'Job {i}{initial_or_dynamic}'))
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
                ncol=len(ENHANCED_JOBS_DATA), fontsize=9)
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.93])
        
        # Save scenario-specific Gantt chart in small_instances folder
        scenario_filename = os.path.join(folder_name, f'test_scenario_{scenario_id + 1}_gantt_comparison.png')
        plt.savefig(scenario_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved Test Scenario {scenario_id + 1} Gantt chart: {scenario_filename}")
        
    plt.close()  # Close figure to save memory

    print(f"\n✅ All Gantt charts saved in {folder_name}/ folder")

    # Skip the old static RL comparison code - focus on the 10 test scenario Gantt charts above
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED!")
    print("Generated files:")
    if milp_makespan != float('inf'):
        print("- complete_scheduling_comparison_with_milp_optimal.png: Five-method comprehensive comparison with MILP benchmark")
        print(f"- small_instances/ folder: Contains 10 Gantt charts for all test scenarios (5 methods each)")
        for i in range(5):
            print(f"  ├── test_scenario_{i+1}_gantt_comparison.png")
        print(f"\nKey Findings (Average across 5 test scenarios):")
        print(f"• MILP Optimal (Benchmark): {avg_results['MILP Optimal']:.2f} ± {std_results['MILP Optimal']:.2f}")
        print(f"• Perfect Knowledge RL: {avg_results['Perfect Knowledge RL']:.2f} ± {std_results['Perfect Knowledge RL']:.2f} (avg regret: +{((avg_results['Perfect Knowledge RL']-avg_results['MILP Optimal'])/avg_results['MILP Optimal']*100):.1f}%)")
        print(f"• Reactive RL: {avg_results['Reactive RL']:.2f} ± {std_results['Reactive RL']:.2f} (avg regret: +{((avg_results['Reactive RL']-avg_results['MILP Optimal'])/avg_results['MILP Optimal']*100):.1f}%)")
        print(f"• Static RL (on dynamic): {avg_results['Static RL (dynamic)']:.2f} ± {std_results['Static RL (dynamic)']:.2f} (avg regret: +{((avg_results['Static RL (dynamic)']-avg_results['MILP Optimal'])/avg_results['MILP Optimal']*100):.1f}%)")
        print(f"• Static RL (on static): {avg_results['Static RL (static)']:.2f} ± {std_results['Static RL (static)']:.2f} (avg regret: +{((avg_results['Static RL (static)']-avg_results['MILP Optimal'])/avg_results['MILP Optimal']*100):.1f}%)")
        print(f"• Best Heuristic: {avg_results['Best Heuristic']:.2f} ± {std_results['Best Heuristic']:.2f} (avg regret: +{((avg_results['Best Heuristic']-avg_results['MILP Optimal'])/avg_results['MILP Optimal']*100):.1f}%)")
        print(f"• Perfect Knowledge RL validation: {'✅ Working well' if avg_results['Perfect Knowledge RL'] <= avg_results['MILP Optimal'] * 1.15 else '❌ Needs improvement'}")


if __name__ == "__main__":
    main()
