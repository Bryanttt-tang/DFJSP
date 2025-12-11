"""
Quick Hyperparameter Tuning using Scikit-Optimize
===================================================

Lighter alternative to Optuna - uses Bayesian optimization.

Installation:
    pip install scikit-optimize

Usage:
    python tune_rule_based_skopt.py

Faster than Optuna but less features.
"""

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import json
import time
from proactive_sche import (
    DispatchingRuleFJSPEnv,
    generate_fjsp_dataset,
    evaluate_rule_based_on_dynamic,
    GLOBAL_SEED
)


# Configuration
TIMESTEPS_PER_TRIAL = 200_000
N_CALLS = 20  # Number of optimization iterations
N_EVAL_SCENARIOS = 5
N_JOBS = 20
N_MACHINES = 5
INITIAL_JOBS = 5
ARRIVAL_RATE = 0.1


def mask_fn(env):
    return env.action_masks()


# Define search space
search_space = [
    Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform'),
    Real(0.05, 0.3, name='clip_range'),
    Real(0.001, 0.1, name='ent_coef', prior='log-uniform'),
    Integer(5, 20, name='n_epochs'),
    Categorical([256, 512, 1024], name='batch_size'),
    Categorical([1024, 2048, 4096], name='n_steps'),
    Real(0.8, 1.0, name='gae_lambda'),
    Real(0.3, 1.0, name='max_grad_norm'),
    Categorical(['small', 'medium', 'large'], name='net_arch')
]


@use_named_args(search_space)
def objective(**params):
    """Objective function to minimize."""
    
    print(f"\n{'='*60}")
    print(f"Testing configuration:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")
    
    # Map network architecture
    if params['net_arch'] == 'small':
        net_arch = [128, 128]
    elif params['net_arch'] == 'medium':
        net_arch = [256, 256, 128]
    else:
        net_arch = [512, 256, 128]
    
    # Generate training data
    jobs_data, machine_list = generate_fjsp_dataset(
        num_initial_jobs=N_JOBS,
        num_future_jobs=0,  # No dynamic arrivals for tuning
        total_num_machines=N_MACHINES,
        seed=GLOBAL_SEED + np.random.randint(1000)
    )
    
    # Create environment
    def make_env():
        env = DispatchingRuleFJSPEnv(
            jobs_data, machine_list,
            initial_jobs=INITIAL_JOBS,
            arrival_rate=ARRIVAL_RATE,
            reward_mode="makespan_increment",
            seed=GLOBAL_SEED,
            max_time_horizon=1000
        )
        return ActionMasker(env, mask_fn)
    
    vec_env = DummyVecEnv([make_env])
    vec_env = VecMonitor(vec_env)
    
    # Create model
    try:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            gamma=1.0,
            gae_lambda=params['gae_lambda'],
            clip_range=params['clip_range'],
            ent_coef=params['ent_coef'],
            vf_coef=0.5,
            max_grad_norm=params['max_grad_norm'],
            normalize_advantage=True,
            seed=GLOBAL_SEED,
            policy_kwargs=dict(
                net_arch=net_arch,
                activation_fn=torch.nn.ReLU
            )
        )
        
        # Train
        model.learn(total_timesteps=TIMESTEPS_PER_TRIAL)
        
        # Evaluate
        test_scenarios = []
        for i in range(N_EVAL_SCENARIOS):
            jobs, machines = generate_fjsp_dataset(
                num_initial_jobs=N_JOBS,
                num_future_jobs=0,  # No dynamic arrivals for tuning
                total_num_machines=N_MACHINES,
                seed=GLOBAL_SEED + 1000 + i
            )
            test_scenarios.append((jobs, machines))
        
        makespans = []
        for test_jobs, test_machines in test_scenarios:
            rl_makespan, _, _ = evaluate_rule_based_on_dynamic(
                model, test_jobs, test_machines,
                initial_jobs=INITIAL_JOBS,
                arrival_rate=ARRIVAL_RATE,
                scenario_name="test",
                verbose=False
            )
            makespans.append(rl_makespan)
        
        avg_makespan = np.mean(makespans)
        print(f"✅ Result: Avg Makespan = {avg_makespan:.2f}\n")
        
        # Cleanup
        vec_env.close()
        del model
        
        return avg_makespan
        
    except Exception as e:
        print(f"❌ Trial failed: {e}")
        return 1e6  # Penalty


def main():
    print("\n" + "="*70)
    print("Bayesian Hyperparameter Optimization (scikit-optimize)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Iterations: {N_CALLS}")
    print(f"  - Timesteps per trial: {TIMESTEPS_PER_TRIAL:,}")
    print(f"  - Eval scenarios: {N_EVAL_SCENARIOS}")
    print("\n" + "="*70 + "\n")
    
    start_time = time.time()
    
    # Run optimization
    result = gp_minimize(
        objective,
        search_space,
        n_calls=N_CALLS,
        random_state=GLOBAL_SEED,
        verbose=True
    )
    
    end_time = time.time()
    
    # Extract best parameters
    best_params = {
        'learning_rate': result.x[0],
        'clip_range': result.x[1],
        'ent_coef': result.x[2],
        'n_epochs': result.x[3],
        'batch_size': result.x[4],
        'n_steps': result.x[5],
        'gae_lambda': result.x[6],
        'max_grad_norm': result.x[7],
        'net_arch': result.x[8]
    }
    
    # Print results
    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)
    print(f"\n⏱️  Time: {(end_time - start_time) / 60:.1f} minutes")
    print(f"\n🏆 Best makespan: {result.fun:.2f}")
    print(f"\n📋 Best hyperparameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # Save results
    results = {
        "best_makespan": float(result.fun),
        "best_params": best_params,
        "n_calls": N_CALLS,
        "tuning_time_minutes": (end_time - start_time) / 60
    }
    
    with open("rule_based_best_hyperparameters_skopt.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n💾 Saved to: rule_based_best_hyperparameters_skopt.json\n")


if __name__ == "__main__":
    main()
