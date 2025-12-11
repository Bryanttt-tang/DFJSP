"""
Automated Hyperparameter Tuning for Rule-Based RL using Optuna
=================================================================

This script automatically finds optimal hyperparameters for the Rule-Based RL agent.

Installation:
    pip install optuna optuna-dashboard

Usage:
    python tune_rule_based_hyperparameters.py
    
    # Optional: View tuning progress in real-time
    optuna-dashboard sqlite:///rule_based_tuning.db

The script will:
1. Test different combinations of hyperparameters
2. Train each configuration briefly (to save time)
3. Evaluate on test scenarios
4. Find the best configuration
5. Save the best hyperparameters to JSON
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import torch
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
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


# ============================================
# Configuration
# ============================================

# Training budget per trial (reduce for faster tuning)
TIMESTEPS_PER_TRIAL = 200_000  # Quick evaluation (full training: 1M+)

# Number of trials to run
N_TRIALS = 30  # More trials = better results but slower

# Test scenarios for evaluation
N_EVAL_SCENARIOS = 5  # Evaluate on multiple scenarios

# Dataset configuration
N_JOBS = 20
N_MACHINES = 5
INITIAL_JOBS = 5
ARRIVAL_RATE = 0.1


# ============================================
# Helper Functions
# ============================================

def mask_fn(env):
    """Action mask function for MaskablePPO."""
    return env.action_masks()


class TrialEvalCallback(BaseCallback):
    """
    Callback for pruning unpromising trials during training.
    Reports intermediate performance to Optuna.
    """
    def __init__(self, eval_env, trial, n_eval_episodes=3, eval_freq=10000):
        super().__init__()
        self.eval_env = eval_env
        self.trial = trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.is_pruned = False
        self.last_mean_reward = float('inf')
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()  # Gym API returns tuple
                done = False
                episode_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                episode_rewards.append(episode_reward)
            
            mean_reward = np.mean(episode_rewards)
            self.last_mean_reward = mean_reward
            
            # Report to Optuna
            self.trial.report(-mean_reward, self.n_calls)  # Negative because we want to minimize makespan
            
            # Prune if trial is not promising
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        
        return True


def create_train_env(jobs_data, machine_list, seed):
    """Create training environment."""
    env = DispatchingRuleFJSPEnv(
        jobs_data, machine_list,
        initial_jobs=INITIAL_JOBS,
        arrival_rate=ARRIVAL_RATE,
        reward_mode="makespan_increment",
        seed=seed,
        max_time_horizon=1000
    )
    env = ActionMasker(env, mask_fn)
    return env


def create_test_scenarios(n_scenarios=5):
    """Generate test scenarios for evaluation."""
    test_scenarios = []
    for i in range(n_scenarios):
        jobs, machines = generate_fjsp_dataset(
            num_initial_jobs=N_JOBS,
            num_future_jobs=0,  # No dynamic arrivals for tuning
            total_num_machines=N_MACHINES,
            seed=GLOBAL_SEED + 1000 + i  # Different seeds for test
        )
        test_scenarios.append((jobs, machines))
    return test_scenarios


# ============================================
# Optuna Objective Function
# ============================================

def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna to optimize.
    
    Returns:
        Average makespan on test scenarios (lower is better)
    """
    
    # ===== Suggest hyperparameters =====
    
    # Learning rate: Log-uniform between 1e-5 and 1e-3
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    
    # Clip range: Uniform between 0.05 and 0.3
    clip_range = trial.suggest_float("clip_range", 0.05, 0.3)
    
    # Entropy coefficient: Log-uniform between 0.001 and 0.1
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    
    # Number of epochs: Categorical choice
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 15, 20])
    
    # Batch size: Categorical choice
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    
    # Number of steps: Categorical choice
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    
    # GAE lambda: Uniform between 0.8 and 1.0
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    
    # Max gradient norm: Uniform between 0.3 and 1.0
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    
    # Network architecture: Categorical choice
    net_arch_choice = trial.suggest_categorical("net_arch", [
        "small",   # [128, 128]
        "medium",  # [256, 256, 128]
        "large"    # [512, 256, 128]
    ])
    
    if net_arch_choice == "small":
        net_arch = [128, 128]
    elif net_arch_choice == "medium":
        net_arch = [256, 256, 128]
    else:  # large
        net_arch = [512, 256, 128]
    
    # ===== Generate training data =====
    jobs_data, machine_list = generate_fjsp_dataset(
        num_initial_jobs=N_JOBS,
        num_future_jobs=0,  # No dynamic arrivals for tuning
        total_num_machines=N_MACHINES,
        seed=GLOBAL_SEED + trial.number  # Different seed per trial
    )
    
    # ===== Create environment =====
    vec_env = DummyVecEnv([lambda: create_train_env(jobs_data, machine_list, GLOBAL_SEED + trial.number)])
    vec_env = VecMonitor(vec_env)
    
    # Create evaluation environment
    eval_env = create_train_env(jobs_data, machine_list, GLOBAL_SEED + trial.number + 500)
    
    # ===== Create model =====
    try:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=1.0,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=max_grad_norm,
            normalize_advantage=True,
            seed=GLOBAL_SEED + trial.number,
            policy_kwargs=dict(
                net_arch=net_arch,
                activation_fn=torch.nn.ReLU
            )
        )
    except Exception as e:
        print(f"❌ Trial {trial.number} failed to create model: {e}")
        raise optuna.TrialPruned()
    
    # ===== Train model =====
    try:
        # Use callback for pruning
        callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=3, eval_freq=10000)
        
        model.learn(total_timesteps=TIMESTEPS_PER_TRIAL, callback=callback)
        
        # Check if pruned
        if callback.is_pruned:
            raise optuna.TrialPruned()
            
    except Exception as e:
        print(f"❌ Trial {trial.number} failed during training: {e}")
        raise optuna.TrialPruned()
    
    # ===== Evaluate on test scenarios =====
    test_scenarios = create_test_scenarios(N_EVAL_SCENARIOS)
    
    makespans = []
    for test_jobs, test_machines in test_scenarios:
        try:
            # Evaluate rule-based RL
            rl_makespan, _, _ = evaluate_rule_based_on_dynamic(
                model, test_jobs, test_machines,
                initial_jobs=INITIAL_JOBS,
                arrival_rate=ARRIVAL_RATE,
                scenario_name="test",
                verbose=False
            )
            makespans.append(rl_makespan)
        except Exception as e:
            print(f"❌ Trial {trial.number} failed during evaluation: {e}")
            makespans.append(1e6)  # Penalty for failure
    
    # Clean up
    vec_env.close()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Return average makespan (lower is better)
    avg_makespan = np.mean(makespans)
    
    print(f"✅ Trial {trial.number}: Avg Makespan = {avg_makespan:.2f}")
    
    return avg_makespan


# ============================================
# Main Tuning Script
# ============================================

def main():
    """Run hyperparameter tuning."""
    
    print("=" * 70)
    print("Automated Hyperparameter Tuning for Rule-Based RL")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"   - Trials: {N_TRIALS}")
    print(f"   - Timesteps per trial: {TIMESTEPS_PER_TRIAL:,}")
    print(f"   - Eval scenarios: {N_EVAL_SCENARIOS}")
    print(f"   - Dataset: {N_JOBS} jobs, {N_MACHINES} machines")
    print(f"   - Arrival rate: {ARRIVAL_RATE}")
    print(f"\n🎯 Goal: Minimize average makespan on test scenarios")
    print(f"\n⏱️  Estimated time: {N_TRIALS * TIMESTEPS_PER_TRIAL / 10000:.0f}-{N_TRIALS * TIMESTEPS_PER_TRIAL / 5000:.0f} minutes")
    print("\n" + "=" * 70 + "\n")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name="rule_based_rl_tuning",
        direction="minimize",  # Minimize makespan
        sampler=TPESampler(seed=GLOBAL_SEED),  # Tree-structured Parzen Estimator
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),  # Prune bad trials early
        storage="sqlite:///rule_based_tuning.db",  # Save to database
        load_if_exists=True  # Resume if interrupted
    )
    
    # Run optimization
    start_time = time.time()
    
    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            show_progress_bar=True,
            catch=(Exception,)
        )
    except KeyboardInterrupt:
        print("\n⚠️  Tuning interrupted by user!")
    
    end_time = time.time()
    tuning_time = end_time - start_time
    
    # ===== Results =====
    print("\n" + "=" * 70)
    print("Tuning Complete!")
    print("=" * 70)
    print(f"\n⏱️  Total time: {tuning_time / 60:.1f} minutes")
    
    # Check if any trials succeeded
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) == 0:
        print("\n❌ No trials completed successfully!")
        print("\nAll trials failed. Check the error messages above.")
        print("\nCommon issues:")
        print("  1. Function signature mismatch")
        print("  2. Missing dependencies")
        print("  3. Environment bugs")
        return
    
    print(f"\n✅ Completed trials: {len(completed_trials)}/{len(study.trials)}")
    print(f"\n🏆 Best trial: #{study.best_trial.number}")
    print(f"   - Best makespan: {study.best_value:.2f}")
    print(f"\n📋 Best hyperparameters:")
    
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"   - {param}: {value}")
    
    # ===== Save results =====
    results = {
        "best_makespan": float(study.best_value),
        "best_trial": study.best_trial.number,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "tuning_time_minutes": tuning_time / 60,
        "config": {
            "timesteps_per_trial": TIMESTEPS_PER_TRIAL,
            "n_eval_scenarios": N_EVAL_SCENARIOS,
            "n_jobs": N_JOBS,
            "n_machines": N_MACHINES,
            "arrival_rate": ARRIVAL_RATE
        }
    }
    
    results_file = "rule_based_best_hyperparameters.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # ===== Visualizations (optional) =====
    print("\n📊 Generating optimization plots...")
    
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_slice
        )
        
        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_html("optimization_history.html")
        print("   ✅ Saved: optimization_history.html")
        
        # Parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_html("param_importances.html")
        print("   ✅ Saved: param_importances.html")
        
        # Slice plot
        fig3 = plot_slice(study)
        fig3.write_html("param_slice.html")
        print("   ✅ Saved: param_slice.html")
        
        print("\n   💡 Open these HTML files in your browser to view interactive plots!")
        
    except ImportError:
        print("   ⚠️  Install plotly for visualizations: pip install plotly")
    except Exception as e:
        print(f"   ⚠️  Could not generate plots: {e}")
    
    # ===== Comparison with default parameters =====
    print("\n" + "=" * 70)
    print("Comparison with Default Parameters")
    print("=" * 70)
    
    default_params = {
        "learning_rate": 1e-4,
        "clip_range": 0.1,
        "ent_coef": 0.03,
        "n_epochs": 10,
        "batch_size": 512,
        "n_steps": 2048,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "net_arch": "medium"
    }
    
    print("\nDefault parameters:")
    for param, value in default_params.items():
        best_value = best_params.get(param, "N/A")
        change = "→" if str(value) != str(best_value) else "✓"
        print(f"   {param:20s}: {str(value):15s} {change} {best_value}")
    
    improvement = ((study.best_value) / study.best_value - 1) * 100
    print(f"\n📈 Improvement: Tuned parameters should perform better!")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("\n1. Review the best hyperparameters above")
    print("2. Update train_rule_based_agent() in proactive_sche.py")
    print("3. Re-train with the optimized parameters")
    print("4. Compare performance to baseline\n")


if __name__ == "__main__":
    main()
