# Automated Hyperparameter Tuning Guide

## Quick Start

### Option 1: Optuna (Recommended) ⭐

**Best for:** Comprehensive tuning with visualization and pruning

```bash
# Install
pip install optuna optuna-dashboard plotly

# Run tuning (30 trials, ~30-60 minutes)
python tune_rule_based_hyperparameters.py

# View real-time progress (optional, in another terminal)
optuna-dashboard sqlite:///rule_based_tuning.db
```

**Features:**
- ✅ Automatic pruning of bad trials (saves time)
- ✅ Interactive HTML visualizations
- ✅ Resume interrupted runs
- ✅ Real-time dashboard
- ✅ Parameter importance analysis

---

### Option 2: Scikit-Optimize (Lightweight) 🚀

**Best for:** Quick tuning without extra dependencies

```bash
# Install
pip install scikit-optimize

# Run tuning (20 iterations, ~20-40 minutes)
python tune_rule_based_skopt.py
```

**Features:**
- ✅ Bayesian optimization (intelligent search)
- ✅ Faster than random/grid search
- ✅ Minimal dependencies
- ⚠️ No pruning or visualization

---

## How It Works

Both scripts:
1. **Search** through hyperparameter combinations
2. **Train** each configuration briefly (200k timesteps)
3. **Evaluate** on multiple test scenarios
4. **Find** the best configuration
5. **Save** results to JSON

### Hyperparameters Tuned

| Parameter | Range | Impact |
|-----------|-------|--------|
| `learning_rate` | 1e-5 to 1e-3 | Speed vs stability |
| `clip_range` | 0.05 to 0.3 | Policy update size |
| `ent_coef` | 0.001 to 0.1 | Exploration vs exploitation |
| `n_epochs` | 5 to 20 | Learning from each batch |
| `batch_size` | 256, 512, 1024 | Gradient stability |
| `n_steps` | 1024, 2048, 4096 | Rollout buffer size |
| `gae_lambda` | 0.8 to 1.0 | Advantage estimation |
| `max_grad_norm` | 0.3 to 1.0 | Gradient clipping |
| `net_arch` | small/medium/large | Network capacity |

---

## Using the Scripts

### 1. Install Dependencies

```bash
# For Optuna (recommended)
pip install optuna optuna-dashboard plotly

# OR for Scikit-Optimize (lightweight)
n_steps = 256             # REDUCED from 512
batch_size = 512          # INCREASED from 256
n_epochs = 4              # REDUCED from 10
ent_coef = 0.01           # INCREASED from 0.001 for more exploration
```

## When to Use Optuna for Hyperparameter Search

### ✅ Use Optuna IF:
1. **After testing optimized settings**, you still see issues
2. You want to **systematically explore** the hyperparameter space
3. You have **compute budget** for 50-100+ trials
4. You need to **adapt to a new problem domain**

### ❌ Don't Use Optuna YET:
1. Standard PPO settings haven't been tried (we just fixed this!)
2. Your environment has bugs (fix env first)
3. Limited compute resources

## Optuna Implementation (if needed later)

### Search Space (Conservative)
Start with these ranges based on stable PPO literature:

```python
import optuna
from optuna.pruners import MedianPruner

def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    n_epochs = trial.suggest_int("n_epochs", 3, 6)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)
    vf_coef = trial.suggest_float("vf_coef", 0.25, 0.75)
    
    # Constraint: batch_size should divide (n_envs × n_steps) evenly
    n_envs = 8
    total_samples = n_envs * n_steps
    if total_samples % batch_size != 0:
        raise optuna.TrialPruned()
    
    # Train model with these hyperparameters
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        # ... other fixed params
    )
    
    # Train for reduced timesteps (e.g., 50k instead of 100k)
    model.learn(total_timesteps=50000)
    
    # Evaluate on test scenarios
    mean_makespan = evaluate_model(model)
    
    return mean_makespan  # Optuna minimizes this

# Run optimization
study = optuna.create_study(
    direction="minimize",
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)
study.optimize(objective, n_trials=50, timeout=3600*10)  # 10 hours

print(f"Best hyperparameters: {study.best_params}")
print(f"Best makespan: {study.best_value}")
```

### Optuna Best Practices
1. **Start with 10-20 trials** to get a sense of the landscape
2. **Use pruning** to stop bad trials early (saves compute)
3. **Reduce training timesteps** during search (e.g., 50k instead of 100k)
4. **Fix random seeds** for fair comparison
5. **Log to TensorBoard** for visualization

## Expected Results with Optimized Settings

### Training Metrics You Should See:
1. **Mean episode reward**: Steady improvement (already working)
2. **Policy gradient loss**: 
   - Should **decrease overall** (not flat oscillation)
   - Some jitter is normal, but should trend down
   - Magnitude should be < 0.01 after initial phase
3. **Value loss**: Should decrease and stabilize
4. **Total loss**: Should decrease steadily
5. **Entropy**: Should gradually decrease (policy becoming more deterministic)
6. **KL divergence**: Should stay < 0.03 (policy updates not too large)

### Warning Signs:
- **Policy loss increasing**: Learning rate too high
- **Policy loss not decreasing**: Batch size too small or n_epochs too high
- **Rewards improving but policy loss flat**: What you had - fixed now!
- **KL divergence > 0.1**: Clip range too large or learning rate too high

## Recommended Testing Sequence

1. **Test current optimized settings** (run training once)
   - Check if policy gradient loss is smoother
   - Compare final makespan to previous results

2. **If still problematic**:
   - Try `batch_size = 1024` (even larger batches)
   - Try `learning_rate = 5e-5` (even lower)
   - Try `n_epochs = 3` (fewer epochs)

3. **Only if manual tuning fails**:
   - Run Optuna search (50 trials)
   - Use best parameters for final training

## References
- [Stable-Baselines3 PPO Guide](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Optuna SB3 Integration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/train.py)
