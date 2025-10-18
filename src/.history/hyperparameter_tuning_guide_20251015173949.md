# Hyperparameter Tuning Guide for PPO Training

## Problem Identified
Your training showed:
- ✅ **Episode rewards improving** (-70 → -40 makespan)
- ❌ **Policy gradient loss highly jittering** (oscillating around 0)

## Root Cause
**Too many gradient updates on the same data**:
```python
# OLD (problematic) settings:
n_envs = 8
n_steps = 256
batch_size = 128      # TOO SMALL
n_epochs = 10         # TOO MANY

# This resulted in:
# - Total samples per rollout: 8 × 256 = 2048
# - Minibatches per epoch: 2048 / 128 = 16
# - Total updates per rollout: 16 × 10 = 160 updates (EXCESSIVE!)
```

## Optimized Settings Applied

### Static RL (train_static_agent)
```python
learning_rate = 1e-4      # Reduced from 3e-4 for stability
n_steps = 256             # Keep: 8 × 256 = 2048 samples
batch_size = 512          # INCREASED: only 4 minibatches now
n_epochs = 4              # REDUCED: standard PPO
gae_lambda = 0.95         # Slightly reduced variance
```

**Expected improvements:**
- Smoother policy gradient loss (less jitter)
- More stable learning
- Better generalization

### Perfect Knowledge RL (train_perfect_knowledge_agent)
```python
learning_rate = 1e-4
n_steps = 256             # INCREASED from 128
batch_size = 512
n_epochs = 4
gae_lambda = 0.95
```

### Dynamic RL (train_dynamic_agent)
```python
learning_rate = 1e-4
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
