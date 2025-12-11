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
pip install scikit-optimize
```

### 2. Run Tuning

```bash
# Optuna (recommended)
python tune_rule_based_hyperparameters.py

# Scikit-Optimize (faster)
python tune_rule_based_skopt.py
```

### 3. Monitor Progress

**Optuna real-time dashboard:**
```bash
# In another terminal
optuna-dashboard sqlite:///rule_based_tuning.db
# Open http://localhost:8080 in browser
```

### 4. View Results

Results saved to JSON:
- **Optuna**: `rule_based_best_hyperparameters.json`
- **Scikit-Optimize**: `rule_based_best_hyperparameters_skopt.json`

Interactive plots (Optuna only):
- `optimization_history.html` - Progress over trials
- `param_importances.html` - Most important parameters
- `param_slice.html` - Parameter effects

---

## Customization

### Adjust Search Budget

```python
# In tune_rule_based_hyperparameters.py

# Quick test (faster but less accurate)
TIMESTEPS_PER_TRIAL = 100_000  # 5-10 min/trial
N_TRIALS = 15

# Thorough search (slower but better)
TIMESTEPS_PER_TRIAL = 500_000  # 30-60 min/trial
N_TRIALS = 50
```

### Change Problem Size

```python
# Test on different scenarios
N_JOBS = 30
N_MACHINES = 8
ARRIVAL_RATE = 0.15
```

---

## Understanding Results

### Example Output

```json
{
  "best_makespan": 245.67,
  "best_params": {
    "learning_rate": 0.00015,
    "clip_range": 0.08,
    "ent_coef": 0.045,
    "n_epochs": 12,
    "batch_size": 512,
    "n_steps": 2048,
    "gae_lambda": 0.92,
    "max_grad_norm": 0.45,
    "net_arch": "medium"
  },
  "n_trials": 30,
  "tuning_time_minutes": 45.2
}
```

### Applying Best Parameters

Update `train_rule_based_agent()` in `proactive_sche.py`:

```python
model = MaskablePPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=0.00015,  # From tuning
    n_steps=2048,           # From tuning
    batch_size=512,         # From tuning
    n_epochs=12,            # From tuning
    # ... etc
)
```

---

## Expected Improvements

After tuning, you should see:

### Training Stability
- **KL Divergence** < 0.01 (stable updates)
- **Entropy** gradually decreasing (learning)
- **Policy Loss** smooth, not oscillating

### Performance
- **Makespan** 5-15% better
- **Convergence** faster
- **Robustness** more consistent

---

## Troubleshooting

### Out of Memory
```python
batch_size = trial.suggest_categorical("batch_size", [128, 256])
net_arch = trial.suggest_categorical("net_arch", ["small"])
```

### Too Slow
```python
TIMESTEPS_PER_TRIAL = 100_000
N_EVAL_SCENARIOS = 3
N_TRIALS = 15
```

### All Trials Fail
1. Test `proactive_sche.py` with default parameters first
2. Check dependencies: `pip list | grep -E "optuna|stable-baselines3"`
3. Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Comparison: Optuna vs Scikit-Optimize

| Feature | Optuna | Scikit-Optimize |
|---------|--------|-----------------|
| Pruning | ✅ Yes | ❌ No |
| Dashboard | ✅ Yes | ❌ No |
| Visualizations | ✅ Rich | ⚠️ Basic |
| Resume | ✅ Yes | ⚠️ Manual |
| Speed | Medium | Fast |
| Dependencies | More | Fewer |

**Recommendation**: Start with Optuna for research, use scikit-optimize if you need speed
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
