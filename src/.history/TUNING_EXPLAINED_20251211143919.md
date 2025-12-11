# Hyperparameter Tuning: Complete Explanation

## 🎯 What This Script Does

The `tune_rule_based_hyperparameters.py` script uses **Bayesian optimization** (via Optuna) to automatically find the best hyperparameters for your Rule-Based RL agent.

### Strategy Overview

```
For each trial (30 trials total):
  1. Optuna suggests a hyperparameter combination
  2. Train RL agent for 200k timesteps (quick test)
  3. Evaluate on 5 test scenarios
  4. Record average makespan
  5. If trial is bad, prune early (stop wasting time)

After all trials:
  → Return the configuration with lowest makespan
```

---

## ⏱️ Time Expectations

### Quick Calculation

**Per Trial:**
- Training: 200k timesteps ≈ **2-5 minutes** (depends on CPU/GPU)
- Evaluation: 5 scenarios ≈ **1-2 minutes**
- **Total per trial: ~3-7 minutes**

**Full Run:**
- 30 trials × 5 minutes/trial = **~2.5 hours** (average)
- With pruning (stopping bad trials early): **~1.5-2 hours**
- Worst case (slow machine): **~3-4 hours**

### Configuration Variables

You can adjust these in the script:

```python
TIMESTEPS_PER_TRIAL = 200_000  # ↓ Reduce for faster (less accurate)
N_TRIALS = 30                   # ↓ Reduce for faster (less thorough)
N_EVAL_SCENARIOS = 5            # ↓ Reduce for faster (less robust)
```

**Quick test (30 minutes):**
```python
TIMESTEPS_PER_TRIAL = 50_000
N_TRIALS = 10
N_EVAL_SCENARIOS = 3
```

**Thorough search (8-10 hours):**
```python
TIMESTEPS_PER_TRIAL = 500_000
N_TRIALS = 50
N_EVAL_SCENARIOS = 10
```

---

## 📊 What You'll See During Execution

### 1. Initial Banner

```
======================================================================
Automated Hyperparameter Tuning for Rule-Based RL
======================================================================

📊 Configuration:
   - Trials: 30
   - Timesteps per trial: 200,000
   - Eval scenarios: 5
   - Dataset: 20 jobs, 5 machines
   - Arrival rate: 0.1

🎯 Goal: Minimize average makespan on test scenarios

⏱️  Estimated time: 120-240 minutes

======================================================================
```

### 2. Progress Bar

```
 23%|████████████▎                        | 7/30 [00:26<01:17,  3.35s/it]
```

Shows:
- Progress percentage
- Trials completed / total
- Time elapsed / estimated remaining
- Speed (seconds per trial)

### 3. Real-Time Updates

```
✅ Trial 5: Avg Makespan = 245.67
[I 2025-12-11 14:25:30,123] Trial 5 finished with value: 245.67
```

For each trial:
- ✅ Success with makespan value
- ❌ Failure with error message
- 🔪 Pruned (stopped early because it's clearly bad)

### 4. Final Results

```
======================================================================
Tuning Complete!
======================================================================

⏱️  Total time: 134.2 minutes

✅ Completed trials: 28/30

🏆 Best trial: #17
   - Best makespan: 238.45

📋 Best hyperparameters:
   - learning_rate: 0.00015
   - clip_range: 0.08
   - ent_coef: 0.045
   - n_epochs: 12
   - batch_size: 512
   - n_steps: 2048
   - gae_lambda: 0.92
   - max_grad_norm: 0.45
   - net_arch: medium

💾 Results saved to: rule_based_best_hyperparameters.json

📊 Generating optimization plots...
   ✅ Saved: optimization_history.html
   ✅ Saved: param_importances.html
   ✅ Saved: param_slice.html

   💡 Open these HTML files in your browser to view interactive plots!
```

---

## 📁 Output Files

### 1. `rule_based_best_hyperparameters.json`

```json
{
  "best_makespan": 238.45,
  "best_trial": 17,
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
  "tuning_time_minutes": 134.2,
  "config": {
    "timesteps_per_trial": 200000,
    "n_eval_scenarios": 5,
    "n_jobs": 20,
    "n_machines": 5,
    "arrival_rate": 0.1
  }
}
```

**Use this to update your training code!**

### 2. `rule_based_tuning.db`

SQLite database with all trial data. Can:
- Resume interrupted runs
- View in dashboard: `optuna-dashboard sqlite:///rule_based_tuning.db`
- Query programmatically

### 3. Interactive HTML Plots

**optimization_history.html:**
- Shows makespan improving over trials
- Identifies when best configuration was found

**param_importances.html:**
- Bar chart showing which hyperparameters matter most
- Example: "learning_rate is 3x more important than batch_size"

**param_slice.html:**
- Shows how each parameter affects makespan
- Helps understand parameter interactions

---

## 🔬 Tuning Strategy Details

### Bayesian Optimization (TPE Sampler)

Optuna uses **Tree-structured Parzen Estimator (TPE)**:

1. **Random trials** (first 5): Explore broadly
2. **Guided trials** (rest): Focus on promising regions
3. **Model building**: Learn which parameters → good results
4. **Smart sampling**: Suggest parameters likely to improve

**Better than grid search:**
- Grid: 5×4×3×4×3×3×3×3×3 = 233,280 combinations (impossible!)
- TPE: 30 smart trials → finds near-optimal in <1% of time

### Pruning Strategy (Median Pruner)

Stops bad trials early:

```python
MedianPruner(n_startup_trials=5, n_warmup_steps=5)
```

**Logic:**
- After 5 startup trials, we know typical performance
- During training, check every 10k timesteps
- If current trial is worse than median → **PRUNE** (stop wasting time)

**Example:**
- Trial 12 gets makespan = 350 after 50k timesteps
- Median at 50k = 270
- Trial 12 is clearly bad → **PRUNE**
- Saved ~4 minutes!

### Search Space

| Parameter | Type | Range | Why |
|-----------|------|-------|-----|
| learning_rate | Log | 1e-5 to 1e-3 | Exponential scale (10x changes) |
| clip_range | Linear | 0.05 to 0.3 | PPO trust region size |
| ent_coef | Log | 0.001 to 0.1 | Exploration bonus |
| n_epochs | Categorical | [5,10,15,20] | Gradient updates per batch |
| batch_size | Categorical | [256,512,1024] | Memory vs stability tradeoff |
| n_steps | Categorical | [1024,2048,4096] | Rollout buffer size |
| gae_lambda | Linear | 0.8 to 1.0 | Advantage estimation |
| max_grad_norm | Linear | 0.3 to 1.0 | Gradient clipping |
| net_arch | Categorical | small/medium/large | Network capacity |

**Total combinations:** Billions
**Trials needed:** 20-50 (Bayesian optimization magic!)

---

## 🖥️ Running on a Cluster

### Option 1: Simple SLURM Script

Create `run_tuning.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=hyperparameter_tuning
#SBATCH --output=tuning_%j.log
#SBATCH --error=tuning_%j.err
#SBATCH --time=04:00:00          # 4 hours max
#SBATCH --cpus-per-task=8        # 8 CPU cores
#SBATCH --mem=16G                # 16GB RAM
#SBATCH --partition=gpu          # Use GPU partition (optional)
#SBATCH --gres=gpu:1             # Request 1 GPU (optional)

# Load modules (adjust for your cluster)
module load python/3.9
module load cuda/11.8  # If using GPU

# Activate virtual environment
source ~/envs/drl/bin/activate

# Navigate to project directory
cd /path/to/Scheduling/src

# Run tuning
python tune_rule_based_hyperparameters.py

# Alternative: Run with reduced output
# python tune_rule_based_hyperparameters.py > tuning_output.txt 2>&1

echo "Tuning completed!"
```

**Submit:**
```bash
sbatch run_tuning.sh
```

**Monitor:**
```bash
squeue -u $USER              # Check status
tail -f tuning_12345.log     # Watch progress
```

### Option 2: Parallel Trials (Advanced)

Optuna supports distributed optimization!

**Master node:**
```bash
#!/bin/bash
#SBATCH --job-name=tuning_master
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Create shared database on network storage
export OPTUNA_DB="sqlite:////shared/storage/rule_based_tuning.db"

python tune_rule_based_hyperparameters.py
```

**Worker nodes (submit multiple):**
```bash
#!/bin/bash
#SBATCH --job-name=tuning_worker
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=1-5              # 5 parallel workers

export OPTUNA_DB="sqlite:////shared/storage/rule_based_tuning.db"

python tune_rule_based_hyperparameters.py
```

**Benefit:** 5 workers → 5x faster!

### Option 3: PBS/Torque

Create `run_tuning.pbs`:

```bash
#!/bin/bash
#PBS -N hyperparameter_tuning
#PBS -l walltime=04:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=16gb
#PBS -j oe
#PBS -o tuning_$PBS_JOBID.log

cd $PBS_O_WORKDIR
source ~/envs/drl/bin/activate

python tune_rule_based_hyperparameters.py
```

**Submit:**
```bash
qsub run_tuning.pbs
```

---

## 💡 Monitoring Tips

### 1. Real-Time Dashboard (Recommended!)

In separate terminal or browser:

```bash
# On cluster (setup SSH tunnel first):
ssh -L 8080:localhost:8080 user@cluster.edu

# Then on cluster:
optuna-dashboard sqlite:///rule_based_tuning.db
```

Open browser: `http://localhost:8080`

See:
- Live progress
- Best trial so far
- Parameter importance
- Trial history

### 2. Log File Monitoring

```bash
# Watch last 20 lines
tail -20 tuning_output.txt

# Follow live
tail -f tuning_output.txt

# Search for best results
grep "Best makespan" tuning_output.txt
```

### 3. Check Database

```python
import optuna

study = optuna.load_study(
    study_name="rule_based_rl_tuning",
    storage="sqlite:///rule_based_tuning.db"
)

print(f"Trials completed: {len(study.trials)}")
print(f"Best so far: {study.best_value}")
print(f"Best params: {study.best_params}")
```

---

## 🛑 Interruption & Resumption

### Safe to Interrupt!

Press `Ctrl+C` anytime:
```
^C
⚠️  Tuning interrupted by user!
```

**What happens:**
- Current trial finishes gracefully
- All completed trials saved to database
- Partial results printed

### Resume Later

Just run again:
```bash
python tune_rule_based_hyperparameters.py
```

Optuna automatically:
- Loads previous trials from database
- Continues from where you left off
- Completes remaining trials

**Example:**
- Run interrupted after 15/30 trials
- Resume: Runs trials 16-30
- Final results use all 30 trials

---

## 🎯 After Tuning: Next Steps

### 1. Review Results

```bash
# View JSON
cat rule_based_best_hyperparameters.json | python -m json.tool

# Open interactive plots
open optimization_history.html
open param_importances.html
```

### 2. Update Training Code

In `proactive_sche.py`, update `train_rule_based_agent()`:

```python
model = MaskablePPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=0.00015,  # ← From tuning
    n_steps=2048,           # ← From tuning
    batch_size=512,         # ← From tuning
    n_epochs=12,            # ← From tuning
    # ... etc (copy all tuned values)
)
```

### 3. Full Training

```python
# Train with optimized hyperparameters
python proactive_sche.py
```

Expect:
- ✅ More stable training (less noisy metrics)
- ✅ Faster convergence
- ✅ Better final performance (5-15% improvement)

### 4. Validate

Compare before/after:
- Training curves (KL div, entropy, policy loss)
- Final makespan on test scenarios
- Robustness across different seeds

---

## 📊 Expected Improvements

### Before Tuning (Default Params)

```
Training: Noisy, high variance
KL Divergence: Spikes to 0.025+
Makespan: 250-280 (inconsistent)
```

### After Tuning (Optimized Params)

```
Training: Stable, low variance
KL Divergence: < 0.01 (smooth)
Makespan: 230-250 (consistent)
Improvement: 8-12% better
```

---

## 🔧 Troubleshooting

### "Out of Memory"

Reduce batch size and network size:

```python
# In tune_rule_based_hyperparameters.py
batch_size = trial.suggest_categorical("batch_size", [128, 256])
net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium"])
```

### "Too Slow"

Reduce training budget:

```python
TIMESTEPS_PER_TRIAL = 100_000  # Half the training
N_TRIALS = 20                   # Fewer trials
N_EVAL_SCENARIOS = 3            # Fewer evaluations
```

### "All Trials Fail"

Check:
1. Does `python proactive_sche.py` work normally?
2. Are all dependencies installed? `pip list | grep optuna`
3. Is CUDA available? `python -c "import torch; print(torch.cuda.is_available())"`

---

## 📚 Further Reading

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Stable-Baselines3 Tuning Guide](https://stable-baselines3.readthedocs.io/en/master/guide/tuning.html)
- [PPO Hyperparameters Explained](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

---

## Summary

**Tuning Strategy:** Bayesian optimization (30 trials, ~2-3 hours)

**What You'll See:**
1. Progress bar with trial updates
2. Real-time makespan results
3. Final best hyperparameters
4. Interactive HTML plots

**Outputs:**
- `rule_based_best_hyperparameters.json` (copy to your code)
- `rule_based_tuning.db` (resumable, viewable)
- `*.html` plots (interactive analysis)

**Cluster:** Yes! Use provided SLURM script

**Resumable:** Yes! Ctrl+C anytime, run again to continue

**Worth it?** Absolutely! 2-3 hours → 8-15% better performance
