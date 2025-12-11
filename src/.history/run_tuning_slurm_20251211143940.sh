#!/bin/bash
#SBATCH --job-name=rl_hyperparameter_tuning
#SBATCH --output=tuning_%j.log
#SBATCH --error=tuning_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@university.edu

# =====================================================
# SLURM Job Script for Hyperparameter Tuning
# =====================================================
#
# Usage:
#   sbatch run_tuning_slurm.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f tuning_<job_id>.log
#
# Cancel:
#   scancel <job_id>
# =====================================================

echo "=================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================="

# Load required modules (adjust for your cluster)
module purge
module load python/3.9
module load cuda/11.8  # If using GPU

# Print loaded modules
echo "Loaded modules:"
module list

# Activate virtual environment (adjust path)
source ~/envs/drl/bin/activate

# OR if using conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate drl

# Verify Python and packages
echo ""
echo "Python version:"
python --version
echo ""
echo "Key packages:"
pip list | grep -E "optuna|stable-baselines3|torch|gymnasium"

# Navigate to project directory (adjust path)
cd /path/to/PhD/Scheduling/src || exit 1

# Verify we're in the right directory
echo ""
echo "Working directory: $(pwd)"
echo "Files present:"
ls -lh tune_rule_based_hyperparameters.py proactive_sche.py

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# If using GPU
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo ""
    echo "GPU available: $CUDA_VISIBLE_DEVICES"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
fi

# Run hyperparameter tuning
echo ""
echo "=================================="
echo "Starting hyperparameter tuning..."
echo "=================================="
echo ""

# Option 1: Standard run with progress output
python tune_rule_based_hyperparameters.py

# Option 2: With detailed logging
# python -u tune_rule_based_hyperparameters.py 2>&1 | tee tuning_detailed.log

# Option 3: Quick test (uncomment and modify script)
# Edit TIMESTEPS_PER_TRIAL=50000, N_TRIALS=10 in the script
# python tune_rule_based_hyperparameters.py

EXIT_CODE=$?

echo ""
echo "=================================="
echo "Job finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================="

# Copy important results to backup location (optional)
if [ $EXIT_CODE -eq 0 ]; then
    echo "Backing up results..."
    mkdir -p ~/hyperparameter_tuning_results
    cp rule_based_best_hyperparameters.json ~/hyperparameter_tuning_results/
    cp rule_based_tuning.db ~/hyperparameter_tuning_results/
    cp *.html ~/hyperparameter_tuning_results/ 2>/dev/null || true
    echo "Results backed up to: ~/hyperparameter_tuning_results/"
fi

# Print final results
if [ -f rule_based_best_hyperparameters.json ]; then
    echo ""
    echo "=================================="
    echo "BEST HYPERPARAMETERS:"
    echo "=================================="
    cat rule_based_best_hyperparameters.json | python -m json.tool
fi

exit $EXIT_CODE
