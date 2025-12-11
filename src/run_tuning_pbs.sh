#!/bin/bash
#PBS -N rl_tuning
#PBS -l walltime=04:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=16gb
#PBS -j oe
#PBS -o tuning_$PBS_JOBID.log
#PBS -m abe
#PBS -M your.email@university.edu

# =====================================================
# PBS/Torque Job Script for Hyperparameter Tuning
# =====================================================
#
# Usage:
#   qsub run_tuning_pbs.sh
#
# Monitor:
#   qstat -u $USER
#   tail -f tuning_<job_id>.log
#
# Cancel:
#   qdel <job_id>
# =====================================================

echo "=================================="
echo "Job started: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "=================================="

# Change to submission directory
cd $PBS_O_WORKDIR || exit 1

# Load required modules (adjust for your cluster)
module purge
module load python/3.9
module load cuda/11.8  # If using GPU

# Activate virtual environment
source ~/envs/drl/bin/activate

# OR if using conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate drl

# Print environment info
echo ""
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo ""

# Set CPU threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Run tuning
echo "Starting hyperparameter tuning..."
echo ""

python tune_rule_based_hyperparameters.py

EXIT_CODE=$?

echo ""
echo "=================================="
echo "Job finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================="

# Show results
if [ -f rule_based_best_hyperparameters.json ]; then
    echo ""
    echo "BEST HYPERPARAMETERS:"
    cat rule_based_best_hyperparameters.json | python -m json.tool
fi

exit $EXIT_CODE
