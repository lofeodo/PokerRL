#!/bin/bash
#SBATCH --job-name=DeepStack
#SBATCH --output=logs/DS_%A.out
#SBATCH --error=logs/DS_%A.err
#SBATCH --partition=nodegpupool
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10     # 10 CPUs
#SBATCH --mem=64G              # Reasonable memory for this task
#SBATCH --gres=gpu:1           # 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Enable error handling
set -e

# Function for timestamped logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "logs/debug_${SLURM_JOB_ID}.log"
}

# Create logs directory
mkdir -p logs

# Print basic job info
log "=== Job Information ==="
log "Job ID: $SLURM_JOB_ID"
log "Running on node: $(hostname)"
log "Working directory: $(pwd)"

# Setup conda
log "=== Setting up Conda ==="
source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate existing environment
log "Activating pokerrl environment..."
conda activate pokerrl

# Run the main Python script
log "=== Starting Training ==="
python scripts/generate_and_train_all.py 2>&1 | tee -a "logs/debug_${SLURM_JOB_ID}.log"

log "=== Training Complete ==="