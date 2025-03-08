#!/bin/bash
#SBATCH --job-name=DS_test
#SBATCH --output=logs/DS_test_%A.out
#SBATCH --error=logs/DS_test_%A.err
#SBATCH --partition=nodegpupool
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4     # Reduced CPUs
#SBATCH --mem=32G            # Reduced memory
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Enable verbose mode for debugging
set -x
set -e

# Function for timestamped logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "logs/debug_test_${SLURM_JOB_ID}.log"
}

# Create logs directory
mkdir -p logs

# Print environment information
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

# Verify Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log "Python version: $PYTHON_VERSION"

# Test only river street with root_nodes
street=4  # river
log "Testing with river street only"

# Function to run training with enhanced logging
run_training() {
    local street=$1
    local approximate=$2
    
    log "Starting $approximate for river"
    
    {
        python scripts/generate_and_train.py \
            --street $street \
            --approximate $approximate \
            --starting_idx 1
    } 2>&1 | tee "logs/python_output_test_${SLURM_JOB_ID}.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "ERROR: Failed during $approximate for river"
        return 1
    fi
    
    log "Completed $approximate for river"
    return 0
}

# Test only with root_nodes
if ! run_training $street "root_nodes"; then
    log "ERROR: Test failed"
    exit 1
fi

log "Test completed successfully"

# Add debug information about available resources
echo "Available nodes:"
sinfo -N -l
echo "Queue status:"
squeue