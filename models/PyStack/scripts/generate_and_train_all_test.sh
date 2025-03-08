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
log "Conda version: $(conda --version)"
log "Checking conda initialization..."
source ~/.bashrc  # Make sure conda is properly initialized

# Check if YAML file exists
if [ ! -f "pokerrl_env.yaml" ]; then
    log "ERROR: pokerrl_env.yaml not found in $(pwd)"
    ls -l  # List directory contents
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "pokerrl"; then
    log "Removing existing pokerrl environment..."
    conda env remove -n pokerrl -y
    if [ $? -ne 0 ]; then
        log "ERROR: Failed to remove existing environment"
        exit 1
    fi
fi

# Create new environment from YAML with detailed output
log "Creating pokerrl environment from YAML..."
conda env create -f pokerrl_env.yaml --verbose
if [ $? -ne 0 ]; then
    log "ERROR: Failed to create conda environment"
    log "Conda debug info:"
    conda info
    log "YAML file contents:"
    cat pokerrl_env.yaml
    exit 1
fi

# Verify Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    log "ERROR: Wrong Python version $PYTHON_VERSION. Expected 3.10"
    exit 1
fi

# Print Python environment info
log "=== Python Environment ==="
log "Python path: $(which python)"
log "Python version: $(python --version)"
log "Conda environment: $CONDA_DEFAULT_ENV"

# Test only river street with root_nodes
street=4  # river
log "Testing with river street only"

# Function to run training with enhanced logging
run_training() {
    local street=$1
    local approximate=$2
    
    log "Starting $approximate for river"
    
    # Run the Python script with minimal epochs
    {
        python scripts/generate_and_train.py \
            --street $street \
            --approximate $approximate \
            --starting_idx 1 \
            --epochs 1
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