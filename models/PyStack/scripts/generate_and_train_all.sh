#!/bin/bash
#SBATCH --job-name=DeepStack_train
#SBATCH --output=logs/DeepStack_train_%A_%a.out  # %A is job ID, %a is array index
#SBATCH --error=logs/DeepStack_train_%A_%a.err
#SBATCH --partition=nodegpupool
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12     # Use all CPUs on one node
#SBATCH --mem=120G             # Almost all memory on one node (122880MB ≈ 120GB)
#SBATCH --gres=gpu:1
#SBATCH --array=1-4            # One task per node
#SBATCH --nodes=1              # Request one full node per task
#SBATCH --ntasks=1             # One task per job

# Enable verbose mode for debugging
set -x  # Print each command before executing
set -e  # Exit on any error

# Function for timestamped logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "logs/debug_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
}

# Create logs directory
mkdir -p logs

# Print environment information
log "=== Job Information ==="
log "Job ID: $SLURM_JOB_ID"
log "Array Task ID: $SLURM_ARRAY_TASK_ID"
log "Running on node: $(hostname)"
log "Working directory: $(pwd)"

# Setup conda
log "=== Setting up Conda ==="
eval "$(conda shell.bash hook)"

# Remove existing environment if it exists
if conda env list | grep -q "pokerrl"; then
    log "Removing existing pokerrl environment..."
    conda env remove -n pokerrl
fi

# Create new environment from YAML
log "Creating pokerrl environment from YAML..."
conda env create -f pokerrl_env.yaml
conda activate pokerrl

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
pip list >> "logs/pip_packages_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"

# Map street numbers to names
declare -A street_names
street_names[1]="preflop"
street_names[2]="flop"
street_names[3]="turn"
street_names[4]="river"

street=$SLURM_ARRAY_TASK_ID
log "Processing street ${street_names[$street]} (street $street)"

# Function to run training with enhanced logging
run_training() {
    local street=$1
    local approximate=$2
    
    log "Starting $approximate for street ${street_names[$street]}"
    
    # Run the Python script and capture both stdout and stderr
    {
        python scripts/generate_and_train.py --street $street --approximate $approximate
    } 2>&1 | tee "logs/python_output_${street}_${approximate}_${SLURM_ARRAY_JOB_ID}.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "ERROR: Failed during $approximate for street ${street_names[$street]}"
        return 1
    fi
    
    log "Completed $approximate for street ${street_names[$street]}"
    return 0
}

# Run training steps
for approximate in "root_nodes" "leaf_nodes"; do
    if ! run_training $street "$approximate"; then
        log "ERROR: Training failed at $approximate"
        exit 1
    fi
done

log "Successfully completed all training for street ${street_names[$street]}"

# Save completion status
echo "Street $street completed at $(date)" >> logs/completion_status.log

# Watch the main output file
tail -f logs/poker_train_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out

# Watch the error file
tail -f logs/poker_train_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err

# Watch debug log
tail -f logs/debug_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log