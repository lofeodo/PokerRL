import subprocess
import os
import sys
from datetime import datetime

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_combination(street, approximate, start_idx=1):
    street_names = {1: "preflop", 2: "flop", 3: "turn", 4: "river"}
    log_message(f"Starting {street_names[street]} with {approximate}")
    
    base_dir = os.getcwd()
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(base_dir, 'src')
    
    try:
        result = subprocess.run([
            'python',
            os.path.join(base_dir, 'scripts', 'generate_and_train.py'),
            '--street', str(street),
            '--start-idx', str(start_idx),
            '--approximate', str(approximate),
        ], env=env, check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"Error processing {street_names[street]} with {approximate}: {str(e)}")
        return False
    except Exception as e:
        log_message(f"Unexpected error for {street_names[street]} with {approximate}: {str(e)}")
        return False

def main():
    # Configuration
    #streets = [4, 3, 2, 1]
    streets = [3, 2, 1]
    approximates = ['root_nodes', 'leaf_nodes']
    start_idx = 1
    
    # Track successful and failed runs
    successful_runs = []
    failed_runs = []
    
    log_message("Starting processing of all combinations")
    
    # Process each combination
    for street in streets:
        for approximate in approximates:
            # Skip leaf nodes for river street since it's not needed
            if street == 4 and approximate == 'leaf_nodes':
                continue
                
            street_name = {1: "preflop", 2: "flop", 3: "turn", 4: "river"}[street]
            log_message(f"Processing street {street_name} with {approximate}")
            
            if run_combination(street, approximate, start_idx):
                successful_runs.append((street_name, approximate))
            else:
                failed_runs.append((street_name, approximate))
                # If a street fails, we should probably stop since later streets depend on it
                log_message(f"Failed to process {street_name}. Stopping since later streets depend on this one.")
                break
        
        if failed_runs:  # If we had a failure, stop processing further streets
            break
    # Print summary
    log_message("="*50)
    log_message("EXECUTION SUMMARY")
    log_message("="*50)
    
    log_message("\nSuccessful runs:")
    for street_name, approximate in successful_runs:
        log_message(f"- {street_name} with {approximate}")
    
    log_message("\nFailed runs:")
    for street_name, approximate in failed_runs:
        log_message(f"- {street_name} with {approximate}")
    
    # Exit with error if any runs failed
    if failed_runs:
        log_message("\nSome runs failed!")
        sys.exit(1)
    else:
        log_message("\nAll runs completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()