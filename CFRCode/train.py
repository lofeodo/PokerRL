import os
import time
import pickle
import torch
from tqdm import tqdm
import sys

# Add path to import modules if needed
if '/content/assignment' not in sys.path:
    sys.path.insert(0, '/content/assignment')

from depthLimitedCFR import DepthLimitedCFR
from pokerGameState import PokerGameState

def save_strategy(solver, path):
    """Save the strategy to a file."""
    avg_strategy = solver.get_average_strategy()
    with open(path, 'wb') as f:
        pickle.dump(avg_strategy, f)
    print(f"Strategy saved to {path} ({len(avg_strategy)} information sets)")
    return path

def train_blueprint(
    num_iterations=20000,
    save_every=1000,
    save_path='/content/gdrive/MyDrive/CFRCode/model',
    max_depth=2,
    batch_size=16
):
    """
    Train blueprint strategy and save checkpoints regularly.
    
    Args:
        num_iterations: Total number of CFR iterations
        save_every: Save checkpoint every N iterations
        save_path: Base path for saving checkpoints and strategy
        max_depth: Maximum depth for CFR search
        batch_size: Batch size for each training step
    """
    print(f"Starting blueprint training for {num_iterations} iterations")
    
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize CFR solver
    cfr = DepthLimitedCFR(max_depth=max_depth)
    print(f"Using device: {cfr.device}")
    
    # Create base directory if it doesn't exist
    base_dir = os.path.dirname(save_path)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
    
    # Add timestamp to save path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_path}_{timestamp}"
    
    # Main training loop with progress bar
    start_time = time.time()
    try:
        with tqdm(total=num_iterations) as pbar:
            for i in range(num_iterations):
                # Process a batch of game states
                for j in range(batch_size):
                    # Initialize a new game state
                    state = PokerGameState()
                    
                    # Deal cards
                    state.deal_hole_cards()
                    
                    # Run CFR from this state
                    reach_probs = torch.ones(2, device=cfr.device)
                    cfr.cfr(state, reach_probs, iteration=i+j+1)
                
                # Update progress bar
                pbar.update(1)
                
                # Display stats in progress bar
                if (i + 1) % 10 == 0:
                    if torch.cuda.is_available():
                        mem = torch.cuda.memory_allocated() / 1024**2
                        pbar.set_postfix({
                            'nodes': len(cfr.nodes),
                            'mem': f"{mem:.1f}MB"
                        })
                    else:
                        pbar.set_postfix({'nodes': len(cfr.nodes)})
                
                # Save checkpoint periodically
                if (i + 1) % save_every == 0 or i == num_iterations - 1:
                    checkpoint_path = f"{save_path}_iter_{i+1}.pkl"
                    save_strategy(cfr, checkpoint_path)
                    
                    # Clear CUDA cache to prevent memory fragmentation
                    if torch.cuda.is_available() and (i + 1) % 5000 == 0:
                        torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        save_strategy(cfr, f"{save_path}_interrupted.pkl")
    
    # Final save
    final_path = f"{save_path}_final.pkl"
    save_strategy(cfr, final_path)
    
    # Print training statistics
    training_time = time.time() - start_time
    iterations_per_second = num_iterations / max(0.1, training_time)
    print(f"Training completed in {training_time:.2f}s ({iterations_per_second:.1f} it/s)")
    print(f"Final nodes in tree: {len(cfr.nodes)}")
    
    return cfr, final_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train poker blueprint strategy")
    parser.add_argument("--iterations", type=int, default=20000,
                        help="Number of iterations to train")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--save-path", type=str, default="/content/gdrive/MyDrive/CFRCode/blueprint",
                        help="Path to save the model")
    parser.add_argument("--depth", type=int, default=2,
                        help="Maximum search depth")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    
    args = parser.parse_args()
    
    # Run training
    cfr, final_path = train_blueprint(
        num_iterations=args.iterations,
        save_every=args.save_every,
        save_path=args.save_path,
        max_depth=args.depth,
        batch_size=args.batch_size
    )
    
    print(f"Final model saved to: {final_path}")