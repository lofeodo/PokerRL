import os
import torch
import argparse
from SelfPlayPokerGame import SelfPlayPokerGame

def main():
    parser = argparse.ArgumentParser(description='Train poker AI using hybrid NN+CFR model')
    parser.add_argument('--sessions', type=int, default=5, help='Number of training sessions')
    parser.add_argument('--games', type=int, default=500, help='Games per session')
    parser.add_argument('--save_path', type=str, default='./models', help='Path to save models')
    parser.add_argument('--plot_path', type=str, default='./plots', help='Path to save training plots')
    parser.add_argument('--use_cfr', action='store_true', help='Enable CFR-guided training')
    parser.add_argument('--cfr_blueprint', type=str, default=None, help='Path to CFR blueprint')
    parser.add_argument('--best_model', type=str, default=None, help='Path to best model to start from')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Print detailed training info')
    
    args = parser.parse_args()
    
    # Create save directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.plot_path, exist_ok=True)
    
    print("Initializing SelfPlayPokerGame...")
    print(f"CFR-guided training: {'Enabled' if args.use_cfr else 'Disabled'}")
    if args.cfr_blueprint:
        print(f"Using CFR blueprint from: {args.cfr_blueprint}")
    
    # Initialize the game
    game = SelfPlayPokerGame(
        learning_rate=args.learning_rate,
        best_model_path=args.best_model,
        cfr_blueprint_path=args.cfr_blueprint,
        use_cfr=args.use_cfr
    )
    
    print("\nStarting training...")
    # Train the model
    session_metrics = game.train(
        num_sessions=args.sessions,
        games_per_session=args.games,
        save_path=args.save_path,
        plot_path=args.plot_path,
        verbose=args.verbose
    )
    
    # Save final metrics
    torch.save(session_metrics, os.path.join(args.save_path, 'final_metrics.pt'))
    
    print("\nTraining complete!")
    print(f"Final models and metrics saved to: {args.save_path}")
    print(f"Training plots saved to: {args.plot_path}")

if __name__ == "__main__":
    main()