from SelfPlayPokerGame import SelfPlayPokerGame
from BayesianHoldem import BayesianHoldem
import torch
import os
from typing import Optional
import glob

class CheckpointTester(SelfPlayPokerGame):
    def __init__(self, checkpoint_path: str):
        """
        Initialize the tester with a checkpoint model and a fresh model.
        
        Args:
            checkpoint_path: Path to the directory containing checkpoints
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Player0 with the most recent checkpoint
        self.Player0 = BayesianHoldem().to(self.device)
        self._load_most_recent_checkpoint(checkpoint_path)
        
        # Initialize Player1 as a fresh model
        self.Player1 = BayesianHoldem().to(self.device)
        
        self.BB = 100
        
        # Pre-allocate tensors for efficiency
        self.p0_action_tensor = torch.zeros((24, 4, 4), device=self.device)
        self.p1_action_tensor = torch.zeros((24, 4, 4), device=self.device)
        
        # Session tracking metrics
        self._init_session_metrics()
    
    def _load_most_recent_checkpoint(self, checkpoint_path: str):
        """Load the most recent checkpoint from the given directory."""
        checkpoint_dir = os.path.join(checkpoint_path, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
        
        # Find all checkpoint files
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_session_*.pt'))
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Get the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Loading checkpoint from {latest_checkpoint}")
        
        # Load the checkpoint
        checkpoint = torch.load(latest_checkpoint)
        self.Player0.load_state_dict(checkpoint['model_state_dict'])
        self.player0_elo = checkpoint['elo']
        print(f"Loaded model from session {checkpoint['session']}")
        print(f"Model metrics:")
        print(f"Win rate: {checkpoint['win_rate']:.4f}")
        print(f"BB/hand: {checkpoint['bb_per_hand']:.4f}")
        print(f"Elo: {self.player0_elo:.1f}")
    
    def load_models_for_training_session(self):
        """
        Load models for a training session. In this case, we don't need to load anything
        since we already have our models set up in __init__.
        """
        pass
    
    def test_games(self, num_games: int = 100, verbose: bool = True) -> dict:
        """
        Run a series of test games between the checkpoint model and a fresh model.
        
        Args:
            num_games: Number of games to play
            verbose: Whether to print detailed game information
            
        Returns:
            dict: Session metrics containing test results
        """
        print(f"\nStarting test session with {num_games} games")
        print(f"Player 0: Checkpoint model (Elo: {self.player0_elo:.1f})")
        print(f"Player 1: Fresh model")
        
        # Run the training session without actual training
        self.run_training_session(
            num_games=num_games,
            verbose=verbose,
        )
        
        # Print final results
        total_games = self.session_metrics['total_player0_wins'] + self.session_metrics['total_player1_wins']
        win_rate = self.session_metrics['total_player0_wins'] / max(1, total_games)
        bb_per_hand = sum(self.session_metrics['session_bb_per_hand']) / max(1, len(self.session_metrics['session_bb_per_hand']))
        
        print("\nTest Results:")
        print(f"Total Games: {total_games}")
        print(f"Checkpoint Model Wins: {self.session_metrics['total_player0_wins']}")
        print(f"Fresh Model Wins: {self.session_metrics['total_player1_wins']}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"BB/Hand: {bb_per_hand:.4f}")
        print(f"Final Elo: {self.player0_elo:.1f}")
        
        return self.session_metrics 