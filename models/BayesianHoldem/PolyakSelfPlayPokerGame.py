from SelfPlayPokerGame import SelfPlayPokerGame
from BayesianHoldem import BayesianHoldem
import torch
import os
from typing import Optional

class PolyakSelfPlayPokerGame(SelfPlayPokerGame):
    def __init__(self, learning_rate: float = 0.001, best_model_path: Optional[str] = None, polyak_tau: float = 0.005):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.polyak_tau = polyak_tau
        
        # Initialize Player0 with polyak enabled
        self.Player0 = BayesianHoldem(use_polyak=True).to(self.device)
        self.Player1 = BayesianHoldem().to(self.device)  # This player will use the best previous version
        
        # Load best model for Player1 if available
        if best_model_path and os.path.exists(best_model_path):
            self.Player1.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        
        self.BB = 100
        
        # Pre-allocate tensors for efficiency
        self.p0_action_tensor = torch.zeros((24, 4, 4), device=self.device)
        self.p1_action_tensor = torch.zeros((24, 4, 4), device=self.device)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.best_win_rate = 0.0
        self.best_model_path = best_model_path
        
        # Session tracking metrics
        self.session_metrics = {
            'session_win_rates': [],  # Win rate per session
            'session_total_losses': [],  # Average total loss per session
            'session_policy_losses': [],  # Average policy loss per session
            'session_value_losses': [],  # Average value loss per session
            'session_win_counts': [],  # Number of wins per session
            'session_game_counts': [],  # Number of games per session
            'session_timestamps': []  # Timestamps for each session
        }

    def load_models_for_training_session(self, best_model_path: str, verbose: bool = False):
        """Load Player1 from best model and update Player0's target network."""
        if os.path.exists(best_model_path):
            # Load best model into Player1
            self.Player1.load_state_dict(torch.load(best_model_path))
            
            # Update Player0's target network using polyak averaging
            self.Player0.update_target_network(self.polyak_tau)
            
            if verbose:
                print(f"Loaded best model from {best_model_path} and updated target network")
        else:
            if verbose:
                print(f"No best model found at {best_model_path}") 