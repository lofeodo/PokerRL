from SelfPlayPokerGame import SelfPlayPokerGame
from typing import Dict, Optional
import torch
import os
from BayesianHoldem import BayesianHoldem
import random

class VanillaSelfPlayPokerGame(SelfPlayPokerGame):
    def __init__(self, learning_rate: float = 0.001, best_model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Player0 = BayesianHoldem().to(self.device)  # This player will be trained
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

    def load_models_for_training_session(self, save_path: Optional[str] = None):
        """
        Load models for a training session. For vanilla self-play, both players are loaded from k-best models.
        
        Args:
            save_path: Directory containing saved models
        """
        if save_path:
            best_models_dir = os.path.join(save_path, 'best_models')
            model_types = ['winrate', 'bbhand', 'elo']
            
            # Load Player0 from k-best
            available_models = [
                m for m in model_types 
                if os.path.exists(os.path.join(best_models_dir, f'best_{m}.pt'))
            ]
            
            if available_models:
                chosen_model = random.choice(available_models)
                model_path = os.path.join(best_models_dir, f'best_{chosen_model}.pt')
                print(f"\nLoading best {chosen_model} model as Player0")
                self.Player0.load_state_dict(torch.load(model_path))
            
            # Load Player1 from k-best
            available_models = [
                m for m in model_types 
                if os.path.exists(os.path.join(best_models_dir, f'best_{m}.pt'))
            ]
            
            if available_models:
                chosen_model = random.choice(available_models)
                model_path = os.path.join(best_models_dir, f'best_{chosen_model}.pt')
                print(f"\nLoading best {chosen_model} model as Player1")
                self.Player1.load_state_dict(torch.load(model_path))

