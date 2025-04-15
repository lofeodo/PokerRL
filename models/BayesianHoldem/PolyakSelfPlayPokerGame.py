from SelfPlayPokerGame import SelfPlayPokerGame
from BayesianHoldem import BayesianHoldem
import torch
import os
from typing import Optional
import random

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
        self._init_session_metrics()

    def load_models_for_training_session(self, save_path: Optional[str] = None):
        """
        Load models for a training session. For polyak self-play, only Player1 is loaded from k-best models.
        Player0 accumulates training from all past sessions.
        
        Args:
            save_path: Directory containing saved models
        """
        if save_path:
            best_models_dir = os.path.join(save_path, 'best_models')
            model_types = ['winrate', 'bbhand', 'elo']
            
            # Load Player1 from k-best
            available_models = [
                m for m in model_types 
                if os.path.exists(os.path.join(best_models_dir, f'best_{m}.pt'))
            ]
            
            if available_models:
                chosen_model = random.choice(available_models)
                model_path = os.path.join(best_models_dir, f'best_{chosen_model}.pt')
                print(f"Loading best {chosen_model} model as Player1")
                
                # Load both model state dict and Elo
                checkpoint = torch.load(model_path)
                self.Player1.load_state_dict(checkpoint['model_state_dict'])
                self.player1_elo = checkpoint['elo']
                
                print(f"Updated Player1 Elo rating to: {self.player1_elo:.1f}")
            else:
                print(f"No best models found in {best_models_dir}")
        else:
            print("No save_path provided") 