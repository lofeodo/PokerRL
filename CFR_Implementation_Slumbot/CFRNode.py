import torch
import numpy as np

class CFRNode:
    """
    A node in the game tree for Counterfactual Regret Minimization (CFR),
    implemented with GPU support using PyTorch.
    """
    
    def __init__(self, info_set, num_actions):
        self.info_set = info_set
        self.num_actions = num_actions
        
        # Use GPU if available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize strategy and regrets as PyTorch tensors
        self.regret_sum = torch.zeros(num_actions, device=self.device)
        self.strategy_sum = torch.zeros(num_actions, device=self.device)
        self.current_strategy = torch.ones(num_actions, device=self.device) / num_actions  # Initially uniform
        
    def get_strategy(self, reach_prob):
        """
        Get current strategy using regret-matching.
        """
        # Convert reach_prob to tensor if it's not already
        if not isinstance(reach_prob, torch.Tensor):
            reach_prob = torch.tensor(reach_prob, device=self.device)
            
        # Compute positive regrets
        regret_positive = torch.maximum(self.regret_sum, torch.zeros_like(self.regret_sum))
        regret_sum_positive = torch.sum(regret_positive)
        
        # If all regrets are 0 or negative, use uniform strategy
        if regret_sum_positive <= 0:
            return torch.ones(self.num_actions, device=self.device) / self.num_actions
        
        # Regret-matching: strategy proportional to positive regrets
        strategy = regret_positive / regret_sum_positive
        
        # Accumulate the weighted strategy for averaging
        self.strategy_sum += reach_prob * strategy
        
        return strategy
    
    def get_average_strategy(self):
        """
        Get the average strategy over all iterations.
        """
        strategy_sum_total = torch.sum(self.strategy_sum)
        
        if strategy_sum_total > 0:
            # Return the normalized strategy
            return self.strategy_sum / strategy_sum_total
        else:
            # Return uniform strategy if no accumulated strategy
            return torch.ones(self.num_actions, device=self.device) / self.num_actions
            
    def update_regrets(self, action, instant_regret):
        """
        Update the cumulative regrets.
        """
        # Convert instant_regret to tensor if it's not already
        if not isinstance(instant_regret, torch.Tensor):
            instant_regret = torch.tensor(instant_regret, device=self.device)
            
        self.regret_sum[action] += instant_regret
        
    def get_numpy_strategy(self):
        """
        Get the average strategy as a NumPy array (for compatibility).
        """
        return self.get_average_strategy().cpu().numpy()