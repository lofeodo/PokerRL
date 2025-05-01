import numpy as np

class CFRNode:
    """
    A node in the game tree for Counterfactual Regret Minimization (CFR).
    """
    
    def __init__(self, info_set, num_actions):
        self.info_set = info_set
        self.num_actions = num_actions
        
        # Initialize strategy and regrets
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.current_strategy = np.ones(num_actions) / num_actions  # Initially uniform
        
    def get_strategy(self, reach_prob):
        """
        Get current strategy using regret-matching.
        """
        # Compute positive regrets
        regret_positive = np.maximum(self.regret_sum, 0)
        regret_sum_positive = np.sum(regret_positive)
        
        # If all regrets are 0 or negative, use uniform strategy
        if regret_sum_positive <= 0:
            return np.ones(self.num_actions) / self.num_actions
        
        # Regret-matching: strategy proportional to positive regrets
        strategy = regret_positive / regret_sum_positive
        
        # Accumulate the weighted strategy for averaging
        self.strategy_sum += reach_prob * strategy
        
        return strategy
    
    def get_average_strategy(self):
        """
        Get the average strategy over all iterations.
        """
        strategy_sum_total = np.sum(self.strategy_sum)
        
        if strategy_sum_total > 0:
            return self.strategy_sum / strategy_sum_total
        else:
            return np.ones(self.num_actions) / self.num_actions
            
    def update_regrets(self, action, instant_regret):
        """
        Update the cumulative regrets.
        """
        self.regret_sum[action] += instant_regret
