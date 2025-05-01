from collections import defaultdict
import numpy as np

class OpponentModel:
    """
    Models opponent strategies based on observed actions.
    Used to adapt to opponents during play.
    """
    
    def __init__(self):
        # Track observed actions by info set
        self.observed_actions = defaultdict(lambda: np.zeros(3))
        
        # Track estimated strategy for each info set
        self.estimated_strategies = {}
        
    def observe_action(self, info_set, action):
        """Record an observed opponent action."""
        self.observed_actions[info_set][action] += 1
        
        # Update estimated strategy
        count = self.observed_actions[info_set]
        total = np.sum(count)
        
        if total > 0:
            self.estimated_strategies[info_set] = count / total
    
    def get_strategy(self, info_set):
        """Get estimated strategy for an information set."""
        if info_set in self.estimated_strategies:
            return self.estimated_strategies[info_set]
        else:
            # Return uniform strategy if no observations
            return np.ones(3) / 3
