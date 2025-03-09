import json
import numpy as np
from collections import defaultdict
import random
import pickle
import os


""" une simple stratégie en termes de %, qui ne prend pas en compte les cartes, les positions, etc.
pour chaque information set, on a 3 valeurs: regret_sum, strategy_sum, current_strategy
regret_sum: somme des regrets pour chaque action
strategy_sum: somme des stratégies pour chaque action
current_strategy: stratégie actuelle pour chaque action
exemple: {'flop_3_50': [array([0., 0., 0.]), array([0., 0., 0.]), array([0., 0., 0.])}
améliorer la stratégie en utilisant des meilleures données et des infos sur nos cartes """

class BlueprintStrategy:
    """
    Blueprint strategy implementation using Counterfactual Regret Minimization (CFR)
    for poker games. 
    """
    
    def __init__(self, num_actions=3):
        # Information sets: map of infoset -> (regret_sum, strategy_sum, current_strategy)
        self.infosets = defaultdict(lambda: [np.zeros(num_actions), np.zeros(num_actions), np.zeros(num_actions)]) #example: {'flop_3_50': [array([0., 0., 0.]), array([0., 0., 0.]), array([0., 0., 0.])]}
        self.num_actions = num_actions
        self.iterations = 0
        
    def get_strategy(self, infoset):
        """Get current strategy for an information set"""
        regret_sum, _, _ = self.infosets[infoset]
        
        # Compute positive regrets
        regret_positive = np.maximum(regret_sum, 0)
        regret_sum_positive = np.sum(regret_positive)
        
        # If all regrets are 0 or negative, use uniform strategy
        if regret_sum_positive <= 0:
            return np.ones(self.num_actions) / self.num_actions
        
        # Regret-matching: strategy proportional to positive regrets
        strategy = regret_positive / regret_sum_positive
        return strategy
    
    def get_average_strategy(self, infoset):
        """Get average strategy for an information set"""
        _, strategy_sum, _ = self.infosets[infoset]
        
        # If no strategy recorded yet, use uniform strategy
        sum_strategy = np.sum(strategy_sum)
        if sum_strategy <= 0:
            return np.ones(self.num_actions) / self.num_actions
            
        # Normalize the average strategy
        avg_strategy = strategy_sum / sum_strategy
        return avg_strategy
        
    def update(self, infoset, action, regret, reach_prob):
        """Update regrets and strategy sums for an information set"""
        regret_sum, strategy_sum, current_strategy = self.infosets[infoset]
        
        # Update regret sum
        regret_sum[action] += regret
        
        # Update strategy sum (weighted by reach probability)
        strategy_sum += reach_prob * current_strategy
        
        # Update current strategy based on new regrets
        self.infosets[infoset][2] = self.get_strategy(infoset)
    
    def train_from_data(self, poker_hands, num_iterations=1000):
        """
        Train blueprint strategy from historical poker hand data using Monte Carlo CFR
        """
        self.iterations = num_iterations
        
        for iteration in range(num_iterations):
            if iteration % 100 == 0:
                print(f"CFR Iteration {iteration}")
            
            # Sample a random hand from data
            hand = random.choice(poker_hands)
            
            # For simplicity, we'll focus on training from flop to river decisions
            if "flop" not in hand or hand["flop"]["num_players"] < 2:
                continue
                
            # Simulate the game with CFR, focusing on betting round transitions
            self._cfr_iteration(hand)
                
        print(f"Training completed after {num_iterations} iterations.")
    
    def _cfr_iteration(self, hand):
        """Perform one iteration of CFR on a sample hand"""
        # For actual implementation, you would perform a full tree traversal with CFR
        # This is a simplified version that would need to be expanded
        
        # Extract features to define information sets
        board = hand.get("board", [])
        
        # Track betting rounds 
        rounds = []
        for round_name in ["flop", "turn", "river"]:
            if round_name in hand and hand[round_name]["num_players"] >= 2:
                rounds.append((round_name, hand[round_name]["pot_size"]))
        
        # Skip hands with insufficient data
        if len(rounds) < 2:
            return
            
        # Simulate decisions at each transition between rounds
        for i in range(len(rounds) - 1):
            current_round, current_pot = rounds[i]
            next_round, next_pot = rounds[i+1]
            
            # Pot growth indicates betting action
            pot_growth = next_pot - current_pot
            
            # Create a simple information set representation
            # In a real implementation, this would include cards, positions, etc.
            infoset = f"{current_round}_{len(board[:i+3])}_{current_pot}"
            
            # Simplified action mapping (fold/check, call, raise)
            if pot_growth == 0:
                action = 0  # check/fold
            elif pot_growth <= 20:
                action = 1  # call
            else:
                action = 2  # raise
                
            # In a real CFR implementation, we would compute counterfactual values
            # Here we use a simplified update based on pot growth as a proxy for reward
            reach_prob = 1.0 / (self.iterations + 1)  # Simplified reach probability
            
            # Simple regret calculation (this is very simplified)
            regret = pot_growth / 100  # Simplified regret calculation
            
            # Update regrets and strategy
            self.update(infoset, action, regret, reach_prob)
    
    def save(self, filename="blueprint_strategy.pkl"):
        """Save the blueprint strategy to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.infosets), f)
        print(f"Strategy saved to {filename}")
    
    def load(self, filename="blueprint_strategy.pkl"):
        """Load a blueprint strategy from a file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.infosets = defaultdict(lambda: [np.zeros(self.num_actions), 
                                                    np.zeros(self.num_actions), 
                                                    np.zeros(self.num_actions)])
                loaded_infosets = pickle.load(f)
                for k, v in loaded_infosets.items():
                    self.infosets[k] = v
            print(f"Strategy loaded from {filename}")
            return True
        return False

    def print_strategy(self, top_n=10):
        """Print the average strategy for the top N most visited information sets"""
        infosets_list = [(k, np.sum(v[1])) for k, v in self.infosets.items()]
        infosets_list.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop information sets by visit count:")
        for i, (infoset, visit_count) in enumerate(infosets_list[:top_n]):
            if visit_count > 0:
                avg_strategy = self.get_average_strategy(infoset)
                actions = ["Fold/Check", "Call", "Raise"]
                strategy_str = ", ".join([f"{actions[i]}: {prob:.3f}" for i, prob in enumerate(avg_strategy)])
                print(f"{i+1}. {infoset} - {strategy_str}")

    def get_action(self, infoset, available_actions=None):
        """
        Get an action based on the current strategy for the given information set
        
        Args:
            infoset: Information set representation
            available_actions: List of available actions (if None, assumes all actions are available)
            
        Returns:
            The chosen action index
        """
        strategy = self.get_average_strategy(infoset)
        
        # If there are restrictions on available actions
        if available_actions is not None:
            # Mask unavailable actions
            mask = np.zeros(self.num_actions)
            for action in available_actions:
                mask[action] = 1
            
            # Apply mask and renormalize
            masked_strategy = strategy * mask
            strategy_sum = np.sum(masked_strategy)
            
            # If no valid actions after masking, use uniform over available
            if strategy_sum <= 0:
                masked_strategy = np.zeros(self.num_actions)
                for action in available_actions:
                    masked_strategy[action] = 1
                strategy = masked_strategy / len(available_actions)
            else:
                strategy = masked_strategy / strategy_sum
        
        # Sample from the strategy
        return np.random.choice(self.num_actions, p=strategy)


def load_poker_data(json_data):
    """Load poker hand data from JSON string"""
    try:
        poker_hands = json.loads(json_data)
        print(f"Loaded {len(poker_hands)} poker hands")
        return poker_hands
    except json.JSONDecodeError:
        print("Error parsing JSON data")
        return []


def test_blueprint_strategy():
    """Test the blueprint strategy with some sample scenarios"""
    blueprint = BlueprintStrategy(num_actions=3)
    
    # Try to load an existing strategy
    if not blueprint.load():
        print("No saved strategy found. Please train the model first.")
        return
    
    # Define some test scenarios
    test_scenarios = [
        {
            "description": "Flop with a medium pot (50 chips)",
            "infoset": "flop_3_50",
            "available_actions": [0, 1, 2]  # All actions available
        },
        {
            "description": "Turn with a big pot (100 chips)",
            "infoset": "turn_4_100",
            "available_actions": [0, 1, 2]
        },
        {
            "description": "River with a huge pot (200 chips)",
            "infoset": "river_5_200",
            "available_actions": [1, 2]  # Can only call or raise
        }
    ]
    
    # Test the strategy on these scenarios
    print("\n===== TESTING BLUEPRINT STRATEGY =====")
    action_names = ["Fold/Check", "Call", "Raise"]
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['description']}")
        print(f"Information set: {scenario['infoset']}")
        
        # Get strategy for this infoset
        strategy = blueprint.get_average_strategy(scenario['infoset'])
        print("Strategy probabilities:")
        for i, prob in enumerate(strategy):
            print(f"  {action_names[i]}: {prob:.3f}")
        
        # Sample actions multiple times to see distribution
        action_counts = [0, 0, 0]
        num_samples = 1000
        
        for _ in range(num_samples):
            action = blueprint.get_action(scenario['infoset'], scenario['available_actions'])
            action_counts[action] += 1
        
        print(f"Action distribution over {num_samples} samples:")
        for i, count in enumerate(action_counts):
            if i in scenario['available_actions']:
                print(f"  {action_names[i]}: {count/num_samples:.3f}")
    
    # Interactive testing
    print("\n===== INTERACTIVE TESTING =====")
    print("Enter information sets to see the strategy (or 'quit' to exit)")
    print("Format: [round]_[num_cards]_[pot_size]")
    print("Example: flop_3_50")
    
    while True:
        user_input = input("\nEnter information set: ")
        if user_input.lower() == 'quit':
            break
        
        strategy = blueprint.get_average_strategy(user_input)
        print("Strategy:")
        for i, prob in enumerate(strategy):
            print(f"  {action_names[i]}: {prob:.3f}")

def main():
    # Load poker data from the data string
    with open('holdem_hands.json', 'r') as f:
        json_data = f.read()
    
    poker_hands = load_poker_data(json_data)
    
    if not poker_hands:
        print("No valid poker hand data found.")
        return
    
    # Initialize and train blueprint strategy
    blueprint = BlueprintStrategy(num_actions=3)
    
    # Try to load existing strategy first
    if not blueprint.load():
        # Train new strategy if no existing one found
        blueprint.train_from_data(poker_hands, num_iterations=5000)
        blueprint.save()
    
    # Print the learned strategy
    blueprint.print_strategy(top_n=15)
    
    # Run tests - ADD THIS NEW PART
    should_test = input("Would you like to test the strategy? (y/n): ")
    if should_test.lower() == 'y':
        test_blueprint_strategy()


if __name__ == "__main__":
    main()