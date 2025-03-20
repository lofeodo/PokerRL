import os
import time
import pickle
import numpy as np
from CFRNode import CFRNode
from handEvaluator import HandEvaluator
from pokerGameState import PokerGameState

class DepthLimitedCFR:
    """
    Implementation of depth-limited Counterfactual Regret Minimization for poker.
    
    This is a modified version of CFR that incorporates depth-limited solving
    similar to what's used in Libratus and Pluribus.
    """
    
    def __init__(self, max_depth=2, num_actions=3):
        self.nodes = {}  # Maps info_set -> CFRNode
        self.max_depth = max_depth
        self.num_actions = num_actions
        self.blueprint = None  # Blueprint strategy for leaf nodes
        self.use_linear_cfr = True  # Use Linear CFR which applies iteration weights
        
    def get_node(self, info_set):
        """Get or create a CFR node for the given information set."""
        if info_set not in self.nodes:
            self.nodes[info_set] = CFRNode(info_set, self.num_actions)
        return self.nodes[info_set]
    
    def cfr(self, state, reach_probs, depth=0, iteration=0):
        """
        Run one iteration of counterfactual regret minimization.
        
        Args:
            state: Current game state
            reach_probs: Reach probabilities for each player
            depth: Current depth in the game tree
            iteration: Current iteration number (used for Linear CFR)
            
        Returns:
            Expected values for each player
        """
        if depth > 10:  # Example threshold
            print(f"Warning: High recursion depth {depth}")
        # If the state is terminal, return the payoffs
        if state.is_terminal():
            return state.get_payoff()
        
        # If we've reached the maximum depth, estimate value using the blueprint
        if depth >= self.max_depth:
            return self.estimate_value(state)
        
        # Current player
        player = state.current_player
        
        # Get the information set for the current player
        info_set = state.get_info_set(player)
        
        # Get or create the CFR node for this information set
        node = self.get_node(info_set)
        
        # Get the current strategy for this information set
        strategy = node.get_strategy(reach_probs[player])
        
        # Get legal actions for the current state
        legal_actions = state.get_legal_actions()
        
        # Initialize expected values for each action and for the entire information set
        action_values = np.zeros(self.num_actions)
        node_value = np.zeros(2)
        
        # For each legal action, compute its expected value
        for action in legal_actions:
            # Create new state after taking this action
            new_state = state.act(action)
            
            # Skip if game ended due to fold
            if new_state is None:
                if action == state.FOLD:
                    # Current player folded
                    action_values[action] = -state.pot if player == 0 else state.pot
                    # Use numpy array operations to update node value
                    fold_value = np.array([-state.pot, state.pot])
                    if player == 1:  # Reverse for player 1
                        fold_value = np.array([state.pot, -state.pot])
                    node_value += strategy[action] * fold_value
                    continue
            
            # Update reach probabilities for this action
            new_reach_probs = reach_probs.copy()
            new_reach_probs[player] *= strategy[action]
            
            # Recursively compute expected values for this action
            action_value = self.cfr(new_state, new_reach_probs, depth + 1, iteration)
            
            # Ensure action_value is a numpy array
            action_value = np.array(action_value)
            
            # Store the expected value for this action
            action_values[action] = action_value[player]
            
            # Accumulate the expected value for the entire information set
            node_value += strategy[action] * action_value
        
        # Compute counterfactual reach probability for the opponent
        counterfactual_reach_prob = reach_probs[1 - player]
        
        # For each legal action, compute its regret and update the node's regrets
        for action in legal_actions:
            regret = counterfactual_reach_prob * (action_values[action] - node_value[player])
            
            # Apply iteration weighting for Linear CFR
            if self.use_linear_cfr and iteration > 0:
                # Linear CFR weights regrets by the iteration number
                regret *= iteration
            
            node.update_regrets(action, regret)
        
        return node_value

    def estimate_value(self, state):
        """
        Estimate the value of a state when depth limit is reached.
        Uses the blueprint strategy or a simple equity calculation.
        """
        # If we have a blueprint strategy, use it
        if self.blueprint:
            # Get information sets for both players
            player_info_set = state.get_info_set(0)
            opponent_info_set = state.get_info_set(1)
            
            # Get strategies from blueprint
            player_strategy = self.blueprint.get_strategy(player_info_set)
            opponent_strategy = self.blueprint.get_strategy(opponent_info_set)
            
            # Estimate expected value based on blueprint strategies
            # This is a simplified implementation
            player_equity = HandEvaluator.calculate_equity(
                state.player_hole_cards[0], 
                state.board, 
                num_simulations=100
            )
            
            # Value is based on pot size and equity
            player_ev = (player_equity * state.pot) - ((1 - player_equity) * state.pot)
            opponent_ev = -player_ev
            
            return [player_ev, opponent_ev]
        
        # Otherwise, use a simple equity calculation
        player_equity = HandEvaluator.calculate_equity(
            state.player_hole_cards[0], 
            state.board, 
            num_simulations=100
        )
        
        # Value is based on pot size and equity
        player_ev = (player_equity * state.pot) - ((1 - player_equity) * state.pot)
        opponent_ev = -player_ev
        
        return [player_ev, opponent_ev]
    
    def train(self, num_iterations=1000):
        """Train the CFR algorithm for a given number of iterations."""
        print(f"Starting CFR training for {num_iterations} iterations...")
        start_time = time.time()
        
        for i in range(num_iterations):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {i} (Elapsed: {elapsed:.2f}s)")
                
            # Initialize a new game state
            state = PokerGameState()
            
            # Deal cards
            state.deal_hole_cards()
            
            # Run CFR from this state
            reach_probs = np.ones(2)  # Initial reach probabilities
            self.cfr(state, reach_probs, iteration=i+1)
            
        print(f"CFR training completed after {num_iterations} iterations.")
    
    def get_average_strategy(self):
        """Return the average strategy for all information sets."""
        avg_strategy = {}
        for info_set, node in self.nodes.items():
            avg_strategy[info_set] = node.get_average_strategy()
        return avg_strategy
    
    def save(self, filename="cfr_strategy.pkl"):
        """Save the current strategy to a file."""
        avg_strategy = self.get_average_strategy()
        with open(filename, 'wb') as f:
            pickle.dump(avg_strategy, f)
        print(f"Strategy saved to {filename}")
    
    def get_strategy(self, info_set):
        """Get strategy for an information set (for blueprint compatibility)."""
        node = self.get_node(info_set)
        return node.get_average_strategy()
    
    def load(self, filename="cfr_strategy.pkl"):
        """Load a strategy from a file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                avg_strategy = pickle.load(f)
                
            # Create nodes for each info set in the loaded strategy
            for info_set, strategy in avg_strategy.items():
                node = self.get_node(info_set)
                node.strategy_sum = strategy * 1000  # Scale up to give it weight
                
            print(f"Strategy loaded from {filename}")
            return True
        return False