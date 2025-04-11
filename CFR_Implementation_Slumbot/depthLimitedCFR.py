import os
import time
import pickle
import torch
import numpy as np
from CFRNode import CFRNode
from handEvaluator import HandEvaluator
from pokerGameState import PokerGameState

class DepthLimitedCFR:
    """
    Implementation of depth-limited Counterfactual Regret Minimization for poker.
    
    This is a GPU-accelerated version of CFR that incorporates depth-limited solving
    similar to what's used in Libratus and Pluribus.
    """
    
    def __init__(self, max_depth=2, num_actions=3):
        self.nodes = {}  # Maps info_set -> CFRNode
        self.max_depth = max_depth
        self.num_actions = num_actions
        self.blueprint = None  # Blueprint strategy for leaf nodes
        self.use_linear_cfr = True  # Use Linear CFR which applies iteration weights
        
        # Initialize device to use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DepthLimitedCFR using device: {self.device}")
        
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
            return torch.tensor(state.get_payoff(), device=self.device)
        
        # If we've reached the maximum depth, estimate value using the blueprint
        if depth >= self.max_depth:
            return torch.tensor(self.estimate_value(state), device=self.device)
        
        # Current player
        player = state.current_player
        
        # Get the information set for the current player
        info_set = state.get_info_set(player)
        
        # Get or create the CFR node for this information set
        node = self.get_node(info_set)
        
        # Convert reach_probs to torch tensor if needed
        if not isinstance(reach_probs, torch.Tensor):
            reach_probs = torch.tensor(reach_probs, device=self.device)
        
        # Get the current strategy for this information set
        strategy = node.get_strategy(reach_probs[player])
        
        # Get legal actions for the current state
        legal_actions = state.get_legal_actions()
        
        # Initialize expected values for each action and for the entire information set
        action_values = torch.zeros(self.num_actions, device=self.device)
        node_value = torch.zeros(2, device=self.device)
        
        # For each legal action, compute its expected value
        for action in legal_actions:
            # Create new state after taking this action
            new_state = state.act(action)
            
            # Skip if game ended due to fold
            if new_state is None:
                if action == state.FOLD:
                    # Current player folded
                    action_values[action] = torch.tensor(-state.pot if player == 0 else state.pot, device=self.device)
                    
                    # Use tensor operations to update node value
                    if player == 0:
                        fold_value = torch.tensor([-state.pot, state.pot], device=self.device)
                    else:  # player == 1
                        fold_value = torch.tensor([state.pot, -state.pot], device=self.device)
                    
                    node_value += strategy[action] * fold_value
                    continue
            
            # Update reach probabilities for this action
            if isinstance(reach_probs, torch.Tensor):
                new_reach_probs = reach_probs.clone()
            else:
                new_reach_probs = torch.tensor(reach_probs, device=self.device).clone()
                
            new_reach_probs[player] *= strategy[action]
            
            # Recursively compute expected values for this action
            action_value = self.cfr(new_state, new_reach_probs, depth + 1, iteration)
            
            # Ensure action_value is a tensor on the same device
            if not isinstance(action_value, torch.Tensor):
                action_value = torch.tensor(action_value, device=self.device)
            elif action_value.device != self.device:
                action_value = action_value.to(self.device)
            
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
    
    def train(self, num_iterations=1000, batch_size=16):
        """Train the CFR algorithm for a given number of iterations."""
        print(f"Starting GPU-accelerated CFR training for {num_iterations} iterations...")
        start_time = time.time()
        
        # Process iterations in batches for better GPU utilization
        for i in range(0, num_iterations, batch_size):
            batch_end = min(i + batch_size, num_iterations)
            current_batch_size = batch_end - i
            
            if i % 100 == 0:
                elapsed = time.time() - start_time
                iterations_per_second = max(1, i) / max(0.1, elapsed)
                remaining = (num_iterations - i) / max(1.0, iterations_per_second)
                print(f"Iteration {i}/{num_iterations} (Elapsed: {elapsed:.2f}s, ETA: {remaining:.2f}s)")
                
                # Report GPU memory usage if available
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    cached = torch.cuda.memory_reserved() / 1024**2
                    print(f"GPU memory: {allocated:.1f}MB allocated, {cached:.1f}MB cached")
            
            # Process a batch of game states
            for j in range(current_batch_size):
                # Initialize a new game state
                state = PokerGameState()
                
                # Deal cards
                state.deal_hole_cards()
                
                # Run CFR from this state
                reach_probs = torch.ones(2, device=self.device)
                self.cfr(state, reach_probs, iteration=i+j+1)
            
            # Periodically clear CUDA cache to prevent fragmentation
            if torch.cuda.is_available() and i % 1000 == 0 and i > 0:
                torch.cuda.empty_cache()
            
        training_time = time.time() - start_time
        iterations_per_second = num_iterations / max(0.1, training_time)
        print(f"CFR training completed after {num_iterations} iterations.")
        print(f"Total time: {training_time:.2f}s ({iterations_per_second:.1f} iterations/second)")
    
    def get_average_strategy(self):
        """Return the average strategy for all information sets."""
        avg_strategy = {}
        for info_set, node in self.nodes.items():
            # Get the strategy and convert to numpy for compatibility
            # Make sure to move tensor to CPU before numpy conversion
            avg_strategy[info_set] = node.get_average_strategy().cpu().numpy()
        return avg_strategy
    
    def save(self, filename="cfr_strategy.pkl"):
        """Save the current strategy to a file."""
        avg_strategy = self.get_average_strategy()
        with open(filename, 'wb') as f:
            pickle.dump(avg_strategy, f)
        print(f"Strategy saved to {filename} ({len(avg_strategy)} information sets)")
    
    def get_strategy(self, info_set):
        """Get strategy for an information set (for blueprint compatibility)."""
        node = self.get_node(info_set)
        # Return as NumPy array for compatibility
        return node.get_average_strategy().cpu().numpy()
    
    def load(self, filename="cfr_strategy.pkl"):
        """Load a strategy from a file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                avg_strategy = pickle.load(f)
                
            # Create nodes for each info set in the loaded strategy
            info_set_count = 0
            for info_set, strategy in avg_strategy.items():
                node = self.get_node(info_set)
                # Convert NumPy array to tensor
                strategy_tensor = torch.tensor(strategy, device=self.device)
                node.strategy_sum = strategy_tensor * 1000  # Scale up to give it weight
                info_set_count += 1
                
                # Print progress for large strategy files
                if info_set_count % 10000 == 0:
                    print(f"Loaded {info_set_count}/{len(avg_strategy)} information sets...")
                
            print(f"Strategy loaded from {filename} ({info_set_count} information sets)")
            return True
        return False