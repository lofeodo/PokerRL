import numpy as np
import random

from depthLimitedCFR import DepthLimitedCFR
from opponentModel import OpponentModel
from pokerGameState import PokerGameState


class DepthLimitedSolver:
    """
    Implements depth-limited solving for poker using a blueprint strategy.
    Similar to the approach used in Libratus and Pluribus.
    """
    
    def __init__(self, blueprint_strategy, max_depth=10, num_actions=3):
        self.blueprint = blueprint_strategy
        self.max_depth = max_depth
        self.num_actions = num_actions
        self.cfr_solver = DepthLimitedCFR(max_depth, num_actions)
        self.cfr_solver.blueprint = blueprint_strategy
        self.opponent_model = OpponentModel()
        self.previously_played_subgames = {}
        
    def solve_subgame(self, root_state, num_iterations=5):
        """
        Solve a depth-limited subgame rooted at the given state.
        Uses safe, nested subgame solving as in Libratus.
        
        Args:
            root_state: The root state of the subgame
            num_iterations: Number of CFR iterations to run
            
        Returns:
            A dictionary mapping information sets to strategies
        """
        # Check if this subgame has been solved before
        subgame_key = self._get_subgame_key(root_state)
        if subgame_key in self.previously_played_subgames:
            print(f"Using cached solution for subgame {subgame_key}")
            return self.previously_played_subgames[subgame_key]
        
        print(f"Solving subgame with {num_iterations} iterations...")
        
        # Use the existing CFR solver 
        solver = self.cfr_solver  # Changed from self.solver to self.cfr_solver
        
        # Override max_depth to limit search depth
        original_max_depth = solver.max_depth
        solver.max_depth = 1  # Temporarily reduce depth
        
        # Run CFR for the specified number of iterations
        for i in range(num_iterations):
            print(f"Subgame solving iteration {i}")
            
            # Run one iteration of CFR
            reach_probs = np.ones(2)
            solver.cfr(root_state, reach_probs)
            
        # Get the computed strategy
        print("Computing average strategy...")
        subgame_strategy = solver.get_average_strategy()
        
        # Restore original depth
        solver.max_depth = original_max_depth
        
        # Cache the solution
        self.previously_played_subgames[subgame_key] = subgame_strategy
        
        return subgame_strategy

    def _get_subgame_key(self, state):
        """Generate a unique key for a subgame to enable caching."""
        # We'll use the round, board cards, and pot size as the key
        board_str = ''.join(str(card) for card in state.board)
        return f"{state.current_round}|{board_str}|{state.pot}"

    def get_action(self, state, player_idx):
        """
        Get the best action for a player in the given state using depth-limited solving.
        
        Args:
            state: Current game state
            player_idx: Index of the player (0 or 1)
            
        Returns:
            The selected action and optional raise amount
        """
        # If it's not the player's turn, return None
        if state.current_player != player_idx:
            return None, None
        
        print("Starting to solve subgame...")
        # Solve the depth-limited subgame
        subgame_strategy = self.solve_subgame(state, num_iterations=5)  # Reduced iterations
        print("Subgame solved!")
        
        # Get the information set for the current state
        info_set = state.get_info_set(player_idx)
        
        # Get legal actions
        legal_actions = state.get_legal_actions()
        
        # If info set exists in the subgame solution, use that strategy
        if info_set in subgame_strategy:
            strategy = subgame_strategy[info_set]
        else:
            # Fall back to blueprint strategy
            strategy = self.blueprint.get_strategy(info_set)
        
        # Create a masked strategy with only legal actions
        masked_strategy = np.zeros(self.num_actions)
        for action in legal_actions:
            masked_strategy[action] = strategy[action]
            
        # Normalize the strategy
        strategy_sum = np.sum(masked_strategy)
        if strategy_sum > 0:
            masked_strategy = masked_strategy / strategy_sum
        else:
            # If all legal actions have zero probability, use uniform strategy
            masked_strategy = np.zeros(self.num_actions)
            for action in legal_actions:
                masked_strategy[action] = 1.0 / len(legal_actions)
        
        # Sample an action from the strategy
        action = np.random.choice(self.num_actions, p=masked_strategy)
        
        # If the action is a raise, determine the raise amount
        raise_amount = None
        if action == PokerGameState.BET_RAISE:
            # Simplify raise sizing for speed
            raise_amount = state.min_raise
        
        return action, raise_amount

    def observe_opponent_action(self, state, action):
        """
        Update the opponent model based on observed actions.
        
        Args:
            state: Current game state before the opponent action
            action: The action taken by the opponent
        """
        # Get opponent's information set
        opponent_idx = state.current_player
        info_set = state.get_info_set(opponent_idx)
        
        # Update opponent model
        self.opponent_model.observe_action(info_set, action)
