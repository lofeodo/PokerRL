from depthLimitedCFR import DepthLimitedCFR
from depthLimitedSolver import DepthLimitedSolver
from pokerGameState import PokerGameState
import time

class PokerAgent:
    """
    A complete poker agent that uses blueprint strategy, depth-limited solving,
    and opponent modeling to make decisions.
    """
    
    def __init__(self, blueprint_path=None, stack_size=1000, small_blind=5, big_blind=10, enable_learning=True):
        """
        Initialize the poker agent.
        
        Args:
            blueprint_path: Path to a saved blueprint strategy
            stack_size: Starting stack size for the game
            small_blind: Small blind amount
            big_blind: Big blind amount
            enable_learning: Whether to enable continuous learning
        """
        # Initialize the blueprint strategy
        self.blueprint_cfr = DepthLimitedCFR(max_depth=2)
        self.blueprint_path = blueprint_path
        self.enable_learning = enable_learning
        
        # Try to load a pre-computed blueprint
        if blueprint_path and self.blueprint_cfr.load(blueprint_path):
            print(f"Loaded blueprint strategy from {blueprint_path}")
        else:
            print("Training new blueprint strategy...")
            self.blueprint_cfr.train(num_iterations=1000)
            
            if blueprint_path:
                self.blueprint_cfr.save(blueprint_path)
        
        # Initialize the depth-limited solver
        self.solver = DepthLimitedSolver(self.blueprint_cfr, max_depth=3)
        
        # Game parameters
        self.stack_size = stack_size
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Current game state
        self.current_state = None
        
        # Learning tracking
        self.hands_played = 0
        self.new_experiences = {}
        
    def start_new_hand(self, position=0):
        """
        Start a new hand with the agent in the given position.
        
        Args:
            position: 0 for small blind, 1 for big blind
            
        Returns:
            The initial game state
        """
        # Initialize a new game state
        self.current_state = PokerGameState(
            stack_size=self.stack_size,
            small_blind=self.small_blind,
            big_blind=self.big_blind
        )
        
        # Deal hole cards
        self.current_state.deal_hole_cards()
        
        # Set player positions
        self.position = position
        self.current_state.current_player = 0  # Small blind acts first
        
        return self.current_state
    
    def act(self):
        """
        Make a decision and take an action in the current state.
        
        Returns:
            The action taken and the new game state
        """
        if self.current_state is None:
            raise ValueError("No active hand. Call start_new_hand() first.")
                
        # If it's not our turn, return None
        if self.current_state.current_player != self.position:
            return None, None
                
        print("Starting to get action from solver...")
        # Use the solver to get an action
        action, raise_amount = self.solver.get_action(self.current_state, self.position)
        print(f"Solver returned action: {action}, raise_amount: {raise_amount}")
            
        # Apply the action to the current state
        new_state = self.current_state.act(action, raise_amount)
        self.current_state = new_state
            
        return action, raise_amount

    def observe_opponent_action(self, action, raise_amount=None):
        """
        Update the current state based on the opponent's action.
        
        Args:
            action: The action taken by the opponent
            raise_amount: The raise amount (if applicable)
            
        Returns:
            The new game state
        """
        if self.current_state is None:
            raise ValueError("No active hand. Call start_new_hand() first.")
            
        # If it's not the opponent's turn, return None
        if self.current_state.current_player == self.position:
            return None
            
        # Update the opponent model
        self.solver.observe_opponent_action(self.current_state, action)
        
        # Apply the action to the current state
        new_state = self.current_state.act(action, raise_amount)
        self.current_state = new_state
        
        return new_state
    
    def deal_next_round(self):
        """
        Deal community cards for the next round.
        
        Returns:
            The new community cards
        """
        if self.current_state is None:
            raise ValueError("No active hand. Call start_new_hand() first.")
            
        # Deal community cards based on the current round
        self.current_state.deal_community_cards()
        
        return self.current_state.board
        
    def update_blueprint_from_solver(self):
        """
        Update the blueprint strategy with what was learned during gameplay.
        """
        if not self.enable_learning:
            return False
            
        # Get subgame solutions from the solver
        for subgame_key, subgame_strategy in self.solver.previously_played_subgames.items():
            # Update the blueprint with new strategies
            for info_set, strategy in subgame_strategy.items():
                # Create node if it doesn't exist
                if info_set not in self.blueprint_cfr.nodes:
                    self.blueprint_cfr.nodes[info_set] = self.blueprint_cfr.get_node(info_set)
                
                # Update strategy with higher weight for real gameplay experience
                experience_weight = 10  # Give more weight to real play vs simulation
                node = self.blueprint_cfr.nodes[info_set]
                node.strategy_sum += experience_weight * strategy
                
                # Also track in new experiences for analysis
                self.new_experiences[info_set] = strategy
        
        # Increment hands played counter
        self.hands_played += 1
        
        # Save after every hand (only to the main file, no backups)
        if self.enable_learning and self.blueprint_path:
            print(f"Saving updated blueprint after hand {self.hands_played}...")
            self.blueprint_cfr.save(self.blueprint_path)
            
        return True