import json
import numpy as np
from collections import defaultdict
import random
import pickle
import os
from itertools import combinations
import time

class EnhancedBlueprintStrategy:
    """
    Enhanced blueprint strategy implementation using Counterfactual Regret Minimization (CFR)
    for poker games, incorporating full card information.
    
    References:
    - Zinkevich et al. (2007) "Regret Minimization in Games with Incomplete Information"
    - Bowling et al. (2015) "Heads-up limit hold'em poker is solved"
    - Brown & Sandholm (2017) "Safe and Nested Subgame Solving for Imperfect-Information Games"
    """
    
    def __init__(self, num_actions=3):
        # Information sets: map of infoset -> (regret_sum, strategy_sum, current_strategy)
        self.infosets = defaultdict(lambda: [np.zeros(num_actions), np.zeros(num_actions), np.zeros(num_actions)])
        self.num_actions = num_actions
        self.iterations = 0
        
        # Card strength calculator
        self.card_evaluator = PokerHandEvaluator()
        
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
    
    def create_infoset_key(self, hole_cards, community_cards, round_name, pot_size, position):
        """
        Create a comprehensive information set key that includes:
        - Player's hole cards
        - Community cards
        - Betting round
        - Pot size (discretized into buckets)
        - Player position
        """
        # Sort hole cards for consistency
        sorted_hole = sorted(hole_cards)
        hole_str = "".join(sorted_hole)
        
        # Sort community cards for consistency
        comm_str = "".join(sorted(community_cards)) if community_cards else ""
        
        # Discretize pot size into buckets
        pot_bucket = self._discretize_pot(pot_size)
        
        # Combine all information into a single infoset key
        return f"{round_name}_{hole_str}_{comm_str}_{pot_bucket}_{position}"
    
    def _discretize_pot(self, pot_size):
        """
        Discretize pot sizes into reasonable buckets to reduce state space
        """
        if pot_size <= 20:
            return "tiny"
        elif pot_size <= 50:
            return "small"
        elif pot_size <= 100:
            return "medium"
        elif pot_size <= 200:
            return "large"
        else:
            return "huge"
            
    def calculate_hand_strength(self, hole_cards, community_cards, round_name):
        """
        Calculate the strength of the hand:
        - Pre-flop: Use preflop hand strength chart
        - Post-flop: Evaluate current hand strength and potential
        """
        if round_name == "preflop":
            return self.card_evaluator.evaluate_preflop(hole_cards)
        else:
            return self.card_evaluator.evaluate_hand(hole_cards, community_cards)
    
    def train_from_data(self, poker_hands, num_iterations=1000):
        """
        Train blueprint strategy from generated poker hand data using Monte Carlo CFR
        """
        self.iterations = num_iterations
        start_time = time.time()
        
        for iteration in range(num_iterations):
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"CFR Iteration {iteration} (Elapsed: {elapsed:.2f}s)")
            
            # Sample a random hand from data
            hand = random.choice(poker_hands)
            
            # Extract hand data
            hole_cards = hand.get("player_hole_cards", [])
            opponent_hole_cards = hand.get("opponent_hole_cards", [])
            board_cards = hand.get("board", [])
            
            # Skip hands with insufficient data
            if not hole_cards:
                continue
                
            # Simulate the game with CFR
            self._cfr_iteration(hand, hole_cards, board_cards)
                
        print(f"Training completed after {num_iterations} iterations.")
    
    def _cfr_iteration(self, hand, hole_cards, board_cards):
        """Perform one iteration of CFR on a sample hand"""
        # Track betting rounds 
        rounds = []
        positions = hand.get("positions", {})
        player_position = positions.get("player", "unknown")
        
        for round_name in ["preflop", "flop", "turn", "river"]:
            if round_name in hand and hand[round_name]["num_players"] >= 2:
                community_cards = []
                if round_name == "flop":
                    community_cards = board_cards[:3] if len(board_cards) >= 3 else []
                elif round_name == "turn":
                    community_cards = board_cards[:4] if len(board_cards) >= 4 else []
                elif round_name == "river":
                    community_cards = board_cards[:5] if len(board_cards) >= 5 else []
                
                rounds.append((round_name, hand[round_name]["pot_size"], community_cards))
        
        # Skip hands with insufficient data
        if len(rounds) < 2:
            return
            
        # Simulate decisions at each transition between rounds
        for i in range(len(rounds) - 1):
            current_round, current_pot, current_cards = rounds[i]
            next_round, next_pot, next_cards = rounds[i+1]
            
            # Calculate hand strength for current state
            hand_strength = self.calculate_hand_strength(hole_cards, current_cards, current_round)
            
            # Create a comprehensive information set
            infoset = self.create_infoset_key(
                hole_cards, 
                current_cards, 
                current_round, 
                current_pot,
                player_position
            )
            
            # Pot growth indicates betting action
            pot_growth = next_pot - current_pot
            
            # Determine the actual action that was taken
            if pot_growth == 0:
                action = 0  # check/fold
            elif pot_growth <= 20:
                action = 1  # call
            else:
                action = 2  # raise
                
            # Calculate counterfactual value based on hand strength and outcome
            # This is a simplified calculation for the example
            regret = self._calculate_regret(hand_strength, pot_growth, current_pot, action)
            
            # Simplified reach probability
            reach_prob = 1.0 / (self.iterations + 1)
            
            # Update regrets and strategy
            self.update(infoset, action, regret, reach_prob)
    
    def _calculate_regret(self, hand_strength, pot_growth, current_pot, action_taken):
        """
        Calculate a more sophisticated regret value based on:
        - Hand strength
        - Pot size
        - Action taken vs optimal action
        """
        # Rough heuristic for optimal action based on hand strength
        optimal_action = 0  # Default to fold/check for weak hands
        
        if hand_strength > 0.8:  # Very strong hand
            optimal_action = 2  # Should raise
        elif hand_strength > 0.5:  # Medium strength
            optimal_action = 1  # Should call
        
        # Calculate regret - how much we regret not taking the optimal action
        # High regret if our hand was strong but we didn't raise
        # or if our hand was weak but we raised
        if optimal_action != action_taken:
            if optimal_action == 2 and action_taken < 2:
                # Regret not raising with a strong hand
                regret = hand_strength * (current_pot / 100)
            elif optimal_action == 0 and action_taken > 0:
                # Regret calling/raising with a weak hand
                regret = (1 - hand_strength) * (pot_growth / 100)
            else:
                regret = 0.1 * (pot_growth / 100)  # Small regret for minor mistakes
        else:
            # Small positive reinforcement for taking the right action
            regret = 0.05 * (current_pot / 100)
            
        return regret
    
    def get_action(self, hole_cards, community_cards, round_name, pot_size, position, available_actions=None):
        """
        Get an action based on the current strategy for the given poker situation
        
        Args:
            hole_cards: Player's private cards
            community_cards: Shared community cards
            round_name: Current betting round (preflop, flop, turn, river)
            pot_size: Current pot size
            position: Player's position (early, middle, late, etc.)
            available_actions: List of available actions (if None, assumes all actions are available)
            
        Returns:
            The chosen action index
        """
        # Create the information set key
        infoset = self.create_infoset_key(hole_cards, community_cards, round_name, pot_size, position)
        
        # Get the strategy for this infoset
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
    
    def save(self, filename="enhanced_blueprint_strategy.pkl"):
        """Save the blueprint strategy to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.infosets), f)
        print(f"Strategy saved to {filename}")
    
    def load(self, filename="enhanced_blueprint_strategy.pkl"):
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


class PokerHandEvaluator:
    """
    A class to evaluate poker hand strength
    """
    def __init__(self):
        # Pre-compute lookup tables if needed
        pass
        
    def evaluate_preflop(self, hole_cards):
        """
        Evaluate preflop hand strength based on common poker heuristics
        Returns a value between 0 (worst) and 1 (best)
        """
        # Extract ranks and suits
        ranks = [card[0] for card in hole_cards]
        suits = [card[1] for card in hole_cards]
        
        # Convert face cards to numeric values
        rank_values = []
        for rank in ranks:
            if rank == 'A':
                rank_values.append(14)
            elif rank == 'K':
                rank_values.append(13)
            elif rank == 'Q':
                rank_values.append(12)
            elif rank == 'J':
                rank_values.append(11)
            elif rank == 'T':
                rank_values.append(10)
            else:
                rank_values.append(int(rank))
        
        # Sort ranks in descending order
        rank_values.sort(reverse=True)
        
        # Check for pairs
        is_pair = rank_values[0] == rank_values[1]
        
        # Check for suited cards
        is_suited = suits[0] == suits[1]
        
        # Calculate base strength
        if is_pair:
            # Pairs strength: higher pairs are stronger
            base_strength = 0.5 + 0.5 * (rank_values[0] / 14)
        else:
            # Non-pairs: consider both card ranks and connectivity
            high_card_strength = rank_values[0] / 14
            low_card_strength = rank_values[1] / 14
            gap = rank_values[0] - rank_values[1] - 1  # -1 for consecutive
            
            # Connectivity bonus (closer cards are better)
            connectivity = max(0, 1 - (gap / 4))
            
            # Base formula for non-paired hands
            base_strength = 0.2 * high_card_strength + 0.1 * low_card_strength + 0.1 * connectivity
            
            # Suited bonus
            if is_suited:
                base_strength += 0.1
        
        # Special case for premium hands
        if (rank_values[0] == 14 and rank_values[1] == 13) or (rank_values[0] == 14 and rank_values[1] == 14):
            base_strength = max(base_strength, 0.9)  # AA, AK
            
        return min(1.0, base_strength)  # Cap at 1.0
    
    def evaluate_hand(self, hole_cards, community_cards):
        """
        Evaluate the strength of a hand given the community cards
        Returns a value between 0 (worst) and 1 (best)
        """
        # If no community cards, use preflop evaluation
        if not community_cards:
            return self.evaluate_preflop(hole_cards)
            
        # Combine hole cards and community cards
        all_cards = hole_cards + community_cards
        
        # Simplified hand strength calculation
        # In a real implementation, you would evaluate the actual poker hand
        # and calculate its strength against possible opponent hands
        
        # Calculate made hand strength (current best 5-card hand)
        made_hand_strength = self._calculate_made_hand_strength(all_cards)
        
        # Calculate potential hand strength (draws to better hands)
        potential_strength = self._calculate_potential_strength(hole_cards, community_cards)
        
        # Combine made hand and potential (weighted by betting round)
        if len(community_cards) == 3:  # Flop
            return 0.7 * made_hand_strength + 0.3 * potential_strength
        elif len(community_cards) == 4:  # Turn
            return 0.85 * made_hand_strength + 0.15 * potential_strength
        else:  # River - no more potential
            return made_hand_strength
    
    def _calculate_made_hand_strength(self, cards):
        """
        Calculate the strength of the current made hand
        Simplified version - in a real implementation, you would use more sophisticated hand evaluation
        """
        # Extract ranks and suits
        ranks = [card[0] for card in cards]
        suits = [card[1] for card in cards]
        
        # Count rank frequencies
        rank_counts = {}
        for rank in ranks:
            if rank in rank_counts:
                rank_counts[rank] += 1
            else:
                rank_counts[rank] = 1
        
        # Count suit frequencies
        suit_counts = {}
        for suit in suits:
            if suit in suit_counts:
                suit_counts[suit] += 1
            else:
                suit_counts[suit] = 1
        
        # Check for common hand types (simplified)
        has_pair = any(count >= 2 for count in rank_counts.values())
        has_two_pair = len([count for count in rank_counts.values() if count >= 2]) >= 2
        has_three_kind = any(count >= 3 for count in rank_counts.values())
        has_straight = self._check_straight(ranks)
        has_flush = any(count >= 5 for count in suit_counts.values())
        has_full_house = has_three_kind and has_pair
        has_four_kind = any(count >= 4 for count in rank_counts.values())
        has_straight_flush = has_straight and has_flush  # Simplified - not accurate
        
        # Assign hand strength based on hand type (simplified)
        if has_straight_flush:
            return 1.0
        elif has_four_kind:
            return 0.9
        elif has_full_house:
            return 0.8
        elif has_flush:
            return 0.7
        elif has_straight:
            return 0.6
        elif has_three_kind:
            return 0.5
        elif has_two_pair:
            return 0.4
        elif has_pair:
            return 0.3
        else:
            # High card - use highest card to determine strength
            highest_rank = max(self._rank_to_value(r) for r in ranks)
            return 0.2 * (highest_rank / 14)
    
    def _rank_to_value(self, rank):
        """Convert rank to numeric value"""
        if rank == 'A':
            return 14
        elif rank == 'K':
            return 13
        elif rank == 'Q':
            return 12
        elif rank == 'J':
            return 11
        elif rank == 'T':
            return 10
        else:
            return int(rank)
    
    def _check_straight(self, ranks):
        """
        Check if the cards contain a straight
        Simplified version - not accurate for all cases
        """
        # Convert ranks to values
        values = sorted([self._rank_to_value(r) for r in ranks], reverse=True)
        
        # Check for 5 consecutive cards
        consecutive_count = 1
        for i in range(1, len(values)):
            if values[i] == values[i-1] - 1:
                consecutive_count += 1
                if consecutive_count >= 5:
                    return True
            elif values[i] != values[i-1]:  # Skip duplicates
                consecutive_count = 1
        
        return False
    
    def _calculate_potential_strength(self, hole_cards, community_cards):
        """
        Calculate the potential strength of the hand (drawing potential)
        Simplified version - in a real implementation, use more sophisticated methods
        """
        # Check for flush draws
        suits = [card[1] for card in hole_cards + community_cards]
        suit_counts = {}
        for suit in suits:
            if suit in suit_counts:
                suit_counts[suit] += 1
            else:
                suit_counts[suit] = 1
                
        has_flush_draw = any(count == 4 for count in suit_counts.values())
        
        # Check for straight draws (simplified)
        ranks = [self._rank_to_value(card[0]) for card in hole_cards + community_cards]
        ranks_set = set(ranks)
        
        # Check for open-ended straight draw
        has_open_straight_draw = False
        for i in range(min(ranks), max(ranks) - 3):
            if all(r in ranks_set for r in range(i, i+4)):
                has_open_straight_draw = True
                break
        
        # Assign potential strength based on draws
        if has_flush_draw and has_open_straight_draw:
            return 0.8  # Strong draw potential
        elif has_flush_draw:
            return 0.6  # Good draw potential
        elif has_open_straight_draw:
            return 0.4  # Moderate draw potential
        else:
            return 0.1  # Low draw potential


def generate_training_data(num_hands=1000):
    """
    Simulate generating training data from self-play or other sources
    In a real implementation, this would come from actual gameplay
    """
    training_data = []
    
    # Card deck
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['h', 'd', 'c', 's']
    deck = [r+s for r in ranks for s in suits]
    
    for _ in range(num_hands):
        # Shuffle deck
        random.shuffle(deck)
        
        # Deal cards
        player_hole = [deck[0], deck[1]]
        opponent_hole = [deck[2], deck[3]]
        community = [deck[4], deck[5], deck[6], deck[7], deck[8]]
        
        # Determine positions
        positions = {
            "player": random.choice(["SB", "BB"]),
            "opponent": "BB" if random.choice(["SB", "BB"]) == "SB" else "SB"
        }
        
        # Simulate a hand with random actions
        hand = {
            "_id": f"simulated_{random.randint(10000, 99999)}",
            "game": "holdem",
            "timestamp": str(int(time.time())),
            "dealer": 1,
            "hand_num": random.randint(1, 10000),
            "num_players": 2,
            "positions": positions,
            "player_hole_cards": player_hole,
            "opponent_hole_cards": opponent_hole,
            "board": community,
        }
        
        # Simulate betting rounds with increasing pot sizes
        preflop_pot = random.choice([20, 30, 40, 60, 80])
        hand["preflop"] = {
            "num_players": 2,
            "pot_size": preflop_pot
        }
        
        # Some hands end before flop
        if random.random() < 0.2:
            hand["showdown"] = {
                "num_players": 1,
                "pot_size": preflop_pot + random.randint(0, 20)
            }
            training_data.append(hand)
            continue
            
        flop_pot = preflop_pot + random.choice([0, 20, 40, 80])
        hand["flop"] = {
            "num_players": 2,
            "pot_size": flop_pot
        }
        
        # Some hands end at flop
        if random.random() < 0.3:
            hand["showdown"] = {
                "num_players": 1,
                "pot_size": flop_pot + random.randint(0, 40)
            }
            training_data.append(hand)
            continue
            
        turn_pot = flop_pot + random.choice([0, 20, 40, 80, 120])
        hand["turn"] = {
            "num_players": 2,
            "pot_size": turn_pot
        }
        
        # Some hands end at turn
        if random.random() < 0.4:
            hand["showdown"] = {
                "num_players": 1,
                "pot_size": turn_pot + random.randint(0, 60)
            }
            training_data.append(hand)
            continue
            
        river_pot = turn_pot + random.choice([0, 40, 80, 120, 200])
        hand["river"] = {
            "num_players": 2,
            "pot_size": river_pot
        }
        
        hand["showdown"] = {
            "num_players": 2,
            "pot_size": river_pot + random.choice([0, 40, 80, 120, 200])
        }
        
        training_data.append(hand)
    
    return training_data


def test_blueprint_strategy():
    """Test the blueprint strategy with some sample scenarios"""
    blueprint = EnhancedBlueprintStrategy(num_actions=3)
    
    # Try to load existing strategy
    if not blueprint.load():
        print("No saved strategy found. Please train the model first.")
        return
    
    # Define some test scenarios
    test_scenarios = [
        {
            "description": "Strong hand on the flop with medium pot",
            "hole_cards": ["Ah", "Kh"],
            "community_cards": ["Jh", "Th", "2s"],
            "round_name": "flop",
            "pot_size": 50,
            "position": "BB",
            "available_actions": [0, 1, 2]  # All actions available
        },
        {
            "description": "Mediocre hand on the turn with big pot",
            "hole_cards": ["9s", "Ts"],
            "community_cards": ["2h", "5c", "Kd", "7s"],
            "round_name": "turn",
            "pot_size": 100,
            "position": "SB",
            "available_actions": [0, 1, 2]
        },
        {
            "description": "Drawing hand on the river with huge pot",
            "hole_cards": ["Jh", "Th"],
            "community_cards": ["9h", "Qc", "2s", "3d", "8d"],
            "round_name": "river",
            "pot_size": 200,
            "position": "BB",
            "available_actions": [1, 2]  # Can only call or raise
        }
    ]
    
    # Test the strategy on these scenarios
    print("\n===== TESTING ENHANCED BLUEPRINT STRATEGY =====")
    action_names = ["Fold/Check", "Call", "Raise"]
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['description']}")
        
        # Get information set key
        infoset = blueprint.create_infoset_key(
            scenario['hole_cards'], 
            scenario['community_cards'], 
            scenario['round_name'], 
            scenario['pot_size'],
            scenario['position']
        )
        print(f"Information set: {infoset}")
        
        # Calculate hand strength
        hand_strength = blueprint.calculate_hand_strength(
            scenario['hole_cards'], 
            scenario['community_cards'], 
            scenario['round_name']
        )
        print(f"Hand strength: {hand_strength:.3f}")
        
        # Get strategy for this infoset
        strategy = blueprint.get_average_strategy(infoset)
        print("Strategy probabilities:")
        for i, prob in enumerate(strategy):
            print(f"  {action_names[i]}: {prob:.3f}")
        
        # Sample actions multiple times to see distribution
        action_counts = [0, 0, 0]
        num_samples = 1000
        
        for _ in range(num_samples):
            action = blueprint.get_action(
                scenario['hole_cards'], 
                scenario['community_cards'], 
                scenario['round_name'], 
                scenario['pot_size'],
                scenario['position'],
                scenario['available_actions']
            )
            action_counts[action] += 1
        
        print(f"Action distribution over {num_samples} samples:")
        for i, count in enumerate(action_counts):
            if i in scenario['available_actions']:
                print(f"  {action_names[i]}: {count/num_samples:.3f}")
    
    # Interactive testing
    print("\n===== INTERACTIVE TESTING =====")
    print("Enter your cards and board cards to see the recommended strategy")
    print("Example hole cards: Ah Kh")
    print("Example board: Jh Th 2s")
    
    while True:
        try:
            print("\n--- New Hand ---")
            hole_input = input("Enter your hole cards (or 'quit' to exit): ")
            if hole_input.lower() == 'quit':
                break
                
            hole_cards = hole_input.split()
            if len(hole_cards) != 2:
                print("Please enter exactly 2 hole cards")
                continue
                
            board_input = input("Enter board cards (or press Enter for preflop): ")
            if board_input.strip():
                community_cards = board_input.split()
            else:
                community_cards = []
                
            round_name = "preflop"
            if len(community_cards) >= 3:
                round_name = "flop"
            if len(community_cards) >= 4:
                round_name = "turn"
            if len(community_cards) >= 5:
                round_name = "river"
                
            pot_size = int(input("Enter pot size: "))
            position = input("Enter position (SB/BB): ").upper()
            
            # Calculate hand strength
            hand_strength = blueprint.calculate_hand_strength(hole_cards, community_cards, round_name)
            print(f"Hand strength: {hand_strength:.3f}")
            
            # Get strategy
            # Get strategy
            infoset = blueprint.create_infoset_key(hole_cards, community_cards, round_name, pot_size, position)
            strategy = blueprint.get_average_strategy(infoset)
           
            print(f"Strategy for this situation:")
            for i, prob in enumerate(["Fold/Check", "Call", "Raise"]):
                print(f"  {prob}: {strategy[i]:.3f}")
                
            recommended_action = np.argmax(strategy)
            print(f"Recommended action: {['Fold/Check', 'Call', 'Raise'][recommended_action]}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again")


def main():
   # Generate training data or load existing data
   print("Generating training data from simulated self-play...")
   training_data = generate_training_data(num_hands=5000)
   print(f"Generated {len(training_data)} poker hands")
   
   # Initialize and train blueprint strategy
   print("Initializing enhanced blueprint strategy...")
   blueprint = EnhancedBlueprintStrategy(num_actions=3)
   
   # Try to load existing strategy first
   if not blueprint.load():
       # Train new strategy if no existing one found
       print("No existing strategy found. Training new strategy...")
       blueprint.train_from_data(training_data, num_iterations=5000)
       blueprint.save()
   
   # Print the learned strategy
   blueprint.print_strategy(top_n=15)
   
   # Run tests
   should_test = input("Would you like to test the strategy? (y/n): ")
   if should_test.lower() == 'y':
       test_blueprint_strategy()


if __name__ == "__main__":
   main()