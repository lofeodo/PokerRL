import os
import sys

os.chdir(os.path.join(os.path.dirname(__file__), '..', 'PyStack'))
print(f"Changed working directory to: {os.getcwd()}")
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
from Settings.arguments import Parameters
from Game.card_tools import CardTools
from Game.card_combinations import CardCombinations
from Game.card_to_string_conversion import CardToStringConversion
from NeuralNetwork.value_nn import ValueNn
from DataGeneration.range_generator import RangeGenerator
from TerminalEquity.terminal_equity import TerminalEquity
from Lookahead.resolving import Resolving
from helper_classes import Node
from Settings.constants import constants

class DeepStackBenchmark:
    def __init__(self):
        self.arguments = Parameters()
        self.card_tools = CardTools()
        self.card_to_str = CardToStringConversion()
        self.range_generator = RangeGenerator()
        self.term_eq = TerminalEquity()
        
    def get_bb_per_hand(self, num_iterations=1000, verbose=False):
        total_value = 0
        HC = constants.hand_count
        
        if verbose:
            print(f"Starting benchmark with {num_iterations} iterations")
            
        for i in range(num_iterations):
            if verbose:
                print(f"\nIteration {i+1}/{num_iterations}")
                
            # Start at flop (street 2) since preflop isn't supported
            # Generate random flop
            board = np.random.choice(52, size=3, replace=False)
            if verbose:
                print(f"Generated flop: {[self.card_to_str.card_to_string(c) for c in board]}")
            
            # Generate random ranges for both players
            self.term_eq.set_board(board)
            hand_strengths = self.term_eq.get_hand_strengths()
            self.range_generator.set_board(hand_strengths, board)
            
            ranges = np.zeros([2, 1, HC], dtype=self.arguments.dtype)
            for player in range(2):
                self.range_generator.generate_range(ranges[player])
                if verbose:
                    print(f"Player {player+1} range generated")
            
            # Initial pot size (assume standard preflop action)
            pot_size = self.arguments.ante + 3 * self.arguments.bb  # Typical pot after preflop
            if verbose:
                print(f"Initial pot size: {pot_size}")
            
            # Solve each street
            for street in [2, 3, 4]:  # flop, turn, river
                if verbose:
                    print(f"\nSolving street {street}")
                    
                resolving = Resolving(self.term_eq)
                current_node = Node()
                current_node.board = board
                current_node.street = street
                current_node.current_player = constants.players.P2
                current_node.bets = np.array([pot_size, pot_size], dtype=self.arguments.dtype)
                
                # Resolve and get values
                results = resolving.resolve(current_node, ranges[0], ranges[1])
                root_values = results.root_cfvs_both_players
                
                # Track P1's expected value
                p1_ev = np.sum(root_values[0, 0, :] * ranges[0, 0, :])
                total_value += p1_ev
                
                if verbose:
                    print(f"Street {street} EV: {p1_ev:.3f}")
                
                if street < 4:  # If not river, deal next card
                    next_card = self._deal_next_card(board)
                    if verbose:
                        print(f"Dealt {self.card_to_str.card_to_string(next_card)}")
                    board = np.append(board, next_card)
                    
                # Update pot size based on typical betting
                pot_size *= 2  # Assume pot roughly doubles each street
                if verbose:
                    print(f"Updated pot size: {pot_size}")
                
        # Convert to BB/hand
        bb_per_hand = (total_value / num_iterations) / self.arguments.bb
        if verbose:
            print(f"\nBenchmark complete")
            print(f"Total value: {total_value:.3f}")
            print(f"BB per hand: {bb_per_hand:.3f}")
        return bb_per_hand
    
    def _deal_next_card(self, current_board):
        available_cards = np.setdiff1d(np.arange(52), current_board)
        return np.random.choice(available_cards)

if __name__ == "__main__":
    benchmark = DeepStackBenchmark()
    print("Running DeepStack benchmark...")
    bb_per_hand = benchmark.get_bb_per_hand(num_iterations=100, verbose=True)  # Adjust iterations as needed
    print(f"Average BB/hand: {bb_per_hand:.3f}")
    print("Note: This benchmark starts from flop since preflop is not supported")