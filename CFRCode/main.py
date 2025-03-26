from collections import defaultdict
import pickle
import random
import numpy as np
import time

from itertools import combinations
from depthLimitedSolver import DepthLimitedSolver
from pokerGameState import PokerGameState
from opponentModel import OpponentModel
from pokerAgent import PokerAgent
from CFRNode import CFRNode
from depthLimitedCFR import DepthLimitedCFR
from handEvaluator import HandEvaluator

def play_against_agent(agent, strategy="random"):
    """
    Play a hand of poker against the agent.
    
    Args:
        agent: The PokerAgent to play against
        strategy: Strategy to use for the opponent ("random", "tight", "loose", "aggressive")
        
    Returns:
        Final game state and outcome
    """
    import time
    
    # Start a new hand with agent in position 0 (small blind)
    state = agent.start_new_hand(position=0)
    
    print("\n--- New Hand ---")
    print(f"Your hole cards: {[str(card) for card in state.player_hole_cards[1]]}")
    print(f"Blinds: {state.small_blind}/{state.big_blind}")
    print(f"Starting stacks: {state.player_stacks[0]}/{state.player_stacks[1]}")
    
    # Set timeout to prevent infinite loops
    start_time = time.time()
    max_time = 60  # 60 seconds max
    
    # Loop until hand is over
    while state is not None:
        # Check for timeout
        if time.time() - start_time > max_time:
            print("Game timed out - breaking loop")
            break
            
        print(f"\nDEBUG: Current player: {state.current_player}, Round: {state.current_round}")
        
        # If it's the agent's turn
        if state.current_player == 0:
            print("\nAgent is thinking...")
            # Save pot value before agent action in case of fold
            pot_before_action = state.pot
            
            action, raise_amount = agent.act()
            
            if action is None:
                print("ERROR: Agent returned None action. This shouldn't happen when it's the agent's turn.")
                break
                
            # Update the reference to the game state after agent's action
            new_state = agent.current_state
            
            print(f"DEBUG: After agent action: player={new_state.current_player if new_state else 'None'}, round={new_state.current_round if new_state else 'None'}")
            
            if action == PokerGameState.FOLD:
                print("Agent folds.")
                print(f"You win {pot_before_action}.")
                
                # Update blueprint before returning
                if agent.enable_learning:
                    agent.update_blueprint_from_solver()
                    
                return new_state, 1  # Player wins
            elif action == PokerGameState.CHECK_CALL:
                if state.player_bets[0] < state.player_bets[1]:
                    print(f"Agent calls {state.player_bets[1] - state.player_bets[0]}.")
                else:
                    print("Agent checks.")
            elif action == PokerGameState.BET_RAISE:
                print(f"Agent raises to {state.player_bets[0] + raise_amount}.")
            
            # Update state after printing action
            state = new_state
            
            # Check if we need to deal community cards
            if state is not None and state.player_bets[0] == state.player_bets[1]:
                if state.current_round < PokerGameState.RIVER:
                    # Round is complete, need to deal cards
                    state.current_round += 1
                    cards = agent.deal_next_round()
                    print(f"\nDealing {'flop' if len(cards) == 3 else 'turn' if len(cards) == 4 else 'river'}: {[str(card) for card in cards]}")
                    state.player_bets = [0, 0]
        
        # If it's the player's turn
        elif state.current_player == 1:
            print("\nYour turn:")
            # Display game state
            print("Current pot:", state.pot)
            if state.board:
                print("Board:", [str(card) for card in state.board])
            print(f"Agent stack: {state.player_stacks[0]}, Your stack: {state.player_stacks[1]}")
            print(f"Agent bet: {state.player_bets[0]}, Your bet: {state.player_bets[1]}")
            
            # Save pot value before player action in case of fold
            pot_before_action = state.pot
            
            # Get legal actions
            legal_actions = state.get_legal_actions()
            action_names = ["fold", "check/call", "bet/raise"]
            legal_action_names = [action_names[a] for a in legal_actions]
            print(f"Available actions: {legal_action_names}")
            
            # Choose action based on strategy
            if strategy == "random":
                # Choose random action
                player_action = random.choice(legal_actions)
                
                # Choose random raise amount if raising
                player_raise = None
                if player_action == PokerGameState.BET_RAISE:
                    call_amount = state.player_bets[0] - state.player_bets[1]
                    max_raise = min(state.player_stacks[1] - call_amount, state.pot * 2)
                    player_raise = random.randint(state.min_raise, max(state.min_raise, max_raise))
                    
            elif strategy == "tight":
                # Tight strategy: only play strong hands
                hand_strength = HandEvaluator.calculate_equity(
                    state.player_hole_cards[1], state.board, num_simulations=100)
                
                if hand_strength > 0.7:
                    # Strong hand: raise if possible, otherwise call
                    if PokerGameState.BET_RAISE in legal_actions:
                        player_action = PokerGameState.BET_RAISE
                        player_raise = min(state.pot, state.player_stacks[1])
                    else:
                        player_action = PokerGameState.CHECK_CALL
                elif hand_strength > 0.5:
                    # Medium hand: call
                    player_action = PokerGameState.CHECK_CALL
                else:
                    # Weak hand: check if possible, otherwise fold
                    if state.player_bets[0] == state.player_bets[1]:
                        player_action = PokerGameState.CHECK_CALL  # Check
                    else:
                        player_action = PokerGameState.FOLD
                        
            elif strategy == "loose":
                # Loose strategy: play many hands
                hand_strength = HandEvaluator.calculate_equity(
                    state.player_hole_cards[1], state.board, num_simulations=100)
                
                if hand_strength > 0.5:
                    # Decent hand: raise
                    if PokerGameState.BET_RAISE in legal_actions:
                        player_action = PokerGameState.BET_RAISE
                        player_raise = state.min_raise
                    else:
                        player_action = PokerGameState.CHECK_CALL
                else:
                    # Any hand: call
                    player_action = PokerGameState.CHECK_CALL
                    
            elif strategy == "aggressive":
                # Aggressive strategy: bet and raise often
                hand_strength = HandEvaluator.calculate_equity(
                    state.player_hole_cards[1], state.board, num_simulations=100)
                
                if PokerGameState.BET_RAISE in legal_actions and (hand_strength > 0.4 or random.random() < 0.3):
                    # Raise with good hands or bluff occasionally
                    player_action = PokerGameState.BET_RAISE
                    player_raise = random.choice([
                        state.min_raise,
                        state.pot // 2,
                        state.pot
                    ])
                else:
                    # Otherwise call
                    player_action = PokerGameState.CHECK_CALL
            
            # Print player action
            if player_action == PokerGameState.FOLD:
                print("You fold.")
                print(f"Agent wins {pot_before_action}.")
                
                # Update blueprint before returning
                if agent.enable_learning:
                    agent.update_blueprint_from_solver()
                    
                return None, 0  # Agent wins
            elif player_action == PokerGameState.CHECK_CALL:
                if state.player_bets[0] > state.player_bets[1]:
                    print(f"You call {state.player_bets[0] - state.player_bets[1]}.")
                else:
                    print("You check.")
            elif player_action == PokerGameState.BET_RAISE:
                print(f"You raise to {state.player_bets[1] + player_raise}.")
            
            # Update the game state
            state = agent.observe_opponent_action(player_action, player_raise)
            print(f"DEBUG: After player action: player={state.current_player if state else 'None'}, round={state.current_round if state else 'None'}")
            
            # Check if we need to deal community cards
            if state is not None and state.player_bets[0] == state.player_bets[1]:
                if state.current_round < PokerGameState.RIVER:
                    # Round is complete, need to deal cards
                    state.current_round += 1
                    cards = agent.deal_next_round()
                    print(f"\nDealing {'flop' if len(cards) == 3 else 'turn' if len(cards) == 4 else 'river'}: {[str(card) for card in cards]}")
                    state.player_bets = [0, 0]
        
        # Check if hand is over due to showdown
        try:
            is_terminal = state is not None and state.is_terminal()
        except Exception as e:
            print(f"ERROR in terminal check: {e}")
            is_terminal = False
            
        if state is not None and is_terminal:
            # Evaluate hands
            print("\n--- Showdown ---")
            print(f"Agent's hand: {[str(card) for card in state.player_hole_cards[0]]}")
            print(f"Your hand: {[str(card) for card in state.player_hole_cards[1]]}")
            print(f"Board: {[str(card) for card in state.board]}")
            
            player_hand = HandEvaluator.evaluate_hand(state.player_hole_cards[0], state.board)
            opponent_hand = HandEvaluator.evaluate_hand(state.player_hole_cards[1], state.board)
            
            result = None
            if player_hand < opponent_hand:  # Lower is better
                print(f"Agent wins {state.pot}.")
                result = 0  # Agent wins
            elif player_hand > opponent_hand:
                print(f"You win {state.pot}.")
                result = 1  # Player wins
            else:
                print(f"Split pot. Each player receives {state.pot / 2}.")
                result = 0.5  # Tie
            
            # Update blueprint before returning
            if agent.enable_learning:
                agent.update_blueprint_from_solver()
                
            return state, result
                
    return state, None  # Shouldn't reach here
      
def self_play_training(num_hands=1000, save_path="self_play_blueprint.pkl", enable_learning=True):
    """
    Train the agent through self-play.
    
    This simulates the self-play training approach used in Pluribus,
    where the agent plays against itself to improve its strategy.
    
    Args:
        num_hands: Number of hands to play
        save_path: Path to save the trained blueprint
        enable_learning: Whether to enable continuous learning
        
    Returns:
        The trained agent
    """
    print(f"Starting self-play training for {num_hands} hands...")
    
    # Initialize the agent with a small blueprint
    agent = PokerAgent(blueprint_path=save_path, enable_learning=enable_learning)
    
    # Create a second agent using the same blueprint
    opponent = PokerAgent(blueprint_path=save_path, enable_learning=False)
    opponent.blueprint_cfr = agent.blueprint_cfr
    opponent.solver = DepthLimitedSolver(agent.blueprint_cfr, max_depth=3)
    
    # Statistics
    agent_wins = 0
    opponent_wins = 0
    ties = 0
    
    for hand in range(num_hands):
        if hand % 100 == 0:
            print(f"Playing hand {hand}/{num_hands}")
        
        # Alternate positions
        agent_position = hand % 2
        opponent_position = 1 - agent_position
        
        # Initialize the game state
        state = PokerGameState()
        state.deal_hole_cards()
        
        # Keep track of the game state for both agents
        agent.current_state = state.clone()
        agent.position = agent_position
        opponent.current_state = state.clone()
        opponent.position = opponent_position
        
        # Play the hand
        while state is not None and not state.is_terminal():
            current_player = state.current_player
            
            if current_player == agent_position:
                # Agent's turn
                action, raise_amount = agent.solver.get_action(state, agent_position)
                
                # Update both agents' states
                new_state = state.act(action, raise_amount)
                
                if new_state is None:  # Hand ended due to fold
                    opponent_wins += 1
                    break
                    
                state = new_state
                agent.current_state = state.clone()
                opponent.observe_opponent_action(action, raise_amount)
                
            else:
                # Opponent's turn
                action, raise_amount = opponent.solver.get_action(state, opponent_position)
                
                # Update both agents' states
                new_state = state.act(action, raise_amount)
                
                if new_state is None:  # Hand ended due to fold
                    agent_wins += 1
                    break
                    
                state = new_state
                opponent.current_state = state.clone()
                agent.observe_opponent_action(action, raise_amount)
            
            # Deal community cards if needed
            if state.player_bets[0] == state.player_bets[1] and state.current_round < PokerGameState.RIVER:
                state.current_round += 1
                state.deal_community_cards()
                agent.current_state = state.clone()
                opponent.current_state = state.clone()
                
        # If hand ended in showdown, evaluate the winner
        if state and state.is_terminal():
            agent_hand = HandEvaluator.evaluate_hand(state.player_hole_cards[agent_position], state.board)
            opponent_hand = HandEvaluator.evaluate_hand(state.player_hole_cards[opponent_position], state.board)
            
            if agent_hand < opponent_hand:  # Lower is better
                agent_wins += 1
            elif agent_hand > opponent_hand:
                opponent_wins += 1
            else:
                ties += 1
                
        # Update and save blueprint after each hand if learning is enabled
        if enable_learning:
            agent.update_blueprint_from_solver()
    
    # Print results
    print("\nSelf-play training completed.")
    print(f"Results: Agent wins: {agent_wins}, Opponent wins: {opponent_wins}, Ties: {ties}")
    
    # No need for final save since we save after each hand
    
    return agent

def analyze_learning(blueprint_path, compare_path=None):
    """
    Analyze what the agent has learned by comparing blueprints.
    
    Args:
        blueprint_path: Path to current blueprint
        compare_path: Optional path to another blueprint for comparison
    """
    import pickle
    import numpy as np
    
    with open(blueprint_path, 'rb') as f:
        current = pickle.load(f)
    
    print(f"Loaded blueprint with {len(current)} information sets")
    print(f"Blueprint file: {blueprint_path}")
    
    # Display some statistics about the blueprint
    action_counts = {"fold": 0, "call": 0, "raise": 0}
    for strategy in current.values():
        max_action = np.argmax(strategy)
        if max_action == 0:
            action_counts["fold"] += 1
        elif max_action == 1:
            action_counts["call"] += 1
        elif max_action == 2:
            action_counts["raise"] += 1
    
    print("\nAction distribution (preferred action):")
    for action, count in action_counts.items():
        percentage = count / len(current) * 100 if len(current) > 0 else 0
        print(f"  {action}: {count} ({percentage:.1f}%)")
    
    if compare_path:
        with open(compare_path, 'rb') as f:
            previous = pickle.load(f)
        
        print(f"\nComparing with: {compare_path}")
        print(f"Comparison blueprint has {len(previous)} information sets")
        
        # Find new info sets
        new_info_sets = set(current.keys()) - set(previous.keys())
        print(f"New information sets learned: {len(new_info_sets)}")
        
        # Find changed strategies
        changed = 0
        for info_set in set(current.keys()).intersection(set(previous.keys())):
            if not np.array_equal(current[info_set], previous[info_set]):
                changed += 1
        
        print(f"Information sets with updated strategies: {changed}")

def main():
    """Main function to demonstrate the poker agent."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Poker AI with depth-limited solving")
    parser.add_argument("--mode", choices=["play", "train", "self_play", "analyze"], default="play",
                        help="Mode: play against AI, train blueprint, self-play training, or analyze learning")
    parser.add_argument("--blueprint", type=str, default="blueprint.pkl",
                        help="Path to blueprint strategy file")
    parser.add_argument("--hands", type=int, default=10,
                        help="Number of hands to play or train")
    parser.add_argument("--strategy", choices=["random", "tight", "loose", "aggressive"], 
                        default="random", help="Strategy to use when playing against the AI")
    parser.add_argument("--enable-learning", action="store_true", default=True,
                        help="Enable continuous learning from gameplay")
    parser.add_argument("--backup", type=str, default=None,
                        help="Backup file path for analysis comparison")
    parser.add_argument("--compare-with", type=str, default=None,
                        help="Alternative blueprint file path for comparison in analysis")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Train a new blueprint strategy
        cfr = DepthLimitedCFR(max_depth=2)
        cfr.train(num_iterations=args.hands * 100)
        cfr.save(args.blueprint)
        print(f"Blueprint strategy trained and saved to {args.blueprint}")
        
    elif args.mode == "self_play":
        # Train through self-play
        self_play_training(num_hands=args.hands, save_path=args.blueprint, enable_learning=args.enable_learning)
        
    elif args.mode == "analyze":
        # Analyze learning progress
        analyze_learning(args.blueprint, args.backup)
        
    elif args.mode == "play":
        # Play against the AI
        print("Initializing poker agent...")
        agent = PokerAgent(blueprint_path=args.blueprint, enable_learning=args.enable_learning)
        
        # Play multiple hands
        agent_wins = 0
        player_wins = 0
        ties = 0
        
        for hand_num in range(args.hands):
            print(f"\nPlaying hand {hand_num+1}/{args.hands}")
            _, result = play_against_agent(agent, strategy=args.strategy)
            
            if result == 0:
                agent_wins += 1
            elif result == 1:
                player_wins += 1
            elif result == 0.5:
                ties += 1
        
        # Print final results
        print("\n--- Final Results ---")
        print(f"Hands played: {args.hands}")
        print(f"Agent wins: {agent_wins}")
        print(f"Player wins: {player_wins}")
        print(f"Ties: {ties}")
        
        # Show learning summary
        if args.enable_learning:
            print(f"\nAI learned from {agent.hands_played} hands")
            print(f"Total information sets in blueprint: {len(agent.blueprint_cfr.nodes)}")
            print(f"New experiences acquired: {len(agent.new_experiences)}")


if __name__ == "__main__":
    main()