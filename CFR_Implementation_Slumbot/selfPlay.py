import os
import time
import pickle
import torch
import numpy as np
from tqdm import tqdm
import sys

# Add path to import modules if needed
if '/content/assignment' not in sys.path:
    sys.path.insert(0, '/content/assignment')

from pokerAgent import PokerAgent
from pokerGameState import PokerGameState
from handEvaluator import HandEvaluator
from depthLimitedSolver import DepthLimitedSolver
from depthLimitedCFR import DepthLimitedCFR

def save_strategy(agent, path):
    """Save the current strategy to a file."""
    agent.blueprint_cfr.save(path)
    print(f"Strategy saved to {path} ({len(agent.blueprint_cfr.nodes)} information sets)")
    return path

def save_checkpoint(agent, stats, hand_num, path):
    """Save a complete checkpoint with training state."""
    checkpoint = {
        'hand': hand_num,
        'nodes': agent.blueprint_cfr.nodes,
        'hands_played': agent.hands_played,
        'new_experiences': agent.new_experiences,
        'stats': stats
    }
    
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved at hand {hand_num} to {path}")
    return path

def load_checkpoint(agent, opponent, path):
    """Load a checkpoint for both agents."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        agent.blueprint_cfr.nodes = checkpoint['nodes']
        agent.hands_played = checkpoint['hands_played']
        agent.new_experiences = checkpoint['new_experiences']
        
        # Update the opponent to use the same blueprint
        opponent.blueprint_cfr = agent.blueprint_cfr
        opponent.solver = DepthLimitedSolver(agent.blueprint_cfr, max_depth=3)
        
        print(f"Loaded checkpoint from hand {checkpoint['hand']} with {len(agent.blueprint_cfr.nodes)} nodes")
        return checkpoint['hand'], checkpoint.get('stats', {'agent_wins': 0, 'opponent_wins': 0, 'ties': 0})
    else:
        print(f"No checkpoint found at {path}, starting from scratch")
        return 0, {'agent_wins': 0, 'opponent_wins': 0, 'ties': 0}

def self_play_training(
    num_hands=10000,
    save_every=500,
    save_path='/content/gdrive/MyDrive/CFRCode/selfplay',
    blueprint_path=None,
    resume_path=None,
    stack_size=1000,
    small_blind=5,
    big_blind=10
):
    """
    Train a poker agent through self-play.
    
    Args:
        num_hands: Number of hands to play
        save_every: Save checkpoint every N hands
        save_path: Base path for saving checkpoints
        blueprint_path: Path to initial blueprint strategy (optional)
        resume_path: Path to resume training from a checkpoint
        stack_size: Starting stack size for each player
        small_blind: Small blind amount
        big_blind: Big blind amount
        
    Returns:
        Trained agent and path to final strategy
    """
    print(f"Starting self-play training for {num_hands} hands")
    
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Add timestamp to save path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_path}_{timestamp}"
    
    # Initialize the agent with a blueprint
    agent = PokerAgent(
        blueprint_path=blueprint_path,
        stack_size=stack_size,
        small_blind=small_blind,
        big_blind=big_blind,
        enable_learning=True
    )
    
    # Create a second agent using the same blueprint
    opponent = PokerAgent(
        blueprint_path=blueprint_path,
        stack_size=stack_size,
        small_blind=small_blind,
        big_blind=big_blind,
        enable_learning=False
    )
    opponent.blueprint_cfr = agent.blueprint_cfr
    
    # Resume from checkpoint if specified
    start_hand = 0
    stats = {'agent_wins': 0, 'opponent_wins': 0, 'ties': 0}
    if resume_path:
        start_hand, stats = load_checkpoint(agent, opponent, resume_path)
    
    # Save initial checkpoint
    if start_hand == 0 and not resume_path:
        initial_checkpoint_path = f"{save_path}_hand_0.pkl"
        save_checkpoint(agent, stats, 0, initial_checkpoint_path)
    
    # Main training loop with progress bar
    start_time = time.time()
    try:
        with tqdm(total=num_hands, initial=start_hand) as pbar:
            for hand in range(start_hand, num_hands):
                # Alternate positions
                agent_position = hand % 2
                opponent_position = 1 - agent_position
                
                # Initialize the game state
                state = PokerGameState(
                    stack_size=stack_size,
                    small_blind=small_blind,
                    big_blind=big_blind
                )
                state.deal_hole_cards()
                
                # Set up both agents with the initial state
                agent.current_state = state.clone()
                agent.position = agent_position
                opponent.current_state = state.clone()
                opponent.position = opponent_position
                
                # Play the hand
                hand_result = None
                while state is not None and not state.is_terminal():
                    current_player = state.current_player
                    
                    if current_player == agent_position:
                        # Agent's turn
                        action, raise_amount = agent.act()
                        
                        # Update both agents' states
                        new_state = state.act(action, raise_amount)
                        
                        if new_state is None:  # Hand ended due to fold
                            stats['opponent_wins'] += 1
                            hand_result = 'opponent_fold'
                            break
                            
                        state = new_state
                        agent.current_state = state.clone()
                        opponent.observe_opponent_action(action, raise_amount)
                        
                    else:
                        # Opponent's turn
                        action, raise_amount = opponent.act()
                        
                        # Update both agents' states
                        new_state = state.act(action, raise_amount)
                        
                        if new_state is None:  # Hand ended due to fold
                            stats['agent_wins'] += 1
                            hand_result = 'agent_fold'
                            break
                            
                        state = new_state
                        opponent.current_state = state.clone()
                        agent.observe_opponent_action(action, raise_amount)
                    
                    # Deal community cards if needed
                    if state and state.player_bets[0] == state.player_bets[1] and state.current_round < PokerGameState.RIVER:
                        state.current_round += 1
                        state.deal_community_cards()
                        agent.current_state = state.clone()
                        opponent.current_state = state.clone()
                        
                # If hand ended in showdown, evaluate the winner
                if state and state.is_terminal() and hand_result is None:
                    agent_hand = HandEvaluator.evaluate_hand(state.player_hole_cards[agent_position], state.board)
                    opponent_hand = HandEvaluator.evaluate_hand(state.player_hole_cards[opponent_position], state.board)
                    
                    if agent_hand < opponent_hand:  # Lower is better
                        stats['agent_wins'] += 1
                        hand_result = 'agent_showdown'
                    elif agent_hand > opponent_hand:
                        stats['opponent_wins'] += 1
                        hand_result = 'opponent_showdown'
                    else:
                        stats['ties'] += 1
                        hand_result = 'tie'
                        
                # Update and save blueprint
                agent.update_blueprint_from_solver()
                
                # Update progress bar
                pbar.update(1)
                win_rate = 100 * stats['agent_wins'] / max(1, sum(stats.values()))
                if (hand + 1) % 10 == 0:  # Update display every 10 hands
                    pbar.set_postfix({
                        'nodes': len(agent.blueprint_cfr.nodes),
                        'win%': f"{win_rate:.1f}%",
                        'hands': sum(stats.values())
                    })
                
                # Save checkpoint periodically
                if (hand + 1) % save_every == 0 or hand == num_hands - 1:
                    checkpoint_path = f"{save_path}_hand_{hand+1}.pkl"
                    save_checkpoint(agent, stats, hand+1, checkpoint_path)
                    
                    # Also save the strategy
                    strategy_path = f"{save_path}_strategy_{hand+1}.pkl"
                    save_strategy(agent, strategy_path)
                    
                    # Clear CUDA cache to prevent memory fragmentation
                    if torch.cuda.is_available() and (hand + 1) % 2000 == 0:
                        torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        interrupted_path = f"{save_path}_interrupted_hand_{hand}.pkl"
        save_checkpoint(agent, stats, hand, interrupted_path)
        
        # Save the strategy too
        interrupted_strategy = f"{save_path}_interrupted_strategy.pkl"
        save_strategy(agent, interrupted_strategy)
    
    # Final save
    final_checkpoint_path = f"{save_path}_final.pkl"
    save_checkpoint(agent, stats, num_hands, final_checkpoint_path)
    
    final_strategy_path = f"{save_path}_final_strategy.pkl"
    save_strategy(agent, final_strategy_path)
    
    # Print training statistics
    training_time = time.time() - start_time
    hands_per_second = (num_hands - start_hand) / max(0.1, training_time)
    
    print(f"\nSelf-play training completed:")
    print(f"- Hands played: {num_hands - start_hand}")
    print(f"- Training time: {training_time:.2f}s ({hands_per_second:.2f} hands/second)")
    print(f"- Agent wins: {stats['agent_wins']} ({100*stats['agent_wins']/max(1,sum(stats.values())):.1f}%)")
    print(f"- Opponent wins: {stats['opponent_wins']} ({100*stats['opponent_wins']/max(1,sum(stats.values())):.1f}%)")
    print(f"- Ties: {stats['ties']} ({100*stats['ties']/max(1,sum(stats.values())):.1f}%)")
    print(f"- Total info sets: {len(agent.blueprint_cfr.nodes)}")
    print(f"Final strategy saved to: {final_strategy_path}")
    
    return agent, final_strategy_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train poker agent through self-play")
    parser.add_argument("--hands", type=int, default=10000,
                        help="Number of hands to play")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Save checkpoint every N hands")
    parser.add_argument("--save-path", type=str, 
                        default="/content/gdrive/MyDrive/CFRCode/selfplay",
                        help="Path to save checkpoints and strategy")
    parser.add_argument("--blueprint", type=str, default=None,
                        help="Path to initial blueprint strategy (optional)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--stack-size", type=int, default=1000,
                        help="Starting stack size")
    parser.add_argument("--small-blind", type=int, default=5,
                        help="Small blind amount")
    parser.add_argument("--big-blind", type=int, default=10,
                        help="Big blind amount")
    
    args = parser.parse_args()
    
    # Run self-play training
    agent, final_path = self_play_training(
        num_hands=args.hands,
        save_every=args.save_every,
        save_path=args.save_path,
        blueprint_path=args.blueprint,
        resume_path=args.resume,
        stack_size=args.stack_size,
        small_blind=args.small_blind,
        big_blind=args.big_blind
    )
    
    print(f"Training complete! Final strategy saved to: {final_path}")