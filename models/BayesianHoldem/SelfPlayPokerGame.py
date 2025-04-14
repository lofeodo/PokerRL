from abc import ABC, abstractmethod
from utilities import Utilities
import time
import pokerkit as pk
from BayesianHoldem import BayesianHoldem
import torch
import torch.optim as optim
from typing import List, Dict, Optional, Tuple, Union
import os
from GameRepresentations import GameRepresentations
from datetime import datetime, timedelta
from tqdm import tqdm  # Add this import at the top of your file
import random  # Add at the top with other imports

# ==================================================

class SelfPlayPokerGame(ABC):
    @abstractmethod
    def __init__(self, learning_rate: float = 0.001, best_model_path: Optional[str] = None):
        pass

    # ==================================================

    def _update_state(self, player0_stack: int, player1_stack: int):
        big_blind = self.BB
        small_blind = 0.5*big_blind
        blinds = [small_blind, big_blind]
        ante = 0
        min_bet = 0.5*big_blind
        player_count = 2

        game = pk.NoLimitTexasHoldem(
            # Automations - everything is automatic except for player actions
            (
                pk.Automation.ANTE_POSTING,
                pk.Automation.BET_COLLECTION,
                pk.Automation.BLIND_OR_STRADDLE_POSTING,
                pk.Automation.CARD_BURNING,
                pk.Automation.HOLE_DEALING,
                pk.Automation.BOARD_DEALING,
                pk.Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                pk.Automation.HAND_KILLING,
                pk.Automation.CHIPS_PUSHING,
                pk.Automation.CHIPS_PULLING,
            ),
            ante_trimming_status=True, # irrelevant but necessary for the game to be created
            raw_antes=ante,
            raw_blinds_or_straddles=blinds,
            min_bet=min_bet,
        )

        self.state = game(raw_starting_stacks=[player0_stack, player1_stack], player_count=player_count)

    # ==================================================

    def _play_action(self, player_id: int, action: int, verbose: bool = False) -> Dict:
        """Execute an action and return transition information."""
        other_player_id = abs(1 - player_id)

        # Store pre-action state
        pre_state = {
            'bets': self.state.bets.copy(),
            'stacks': self.state.stacks.copy(),
            'pot_size': self.state.total_pot_amount,
            'street_index': self.state.street_index,
            'board_cards': self.state.board_cards.copy() if self.state.board_cards else None
        }

        # Check the current state of the game
        current_bet = self.state.bets[other_player_id]  # The bet made by the other player
        player_stack = self.state.stacks[player_id]  # The current stack of the player
        other_player_stack = self.state.stacks[other_player_id]

        if action == 0:  # fold
            prev_action = self.state.operations[-1]
            if isinstance(prev_action, pk.state.CheckingOrCalling) or \
                isinstance(prev_action, pk.state.HoleDealing) or \
                    isinstance(prev_action, pk.state.BoardDealing): # don't fold if the other player called or hasn't played
                if verbose:
                    print(f"Player {player_id} checks/calls")
                self.state.check_or_call()
            else:
                if verbose:
                    print(f"Player {player_id} folds")
                self.state.fold()

        elif action == 1:  # check/call
            if current_bet > self.state.bets[player_id]:  # If the other player has bet
                if player_stack < current_bet:  # Not enough chips to call
                    if verbose:
                        print(f"Player {player_id} cannot call, going all-in instead")
                    self.state.check_or_call()
                else:
                    if verbose:
                        print(f"Player {player_id} checks/calls")
                    self.state.check_or_call()
            else:  # If the current bet is equal to or less than the player's bet
                if verbose:
                    print(f"Player {player_id} checks")
                self.state.check_or_call()  # Just check

        elif action == 2:  # bet/raise BB
            if other_player_stack == 0:
                if verbose:
                    print(f"Player {player_id} matches player {other_player_id}'s all-in")
                self.state.check_or_call()
            else:
                bet_amount = max(self.state.bets[player_id], self.state.bets[other_player_id])
                raise_amount = bet_amount + self.BB
                if self.state.bets[player_id] + player_stack < raise_amount:
                    if verbose:
                        print(f"Player {player_id} does not have enough chips to raise, going all-in instead")
                    self.state.check_or_call()
                else:
                    if verbose:
                        print(f"Player {player_id} bets/raises {self.BB} to {raise_amount}")
                    self.state.complete_bet_or_raise_to(raise_amount)

        elif action == 3:  # all-in
            all_in_amount = player_stack + self.state.bets[player_id]
            if other_player_stack == 0 or self.state.bets[other_player_id] == all_in_amount:
                if verbose:
                    print(f"Player {player_id} matches other player's all-in")
                self.state.check_or_call()
            else:
                other_player_stack_and_bet = self.state.bets[other_player_id] + self.state.stacks[other_player_id]
                all_in_amount = min(all_in_amount, other_player_stack_and_bet) # don't bet more than the other play has
                if verbose:
                    print(f"Player {player_id} goes all-in for {all_in_amount}")
                self.state.complete_bet_or_raise_to(all_in_amount)

        # Store post-action state
        post_state = {
            'bets': self.state.bets.copy(),
            'stacks': self.state.stacks.copy(),
            'pot_size': self.state.total_pot_amount,
            'street_index': self.state.street_index,
            'board_cards': self.state.board_cards.copy() if self.state.board_cards else None
        }

        # Calculate state changes
        state_changes = {
            'bet_delta': post_state['bets'][player_id] - pre_state['bets'][player_id],
            'stack_delta': post_state['stacks'][player_id] - pre_state['stacks'][player_id],
            'pot_delta': post_state['pot_size'] - pre_state['pot_size'],
            'street_changed': post_state['street_index'] != pre_state['street_index'],
            'new_cards': None
        }

        # Check if new cards were dealt
        if post_state['board_cards'] and pre_state['board_cards']:
            if len(post_state['board_cards']) > len(pre_state['board_cards']):
                state_changes['new_cards'] = post_state['board_cards'][-1]

        # Return transition information
        return {
            'action': action,
            'player_id': player_id,
            'pre_state': pre_state,
            'post_state': post_state,
            'state_changes': state_changes,
            'is_terminal': self.state.street_index is None or any(stack <= 0 for stack in self.state.stacks)
        }

    # ==================================================

    def _train_player(self, 
                     player_id: int, 
                     pot_size: float,
                     action_representation: torch.Tensor,
                     card_representation: torch.Tensor,
                     transition: dict) -> Tuple[float, float, float]:
        """Train Player0 based on their action and the current game state."""
        if player_id != 0:  # Only train Player0
            return 0.0, 0.0, 0.0
            
        stack_size = float(self.state.stacks[player_id])
        bet_size = float(self.BB)
        max_stack = 20000  # Match the default max_stack in BayesianHoldem
        
        # Calculate normalized return
        if transition['is_terminal']:
            stack_delta = transition['post_state']['stacks'][player_id] - transition['pre_state']['stacks'][player_id]
            actual_return = stack_delta / max_stack  # Normalize by max_stack
            # Determine if player won based on stack delta
            is_winner = stack_delta > 0
        else:
            stack_delta = transition['state_changes']['stack_delta']
            # For non-terminal states, also include potential future value from pot
            pot_contribution = 0.5 * (pot_size / max_stack)  # Expected value from pot
            actual_return = (stack_delta / max_stack) + pot_contribution
            is_winner = None
        
        # Ensure tensors are on the correct device
        action_representation = action_representation.to(self.device)
        card_representation = card_representation.to(self.device)
        
        return self.Player0.train_step(
            action_representation=action_representation,
            card_representation=card_representation,
            pot_size=pot_size,
            stack_size=stack_size,
            bet_size=bet_size,
            actual_return=actual_return,
            is_terminal=transition['is_terminal'],
            is_winner=is_winner
        )

    # ==================================================

    def run_round(self, verbose: bool = False, training: bool = True) -> List[dict]:
        """Run a round of poker with optional training."""
        round_metrics = []
        
        if verbose:
            print("======= New Round =======")
            print(f"Player 0 cards: {self.state.hole_cards[0]}, chips: {self.state.stacks[0]}")
            print(f"Player 1 cards: {self.state.hole_cards[1]}, chips: {self.state.stacks[1]}")
        
        self.p0_action_tensor = torch.zeros((24, 4, 4))
        self.p1_action_tensor = torch.zeros((24, 4, 4))
        while self.state.street_index is not None:
            if verbose:
                print(f"Street index: {self.state.street_index}")
                print(f"Public cards: {self.state.board_cards}")
                print(f"Bets: P0: {self.state.bets[0]}, P1: {self.state.bets[1]}, Chips: P0: {self.state.stacks[0]}, P1: {self.state.stacks[1]}")
            
            # Current player's turn
            player_id = self.state.actor_index
            player = self.Player0 if player_id == 0 else self.Player1

            # Get state representations
            if player_id == 0:
                self.p0_action_tensor = GameRepresentations.get_action_representations(self.state, self.p0_action_tensor, player_id)
                action_representation = self.p0_action_tensor
            else:
                self.p1_action_tensor = GameRepresentations.get_action_representations(self.state, self.p1_action_tensor, player_id)
                action_representation = self.p1_action_tensor
            card_representation = GameRepresentations.get_card_representations(self.state, player_id)
            
            # Get action from player
            action = player.predict_action(action_representation, card_representation)
            
            # Execute action and get state transition info
            transition = self._play_action(player_id, action, verbose)

            pot_size = transition['pre_state']['pot_size'] if transition['is_terminal'] else transition['post_state']['pot_size']
            
            # Train if enabled
            if training:
                losses = self._train_player(
                    player_id,
                    pot_size,
                    action_representation,
                    card_representation,
                    transition  # Pass the transition information
                )
                
                # Record metrics
                round_metrics.append({
                    'player_id': player_id,
                    'action': action,
                    'losses': losses,
                    'transition': transition
                })
        
        return round_metrics

    # ==================================================

    def run_game(self, verbose: bool = False, training: bool = True) -> Dict:
        """Run a complete game with optional training."""
        initial_stack = 100 * self.BB  # Store initial stack size
        self._update_state(initial_stack, initial_stack)
        game_metrics = []

        while True:
            # Run a round and collect metrics
            round_metrics = self.run_round(verbose=verbose, training=training)
            game_metrics.extend(round_metrics)

            # Check if someone lost all their chips
            for idx, stack in enumerate(self.state.stacks):
                if stack <= 0:
                    if verbose:
                        print(f"Player {idx} has lost all their chips.")
                        print(f"Final stacks: {self.state.stacks}")
                    
                    # The player who lost all their chips (idx) is the loser
                    loser = idx
                    winner = abs(1 - loser)  # The other player is the winner
                    
                    # Calculate player 0's profit/loss directly from their stack change
                    player0_profit = self.state.stacks[0] - initial_stack
                    
                    final_rewards = {
                        0: float(player0_profit),      # Player 0's profit/loss
                        1: float(-player0_profit)      # Player 1's profit/loss
                    }
                    
                    if verbose:
                        print(f"Player {loser} lost all chips")
                        print(f"Player {winner} wins")
                        print(f"Player 0 profit: {player0_profit/self.BB:.2f} BB")
                        print(f"Player 1 profit: {-player0_profit/self.BB:.2f} BB")
                    
                    return {
                        'metrics': game_metrics,
                        'final_rewards': final_rewards,
                        'winner': winner
                    }

            self._update_state(self.state.stacks[0], self.state.stacks[1])

    # ==================================================

    def _calculate_bb_per_hand(self, stack_delta: Union[torch.Tensor, float], num_hands: int) -> torch.Tensor:
        """Calculate the average big blinds won per hand."""
        if num_hands == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Ensure stack_delta is a tensor
        if not isinstance(stack_delta, torch.Tensor):
            stack_delta = torch.tensor(stack_delta, device=self.device)
        
        # Convert stack delta to BB units and divide by number of hands
        bb_delta = stack_delta.float() / float(self.BB)
        return bb_delta / float(num_hands)

    def _update_elo(self, player0_win: bool, k_factor: float = 32) -> None:
        """
        Update Elo ratings after a game.
        Uses standard Elo formula with configurable K-factor.
        """
        # Convert current Elo ratings to tensors
        player0_elo_tensor = torch.tensor(self.player0_elo, device=self.device)
        player1_elo_tensor = torch.tensor(self.player1_elo, device=self.device)

        # Get expected win probabilities
        expected_p0 = 1.0 / (1.0 + 10**((player1_elo_tensor - player0_elo_tensor) / 400))
        expected_p1 = 1.0 - expected_p0
        
        # Actual results (1 for win, 0 for loss)
        actual_p0 = torch.tensor(1.0 if player0_win else 0.0, device=self.device)
        actual_p1 = 1.0 - actual_p0
        
        # Update Elo ratings
        self.player0_elo += k_factor * (actual_p0.item() - expected_p0.item())
        self.player1_elo += k_factor * (actual_p1.item() - expected_p1.item())

    def _init_session_metrics(self):
        """Initialize all session metrics."""
        self.session_metrics = {
            'session_win_rates': [],  # Win rate per session
            'session_total_losses': [],  # Average total loss per session
            'session_policy_losses': [],  # Average policy loss per session
            'session_value_losses': [],  # Average value loss per session
            'session_win_counts': [],  # Number of wins per session
            'session_game_counts': [],  # Number of games per session
            'session_timestamps': [],  # Timestamps for each session
            'session_bb_per_hand': [],  # BB/hand per session
            'session_player0_elo': [],  # Player 0 Elo per session
            'session_player1_elo': [],  # Player 1 Elo per session
            'session_hands_played': [],  # Number of hands played per session
            'session_durations': [],  # Duration of each session
            'total_player0_wins': 0,  # Total wins for Player 0
            'total_player1_wins': 0,  # Total wins for Player 1
            'total_hands_played': 0,  # Total hands played
            'total_stack_delta': 0.0  # Total stack delta
        }

    def _update_session_metrics(self, training_stats: dict, session_duration: float):
        """Update session metrics with the latest training statistics."""
        total_games = training_stats['player0_wins'] + training_stats['player1_wins']
        win_rate = training_stats['player0_wins'] / max(1, total_games)
        avg_total_loss = sum(training_stats['total_losses']) / max(1, len(training_stats['total_losses']))
        avg_policy_loss = sum(training_stats['policy_losses']) / max(1, len(training_stats['policy_losses']))
        avg_value_loss = sum(training_stats['value_losses']) / max(1, len(training_stats['value_losses']))
        avg_bb_per_hand = sum(training_stats['bb_per_hand']) / max(1, len(training_stats['bb_per_hand']))
        
        # Update session metrics
        self.session_metrics['session_win_rates'].append(win_rate)
        self.session_metrics['session_total_losses'].append(avg_total_loss)
        self.session_metrics['session_policy_losses'].append(avg_policy_loss)
        self.session_metrics['session_value_losses'].append(avg_value_loss)
        self.session_metrics['session_win_counts'].append(training_stats['player0_wins'])
        self.session_metrics['session_game_counts'].append(total_games)
        self.session_metrics['session_timestamps'].append(time.time())
        self.session_metrics['session_bb_per_hand'].append(avg_bb_per_hand)
        self.session_metrics['session_player0_elo'].append(self.player0_elo)
        self.session_metrics['session_player1_elo'].append(self.player1_elo)
        self.session_metrics['session_hands_played'].append(training_stats['total_hands'])
        self.session_metrics['session_durations'].append(session_duration)
        self.session_metrics['total_player0_wins'] += training_stats['player0_wins']
        self.session_metrics['total_player1_wins'] += training_stats['player1_wins']
        self.session_metrics['total_hands_played'] += training_stats['total_hands']
        self.session_metrics['total_stack_delta'] += training_stats['total_stack_delta']

    def run_training_session(self, 
                           num_games: int = 1000, 
                           verbose: bool = False,
                           save_interval: int = 100,
                           save_path: Optional[str] = None) -> dict:
        """
        Run a full training session with multiple games.
        
        Args:
            num_games: Number of games to play in this session
            verbose: Whether to print detailed progress information
            save_interval: How often to save checkpoints
            save_path: Directory to save checkpoints
            
        Returns:
            Dict containing training statistics for the session
        """
        session_start_time = time.time()
        training_stats = {
            'player0_wins': 0,
            'player1_wins': 0,
            'total_losses': [],
            'policy_losses': [],
            'value_losses': [],
            'bb_per_hand': [],
            'player0_elo': [],
            'player1_elo': [],
            'total_hands': 0,
            'total_stack_delta': 0.0
        }

        # Initialize Elo ratings if not already set
        if not hasattr(self, 'player0_elo'):
            self.player0_elo = 1000.0  # Changed from 1500 to 1000
        if not hasattr(self, 'player1_elo'):
            self.player1_elo = 1000.0  # Changed from 1500 to 1000
        
        # Use tqdm to create a progress bar for the number of games
        for game_idx in tqdm(range(num_games), desc="Training Games", unit="game"):
            game_result = self.run_game(verbose=verbose, training=True)
            
            winner = game_result['winner']
            if winner == 0:
                training_stats['player0_wins'] += 1
            else:
                training_stats['player1_wins'] += 1
            
            # Update Elo ratings
            self._update_elo(winner == 0)
            training_stats['player0_elo'].append(self.player0_elo)
            training_stats['player1_elo'].append(self.player1_elo)
            
            # Calculate BB/Hand
            num_hands = len([m for m in game_result['metrics'] if m['player_id'] == 0])
            stack_delta = game_result['final_rewards'][0]  # Player 0's profit/loss
            if verbose and (game_idx + 1) % 10 == 0:  # Add debug prints
                print(f"\nDebug - Game {game_idx + 1}:")
                print(f"Stack Delta: {stack_delta:.2f} chips ({stack_delta/self.BB:.2f} BB)")
                print(f"Number of Hands: {num_hands}")

            training_stats['total_hands'] += num_hands
            training_stats['total_stack_delta'] += stack_delta
            bb_per_hand = self._calculate_bb_per_hand(stack_delta, num_hands)
            training_stats['bb_per_hand'].append(bb_per_hand.item())
            
            # Aggregate losses
            for metric in game_result['metrics']:
                if metric['player_id'] == 0:
                    total_loss, policy_loss, value_loss = metric['losses']
                    training_stats['total_losses'].append(total_loss)
                    training_stats['policy_losses'].append(policy_loss)
                    training_stats['value_losses'].append(value_loss)
            
            total_games = training_stats['player0_wins'] + training_stats['player1_wins']
            current_win_rate = training_stats['player0_wins'] / max(1, total_games)
            
            if verbose and (game_idx + 1) % 10 == 0:
                print(f"Game {game_idx + 1}/{num_games}")
                print(f"Player 0 wins: {training_stats['player0_wins']}")
                print(f"Player 1 wins: {training_stats['player1_wins']}")
                print(f"Win rate: {current_win_rate:.2%}")
                print(f"BB/Hand: {sum(training_stats['bb_per_hand'])/len(training_stats['bb_per_hand']):.4f}")
                print(f"Player 0 Elo: {self.player0_elo:.1f}")
                print(f"Player 1 Elo: {self.player1_elo:.1f}")
                if training_stats['total_losses']:
                    print(f"Average total loss: {sum(training_stats['total_losses'][-10:])/10:.4f}")
        
        # Calculate session duration
        session_duration = time.time() - session_start_time
        
        # Update session metrics
        self._update_session_metrics(training_stats, session_duration)
        
        return training_stats

    @abstractmethod
    def load_models_for_training_session(self, save_path: Optional[str] = None):
        """
        Load models for a training session. Implementation depends on the training strategy.
        
        Args:
            save_path: Directory containing saved models
        """
        pass

    def _save_best_models(self, save_path: str, current_win_rate: float, current_bb_per_hand: float, 
                         current_elo: float, best_win_rate: float, best_bb_per_hand: float, 
                         best_elo: float) -> Tuple[float, float, float]:
        """
        Save best models based on different metrics.
        
        Args:
            save_path: Directory to save models
            current_*: Current metrics
            best_*: Best metrics so far
            
        Returns:
            Updated best metrics
        """
        best_models_dir = os.path.join(save_path, 'best_models')
        
        # Update and save best win rate model
        if current_win_rate > best_win_rate:
            best_win_rate = current_win_rate
            # Save model state dict and Elo
            torch.save({
                'model_state_dict': self.Player0.state_dict(),
                'elo': self.player0_elo
            }, os.path.join(best_models_dir, 'best_winrate.pt'))
            print(f"\nSaved new best win rate model: {best_win_rate:.4f} with Elo: {self.player0_elo:.1f}")
        
        # Update and save best BB/hand model
        if current_bb_per_hand > best_bb_per_hand:
            best_bb_per_hand = current_bb_per_hand
            # Save model state dict and Elo
            torch.save({
                'model_state_dict': self.Player0.state_dict(),
                'elo': self.player0_elo
            }, os.path.join(best_models_dir, 'best_bbhand.pt'))
            print(f"\nSaved new best BB/hand model: {best_bb_per_hand:.4f} with Elo: {self.player0_elo:.1f}")
        
        # Update and save best Elo model
        if current_elo > best_elo:
            best_elo = current_elo
            # Save model state dict and Elo
            torch.save({
                'model_state_dict': self.Player0.state_dict(),
                'elo': self.player0_elo
            }, os.path.join(best_models_dir, 'best_elo.pt'))
            print(f"\nSaved new best Elo model: {best_elo:.1f}")
            
        return best_win_rate, best_bb_per_hand, best_elo

    def train_sessions(self,
                      num_sessions: int = 10,
                      games_per_session: int = 1000,
                      save_path: Optional[str] = None,
                      plot_path: Optional[str] = None,
                      verbose: bool = False) -> Dict:
        """Train the model over multiple sessions with k-best self play."""
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            best_models_dir = os.path.join(save_path, 'best_models')
            os.makedirs(best_models_dir, exist_ok=True)
        
        best_win_rate = 0.5
        best_bb_per_hand = 0.0
        best_elo = 1000.0
        
        for session in range(num_sessions):
            session_start_time = time.time()
            print(f"\n========= Starting Training Session {session + 1}/{num_sessions} =========")
            
            # Every 5 sessions, load new models
            if session > 0 and session % 5 == 0:
                self.load_models_for_training_session(save_path)
            
            # Run training session
            session_stats = self.run_training_session(
                num_games=games_per_session,
                verbose=verbose,
                save_interval=games_per_session,
                save_path=save_path
            )
            
            # Get current metrics
            current_win_rate = session_stats['win_rate']
            current_bb_per_hand = session_stats['bb_per_hand']
            current_elo = session_stats['player0_elo']
            
            # Update and save best models if current metrics are better
            if save_path:
                best_win_rate, best_bb_per_hand, best_elo = self._save_best_models(
                    save_path, current_win_rate, current_bb_per_hand, current_elo,
                    best_win_rate, best_bb_per_hand, best_elo
                )
            
            # Print session summary
            print(f"Average loss: {sum(session_stats['total_losses'])/len(session_stats['total_losses']):.4f}")
            print(f"\nSession {session + 1} completed in {time.time() - session_start_time:.1f}s")
            print(f"Win rate: {current_win_rate:.3f}")
            print(f"BB/hand: {current_bb_per_hand:.3f}")
            print(f"Elo rating: {current_elo:.1f}")
            
            # Save session metrics
            if save_path:
                metrics_path = os.path.join(save_path, f'session_{session + 1}_metrics.pt')
                torch.save(self.session_metrics, metrics_path)
            
            # Plot training progress
            if plot_path:
                os.makedirs(plot_path, exist_ok=True)
                Utilities.plot_all_stats(
                    self.session_metrics,
                    os.path.join(plot_path, f'session_{session + 1}'),
                    show=False
                )
        
        # Print final training summary
        print("\n========= Training Complete =========")
        print(f"Total Sessions: {num_sessions}")
        print(f"Total Games: {sum(self.session_metrics['session_game_counts'])}")
        print(f"Best Win Rate: {best_win_rate:.2%}")
        print(f"Best BB/Hand: {best_bb_per_hand:.4f}")
        print(f"Best Elo: {best_elo:.1f}")
        print(f"Final Player 0 Elo: {self.player0_elo:.1f}")
        print(f"Final Player 1 Elo: {self.player1_elo:.1f}")
        
        return self.session_metrics

    def train_for_duration(self, 
                           duration: timedelta, 
                           games_per_session: int = 50, 
                           save_path: str = None, 
                           plot_path: str = None,
                           verbose: bool = False):
        """
        Train the model for a specified duration with k-best self play.
        
        Args:
            duration (timedelta): Duration to train
            games_per_session (int): Number of games to play per session
            save_path (str): Directory to save models and checkpoints
            plot_path (str): Directory to save plots
            verbose (bool): Whether to print detailed progress
        """
        start_time = datetime.now()
        end_time = start_time + duration
        session = 0
        best_win_rate = 0.5
        best_bb_per_hand = 0.0
        best_elo = 1000.0
        
        # Create directories if they don't exist
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            best_models_dir = os.path.join(save_path, 'best_models')
            os.makedirs(best_models_dir, exist_ok=True)
        
        while datetime.now() < end_time:
            session_start_time = time.time()
            session += 1
            print(f"\n========= Starting Training Session {session} =========")
            print(f"Time remaining: {end_time - datetime.now()}")
            
            # Every 5 sessions, load new models
            if session > 1 and session % 5 == 0:
                self.load_models_for_training_session(save_path)
            
            # Run training session
            session_stats = self.run_training_session(
                num_games=games_per_session,
                verbose=verbose,
                save_interval=games_per_session,
                save_path=save_path
            )
            
            # Calculate current metrics from session_stats
            total_games = session_stats['player0_wins'] + session_stats['player1_wins']
            current_win_rate = session_stats['player0_wins'] / max(1, total_games)
            current_bb_per_hand = sum(session_stats['bb_per_hand']) / max(1, len(session_stats['bb_per_hand']))
            current_elo = self.player0_elo  # Use the current Elo rating
            
            # Update and save best models if current metrics are better
            if save_path:
                best_win_rate, best_bb_per_hand, best_elo = self._save_best_models(
                    save_path, current_win_rate, current_bb_per_hand, current_elo,
                    best_win_rate, best_bb_per_hand, best_elo
                )
            
            # Print session summary
            print(f"\nSession {session} completed in {time.time() - session_start_time:.1f}s")
            print(f"Win rate: {100*current_win_rate:.1f}%")
            print(f"BB/hand: {current_bb_per_hand:.3f}")
            print(f"Elo rating: {current_elo:.1f}")
            
            # Check if we should continue training
            if datetime.now() >= end_time:
                print("\nTraining duration completed")
                break
            
            # Optional: Add a small delay between sessions
            time.sleep(1)

        if plot_path is not None:
            os.makedirs(plot_path, exist_ok=True)
            Utilities.plot_all_stats(
                self.session_metrics,
                os.path.join(plot_path, f'session_{session + 1}'),
                show=False
            )
        
        # Print final best metrics
        print("\nTraining completed!")
        print(f"Best metrics achieved:")
        print(f"Win Rate: {best_win_rate:.4f}")
        print(f"BB/Hand: {best_bb_per_hand:.4f}")
        print(f"Elo: {best_elo:.1f}")
        
        return self.session_metrics

# ==================================================
