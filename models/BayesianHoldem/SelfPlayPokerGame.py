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
                     action_representation: torch.Tensor,
                     card_representation: torch.Tensor,
                     transition: dict) -> Tuple[float, float, float]:
        """Train Player0 based on their action and the current game state."""
        if player_id != 0:  # Only train Player0
            return 0.0, 0.0, 0.0
            
        pot_size = self.state.total_pot_amount
        stack_size = float(self.state.stacks[player_id])
        bet_size = float(self.BB)
        max_stack = 20000  # Match the default max_stack in BayesianHoldem
        
        # Calculate normalized return
        if transition['is_terminal']:
            stack_delta = transition['post_state']['stacks'][player_id] - transition['pre_state']['stacks'][player_id]
            actual_return = stack_delta / max_stack  # Normalize by max_stack
        else:
            stack_delta = transition['state_changes']['stack_delta']
            # For non-terminal states, also include potential future value from pot
            pot_contribution = 0.5 * (pot_size / max_stack)  # Expected value from pot
            actual_return = (stack_delta / max_stack) + pot_contribution
        
        # Ensure tensors are on the correct device
        action_representation = action_representation.to(self.device)
        card_representation = card_representation.to(self.device)
        
        return self.Player0.train_step(
            action_representation=action_representation,
            card_representation=card_representation,
            pot_size=pot_size,
            stack_size=stack_size,
            bet_size=bet_size,
            actual_return=actual_return
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
            
            # Train if enabled
            if training:
                losses = self._train_player(
                    player_id,
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
            self.player0_elo = 1500.0
        if not hasattr(self, 'player1_elo'):
            self.player1_elo = 1500.0

        best_model_path = f"{save_path}/best.pt"
        self.load_models_for_training_session(best_model_path)
        
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
            
            if save_path and (game_idx + 1) % save_interval == 0:
                self._save_checkpoint(save_path, game_idx + 1, training_stats)
                
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
        
        print(f"Total stack delta: {training_stats['total_stack_delta']}, total hands: {training_stats['total_hands']}")
        overall_bb_per_hand = self._calculate_bb_per_hand(
            training_stats['total_stack_delta'],
            training_stats['total_hands']
        )
        
        print("\n========= Performance Metrics =========")
        print(f"Overall BB/Hand: {overall_bb_per_hand:.4f}")
        print(f"Final Player 0 Elo: {self.player0_elo:.1f}")
        print(f"Final Player 1 Elo: {self.player1_elo:.1f}")
        print(f"Total hands played: {training_stats['total_hands']}")
        
        return training_stats

    # ==================================================

    @abstractmethod
    def load_models_for_training_session(self, best_model_path: str, verbose: bool = False):
        """Load models for a training session. Implementation depends on the training strategy."""
        pass

    # ==================================================

    def _save_checkpoint(self, save_path: str, game_idx: int, stats: dict):
        """Save model checkpoints and training statistics."""
        checkpoint = {
            'player0_state': self.Player0.state_dict(),
            'game_idx': game_idx,
            'stats': stats,
            'best_win_rate': self.best_win_rate,
            'player0_elo': self.player0_elo,
            'player1_elo': self.player1_elo
        }
        
        torch.save(checkpoint, f"{save_path}/game_{game_idx}.pt")

    def train_sessions(self,
                      num_sessions: int = 10,
                      games_per_session: int = 1000,
                      save_path: Optional[str] = None,
                      plot_path: Optional[str] = None,
                      verbose: bool = False) -> Dict:
        """
        Train the model over multiple sessions.
        
        Args:
            num_sessions: Number of training sessions to run
            games_per_session: Number of games per training session
            save_path: Directory to save models and checkpoints
            plot_path: Directory to save training plots
            verbose: Whether to print progress information
            
        Returns:
            Dict containing training statistics across all sessions
        """
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        best_win_rate = 0.0
        total_bb_per_hand = []
        
        for session in range(num_sessions):
            session_start_time = time.time()
            print(f"\n========= Starting Training Session {session + 1}/{num_sessions} =========")
            
            # Run training session
            session_stats = self.run_training_session(
                num_games=games_per_session,
                verbose=verbose,
                save_interval=games_per_session // 10,
                save_path=save_path
            )
            
            # Update session metrics
            total_games = session_stats['player0_wins'] + session_stats['player1_wins']
            session_win_rate = session_stats['player0_wins'] / max(1, total_games)
            
            self.session_metrics['session_win_rates'].append(session_win_rate)
            self.session_metrics['session_win_counts'].append(session_stats['player0_wins'])
            self.session_metrics['session_game_counts'].append(total_games)
            self.session_metrics['session_timestamps'].append(time.time() - session_start_time)
            
            # Track BB/Hand for the session
            session_bb_per_hand = self._calculate_bb_per_hand(
                session_stats['total_stack_delta'],
                session_stats['total_hands']
            )
            total_bb_per_hand.append(session_bb_per_hand.item())
            
            # Calculate average losses for the session
            if session_stats['total_losses']:
                self.session_metrics['session_total_losses'].append(
                    sum(session_stats['total_losses']) / len(session_stats['total_losses'])
                )
                self.session_metrics['session_policy_losses'].append(
                    sum(session_stats['policy_losses']) / len(session_stats['policy_losses'])
                )
                self.session_metrics['session_value_losses'].append(
                    sum(session_stats['value_losses']) / len(session_stats['value_losses'])
                )
            
            # Print session summary
            print(f"\nSession {session + 1} Summary:")
            print(f"Win Rate: {session_win_rate:.2%}")
            print(f"Games Played: {total_games}")
            print(f"BB/Hand: {session_bb_per_hand.item():.4f}")
            print(f"Player 0 Elo: {self.player0_elo:.1f}")
            print(f"Player 1 Elo: {self.player1_elo:.1f}")
            print(f"Time Taken: {self.session_metrics['session_timestamps'][-1]:.2f} seconds")
            if self.session_metrics['session_total_losses']:
                print(f"Average Total Loss: {self.session_metrics['session_total_losses'][-1]:.4f}")
            
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
            
            # Save the model if the current session's win rate is better than the best win rate
            if session_win_rate > best_win_rate and save_path:
                best_win_rate = session_win_rate
                torch.save(self.Player0.state_dict(), os.path.join(save_path, 'best_model.pt'))
                print(f"\nNew best model saved with win rate: {best_win_rate:.2%}")
        
        # Print final training summary with new metrics
        print("\n========= Training Complete =========")
        print(f"Total Sessions: {num_sessions}")
        print(f"Total Games: {sum(self.session_metrics['session_game_counts'])}")
        print(f"Best Win Rate: {best_win_rate:.2%}")
        print(f"Average BB/Hand: {sum(total_bb_per_hand)/len(total_bb_per_hand):.4f}")
        print(f"Final Player 0 Elo: {self.player0_elo:.1f}")
        print(f"Final Player 1 Elo: {self.player1_elo:.1f}")
        print(f"Average Session Time: {sum(self.session_metrics['session_timestamps'])/num_sessions:.2f} seconds")
        
        return self.session_metrics

    def train_for_duration(self,
                          duration: timedelta,
                          games_per_session: int = 50,
                          save_path: Optional[str] = None,
                          plot_path: Optional[str] = None,
                          verbose: bool = False) -> Dict:
        """
        Train the model for a specified duration.
        
        Args:
            duration: How long to train for (e.g., timedelta(hours=48))
            games_per_session: Number of games per training session
            save_path: Directory to save models and checkpoints
            plot_path: Directory to save training plots
            verbose: Whether to print progress information
            
        Returns:
            Dict containing training statistics across all sessions
        """
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        best_win_rate = 0.0
        start_time = datetime.now()
        end_time = start_time + duration
        session = 0
        
        while datetime.now() < end_time:
            session_start_time = time.time()
            session += 1
            print(f"\n========= Starting Training Session {session} =========")
            print(f"Time remaining: {end_time - datetime.now()}")
            
            # Run training session
            session_stats = self.run_training_session(
                num_games=games_per_session,
                verbose=verbose,
                save_interval=games_per_session // 10,  # Save 10 checkpoints per session
                save_path=save_path
            )
            
            # Update session metrics
            total_games = session_stats['player0_wins'] + session_stats['player1_wins']
            session_win_rate = session_stats['player0_wins'] / max(1, total_games)
            
            self.session_metrics['session_win_rates'].append(session_win_rate)
            self.session_metrics['session_win_counts'].append(session_stats['player0_wins'])
            self.session_metrics['session_game_counts'].append(total_games)
            self.session_metrics['session_timestamps'].append(time.time() - session_start_time)
            
            # Calculate average losses for the session
            if session_stats['total_losses']:
                self.session_metrics['session_total_losses'].append(
                    sum(session_stats['total_losses']) / len(session_stats['total_losses'])
                )
                self.session_metrics['session_policy_losses'].append(
                    sum(session_stats['policy_losses']) / len(session_stats['policy_losses'])
                )
                self.session_metrics['session_value_losses'].append(
                    sum(session_stats['value_losses']) / len(session_stats['value_losses'])
                )
            
            # Print session summary
            print(f"\nSession {session} Summary:")
            print(f"Win Rate: {session_win_rate:.2%}")
            print(f"Games Played: {total_games}")
            print(f"Time Taken: {self.session_metrics['session_timestamps'][-1]:.2f} seconds")
            if self.session_metrics['session_total_losses']:
                print(f"Average Total Loss: {self.session_metrics['session_total_losses'][-1]:.4f}")
            
            # Save session metrics
            if save_path:
                metrics_path = os.path.join(save_path, f'session_{session}_metrics.pt')
                torch.save(self.session_metrics, metrics_path)
            
            # Plot training progress
            if plot_path:
                os.makedirs(plot_path, exist_ok=True)
                Utilities.plot_all_stats(
                    self.session_metrics,
                    os.path.join(plot_path, f'session_{session}'),
                    show=False
                )
            
            # Save the model if the current session's win rate is better than the best win rate
            if session_win_rate > best_win_rate and save_path:
                best_win_rate = session_win_rate
                torch.save(self.Player0.state_dict(), os.path.join(save_path, 'best_model.pt'))
                print(f"\nNew best model saved with win rate: {best_win_rate:.2%}")
        
        # Print final training summary
        print("\n========= Training Complete =========")
        print(f"Total Sessions: {session}")
        print(f"Total Games: {sum(self.session_metrics['session_game_counts'])}")
        print(f"Best Win Rate: {best_win_rate:.2%}")
        print(f"Total Training Time: {datetime.now() - start_time}")
        print(f"Average Session Time: {sum(self.session_metrics['session_timestamps'])/session:.2f} seconds")
        
        return self.session_metrics

# ==================================================
