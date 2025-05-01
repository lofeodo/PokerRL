from utilities import Utilities
import time
import pokerkit as pk
import torch
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
import os
from BayesianCfr import BayesianHoldemWithCFR
from BayesianHoldem import BayesianHoldem

# ==================================================

class SelfPlayPokerGame():
    def __init__(self, learning_rate: float = 0.001, best_model_path: Optional[str] = None, 
                 cfr_blueprint_path: Optional[str] = None, use_cfr: bool = True):
        self.use_cfr = use_cfr
        
        # Initialize base models
        base_model0 = BayesianHoldem(learning_rate=learning_rate)
        base_model1 = BayesianHoldem(learning_rate=learning_rate)
        
        if use_cfr and cfr_blueprint_path is not None:
            # Create hybrid models with CFR guidance
            self.Player0 = BayesianHoldemWithCFR(base_model0, cfr_blueprint_path, cfr_weight=0.5)
            self.Player1 = BayesianHoldemWithCFR(base_model1, cfr_blueprint_path, cfr_weight=0.5)
            print(f"Using hybrid NN+CFR models with blueprint from {cfr_blueprint_path}")
        # else:
        #     # Use standard BayesianHoldem models
        #     self.Player0 = base_model0
        #     self.Player1 = base_model1
        #     print("Using standard BayesianHoldem models")
        
        # Load best model for Player1 if available
        if best_model_path and os.path.exists(best_model_path):
            if use_cfr:
                self.Player1.base_model.load_state_dict(torch.load(best_model_path))
            else:
                self.Player1.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        
        self.BB = 100
        
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.best_win_rate = 0.0
        self.best_model_path = best_model_path
        
        # Session tracking metrics
        self.session_metrics = {
            'session_win_rates': [],  # Win rate per session
            'session_total_losses': [],  # Average total loss per session
            'session_policy_losses': [],  # Average policy loss per session
            'session_value_losses': [],  # Average value loss per session
            'session_win_counts': [],  # Number of wins per session
            'session_game_counts': [],  # Number of games per session
            'session_timestamps': []  # Timestamps for each session
        }

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

    def _update_representations(self):
        # TODO: Implement this when Jie has completed the representation functions
        pass

    # ==================================================

    def _play_action(self, player_id: int, action: int, verbose: bool = False) -> Dict:
        """Execute an action and return transition information."""
        other_player_id = abs(1 - player_id)

        # Store pre-action state
        pre_state = {
            'bets': self.state.bets.copy(),
            'stacks': self.state.stacks.copy(),
            'pot_size': tuple(self.state.pot_amounts)[0] if len(tuple(self.state.pot_amounts)) > 0 else 0,
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
            'pot_size': tuple(self.state.pot_amounts)[0] if len(tuple(self.state.pot_amounts)) > 0 else 0,
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
            
        pot_size = float(tuple(self.state.pot_amounts)[0]) if len(tuple(self.state.pot_amounts)) > 0 else 0
        stack_size = float(self.state.stacks[player_id])
        bet_size = float(self.BB)
        
        if transition['is_terminal']:
            actual_return = transition['post_state']['stacks'][player_id] - transition['pre_state']['stacks'][player_id]
        else:
            immediate_reward = transition['state_changes']['stack_delta']
            actual_return = immediate_reward
        
        if self.use_cfr:
            # Use the CFR-guided training method if enabled
            return self.Player0.train_step(
                action_representation=action_representation,
                card_representation=card_representation,
                pot_size=pot_size,
                stack_size=stack_size,
                bet_size=bet_size,
                actual_return=actual_return,
                pokerkit_state=self.state,  # Pass the pokerkit state for CFR
                player_id=player_id
            )
        else:
            # Use the standard training method
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
            
        while self.state.street_index is not None:
            if verbose:
                print(f"Street index: {self.state.street_index}")
                print(f"Public cards: {self.state.board_cards}")
                print(f"Bets: P0: {self.state.bets[0]}, P1: {self.state.bets[1]}, Chips: P0: {self.state.stacks[0]}, P1: {self.state.stacks[1]}")
            
            # Current player's turn
            player_id = self.state.actor_index
            player = self.Player0 if player_id == 0 else self.Player1

            # Get state representations
            # TODO: Replace with actual representation
            # action_representation = InputRepresentations.get_action_representation(self.state, player_id)
            action_representation = torch.randn(24, 4, 4)
            # card_representation = InputRepresentations.get_card_representation(self.state, player_id)
            card_representation = torch.randn(6, 13, 4)
            
            # Get action from player
            if self.use_cfr:
                # Use predict_action from base model or hybrid model depending on implementation
                action = player.predict_action(action_representation, card_representation)
            else:
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
            
            # Check if the round ended due to a fold or all-in
            if transition['is_terminal']:
                break
        
        return round_metrics

    # ==================================================

    def run_game(self, verbose: bool = False, training: bool = True) -> Dict:
        """Run a complete game with optional training."""
        self._update_state(100 * self.BB, 100 * self.BB)
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
                    
                    # Calculate final rewards
                    winner = 1 - idx  # The player who still has chips
                    loser = idx
                    final_rewards = {
                        winner: float(self.state.stacks[winner]),
                        loser: -float(self.BB * 100)  # Lost the initial stack
                    }
                    
                    return {
                        'metrics': game_metrics,
                        'final_rewards': final_rewards,
                        'winner': winner
                    }

            self._update_state(self.state.stacks[0], self.state.stacks[1])

    # ==================================================

    def run_training_session(self, 
                           num_games: int = 1000, 
                           verbose: bool = False,
                           save_interval: int = 100,
                           save_path: Optional[str] = None) -> dict:
        """Run a full training session with multiple games."""
        training_stats = {
            'player0_wins': 0,
            'player1_wins': 0,
            'total_losses': [],
            'policy_losses': [],
            'value_losses': []
        }

        if save_path and os.path.exists(f"{save_path}/best.pt"):
            best_model_path = f"{save_path}/best.pt"
            self.set_models_to_previous_best(best_model_path)
        
        for game_idx in range(num_games):
            # Run a game and collect metrics
            if verbose:
                print(f"========= New game {game_idx + 1} =========")
            game_result = self.run_game(verbose=verbose, training=True)
            
            # Update statistics
            winner = game_result['winner']
            if winner == 0:
                training_stats['player0_wins'] += 1
            else:
                training_stats['player1_wins'] += 1
            
            # Aggregate losses
            for metric in game_result['metrics']:
                if metric['player_id'] == 0:  # Only track losses for Player0
                    total_loss, policy_loss, value_loss = metric['losses']
                    training_stats['total_losses'].append(total_loss)
                    training_stats['policy_losses'].append(policy_loss)
                    training_stats['value_losses'].append(value_loss)
            
            # Calculate current win rate
            total_games = training_stats['player0_wins'] + training_stats['player1_wins']
            current_win_rate = training_stats['player0_wins'] / max(1, total_games)
            
            # Save checkpoints periodically
            if save_path and (game_idx + 1) % save_interval == 0:
                self._save_checkpoint(save_path, game_idx + 1, training_stats)
                
                # Update best model if win rate has improved
                if current_win_rate > self.best_win_rate:
                    self.best_win_rate = current_win_rate
                    if self.use_cfr:
                        torch.save(self.Player0.base_model.state_dict(), f"{save_path}/best.pt")
                    else:
                        torch.save(self.Player0.state_dict(), f"{save_path}/best.pt")
                    print(f"\nNew best model saved with win rate: {self.best_win_rate:.2%}")
                
            # Print progress
            if (game_idx + 1) % 10 == 0:
                print(f"Game {game_idx + 1}/{num_games} - Win rate: {current_win_rate:.2%}")
                if training_stats['total_losses']:
                    print(f"Average loss: {sum(training_stats['total_losses'][-10:])/min(10, len(training_stats['total_losses'])):.4f}")
        
        return training_stats

    # ==================================================

    def set_models_to_previous_best(self, best_model_path: str, verbose: bool = False):
        """Load the best model for both players."""
        if os.path.exists(best_model_path):
            if self.use_cfr:
                self.Player0.base_model.load_state_dict(torch.load(best_model_path))
                self.Player1.base_model.load_state_dict(torch.load(best_model_path))
            else:
                self.Player0.load_state_dict(torch.load(best_model_path))
                self.Player1.load_state_dict(torch.load(best_model_path))
            if verbose:
                print(f"Loaded best model from {best_model_path}")
        else:
            if verbose:
                print(f"No best model found at {best_model_path}")
    
    # ==================================================

    def _save_checkpoint(self, save_path: str, game_idx: int, stats: dict):
        """Save model checkpoints and training statistics."""
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        if self.use_cfr:
            model_state = self.Player0.base_model.state_dict()
        else:
            model_state = self.Player0.state_dict()
            
        # Create checkpoint
        checkpoint = {
            'model_state': model_state,
            'game_idx': game_idx,
            'stats': stats,
            'best_win_rate': self.best_win_rate
        }
        
        # Save checkpoint
        torch.save(checkpoint, f"{save_path}/checkpoint_game_{game_idx}.pt")
        print(f"Saved checkpoint at game {game_idx}")

    def train(self,
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
        
        best_win_rate = 0.0  # Track the best win rate across all sessions
        
        for session in range(num_sessions):
            session_start_time = time.time()
            print(f"\n========= Starting Training Session {session + 1}/{num_sessions} =========")
            
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
            print(f"\nSession {session + 1} Summary:")
            print(f"Win Rate: {session_win_rate:.2%}")
            print(f"Games Played: {total_games}")
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
                if self.use_cfr:
                    torch.save(self.Player0.base_model.state_dict(), os.path.join(save_path, 'best_model.pt'))
                else:
                    torch.save(self.Player0.state_dict(), os.path.join(save_path, 'best_model.pt'))
                print(f"\nNew best model saved with win rate: {best_win_rate:.2%}")
        
        # Print final training summary
        print("\n========= Training Complete =========")
        print(f"Total Sessions: {num_sessions}")
        print(f"Total Games: {sum(self.session_metrics['session_game_counts'])}")
        print(f"Best Win Rate: {best_win_rate:.2%}")
        print(f"Average Session Time: {sum(self.session_metrics['session_timestamps'])/num_sessions:.2f} seconds")
        
        return self.session_metrics

# ==================================================