import pokerkit as pk
from BayesianHoldem import BayesianHoldem
import torch
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
import os
#from InputRepresentations import InputRepresentations

# ==================================================

class SelfPlayPokerGame():
    def __init__(self, learning_rate: float = 0.001, best_model_path: Optional[str] = None):
        self.Player0 = BayesianHoldem()  # This player will be trained
        self.Player1 = BayesianHoldem()  # This player will use the best previous version
        
        # Load best model for Player1 if available
        if best_model_path and os.path.exists(best_model_path):
            self.Player1.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        
        self.BB = 100
        
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.best_win_rate = 0.0
        self.best_model_path = best_model_path

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
                    winner = idx
                    loser = 1 - idx
                    final_rewards = {
                        winner: float(self.state.stacks[winner]),
                        loser: -float(self.state.stacks[winner])
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

        best_model_path = f"{save_path}/best.pt"
        self.set_models_to_previous_best(best_model_path)
        
        for game_idx in range(num_games):
            # Run a game and collect metrics
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
            
            # Save best model if win rate improved
            if current_win_rate > self.best_win_rate and save_path:
                self.best_win_rate = current_win_rate
                torch.save(self.Player0.state_dict(), best_model_path)
                print(f"\nNew best model saved with win rate: {current_win_rate:.2%}")
            
            # Save checkpoints periodically
            if save_path and (game_idx + 1) % save_interval == 0:
                self._save_checkpoint(save_path, game_idx + 1, training_stats)
                
            # Print progress
            if verbose and (game_idx + 1) % 10 == 0:
                print(f"Game {game_idx + 1}/{num_games}")
                print(f"Player 0 wins: {training_stats['player0_wins']}")
                print(f"Player 1 wins: {training_stats['player1_wins']}")
                print(f"Win rate: {current_win_rate:.2%}")
                if training_stats['total_losses']:
                    print(f"Average total loss: {sum(training_stats['total_losses'][-10:])/10:.4f}")
        
        return training_stats

    # ==================================================

    def set_models_to_previous_best(self, best_model_path: str, verbose: bool = False):
        if os.path.exists(best_model_path):
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
        checkpoint = {
            'player0_state': self.Player0.state_dict(),
            'game_idx': game_idx,
            'stats': stats,
            'best_win_rate': self.best_win_rate
        }
        
        torch.save(checkpoint, f"{save_path}/game_{game_idx}.pt")

# ==================================================
