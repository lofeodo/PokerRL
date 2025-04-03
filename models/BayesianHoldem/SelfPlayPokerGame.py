import pokerkit as pk
from BayesianHoldem import BayesianHoldem
import torch

# ==================================================

class SelfPlayPokerGame():
    def __init__(self):
        self.Player0 = BayesianHoldem()
        self.Player1 = BayesianHoldem()
        self.BB = 100

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

    def _is_all_in(self, player_id: int):
        return self.state.bets[player_id] + self.state.stacks[player_id] == self.state.stacks[player_id]

    # ==================================================

    def _play_action(self, player_id: int, action: int, verbose: bool = False):
        other_player_id = abs(1 - player_id)

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
                    #self.state.complete_bet_or_raise_to(player_stack)  # Go all-in
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
                    #self.state.complete_bet_or_raise_to(self.state.bets[player_id] + player_stack)
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

        # TODO: Implement this when Jie has completed the representation functions
        #self.update_representations()

    # ==================================================

    def run_round(self, verbose: bool = False):
        if verbose:
            print("======= New Round =======")
            print(f"Player 0 cards: {self.state.hole_cards[0]}, chips: {self.state.stacks[0]}")
            print(f"Player 1 cards: {self.state.hole_cards[1]}, chips: {self.state.stacks[1]}")
        while self.state.street_index is not None:
            # TODO: Implement this when Jie has completed the representation functions
            if verbose:
                print(f"Street index: {self.state.street_index}")
                print(f"Public cards: {self.state.board_cards}")
                print(f"Bets: P0: {self.state.bets[0]}, P1: {self.state.bets[1]}, Chips: P0: {self.state.stacks[0]}, P1: {self.state.stacks[1]}")

            action_representation = torch.randn(24,4,4)
            card_representation = torch.randn(6,13,4)
            if self.state.actor_index == 0:
                action = self.Player0.predict_action(action_representation, card_representation)
                self._play_action(0, action, verbose)
            else:
                action = self.Player1.predict_action(action_representation, card_representation)
                self._play_action(1, action, verbose)

    # ==================================================

    def run_game(self, verbose: bool = False):
        self._update_state(100 * self.BB, 100 * self.BB)

        while True:
            self.run_round(verbose)

            # Check if someone lost all their chips
            for idx, stack in enumerate(self.state.stacks):
                if stack <= 0:
                    if verbose:
                        print(f"Player {idx} has lost all their chips.")
                        print(f"Final stacks: {self.state.stacks}")
                        return

            self._update_state(self.state.stacks[0], 
                               self.state.stacks[1])
            
    # ==================================================

# ==================================================
