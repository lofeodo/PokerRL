import pokerkit as pk
from BayesianHoldem import BayesianHoldem

class SelfPlayPokerGame():
    def __init__(self):
        self.Player0 = BayesianHoldem()
        self.Player1 = BayesianHoldem()
        self.BB = 100

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

        return game(raw_starting_stacks=[player0_stack, player1_stack], player_count=player_count)

    def _update_representations(self):
        # TODO: Implement this when Jie has completed the representation functions
        pass

    def _play_action(self, player_id: int, action: int):
        if action == 0: # fold
            self.state.fold()
        elif action == 1: # check/call
            self.state.check_or_call()
        elif action == 2: # bet/raise BB
            self.state.complete_bet_or_raise_to(self.state.bets[player_id] + 
                                                self.BB)
        elif action == 3: # all-in
            self.state.complete_bet_or_raise_to(self.state.stacks[player_id] + 
                                                self.state.stacks[player_id])

        self.update_representations()

    def run_round(self):
        while self.state.street_index is not None:
            if self.state.actor_index == 0:
                action = self.Player0.predict_action(self.action_representation, self.card_representation)
                self._play_action(0, action)
            else:
                action = self.Player1.predict_action(self.action_representation, self.card_representation)
                self._play_action(1, action)

    def run_game(self):
        self.state = self._update_state(100 * self.BB, 100 * self.BB)
        gameover = False

        while not gameover:
            self.run_round()

            # Check if someone lost all their chips
            for idx, stack in enumerate(self.state.stacks):
                if stack == 0:
                    print(f"Player {idx} has lost all their chips.")
                    gameover = True
                    break

            self.state = self._update_state(self.state.stacks[0], 
                                            self.state.stacks[1])
            

        print(f"Final stacks: {self.state.stacks}")

        
        
