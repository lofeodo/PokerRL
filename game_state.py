import numpy as np
import pokerkit as pk

class HUNLPoker:
    def __init__(self):
        self.big_blind = 100
        self.small_blind = 0.5 * self.big_blind
        self.blinds = [self.small_blind, self.big_blind]
        self.ante = 0.1 * self.big_blind
        self.min_bet = 0.5 * self.big_blind
        self.starting_stack = 100 * self.big_blind
        self.player_count = 2
        self.board_idx = 0
        self.hand_type_idx = 0  # pk.hand_types.StdHighHand

        self.game = pk.NoLimitTexasHoldem(
            automations=(
                pk.Automation.ANTE_POSTING,
                pk.Automation.BET_COLLECTION,
                pk.Automation.BLIND_OR_STRADDLE_POSTING,
                pk.Automation.CARD_BURNING,
                pk.Automation.HOLE_DEALING,
                pk.Automation.BOARD_DEALING,
                pk.Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                pk.Automation.HAND_KILLING,
                pk.Automation.CHIPS_PUSHING,
                pk.Automation.CHIPS_PULLING
            ),
            ante_trimming_status=True,
            raw_antes=self.ante,
            raw_blinds_or_straddles=self.blinds,
            min_bet=self.min_bet
        )

        # Start the initial state with 2 players
        self.state = self.game(
            raw_starting_stacks=self.starting_stack,
            player_count=self.player_count
        )

        # Example card tensor (6x4x13)
        self.tensor = np.zeros((6, 4, 13), dtype=np.int8)

        # Example action tensor (24x4x4)
        self.action_tensor = np.zeros((24, 4, 4), dtype=np.int8)
        # One counter for each of the four rounds (preflop=0, flop=1, turn=2, river=3)
        self.round_counters = [0, 0, 0, 0]
        self.action_counter = 0

        # Initialize for player 0
        self.update_tensor_for_player(0)

    def get_legal_actions(self, player_id: int):
        """
        Returns a subset of ['fold', 'call', 'raise_bb', 'allin'] as strings
        in line with your action map = {'fold': 0, 'call': 1, 'raise_bb': 2, 'allin': 3}.

        This is just an example approach for heads-up:
          - fold: only if there's an actual bet to fold against
          - call: always an option if we can match the current bet (or check if no bet)
          - raise_bb: only if the other side isn't all-in and we have enough chips
          - allin: if we have any stack left
        """
        legal = []

        other_id = 1 - player_id
        current_bet = self.state.bets[other_id]          
        player_stack = self.state.stacks[player_id]     
        player_bet = self.state.bets[player_id]          
        last_action = self.state.operations[-1] if self.state.operations else None

        # (fold) - only if there's a real bet to fold against
        # i.e. skip fold if last action was check/hole deal/board deal
        if not (
            isinstance(last_action, pk.state.CheckingOrCalling)
            or isinstance(last_action, pk.state.HoleDealing)
            or isinstance(last_action, pk.state.BoardDealing)
        ):
            legal.append('fold')

        # (call) - lumps “check” if cost_to_call is 0, or partial all-in if stack < cost_to_call
        legal.append('call')

        # (raise_bb) - if other side is not all-in, we have enough to raise
        other_stack = self.state.stacks[other_id]
        bet_amount = max(player_bet, current_bet)
        raise_amount = bet_amount + self.big_blind
        if other_stack != 0 and (player_bet + player_stack >= raise_amount):
            legal.append('raise_bb')

        # (allin) - if we have any stack
        if player_stack > 0:
            legal.append('allin')

        return legal

    def complete_bet_or_raise_to(self, amount):
        player_id = self.state.actor_index
        legal_actions = self.get_legal_actions(player_id)

        stack = self.state.stacks[player_id]
        if amount >= stack:
            action_type = 'allin'
        elif amount >= self.big_blind:
            action_type = 'raise_bb'
        else:
            action_type = 'call'

        # Check if our chosen 'action_type' is actually in 'legal_actions'
        if action_type not in legal_actions:
            action_type = 'call'

        self.update_action_tensor(action_type, player_id, legal_actions)
        self._apply_action(action_type, player_id)

        if self.state.actor_index is not None:
            self.update_tensor_for_player(self.state.actor_index)

    def check_or_call(self):
        player_id = self.state.actor_index
        if player_id is None:
            return
        legal_actions = self.get_legal_actions(player_id)

        action_type = 'call'

        self.update_action_tensor(action_type, player_id, legal_actions)
        self._apply_action(action_type, player_id)

        if self.state.actor_index is not None:
            self.update_tensor_for_player(self.state.actor_index)

    def fold(self):
        player_id = self.state.actor_index
        legal_actions = self.get_legal_actions(player_id)

        action_type = 'fold' if 'fold' in legal_actions else 'call'

        self.update_action_tensor(action_type, player_id, legal_actions)
        self._apply_action(action_type, player_id)

        if self.state.actor_index is not None:
            self.update_tensor_for_player(self.state.actor_index)

    def _apply_action(self, action_type, player_id):
        other_id = 1 - player_id
        current_bet = self.state.bets[other_id]
        player_stack = self.state.stacks[player_id]

        if action_type == 'fold':
            prev_action = self.state.operations[-1] if self.state.operations else None
            # If last action was check or dealing => skip folding => just check/call
            if isinstance(prev_action, pk.state.CheckingOrCalling) or \
               isinstance(prev_action, pk.state.HoleDealing) or \
               isinstance(prev_action, pk.state.BoardDealing):
                self.state.check_or_call()
            else:
                self.state.fold()

        elif action_type == 'call':
            # partial call if stack < cost_to_call
            self.state.check_or_call()

        elif action_type == 'raise_bb':
            # raise to (max_bet + big_blind)
            bet_amount = max(self.state.bets[player_id], current_bet)
            raise_amount = bet_amount + self.big_blind
            # If not enough for that => partial call
            if (self.state.bets[player_id] + player_stack) < raise_amount:
                self.state.check_or_call()
            else:
                self.state.complete_bet_or_raise_to(raise_amount)

        elif action_type == 'allin':
            # all-in for (player_stack + player_bet), but not exceeding opponent’s total
            all_in_amount = self.state.bets[player_id] + player_stack
            other_total = self.state.bets[other_id] + self.state.stacks[other_id]
            final_all_in = min(all_in_amount, other_total)
            self.state.complete_bet_or_raise_to(final_all_in)

    def update_tensor_for_player(self, player_idx=0):
        """
        Example code that updates a 6x4x13 card tensor for the given player
        """
        self.tensor.fill(0)
        hole_cards = self.get_hole_cards(player_idx)
        raw_board = [item for sublist in self.state.board_cards for item in sublist]
        community_cards = []
        for item in raw_board:
            community_cards.extend(item) if isinstance(item, list) else community_cards.append(item)

        # channel 0 => hole, 1 => flop, 2 => turn, 3 => river, 4 => all com, 5 => all com+hole
        self.add_cards_to_tensor(hole_cards, 0)
        if len(community_cards) >= 3:
            self.add_cards_to_tensor(community_cards[:3], 1)
        if len(community_cards) >= 4:
            self.add_cards_to_tensor([community_cards[3]], 2)
        if len(community_cards) == 5:
            self.add_cards_to_tensor([community_cards[4]], 3)

        self.add_cards_to_tensor(community_cards, 4)
        self.add_cards_to_tensor(hole_cards + community_cards, 5)

    def get_hole_cards(self, player_idx=0):
        if player_idx < len(self.state.hole_cards):
            cards = list(self.state.hole_cards[player_idx])
            return [c for c in cards if isinstance(c, pk.Card)]
        return []

    def add_cards_to_tensor(self, cards, channel):
        for card in cards:
            suit_index = self.get_suit_index(card.suit)
            rank_index = self.get_rank_index(card.rank)
            self.tensor[channel, suit_index, rank_index] = 1

    def get_suit_index(self, s):
        return {'s': 0, 'h': 1, 'd': 2, 'c': 3}[s]

    def get_rank_index(self, r):
        return {'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5,
                '7': 6, '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12}[r]

    def get_card_tensor(self):
        return self.tensor

    # ------------------------------------------------------------------------
    # 24 x 4 x 4 Action Tensor
    # ------------------------------------------------------------------------

    def get_round_idx(self):
        total_cards = sum(len(cards) for cards in self.state.board_cards)
        if total_cards == 0:
            return 0  # preflop
        elif total_cards == 3:
            return 1  # flop
        elif total_cards == 4:
            return 2  # turn
        elif total_cards == 5:
            return 3  # river
        return 3      

    def update_action_tensor(self, action_type, player_idx, legal_actions):
        action_map = {'fold': 0, 'call': 1, 'raise_bb': 2, 'allin': 3}
        col = action_map.get(action_type)
        if col is None:
            return

        round_idx = self.get_round_idx()

        action_in_round = self.round_counters[round_idx]
        if action_in_round >= 6:
            print(f"Warning: Round {round_idx} has 6 actions already.")
            return

        channel_idx = round_idx*6 + action_in_round

        self.action_tensor[channel_idx, player_idx, col] = 1
        self.action_tensor[channel_idx, 2, col] = self.action_tensor[channel_idx, 0:2, col].sum()

        for la in legal_actions:
            c = action_map.get(la)
            if c is not None:
                self.action_tensor[channel_idx, 3, c] = 1

        self.round_counters[round_idx] += 1


    def get_action_tensor(self):
        return self.action_tensor



if __name__ == "__main__":
    hunl = HUNLPoker()

    print("\n--- PRE-FLOP BETTING ROUND ---")
    print(
        f"Player {hunl.state.actor_index}'s turn,\n"
        f"  hole_cards for this player: {hunl.get_hole_cards(hunl.state.actor_index)},\n"
        f"  bets: {hunl.state.bets},\n"
        f"  queue: {hunl.state.actor_indices}"
    )

    hunl.complete_bet_or_raise_to(150)
    print(
        f"Player {hunl.state.actor_index}'s turn,\n"
        f"  hole_cards: {hunl.get_hole_cards(hunl.state.actor_index)},\n"
        f"  bets: {hunl.state.bets},\n"
        f"  queue: {hunl.state.actor_indices}"
    )

    hunl.complete_bet_or_raise_to(200)
    print(
        f"Player {hunl.state.actor_index}'s turn,\n"
        f"  hole_cards: {hunl.get_hole_cards(hunl.state.actor_index)},\n"
        f"  bets: {hunl.state.bets},\n"
        f"  queue: {hunl.state.actor_indices}"
    )

    hunl.check_or_call()
    
    print(f'flop card len: {len([item for sublist in hunl.state.board_cards for item in sublist])}')
    print(f'flop card value: {[item for sublist in hunl.state.board_cards for item in sublist]}')


    print("\n--- FLOP BETTING ROUND ---")
    flop_cards = [item for sublist in hunl.state.board_cards for item in sublist][:3]
    print(f"Flop cards revealed: {flop_cards}")

    # print("Updated flop cards in the tensor for Player 0:")
    # hunl.update_tensor_for_player(0)
    # print(hunl.get_card_tensor()[1])  # Channel 1 is the flop

    # Let's see who's acting
    print(
        f"Player {hunl.state.actor_index}'s turn, "
        f"hand: {hunl.state.get_hand(hunl.state.actor_index, hunl.board_idx, hunl.hand_type_idx)}, "
        f"current bets: {hunl.state.bets}, "
        f"queue: {hunl.state.actor_indices}"
    )

    hunl.complete_bet_or_raise_to(150)

    print(
        f"Player {hunl.state.actor_index}'s turn, "
        f"hand: {hunl.state.get_hand(hunl.state.actor_index, hunl.board_idx, hunl.hand_type_idx)}, "
        f"current bets: {hunl.state.bets}, "
        f"queue: {hunl.state.actor_indices}"
    )

    hunl.complete_bet_or_raise_to(200)


    print(
        f"Player {hunl.state.actor_index}'s turn, "
        f"hand: {hunl.state.get_hand(hunl.state.actor_index, hunl.board_idx, hunl.hand_type_idx)}, "
        f"current bets: {hunl.state.bets}, "
        f"queue: {hunl.state.actor_indices}"
    )
    hunl.check_or_call()

    print("Flop betting complete.\n")


    print("\n--- FOURTH STREET BETTING ROUND ---")
    fourth_street_cards = [item for sublist in hunl.state.board_cards for item in sublist][:3]
    print(f"Fourth street cards revealed: {fourth_street_cards}")

    print("Updated fourth street cards in the tensor for Player 0:")
    # print(hunl.get_card_tensor()[1])  # Channel 1 is the flop

    # Let's see who's acting
    print(
        f"Player {hunl.state.actor_index}'s turn, "
        f"hand: {hunl.state.get_hand(hunl.state.actor_index, hunl.board_idx, hunl.hand_type_idx)}, "
        f"current bets: {hunl.state.bets}, "
        f"queue: {hunl.state.actor_indices}"
    )

    hunl.complete_bet_or_raise_to(150)

    print(
        f"Player {hunl.state.actor_index}'s turn, "
        f"hand: {hunl.state.get_hand(hunl.state.actor_index, hunl.board_idx, hunl.hand_type_idx)}, "
        f"current bets: {hunl.state.bets}, "
        f"queue: {hunl.state.actor_indices}"
    )
    hunl.check_or_call()
    
    print("Fourth Street betting complete.\n")


    print("\n--- FIFTH STREET BETTING ROUND ---")
    fifth_street_cards = [item for sublist in hunl.state.board_cards for item in sublist][:3]
    print(f"Fifth street cards revealed: {fifth_street_cards}")

    print("Updated fifth street cards in the tensor for Player 0:")
    # print(hunl.get_card_tensor()[1])  # Channel 1 is the flop

    # Let's see who's acting
    print(
        f"Player {hunl.state.actor_index}'s turn, "
        f"hand: {hunl.state.get_hand(hunl.state.actor_index, hunl.board_idx, hunl.hand_type_idx)}, "
        f"current bets: {hunl.state.bets}, "
        f"queue: {hunl.state.actor_indices}"
    )

    hunl.complete_bet_or_raise_to(150)

    print(
        f"Player {hunl.state.actor_index}'s turn, "
        f"hand: {hunl.state.get_hand(hunl.state.actor_index, hunl.board_idx, hunl.hand_type_idx)}, "
        f"current bets: {hunl.state.bets}, "
        f"queue: {hunl.state.actor_indices}"
    )
    hunl.check_or_call()
    
    print("Fifth Street betting complete.\n")

    # Show final tensor for demonstration
    card_tensor = hunl.get_card_tensor()
    print("Current card tensor shape:", card_tensor.shape)
    print(card_tensor)
    print(f"Final stacks: {hunl.state.stacks}")
    
    print("\n")

    action_tensor = hunl.get_action_tensor()
    print("Current action tensor shape:", action_tensor.shape)
    print(action_tensor)
