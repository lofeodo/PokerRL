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
            ante_trimming_status=True,
            raw_antes=self.ante,
            raw_blinds_or_straddles=self.blinds,
            min_bet=self.min_bet,
        )

        self.state = self.game(
            raw_starting_stacks=self.starting_stack,
            player_count=self.player_count
        )

        # 6x4x13 tensor
        self.tensor = np.zeros((6, 4, 13), dtype=np.int8)
        self.update_tensor_for_player(0)

    def update_tensor_for_player(self, player_idx=0):
        """Update the 6x4x13 tensor for the given player's viewpoint."""
        self.tensor.fill(0)

        hole_cards = self.get_hole_cards(player_idx)

        # Instead of taking the entire state.board_cards, pick the single board
        # at self.board_idx (since we only have one board in HUNL).
        # print(f'board_idx: {self.board_idx}')
        # print(f'board_cards: {self.state.board_cards}')
        # if self.board_idx < len(self.state.board_cards):

        raw_board = [item for sublist in self.state.board_cards for item in sublist]

        # print(f'raw_board: {raw_board}')
        community_cards = []
        for item in raw_board:
            if isinstance(item, list):
                community_cards.extend(item)  # add the sub-list’s cards
            else:
                community_cards.append(item)
        # else:
        #     community_cards = []

        print(f'community_cards: {community_cards}')

        # Channel 0: Hole cards
        self.add_cards_to_tensor(hole_cards, 0)

        # Channel 1: Flop (if at least 3 total cards are present)
        if len(community_cards) >= 3:
            flop = community_cards[:3]
            self.add_cards_to_tensor(flop, 1)

        # Channel 2: Turn
        if len(community_cards) >= 4:
            turn = [community_cards[3]]
            self.add_cards_to_tensor(turn, 2)

        # Channel 3: River
        if len(community_cards) == 5:
            river = [community_cards[4]]
            self.add_cards_to_tensor(river, 3)

        # Channel 4: All community cards
        self.add_cards_to_tensor(community_cards, 4)

        # Channel 5: All cards combined
        all_cards = list(community_cards) + hole_cards
        self.add_cards_to_tensor(all_cards, 5)

    def get_hole_cards(self, player_idx=0):
        if player_idx < len(self.state.hole_cards):
            cards = list(self.state.hole_cards[player_idx])
            return [c for c in cards if isinstance(c, pk.Card)]
        return []

    def add_cards_to_tensor(self, cards, channel):
        """Adds each pk.Card object to the tensor at the specified channel."""
        for card in cards:
            suit_index = self.get_suit_index(card.suit)
            rank_index = self.get_rank_index(card.rank)
            self.tensor[channel, suit_index, rank_index] = 1

    def get_suit_index(self, suit_char):
        mapping = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
        return mapping[suit_char]

    def get_rank_index(self, rank_char):
        rank_map = {
            'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5,
            '7': 6, '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12
        }
        return rank_map[rank_char]

    def get_card_tensor(self):
        """Return the 6x4x13 tensor."""
        return self.tensor

    def complete_bet_or_raise_to(self, amount):
        self.state.complete_bet_or_raise_to(amount)
        self.update_tensor_for_player(self.state.actor_index)

    def check_or_call(self):
        # If there's no valid next actor, do nothing
        if self.state.actor_index is None:
            print("No next actor (hand is probably over).")
            return
        self.state.check_or_call()
        # If the actor now becomes None, skip
        if self.state.actor_index is not None:
            self.update_tensor_for_player(self.state.actor_index)

    def fold(self):
        self.state.fold()
        self.update_tensor_for_player(self.state.actor_index)


if __name__ == "__main__":
    hunl = HUNLPoker()

    print("Pre-flop betting")
    print(
        f"Player {hunl.state.actor_index}'s turn,\n"
        f"  hole_cards for this player: {hunl.get_hole_cards(hunl.state.actor_index)},\n"
        f"  bets: {hunl.state.bets},\n"
        f"  queue: {hunl.state.actor_indices}"
    )

    # Example bet
    hunl.complete_bet_or_raise_to(150)
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
    tensor = hunl.get_card_tensor()
    print("Current card tensor shape:", tensor.shape)
    print("Current card tensor for player", hunl.state.actor_index, "\n", tensor)
    print(f"Final stacks: {hunl.state.stacks}")
