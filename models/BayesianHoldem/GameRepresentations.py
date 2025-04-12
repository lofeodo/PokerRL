import torch
import pokerkit as pk

class GameRepresentations:
    @staticmethod
    def get_card_representations(state: pk.state.State, player_id: int) -> torch.Tensor:
        """
        Generate a card input tensor for the given player. Tensor dimensions are (6, 4, 13)
        Channel one: player's cards
        Channel two: flop cards
        Channel three: turn card
        Channel four: river card
        Channel five: current board cards
        Channel six: current board cards + player's cards

        Columns: card number (i.e. ace = 1, king = 13)
        Rows: suit number (i.e. clubs = 1, diamonds = 2, hearts = 3, spades = 4)
        """
        # Initialize the tensor with zeros on the GPU
        card_tensor = torch.zeros((6, 4, 13), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Helper function to convert card representation to tensor indices
        def card_to_tensor_indices(card: list[str]):
            rank = int(card[:-1])  # Get the rank (e.g., '5' from '5d')
            suit = {'c': 1, 'd': 2, 'h': 3, 's': 4}[card[-1]]  # Map suit to index
            return suit - 1, rank - 1  # Convert to zero-based index

        # Player's cards
        for card in state.hole_cards[player_id]:
            suit_idx, rank_idx = card_to_tensor_indices(card)
            card_tensor[0, suit_idx, rank_idx] = 1

        # Board cards
        for i, card in enumerate(state.board_cards):
            if card:  # Check if the card exists
                card = card[0]  # Extract the card from the list
                suit_idx, rank_idx = card_to_tensor_indices(card)
                if i < 3:  # Flop cards
                    card_tensor[1 + i, suit_idx, rank_idx] = 1
                else:  # Turn and river cards
                    card_tensor[i, suit_idx, rank_idx] = 1

        # Current board cards
        for card in state.board_cards:
            if card:  # Check if the card exists
                card = card[0]  # Extract the card from the list
                suit_idx, rank_idx = card_to_tensor_indices(card)
                card_tensor[4, suit_idx, rank_idx] = 1

        # Current board cards + player's cards
        card_tensor[5] = card_tensor[0] + card_tensor[4]

        return card_tensor

    @staticmethod
    def get_action_representations(state: pk.state.State) -> torch.Tensor:
        """
        Generate an action input tensor for the given player. Tensor dimensions are (24, 4, 4)
        Channel 1 - 6: Actions for the first betting round
        Channel 7 - 12: Actions for the second betting round
        Channel 13 - 18: Actions for the third betting round
        Channel 19 - 24: Actions for the fourth betting round

        Row 1: first player's action
        Row 2: second player's action
        Row 3: sum of actions
        Row 4: legal actions allowed
        """
        pass
    