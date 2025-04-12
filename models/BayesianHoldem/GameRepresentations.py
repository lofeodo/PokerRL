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
        Channel 1 - 6: Actions for the preflop betting round
        Channel 7 - 12: Actions for the flop betting round
        Channel 13 - 18: Actions for the turn betting round
        Channel 19 - 24: Actions for the river betting round

        Row 1: first player's action
        Row 2: second player's action
        Row 3: sum of player 0 and player 1 actions
        Row 4: legal actions allowed
        """
        action_tensor = torch.zeros((24, 4, 4), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

        current_round = 0  # Track the current betting round
        player_actions = [[], []]  # Store actions for player 0 and player 1

        for idx, operation in enumerate(state.operations):
            prev_operation = state.operations[idx - 1] if idx > 0 else None
            if isinstance(prev_operation, (pk.state.HoleDealing, pk.state.BoardDealing)) and \
                prev_operation is not None and \
                not isinstance(operation, (pk.state.HoleDealing, pk.state.BoardDealing)):
                if player_actions[0] and player_actions[1]:
                    for idx in range(len(player_actions[0])):
                        channel_index = (current_round - 1) * 6 + idx
                        action_tensor[channel_index, 0, player_actions[0][idx]] = 1
                        action_tensor[channel_index, 1, player_actions[1][idx]] = 1
                        action_tensor[channel_index, 2] = action_tensor[channel_index, 0] + action_tensor[channel_index, 1]
                        action_tensor[channel_index, 3] = GameRepresentations.get_legal_actions(state, idx)
                        
                current_round += 1  # Move to the next betting round
                player_actions = [[], []]  # Reset player actions for the new round

            if isinstance(operation, pk.state.Folding):
                player_index = operation.player_index
                player_actions[player_index].append(0)  # Mark as fold

            elif isinstance(operation, pk.state.CheckingOrCalling):
                player_index = operation.player_index
                player_actions[player_index].append(1)  # Mark as check/call

            elif isinstance(operation, pk.state.CompletionBettingOrRaisingTo):
                player_index = operation.player_index
                # Check if player has gone all in:
                if state.stacks[player_index] == 0:
                    player_actions[player_index].append(3)  # Mark as all-in
                else:
                    player_actions[player_index].append(2)  # Mark as raise

        return action_tensor

    @staticmethod
    def get_legal_actions(state: pk.state.State, operation_idx: int) -> torch.Tensor:
        """
        Generate a legal actions tensor for the given player. Tensor dimensions are (1, 4)
        """
        legal_actions = torch.zeros((4, 4), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        operations = state.operations[:operation_idx]
        return None
    