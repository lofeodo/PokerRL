import torch
import pokerkit as pk

class GameRepresentations:
    """
    This class is used to obtain card and action representations
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Pre-allocate tensors for efficiency
        self.card_tensor = torch.zeros((6, 4, 13), dtype=torch.float32, device=self.device)
        self.legal_actions = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
        self.action_tensor = torch.zeros((24, 4, 4), dtype=torch.float32, device=self.device)

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

        # Helper function to convert suit and rank to tensor indices
        def card_to_tensor_indices(card):
            suit_mapping = {'c': 1, 'd': 2, 'h': 3, 's': 4}
            rank_mapping = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}
            
            suit_idx = suit_mapping[card.suit] - 1  # Convert to zero-based index
            rank_idx = rank_mapping[card.rank] - 1  # Convert to zero-based index
            return suit_idx, rank_idx

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
                    card_tensor[1, suit_idx, rank_idx] = 1
                else:  # Turn and river cards
                    card_tensor[i - 1, suit_idx, rank_idx] = 1

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
    def get_action_representations(state: pk.state.State, prev_action_tensor: torch.Tensor, player_id: int) -> torch.Tensor:
        """
        This function is used to get the action representations for the given player. It should be called after each action is taken
        to update the action tensor and keep it up to date.

        Generate an action input tensor for the given player. Tensor dimensions are (24, 4, 4)
        Channel 0 - 5: Actions for the preflop betting round
        Channel 6 - 11: Actions for the flop betting round
        Channel 12 - 17: Actions for the turn betting round
        Channel 18 - 23: Actions for the river betting round

        Row 1: first player's action
        Row 2: second player's action
        Row 3: sum of player 0 and player 1 actions
        Row 4: legal actions allowed at the time of the action
        """        
        # Ensure input tensor is on the correct device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        action_tensor = prev_action_tensor.to(device)

        # Get the index of the operation that starts the current round
        start_idx, current_round = GameRepresentations.get_beginning_of_round_idx(state)

        player_actions = [[], []]  # Current action pair being processed
        other_player_id = abs(1 - player_id)  # Determine the other player's ID

        # Iterate forward from start_idx to populate the tensor
        for idx in range(start_idx, len(state.operations)):
            operation = state.operations[idx]

            # Record the action
            if isinstance(operation, pk.state.Folding):
                player_actions[operation.player_index].append(0)
            elif isinstance(operation, pk.state.CheckingOrCalling):
                player_actions[operation.player_index].append(1)
            elif isinstance(operation, pk.state.CompletionBettingOrRaisingTo):
                if state.stacks[operation.player_index] == 0:
                    player_actions[operation.player_index].append(3)  # All-in
                else:
                    player_actions[operation.player_index].append(2)  # Raise

        channel_idx = current_round * 6
        # Populate the action tensor for the specified player
        for idx, action in enumerate(player_actions[player_id]):
            if idx >= 6:
                break
            action_tensor[channel_idx + idx, 0, action] = 1  # Player's actions in row 0

        # Populate the action tensor for the other player
        for idx, action in enumerate(player_actions[other_player_id]):
            if idx >= 6:
                break
            action_tensor[channel_idx + idx, 1, action] = 1  # Other player's actions in row 1

        # Populate the sum of actions
        for idx in range(max(len(player_actions[player_id]), len(player_actions[other_player_id]))):
            if idx >= 6:
                break
            action_tensor[channel_idx + idx, 2] = action_tensor[channel_idx + idx, 0] + action_tensor[channel_idx + idx, 1]

        # Populate legal actions if applicable
        if len(player_actions[player_id]) <= 6 and len(player_actions[player_id]) > 0:
            action_tensor[channel_idx + len(player_actions[player_id]) - 1, 3] = GameRepresentations.get_legal_actions(state, player_id)

        return action_tensor

    @staticmethod
    def get_legal_actions(state: pk.state.State, player_id: int) -> torch.Tensor:
        """
        Generate a legal actions tensor for the given player. Tensor dimensions are (1, 4)
        """
        legal_actions = torch.zeros((1, 4), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

        last_player_action = None
        last_other_player_action = None
        other_player_id = abs(1 - player_id)  # Determine the other player's ID

        for idx in range(len(state.operations) - 1, -1, -1):
            operation = state.operations[idx]
            if isinstance(operation, pk.state.BoardDealing):
                if idx == len(state.operations) - 1:
                    continue
                else:
                    break
            elif isinstance(operation, pk.state.HoleDealing):
                break
            elif isinstance(operation, (pk.state.CheckingOrCalling, pk.state.CompletionBettingOrRaisingTo, pk.state.Folding)):
                if operation.player_index == player_id:
                    last_player_action = operation
                else:
                    last_other_player_action = operation

        # If the player has folded, no available actions
        if isinstance(last_player_action, pk.state.Folding) or isinstance(last_other_player_action, pk.state.Folding):
            return legal_actions  # No available actions if player has folded

        # No action has been played this round.
        if last_player_action is None and last_other_player_action is None:
            return torch.tensor([[0, 1, 1, 1]], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Check if the player can fold
        if last_other_player_action is not None and \
            not isinstance(last_other_player_action, (pk.state.CheckingOrCalling, pk.state.HoleDealing, pk.state.BoardDealing)):
            legal_actions[0, 0] = 1

        # Check/call is always legal
        legal_actions[0, 1] = 1

        # Check if the player can bet/raise
        bb = tuple(state.blinds_or_straddles)[1]
        if bb <= state.stacks[player_id] and bb <= state.stacks[other_player_id]:
            legal_actions[0, 2] = 1

        # Check if player can go all-in
        if state.stacks[player_id] != 0:
            legal_actions[0, 3] = 1

        return legal_actions
    
    @staticmethod
    def get_beginning_of_round_idx(state: pk.state.State) -> int:
        """
        Get the index of the operation that starts the current round.
        """
        current_round = state.street_index
        for idx in range(len(state.operations) - 1, -1, -1):
            operation = state.operations[idx]
            
            if isinstance(operation, pk.state.BoardDealing):
                if idx == len(state.operations) - 1:  # If this is the most recent operation
                    current_round -= 1  # We need to populate the previous round
                    continue
                return idx + 1, current_round  # Start from the operation after the dealing
            
        return 0, 0