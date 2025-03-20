from deck import Deck


class PokerGameState:
    """
    Represents the state of a poker game.
    """
    
    # Action constants
    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2
    
    # Betting round constants
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    
    def __init__(self, 
                 stack_size=1000, 
                 small_blind=5, 
                 big_blind=10, 
                 current_round=PREFLOP):
        
        self.stack_size = stack_size
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.current_round = current_round
        
        self.deck = Deck()
        self.pot = 0
        self.board = []
        self.player_stacks = [stack_size, stack_size]
        self.player_bets = [0, 0]
        self.player_hole_cards = [[], []]
        self.current_player = 0  # 0 for player, 1 for opponent
        self.last_raise = 0
        self.min_raise = big_blind
        
        # Initialize with blinds
        self.post_blinds()
        
    def post_blinds(self):
        """Post the small and big blinds."""
        self.player_bets[0] = self.small_blind
        self.player_bets[1] = self.big_blind
        self.player_stacks[0] -= self.small_blind
        self.player_stacks[1] -= self.big_blind
        self.pot = self.small_blind + self.big_blind
        self.last_raise = self.big_blind
        self.min_raise = self.big_blind
        
    def deal_hole_cards(self):
        """Deal hole cards to both players."""
        self.player_hole_cards[0] = self.deck.deal(2)
        self.player_hole_cards[1] = self.deck.deal(2)
        
    def deal_community_cards(self):
        """Deal community cards based on the current round."""
        if self.current_round == self.FLOP:
            self.board = self.deck.deal(3)
        elif self.current_round == self.TURN or self.current_round == self.RIVER:
            self.board.extend(self.deck.deal(1))
    
    def get_legal_actions(self):
        """Get the legal actions for the current player."""
        legal_actions = []
        
        # Can always fold unless checking is free
        if self.player_bets[0] != self.player_bets[1]:
            legal_actions.append(self.FOLD)
            
        # Can always check or call
        legal_actions.append(self.CHECK_CALL)
        
        # Can raise if player has enough chips
        call_amount = abs(self.player_bets[1 - self.current_player] - self.player_bets[self.current_player])
        if self.player_stacks[self.current_player] > call_amount + self.min_raise:
            legal_actions.append(self.BET_RAISE)
            
        return legal_actions
    
    def act(self, action, raise_amount=None):
        """
        Execute an action for the current player.
        
        Args:
            action: FOLD, CHECK_CALL, or BET_RAISE
            raise_amount: Amount to raise (only used for BET_RAISE)
        
        Returns:
            A new game state after the action is applied
        """
        new_state = self.clone()
        
        # Execute the action
        if action == self.FOLD:
            # Current player folds, opponent wins the pot
            new_state.player_stacks[1 - new_state.current_player] += new_state.pot
            # Start a new hand
            return None
            
        elif action == self.CHECK_CALL:
            # Calculate call amount
            call_amount = new_state.player_bets[1 - new_state.current_player] - new_state.player_bets[new_state.current_player]
            
            # Update player's bet and stack
            new_state.player_bets[new_state.current_player] += call_amount
            new_state.player_stacks[new_state.current_player] -= call_amount
            new_state.pot += call_amount
            
        elif action == self.BET_RAISE:
            # If raise amount not specified, use minimum raise
            if raise_amount is None:
                raise_amount = new_state.min_raise
                
            # Calculate total amount (call + raise)
            call_amount = new_state.player_bets[1 - new_state.current_player] - new_state.player_bets[new_state.current_player]
            total_amount = call_amount + raise_amount
            
            # Update player's bet and stack
            new_state.player_bets[new_state.current_player] += total_amount
            new_state.player_stacks[new_state.current_player] -= total_amount
            new_state.pot += total_amount
            
            # Update last raise and minimum raise
            new_state.last_raise = raise_amount
            new_state.min_raise = raise_amount
        
        # Update current player
        new_state.current_player = 1 - new_state.current_player
        
        # Check if betting round is complete
        if new_state.player_bets[0] == new_state.player_bets[1]:
            # If all players have acted at least once
            if new_state.current_player == 1:  # Back to the first player
                # Advance to the next round
                new_state.current_round += 1
                
                # Reset bets for the new round
                new_state.player_bets = [0, 0]
                
                # Deal community cards for the new round
                new_state.deal_community_cards()
                
                # Reset minimum raise
                new_state.min_raise = new_state.big_blind
        
        return new_state
    
    def is_terminal(self):
        """Check if the game state is terminal (hand is over)."""
        return self.current_round > self.RIVER
    
    def get_payoff(self):
        """
        Calculate the payoff for each player.
        Returns a list [player_payoff, opponent_payoff].
        """
        if self.is_terminal():
            from handEvaluator import HandEvaluator
            # Evaluate hands
            player_hand = HandEvaluator.evaluate_hand(self.player_hole_cards[0], self.board)
            opponent_hand = HandEvaluator.evaluate_hand(self.player_hole_cards[1], self.board)
            
            # Compare hands and distribute pot
            if player_hand < opponent_hand:  # Player wins (lower value is better)
                return [self.pot, -self.pot]
            elif player_hand > opponent_hand:  # Opponent wins
                return [-self.pot, self.pot]
            else:  # Tie
                return [0, 0]
        
        return [0, 0]  # Default if not terminal
    
    def get_info_set(self, player_idx):
        """
        Generate a string representation of the information set for a player.
        """
        # Include hole cards
        info_set = ''.join(str(card) for card in self.player_hole_cards[player_idx]) + '|'
        
        # Include community cards
        info_set += ''.join(str(card) for card in self.board) + '|'
        
        # Include betting history (simplified for this example)
        info_set += f"{self.current_round}|{self.pot}|{self.player_bets[0]}|{self.player_bets[1]}"
        
        return info_set
    
    def clone(self):
        """Create a deep copy of the current game state."""
        new_state = PokerGameState(
            stack_size=self.stack_size,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            current_round=self.current_round
        )
        
        new_state.deck = Deck()
        new_state.deck.cards = self.deck.cards.copy()
        
        new_state.pot = self.pot
        new_state.board = self.board.copy()
        new_state.player_stacks = self.player_stacks.copy()
        new_state.player_bets = self.player_bets.copy()
        new_state.player_hole_cards = [cards.copy() for cards in self.player_hole_cards]
        new_state.current_player = self.current_player
        new_state.last_raise = self.last_raise
        new_state.min_raise = self.min_raise
        
        return new_state

