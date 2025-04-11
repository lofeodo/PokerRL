from deck import Deck
from card import Card
from collections import defaultdict


class HandEvaluator:
    """Evaluates poker hands and calculates their strength."""
    
    # Hand rankings from highest to lowest
    HAND_RANKINGS = [
        "straight_flush", "four_of_a_kind", "full_house", 
        "flush", "straight", "three_of_a_kind", 
        "two_pair", "one_pair", "high_card"
    ]
    
    @staticmethod
    def evaluate_hand(hole_cards, board_cards):
        """
        Evaluate the best 5-card hand from hole cards and board cards.
        Returns a tuple of (hand_rank, hand_value) where hand_rank is the type of hand
        and hand_value is used to break ties within the same hand rank.
        Lower values are stronger (1 is the strongest).
        """
        # Combine hole cards and board cards
        cards = hole_cards + board_cards
        
        # Check for each hand type, from best to worst
        if HandEvaluator._has_straight_flush(cards):
            return (0, HandEvaluator._straight_flush_value(cards))
        elif HandEvaluator._has_four_of_a_kind(cards):
            return (1, HandEvaluator._four_of_a_kind_value(cards))
        elif HandEvaluator._has_full_house(cards):
            return (2, HandEvaluator._full_house_value(cards))
        elif HandEvaluator._has_flush(cards):
            return (3, HandEvaluator._flush_value(cards))
        elif HandEvaluator._has_straight(cards):
            return (4, HandEvaluator._straight_value(cards))
        elif HandEvaluator._has_three_of_a_kind(cards):
            return (5, HandEvaluator._three_of_a_kind_value(cards))
        elif HandEvaluator._has_two_pair(cards):
            return (6, HandEvaluator._two_pair_value(cards))
        elif HandEvaluator._has_one_pair(cards):
            return (7, HandEvaluator._one_pair_value(cards))
        else:
            return (8, HandEvaluator._high_card_value(cards))
    
    @staticmethod
    def _has_straight_flush(cards):
        # Group cards by suit
        by_suit = defaultdict(list)
        for card in cards:
            by_suit[card.suit].append(card)
        
        # Check each suit for a straight
        for suit, suited_cards in by_suit.items():
            if len(suited_cards) >= 5 and HandEvaluator._has_straight(suited_cards):
                return True
        return False
    
    @staticmethod
    def _straight_flush_value(cards):
        # Group cards by suit
        by_suit = defaultdict(list)
        for card in cards:
            by_suit[card.suit].append(card)
        
        # Get the highest straight flush
        highest_straight = -1
        for suit, suited_cards in by_suit.items():
            if len(suited_cards) >= 5:
                straight_value = HandEvaluator._straight_value(suited_cards)
                if straight_value != -1:
                    highest_straight = max(highest_straight, HandEvaluator._straight_value(suited_cards))
        
        return highest_straight
    
    @staticmethod
    def _has_four_of_a_kind(cards):
        rank_counts = defaultdict(int)
        for card in cards:
            rank_counts[card.rank] += 1
        
        return any(count >= 4 for count in rank_counts.values())
    
    @staticmethod
    def _four_of_a_kind_value(cards):
        rank_counts = defaultdict(int)
        for card in cards:
            rank_counts[card.rank] += 1
        
        four_rank = [rank for rank, count in rank_counts.items() if count >= 4][0]
        four_value = Card.RANKS.index(four_rank)
        
        # Find the highest kicker
        kickers = [card.rank_value for card in cards if card.rank != four_rank]
        kicker_value = max(kickers) if kickers else 0
        
        return four_value * 13 + kicker_value
    
    @staticmethod
    def _has_full_house(cards):
        rank_counts = defaultdict(int)
        for card in cards:
            rank_counts[card.rank] += 1
        
        has_three = any(count >= 3 for count in rank_counts.values())
        pairs = sum(1 for count in rank_counts.values() if count >= 2)
        
        return has_three and pairs >= 2
    
    @staticmethod
    def _full_house_value(cards):
        rank_counts = defaultdict(int)
        for card in cards:
            rank_counts[card.rank] += 1
        
        three_ranks = [rank for rank, count in rank_counts.items() if count >= 3]
        pair_ranks = [rank for rank, count in rank_counts.items() if count >= 2]
        
        three_rank = max(three_ranks, key=lambda r: Card.RANKS.index(r))
        three_value = Card.RANKS.index(three_rank)
        
        pair_ranks = [r for r in pair_ranks if r != three_rank]
        pair_value = max([Card.RANKS.index(r) for r in pair_ranks])
        
        return three_value * 13 + pair_value
    
    @staticmethod
    def _has_flush(cards):
        suit_counts = defaultdict(int)
        for card in cards:
            suit_counts[card.suit] += 1
        
        return any(count >= 5 for count in suit_counts.values())
    
    @staticmethod
    def _flush_value(cards):
        suit_counts = defaultdict(int)
        suited_cards = defaultdict(list)
        
        for card in cards:
            suit_counts[card.suit] += 1
            suited_cards[card.suit].append(card)
        
        flush_suit = next(suit for suit, count in suit_counts.items() if count >= 5)
        flush_cards = sorted(suited_cards[flush_suit], key=lambda c: c.rank_value, reverse=True)
        
        # Take the 5 highest cards of the flush suit
        top_five = flush_cards[:5]
        
        # Calculate the value - higher ranks are more valuable
        value = 0
        for i, card in enumerate(top_five):
            value += card.rank_value * (13 ** (4 - i))
        
        return value
    
    @staticmethod
    def _has_straight(cards):
        # Get unique ranks
        ranks = sorted(set(card.rank_value for card in cards))
        
        # Check for standard straight
        for i in range(len(ranks) - 4):
            if ranks[i+4] - ranks[i] == 4:
                return True
        
        # Check for A-5 straight (wheel)
        if 12 in ranks and 0 in ranks and 1 in ranks and 2 in ranks and 3 in ranks:
            return True
            
        return False
    
    @staticmethod
    def _straight_value(cards):
        # Get unique ranks
        ranks = sorted(set(card.rank_value for card in cards))
        
        # Check for standard straight
        for i in range(len(ranks) - 4):
            if ranks[i+4] - ranks[i] == 4:
                return ranks[i+4]  # Return the high card of the straight
        
        # Check for A-5 straight (wheel)
        if 12 in ranks and 0 in ranks and 1 in ranks and 2 in ranks and 3 in ranks:
            return 3  # A-5 straight has 5 as the highest card
            
        return -1  # No straight
    
    @staticmethod
    def _has_three_of_a_kind(cards):
        rank_counts = defaultdict(int)
        for card in cards:
            rank_counts[card.rank] += 1
        
        return any(count >= 3 for count in rank_counts.values())
    
    @staticmethod
    def _three_of_a_kind_value(cards):
        rank_counts = defaultdict(int)
        for card in cards:
            rank_counts[card.rank] += 1
        
        three_rank = [rank for rank, count in rank_counts.items() if count >= 3][0]
        three_value = Card.RANKS.index(three_rank)
        
        # Find the two highest kickers
        kickers = sorted([card.rank_value for card in cards if card.rank != three_rank], reverse=True)
        kicker_values = kickers[:2] if len(kickers) >= 2 else kickers + [0] * (2 - len(kickers))
        
        return three_value * 13**2 + kicker_values[0] * 13 + kicker_values[1]
    
    @staticmethod
    def _has_two_pair(cards):
        rank_counts = defaultdict(int)
        for card in cards:
            rank_counts[card.rank] += 1
        
        pairs = sum(1 for count in rank_counts.values() if count >= 2)
        return pairs >= 2
    
    @staticmethod
    def _two_pair_value(cards):
        rank_counts = defaultdict(int)
        for card in cards:
            rank_counts[card.rank] += 1
        
        pair_ranks = [rank for rank, count in rank_counts.items() if count >= 2]
        pair_values = sorted([Card.RANKS.index(r) for r in pair_ranks], reverse=True)
        
        # Use the two highest pairs
        high_pair, low_pair = pair_values[0], pair_values[1]
        
        # Find the highest kicker
        kickers = [card.rank_value for card in cards 
                  if card.rank != Card.RANKS[high_pair] and card.rank != Card.RANKS[low_pair]]
        kicker_value = max(kickers) if kickers else 0
        
        return high_pair * 13**2 + low_pair * 13 + kicker_value
    
    @staticmethod
    def _has_one_pair(cards):
        rank_counts = defaultdict(int)
        for card in cards:
            rank_counts[card.rank] += 1
        
        return any(count >= 2 for count in rank_counts.values())
    
    @staticmethod
    def _one_pair_value(cards):
        rank_counts = defaultdict(int)
        for card in cards:
            rank_counts[card.rank] += 1
        
        pair_rank = [rank for rank, count in rank_counts.items() if count >= 2][0]
        pair_value = Card.RANKS.index(pair_rank)
        
        # Find the three highest kickers
        kickers = sorted([card.rank_value for card in cards if card.rank != pair_rank], reverse=True)
        kicker_values = kickers[:3] if len(kickers) >= 3 else kickers + [0] * (3 - len(kickers))
        
        return pair_value * 13**3 + kicker_values[0] * 13**2 + kicker_values[1] * 13 + kicker_values[2]
    
    @staticmethod
    def _high_card_value(cards):
        # Sort by rank (highest first)
        sorted_cards = sorted(cards, key=lambda c: c.rank_value, reverse=True)
        
        # Take the 5 highest cards
        top_five = sorted_cards[:5]
        
        # Calculate the value
        value = 0
        for i, card in enumerate(top_five):
            value += card.rank_value * (13 ** (4 - i))
        
        return value
    
    @staticmethod
    def calculate_equity(hole_cards, board_cards, num_simulations=1000):
        """
        Calculate the equity (winning probability) of a hand using Monte Carlo simulation.
        """
        wins = 0
        ties = 0
        
        # Convert string representations to Card objects if needed
        if isinstance(hole_cards[0], str):
            hole_cards = [Card(c) for c in hole_cards]
        if board_cards and isinstance(board_cards[0], str):
            board_cards = [Card(c) for c in board_cards]
        
        # Cards that are already dealt
        used_cards = hole_cards + board_cards
        
        for _ in range(num_simulations):
            # Create a new deck excluding used cards
            deck = Deck()
            deck.cards = [card for card in deck.cards if card not in used_cards]
            
            # Deal opponent's hole cards
            opponent_hole = deck.deal(2)
            
            # Complete the board if needed
            remaining_board = deck.deal(5 - len(board_cards))
            complete_board = board_cards + remaining_board
            
            # Evaluate both hands
            player_hand = HandEvaluator.evaluate_hand(hole_cards, complete_board)
            opponent_hand = HandEvaluator.evaluate_hand(opponent_hole, complete_board)
            
            # Compare hands
            if player_hand < opponent_hand:  # Lower values are stronger
                wins += 1
            elif player_hand == opponent_hand:
                ties += 0.5
        
        # Return equity (probability of winning)
        return (wins + ties) / num_simulations
