class Card:
    """Representation of a playing card."""
    
    RANKS = '23456789TJQKA'
    SUITS = 'hdcs'  # hearts, diamonds, clubs, spades
    
    
    def __init__(self, card_str):
        self.rank = card_str[0]
        self.suit = card_str[1]
        self.rank_value = Card.RANKS.index(self.rank)
        
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
 