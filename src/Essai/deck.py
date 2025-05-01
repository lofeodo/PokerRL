from card import Card
import random

class Deck:
    """A deck of cards."""
    
    def __init__(self):
        self.cards = [Card(r+s) for r in Card.RANKS for s in Card.SUITS]
        self.reset()
    
    def reset(self):
        """Reset and shuffle the deck."""
        self.cards = [Card(r+s) for r in Card.RANKS for s in Card.SUITS]
        random.shuffle(self.cards)
        
    def deal(self, n=1):
        """Deal n cards from the deck."""
        if n > len(self.cards):
            raise ValueError(f"Cannot deal {n} cards. Only {len(self.cards)} remaining.")
        cards = self.cards[:n]
        self.cards = self.cards[n:]
        return cards
