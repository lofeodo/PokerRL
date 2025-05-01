"""
This module provides functionality to browse and visualize extracted poker hands data in the console.
It works with the simplified JSON format produced by the extract.py script.
"""

import json
import os
import argparse
from typing import Optional, List, Dict, Any

class PokerHandsBrowser:
    def __init__(self, filename: str):
        """
        Initialize browser for poker hands data
        
        Args:
            filename (str): Path to the JSON file containing poker hand data
        """
        self.filename = filename
        self.hands = []
        self.load_data()
        
    def load_data(self):
        """Load poker hands data from the JSON file"""
        if not os.path.exists(self.filename):
            print(f"Error: File {self.filename} not found")
            return
        
        try:
            with open(self.filename, 'r') as f:
                self.hands = json.load(f)
            print(f"Loaded {len(self.hands)} poker hands from {self.filename}")
        except json.JSONDecodeError:
            print(f"Error: {self.filename} is not a valid JSON file")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def browse(self, game_type: Optional[str] = None, limit: int = 100, manual_mode: bool = False):
        """
        Browse through poker hands data
        
        Args:
            game_type (Optional[str]): Filter hands by game type (e.g., 'holdem')
            limit (int): Maximum number of hands to display
            manual_mode (bool): If True, wait for user input between hands
        """
        count = 0
        
        for hand in self.hands:
            # Apply game type filter if specified
            if game_type and not hand.get("game", "") == game_type:
                continue
                
            self._display_hand(hand)
            count += 1
            
            if count >= limit:
                break
                
            if manual_mode:
                input("\nPress Enter to see next hand...")
            else:
                print()
    
    def _display_hand(self, hand: Dict[str, Any]):
        """
        Display a single poker hand in a formatted way
        
        Args:
            hand: Dictionary containing hand data
        """
        # Extract basic information
        hand_id = hand.get("_id", "unknown")
        game = hand.get("game", "unknown")
        timestamp = hand.get("timestamp", "unknown")
        dealer = hand.get("dealer", 0)
        hand_num = hand.get("hand_num", 0)
        num_players = hand.get("num_players", 0)
        board = hand.get("board", [])
        
        # Extract pot information for each stage
        stages = ["flop", "turn", "river", "showdown"]
        pot_info = []
        
        for stage in stages:
            if stage in hand:
                players = hand[stage].get("num_players", 0)
                pot = hand[stage].get("pot_size", 0)
                pot_info.append(f"{stage.capitalize()}: {players} players, pot: ${pot}")
        
        # Create the display
        border = "=" * 80
        print(border)
        print(f"Hand ID: {hand_id} | Game: {game}")
        print(f"Time: {timestamp} | Hand #: {hand_num} | Dealer: {dealer} | Players: {num_players}")
        
        if board:
            print(f"Board: {' '.join(board)}")
        
        print("\nPot progression:")
        for info in pot_info:
            print(f"  {info}")
            
        print(border)
    
    def search(self, min_players: int = 0, min_pot: int = 0, has_showdown: bool = False):
        """
        Search for hands matching specific criteria
        
        Args:
            min_players (int): Minimum number of players
            min_pot (int): Minimum pot size at showdown
            has_showdown (bool): If True, only return hands that reached showdown
        """
        matching_hands = []
        
        for hand in self.hands:
            # Check minimum players
            if hand.get("num_players", 0) < min_players:
                continue
                
            # Check if hand reached showdown
            if has_showdown and "showdown" not in hand:
                continue
                
            # Check minimum pot size
            if "showdown" in hand and hand["showdown"].get("pot_size", 0) < min_pot:
                continue
                
            matching_hands.append(hand)
        
        print(f"Found {len(matching_hands)} hands matching criteria")
        
        # Display first 10 matching hands
        for i, hand in enumerate(matching_hands[:10]):
            print(f"\nResult {i+1}:")
            self._display_hand(hand)
            
        return matching_hands
    
    def get_statistics(self):
        """Calculate and display statistics about the poker hands data"""
        if not self.hands:
            print("No data loaded")
            return
        
        total_hands = len(self.hands)
        game_types = {}
        player_counts = {}
        showdown_count = 0
        pot_sizes = []
        
        for hand in self.hands:
            # Count game types
            game = hand.get("game", "unknown")
            game_types[game] = game_types.get(game, 0) + 1
            
            # Count number of players
            players = hand.get("num_players", 0)
            player_counts[players] = player_counts.get(players, 0) + 1
            
            # Count showdowns
            if "showdown" in hand:
                showdown_count += 1
                pot_sizes.append(hand["showdown"].get("pot_size", 0))
        
        # Calculate statistics
        showdown_percentage = (showdown_count / total_hands) * 100 if total_hands > 0 else 0
        avg_pot = sum(pot_sizes) / len(pot_sizes) if pot_sizes else 0
        max_pot = max(pot_sizes) if pot_sizes else 0
        
        # Display statistics
        border = "=" * 80
        print(border)
        print(f"POKER HANDS STATISTICS ({total_hands} hands)")
        print(border)
        
        print("\nGame Types:")
        for game, count in sorted(game_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_hands) * 100
            print(f"  {game}: {count} hands ({percentage:.1f}%)")
        
        print("\nPlayer Distribution:")
        for players, count in sorted(player_counts.items()):
            percentage = (count / total_hands) * 100
            print(f"  {players} players: {count} hands ({percentage:.1f}%)")
        
        print(f"\nHands reaching showdown: {showdown_count} ({showdown_percentage:.1f}%)")
        print(f"Average pot at showdown: ${avg_pot:.2f}")
        print(f"Maximum pot: ${max_pot}")
        print(border)


def main():
    """Main function to run the poker hands browser"""
    parser = argparse.ArgumentParser(description="Browse poker hands data")
    parser.add_argument("--file", "-f", default="holdem_hands.json", help="Path to the JSON file containing poker hand data")
    parser.add_argument("--game", "-g", help="Filter by game type (e.g., holdem)")
    parser.add_argument("--limit", "-l", type=int, default=50, help="Maximum number of hands to display")
    parser.add_argument("--manual", "-m", action="store_true", help="Manual mode (press Enter between hands)")
    parser.add_argument("--stats", "-s", action="store_true", help="Display statistics about the data")
    parser.add_argument("--search", action="store_true", help="Search for hands matching criteria")
    parser.add_argument("--min-players", type=int, default=0, help="Minimum number of players")
    parser.add_argument("--min-pot", type=int, default=0, help="Minimum pot size at showdown")
    parser.add_argument("--has-showdown", action="store_true", help="Only show hands that reached showdown")
    
    args = parser.parse_args()
    
    browser = PokerHandsBrowser(args.file)
    
    if args.stats:
        browser.get_statistics()
    elif args.search:
        browser.search(args.min_players, args.min_pot, args.has_showdown)
    else:
        browser.browse(args.game, args.limit, args.manual)


if __name__ == "__main__":
    main()