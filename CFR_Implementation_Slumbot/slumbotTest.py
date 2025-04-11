import requests
import sys
import argparse
import json
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import os

from depthLimitedCFR import DepthLimitedCFR
from pokerAgent import PokerAgent
from pokerGameState import PokerGameState
from card import Card
from handEvaluator import HandEvaluator

class SlumbotIntegration:
    """
    Integrates the poker agent with Slumbot API.
    """
    
    # Slumbot constants
    SLUMBOT_HOST = 'slumbot.com'
    SMALL_BLIND = 50
    BIG_BLIND = 100
    STACK_SIZE = 20000
    
    # Action mapping
    FOLD = 'f'
    CHECK = 'k'
    CALL = 'c'
    BET = 'b'
    
    def __init__(self, blueprint_path=None, max_depth=3, enable_learning=True):
        """
        Initialize the integration with a poker agent.
        
        Args:
            blueprint_path: Path to a saved blueprint strategy
            max_depth: Maximum depth for the solver
            enable_learning: Whether to continuously update the blueprint
        """
        print("Initializing Slumbot integration...")
        
        # Create the poker agent
        self.agent = PokerAgent(
            blueprint_path=blueprint_path,
            stack_size=self.STACK_SIZE,
            small_blind=self.SMALL_BLIND,
            big_blind=self.BIG_BLIND,
            enable_learning=enable_learning
        )
        
        # Session data
        self.token = None
        self.total_winnings = 0
        self.hands_played = 0
        self.wins = 0
        self.losses = 0
        self.ties = 0
        
        print(f"Poker agent initialized with blueprint: {blueprint_path}")
        print(f"Using depth-limited solver with max depth: {max_depth}")
        print(f"Continuous learning: {'Enabled' if enable_learning else 'Disabled'}")
        
    def slumbot_card_to_card(self, slumbot_card: str) -> Card:
        """
        Convert Slumbot card format to our Card object.
        
        Args:
            slumbot_card: Card string in Slumbot format (e.g., 'Ac', 'Th')
            
        Returns:
            Card object
        """
        # Slumbot uses 'h' for hearts, 'd' for diamonds, 'c' for clubs, 's' for spades
        # Our code might use the same format, but we'll ensure compatibility
        return Card(slumbot_card)
    
    def parse_action(self, action: str) -> Dict:
        """
        Parse Slumbot action string and return action information.
        This is a modified version of the ParseAction function from the Slumbot API example.
        
        Args:
            action: Action string from Slumbot API
            
        Returns:
            Dict with action details
        """
        st = 0
        street_last_bet_to = self.BIG_BLIND
        total_last_bet_to = self.BIG_BLIND
        last_bet_size = self.BIG_BLIND - self.SMALL_BLIND
        last_bettor = 0
        sz = len(action)
        pos = 1
        
        if sz == 0:
            return {
                'st': st,
                'pos': pos,
                'street_last_bet_to': street_last_bet_to,
                'total_last_bet_to': total_last_bet_to,
                'last_bet_size': last_bet_size,
                'last_bettor': last_bettor,
            }

        check_or_call_ends_street = False
        i = 0
        action_history = []  # Keep track of actions for our game state
        
        while i < sz:
            if st >= 4:  # 4 streets in NL Hold'em
                return {'error': 'Unexpected error'}
            
            c = action[i]
            i += 1
            
            if c == 'k':  # Check
                action_history.append({'player': pos, 'action': 'check'})
                if last_bet_size > 0:
                    return {'error': 'Illegal check'}
                if check_or_call_ends_street:
                    # After a check that ends a pre-river street, expect either a '/' or end of string
                    if st < 3 and i < sz:  # 0-3 for streets
                        if action[i] != '/':
                            return {'error': 'Missing slash'}
                        i += 1
                    if st == 3:  # River
                        # Reached showdown
                        pos = -1
                    else:
                        pos = 0
                        st += 1
                    street_last_bet_to = 0
                    check_or_call_ends_street = False
                else:
                    pos = (pos + 1) % 2
                    check_or_call_ends_street = True
                    
            elif c == 'c':  # Call
                action_history.append({'player': pos, 'action': 'call', 'amount': street_last_bet_to})
                if last_bet_size == 0:
                    return {'error': 'Illegal call'}
                if total_last_bet_to == self.STACK_SIZE:
                    # Call of an all-in bet
                    if i != sz:
                        for st1 in range(st, 3):  # 0-3 for streets
                            if i == sz:
                                return {'error': 'Missing slash (end of string)'}
                            else:
                                c = action[i]
                                i += 1
                                if c != '/':
                                    return {'error': 'Missing slash'}
                    if i != sz:
                        return {'error': 'Extra characters at end of action'}
                    st = 3  # River
                    pos = -1
                    last_bet_size = 0
                    return {
                        'st': st,
                        'pos': pos,
                        'street_last_bet_to': street_last_bet_to,
                        'total_last_bet_to': total_last_bet_to,
                        'last_bet_size': last_bet_size,
                        'last_bettor': last_bettor,
                        'action_history': action_history
                    }
                if check_or_call_ends_street:
                    # After a call that ends a pre-river street, expect either a '/' or end of string
                    if st < 3 and i < sz:  # 0-3 for streets
                        if action[i] != '/':
                            return {'error': 'Missing slash'}
                        i += 1
                    if st == 3:  # River
                        # Reached showdown
                        pos = -1
                    else:
                        pos = 0
                        st += 1
                    street_last_bet_to = 0
                    check_or_call_ends_street = False
                else:
                    pos = (pos + 1) % 2
                    check_or_call_ends_street = True
                last_bet_size = 0
                last_bettor = -1
                
            elif c == 'f':  # Fold
                action_history.append({'player': pos, 'action': 'fold'})
                if last_bet_size == 0:
                    return {'error': 'Illegal fold'}
                if i != sz:
                    return {'error': 'Extra characters at end of action'}
                pos = -1
                return {
                    'st': st,
                    'pos': pos,
                    'street_last_bet_to': street_last_bet_to,
                    'total_last_bet_to': total_last_bet_to,
                    'last_bet_size': last_bet_size,
                    'last_bettor': last_bettor,
                    'action_history': action_history
                }
                
            elif c == 'b':  # Bet/Raise
                j = i
                while i < sz and action[i] >= '0' and action[i] <= '9':
                    i += 1
                if i == j:
                    return {'error': 'Missing bet size'}
                try:
                    new_street_last_bet_to = int(action[j:i])
                except (TypeError, ValueError):
                    return {'error': 'Bet size not an integer'}
                
                new_last_bet_size = new_street_last_bet_to - street_last_bet_to
                action_history.append({'player': pos, 'action': 'bet/raise', 'amount': new_street_last_bet_to})
                
                # Validate that the bet is legal
                remaining = self.STACK_SIZE - total_last_bet_to
                if last_bet_size > 0:
                    min_bet_size = last_bet_size
                    # Make sure minimum opening bet is the size of the big blind
                    if min_bet_size < self.BIG_BLIND:
                        min_bet_size = self.BIG_BLIND
                else:
                    min_bet_size = self.BIG_BLIND
                    
                # Can always go all-in
                if min_bet_size > remaining:
                    min_bet_size = remaining
                if new_last_bet_size < min_bet_size:
                    return {'error': 'Bet too small'}
                max_bet_size = remaining
                if new_last_bet_size > max_bet_size:
                    return {'error': 'Bet too big'}
                    
                last_bet_size = new_last_bet_size
                street_last_bet_to = new_street_last_bet_to
                total_last_bet_to += last_bet_size
                last_bettor = pos
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
                
            else:
                return {'error': 'Unexpected character in action'}

        return {
            'st': st,
            'pos': pos,
            'street_last_bet_to': street_last_bet_to,
            'total_last_bet_to': total_last_bet_to,
            'last_bet_size': last_bet_size,
            'last_bettor': last_bettor,
            'action_history': action_history
        }
        
    def convert_to_game_state(self, slumbot_data: Dict) -> PokerGameState:
        """
        Convert Slumbot response to our PokerGameState.
        
        Args:
            slumbot_data: Response from Slumbot API
            
        Returns:
            PokerGameState object
        """
        # Create a new game state
        state = PokerGameState(
            stack_size=self.STACK_SIZE,
            small_blind=self.SMALL_BLIND,
            big_blind=self.BIG_BLIND
        )
        
        # Set hole cards
        if 'hole_cards' in slumbot_data and slumbot_data['hole_cards']:
            state.player_hole_cards[0] = [self.slumbot_card_to_card(card) for card in slumbot_data['hole_cards']]
        
        # Set board cards
        if 'board' in slumbot_data and slumbot_data['board']:
            state.board = [self.slumbot_card_to_card(card) for card in slumbot_data['board']]
            # Set current round based on board cards
            if len(state.board) == 0:
                state.current_round = PokerGameState.PREFLOP
            elif len(state.board) == 3:
                state.current_round = PokerGameState.FLOP
            elif len(state.board) == 4:
                state.current_round = PokerGameState.TURN
            elif len(state.board) == 5:
                state.current_round = PokerGameState.RIVER
        
        # Set position
        client_pos = slumbot_data.get('client_pos', 0)
        # In Slumbot, 0 is big blind, 1 is small blind
        # Our code might use the opposite convention, adjust accordingly
        state.current_player = client_pos
        
        # Parse action history to recreate betting state
        action = slumbot_data.get('action', '')
        if action:
            action_info = self.parse_action(action)
            
            # Set pot and bets based on action
            if 'street_last_bet_to' in action_info:
                street_last_bet_to = action_info['street_last_bet_to']
                if action_info['st'] == 0:  # Preflop
                    # Initialize with blinds
                    state.player_bets = [self.SMALL_BLIND, self.BIG_BLIND]
                    state.pot = self.SMALL_BLIND + self.BIG_BLIND
                    
                    # Adjust for additional bets
                    if street_last_bet_to > self.BIG_BLIND:
                        # Account for raises
                        state.player_bets[action_info['last_bettor']] = street_last_bet_to
                        state.pot += (street_last_bet_to - self.BIG_BLIND)
                else:
                    # Postflop streets
                    state.player_bets = [0, 0]
                    if street_last_bet_to > 0:
                        state.player_bets[action_info['last_bettor']] = street_last_bet_to
                        state.pot += street_last_bet_to
            
            # Update player stacks
            state.player_stacks[0] = self.STACK_SIZE - state.player_bets[0]
            state.player_stacks[1] = self.STACK_SIZE - state.player_bets[1]
            
        return state
    
    def map_agent_action_to_slumbot(self, action: int, raise_amount: Optional[int] = None) -> str:
        """
        Map our agent's action to Slumbot API format.
        
        Args:
            action: Agent action (0=fold, 1=check/call, 2=bet/raise)
            raise_amount: Amount to raise by (only for bet/raise)
            
        Returns:
            Action string for Slumbot API
        """
        if action == PokerGameState.FOLD:
            return self.FOLD
        
        # Get current game state and betting context
        current_state = self.agent.current_state
        our_bet = current_state.player_bets[self.agent.position]
        opponent_bet = current_state.player_bets[1 - self.agent.position]
        our_stack = current_state.player_stacks[self.agent.position]
        
        # Determine if we're in small blind or big blind
        is_small_blind = (current_state.current_round == 0 and self.agent.position == 1)
        
        if action == PokerGameState.CHECK_CALL:
            # Determine if we need to call or check
            if opponent_bet > our_bet:
                # We're facing a bet, so need to call
                call_amount = opponent_bet - our_bet
                if call_amount >= our_stack:
                    return self.CALL  # All-in call
                return self.CALL
            else:
                # No bet to call, so check
                return self.CHECK
        
        elif action == PokerGameState.BET_RAISE:
            # Slumbot expects total bet amount, not increment
            # Calculate minimum raise amounts
            min_raise = max(
                self.BIG_BLIND,  # Minimum raise is big blind
                (opponent_bet * 2) - our_bet  # At least double opponent's current bet
            )
            
            # Special handling for small blind
            if is_small_blind:
                # In small blind, ensure a meaningful raise
                min_raise = max(
                    self.BIG_BLIND * 2,  # At least 2 big blinds
                    self.BIG_BLIND + self.SMALL_BLIND  # Small blind + at least a big blind
                )
            
            # Use the provided raise amount or default to minimum
            if raise_amount is None or raise_amount < min_raise:
                raise_amount = min_raise
            
            # Calculate total bet amount
            total_bet_amount = opponent_bet + raise_amount
            
            # Prevent over-betting our stack
            max_bet = our_stack + our_bet
            total_bet_amount = min(total_bet_amount, max_bet)
            
            # Ensure we're not violating Slumbot's betting rules
            total_bet_amount = max(total_bet_amount, opponent_bet + self.BIG_BLIND)
            
            return f"{self.BET}{total_bet_amount}"
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def login(self, username: str, password: str) -> str:
        """
        Login to Slumbot API.
        
        Args:
            username: Slumbot username
            password: Slumbot password
            
        Returns:
            Authentication token
        """
        data = {"username": username, "password": password}
        response = requests.post(
            f'https://{self.SLUMBOT_HOST}/api/login', 
            json=data
        )
        
        success = getattr(response, 'status_code') == 200
        if not success:
            print(f'Login failed. Status code: {response.status_code}')
            try:
                print(f'Error response: {response.json()}')
            except ValueError:
                pass
            sys.exit(-1)

        try:
            r = response.json()
        except ValueError:
            print('Could not get JSON from response')
            sys.exit(-1)

        if 'error_msg' in r:
            print(f'Login error: {r["error_msg"]}')
            sys.exit(-1)
            
        token = r.get('token')
        if not token:
            print('Did not get token in response to /api/login')
            sys.exit(-1)
            
        print(f"Successfully logged in as {username}")
        return token
    
    def new_hand(self) -> Dict:
        """
        Start a new hand with Slumbot.
        
        Returns:
            Response from Slumbot API
        """
        data = {}
        if self.token:
            data['token'] = self.token
            
        response = requests.post(
            f'https://{self.SLUMBOT_HOST}/api/new_hand', 
            json=data
        )
        
        success = getattr(response, 'status_code') == 200
        if not success:
            print(f'New hand request failed. Status code: {response.status_code}')
            try:
                print(f'Error response: {response.json()}')
            except ValueError:
                pass
            sys.exit(-1)

        try:
            r = response.json()
        except ValueError:
            print('Could not get JSON from response')
            sys.exit(-1)

        if 'error_msg' in r:
            print(f'New hand error: {r["error_msg"]}')
            sys.exit(-1)
            
        # Update token if a new one is provided
        if 'token' in r:
            self.token = r['token']
            
        # Determine whether we're facing a bet or not
        self.is_facing_bet = False
        if 'action' in r and r['action']:
            # Check the action string for betting patterns
            action_str = r['action']
            
            # Check if the last character sequence contains a bet
            if 'b' in action_str:
                # Find the last betting sequence
                last_bet_pos = action_str.rfind('b')
                bet_str = ''
                for i in range(last_bet_pos + 1, len(action_str)):
                    if action_str[i].isdigit():
                        bet_str += action_str[i]
                    else:
                        break
                
                # If there was a bet followed by a call, we're not facing a bet
                # If there was a bet and no call, we are facing a bet
                if bet_str and not action_str.endswith('c'):
                    self.is_facing_bet = True
                    
            # Handle slash-separated streets
            streets = action_str.split('/')
            if streets and streets[-1]:
                last_street = streets[-1]
                # Check if there's a bet in the last street without a call
                if 'b' in last_street and not last_street.endswith('c'):
                    self.is_facing_bet = True
                    
        return r
    
    def act(self, action_str: str) -> Dict:
        """
        Send an action to Slumbot.
        
        Args:
            action_str: Action string for Slumbot API
            
        Returns:
            Response from Slumbot API
        """
        print(f"Sending action to Slumbot: {action_str}")
        data = {'token': self.token, 'incr': action_str}
        
        try:
            response = requests.post(
                f'https://{self.SLUMBOT_HOST}/api/act', 
                json=data
            )
            
            success = getattr(response, 'status_code') == 200
            if not success:
                print(f'Act request failed. Status code: {response.status_code}')
                try:
                    print(f'Error response: {response.json()}')
                except ValueError:
                    pass
                
                # Handle common errors
                if response.status_code == 400:
                    print("Bad request - likely an illegal action. Trying a safe action instead.")
                    # Try a safer action (checking) if possible
                    if action_str != self.CHECK:
                        print(f"Retrying with {self.CHECK} instead of {action_str}")
                        return self.act(self.CHECK)
                sys.exit(-1)

            try:
                r = response.json()
            except ValueError:
                print('Could not get JSON from response')
                sys.exit(-1)

            if 'error_msg' in r:
                print(f'Act error: {r["error_msg"]}')
                
                # Try to recover from common errors
                if "Illegal call" in r['error_msg'] and action_str == self.CALL:
                    print("Calling isn't legal here. Trying to check instead.")
                    return self.act(self.CHECK)
                elif "Illegal check" in r['error_msg'] and action_str == self.CHECK:
                    print("Checking isn't legal here. Trying to call instead.")
                    return self.act(self.CALL)
                else:
                    sys.exit(-1)
            
            # Check if we're facing a bet in the next action
            self.is_facing_bet = False
            if 'action' in r and r['action']:
                action_str = r['action']
                
                # Check the action string for betting patterns
                if 'b' in action_str:
                    # Find the last betting sequence
                    last_bet_pos = action_str.rfind('b')
                    bet_str = ''
                    for i in range(last_bet_pos + 1, len(action_str)):
                        if action_str[i].isdigit():
                            bet_str += action_str[i]
                        else:
                            break
                    
                    # If there was a bet followed by a call, we're not facing a bet
                    # If there was a bet and no call, we are facing a bet
                    if bet_str and not action_str.endswith('c'):
                        self.is_facing_bet = True
                
                # Handle slash-separated streets
                streets = action_str.split('/')
                if streets and streets[-1]:
                    last_street = streets[-1]
                    # Check if there's a bet in the last street without a call
                    if 'b' in last_street and not last_street.endswith('c') and not last_street.endswith('k'):
                        self.is_facing_bet = True
                    
            return r
            
        except Exception as e:
            print(f"Unexpected error while acting: {e}")
            sys.exit(-1)
    
    def play_hand(self) -> int:
        """
        Play a single hand against Slumbot.
        
        Returns:
            Hand winnings (positive for profit, negative for loss)
        """
        # Start a new hand
        r = self.new_hand()
        print("\n--- New Hand ---")
        client_pos = r.get('client_pos', 0)
        position = 'BB' if client_pos == 0 else 'SB'
        print(f"Position: {position}")
        print(f"Hole cards: {r.get('hole_cards')}")
        
        # Check if Slumbot has already acted
        action = r.get('action', '')
        if action:
            print(f"Current action history: {action}")
            
        # If we get a 'winnings' value, the hand is already over
        if 'winnings' in r:
            winnings = r.get('winnings', 0)
            print(f"Hand over. Winnings: {winnings}")
            return winnings
            
        # Create a game state from the Slumbot response
        game_state = self.convert_to_game_state(r)
        self.agent.current_state = game_state
        self.agent.position = client_pos
        
        # In Slumbot, SB (position 1) acts first preflop
        # If we're SB and there's no action yet, we need to make the first move
        # The first action for SB is often a call (completing the blind) or a raise
        if position == 'SB' and not action:
            # Initial actions in the small blind are usually to call (complete) or raise
            # In Slumbot API, "c" means call when facing a bet (the BB in this case)
            print("We're in the small blind, making the first action")
            
            # Use our agent to decide the action
            agent_action, raise_amount = self.agent.act()
            
            # Safety check - if agent returned None, use a default action
            if agent_action is None:
                print("WARNING: Agent returned None action. Using default call action.")
                agent_action = PokerGameState.CHECK_CALL
            
            # When in SB, we're always facing the big blind initially
            self.is_facing_bet = True
            
            # Convert to Slumbot format
            slumbot_action = self.map_agent_action_to_slumbot(agent_action, raise_amount)
            print(f"Our action (small blind): {slumbot_action}")
            
            # Send the action to Slumbot
            r = self.act(slumbot_action)
            print(f"Updated action history: {r.get('action', '')}")
            
            # Update board if available
            if 'board' in r and r['board']:
                print(f"Board: {r['board']}")
            
            # Check if hand is over
            if 'winnings' in r:
                winnings = r.get('winnings', 0)
                print(f"Hand over. Winnings: {winnings}")
                
                # Update the agent's blueprint if learning is enabled
                if self.agent.enable_learning:
                    self.agent.update_blueprint_from_solver()
                    
                return winnings
        
        # Play the hand until it's over
        while True:
            # In Slumbot, if it's our turn to act, we need to send an action
            if 'action' in r and ('old_action' not in r or r['old_action'] != r['action']):
                # Parse the action to determine the bet amount and whose turn it is
                action_info = self.parse_action(r['action'])
                
                # Determine if we need to call or check
                # At the beginning of the hand in the Big Blind, we're facing a "call" if SB did anything
                # other than fold
                if position == 'BB' and action_info['st'] == 0 and 'f' not in r['action']:
                    self.is_facing_bet = True
                else:
                    # Otherwise, we're facing a bet if the last action includes a bet that we haven't called
                    self.is_facing_bet = action_info.get('last_bet_size', 0) > 0
                
                # Use our agent to decide the action
                agent_action, raise_amount = self.agent.act()
                
                # Safety check - if agent returned None, use a default action
                if agent_action is None:
                    print("WARNING: Agent returned None action. Using default call/check action.")
                    agent_action = PokerGameState.CHECK_CALL
                
                # Convert to Slumbot format
                slumbot_action = self.map_agent_action_to_slumbot(agent_action, raise_amount)
                print(f"Our action: {slumbot_action}")
                
                # Send the action to Slumbot
                r = self.act(slumbot_action)
                print(f"Updated action history: {r.get('action', '')}")
                
                # Update board if available
                if 'board' in r and r['board']:
                    print(f"Board: {r['board']}")
                
                # Check if hand is over
                if 'winnings' in r:
                    winnings = r.get('winnings', 0)
                    print(f"Hand over. Winnings: {winnings}")
                    
                    # Update the agent's blueprint if learning is enabled
                    if self.agent.enable_learning:
                        self.agent.update_blueprint_from_solver()
                        
                    return winnings
            else:
                # If we're here and not in the SB at the start of the hand,
                # there might be some issue with the game state
                print("Determining our next move...")
                
                # Use our agent to decide the action
                agent_action, raise_amount = self.agent.act()
                
                # Safety check - if agent returned None, use a default action
                if agent_action is None:
                    print("WARNING: Agent returned None action. Using default call/check action.")
                    agent_action = PokerGameState.CHECK_CALL
                
                slumbot_action = self.map_agent_action_to_slumbot(agent_action, raise_amount)
                print(f"Our action: {slumbot_action}")
                
                # Send the action to Slumbot
                r = self.act(slumbot_action)
                print(f"Updated action history: {r.get('action', '')}")
                
                # Update board if available
                if 'board' in r and r['board']:
                    print(f"Board: {r['board']}")
                
                # Check if hand is over
                if 'winnings' in r:
                    winnings = r.get('winnings', 0)
                    print(f"Hand over. Winnings: {winnings}")
                    
                    # Update the agent's blueprint if learning is enabled
                    if self.agent.enable_learning:
                        self.agent.update_blueprint_from_solver()
                        
                    return winnings
                    
    def play_session(self, num_hands: int, username: Optional[str] = None, password: Optional[str] = None) -> Dict:
        """
        Play multiple hands against Slumbot.
        
        Args:
            num_hands: Number of hands to play
            username: Slumbot username (optional)
            password: Slumbot password (optional)
            
        Returns:
            Session statistics
        """
        # Login if credentials are provided
        if username and password:
            self.token = self.login(username, password)
        
        print(f"\nStarting session against Slumbot ({num_hands} hands)")
        start_time = time.time()
        
        # Play hands
        for h in range(num_hands):
            print(f"\nHand {h+1}/{num_hands}")
            hand_winnings = self.play_hand()
            self.total_winnings += hand_winnings
            self.hands_played += 1
            
            # Track result
            if hand_winnings > 0:
                self.wins += 1
            elif hand_winnings < 0:
                self.losses += 1
            else:
                self.ties += 1
                
            # Print current statistics
            win_rate = self.wins / self.hands_played * 100
            print(f"Current stats: {self.wins} wins, {self.losses} losses, {self.ties} ties ({win_rate:.1f}% win rate)")
            print(f"Total winnings: {self.total_winnings} chips")
            
            # Save blueprint periodically if learning is enabled
            if self.agent.enable_learning and h > 0 and h % 10 == 0:
                if self.agent.blueprint_path:
                    self.agent.blueprint_cfr.save(self.agent.blueprint_path)
                    print(f"Blueprint saved after {h+1} hands")
        
        # Calculate final statistics
        elapsed_time = time.time() - start_time
        hands_per_minute = self.hands_played / (elapsed_time / 60)
        win_rate = self.wins / self.hands_played * 100
        bb_per_hand = self.total_winnings / self.hands_played / self.BIG_BLIND
        
        stats = {
            'hands_played': self.hands_played,
            'wins': self.wins,
            'losses': self.losses,
            'ties': self.ties,
            'win_rate': win_rate,
            'total_winnings': self.total_winnings,
            'bb_per_hand': bb_per_hand,
            'hands_per_minute': hands_per_minute,
            'elapsed_time': elapsed_time
        }
        
        # Print final statistics
        print("\n--- Session Summary ---")
        print(f"Hands played: {self.hands_played}")
        print(f"Wins: {self.wins}, Losses: {self.losses}, Ties: {self.ties}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Total winnings: {self.total_winnings} chips ({bb_per_hand:.2f} BB/hand)")
        print(f"Time elapsed: {elapsed_time:.1f} seconds ({hands_per_minute:.1f} hands/minute)")
        
        # Save final blueprint if learning is enabled
        if self.agent.enable_learning and self.agent.blueprint_path:
            self.agent.blueprint_cfr.save(self.agent.blueprint_path)
            print(f"Final blueprint saved to {self.agent.blueprint_path}")
            
        return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against Slumbot using a CFR poker agent")
    parser.add_argument("--username", type=str, help="Slumbot username")
    parser.add_argument("--password", type=str, help="Slumbot password")
    parser.add_argument("--blueprint", type=str, default="blueprint.pkl", help="Path to blueprint strategy")
    parser.add_argument("--hands", type=int, default=100, help="Number of hands to play")
    parser.add_argument("--depth", type=int, default=3, help="Maximum depth for solver")
    parser.add_argument("--learning", action="store_true", default=True, help="Enable continuous learning")
    
    args = parser.parse_args()
    
    # Create integration
    integration = SlumbotIntegration(
        blueprint_path=args.blueprint,
        max_depth=4,
        enable_learning=args.learning
    )
    
    # Play session
    integration.play_session(
        num_hands=args.hands,
        username=args.username,
        password=args.password
    )