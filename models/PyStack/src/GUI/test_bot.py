


class TestBot():
    def __init__(self):
        pass

    def start_new_hand(self, card1, card2, player_is_small_blind):
        print('starting new hand:')
        print(card1, card2, player_is_small_blind)


    # def compute_action(self, board_string, player_bet, opponent_bet):
    #     while True:
    #         action = input('enter action:')
    #         if action in ['fold','call','raise','allin']:
    #             break
    #     amount = -1
    #     if action == 'raise':
    #         amount = input('enter amount:')
    #     return {'action':action, 'amount': int(amount)}

    # def compute_action(self, board_string, player_bet, opponent_bet):
    #     import time
    #     time.sleep(5)
    #     print('===PLAYER BET: {} OPP BET: {}'.format(player_bet, opponent_bet))
    #     return {'action':'call', 'amount': int(-1)}


    def compute_action(self, board_string, player_bet, opponent_bet):
        import time
        time.sleep(3)
        print('===PLAYER BET: {} OPP BET: {}'.format(player_bet, opponent_bet))
        return {'action':'raise', 'amount': int(1000)}
