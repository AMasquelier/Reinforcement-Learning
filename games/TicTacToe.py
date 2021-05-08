import random
class State:
    def __init__(self, board):
        self.board = board
        
    def repr(self):
        return self.board
        
    def __eq__(self, other):
        return other.board == self.board
        
    def __hash__(self):
        return hash(self.board)

class TicTacToe:
    def __init__(self):
        self.n_actions = 9+1
        
    def action_id(self, a):
        if a == None: return -1
        return a
    
    def initial_state(self):
        return State((0,0,0,  0,0,0,  0,0,0))
    
    def first_player(self):
        return 1
        
    def possible_actions(self, s, player):
        ret = []
        for i in range(9):
            if s.board[i] == 0: ret.append(i)
        return ret
        
    def is_terminal(self, s):
        if not (0 in s.board): return True
        for i in [-1, 1]:
            if  s.board[0] == i and s.board[1] == i and s.board[2] == i : return True
            if  s.board[3] == i and s.board[4] == i and s.board[5] == i : return True
            if  s.board[6] == i and s.board[7] == i and s.board[8] == i : return True
            
            if  s.board[0] == i and s.board[3] == i and s.board[6] == i : return True
            if  s.board[1] == i and s.board[4] == i and s.board[7] == i : return True
            if  s.board[2] == i and s.board[5] == i and s.board[8] == i : return True
            
            if  s.board[0] == i and s.board[4] == i and s.board[8] == i : return True
            if  s.board[6] == i and s.board[4] == i and s.board[2] == i : return True
            
        return False
    
    def reward(self, s):
        if not (0 in s.board): return 0
        for i in [-1, 1]:
            if  s.board[0] == i and s.board[1] == i and s.board[2] == i : return i
            if  s.board[3] == i and s.board[4] == i and s.board[5] == i : return i
            if  s.board[6] == i and s.board[7] == i and s.board[8] == i : return i
            
            if  s.board[0] == i and s.board[3] == i and s.board[6] == i : return i
            if  s.board[1] == i and s.board[4] == i and s.board[7] == i : return i
            if  s.board[2] == i and s.board[5] == i and s.board[8] == i : return i
            
            if  s.board[0] == i and s.board[4] == i and s.board[8] == i : return i
            if  s.board[6] == i and s.board[4] == i and s.board[2] == i : return i
        return 0
    
    def winner(self, s):
        if not (0 in s.board): return 0
        for i in [-1, 1]:
            if  s.board[0] == i and s.board[1] == i and s.board[2] == i : return i
            if  s.board[3] == i and s.board[4] == i and s.board[5] == i : return i
            if  s.board[6] == i and s.board[7] == i and s.board[8] == i : return i
            
            if  s.board[0] == i and s.board[3] == i and s.board[6] == i : return i
            if  s.board[1] == i and s.board[4] == i and s.board[7] == i : return i
            if  s.board[2] == i and s.board[5] == i and s.board[8] == i : return i
            
            if  s.board[0] == i and s.board[4] == i and s.board[8] == i : return i
            if  s.board[6] == i and s.board[4] == i and s.board[2] == i : return i
        return 0
    
    
    def step(self, s, a, player):
        if a == None: return s.board
        buf = list(s.board)
        buf[a] = player
        return State(tuple(buf))
    
    
    def random_action(self, s, player):
        actions = self.possible_actions(s, player)
        random.shuffle(actions)
        if len(actions) == 0: return None
        return actions[0]
                
            