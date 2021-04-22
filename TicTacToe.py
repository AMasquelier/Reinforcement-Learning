import random
class TicTacToe:
    def __init__(self):
        pass
    
    def initial_state(self):
        return (0,0,0,  0,0,0,  0,0,0)
    
    def first_player(self):
        return 1
        
    def possible_actions(self, s):
        ret = []
        for i in range(9):
            if s[i] == 0: ret.append(i)
        return ret
        
    def is_terminal(self, s):
        if not (0 in s): return True
        for i in [-1, 1]:
            if  s[0] == i and s[1] == i and s[2] == i : return True
            if  s[3] == i and s[4] == i and s[5] == i : return True
            if  s[6] == i and s[7] == i and s[8] == i : return True
            
            if  s[0] == i and s[3] == i and s[6] == i : return True
            if  s[1] == i and s[4] == i and s[7] == i : return True
            if  s[2] == i and s[5] == i and s[8] == i : return True
            
            if  s[0] == i and s[4] == i and s[8] == i : return True
            if  s[6] == i and s[4] == i and s[2] == i : return True
            
        return False
    
    def reward(self, s):
        if not (0 in s): return 0
        for i in [-1, 1]:
            if  s[0] == i and s[1] == i and s[2] == i : return i
            if  s[3] == i and s[4] == i and s[5] == i : return i
            if  s[6] == i and s[7] == i and s[8] == i : return i
            
            if  s[0] == i and s[3] == i and s[6] == i : return i
            if  s[1] == i and s[4] == i and s[7] == i : return i
            if  s[2] == i and s[5] == i and s[8] == i : return i
            
            if  s[0] == i and s[4] == i and s[8] == i : return i
            if  s[6] == i and s[4] == i and s[2] == i : return i
        return 0
    
    def winner(self, s):
        if not (0 in s): return 0
        for i in [-1, 1]:
            if  s[0] == i and s[1] == i and s[2] == i : return i
            if  s[3] == i and s[4] == i and s[5] == i : return i
            if  s[6] == i and s[7] == i and s[8] == i : return i
            
            if  s[0] == i and s[3] == i and s[6] == i : return i
            if  s[1] == i and s[4] == i and s[7] == i : return i
            if  s[2] == i and s[5] == i and s[8] == i : return i
            
            if  s[0] == i and s[4] == i and s[8] == i : return i
            if  s[6] == i and s[4] == i and s[2] == i : return i
        return 0
    
    
    def step(self, s, a, player):
        if a == None: return s
        buf = list(s)
        buf[a] = player
        return tuple(buf)
    
    
    def random_action(self, s):
        actions = self.possible_actions(s)
        random.shuffle(actions)
        if len(actions) == 0: return None
        return actions[0]
                
            