import random
def tuple_to_matrix(tup):
    ret = []
    for i in tup:
        ret.append(list(i))
    return ret

def matrix_to_tuple(mat):
    ret = []
    for i in mat:
        ret.append(tuple(i))
    return tuple(ret)

def sign(x):
    if x == 0: return 0 
    return int(abs(x)/x)

class State:
    def __init__(self, board, passive_moves, next_player):
        self.board = board
        self.passive_moves = passive_moves
        self.next_player = next_player
        
    def repr(self):
        return [self.board]
        
    def __eq__(self, other):
        return (other.board, other.next_player) == (self.board, self.next_player)
        
    def __hash__(self):
        return hash((self.board, self.next_player))
    

class Checkers:
    def __init__(self):
        self.n_actions = 64*64+1
    
    def action_id(self, a):
        if a == None: return -1
        p1, p2 = a
        return 64 * (8 * p1[0] + p1[1]) + 8 * p2[0] + p2[1]
    
    def initial_state(self):
        return State(((1,0,1,0,1,0,1,0), (0,1,0,1,0,1,0,1), (1,0,1,0,1,0,1,0), (0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0), (0,-1,0,-1,0,-1,0,-1), (-1,0,-1,0,-1,0,-1,0), (0,-1,0,-1,0,-1,0,-1)), 0, 1)
    
    def first_player(self):
        return 1
    
    def recursive_moves(self, s, player, pos, moves, seq=[]):
        y,x = pos
        board = s.board
        king = (s.board[y][x]==player*2)
        p = sign(player)
        # Basic moves
        if y+p < 8 and y+p >= 0:
            if x+1 < 8 and board[y+p][x+1] == 0:  moves.append((pos, (y+p,x+1)))
            if x-1 >= 0 and board[y+p][x-1] == 0: moves.append((pos, (y+p,x-1)))
            if king and y-p < 8 and y-p > 0:
                if x+1 < 8 and board[y-p][x+1] == 0:  moves.append((pos, (y-p,x+1)))
                if x-1 >= 0 and board[y-p][x-1] == 0: moves.append((pos, (y-p,x-1)))
                
        # Capture moves (should be recursive)
        if y+2*p < 8 and y+2*p >= 0:
            if x+2 < 8 and board[y+p][x+1] == -p and board[y+2*p][x+2] == 0:  moves.append((pos, (y+2*p,x+2)))
            if x-2 >= 0 and board[y+p][x-1] == -p and board[y+2*p][x-2] == 0: moves.append((pos, (y+2*p,x-2)))
            if king and y-2*p < 8 and y-2*p >= 0:
                if x+2 < 8 and board[y-p][x+1] == -p and board[y+2*p][x+2] == 0:  moves.append((pos, (y-2*p,x+2)))
                if x-2 >= 0 and board[y-p][x-1] == -p and board[y+2*p][x-2] == 0: moves.append((pos, (y-2*p,x-2)))
        
    def possible_actions(self, s, player):
        ret = []
        for i in range(8):
            for j in range(8):
                if sign(s.board[i][j]) == player: self.recursive_moves(s, player, (i,j), ret)
        return ret
        
    def is_terminal(self, s):
        if s.passive_moves > 20: return True
        score = {-1:12, 0:0, 1:12}
        for i in range(8):
            for j in range(8):
                score[-sign(s.board[i][j])]-=1
        return score[-1] == 12 or score[1] == 12
    
    def reward(self, s):
        score = {-1:12, 0:0, 1:12}
        for i in range(8):
            for j in range(8):
                score[-sign(s.board[i][j])]-=1
    
        return self.winner(s)
    
    def winner(self, s):
        score = {-1:12, 0:0, 1:12}
        for i in range(8):
            for j in range(8):
                score[-sign(s.board[i][j])]-=1
                
        if score[1] < score[-1]: return -1
        if score[1] > score[-1]: return 1
        return 0
        
    
    
    def step(self, s, a, player):
        if a == None: return s
        buf = tuple_to_matrix(s.board)
        pm = s.passive_moves
        p1,p2 = a
        dx, dy = p2[1]-p1[1], p2[0]-p1[0]
        
        buf[p2[0]][p2[1]] = buf[p1[0]][p1[1]]
        if abs(dx) == 2: buf[p1[0]+int(dy/2)][p1[1]+int(dx/2)] = 0
        else: pm += 1
        buf[p1[0]][p1[1]] = 0
        return State(matrix_to_tuple(buf), pm, -player)
    
    
    def random_action(self, s, player):
        actions = self.possible_actions(s, player)
        random.shuffle(actions)
        if len(actions) == 0: return None
        return actions[0]
                
            