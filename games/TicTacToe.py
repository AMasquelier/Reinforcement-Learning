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


class State:
    def __init__(self, board):
        self.board = board
        
    def repr(self):
        return self.board
        
    def __eq__(self, other):
        return other.board == self.board
    
    def __str__(self):
        return str(self.board)
        
    def __hash__(self):
        return hash(self.board)

class TicTacToe:
    def __init__(self, size=3, goal=3):
        self.n_actions = size*size+1
        self.size = size
        self.goal = goal
        
    def action_id(self, a):
        if a == None: return -1
        return a[0]*self.size+a[1]
    
    def initial_state(self):
        board = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                row.append(0)
            board.append(row)
        return State(matrix_to_tuple(board))
    
    def first_player(self):
        return 1
        
    def possible_actions(self, s, player):
        ret = []
        for i in range(self.size):
            for j in range(self.size):
                if s.board[i][j] == 0: ret.append((i,j))
        return ret
    
    def found_win(self, s, player):
        size = self.size
        goal = self.goal
        for i in range(size):
            for j in range(size):
                count = 0
                for k in range(size-i):
                    if s.board[i+k][j] == player: count += 1
                    else: break
                if count == goal: return True
                count = 0
                for k in range(size-j):
                    if s.board[i][j+k] == player: count += 1
                    else: break
                if count == goal: return True
                count = 0
                for k in range(size-max(i,j)):
                    if s.board[i+k][j+k] == player: count += 1
                    else: break
                if count == goal: return True
                count = 0
                for k in range(min(size-j, i)):
                    if s.board[i-k][j+k] == player: count += 1
                    else: break
                if count == goal: return True
                count = 0
        return False
        
    def is_terminal(self, s):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if s.board[i][j] == 0: count += 1
        if count == 0: return True
        for i in [-1, 1]:
            if self.found_win(s, i): return True
            
        return False
    
    def reward(self, s):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if s.board[i][j] == 0: count += 1
        if count == 0: return True
        for i in [-1, 1]:
            if self.found_win(s, i): return i
        return 0
    
    def winner(self, s):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if s.board[i][j] == 0: count += 1
        if count == 0: return 0
        for i in [-1, 1]:
            if self.found_win(s, i): return i
        return 0
    
    
    def step(self, s, a, player):
        if a == None: return s.board
        buf = tuple_to_matrix(s.board)
        buf[a[0]][a[1]] = player
        return State(matrix_to_tuple(buf))
    
    
    def random_action(self, s, player):
        actions = self.possible_actions(s, player)
        random.shuffle(actions)
        if len(actions) == 0: return None
        return actions[0]
                
            