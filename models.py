import random
from TicTacToe import TicTacToe

##########################
# Monte-Carlo Prediction #
##########################

class StoreMCP:
    def __init__(self):
        self.x = {}
        self.N = {}
        
    def Add(self, key, val):
        if key not in self.x: 
            self.x[key] = val
            self.N[key] = 1
        else: 
            V, N = self.x[key], self.N[key]
            self.x[key] = (N * V + val) / (N + 1)
            self.N[key] = N + 1
        
    def Average(self, key):
        if key in self.x: 
            return self.x[key]
        return 0
    
    def Contains(self, key):
        return key in self.x
    

class MCPrediction:
    def __init__(self, gamma=0.8):
        self.V = {}
        self.Returns = StoreMCP()
        self.gamma = gamma
        self.game = TicTacToe()
        self.name = "Monte-Carlo prediction"
        
    def choice(self, s, player, curious=False, verbose=False):
        actions = self.game.possible_actions(s)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            sp = self.game.step(s, a, player)
            score = 0
            if sp in self.V: score = self.V[sp]
            elif curious: score = player * 999
            scores.append((score, a))
            if verbose: print('   ', score, a)
            
        if player == 1: return max(scores)[1]
        if player == -1: return min(scores)[1]
        
    def make_game(self, initial_state, player=1, smart={-1:False, 1:False}):
        s = initial_state
        states = [s]
        results = [self.game.reward(s)]
        while not self.game.is_terminal(s):
            if not smart[player]: a = self.game.random_action(s)
            else: a = self.choice(s, player, True)
            s = self.game.step(s, a, player)
            states.append(s)
            results.append(self.game.reward(s))
            player *= -1
        return states, results
    
    def train(self, n=1000, smart={-1:False, 1:False}):
        initial_state = self.game.initial_state()
        for i in range(n):
            G = 0
            S, R = self.make_game(initial_state, self.game.first_player(), smart)
            S, R = reversed(S), reversed(R)
            for s, r in zip(S, R):
                G = self.gamma * G + r
                self.Returns.Add(s, G)
                self.V[s] = self.Returns.Average(s)
                
                
                
##########################
#       Q-Learning       #
##########################

class StoreQ:
    def __init__(self):
        self.x = {}
        
    def Set(self, key, val):
        if key not in self.x: self.x[key] = val
        else: self.x[key] = val
        
    def Get(self, key):
        if key in self.x: return self.x[key]
        return 0
    
    def Contains(self, key):
        return key in self.x
    

class QLearning:
    def __init__(self, alpha=0.3, gamma=0.8, game=TicTacToe()):
        self.Q = StoreQ()
        self.alpha = alpha
        self.gamma = gamma
        self.game = game
        self.name = "Q-Learning"
        
    def best_Q(self, s):
        scores = []
        A = self.game.possible_actions(s)
        random.shuffle(A)
        A.append(None)
        for a in A: 
            scores.append(self.Q.Get((a, s)))
        
        return max(scores)
        
    def choice(self, s, player, curious=False, verbose=False):
        actions = self.game.possible_actions(s)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            scores.append((self.Q.Get((a, s)), a))
            
        if player == 1: return max(scores)[1]
        if player == -1: return min(scores)[1]
        
    def make_game(self, initial_state, player=1, epsilon=0.5):
        s = initial_state
        actions = []
        while not self.game.is_terminal(s):
            rand = random.uniform(0, 1)
            if rand < epsilon: a = self.game.random_action(s)
            else: a = self.choice(s, player, True)
            sp = self.game.step(s, a, player)
            r = self.game.reward(s)
            
            self.Q.Set((a, s), self.Q.Get((a, s)) + self.alpha * (r + self.gamma * self.best_Q(sp) - self.Q.Get((a, s))))
            s = sp
            player *= -1
        
        r = self.game.reward(s)     
        self.Q.Set((None, s), self.Q.Get((None, s)) + self.alpha * r)
        
        return actions
    
    def train(self, n=1000, epsilon=0.5):
        initial_state = self.game.initial_state()
        for i in range(n):
            s = initial_state
            A = self.make_game(initial_state, self.game.first_player(), epsilon)
                