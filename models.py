import random
from TicTacToe import TicTacToe

##########################
# Monte-Carlo Prediction #
##########################
# Source : Reinforcement Learning : an introcuction, Sutton & Barto, p.92
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
    def __init__(self, game, gamma=0.8):
        self.V = {}
        self.Returns = StoreMCP()
        self.gamma = gamma
        self.game = game
        self.name = "Monte-Carlo prediction"
        
    def argmax(self, s, player):
        actions = self.game.possible_actions(s, player)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            sp = self.game.step(s, a, player)
            score = 999
            if sp in self.V: score = self.V[sp]
            scores.append((score, a))
            
        if player == 1: return max(scores, key=lambda x: x[0])[1]
        if player == -1: return min(scores, key=lambda x: x[0])[1]
        
    def make_game(self, initial_state, player=1, epsilon={-1:0.5, 1:0.5}):
        s = initial_state
        states = [s]
        results = [self.game.reward(s)]
        while not self.game.is_terminal(s):
            rand = random.uniform(0, 1)
            if rand < epsilon[player]: a = self.game.random_action(s, player)
            else: a = self.argmax(s, player)
                
            s = self.game.step(s, a, player)
            states.append(s)
            results.append(self.game.reward(s))
            player *= -1
        return states, results
    
    def train(self, n=1000, epsilon={-1:0.5, 1:0.5}):
        initial_state = self.game.initial_state()
        for i in range(n):
            G = 0
            S, R = self.make_game(initial_state, self.game.first_player(), epsilon)
            S, R = reversed(S), reversed(R)
            for s, r in zip(S, R):
                G = self.gamma * G + r
                self.Returns.Add(s, G)
                self.V[s] = self.Returns.Average(s)
                
                
################################
# Monte-Carlo Exploring Starts #
################################
# Source : Reinforcement Learning : an introcuction, Sutton & Barto, p.99

class MCES:
    def __init__(self, game, gamma=0.8):
        self.P = {}
        self.Returns = StoreMCP()
        self.Q = StoreQ()
        self.gamma = gamma
        self.game = game
        self.name = "Monte-Carlo Exploring Starts"
        
    def argmax(self, s, player):
        actions = self.game.possible_actions(s, player)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            scores.append((self.Q.Get((a, s)), a))
            
        if player == 1: return max(scores, key=lambda x: x[0])[1]
        if player == -1: return min(scores, key=lambda x: x[0])[1]
        
    def make_game(self, initial_state, player=1, epsilon={-1:0.5, 1:0.5}):
        s = initial_state
        states = [s]
        rewards = [0]
        actions = []
        while not self.game.is_terminal(s):
            rand = random.uniform(0, 1)
            if rand < epsilon[player]: a = self.game.random_action(s, player)
            else: a = self.argmax(s, player)
        
            actions.append(a)
            s = self.game.step(s, a, player)
            states.append(s)
            rewards.append(self.game.reward(s))
            player *= -1
        actions.append(None)
        
        return states, rewards, actions
    
    def train(self, n=1000, epsilon={-1:0.5, 1:0.5}):
        initial_state = self.game.initial_state()
        for i in range(n):
            G = 0
            S, R, A = self.make_game(initial_state, self.game.first_player(), epsilon)
            S, R, A = reversed(S), reversed(R), reversed(A)
            for s, r, a in zip(S, R, A):
                G = self.gamma * G + r
                self.Returns.Add((a,s), G)
                self.Q.Set((a, s), self.Returns.Average((a,s)))
                self.P[s] = self.argmax(s, 0)
                
                
                
##########################
#       Q-Learning       #
##########################
# Source : Reinforcement Learning : an introcuction, Sutton & Barto, p.131
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
    def __init__(self, game, alpha=0.3, gamma=0.8):
        self.Q = StoreQ()
        self.alpha = alpha
        self.gamma = gamma
        self.game = game
        self.name = "Q-Learning"
        
        
    def best_Q(self, s, player):
        scores = []
        A = self.game.possible_actions(s, player)
        random.shuffle(A)
        A.append(None)
        for a in A: 
            scores.append(self.Q.Get((a, s)))
        
        return max(scores)
        
    def argmax(self, s, player):
        actions = self.game.possible_actions(s, player)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            scores.append((self.Q.Get((a, s)), a))
            
        if player == 1: return max(scores, key=lambda x: x[0])[1]
        if player == -1: return min(scores, key=lambda x: x[0])[1]
        
    def make_game(self, initial_state, player=1, epsilon={-1:0.5, 1:0.5}):
        s = initial_state
        actions = []
        while not self.game.is_terminal(s):
            rand = random.uniform(0, 1)
            if rand < epsilon[player]: a = self.game.random_action(s, player)
            else: a = self.argmax(s, player)
            sp = self.game.step(s, a, player)
            r = self.game.reward(s)
            
            self.Q.Set((a, s), self.Q.Get((a, s)) + self.alpha * (r + self.gamma * self.best_Q(sp, player) - self.Q.Get((a, s))))
            s = sp
            player *= -1
        
        r = self.game.reward(s)     
        self.Q.Set((None, s), self.Q.Get((None, s)) + self.alpha * r)
        
        return actions
    
    def train(self, n=1000, epsilon={-1:0.5, 1:0.5}):
        initial_state = self.game.initial_state()
        for i in range(n):
            s = initial_state
            A = self.make_game(initial_state, self.game.first_player(), epsilon)
            
            
##############################
#       Bandit Problem       #
##############################
# Source : Reinforcement Learning : an introcuction, Sutton & Barto, p.32
class StoreBP:
    def __init__(self, default=0):
        self.x = {}
        self.default = default
        
    def Set(self, key, val):
        if key not in self.x: self.x[key] = val
        else: self.x[key] = val
        
    def Get(self, key):
        if key in self.x: return self.x[key]
        return self.default
    
    def Contains(self, key):
        return key in self.x
    

class Bandit:
    def __init__(self, game, alpha=0.3, gamma=0.8):
        self.Q = StoreBP()
        self.N = StoreBP()
        self.alpha = alpha
        self.gamma = gamma
        self.game = game
        self.name = "Bandit problem"
        
        
    def choice(self, s, player, curious=False, verbose=False):
        actions = self.game.possible_actions(s, player)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            scores.append((self.Q.Get(s), a))
            
        if player == 1: return max(scores, key=lambda x: x[0])[1]
        if player == -1: return min(scores, key=lambda x: x[0])[1]
        
    def make_game(self, initial_state, player=1, epsilon=0.5):
        s = initial_state
        actions = []
        while not self.game.is_terminal(s):
            rand = random.uniform(0, 1)
            if rand < epsilon: a = self.game.random_action(s, player)
            else: a = self.choice(s, True)
                
            sp = self.game.step(s, a, player)
            r = self.game.reward(s)
            
            self.N.Set(a, self.N.Get(a)+1)
            Qa = self.Q.Get(a)
            self.Q.Set(a, Qa + (r-Qa) / self.N.Get(a))
            s = sp
            player *= -1
        
        r = self.game.reward(s)   
        Qa = self.Q.Get(None)
        self.N.Set(None, self.N.Get(None)+1)
        self.Q.Set(None, Qa + (r-Qa) / self.N.Get(None))
        
        return actions
    
    def train(self, n=1000, epsilon=0.5):
        initial_state = self.game.initial_state()
        for i in range(n):
            s = initial_state
            A = self.make_game(initial_state, self.game.first_player(), epsilon)
            
            
##############################
#        n-step SARSA        #
##############################       
# Source : Reinforcement Learning : an introcuction, Sutton & Barto, p.147
import numpy as np
class StoreNSS:
    def __init__(self, default=0):
        self.x = {}
        self.default = default
        
    def Set(self, key, val):
        if key not in self.x: self.x[key] = val
        else: self.x[key] = val
        
    def Get(self, key):
        if key in self.x: return self.x[key]
        return self.default
    
    def Contains(self, key):
        return key in self.x
            
class NStepSarsa:
    def __init__(self, game, n=1, alpha=0.5, gamma=0.8):
        self.Q = StoreNSS()
        self.alpha = alpha
        self.gamma = gamma
        self.game = game
        self.n = n
        self.name = "n-Step SARSA"
        
    def choice(self, s, player, curious=False, verbose=False):
        actions = self.game.possible_actions(s, player)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            scores.append((self.Q.Get((a, s)), a))
            
        if player == 1: return max(scores, key=lambda x: x[0])[1]
        if player == -1: return min(scores, key=lambda x: x[0])[1]
    
    def choose_action(self, s, player, epsilon):
        rand = random.uniform(0, 1)
        if rand < epsilon: a = self.game.random_action(s, player)
        else: a = self.choice(s, True)
        return a
        
    def make_game(self, initial_state, player=1, epsilon=0.5):
        t = 0
        T = 999999999
        
        s = initial_state
        a = self.choose_action(s, player, epsilon)
        actions = [a]
        states = [s]
        rewards = [0]
        
        while True:
            if t < T:
                s = self.game.step(s, a, player)
                r = self.game.reward(s)
                player *= -1
                states.append(s)
                rewards.append(r)
                
                if self.game.is_terminal(s): T = t+1
                else:
                    a = self.choose_action(s, player, epsilon)
                    actions.append(a)
                    
            tau = t-self.n+1
            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau+self.n+1, T+1)):
                    G += np.power(self.gamma, i-tau-1) * rewards[i]
                if tau + self.n < T:
                    S,A = states[tau+self.n], actions[tau+self.n]
                    G += np.power(self.gamma, self.n) * self.Q.Get((A,S))
            
                S,A = states[tau], actions[tau]
                QSA = self.Q.Get((A,S))
                self.Q.Set((A,S), QSA + self.alpha * (G - QSA))
        
            if tau == T - 1: break
            t += 1
        
        return actions
    
    def train(self, n=1000, epsilon=0.5):
        initial_state = self.game.initial_state()
        for i in range(n):
            s = initial_state
            A = self.make_game(initial_state, self.game.first_player(), epsilon)