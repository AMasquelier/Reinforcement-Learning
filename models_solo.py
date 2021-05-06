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
        
    def choice(self, s, curious=False, verbose=False):
        actions = self.game.possible_actions(s)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            sp = self.game.step(s, a)
            score = 0
            if sp in self.V: score = self.V[sp]
            elif curious: score = 999
            scores.append((score, a))
            if verbose: print('   ', score, a)
            
        return max(scores)[1]
        
    def make_game(self, initial_state, smart=False):
        s = initial_state
        states = [s]
        results = [self.game.reward(s)]
        while not self.game.is_terminal(s):
            if not smart: a = self.game.random_action(s)
            else: a = self.choice(s, True)
                
            s = self.game.step(s, a)
            states.append(s)
            results.append(self.game.reward(s))
        return states, results
    
    def train(self, n=1000, smart=False):
        initial_state = self.game.initial_state()
        for i in range(n):
            G = 0
            S, R = self.make_game(initial_state, smart)
            S, R = reversed(S), reversed(R)
            for s, r in zip(S, R):
                G = self.gamma * G + r
                self.Returns.Add(s, G)
                self.V[s] = self.Returns.Average(s)
                
                
                
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
        return 99999
    
    def Contains(self, key):
        return key in self.x
    

class QLearning:
    def __init__(self, game, alpha=0.3, gamma=0.8):
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
        
    def choice(self, s, curious=False, verbose=False):
        actions = self.game.possible_actions(s)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            scores.append((self.Q.Get((a, s)), a))
            
        return max(scores, key=lambda x: (x[0], -x[1]))[1]
        
    def make_game(self, initial_state, epsilon=0.5):
        s = initial_state
        actions = []
        while not self.game.is_terminal(s):
            rand = random.uniform(0, 1)
            if rand < epsilon: a = self.game.random_action(s)
            else: a = self.choice(s, True)
            sp = self.game.step(s, a)
            r = self.game.reward(s)
            
            self.Q.Set((a, s), self.Q.Get((a, s)) + self.alpha * (r + self.gamma * self.best_Q(sp) - self.Q.Get((a, s))))
            s = sp
        
        r = self.game.reward(s)     
        self.Q.Set((None, s), self.Q.Get((None, s)) + self.alpha * r)
        
        return actions
    
    def train(self, n=1000, epsilon=0.5):
        initial_state = self.game.initial_state()
        for i in range(n):
            s = initial_state
            A = self.make_game(initial_state, epsilon)
            
            
            
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
        self.Q = StoreBP(9999)
        self.N = StoreBP()
        self.alpha = alpha
        self.gamma = gamma
        self.game = game
        self.name = "Bandit problem"
        
    def choice(self, s, curious=False, verbose=False):
        actions = self.game.possible_actions(s)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            scores.append((self.Q.Get((a, s)), a))
            
        return max(scores, key=lambda x: (x[0]))[1]
        
    def make_game(self, initial_state, epsilon=0.5):
        s = initial_state
        actions = []
        while not self.game.is_terminal(s):
            rand = random.uniform(0, 1)
            if rand < epsilon: a = self.game.random_action(s)
            else: a = self.choice(s, True)
                
            sp = self.game.step(s, a)
            r = self.game.reward(s)
            
            self.N.Set((a, s), self.N.Get((a, s))+1)
            Qa = self.Q.Get((a, s))
            self.Q.Set((a, s), Qa + (r-Qa) / self.N.Get((a, s)))
            s = sp
        
        r = self.game.reward(s)   
        Qa = self.Q.Get((None, s))
        self.N.Set((None, s), self.N.Get((None, s))+1)
        self.Q.Set((None, s), Qa + (r-Qa) / self.N.Get((None, s)))
        
        return actions
    
    def train(self, n=1000, epsilon=0.5):
        initial_state = self.game.initial_state()
        for i in range(n):
            s = initial_state
            A = self.make_game(initial_state, epsilon)
            A = self.make_game(initial_state, epsilon)
            
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
    def __init__(self, game, n=5, alpha=0.5, gamma=0.8):
        self.Q = StoreNSS(9999)
        self.alpha = alpha
        self.gamma = gamma
        self.game = game
        self.n = n
        self.name = "n-Step SARSA"
        
    def choice(self, s, curious=False, verbose=False):
        actions = self.game.possible_actions(s)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            scores.append((self.Q.Get((a, s)), a))
            
        return max(scores, key=lambda x: (x[0]))[1]
    
    def choose_action(self, s, epsilon):
        rand = random.uniform(0, 1)
        if rand < epsilon: a = self.game.random_action(s)
        else: a = self.choice(s, True)
        return a
        
    def make_game(self, initial_state, epsilon=0.5):
        t = 0
        T = 999999999
        
        s = initial_state
        a = self.choose_action(s, epsilon)
        actions = [a]
        states = [s]
        rewards = [0]
        
        while True:
            if t < T:
                s = self.game.step(s, a)
                r = self.game.reward(s)
                
                states.append(s)
                rewards.append(r)
                
                if self.game.is_terminal(s): T = t+1
                else:
                    a = self.choose_action(s, epsilon)
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
            A = self.make_game(initial_state, epsilon)