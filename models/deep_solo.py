import random
from models.storage import *

                
################################
# Monte-Carlo Exploring Starts #
################################
# Source : Reinforcement Learning : an introcuction, Sutton & Barto, p.99

class MCES:
    def __init__(self, game, gamma=0.8, defaultq=0):
        self.P = {}
        self.Returns = StoreAverage()
        self.Q = StoreValue(defaultq)
        self.gamma = gamma
        self.game = game
        self.name = "Monte-Carlo Exploring Starts"
        
    def argmax(self, s):
        actions = self.game.possible_actions(s)
        if len(actions) == 0: return None
        
        random.shuffle(actions)
        scores = []
        for a in actions:
            scores.append((self.Q.Get((a, s)), a))
            
        return max(scores, key=lambda x: x[0])[1]
        
    def make_game(self, initial_state, epsilon=0.5):
        s = initial_state
        states = [s]
        rewards = [0]
        actions = []
        while not self.game.is_terminal(s):
            rand = random.uniform(0, 1)
            if rand < epsilon: a = self.game.random_action(s)
            else: a = self.argmax(s)
        
            actions.append(a)
            s = self.game.step(s, a)
            states.append(s)
            rewards.append(self.game.reward(s))
        actions.append(None)
        
        return states, rewards, actions
    
    def train(self, n=1000, epsilon=0.5):
        initial_state = self.game.initial_state()
        for i in range(n):
            G = 0
            S, R, A = self.make_game(initial_state, epsilon)
            S, R, A = reversed(S), reversed(R), reversed(A)
            for s, r, a in zip(S, R, A):
                G = self.gamma * G + r
                self.Returns.Add((a,s), G)
                self.Q.Set((a, s), self.Returns.Average((a,s)))
                self.P[s] = self.argmax(s)
                
                
##########################
#       Q-Learning       #
##########################
# Source : Reinforcement Learning : an introcuction, Sutton & Barto, p.131

class QLearning:
    def __init__(self, game, alpha=0.3, gamma=0.8, defaultq=0):
        self.Q = StoreApproximation(game, default=defaultq)
        self.alpha = alpha
        self.gamma = gamma
        self.game = game
        self.name = "Deep Q-Learning"
        
    def best_Q(self, s):
        scores = []
        A = self.game.possible_actions(s)
        random.shuffle(A)
        A.append(None)
        for a in A: 
            scores.append(self.Q.Get((a, s)))
        
        return max(scores)
        
    def argmax(self, s):
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
            else: a = self.argmax(s)
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
            
