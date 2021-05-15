import random
import numpy as np
from models.storage import *

# Globaly the same as the models in models.py but using an estimator to approximate the value
# instead of searching it in a hashmap
                
################################
# Monte-Carlo Exploring Starts #
################################
# Source : Reinforcement Learning : an introcuction, Sutton & Barto, p.99

class DeepMCES:
    def __init__(self, game, gamma=0.8, model=None, defaultq=0, retrain_iter=1000, max_data=50000):
        self.P = {}
        self.Returns = StoreAverage()
        self.Q = StoreApproximation(game, model=model, default=defaultq, retrain_iter=retrain_iter, max_data=max_data)
        self.gamma = gamma
        self.game = game
        self.name = "Deep Monte-Carlo Exploring Starts"
        
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

class DeepQLearning:
    def __init__(self, game, alpha=0.3, gamma=0.8, model=None, defaultq=0, retrain_iter=1000, max_data=50000):
        self.Q = StoreApproximation(game, model=model, default=defaultq, retrain_iter=retrain_iter, max_data=max_data)
        self.alpha = alpha
        self.gamma = gamma
        self.game = game
        self.name = "Deep Q-Learning"
        
        
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
            
