import random
import numpy as np
from collections import deque 
from sklearn.neural_network import MLPRegressor

class StoreAverage:
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
    

class StoreValue:
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
    

# Approximate the values with a given model
class StoreApproximation:
    def __init__(self, game, model=None, default=0, retrain_iter=1000):
        self.x = deque(maxlen=50000)
        self.y = deque(maxlen=50000)
        self.trained = False
        if model != None: self.model = model
        else:             self.model = MLPRegressor()
        self.default = default
        self.game = game
        self.retrain_iter = retrain_iter
        self.iter = 0
        
    def Set(self, x, y):
        a,s = x
        a = self.game.action_id(a)
        Y = np.zeros(self.game.n_actions)
        Y[a] = y
        self.y.append(Y)
        self.x.append(np.array(s.repr()).flatten())
        self.iter += 1
        if self.iter >= self.retrain_iter:
            self.iter = 0
            self.Retrain()
        
    def Get(self, x):
        a,s = x
        a = self.game.action_id(a)
        x = np.array(s.repr()).flatten().reshape(1, -1)
        if self.trained: return self.model.predict(x)[0][a]
        return self.default
    
    def Retrain(self):
        x = np.array(self.x)
        y = np.array(self.y)
        self.model.fit(x, y)
        self.trained = True