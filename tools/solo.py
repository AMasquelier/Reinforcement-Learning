import numpy as np
import random
from matplotlib import pyplot as plt

def make_games(game, model, n=10000, verbose=False, epsilon=0.5):
    w,l = 0,0
    initial_state = game.initial_state()
    cost = []

    for i in range(n):
        s = initial_state
        
        while not game.is_terminal(s):
            rand = random.uniform(0, 1)
            if rand < epsilon: a = game.random_action(s)
            else: a = model.argmax(s)
            if verbose: print(a, s.square, s.t)
            s = game.step(s, a)
        if verbose: print(a, s.square, s.t)
        cost.append(s.t)
    if verbose: print("Mean cost :", np.mean(cost))
    return np.mean(cost)
    
    
def learning_curve(game, model, n=1000, steps=100, n_games=1000, epsilon=0.5):
    x = np.linspace(0, n, steps+1)
    R = []
    for i, j in zip(x[1:], x[:-1]):
        model.train(int(i-j), epsilon=epsilon)
        r = make_games(game, model, n_games, False, epsilon=0)
        R.append(r)
    
    plt.figure(figsize=(10,7))
    plt.plot(x[1:], R, label='1: win')
    plt.legend(loc='lower right')
    plt.xlabel('Number of training games')
    plt.ylabel('Cost')
    plt.show()
    print("Final :", R[-1])
    
    
def compare_learning_curve(game, model1, model2, n=10000, steps=100, n_games=1000, epsilon=0.5):
    x = np.linspace(0, n, steps+1)
    R1 = []
    R2 = []
    for i, j in zip(x[1:], x[:-1]):
        model1.train(int(i-j), epsilon=epsilon)
        r = make_games(game, model1, n_games, False, epsilon=0)
        R1.append(r)
        model2.train(int(i-j), epsilon=epsilon)
        r = make_games(game, model2, n_games, False, epsilon=0)
        R2.append(r)
    
    plt.figure(figsize=(10,7))
    plt.plot(x[1:], R1, 'c', label= model1.name)
    
    plt.plot(x[1:], R2, 'r', label= model2.name)
    plt.legend(loc='center right')
    plt.xlabel('Number of training games')
    plt.ylabel('Cost')
    plt.show()