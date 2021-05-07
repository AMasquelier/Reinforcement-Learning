from matplotlib import pyplot as plt
import numpy as np

def make_games(game, model, n=1000, verbose=False, smart={-1:True, 1:True}):
    w,l = 0,0
    initial_state = game.initial_state()

    for i in range(n):
        s = initial_state
        player = game.first_player()
        
        while not game.is_terminal(s):
            if not smart[player]: a = game.random_action(s, player)
            else: a = model.choice(s, player, True)
            s = game.step(s, a, player)
            player *= -1

        if game.winner(s) == 1: w += 1
        if game.winner(s) == -1: l += 1
    if verbose: print("win:", 100 * w / n, "% , lost:", 100 * l / n, "%")
    return w / n, l / n
    
    
def learning_curve(game, model, n=10000, steps=100, n_games=1000):
    x = np.linspace(0, n, steps+1)
    W, L = [], []
    W2, L2 = [], []
    for i, j in zip(x[1:], x[:-1]):
        model.train(int(i-j))
        w, l = make_games(game, model, n_games, False, smart={-1:False, 1:True})
        W.append(w)
        L.append(l)
        w, l = make_games(game, model, n_games, False, smart={-1:True, 1:False})
        W2.append(w)
        L2.append(l)
    
    plt.figure(figsize=(10,7))
    plt.plot(x[1:], W, label='1: win')
    plt.plot(x[1:], L2, label='-1: win')
    plt.legend(loc='lower right')
    plt.xlabel('Number of training games')
    plt.ylabel('Frequence')
    plt.show()
    
    
def compare_learning_curve(game, model1, model2, n=10000, steps=100, n_games=1000):
    x = np.linspace(0, n, steps+1)
    W, L = [], []
    W2, L2 = [], []
    for i, j in zip(x[1:], x[:-1]):
        model1.train(int(i-j))
        model2.train(int(i-j))
        w, l = make_games(game, model1, n_games, False, smart={-1:False, 1:True})
        W.append(w)
        L.append(l)
        w, l = make_games(game, model2, n_games, False, smart={-1:False, 1:True})
        W2.append(w)
        L2.append(l)
    
    plt.figure(figsize=(10,7))
    plt.plot(x[1:], W, 'c', label= model1.name + ' win')
    plt.plot(x[1:], L, 'c--', label= model1.name + ' lost')
    
    plt.plot(x[1:], W2, 'r', label= model2.name + ' win')
    plt.plot(x[1:], L2, 'r--', label= model2.name + ' lost')
    plt.legend(loc='center right')
    plt.xlabel('Number of training games')
    plt.ylabel('Frequence')
    plt.show()