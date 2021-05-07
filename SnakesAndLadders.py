import random
class State:
    def __init__(self, square, t):
        self.square = square
        self.t = t
        
    def repr(self):
        return self.square
        
    def __eq__(self, other):
        return other.square == self.square
        
    def __hash__(self):
        return hash(self.square)
    
    def __str__(self):
        return str(self.square)
    
    def __repr__(self):
        return str(self.square)

class SnakesAndLadders:
    def __init__(self, layout=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], circular=True):
        self.layout = layout
        self.max_iter = 100
        self.circular = circular
    
    def initial_state(self):
        return State(0,0)
    
    def distance_to_goal(self, s):
        square = s.square
        if square >= 0 and square < 10: return 10-square
        if square >= 10: return 14-square
        return 0
        
    def possible_actions(self, s):
        return [0,1,2]
        
    def is_terminal(self, s):
        if s.square >= len(self.layout)-1 or s.t >= self.max_iter: return True
        return False
    
    def reward(self, s):
        if self.is_terminal(s): return self.max_iter-s.t
        return 15-self.distance_to_goal(s)
    
    
    def step(self, s, a):
        square = s.square
        t = s.t
        step = 0
        
        if a == 0: step = random.choice([0,1])
        if a == 1: step = random.choice([0,1,2])
        if a == 2: step = random.choice([0,1,2,3])
            
        was_on_slow_lane = (square > 2 and square < 10)
        
        go_to_fast = False
        if square == 2: go_to_fast = random.choice([True, False])
        if go_to_fast and step > 0: square = 9
            
        
        if self.circular: square = (square+step) % len(self.layout)
        else: square = min(len(self.layout)-1, square+step)
        
        
        if was_on_slow_lane and square >= 10: square=14
        if square >= len(self.layout)-1: return State(square, t+1)
        #    
        # Restart
        if a != 0 and self.layout[square] == 1:
            restart = random.choice([True, False])
            if a == 1 and restart: square = 0 
            if a == 2: square = 0 
                
        # Penalty
        elif a != 0 and self.layout[square] == 2: 
            penalty = a==2 or (a==1 and random.choice([True, False]))
            if penalty:
                if square < 10 or square >= 13: square -= 3
                if square >= 10 and square < 13: square = 3 - (13 - square) 
        
        # Prison
        elif a != 0 and self.layout[square] == 3: 
            if a == 1: t += random.choice([0,1])
            if a == 2: t += 1
                
        # Gamble
        elif a != 0 and self.layout[square] == 4: 
            gamble = random.choice([True, False])
            if a == 1 and gamble: square = random.choice(range(len(self.layout)))
            if a == 2: square = random.choice(range(len(self.layout)))
                
        square = max(0, square)
                
        return State(square, t+1)
    
    
    def random_action(self, s):
        actions = self.possible_actions(s)
        random.shuffle(actions)
        if len(actions) == 0: return None
        return actions[0]
                
            