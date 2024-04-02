import random as rd
import sys
from matplotlib import pyplot as plt
from gridworld import *

class q_function:
    def __init__(self, Qvalues):
        self.Qvalues = Qvalues.copy()
    
    def get_q_value(self, state, action):
        return self.Qvalues[state][action]

class q_agent:
    mdp=None
    state= None
    Qvalues = None
    epochs = 1000
    alpha = 0.001
    gamma = 0.99

    def __init__(self, mdp): # and here...
        self.mdp = mdp
        self.Qvalues = dict()
        for s in mdp.get_states():
            self.Qvalues[s] = dict() # Q(s, a)
            for a in mdp.get_actions(s):
                self.Qvalues[s][a] = rd.random()
        

    def greedy(self,state,eps):
        r = rd.random() # random nb 0,1s
        if r > eps:
            a_max = self.mdp.get_actions(state)[0]
            q_max = self.Qvalues[state][a_max]
            for a in self.mdp.get_actions(state):
                if self.Qvalues[state][a] > q_max:
                    a_max, q_max = a, self.Qvalues[state][a]
            return a_max
        else:
            return rd.choice(self.mdp.get_actions(state))

    def solve(self):
        for k in range(1, self.epochs+1):
            sys.stdout.write("\rEpoch : " + str(k) + "/" + str(self.epochs))
            sys.stdout.flush()
            epsilon = 1 - (k/self.epochs)
            self.state = self.mdp.get_initial_state()
            while not self.mdp.is_terminal(self.state):
                if k == self.epochs:
                    #self.mdp.visualise(self.state)
                    pass
                action = self.greedy(self.state, epsilon)
                nextState, reward = self.mdp.execute(self.state, action)
                delta = self.get_delta(reward, self.Qvalues, self.state, nextState, action)
                self.Qvalues[self.state][action] = self.Qvalues[self.state][action] + self.alpha*delta
                self.state=nextState
            if k == self.epochs:
                qf = q_function(self.Qvalues)
                self.mdp.visualise_q_function(qf)
        return self.Qvalues

    def get_delta(self, reward, q_value, state, next_state, action):
        max = max(q_value[next_state][a] for a in self.mdp.get_actions(next_state))
        return reward + (self.gamma*max) - q_value[state][action]