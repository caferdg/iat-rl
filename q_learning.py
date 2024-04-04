import random as rd
import sys
from matplotlib import pyplot as plt
from gridworld import *
import time as tm

class q_function:
    def __init__(self, Qvalues):
        self.Qvalues = Qvalues.copy()
    
    def get_q_value(self, state, action):
        return self.Qvalues[state][action]

class q_agent:
    # HYPERPARAMETERS
    nbEpisodes = None
    alpha = None
    gamma = None
    
    mdp=None
    state= None
    Qvalues = None


    def __init__(self, mdp, alpha=0.01, gamma=0.99, nbEp=1000): # and here...
        self.mdp = mdp
        self.Qvalues = dict()
        for s in mdp.get_states():
            self.Qvalues[s] = dict() # Q(s, a)
            for a in mdp.get_actions(s):
                #self.Qvalues[s][a] = rd.random()
                self.Qvalues[s][a] = rd.gauss(0, 0.1)
        self.alpha = alpha
        self.gamma = gamma
        self.nbEpisodes = nbEp
        

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
        print("Training Q-Learning agent with the following hyperparameters:")
        print(" - Alpha: ", self.alpha)
        print(" - Gamma: ", self.gamma)
        print(" - Number of episodes: ", self.nbEpisodes)
        start = tm.time()
        y=[]
        episodes = range(1, self.nbEpisodes+1)
        for k in range(1, self.nbEpisodes+1):
            sys.stdout.write("\rEpisode : " + str(k) + "/" + str(self.nbEpisodes))
            sys.stdout.flush()
            epsilon = 1 - (k/self.nbEpisodes)
            self.state = self.mdp.get_initial_state()
            while not self.mdp.is_terminal(self.state):
                # finalAlpha = 0.0001
                # lrScheduling = (finalAlpha-self.alpha)*k/self.nbEpisodes + self.alpha
                action = self.greedy(self.state, epsilon)
                nextState, reward = self.mdp.execute(self.state, action)
                delta = self.get_delta(reward, self.Qvalues, self.state, nextState, action)
                self.Qvalues[self.state][action] = self.Qvalues[self.state][action] + self.alpha*delta
                self.state=nextState
            y.append(self.Qvalues[(0,0)][UP])
        print("\nTraining time: ", tm.time()-start)

        self.mdp.visualise_q_function(q_function(self.Qvalues))
        plt.plot(episodes,y)
        plt.xlabel("Episode")
        plt.ylabel("Q((0,0), ▲)")
        plt.show()
        return self.Qvalues

    def get_delta(self, reward, q_value, state, next_state, action):
        maxx = max(q_value[next_state][a] for a in self.mdp.get_actions(next_state))
        return reward + (self.gamma*maxx) - q_value[state][action]