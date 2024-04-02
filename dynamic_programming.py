import random as rd
import matplotlib.pyplot as plt
import time as tm

class value_function:
    def __init__(self, values):
        self.values = values

    def get_value(self, state):
        return self.values[state]
    
class policy:
    def __init__(self, mdp, values, gamma):
        self.mdp = mdp
        self.values = values
        self.gamma = gamma

    def select_action(self, state):
        max = -1
        a_max = self.mdp.get_actions(state)[0]
        for a in self.mdp.get_actions(state):
            r = 0
            for new_state, proba in self.mdp.get_transitions(state, a):
                r += proba * (self.mdp.get_reward(state,a,new_state) + self.gamma*self.values[new_state])
            if r > max:
                max = r
                a_max = a
        return a_max



class dp_agent():
    # HYPERPARAMETERS
    epsilon = None
    gamma = None

    mdp=None
    values=None


    def __init__(self,mdp, eps=0.001, gamma=0.99):
        self.mdp=mdp
        self.values = dict()
        for s in mdp.get_states():
            self.values[s] = rd.gauss(0, 0.1)
        self.epsilon = eps
        self.gamma = gamma

    def get_value(self,s,v):
        #return the value of a specific state s according to value function v
        return v[s]
        
    def get_width(self,v,v_bis):
        return max(abs(v[s] - v_bis[s]) for s in self.mdp.get_states())

    def solve(self):
        print("Training Dynamic Programming agent with the following hyperparameters:")
        print(" - Epsilon: ", self.epsilon)
        print(" - Gamma: ", self.gamma)
        start = tm.time()
        diff = self.epsilon+1
        firstV = [self.values[(0,0)]] # value of the initial state history
        k = 0
        while diff > self.epsilon:
            k += 1
            v_ancien = self.values.copy()
            for s in self.mdp.get_states():
                self.update(s)
            diff = self.get_width(self.values, v_ancien)
            firstV.append(self.values[(0,0)])
        print("Convergence reached in", k, "iterations", "in", tm.time()-start, "seconds")
        plt.plot(range(0, k+1), firstV)
        plt.xlabel('Iterations')
        plt.ylabel('Value of the initial state')
        plt.show()
        self.mdp.visualise_value_function(value_function(self.values))
        self.mdp.visualise_policy(policy(self.mdp, self.values, self.gamma))
        return(self.values)
    
    def update(self,s):
        # get Max
        max = -1
        for a in self.mdp.get_actions(s):
            r = 0
            for new_state, proba in self.mdp.get_transitions(s, a):
                r += proba * (self.mdp.get_reward(s,a,new_state) + self.gamma*self.values[new_state])
            if r > max:
                max = r

        self.values[s] = max
