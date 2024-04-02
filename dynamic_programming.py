import random as rd

class value_function:
    def __init__(self, values):
        self.values = values

    def get_value(self, state):
        return self.values[state]
    
class policy:
    def __init__(self, values, mdp):
        self.values = values
        self.mdp = mdp

    def select_action(self, state):
        action = self.mdp.get_actions(state)[0]
        for a in self.mdp.get_actions(state):
            maxProba = -1000
            candidate_state = (0,0)
            candidate_action = a
            for new_state, proba in self.mdp.get_transitions(state, a):
                if proba > maxProba:
                    maxProba = proba
                    candidate_state = new_state
                    candidate_action = a
            if self.values[candidate_state] > self.values[state]:
                action = candidate_action
        return action


class dp_agent():
    mdp=None
    values=None

    # HYPERPARAMETERS
    epsilon = None
    gamma = None

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
        diff = 99
        k = 0
        while diff > self.epsilon:
            k += 1
            v_ancien = self.values.copy()
            for s in self.mdp.get_states():
                self.update(s)
            diff = self.get_width(self.values, v_ancien)
        print("Convergence reached in", k, "iterations")
        self.mdp.visualise_value_function(value_function(self.values))
        self.mdp.visualise_policy(policy(self.values, self.mdp))
        return(self.values)
    
    def update(self,s):
        # get Max
        max = -1
        for a in self.mdp.get_actions():
            r = 0
            for new_state, proba in self.mdp.get_transitions(s, a):
                r += proba * self.mdp.get_reward(s,a,new_state) + (self.gamma*proba*self.values[new_state])
            if r > max:
                max = r

        self.values[s] = max
