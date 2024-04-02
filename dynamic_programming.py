# class value_function:
#     def __init__(self, values):
#         self.values = values

#     def get_value(self, state):
#         return self.values[state]
    
# class policy:
#     def __init__(self, values, mdp):
#         self.values = values
#         self.mdp

#     def select_action(self, state):
#         listAct = []
#         for action in self.mdp.get_actions():
#             maxProba = 0
#             for new_state, proba in self.mdp.get_transitions(state, action):
#                 if proba > maxProba:
#                     maxProba = proba

#         return self.values[state]

class dp_agent():
    mdp=None
    values=None
    epsilon = 0.00001

    def __init__(self,mdp): #and here...
        self.mdp=mdp
        self.values = dict()
        for s in mdp.get_states():
            self.values[s] = 0

    def get_value(self,s,v):
        #return the value of a specific state s according to value function v
        return v[s]
        
    def get_width(self,v,v_bis):
        return max(abs(v[s] - v_bis[s]) for s in self.mdp.get_states())

    def solve(self):
        diff = 99
        while diff > self.epsilon:
            v_ancien = self.values.copy()
            for s in self.mdp.get_states():
                self.update(s)
            diff = self.get_width(self.values, v_ancien)
        #self.mdp.visualise_value_function(value_function(self.values))
        #self.mdp.visualise_policy(policy(self.values))
        return(self.values)
    
    def update(self,s):
        gamma = 0.9
        # get Max
        max = -1
        for a in self.mdp.get_actions():
            r = 0
            for new_state, proba in self.mdp.get_transitions(s, a):
                r += proba * self.mdp.get_reward(s,a,new_state) + (gamma*proba*self.values[new_state])
            if r > max:
                max = r

        self.values[s] = max
