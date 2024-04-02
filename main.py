from dynamic_programming import *
from gridworld import *
from q_learning import *
from dynamic_programming import *

# QL = q_agent(GridWorld())
# print (" states :" , QL.mdp.get_states () )
# print (" terminal states :" , QL.mdp.get_goal_states () )
# print (" actions :" , QL.mdp.get_actions () )

# bestQ = QL.solve()

DL = dp_agent(GridWorld())
print (" states :" , DL.mdp.get_states () )
print (" terminal states :" , DL.mdp.get_goal_states () )
print (" actions :" , DL.mdp.get_actions () )

bestV = DL.solve()
print(bestV)