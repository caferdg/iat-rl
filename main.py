from dynamic_programming import *
from gridworld import *
from q_learning import *
from dynamic_programming import *
import sys

if len(sys.argv) != 2:
    print("Usage: python main.py <q or dp>")
    exit()

if (sys.argv[1] == "q"):
    QL = q_agent(GridWorld(), alpha=0.01, gamma=0.95, nbEp=6000)
    start = tm.time()
    bestQ = QL.solve()

elif(sys.argv[1] == "dp"):
    DP = dp_agent(GridWorld(), eps=0.0001, gamma=0.80)
    start = tm.time()
    bestV = DP.solve()
else:
    print("Please enter a valid argument : q or dp")
    exit()
